# Lesson 02 — Write Your Own Dataset

Close the loop on the format: script SO-101 trajectories in MuJoCo and serialize them into a valid v3 dataset on the Hub — the exact pipeline the hardware track will run with real teleop instead of a script.

| | |
|---|---|
| **Phase** | 1 — Data |
| **Time** | 1 session (4–6 h desk time) |
| **Cost** | $0 (Mac-local) |
| **Prerequisites** | 01 (your parser is this lesson's acceptance test), 00 (SO-101 models cloned) |
| **Feeds into** | H2 (identical pipeline, real robot), 14/15 (you now know exactly what a training sample contains), 03 (the scripted trajectories get replaced by real IK) |

## Learning objectives

After this lesson you can:

1. **Define** a v3 `features` schema (dtypes, shapes, names) for a robot with two cameras and defend each field.
2. **Drive** the creation API — `LeRobotDataset.create()` → `add_frame()` → `save_episode()` → `finalize()` — and explain what `finalize()` writes and why skipping it corrupts the dataset.
3. **Generate** kinematically smooth scripted trajectories at a fixed control rate locked to the dataset fps.
4. **Publish** a Hub dataset that loads in `LeRobotDataset`, renders in the visualizer, and passes your Lesson 01 parser's parity tests.
5. **Enumerate** what the format forced you to get right — fps sync, image conventions, stats — before H2 raises the stakes with real hardware.

## Background

**The creation lifecycle.** Verified against current source — `LeRobotDataset.create()`'s signature (abridged to what you'll use):

```python
LeRobotDataset.create(
    repo_id: str, fps: int, features: dict,
    root=None, robot_type=None, use_videos=True,
    tolerance_s=1e-4, image_writer_processes=0, image_writer_threads=0,
    video_backend=None, batch_encoding_size=1, ...)
```

Then per episode: `add_frame(frame: dict)` per timestep, `save_episode()` once per episode, and — non-negotiable — **`finalize()` once at the end, before `push_to_hub()`**. v3 writes parquet incrementally with buffered metadata; `finalize()` flushes buffered episode metadata and closes parquet writers so footer metadata gets written. Skip it and your parquet files are structurally corrupt (LeRobot PR #1903 is the receipt).

**The features schema.** `features` maps name → `{"dtype", "shape", "names"}`. Yours, matching `svla_so101_pickplace` conventions so your Lesson 01 knowledge transfers verbatim:

```python
MOTORS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
          "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
features = {
    "action":             {"dtype": "float32", "shape": (6,), "names": MOTORS},
    "observation.state":  {"dtype": "float32", "shape": (6,), "names": MOTORS},
    "observation.images.up":   {"dtype": "video", "shape": (480, 640, 3),
                                "names": ["height", "width", "channels"]},
    "observation.images.side": {"dtype": "video", "shape": (480, 640, 3),
                                "names": ["height", "width", "channels"]},
}
```

`"video"` dtype means frames buffer to an image writer and encode to MP4 shards at `save_episode()`; `use_videos=False` would store per-frame images instead (bigger, faster random access — you'll benchmark this trade-off only if the stretch calls).

**State vs action, decided once.** State = measured joint positions (`qpos` in sim, encoder readings on the real arm). Action = *commanded* targets (`ctrl` in sim, goal positions on the real arm). They differ by tracking error — that difference is exactly what makes BC-on-actions work, so do not log `qpos` for both.

**Rate-locking.** The dataset declares `fps=30`; MuJoCo steps at `model.opt.timestep` (typically 2 ms). Record a frame every `n_sub = round(1 / (fps * timestep))` physics steps and assert `abs(1/(fps*timestep) - n_sub) < 1e-9` — silent drift here becomes Lesson 01's `tolerance_s` violations downstream.

| Source | Read for |
|---|---|
| Tutorial §1.3 | the data-collection recipe this lesson automates in sim |
| [Dataset v3 docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3), "Record a dataset" + "Common Issues" | current `create()`/`add_frame()`/`save_episode()`/`finalize()` semantics — the API of record if flags drift |
| `SO-ARM100/Simulation/SO101/so101_new_calib.xml` | joint names/limits/zeros — this model, not menagerie, because its calibration matches LeRobot's |

## Part 1 — Scene + cameras (1 h)

1. Load `so101_new_calib.xml`; add a tabletop (a box geom) and two `<camera>` elements: `up` (overhead, looking down at the workspace) and `side` (three-quarter view). Save as `scene_record.xml`.
2. Offscreen render both (plain `python` is fine — no viewer):
   ```python
   import mujoco
   model = mujoco.MjModel.from_xml_path("scene_record.xml")
   data = mujoco.MjData(model)
   r = mujoco.Renderer(model, height=480, width=640)
   mujoco.mj_forward(model, data)
   r.update_scene(data, camera="up"); img_up = r.render()
   ```
   If the render errors on buffer size, raise `<visual><global offheight="480" offwidth="640"/></visual>` in the XML.
3. Verify joint ranges: print `model.jnt_range` and map each row to a motor name.

**✅ Checkpoint:** two 480×640 PNGs from the named cameras; a printed joint-name → range table in your notes.

## Part 2 — Scripted trajectories (1.5 h)

Real IK arrives in Lesson 03; here, joint-space waypoints keep it honest and simple.

1. Define a reach-and-return primitive: home configuration → hover above a target on the table → descend → close gripper → return. 4–6 joint-space waypoints, hand-tuned once in the viewer (`mjpython`) by dragging joints and recording `qpos`.
2. Interpolate with a smooth profile — cosine (minimum-jerk-ish) between waypoints:
   $q(s) = q_a + (q_b - q_a)\,\tfrac{1 - \cos(\pi s)}{2},\ s \in [0,1]$ — sized so each segment respects a max joint velocity you set (e.g. 1.5 rad/s).
3. Per episode, randomize the target position uniformly over a 10×10 cm table zone (recompute the hover/descend waypoints from it) and randomize the home pose slightly. 50 episodes, each 5–10 s (150–300 frames at 30 fps).
4. Drive the sim: set `data.ctrl` to the interpolated target each physics step; record `(qpos → state, ctrl → action, renders)` every `n_sub` steps.

**✅ Checkpoint:** overlaid joint-position traces for 5 episodes are smooth (no steps/spikes), all within `jnt_range`; commanded-vs-measured traces visibly differ by a small tracking lag (that's your state≠action evidence).

## Part 3 — Serialize (1.5 h)

1. `writer.py`:
   ```python
   ds = LeRobotDataset.create(repo_id=f"{USER}/so101_scripted_reach",
                              fps=30, features=features, robot_type="so101")
   for ep in episodes:
       for t in range(len(ep)):
           ds.add_frame({"action": ep.action[t], "observation.state": ep.state[t],
                         "observation.images.up": ep.up[t],      # HWC uint8
                         "observation.images.side": ep.side[t],
                         "task": "Reach the target and grasp."})
       ds.save_episode()
   ds.finalize()
   ```
   The exact `task` plumbing (per-frame key vs `save_episode` argument) has moved between minor versions — check `add_frame`'s docstring in your installed source and note which your version wants.
2. Load-back test *before* pushing: `LeRobotDataset(repo_id, root=...)` locally; window it with `delta_timestamps={"action": [i/30 for i in range(10)]}`; check `meta/stats.json` exists and per-feature mean/std are sane (state means inside joint ranges, image stats in [0,1]).

**✅ Checkpoint:** local load-back works; windowed `action` comes back `(10, 6)`; episode count 50 and total frames match your generation logs exactly.

## Part 4 — Publish + validate (1 h)

1. `ds.push_to_hub()`; confirm the card renders and note what boilerplate the library generated.
2. Open the dataset in the visualizer (`huggingface.co/spaces/lerobot/visualize_dataset`, your repo_id): scrub 3 episodes; both cameras play; action traces plot.
3. **The real acceptance test:** point Lesson 01's parser + parity suite at your dataset. Your writer and your parser were built from the same FORMAT.md — if they disagree, one of them just taught you something; fix and record which.
4. Write the dataset card: task description, features table, fps, episode count, generation script link, camera layout screenshot.

**✅ Checkpoint:** visualizer renders; Lesson 01 parity suite green against your dataset.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `scene_record.xml` + `generate.py` + `writer.py` | one command regenerates the full dataset deterministically (seeded) |
| Hub dataset `<you>/so101_scripted_reach` | loads in `LeRobotDataset`; renders in visualizer; proper card |
| Lesson 01 suite run | parity green against this dataset (link the CI run / output in RESULTS.md) |
| `RESULTS.md` | what the format forced you to get right (fps lock, uint8 HWC input convention, stats, finalize); the state-vs-action trace figure; any writer/parser disagreement and its resolution |

## Done when

- [ ] 50 seeded episodes regenerate byte-identical metadata from one command.
- [ ] Dataset loads with `delta_timestamps` and displays correctly in the visualizer.
- [ ] Lesson 01 parser passes against it.
- [ ] Card would let a stranger regenerate the dataset.

## Self-check

1. What exactly does `finalize()` write, and what error surfaces (and when) if you skip it?
2. Why must `action` be the command and `observation.state` the measurement? What would a policy trained on state-as-action learn to do at deployment?
3. Your fps is 30 and MuJoCo's timestep is 2 ms. Derive `n_sub` and state the assertion that catches a bad pair (e.g. fps=29).
4. `add_frame` takes HWC uint8 images but `dataset[i]` returns CHW float32. Where does the conversion live and why store uint8?
5. Which fields of `meta/stats.json` will Lesson 14's training actually consume, and for what?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Dataset won't load after recording | `finalize()` never called — corrupt parquet footers | it's in the lifecycle for a reason; re-run the writer |
| Washed-out or black frames in the visualizer | passed float [0,1] or CHW images to `add_frame` | HWC uint8 in; the library owns conversion |
| `tolerance_s` violations when windowing your own data | frame recording not locked to fps (drifting substep count) | Part 2's `n_sub` assertion; never record on wall-clock |
| Arm teleports between waypoints | interpolating `qpos` directly instead of driving `ctrl` through the profile | set `ctrl` targets each physics step; let the actuators track |
| `save_episode()` extremely slow | synchronous video encode per episode | `image_writer_processes/threads` in `create()`; encode in background |
| Push succeeds, visualizer shows nothing | card metadata not committed / wrong repo type | ensure `repo_type="dataset"`; re-run `push_to_hub()` after `finalize()` |

## Stretch

Record 5 episodes with `use_videos=False` (per-frame images) and compare on your Mac: dataset size, `save_episode()` time, and random-access `__getitem__` latency vs the video version. One table, three sentences on when you'd choose each.

## References

- LeRobot dataset v3 docs (creation lifecycle, "Always call `finalize()`", PR #1903).
- Tutorial §1.3 (collection recipe).
- `TheRobotStudio/SO-ARM100` — `Simulation/SO101/so101_new_calib.xml`.
- MuJoCo docs: `mujoco.Renderer`, offscreen framebuffer sizing (`<visual><global>`).
