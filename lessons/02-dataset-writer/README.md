# Lesson 02 — Write Your Own Dataset

Close the loop on the format: the same `create → add_frame → save_episode → finalize → push` pipeline the hardware track will run with real teleop, driven here by scripted SO-101 trajectories in MuJoCo — and the two things the format forces you to get right (rate-locking, state ≠ action) experienced as failures you planted yourself.

| | |
|---|---|
| **Phase** | 1 — Data |
| **Time** | 1 session (3–4 h desk time) |
| **Cost** | $0 (Mac-local) |
| **Prerequisites** | 01 (`window()` is this lesson's boundary check), 00 (SO-101 models cloned) |
| **Feeds into** | H2 (identical pipeline, real robot), 14/15 (you know exactly what a training sample contains), 03 (the scripted trajectories get replaced by real IK) |

## Learning objectives

After this lesson you can:

1. **Define** a v3 `features` schema for a robot with two cameras and defend each field.
2. **Explain** what `finalize()` writes and predict exactly what breaks, and when, if it is skipped.
3. **Predict** the sign and shape of the commanded-vs-measured gap that makes `action` ≠ `observation.state`, and why BC needs it.
4. **Diagnose** a rate-lock violation from the symptom Lesson 01's `window()` reports.
5. **Publish** a Hub dataset that loads in `LeRobotDataset`, renders in the visualizer, and passes the boundary check.

## Principles

**The creation lifecycle.** Verified against current source — `LeRobotDataset.create()`'s signature, abridged to what you'll use:

```python
LeRobotDataset.create(
    repo_id: str, fps: int, features: dict,
    root=None, robot_type=None, use_videos=True,
    tolerance_s=1e-4, image_writer_processes=0, image_writer_threads=0,
    video_backend=None, batch_encoding_size=1, ...)
```

Then per episode: `add_frame(frame: dict)` per timestep, `save_episode()` once per episode, and — non-negotiable — **`finalize()` once at the end, before `push_to_hub()`**. v3 writes parquet incrementally with buffered metadata; `finalize()` flushes the episode metadata and closes the parquet writers so footers get written. Skip it and the files are structurally corrupt (LeRobot PR #1903 is the receipt); Exercise 5 makes you watch it.

**The features schema.** `features` maps name → `{"dtype", "shape", "names"}`. Yours, matching `svla_so101_pickplace` conventions so Lesson 01 transfers verbatim:

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

`"video"` dtype means frames buffer to an image writer and encode to MP4 shards at `save_episode()`; `use_videos=False` would store per-frame images instead (bigger, faster random access).

**State vs action, decided once.** State = measured joint positions (`qpos` in sim, encoder readings on the real arm). Action = *commanded* targets (`ctrl` in sim, goal positions on the real arm). They differ by tracking error: the measurement lags the command and undershoots at reversals. That gap is what makes BC-on-actions work — a policy trained on state-as-action learns to command where the arm already is, and stalls. Never log `qpos` for both.

**Rate-locking.** The dataset declares `fps=30`; MuJoCo steps at `model.opt.timestep` (typically 2 ms). Record a frame every `n_sub = round(1 / (fps * timestep))` physics steps and assert `abs(1/(fps*timestep) - n_sub) < 1e-9`. Silent drift here becomes Lesson 01's `tolerance_s` violations downstream — frames land at timestamps that are not multiples of 1/fps, and every window over them is invalid.

**Carry forward**

- `finalize()` is part of the lifecycle, not cleanup: without it the parquet footers never get written.
- `action` is the command, `observation.state` the measurement; the gap between them is the training signal, not noise.
- Lock recording to the physics step count, never to wall-clock; assert the ratio is an integer.
- `add_frame` takes HWC uint8; `dataset[i]` returns CHW float32 in [0,1]. The library owns the conversion; you own the input convention.

| Source | Read for |
|---|---|
| Tutorial §1.3 | the data-collection recipe this lesson automates in sim |
| [Dataset v3 docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3), "Record a dataset" + "Common Issues" | current `create()`/`add_frame()`/`save_episode()`/`finalize()` semantics — the API of record if flags drift |
| `SO-ARM100/Simulation/SO101/so101_new_calib.xml` | joint names/limits/zeros — this model, not menagerie, because its calibration matches LeRobot's |

## Exercise 1 — Scene and cameras [Build]

Tests the schema principle: two named cameras whose renders match the `features` shapes exactly. Spec for `scene_record.xml` + a 20-line render check:

- Load `so101_new_calib.xml` via `<include>`; add a tabletop box geom and two `<camera>` elements: `up` (overhead, looking down at the workspace) and `side` (three-quarter view). If offscreen rendering errors on buffer size, add `<visual><global offheight="480" offwidth="640"/></visual>`.
- The check: offscreen render both cameras at 480×640 under plain `python` (no viewer) and save PNGs:
  ```python
  import mujoco
  model = mujoco.MjModel.from_xml_path("scene_record.xml")
  data = mujoco.MjData(model)
  r = mujoco.Renderer(model, height=480, width=640)
  mujoco.mj_forward(model, data)
  r.update_scene(data, camera="up"); img_up = r.render()
  ```
- Also print `model.jnt_range` mapped to motor names.

**✅ Checkpoint:** two 480×640 PNGs from the named cameras; a joint-name → range table in `RESULTS.md`.

## Exercise 2 — Scripted trajectories [Build]

Tests rate-locking and the actuator-tracking principle. Real IK arrives in Lesson 03; joint-space waypoints keep this honest. Spec for `generate.py`:

- A reach-and-return primitive: home → hover above a table target → descend → close gripper → return. 4–6 joint-space waypoints, hand-tuned once in the viewer by dragging joints and reading `qpos`.
- Cosine interpolation between waypoints, $q(s) = q_a + (q_b - q_a)\,\tfrac{1 - \cos(\pi s)}{2}$, segment durations sized to a max joint velocity of 1.5 rad/s.
- Per episode (seeded): target uniform over a 10×10 cm table zone (hover/descend waypoints recomputed from it), home pose jittered slightly. 50 episodes, 5–10 s each (150–300 frames at 30 fps).
- Drive the sim by setting `data.ctrl` to the interpolated target every physics step; record `(qpos → state, ctrl → action, both renders)` every `n_sub` steps, with the `n_sub` assertion from Principles.
- The check: overlaid joint traces for 5 episodes are smooth (no steps) and inside `jnt_range`.

**✅ Checkpoint:** the 5-episode overlay is smooth and in-range; the `n_sub` assertion passes at fps=30, timestep=0.002.

## Exercise 3 — State ≠ action [Predict → Run]

Tests objective 3.

1. **Write first**, in `RESULTS.md`: on a plot of commanded (`ctrl`) vs measured (`qpos`) for `shoulder_lift` over one episode, which trace leads, by roughly how many frames, and what happens at each waypoint reversal. Then: what a policy trained with `qpos` as both state *and* action would do at deployment, in one sentence.
2. Plot the two traces for 2 joints across one episode.
3. Reconcile. The lag is your evidence; its size sets a floor on how precisely any BC policy can be expected to track.

**✅ Checkpoint:** the plot shows a visible lag; your predicted direction was right or the reconciliation says why not.

## Exercise 4 — Break the rate lock [Diagnose]

Tests objective 4: what the `n_sub` assertion protects.

1. Predict, in writing: with `fps=29` and `timestep=0.002`, what is `1/(fps·timestep)`? Does `n_sub` exist? If you *skip* the assertion and record every `round(...)` steps anyway, what timestamps do frames land on, and what does Lesson 01's `window()` report for `deltas=[-1/29, 0, 1/29]` at the default `tolerance_s`?
2. Do it: generate 2 episodes at fps=29 with the assertion disabled, write them (Exercise 6's writer), load back, and run `window()` and `LeRobotDataset(..., delta_timestamps=...)` over them.
3. Restore fps=30. In `RESULTS.md`: the mechanism (frame timestamps drift off the 1/fps grid) and which downstream consumer would fail silently versus loudly.

**✅ Checkpoint:** the fps=29 dataset raises a tolerance error on windowing (or reports padded/invalid indices — record which); fps=30 is clean.

## Exercise 5 — Skip `finalize()` [Diagnose]

Tests objective 2.

1. Predict, in writing: if you call `save_episode()` for all 50 episodes and then `LeRobotDataset(repo_id, root=...)` *without* `finalize()`, what fails — the load, the metadata counts, or the parquet read — and with what kind of error?
2. Run it on a throwaway `root`. Record the actual error and the file that is malformed (`pyarrow.parquet.read_metadata` on a data shard tells you).
3. Delete the throwaway root.

**✅ Checkpoint:** the failure is reproduced and named by mechanism (unflushed episode metadata, unwritten parquet footers), not just by error string.

## Exercise 6 — Serialize and check the boundaries [Build]

Tests the lifecycle end to end, with Lesson 01's rules as the acceptance test. Spec for `writer.py`:

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

The check (`check_dataset.py`), *before* pushing: local load-back with `LeRobotDataset(repo_id, root=...)`; `delta_timestamps={"action": [i/30 for i in range(10)]}` returns `(10, 6)`; episode count 50 and total frames match the generation log exactly; `meta/stats.json` exists with state means inside joint ranges and image stats in [0,1]; and Lesson 01's `window()` agrees with the library's indices and `_is_pad` at the first and last frame of 3 episodes.

**✅ Checkpoint:** all five checks pass locally.

## Exercise 7 — Publish and inspect [Write]

1. `ds.push_to_hub()`; open the dataset in the visualizer (`huggingface.co/spaces/lerobot/visualize_dataset`, your repo_id): scrub 3 episodes; both cameras play; action traces plot.
2. Write the dataset card over the library's boilerplate: task description, features table, fps, episode count, generation script link, camera layout screenshot. Bar: a stranger regenerates the dataset from the card plus `generate.py`.

**✅ Checkpoint:** visualizer renders; card passes the stranger bar.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `scene_record.xml`, `generate.py`, `writer.py`, `check_dataset.py` | one command regenerates the full dataset deterministically (seeded) and the five checks pass |
| Hub dataset `<you>/so101_scripted_reach` | loads in `LeRobotDataset`; renders in visualizer; card passes the stranger bar |
| `RESULTS.md` | Exercise 3 prediction + plot + reconciliation; Exercise 4 and 5 predictions, observed failures, mechanisms; joint-range table |

## Done when

- [ ] 50 seeded episodes regenerate byte-identical metadata from one command.
- [ ] `check_dataset.py` passes, including the `window()` boundary agreement on 3 episodes.
- [ ] The fps=29 and skipped-`finalize()` failures are reproduced and explained by mechanism.
- [ ] Dataset displays correctly in the visualizer; card would let a stranger regenerate it.

## Self-check

1. What exactly does `finalize()` write, and what error surfaces (and when) if you skip it?
2. Why must `action` be the command and `observation.state` the measurement? What would a policy trained on state-as-action learn to do at deployment?
3. Your fps is 30 and MuJoCo's timestep is 2 ms. Derive `n_sub` and state the assertion that catches a bad pair (e.g. fps=29).
4. `add_frame` takes HWC uint8 images but `dataset[i]` returns CHW float32. Where does the conversion live and why store uint8?
5. Which fields of `meta/stats.json` will Lesson 14's training actually consume, and for what?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Dataset won't load after recording | `finalize()` never called — corrupt parquet footers | it's in the lifecycle for a reason; re-run the writer (Exercise 5 shows you the signature) |
| Washed-out or black frames in the visualizer | passed float [0,1] or CHW images to `add_frame` | HWC uint8 in; the library owns conversion |
| `tolerance_s` violations when windowing your own data | frame recording not locked to fps (drifting substep count) | the `n_sub` assertion; never record on wall-clock (Exercise 4) |
| Arm teleports between waypoints | interpolating `qpos` directly instead of driving `ctrl` through the profile | set `ctrl` targets each physics step; let the actuators track |
| `save_episode()` extremely slow | synchronous video encode per episode | `image_writer_processes/threads` in `create()`; encode in background |
| Push succeeds, visualizer shows nothing | card metadata not committed / wrong repo type | ensure `repo_type="dataset"`; re-run `push_to_hub()` after `finalize()` |

## Going deeper

- **Storage trade-off.** Record 5 episodes with `use_videos=False` (per-frame images) and compare on your Mac: dataset size, `save_episode()` time, and random-access `__getitem__` latency vs the video version. One table, three sentences on when you'd choose each.

## References

- LeRobot dataset v3 docs (creation lifecycle, "Always call `finalize()`", PR #1903).
- Tutorial §1.3 (collection recipe).
- `TheRobotStudio/SO-ARM100` — `Simulation/SO101/so101_new_calib.xml`.
- MuJoCo docs: `mujoco.Renderer`, offscreen framebuffer sizing (`<visual><global>`).
