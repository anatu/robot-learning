# Lesson 02 — Write Your Own Dataset

In Lesson 01 you learned to read the LeRobotDataset format; in this lesson you write one. You will script reach-and-grasp trajectories for the SO-101 in MuJoCo, render two cameras, and serialize the result through the same `create → add_frame → save_episode → finalize → push` pipeline that the hardware track later drives with real teleoperation. Along the way you deliberately break the two things the format forces you to get right, the locking of the recording rate to the physics step and the distinction between commanded and measured joint positions, so that you recognise those failures when they happen on real hardware.

| | |
|---|---|
| **Phase** | 1 — Data |
| **Time** | 1 session (3–4 h desk time) |
| **Cost** | $0 (Mac-local) |
| **Prerequisites** | 01 (`window()` is this lesson's boundary check), 00 (SO-101 models cloned) |
| **Feeds into** | H2 (the identical pipeline on the real robot), 14 and 15 (you will know exactly what a training sample contains), 03 (the scripted trajectories are replaced by real inverse kinematics) |

## Learning objectives

After this lesson you can:

1. **Define** a v3 `features` schema for a robot with two cameras and defend each field.
2. **Explain** what `finalize()` writes, and predict exactly what breaks, and when, if it is skipped.
3. **Predict** the sign and shape of the gap between commanded and measured joint positions that makes `action` differ from `observation.state`, and explain why behaviour cloning needs that gap.
4. **Diagnose** a rate-lock violation from the symptom that Lesson 01's `window()` reports.
5. **Publish** a Hub dataset that loads in `LeRobotDataset`, renders in the visualizer, and passes the boundary check.

## Principles

### The creation lifecycle

A v3 dataset is created through a fixed sequence of calls, and the sequence matters because the writer buffers metadata and streams parquet incrementally. The entry point is `LeRobotDataset.create()`. Its signature, verified against the installed source and abridged to the arguments this lesson uses, is:

```python
LeRobotDataset.create(
    repo_id: str, fps: int, features: dict,
    root=None, robot_type=None, use_videos=True,
    tolerance_s=1e-4, image_writer_processes=0, image_writer_threads=0,
    video_backend=None, batch_encoding_size=1, ...)
```

After creation you call `add_frame(frame: dict)` once per timestep and `save_episode()` once per episode. When all episodes have been written you must call `finalize()` once, before `push_to_hub()`. The reason is that v3 writes parquet incrementally with buffered episode metadata. `finalize()` flushes that metadata and closes the parquet writers, and closing a parquet writer is what writes the file's footer. A parquet file without a footer is structurally corrupt, so a dataset that skipped `finalize()` cannot be read back. LeRobot pull request #1903 documents the bug reports that led to this rule, and Exercise 5 has you reproduce the failure so that you recognise it when it happens on real data.

### The features schema

The `features` dictionary maps each feature name to its dtype, shape, and per-dimension names. The schema below matches the conventions of `svla_so101_pickplace`, so that everything you learned about that dataset in Lesson 01 applies to yours unchanged:

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

The `"video"` dtype means that frames are buffered to an image writer and encoded into MP4 shards when `save_episode()` is called. Setting `use_videos=False` would store one image file per frame instead, which takes more space but gives faster random access; that trade-off is measured under Going deeper.

### State and action are different quantities

Two of the features above look interchangeable and are not. `observation.state` holds the measured joint positions, which is `qpos` in simulation and the encoder readings on a real arm. `action` holds the commanded targets, which is `ctrl` in simulation and the goal positions sent to the motors on a real arm. The two differ by the controller's tracking error: the measurement lags the command and undershoots at reversals. This difference is what behaviour cloning on actions relies on. A policy trained with the measured state as its action target would learn to command the position the arm is already at, and at deployment it would stall. For that reason you must never log `qpos` for both features.

### Locking the recording rate to the physics step

The dataset declares `fps=30`, while MuJoCo advances physics in steps of `model.opt.timestep`, typically 2 ms. To record at exactly 30 frames per second you record one frame every `n_sub = round(1 / (fps * timestep))` physics steps, and you assert that `abs(1/(fps*timestep) - n_sub) < 1e-9`, which is to say that the ratio of physics steps per frame is an integer. If the ratio is not an integer and you record anyway, frame timestamps drift off the $1/\text{fps}$ grid. That drift appears downstream as the `tolerance_s` violations from Lesson 01: every window over such a dataset is invalid, because the requested offsets no longer correspond to real frames.

**Carry forward**

- `finalize()` is part of the write lifecycle rather than an optional cleanup step, because it flushes the buffered episode metadata and writes the parquet footers; without it the files cannot be read.
- `action` records the command and `observation.state` records the measurement. The gap between them is the controller's tracking error, and it is the signal that behaviour cloning learns from, not noise to be removed.
- Recording must be locked to the physics step count, never to wall-clock time, and the number of physics steps per frame must be asserted to be an integer, because a non-integer ratio puts frames off the frame grid and invalidates every window over them.
- `add_frame` takes images as HWC uint8 arrays, and `dataset[i]` returns them as CHW float32 tensors in [0, 1]. The library owns that conversion; you are responsible for supplying the input in the expected convention.

| Source | Read for |
|---|---|
| Tutorial §1.3 | the data-collection recipe this lesson automates in simulation |
| [Dataset v3 docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3), "Record a dataset" + "Common Issues" | current `create()`/`add_frame()`/`save_episode()`/`finalize()` semantics; the API of record if flags drift |
| `SO-ARM100/Simulation/SO101/so101_new_calib.xml` | joint names, limits and zeros; this model rather than menagerie, because its calibration matches LeRobot's |

## Exercise 1 — Build the scene and cameras [Build]

In this exercise you build the MuJoCo scene that the dataset is recorded from: the SO-101 on a tabletop with two named cameras whose renders match the shapes declared in the `features` schema. Agreement between camera names, render sizes and the schema is the first place the format constrains you, and checking it now avoids a mismatch that would otherwise only surface at load time.

Write the specification for `scene_record.xml` and a short render check (about twenty lines), and have an AI tool draft them:

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

**✅ Checkpoint:** two 480×640 PNGs from the named cameras, and a joint-name → range table in `RESULTS.md`.

## Exercise 2 — Generate scripted trajectories [Build]

Here you write the trajectory generator that plays the role teleoperation plays on real hardware. Proper inverse kinematics arrives in Lesson 03; for now, hand-tuned joint-space waypoints with smooth interpolation are enough, and they keep attention on the two recording principles: driving the actuators rather than teleporting the joints, and recording at a rate locked to the physics step.

Write the specification for `generate.py` and have an AI tool draft it:

- A reach-and-return primitive: home → hover above a table target → descend → close gripper → return. Use 4–6 joint-space waypoints, hand-tuned once in the viewer by dragging joints and reading `qpos`.
- Cosine interpolation between waypoints, $q(s) = q_a + (q_b - q_a)\,\tfrac{1 - \cos(\pi s)}{2}$, with segment durations sized to a maximum joint velocity of 1.5 rad/s.
- Per episode (seeded): the target is drawn uniformly over a 10×10 cm table zone, with the hover and descend waypoints recomputed from it, and the home pose is jittered slightly. Generate 50 episodes of 5–10 s each (150–300 frames at 30 fps).
- Drive the simulation by setting `data.ctrl` to the interpolated target at every physics step, and record `(qpos → state, ctrl → action, both renders)` every `n_sub` steps, with the `n_sub` assertion from the Principles section.
- The check: overlaid joint traces for 5 episodes are smooth (no steps) and stay inside `jnt_range`.

**✅ Checkpoint:** the 5-episode overlay is smooth and in range, and the `n_sub` assertion passes at fps=30, timestep=0.002.

## Exercise 3 — Compare commanded and measured positions [Predict → Run]

This exercise makes the difference between `action` and `observation.state` visible. Before plotting, you predict the relationship between the commanded and measured traces. The prediction is worth making because the direction and size of the lag are what show that the two features carry different information, and a wrong guess here usually means the roles of `ctrl` and `qpos` have been confused.

1. Before running anything, write in `RESULTS.md` what you expect to see on a plot of commanded (`ctrl`) against measured (`qpos`) for `shoulder_lift` over one episode: which trace leads, by roughly how many frames, and what happens to the measured trace at each waypoint reversal. Then write, in one sentence, what a policy trained with `qpos` as both state and action would do at deployment.
2. Plot the two traces for 2 joints across one episode.
3. Reconcile the plot with your prediction. The lag you observe sets a floor on how precisely any behaviour-cloning policy trained on this data can be expected to track its own commands.

**✅ Checkpoint:** the plot shows a visible lag, and either your predicted direction was right or the reconciliation explains why not.

## Exercise 4 — Break the rate lock [Diagnose]

The `n_sub` assertion in Exercise 2 is easy to regard as pedantry. This exercise shows what it protects by choosing a frame rate that does not divide the physics step, disabling the assertion, and observing how the resulting dataset fails under Lesson 01's windowing rules.

1. Before running, predict in writing: with `fps=29` and `timestep=0.002`, what is `1/(fps·timestep)`? Does an integer `n_sub` exist? If you skip the assertion and record every `round(...)` steps anyway, on what timestamps do the frames land, and what does Lesson 01's `window()` report for `deltas=[-1/29, 0, 1/29]` at the default `tolerance_s`?
2. Generate 2 episodes at fps=29 with the assertion disabled, write them with Exercise 6's writer, load them back, and run both `window()` and `LeRobotDataset(..., delta_timestamps=...)` over them.
3. Restore fps=30. In `RESULTS.md`, describe the mechanism (frame timestamps drift off the 1/fps grid) and say which downstream consumer would fail loudly and which would fail silently.

**✅ Checkpoint:** the fps=29 dataset raises a tolerance error on windowing, or reports padded or invalid indices (record which); the fps=30 dataset is clean.

## Exercise 5 — Skip `finalize()` [Diagnose]

The rule that `finalize()` must be called is stated in the documentation, but the failure it prevents is easier to remember once you have seen it. In this exercise you write a dataset without calling `finalize()`, predict how the load will fail, and then inspect the malformed file.

1. Before running, predict in writing: if you call `save_episode()` for all 50 episodes and then `LeRobotDataset(repo_id, root=...)` without `finalize()`, what fails first, the load, the metadata counts, or the parquet read, and with what kind of error?
2. Run it on a throwaway `root`. Record the actual error and identify the malformed file; `pyarrow.parquet.read_metadata` on a data shard will tell you.
3. Delete the throwaway root.

**✅ Checkpoint:** the failure is reproduced and named by mechanism (unflushed episode metadata, unwritten parquet footers), not only by its error string.

## Exercise 6 — Serialize and check the boundaries [Build]

With the generator and scene in place, you write the dataset through the full lifecycle and check it locally before pushing. The check uses Lesson 01's `window()` at episode boundaries, because that is where a subtle writer bug such as an off-by-one episode length or a dropped frame first shows.

Write the specification for `writer.py` and have an AI tool draft it:

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

The exact `task` plumbing (a per-frame key versus a `save_episode` argument) has moved between minor versions. Check `add_frame`'s docstring in your installed source and note which form your version wants.

The check, in `check_dataset.py` and run before pushing, has five parts: a local load-back with `LeRobotDataset(repo_id, root=...)` succeeds; `delta_timestamps={"action": [i/30 for i in range(10)]}` returns a tensor of shape `(10, 6)`; the episode count is 50 and the total frame count matches the generation log exactly; `meta/stats.json` exists, with state means inside the joint ranges and image statistics in [0, 1]; and Lesson 01's `window()` agrees with the library's indices and `_is_pad` at the first and last frame of 3 episodes.

**✅ Checkpoint:** all five checks pass locally.

## Exercise 7 — Publish and write the card [Write]

Finally you push the dataset to the Hub, confirm that it renders in the visualizer, and write a card that would let someone else regenerate it.

1. Call `ds.push_to_hub()`, then open the dataset in the visualizer (`huggingface.co/spaces/lerobot/visualize_dataset`, with your repo_id). Scrub through 3 episodes: both cameras should play and the action traces should plot.
2. Write the dataset card over the library's boilerplate: task description, features table, fps, episode count, a link to the generation script, and a camera-layout screenshot. The bar is that a stranger could regenerate the dataset from the card plus `generate.py`.

**✅ Checkpoint:** the visualizer renders, and the card meets the stranger bar.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `scene_record.xml`, `generate.py`, `writer.py`, `check_dataset.py` | one command regenerates the full dataset deterministically (seeded) and the five checks pass |
| Hub dataset `<you>/so101_scripted_reach` | loads in `LeRobotDataset`; renders in the visualizer; card meets the stranger bar |
| `RESULTS.md` | Exercise 3 prediction, plot and reconciliation; Exercise 4 and 5 predictions, observed failures and mechanisms; the joint-range table |

## Done when

- [ ] 50 seeded episodes regenerate byte-identical metadata from one command.
- [ ] `check_dataset.py` passes, including the `window()` boundary agreement on 3 episodes.
- [ ] The fps=29 and skipped-`finalize()` failures are reproduced and explained by mechanism.
- [ ] The dataset displays correctly in the visualizer, and the card would let a stranger regenerate it.

## Self-check

1. What exactly does `finalize()` write, and what error surfaces, and when, if you skip it?
2. Why must `action` be the command and `observation.state` the measurement? What would a policy trained on state-as-action learn to do at deployment?
3. Your fps is 30 and MuJoCo's timestep is 2 ms. Derive `n_sub` and state the assertion that catches a bad pair such as fps=29.
4. `add_frame` takes HWC uint8 images but `dataset[i]` returns CHW float32. Where does the conversion live, and why store uint8?
5. Which fields of `meta/stats.json` will Lesson 14's training actually consume, and for what?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Dataset won't load after recording | `finalize()` never called, so the parquet footers are missing | call it after the last `save_episode()`; re-run the writer (Exercise 5 shows the failure signature) |
| Washed-out or black frames in the visualizer | float [0,1] or CHW images passed to `add_frame` | pass HWC uint8; the library owns the conversion |
| `tolerance_s` violations when windowing your own data | frame recording not locked to fps (drifting substep count) | the `n_sub` assertion; never record on wall-clock time (Exercise 4) |
| Arm teleports between waypoints | `qpos` interpolated directly instead of driving `ctrl` through the profile | set `ctrl` targets each physics step and let the actuators track |
| `save_episode()` extremely slow | synchronous video encode per episode | `image_writer_processes/threads` in `create()`; encode in the background |
| Push succeeds, visualizer shows nothing | card metadata not committed, or wrong repo type | ensure `repo_type="dataset"`; re-run `push_to_hub()` after `finalize()` |

## Going deeper

- **Storage trade-off.** Record 5 episodes with `use_videos=False` (per-frame images) and compare, on your Mac, dataset size, `save_episode()` time, and random-access `__getitem__` latency against the video version. Report one table and three sentences on when you would choose each.

## References

- LeRobot dataset v3 docs (creation lifecycle, "Always call `finalize()`", PR #1903).
- Tutorial §1.3 (collection recipe).
- `TheRobotStudio/SO-ARM100` — `Simulation/SO101/so101_new_calib.xml`.
- MuJoCo docs: `mujoco.Renderer`, offscreen framebuffer sizing (`<visual><global>`).
