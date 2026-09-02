# Lesson 00 — Setup & Repo Bootstrap

This lesson sets up everything the rest of the course depends on. You order the SO-101 hardware first, because its lead time of one to two weeks overlaps with the theory lessons; then you build the Python environment from a lockfile, get MuJoCo and both SO-101 simulation models running, create the accounts the course pushes artifacts to, and prove that the whole stack works by loading a real robot dataset on your Mac. Nothing here is conceptually difficult, but every later lesson assumes it has been done.

| | |
|---|---|
| **Phase** | 0 — Bootstrap |
| **Time** | ~half a day desk time; hardware ships in 1–2 weeks, which is why it is ordered first |
| **Cost** | ~$375 hardware order placed now; $0 compute (everything here runs Mac-local) |
| **Prerequisites** | none |
| **Feeds into** | every lesson (the environment), H1 (the hardware), 01 (the hello-dataset script is the starting point for the format study) |

## Learning objectives

After this lesson you can:

1. **Justify** the hardware order you placed: which SO-101 kit, which cameras, and what trade-off each choice made.
2. **Reproduce** your Python environment from a lockfile on a fresh machine in under five minutes.
3. **Run** MuJoCo's interactive viewer on macOS, load an SO-101 model from both available sources, and explain when to use which.
4. **Load** `lerobot/svla_so101_pickplace` with `delta_timestamps` and read every field of the returned sample.
5. **Publish** the course repo with the scaffold every later lesson drops into.

## Background

### What runs where

The course splits its computation in one fixed way. Interactive work, meaning MuJoCo, dataset inspection, policy inference and small training runs, happens locally on the Mac's `mps` device. Heavy training rents a Linux machine with an NVIDIA GPU from Vast.ai or RunPod, and artifacts travel between the two through the Hugging Face Hub. Nothing in the core track requires a local GPU.

### Version policy

The course targets LeRobot 0.6.1 or newer, which requires Python 3.12 or newer and PyTorch between 2.7 and 2.11. The course backbone, the LeRobot tutorial, was written against version 0.4.0 and no longer runs verbatim, so each lesson notes where the API has moved. You will meet two such changes today. The first is the split of optional dependencies into extras: the base package no longer bundles the training dependencies, so you install `"lerobot[training]"`. The second is the dataset codebase version: the v3 file layout is current, but many datasets on the Hub still carry the older v2.1 layout, and Part 5 explains what to do when you meet one.

### Why `uv`

The environment is managed with `uv` because it produces a deterministic lockfile, resolves dependencies quickly, and its `uv run` command activates the environment per command. Those properties matter here because you will rebuild this environment on every cloud machine you rent.

### Why `mjpython`

On macOS, MuJoCo's interactive viewer must own the process's main thread, which plain `python` cannot guarantee; running the viewer under plain `python` raises `launch_passive requires that the Python script be run under mjpython on macOS`. Offscreen rendering, which Lesson 02 uses, has no such requirement and works under plain `python`. On this particular machine `mjpython` itself is broken in two ways, and the Pitfalls table records both the fixes and the workaround.

### Two SO-101 simulation models

Two MuJoCo models of the SO-101 exist, and the course uses both for different purposes. The first is the curated model in DeepMind's menagerie, which has tuned contact physics and is the right choice for the classical lessons. The second is the model in TheRobotStudio's repository, whose joint zeros and signs match LeRobot's calibration convention, which makes it the right choice for anything that has to agree with the real arm or with LeRobot datasets.

| Source | File | Use when |
|---|---|---|
| `google-deepmind/mujoco_menagerie` | `trs_so_arm100/so_arm100.xml` + `scene.xml` | curated physics (tuned contacts, convex gripper collision meshes), MuJoCo ≥ 3.1.6, Apache-2.0 — Lessons 03–07 |
| `TheRobotStudio/SO-ARM100` | `Simulation/SO101/so101_new_calib.xml` | joint zero/sign conventions match LeRobot's calibration — anything that must agree with the real arm or with LeRobot datasets (Lesson 02, H-track) |

| Source | Read for |
|---|---|
| [LeRobot installation docs](https://huggingface.co/docs/lerobot/installation) | current extras names and the source-install fallback |
| [LeRobot SO-101 page](https://huggingface.co/docs/lerobot/so101) | the assembly guide you'll follow in H1; skim it now to sanity-check your kit order |
| `uv` docs (astral.sh/uv) | `uv venv` / `uv pip` / lockfile workflow |

## Part 1 — Order the hardware (30 min, do this first)

The kit's lead time is one to two weeks, and every day of delay pushes the whole hardware track back, so this is the first thing to do.

1. Pick a kit:

| Kit | Price | Trade-off |
|---|---|---|
| **Seeed SO-ARM101 Pro (US warehouse)** — default | $289 | fast US shipping, both arms' servos, known-good QC |
| WowRobo | ~$199–269 | cheapest, ships from China (adds 1–3 weeks, which usually defeats the purpose) |
| Partabot US full kit | $329 (assembled: $479) | includes printed parts; buy if in stock and you don't want to deal with printing. Assembled skips H1's build steps — chosen path, see `hardware/ORDER.md` |

2. Printed parts: the official set costs $35, or about $20 of PLA+ if you have printer access (`STL/` in the SO-ARM100 repo; print the gauge in `STL/Gauges/` first to verify tolerances).
3. Cameras: an InnoMaker 1080p for the wrist plus any second 1080p webcam **of a different model**. Two identical models are hard to tell apart when the operating system enumerates them, and that becomes a recurring nuisance in H1. Budget about $50 for both.
4. Log the order numbers and ETAs in `hardware/ORDER.md`.

**✅ Checkpoint:** both order confirmations exist; ETA is on your calendar; total ≤ ~$375 on the DIY path (the assembled Partabot path runs ~$610 all-in; actuals in `hardware/ORDER.md`). ✅ Done 2026-09-01.

## Part 2 — Python environment (30 min)

1. Install `uv`, then:
   ```bash
   mkdir -p ~/robot-learning && cd ~/robot-learning
   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install "lerobot[training]" mujoco
   uv pip freeze > requirements.lock
   ```
   If the extras name is rejected, the extras split has moved again. Check the installation docs page above rather than guessing.
2. Verify:
   ```bash
   python -c "import lerobot, torch; print(lerobot.__version__, torch.__version__, torch.backends.mps.is_available())"
   ```

**✅ Checkpoint:** prints lerobot ≥ 0.6.1, torch in 2.7–2.11, `True` for MPS. `requirements.lock` is committed. ✅ Done 2026-09-01: lerobot 0.6.1, torch 2.11.0, MPS `True`, Python 3.12.12.

## Part 3 — MuJoCo + models (30 min)

1. ```bash
   git clone https://github.com/google-deepmind/mujoco_menagerie ~/models/mujoco_menagerie
   git clone https://github.com/TheRobotStudio/SO-ARM100 ~/models/SO-ARM100
   python -m mujoco.viewer --mjcf ~/models/mujoco_menagerie/trs_so_arm100/scene.xml
   ```
   Use `python` here, not `mjpython`. The managed viewer runs on the main thread without it, and on this machine `mjpython`'s UI-thread bridge is broken (see Pitfalls). `mjpython` is still required for `launch_passive` in later lessons.
2. In the viewer, drag each joint. Count the degrees of freedom: five revolute joints plus the gripper. Cross-reference the names against the dataset's state features in Part 5 (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`).
3. Load `~/models/SO-ARM100/Simulation/SO101/so101_new_calib.xml` the same way and note any joint-zero differences from the menagerie model by moving a joint to its zero in both.

**✅ Checkpoint:** both models open; you can name all six actuated DoF and have written one sentence in `setup.md` on which model you'd use for Lesson 02 and why. ✅ Done 2026-09-01 — both models verified (nq=6, 6 actuators); `setup.md` written.

## Part 4 — Accounts (15 min)

```bash
hf auth login   # use a WRITE token — you push datasets/checkpoints all course
wandb login
```
Note that `huggingface-cli` no longer works in `huggingface_hub` 1.x: the binary still installs but only prints a deprecation error. All authentication subcommands now live under `hf auth`, and uploads use `hf upload`.

**✅ Checkpoint:** `hf auth whoami` shows your username; `wandb login` confirms. ✅ Done 2026-09-01 — HF `natuanand93` (token `robot-learning-token`), W&B `natu-anand`; `check_no_leaked_keys.sh` passes (run it after any credential change).

## Part 5 — Hello-dataset (1 h)

This is the first time you touch the data format that the whole course is built on. The dataset is `lerobot/svla_so101_pickplace`: 50 episodes, 11,939 frames at 30 fps, two cameras (`observation.images.up` and `observation.images.side`, 480×640, AV1-encoded), and 6-dimensional state and action vectors covering the five joints plus the gripper, matching the names you saw in Part 3.

1. Write `hello_dataset.py`:
   ```python
   from lerobot.datasets.lerobot_dataset import LeRobotDataset
   ds = LeRobotDataset(
       "lerobot/svla_so101_pickplace",
       delta_timestamps={"observation.images.up": [-2/30, -1/30, 0.0]},
   )
   print(ds.meta.fps, ds.meta.total_episodes, ds.meta.total_frames)
   item = ds[100]
   for k, v in item.items():
       print(k, getattr(v, "shape", None), getattr(v, "dtype", type(v)))
   ```
2. Expected output: fps 30, 50 episodes, 11,939 frames; `observation.images.up` has shape `(3, 3, 480, 640)` (a history of T=3 frames), and state and action are 6-dimensional. If loading fails with a dataset-version or backward-compatibility error, the Hub copy is still codebase v2.1. Convert it once and move on; the details are Lesson 01 material, and today it only needs to load:
   ```bash
   python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id=lerobot/svla_so101_pickplace
   ```
   Verify the exact module path with `python -m lerobot.scripts --help` or the datasets doc page, because converter locations have moved between minor versions.
3. Render a 3×3 grid of `up`-camera frames spanning one episode and save it as `hello_grid.png`.

**✅ Checkpoint:** script runs clean on `mps`, shapes match the above, grid PNG shows a pick-place progressing left-to-right. ✅ Done 2026-09-01 — all numbers/shapes matched; no v2.1→v3 conversion needed (Hub copy loads directly); grid verified.

## Part 6 — Repo scaffold + post #0 (45 min)

1. Create the scaffold that every lesson directory will follow:
   ```
   robot-learning/
     README.md  TEMPLATE.md  requirements.lock  setup.md
     lessons/NN-slug/   → README.md (the assignment) + code + RESULTS.md (the evidence)
     hardware/HN-slug/  → same convention
   ```
2. Push the repository public. The convention is one branch per lesson and one merged PR per lesson, with the PR description linking `RESULTS.md`.
3. Outline post #0, to be published wherever you build in public: what the course is; the arc in five sentences (classical → RL → generative imitation → VLAs → beyond); the budget table; the public-accountability rule that every lesson ends in a merged PR; and what post #1 will be (Lesson 01's format study).

**✅ Checkpoint:** repo is public, post #0 is live, `setup.md` records every version pin. ✅ Done 2026-09-01 — repo public, `setup.md` pins recorded; post #0 deliberately skipped (journaled).

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `setup.md` | OS/Python/lerobot/mujoco/torch versions; the two-model comparison sentence; account names |
| `requirements.lock` | fresh `uv venv` + install from it reproduces the env |
| `hello_dataset.py` + `hello_grid.png` | runs clean on `mps`; grid renders |
| `hardware/ORDER.md` | kit choice + rationale, order numbers, ETAs |
| Public repo + post #0 | scaffold as above; post covers the outline |

## Done when

- [x] Hardware ordered; ETA logged.
- [x] Both SO-101 models open in the viewer (plain `python -m mujoco.viewer`; `mjpython` broken on this machine, see Pitfalls).
- [x] `hello_dataset.py` runs clean on `mps` with correct shapes.
- [x] Repo public with scaffold and lockfile; post #0 deliberately skipped (journaled).

## Self-check

1. Why does the interactive viewer need `mjpython` on macOS when Lesson 02's offscreen rendering does not?
2. Which SO-101 model matches LeRobot's calibration convention, and what is one concrete bug you would cause by using the other one for real-arm work?
3. What does the `[training]` extra pull in that the base `lerobot` package does not, and why does the split exist?
4. Your `delta_timestamps` asked for `[-2/30, -1/30, 0.0]`. Why must these be multiples of 1/fps, and what parameter governs the slack? (The full answer is Lesson 01; you should have a hypothesis now.)

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `launch_passive requires ... mjpython` | interactive viewer under plain `python` on macOS | use `mjpython` |
| Video decode errors / green frames on `svla_so101_pickplace` | AV1-encoded MP4s vs old ffmpeg/pyav | `brew install ffmpeg` (≥ 6), reinstall `av`; or use the torchcodec backend if your lerobot version exposes it |
| `403 Forbidden` pushing to Hub later | READ-scoped token | re-login with a WRITE token |
| Dataset load fails with version/compat error | Hub copy is codebase v2.1, your lerobot expects v3 | run the v2.1→v3 converter (Part 5 step 2) |
| `uv pip install` succeeds but `python` finds nothing | venv not activated / wrong interpreter | `which python` must point into `.venv`; or prefix commands with `uv run` |
| Menagerie model errors on load | MuJoCo < 3.1.6 | `uv pip install -U mujoco` |
| `mjpython` dies with `dlopen ... libpython3.12.dylib` not loaded | uv's standalone CPython keeps the dylib outside the rpaths mjpython's app bundle searches | `ln -sf ~/.local/share/uv/python/cpython-<ver>-macos-aarch64-none/lib/libpython3.12.dylib .venv/libpython3.12.dylib` (`.venv/bin/../` is on the search list) |
| `mjpython -m mujoco.viewer` → `RuntimeError: Caught an unknown exception!` in `_Simulate` | mjpython's UI-thread bridge broken on this machine (mujoco 3.11/3.12, uv and Homebrew Pythons alike; raw GLFW works) | use plain `python -m mujoco.viewer` for the managed viewer; retest mjpython when a lesson first needs `launch_passive` |

## References

- LeRobot docs: installation, SO-101 assembly, dataset v3 pages (huggingface.co/docs/lerobot).
- `google-deepmind/mujoco_menagerie` (`trs_so_arm100/`, Apache-2.0, requires MuJoCo ≥ 3.1.6).
- `TheRobotStudio/SO-ARM100` (`Simulation/SO101/so101_new_calib.xml`, `STL/`, `Optional/` camera mounts).
- Dataset card: `lerobot/svla_so101_pickplace`.
