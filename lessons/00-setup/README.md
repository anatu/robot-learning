# Lesson 00 — Setup & Repo Bootstrap

Stand up the full toolchain, order the hardware so its lead time overlaps the theory lessons, and prove the stack works end-to-end by loading a real robot dataset on your Mac.

| | |
|---|---|
| **Phase** | 0 — Bootstrap |
| **Time** | ~half a day desk time; hardware ships in 1–2 weeks (that's why it's ordered *today*) |
| **Cost** | ~$375 hardware order placed now; $0 compute (everything here runs Mac-local) |
| **Prerequisites** | none |
| **Feeds into** | every lesson (the environment), H1 (the hardware), 01 (the hello-dataset script grows into the parser) |

## Learning objectives

After this lesson you can:

1. **Justify** the hardware order you placed — which SO-101 kit, which cameras, and what trade-off each choice made.
2. **Reproduce** your Python environment from a lockfile on a fresh machine in under five minutes.
3. **Run** MuJoCo's interactive viewer on macOS via `mjpython` and load an SO-101 model from both available sources, and explain when to use which.
4. **Load** `lerobot/svla_so101_pickplace` with `delta_timestamps` and read every field of the returned sample.
5. **Publish** the course repo with the scaffold every later lesson drops into.

## Background

**What runs where.** The course's split, fixed here once: interactive work (MuJoCo, dataset inspection, policy inference, small training) runs Mac-local on `mps`; heavy training rents a Linux+NVIDIA box (Vast.ai/RunPod) and round-trips artifacts through the HF Hub. Nothing in the core track requires a local GPU.

**Version policy.** LeRobot ≥ 0.6.1 (Python ≥ 3.12, PyTorch 2.7–2.11). The course backbone — the tutorial — targets v0.4.0 and no longer runs verbatim; each lesson notes the deltas. Two you'll hit today: the extras split (`pip install "lerobot[training]"` — the base package no longer bundles training deps) and dataset codebase versions (v3 file layout vs the v2.1 many datasets on the Hub still carry — see Part 5).

**Why `uv`.** Deterministic lockfile, fast resolves, and `uv run` gives per-command env activation — matters because you'll rebuild this env on every rented cloud box.

**Why `mjpython`.** On macOS, MuJoCo's interactive viewer must own the main thread; plain `python` raises `launch_passive requires that the Python script be run under mjpython on macOS`. Offscreen rendering (what Lesson 02 uses) works under plain `python`.

**Two SO-101 sim models exist; know both:**

| Source | File | Use when |
|---|---|---|
| `google-deepmind/mujoco_menagerie` | `trs_so_arm100/so_arm100.xml` + `scene.xml` | curated physics (tuned contacts, convex gripper collision meshes), MuJoCo ≥ 3.1.6, Apache-2.0 — Lessons 03–07 |
| `TheRobotStudio/SO-ARM100` | `Simulation/SO101/so101_new_calib.xml` | joint zero/sign conventions match LeRobot's calibration — anything that must agree with the real arm or with LeRobot datasets (Lesson 02, H-track) |

| Source | Read for |
|---|---|
| [LeRobot installation docs](https://huggingface.co/docs/lerobot/installation) | current extras names and the source-install fallback |
| [LeRobot SO-101 page](https://huggingface.co/docs/lerobot/so101) | the assembly guide you'll follow in H1 — skim now to sanity-check your kit order |
| `uv` docs (astral.sh/uv) | `uv venv` / `uv pip` / lockfile workflow |

## Part 1 — Order the hardware (30 min, do this first)

Lead time is 1–2 weeks; every day of delay pushes the whole H-track.

1. Pick a kit:

| Kit | Price | Trade-off |
|---|---|---|
| **Seeed SO-ARM101 Pro (US warehouse)** — default | $289 | fast US shipping, both arms' servos, known-good QC |
| WowRobo | ~$199–269 | cheapest, ships from China (adds 1–3 weeks — usually defeats the purpose) |
| Partabot US full kit | $329 (assembled: $479) | includes printed parts; buy if in stock and you don't want to deal with printing. Assembled skips H1's build steps — chosen path, see `hardware/ORDER.md` |

2. Printed parts: official set $35, or ~$20 of PLA+ if you have printer access (`STL/` in the SO-ARM100 repo; print the gauge in `STL/Gauges/` first to verify tolerances).
3. Cameras: InnoMaker 1080p (wrist) + any second 1080p webcam **of a different model** (identical models make USB device disambiguation miserable in H1). ~$50 total.
4. Log order numbers + ETAs in `hardware/ORDER.md`.

**✅ Checkpoint:** both order confirmations exist; ETA is on your calendar; total ≤ ~$375 (DIY path — the assembled Partabot path runs ~$610 all-in; actuals in `hardware/ORDER.md`). ✅ Done 2026-09-01.

## Part 2 — Python environment (30 min)

1. Install `uv`, then:
   ```bash
   mkdir -p ~/robot-learning && cd ~/robot-learning
   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install "lerobot[training]" mujoco
   uv pip freeze > requirements.lock
   ```
   If the extras name is rejected, the extras split has moved again — check the installation docs page above; do not guess.
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
   (`python`, not `mjpython`: the managed viewer runs fine on the main thread, and on this machine `mjpython`'s UI-thread bridge is broken — see Pitfalls. `mjpython` is still required for `launch_passive` in later lessons.)
2. In the viewer: drag each joint. Count DoF — 5 revolute joints + gripper. Cross-reference the names against the dataset's state features in Part 5 (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`).
3. Load `~/models/SO-ARM100/Simulation/SO101/so101_new_calib.xml` the same way and note any joint-zero differences from the menagerie model (move a joint to its zero in both).

**✅ Checkpoint:** both models open; you can name all six actuated DoF and have written one sentence in `setup.md` on which model you'd use for Lesson 02 and why. ✅ Done 2026-09-01 — both models verified (nq=6, 6 actuators); `setup.md` written.

## Part 4 — Accounts (15 min)

```bash
huggingface-cli login   # use a WRITE token — you push datasets/checkpoints all course
wandb login
```
(Current `huggingface_hub` also ships the shorter `hf` CLI; either works — pick one and use it consistently.)

**✅ Checkpoint:** `huggingface-cli whoami` shows your username; `wandb login` confirms.

## Part 5 — Hello-dataset (1 h)

The first contact with the substrate of the whole course. Dataset: `lerobot/svla_so101_pickplace` — 50 episodes, 11,939 frames, 30 fps, two cameras (`observation.images.up`, `observation.images.side`, 480×640, AV1-encoded), 6-D state/action (the five joints + gripper, matching Part 3).

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
2. Expected: fps 30, 50 episodes, 11,939 frames; `observation.images.up` is `(3, 3, 480, 640)` (T=3 history), state/action are 6-D. If loading fails with a dataset-version/backward-compatibility error, the Hub copy is still codebase v2.1 — convert once and move on (this is Lesson 01 material; today it just needs to load):
   ```bash
   python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id=lerobot/svla_so101_pickplace
   ```
   (verify the exact module path with `python -m lerobot.scripts --help` or the datasets doc page — converter locations have moved between minor versions).
3. Render a 3×3 grid of `up`-camera frames spanning one episode; save as `hello_grid.png`.

**✅ Checkpoint:** script runs clean on `mps`, shapes match the above, grid PNG shows a pick-place progressing left-to-right.

## Part 6 — Repo scaffold + post #0 (45 min)

1. Scaffold (every lesson directory will follow this):
   ```
   robot-learning/
     README.md  TEMPLATE.md  requirements.lock  setup.md
     lessons/NN-slug/   → README.md (the assignment) + code + RESULTS.md (the evidence)
     hardware/HN-slug/  → same convention
   ```
2. Push public. Branch-per-lesson, one merged PR per lesson — the PR description links `RESULTS.md`.
3. Post #0 outline (publish anywhere you build in public): what the course is; the arc in five sentences (classical → RL → generative imitation → VLAs → beyond); the budget table; the public-accountability rule (every lesson ends in a merged PR); what post #1 will be (Lesson 01's format spec).

**✅ Checkpoint:** repo is public, post #0 is live, `setup.md` records every version pin.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `setup.md` | OS/Python/lerobot/mujoco/torch versions; the two-model comparison sentence; account names |
| `requirements.lock` | fresh `uv venv` + install from it reproduces the env |
| `hello_dataset.py` + `hello_grid.png` | runs clean on `mps`; grid renders |
| `hardware/ORDER.md` | kit choice + rationale, order numbers, ETAs |
| Public repo + post #0 | scaffold as above; post covers the outline |

## Done when

- [ ] Hardware ordered; ETA logged.
- [ ] `mjpython` opens both SO-101 models.
- [ ] `hello_dataset.py` runs clean on `mps` with correct shapes.
- [ ] Repo public with scaffold, lockfile, and post #0 live.

## Self-check

1. Why does the interactive viewer need `mjpython` on macOS but Lesson 02's offscreen rendering won't?
2. Which SO-101 model matches LeRobot's calibration convention, and name one concrete bug you'd cause by using the other one for real-arm work.
3. What does the `[training]` extra pull in that the base `lerobot` package doesn't, and why does the split exist?
4. Your `delta_timestamps` asked for `[-2/30, -1/30, 0.0]`. Why must these be multiples of 1/fps, and what parameter governs the slack? (Full answer is Lesson 01; you should have a hypothesis now.)

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
