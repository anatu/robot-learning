# Lesson 00 — Setup & Repo Bootstrap

**Goal:** working toolchain, repo hygiene, and hardware ordered so lead time overlaps the theory lessons.

## Tasks
1. **Order hardware now** (arrives in 1–2 weeks): Seeed SO-ARM101 Pro kit ($289) + printed parts ($35) + InnoMaker 1080p wrist cam + a second (different-model) 1080p webcam. Alternatives: WowRobo kit (~$199–269, ships from China) or Partabot US full kit ($329) if in stock.
2. Python ≥ 3.12 env via `uv`; `pip install "lerobot[training]"` (v0.6.x — note the extras split introduced in v0.6.0).
3. MuJoCo native install; verify the interactive viewer via `mjpython`. Clone `mujoco_menagerie` (has `trs_so_arm100`) and TheRobotStudio `SO-ARM100` repo (`Simulation/SO101/so101_new_calib.xml` — matches LeRobot's calibration convention).
4. HF account + `huggingface-cli login`; W&B account for training logs.
5. Hello-dataset script: load `lerobot/svla_so101_pickplace` with `delta_timestamps`, print sample structure, render a frame grid.

## Deliverables
- `setup.md` documenting the environment (versions pinned), plus the hello-dataset script and its output.
- Repo scaffold pushed to GitHub; short post #0: what this course is, the plan, the budget.

## Done when
`mjpython` opens an SO-101 scene, the hello-dataset script runs clean on `mps`, and the repo is public.
