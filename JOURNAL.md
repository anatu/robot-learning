# Course Journal

Cross-lesson log. Per-lesson numbers and interpretation live in each lesson's `RESULTS.md`; this file holds what spans lessons: environment and tooling fixes, cloud-provider notes, cost tracking, plan deviations, and transferable learnings.

## Running totals

| | |
|---|---|
| Cloud GPU spend | $0 |
| Hardware spend | $608.49 |
| API spend | $0 |
| Core lessons done | 0 / 23 |
| Hardware lessons done | 0 / 6 |

## Entries

Newest first. Format: `### YYYY-MM-DD — [NN / HN / meta] title` + up to 5 bullets.

### 2026-09-01 — [00] mjpython broken on this Mac — dlopen fix + viewer workaround

- Stage 1: `mjpython` died on `dlopen ... @rpath/libpython3.12.dylib` (uv's CPython keeps the dylib outside the app bundle's rpaths). Fix: `ln -sf ~/.local/share/uv/python/cpython-3.12.12-macos-aarch64-none/lib/libpython3.12.dylib .venv/libpython3.12.dylib`. Symlink lives in gitignored `.venv` — recreate after any venv rebuild.
- Stage 2: with dlopen fixed, `mjpython -m mujoco.viewer` still dies in `_Simulate` ("Caught an unknown exception") — reproduced on mujoco 3.12.0 and 3.11.0, uv and Homebrew-framework Pythons; raw GLFW window creation works, so it's mjpython's UI-thread bridge, not the window server. No matching upstream issue found (Darwin 25.6).
- Workaround: the managed viewer runs fine as plain `python -m mujoco.viewer` (verified, window opens). Lesson 00 Part 3 updated; both failure modes added to its Pitfalls.
- Open risk: `launch_passive` genuinely requires mjpython — retest at the first lesson that needs it (likely 03/04), and file/find an upstream issue if still broken.

### 2026-09-01 — [00] Lesson 00 started — Parts 1–3 complete

- Cameras ordered: EMEET C960 (overhead) + InnoMaker U20CAM-1080P (wrist), $51.70 all-in, ETA Fri 2026-09-04; details in `hardware/ORDER.md` (moved there from repo root, where Part 1 expects it). Running totals updated.
- With the assembled Partabot arm, hardware totals $608.49 vs the lesson's ≤ ~$375 checkpoint — deliberate deviation ($150 assembly premium + shipping/tax); Lesson 00 README updated to note the assembled path.
- Part 2 verified: lerobot 0.6.1, torch 2.11.0, MPS available, Python 3.12.12; `requirements.lock` committed.
- Part 3 verified: menagerie `trs_so_arm100` and `so101_new_calib` both load and open (nq=6, 6 actuators each); `so101_new_calib` joint names match LeRobot dataset features verbatim → chosen for Lesson 02, rationale in `lessons/00-setup/setup.md`.
- Viewer runs via plain `python -m mujoco.viewer` — mjpython is broken on this machine, see the dedicated entry above.

### 2026-09-01 — [meta] SO-101 ordered — Partabot, assembled

- Ordered SO-ARM101 Full Kit / Assembled from Partabot: $556.79 all-in. Details in `hardware/ORDER.md`.
- Assembled over DIY: H1's build steps collapse to unbox + calibrate + teleop; log the deviation in H1's `RESULTS.md` when started.
- README budget updated to actuals: arm line $324 est → $556.79; projected total ~$425 → ~$657.
- Running totals updated (hardware spend $556.79).

### 2026-09-01 — [meta] Skild S1 folded into Lesson 20

- Reviewed Skild's S1 blog (in-context video demo → actions, no fine-tune; claimed 66% vs 9% OOD at ~100k pre-training hours) against the curriculum: no redesign — closed weights, vendor-only evidence, fundamentals unchanged.
- Lesson 20 gains a task-specification axis in the survey note (language → goal image → sketch → in-context demo; readings RT-Trajectory, Vid2Robot, ICRT) and a new Part 4: S1 claims audit ending in a dated, falsifiable expectation.
- README: Lesson 20 deliverable line updated; S1 added to the frontier reading spine.
- Transferable rule: a frontier lab blog changes reading lists and exercises, not course structure, until weights/paper/independent eval exist.

### 2026-08-11 — [meta] Journal created

- Curriculum quality pass published (`b11325f`); course not yet started.
- Next: Lesson 00 (setup + order hardware — lead time overlaps Phases 1–2).
