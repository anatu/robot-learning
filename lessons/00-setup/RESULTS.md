# Lesson 00 — Results

Completed 2026-09-01. All six parts done; post #0 deliberately skipped (see JOURNAL).

| Part | Evidence |
|---|---|
| 1 — hardware ordered | `hardware/ORDER.md`: Partabot SO-ARM101 assembled ($556.79) + 2 cameras ($51.70); ETAs on calendar |
| 2 — Python env | `requirements.lock` (repo root); lerobot 0.6.1 / torch 2.11.0 / MPS on Python 3.12.12 |
| 3 — MuJoCo + models | `setup.md`: both SO-101 models load, 6 DoF verified; `so101_new_calib` chosen for Lesson 02 |
| 4 — accounts | HF + W&B logged in; `check_no_leaked_keys.sh` passes (5 checks) |
| 5 — hello-dataset | `hello_dataset.py` (shapes/fps/episodes match spec), `hello_grid.png` (ep0 pick-place progression) |
| 6 — scaffold | repo public at github.com/anatu/robot-learning; post #0 skipped |

Deviations from the lesson README (all journaled):

- Assembled Partabot kit ($479) over DIY — total $608.49 vs the ≤ ~$375 checkpoint; H1 becomes unbox + calibrate.
- `mjpython` broken on this machine two ways (dlopen fix + `_Simulate` crash) — managed viewer runs via plain `python -m mujoco.viewer`.
- `huggingface-cli` dead in huggingface_hub 1.29 — all commands migrated to `hf`.
- No v2.1→v3 dataset conversion needed; torchcodec dylib broken → pyav fallback.
- Post #0 not published; lesson worked as direct commits to main rather than a single PR.
