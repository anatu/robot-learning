# Course Journal

Cross-lesson log. Per-lesson numbers and interpretation live in each lesson's `RESULTS.md`; this file holds what spans lessons: environment and tooling fixes, cloud-provider notes, cost tracking, plan deviations, and transferable learnings.

## Running totals

| | |
|---|---|
| Cloud GPU spend | $0 |
| Hardware spend | $608.49 |
| API spend | $0 |
| Core lessons done | 1 / 23 |
| Hardware lessons done | 0 / 6 |

## Entries

Newest first. Format: `### YYYY-MM-DD — [NN / HN / meta] title` + up to 5 bullets.

### 2026-09-02 — [meta] Course restructured: principles-first, AI-assisted coding

- Why: several years away from robotics code plus current AI coding tools make hand-implementing parsers, planners, and RL agents low-yield per hour; the learning value is in principles and practical exercises (predict → run → reconcile, diagnose, decide). Every lesson was re-derived from that premise.
- `TEMPLATE.md` rewritten: `Principles` (+ **Carry forward** block) replaces Background; numbered `## Exercise N — name [Type]` with eight exercise types ([Derive] [Build] [Read the kernel] [Predict → Run] [Diagnose] [Decide] [Read] [Write]) replaces Parts; new execution contract — code is AI-drafted from a student-written spec + check; non-delegable: predictions, interpretation, derivations, kernel annotations, diagnoses, decisions, pre-registration docs. `CLAUDE.md` division of labor flipped to match. `PRINCIPLES.md` added (one line per principle, from each lesson's Carry-forward block).
- All 28 remaining lesson READMEs (01–22, H1–H6) rewritten. Biggest coding cuts: 01 byte-level parser → `window()` + field guide; 03 tests → one check script, constrained IK to Going deeper; 08 from-scratch `SACAgent` package → vendored CleanRL single-file scripts, annotated (09 patches `sac.py`, 11 reuses it); 14 from-scratch ensembler → annotate LeRobot's; 15 `dp_cfm/` fork → FM-vs-DDPM few-step study on Lesson 12's heads; 19 three models + PDF paper → two models + `leaderboard.md`. Grids shrunk everywhere to the minimum that shows the effect (09: 18 → 8 runs; 11: 12 → 6 arms, 49 → 25 cells; 14: 4 → 3 chunk arms; 16: 15 → 8 sweep cells). Hardware lessons keep every command and protocol; H1 Part 1 is now unbox + inspect (assembled kit).
- Stub scaffold retired: the ~75 Python/test/config stubs from 2026-09-02's first entry encode the old "implement yourself" contract and were deleted (`git rm`, 93 paths). Markdown deliverable skeletons (`NOTE.md`, `PROTOCOL.md`, `TASK.md`, `FORMAT.md`, logs, sheets) kept, old-contract headers stripped, sizes updated (17 NOTE ≤ 2 pp, 20 NOTE 2–3 pp, 19 pair not trio).
- Transferable rule: a from-scratch implementation earns its place only when the code *is* the principle (iLQR backward pass, SAC updates, generative-head losses, the π0 mask) — and even then, annotation against the equations + the checkpoint test is the requirement; typing it is optional.

### 2026-09-02 — [meta] Execution contract + course-wide stub scaffold

- Added `TEMPLATE.md` § Execution contract: bash blocks = terminal verbatim; unnamed python blocks = run-once; named modules/signatures = student-authored files; deliverables are the student's, Claude debugs/scaffolds/verifies/reviews. `CLAUDE.md` gained the division-of-labor pointer. Motivation: lesson instructions were ambiguous about file-vs-terminal-vs-delegate.
- Scaffolded stubs for all 28 remaining lessons (01–22, H1–H6): ~90 files — Python modules with README signatures + `NotImplementedError`, module-level-skipped pytest suites, heading-skeleton markdown deliverables (NOTE/PROTOCOL/FORMAT/logs). All compile; `pytest lessons hardware` = 24 skipped, 0 errors.
- pytest added to the env; `requirements.lock` refreshed.
- Known judgment calls are documented in each stub's docstring; XML/MJCF assets and notebooks were not stubbed (no sensible stub form).

### 2026-09-01 — [00] Lesson 00 complete (1/23)

- All six parts done in one day; evidence table in `lessons/00-setup/RESULTS.md`, version pins + accounts in `setup.md`.
- Deviation: post #0 skipped by choice — building in public via the public repo only for now; revisit if/when a blog exists.
- Deviation: lesson ran as direct commits to `main`, not the branch-per-lesson PR convention — decide from Lesson 01 whether to adopt PRs.

### 2026-09-01 — [00] Part 5 done — hello-dataset

- `hello_dataset.py`: 30 fps / 50 episodes / 11,939 frames; all shapes as documented, `delta_timestamps` stack + `_is_pad` mask working. No v2.1→v3 conversion needed — the Hub copy loads directly, lesson step 2 is skippable.
- `hello_grid.png`: episode 0 (303 frames, 9 sampled), full pick-place visible ("pink lego brick into the transparent box").
- matplotlib was missing from the env — installed, `requirements.lock` refreshed.
- torchcodec's dylib fails to load on this Mac; lerobot silently falls back to pyav (works, slower). Revisit before dataloader-heavy lessons (01/02, 14+).

### 2026-09-01 — [00] Part 4 done + credential hygiene test

- HF login verified (`natuanand93`, dedicated `robot-learning-token`); W&B verified (`natu-anand`).
- Added `lessons/00-setup/check_no_leaked_keys.sh`: scans working tree + full git history for token patterns, enforces 600 perms on credential files, checks shell rc files and tracked `.env`s. Passing.
- Found and fixed: `~/.cache/huggingface/token` and `stored_tokens` were world-readable 644 → chmod 600.
- Watch-out: W&B session is bound to the `natu-anand-descript` entity — pass a personal `--entity`/`WANDB_ENTITY` on course runs or they land in that org.

### 2026-09-01 — [meta] huggingface-cli is dead — use `hf`

- `huggingface-cli` in `huggingface_hub` 1.29.0 is a stub that only prints a deprecation error (verified). Replacement: `hf`, auth under `hf auth` (`hf auth login`, `hf auth whoami`), uploads via `hf upload`.
- Fixed in lessons 00 (login/whoami), 14 (login + upload), 18 (upload) — all `huggingface-cli` references in the repo.

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
