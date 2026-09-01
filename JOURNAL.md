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

### 2026-09-01 — [00] Lesson 00 started — Parts 1–2 complete

- Cameras ordered: EMEET C960 (overhead) + InnoMaker U20CAM-1080P (wrist), $51.70 all-in, ETA Fri 2026-09-04. Details in `hardware/ORDER.md`.
- With the assembled Partabot arm, hardware totals $608.49 vs the lesson's ≤ ~$375 checkpoint — deliberate deviation ($150 assembly premium + shipping/tax); Lesson 00 README updated to note the assembled path.
- `ORDER.md` moved to `hardware/ORDER.md` where Lesson 00 Part 1 expects it.
- Running totals updated.
- Part 2 verified: lerobot 0.6.1, torch 2.11.0, MPS available, Python 3.12.12; `requirements.lock` committed.

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
