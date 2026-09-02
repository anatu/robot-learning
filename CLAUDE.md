# CLAUDE.md

Self-study robot-learning course. Map: `README.md`. Lesson format and execution contract: `TEMPLATE.md`. One lesson = one directory = one PR, with a `RESULTS.md` per lesson.

## Course philosophy (read before touching a lesson)

Principles and practical exercises are the point; code is instrumental. Each lesson names its principles, then runs typed exercises ([Predict → Run], [Diagnose], [Decide], [Derive], [Read the kernel], [Build], [Read], [Write]). Code is AI-assisted from a spec the user writes. From-scratch reimplementation lives under "Going deeper" and never gates progression.

## Division of labor

- **User (non-delegable):** predictions written before runs; `RESULTS.md` interpretation; derivations; kernel annotations; diagnoses; decisions and their defense; pre-registration documents (`PROTOCOL.md`, `TASK.md`); self-check answers.
- **Claude:** drafts code from the user's spec; debugs the environment; runs checkpoints and parity checks; reviews drafts; keeps the journal and bookkeeping. Claude does not write the user's predictions, interpretations, or decisions, and does not fill in a [Derive] or [Read the kernel] exercise.
- When asked to "do the exercise", draft the code and the check, then stop and hand back for the prediction/interpretation step.

## Journal discipline (`JOURNAL.md`)

Append a journal entry in the same session whenever any of these happens:

- An environment or tooling problem gets solved — record the fix in one line so it is never re-debugged.
- Money is spent (GPU rental, hardware, API) — also update the running-totals table.
- A lesson is started, completed, or deviates from its README — what changed and why.
- A transferable learning surfaces that doesn't belong to a single lesson's `RESULTS.md` (cloud-provider quirks, LeRobot version drift, workflow improvements).

Entry rules:

- Newest first under `## Entries`: `### YYYY-MM-DD — [NN / HN / meta] title` + up to 5 bullets.
- Journal is cross-lesson; `RESULTS.md` is per-lesson. Never duplicate content between them — link instead.
- Update the running-totals table in place, don't append rows.
- On lesson completion: tick the lesson's checkbox in `README.md`'s Progress section and increment the journal's done counters in the same commit.

## Conventions

- Costs in USD, actuals not estimates.
- Version drift discovered against a lesson README (LeRobot flags, model IDs) gets fixed in that lesson's README in the same PR, and journaled.
- Every trained checkpoint and dataset goes to the HF Hub; the producing lesson's `RESULTS.md` links it.
- No pre-scaffolded code stubs. A lesson's Deliverables table is its file manifest; files are created when an exercise reaches them.
