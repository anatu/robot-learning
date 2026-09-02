# CLAUDE.md

Self-study robot-learning course. Map: `README.md`. Lesson format: `TEMPLATE.md`. One lesson = one directory = one PR, with a `RESULTS.md` per lesson.

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

## Division of labor

Lesson deliverables (everything in a lesson's Deliverables table) are authored by the user — the implementation is the lesson. Claude: environment debugging, stub/harness scaffolding, checkpoint verification, post-draft code review, journal and bookkeeping. Full contract in `TEMPLATE.md` § Execution contract.

## Conventions

- Costs in USD, actuals not estimates.
- Version drift discovered against a lesson README (LeRobot flags, model IDs) gets fixed in that lesson's README in the same PR, and journaled.
- Every trained checkpoint and dataset goes to the HF Hub; the producing lesson's `RESULTS.md` links it.
