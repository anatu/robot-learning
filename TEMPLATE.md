# Lesson Template

Every lesson README follows this structure. The design is backward: each lesson was built by fixing the **Done when** bar first, then the deliverables that evidence it, then the instruction that gets you there. Sections marked *(optional)* may be dropped when they'd be empty, never to save writing effort.

## Structure

```markdown
# Lesson NN — Title

One-sentence statement of what you build and why it matters.

| | |
|---|---|
| **Phase** | e.g. 4 — Generative imitation policies |
| **Time** | honest estimate, split into desk time vs compute wall-clock |
| **Cost** | cloud GPU spend, $0 if Mac-local |
| **Prerequisites** | lesson numbers + the specific artifact reused |
| **Feeds into** | later lessons that consume this lesson's outputs |

## Learning objectives
3–6 measurable "you can ..." statements. Verbs like derive, implement,
quantify, diagnose, defend — never "understand" or "be familiar with."
These are the exam; everything below exists to make them true.

## Background
The conceptual briefing: the mental model, the key equations, the design
choices and why they were made. Written so the reading list becomes
targeted rather than prerequisite. Ends with a reading table:

| Source | Read for |
|---|---|
| Paper §X | the specific thing to extract, stated as a question |

## Part N — <name> (per part: context → steps → checkpoint)
Numbered parts in dependency order. Each part:
- One sentence of context: what this part produces and why.
- Concrete steps: exact commands, dataset/model IDs, file names, function
  signatures, hyperparameters with their values. "Train the model" is not
  a step; the command that trains it is.
- **✅ Checkpoint** — an observable result (a number, a plot shape, a
  passing test) you must see before moving on. If a checkpoint fails,
  the Pitfalls table is the first stop.

## Deliverables
A file manifest with acceptance criteria per artifact:

| Artifact | Acceptance criteria |
|---|---|
| `eval.py` | seeded, produces the table in RESULTS.md from one command |

Every lesson ships `RESULTS.md`: the numbers, the plots, and 3–10
sentences of interpretation. The writeup is part of the work, not
overhead — it's where the "defend/explain" objectives get tested.

## Done when
A short checklist of verifiable criteria. Binary, no judgment calls.

## Self-check
3–6 questions answerable without notes if the lesson landed. If one
stumps you, the Background section names where to re-read.

## Pitfalls
| Symptom | Cause | Fix |
Known failure modes with diagnostics — the debugging time the lesson
author already spent, so you don't spend it again.

## Stretch *(optional)*
Extensions that deepen but don't gate progression.

## References
Full citations with links, including exact package/doc versions where
the API surface matters.
```

## Execution contract

How to read any code that appears in a lesson README, and who writes what:

1. **`bash` block** → paste into the terminal verbatim. Never becomes a file unless the step says so.
2. **`python` block with no file or module named** → run-once (REPL or scratch); never committed.
3. **"Module: `path.py`" header, a function signature, or a numbered list of functions** → a file *you* create and implement in the lesson directory. Stub files with the exact signatures are pre-scaffolded per lesson — fill them in; the stub docstring points at the README part that specifies it.
4. **Division of labor.** Everything in a lesson's Deliverables table is student-authored — that is the lesson. Claude's lane: environment debugging, scaffolding stubs and test harnesses, running checkpoints and parity verification, code review after a working draft, journal/bookkeeping. Lesson 00 (pure setup) was the one exception.
5. **Test stubs** ship `pytest.skip(..., allow_module_level=True)` so a fresh clone is green; remove the skip line as you implement.

## Writing rules

1. **Specificity over brevity, but no padding.** Every sentence either instructs, explains a mechanism, or states a criterion. No motivational filler.
2. **Commands are verbatim.** If a step can be a command, it is one, with flags. Where an API is fast-moving (LeRobot minor releases, frontier model endpoints), the step says what to run *and* where to verify the current syntax.
3. **Expected results are quantified.** "It should work" is banned. State the number, the range, or the plot shape you should see, and what deviation means.
4. **Checkpoints are cheap.** Each one is observable in under a minute from artifacts the steps already produced.
5. **Post-cutoff honesty.** For tools newer than the primary sources (MolmoAct 2, GR00T N1.7, LeRobot world-model policies), instructions give the procedure and the authoritative doc to check, and never invent API details.
6. **Interfaces are contracts.** When a later lesson reuses an artifact (`q̇ = f(q, target)` from Lesson 04, the eval harness from Lesson 14), the producing lesson states the interface explicitly.
