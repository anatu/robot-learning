# Lesson Template

Every lesson README follows this structure. Design is backward from the **Done when** bar: the principle the lesson installs comes first, then the exercises that prove you have it, then whatever code the exercises need. The code is the least important part. Sections marked *(optional)* may be dropped when they would be empty, never to save writing effort.

## What a lesson is for

The learning benefit of this course is in principles and practical exercises: predicting what an experiment will show, running it, reconciling, and deciding. Code is instrumental. With current AI coding tools, hand-implementing a dataset parser or an RRT teaches less per hour than specifying it precisely, verifying it, and then *using* it to test a prediction. So:

- Every lesson names its principles up front and ties each exercise to one.
- Code is written with AI assistance from a spec you author. What you cannot delegate: the spec, the check, the prediction, the interpretation, and the decision.
- From-scratch reimplementation is offered under **Going deeper**, never required.

## Structure

```markdown
# Lesson NN — Title

One sentence: the principle this lesson installs and the experiment that proves you have it.

| | |
|---|---|
| **Phase** | e.g. 4 — Generative imitation policies |
| **Time** | desk time vs compute wall-clock; desk time assumes AI-assisted coding |
| **Cost** | cloud GPU spend, $0 if Mac-local |
| **Prerequisites** | lesson numbers + the specific artifact reused |
| **Feeds into** | later lessons that consume this lesson's outputs |

## Learning objectives
3–5 measurable "you can ..." statements. Verbs: explain, predict, derive,
diagnose, decide, quantify, defend. "Implement" appears only when the
implementation is itself the understanding, and then it is a small kernel.

## Principles
The mental model, the key equations, the design choices and why. The
largest section. Ends with two things:

**Carry forward** — 3–5 bullets: what you must still know a year from now.

| Source | Read for |
|---|---|
| Paper §X | the specific thing to extract, stated as a question |

## Exercise N — <name> [Type]
Numbered, in dependency order, each tagged with one type from the table
below. Each exercise:
- One sentence: which principle it tests and what it produces.
- Concrete steps. For [Build]: the spec you give the AI tool (interface,
  behavior, the check). For [Predict → Run]: what to write down before
  running. Commands verbatim; dataset/model IDs; hyperparameters with values.
- **✅ Checkpoint** — an observable result (a number, a plot shape, a passing
  check) before moving on. If it fails, Pitfalls is the first stop.

## Deliverables
| Artifact | Acceptance criteria |
|---|---|
| `RESULTS.md` | predictions vs outcomes for every [Predict → Run]; figures; interpretation |
| `run.py` | reproduces every figure/table from one command |
Few artifacts. RESULTS.md is the primary one: it is where the objectives
get tested.

## Done when
Short checklist. Binary, no judgment calls.

## Self-check
3–6 questions answerable without notes if the lesson landed.

## Pitfalls
| Symptom | Cause | Fix |

## Going deeper *(optional)*
Where from-scratch reimplementation or a bigger experiment would pay off.
Never gates progression.

## References
Full citations with links; exact package/doc versions where the API
surface matters.
```

## Exercise types

| Tag | What you do | What proves it |
|---|---|---|
| **[Derive]** | on paper, a result the lesson needs | it appears in RESULTS.md and a numeric check agrees with it |
| **[Build]** | author the spec (interface, behavior, the check that proves correctness); have an AI tool draft the code; read the draft; run the check | the check passes and you can explain every line that touches the principle |
| **[Read the kernel]** | annotate a piece of code, yours or a library's, line by line with the equation or rule each line implements | the annotated file is committed |
| **[Predict → Run]** | write the expected result and the reason *before* running; run; reconcile | prediction, outcome, and reconciliation all appear in RESULTS.md |
| **[Diagnose]** | a planted bug or a known failure mode: find it from symptoms and explain the mechanism | the diagnosis names the mechanism, not just the fix |
| **[Decide]** | choose a config, design, or threshold and defend it with your numbers | the decision and its supporting row or figure in RESULTS.md |
| **[Read]** | targeted reading with a specific question | the answer, in RESULTS.md or the note |
| **[Write]** | a document: note, protocol, field guide, audit | the document passes its stated bar (usually the stranger test) |

## Execution contract

1. **`bash` block** → paste into the terminal verbatim.
2. **`python` block with no file named** → run-once (REPL or scratch); never committed.
3. **A named file in an exercise or the Deliverables table** → a file that ends up in the lesson directory. By default you author its spec and an AI tool drafts it; you read the draft and run the check. A lesson says explicitly when a kernel is worth typing yourself.
4. **Non-delegable** (this is the coursework): predictions written before runs; RESULTS.md interpretation; [Derive] work; [Read the kernel] annotations; diagnoses; decisions and their defense; pre-registration documents (PROTOCOL.md, TASK.md); self-check answers.
5. **Claude in-session:** drafts code from your spec, debugs the environment, runs checkpoints, reviews drafts, keeps the journal. Claude does not write your predictions, interpretations, or decisions.
6. **No pre-scaffolded stubs.** The Deliverables table is the file manifest; files are created when an exercise reaches them.
7. **Interfaces are contracts.** When a later lesson reuses an artifact (`fk(q)` from Lesson 03, `evaluate(...)` from Lesson 14), the producing lesson states the interface explicitly and it survives any simplification.

## Writing rules

1. **Principle before procedure.** Every exercise names the principle it tests in its first sentence. A step that tests nothing is cut.
2. **Commands are verbatim.** If a step can be a command, it is one, with flags. Where an API moves fast (LeRobot minor releases, hosted model endpoints), the step says what to run *and* where to verify the current syntax.
3. **Expected results are quantified.** "It should work" is banned. State the number, the range, or the plot shape, and what a deviation means.
4. **Predictions are written first.** Every [Predict → Run] tells you exactly what to write down before the run.
5. **Checkpoints are cheap.** Observable in under a minute from artifacts the steps already produced.
6. **Experiments are minimal.** The smallest grid that shows the effect, with the caveat stated. Three seeds when the claim is a ranking under noise; one seed when the claim is a mechanism.
7. **Post-cutoff honesty.** For tools newer than the primary sources, give the procedure and the authoritative doc to check; never invent API details.
8. **Specificity over brevity, no padding.** Every sentence instructs, explains a mechanism, or states a criterion.
