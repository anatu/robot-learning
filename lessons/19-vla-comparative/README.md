# Lesson 19 — Comparative VLA Lab

Fine-tuning one model, as you did in Lesson 18, teaches you a recipe. Fine-tuning two architecturally different models on the same data, under the same compute budget, and evaluating them under the same protocol teaches you which design choices actually matter, because it removes every other explanation for a difference in results. In this lesson you adapt two open vision-language-action models with two different adaptation mechanisms, evaluate them under a protocol you commit to before training, and write the result up with a section that states honestly which of your conclusions would survive which objection. The methodology is what the capstone reuses.

| | |
|---|---|
| **Phase** | 5 — Generalist policies |
| **Time** | ~6–8 h desk time + ~8 GPU-hours (2 models × 4 h budget, which can run in parallel on separate instances); add ~4 GPU-hours for an optional third model |
| **Cost** | ~$8–15 of cloud GPU, more if a GR00T-class model is included |
| **Prerequisites** | 18 (one fine-tune done properly; this lesson repeats it twice under a protocol, and the SmolVLA fine-tune is entry #1), 14 (`eval.py`), 17 (`NOTE.md`, which is the input to model selection) |
| **Feeds into** | 22 (the capstone options reuse this protocol), H4 (the better model is a candidate for your arm) |

## Learning objectives

After this lesson you can:

1. **Design** a matched-budget comparison protocol and defend its fairness choices in writing before you have seen any result.
2. **Adapt** two VLAs using two different adaptation mechanisms (expert fine-tuning or LoRA on one, soft prompts on the other) from their own documentation.
3. **Predict** a leaderboard and a per-task heatmap, and then **report** the real ones with confidence intervals, cost columns, and a failure taxonomy.
4. **Attribute** a performance gap to an architectural choice by running one cheap follow-up test rather than by speculating.
5. **Write** a threats-to-validity section that says, for each threat, which conclusions survive it.

## Principles

### The roster of models

By mid-2026 LeRobot hosts several open vision-language-action models natively, and they differ in three ways that matter for a comparison: the scale of the backbone, the action interface, and the mechanism by which the model is adapted to a new task or embodiment. The third difference is the most interesting one, because it determines how much of the model you train and therefore how much data and compute adaptation needs. The table summarises the candidates.

| Model | Scale | Distinguishing bet | Adaptation path |
|---|---|---|---|
| **SmolVLA** | ~450M | efficiency: layer skip, 64 visual tokens, small expert | expert FT or LoRA (Lesson 18 — done) |
| **X-VLA** (arXiv 2510.10274) | 0.9B | soft prompts: frozen unified backbone, per-embodiment learnable prompts (~9M params, ~1%) | train prompts only — a categorically different mechanism |
| **EO-1** | ~3B | unified interleaved vision-text-action model (Qwen2.5-VL lineage) with a generative action head | LoRA or expert FT — verify on the model card |
| **Evo-1** (arXiv 2511.04555) | ~0.8B | lightweight, semantic-alignment-preserving, no robot-data pretraining, consumer-GPU friendly | full FT is actually affordable |
| **GR00T N1.7** | large | dual-system reasoning VLM + DiT expert | only if you rent ≥ 48 GB VRAM |

This lesson compares SmolVLA and X-VLA. The pair spans two scales and two adaptation mechanisms: SmolVLA trains the weights of its action expert (or low-rank adapters on them), whereas X-VLA keeps its entire backbone frozen and trains only a set of soft prompts amounting to about one percent of its parameters. A third model from EO-1 or Evo-1 can be added if budget allows. Every claim in the roster table can be checked: X-VLA has a LeRobot documentation page and `lerobot/xvla-base` on the Hub, and EO-1 and Evo-1 arrived with the v0.6 release. Exercise 1 makes that verification a deliverable rather than an assumption.

### Matching the budget

A comparison between models of different sizes has to decide what to hold equal, and there are two defensible answers. Holding the number of training steps equal isolates the architecture at a fixed optimisation length, but a 3B-parameter model then consumes about four times the floating-point operations of a 0.5B model, so it receives four times the compute. Holding GPU-hours equal on identical hardware equalises the money spent, which is the constraint that a practitioner actually faces. This course chooses equal GPU-hours, because the question a comparison answers for you is which model to fine-tune with a fixed budget, not which model would win with unlimited compute. Two safeguards go with that choice: record the number of steps each model completed, so that a reader can reinterpret the result at equal steps, and check whether the ranking is already stable at half the budget, which addresses the objection that one model simply had not converged.

### Pre-registration

A decision made after seeing results is analysis, not protocol, and analysis performed after the fact is where comparisons quietly become unfair. The protocol for this lesson, meaning the models, the budget, the adaptation mechanism for each, the evaluation suite, the seeds, the episode counts, and the metrics, is written into `PROTOCOL.md` and committed before any GPU is rented, so that the git history proves it predates every training run. Amendments are permitted, but each must be dated and justified in the file.

### Why aggregate success is not enough

Two models can have the same average success rate and disagree about which tasks are hard. That disagreement is where the architectural signal lives, because it tells you what each model's inductive bias helps with. A per-task heatmap and a taxonomy of failure modes are therefore not decoration: they are the starting point for attributing a difference to a design choice, which is objective 4.

**Carry forward**

- A comparison is only as good as its budget rule and its pre-registration, because those two decisions are the ones most easily bent after the results are in; choose both before training.
- Equal GPU-hours gives large models fewer steps, and that is appropriate when the question is which model to fine-tune with a fixed amount of money rather than which model is best in principle.
- Report per-task results and confidence intervals alongside the aggregate, and report cost, because aggregates hide the task-level disagreements that explain differences.
- A surprising result that contradicts your prediction is only evidence once you have run a test that could have falsified your explanation for it.
- A threats-to-validity section is stated per conclusion, so that a reader knows which claims survive which objection.

| Source | Read for |
|---|---|
| Your Lesson 17 `NOTE.md` | the design axes your pair should span; justify the choice in one paragraph |
| Each model's LeRobot documentation page and model card | the documented adaptation recipe, VRAM requirements, and expected baseline numbers |
| X-VLA paper §4 | what Phase-II soft-prompt adaptation trains and what it freezes, since that is what you are about to run |
| Agarwal et al. 2021 (*Statistical Precipice*) | why confidence intervals and per-task reporting matter, and what aggregate scores hide |

## Exercise 1 — Verify the roster [Read]

Nothing enters the protocol until you have confirmed that it exists as documented. Several of the roster entries postdate this course's primary sources, and a comparison built on a model whose documented fine-tune path turns out not to exist would fail after the budget is spent. This exercise makes the verification explicit.

For SmolVLA, X-VLA, and, if you plan to include one, your third candidate: confirm that the model card and the LeRobot documentation page exist, and record the documented fine-tune entry point, the adaptation mechanism the model ships with, and its VRAM requirement. Drop any candidate you cannot verify.

**✅ Checkpoint:** a verification table of two or three rows in `RESULTS.md`, with every cell attributed to its source.

## Exercise 2 — Pre-register the protocol [Write]

This exercise fixes every decision that could otherwise be made after seeing results. It also records your predictions, so that Exercise 4 can be a genuine comparison between what you expected and what happened. Commit `PROTOCOL.md` before launching any training run.

1. The pair, plus the optional third model, with a one-paragraph justification drawn from the design axes in `NOTE.md`.
2. The dataset: the benchmark-paired dataset from Lesson 18. Your SmolVLA numbers from that lesson transfer as entry #1; rerun them under this lesson's budget if you want strict comparability.
3. The budget: 4 A100-hours per model, on one instance type for all models.
4. The adaptation mechanism per model: the documented default. You are comparing the models as their authors ship them, not your skill at tuning each one.
5. The evaluation: the suite, the seed list, at least 50 episodes per model, and the success definition, using Lesson 14's `evaluate()`.
6. The metrics: success with a Wilson confidence interval, milliseconds per chunk (batch size 1, fixed hardware, median and p95), peak inference VRAM, trainable and total parameters, steps completed within the budget, dollars spent, and mean squared jerk.
7. Your predictions, which are what Exercise 4 will be reconciled against: the ranking you expect and why, the tasks you expect to separate the models, and the fraction of parameters you expect X-VLA's soft prompts to amount to.

**✅ Checkpoint:** `PROTOCOL.md` is committed, and the git log shows it predates every training run.

## Exercise 3 — Adapt two models under one budget [Build]

Here you run the two fine-tunes. The discipline is that both models see the same dataset, the same wall-clock budget, and the same hardware, so that the only difference between the runs is the model and its adaptation mechanism.

1. Launch each model's documented fine-tune with a hard wall-clock stop at the budget, either by wrapping the command (`timeout 4h lerobot-train ...`) or by setting a step cap calibrated from the trainer's measured step time. Use the same GPU type for both models; a comparison between a run on a 4090 and a run on an A100 measures the hardware rather than the models.
2. For X-VLA, follow its LeRobot documentation for Phase-II adaptation, in which the backbone is frozen and only the soft prompts are trained. Log the fraction of the model's parameters that this represents.
3. If a model's fine-tune script expects a different dataset format, write a specification for a per-model adapter shim and have an AI tool draft it. Commit the shim with the model's name; never modify the shared dataset.
4. For each model, capture the W&B curves, the number of steps completed, the peak training VRAM, and the cost from the provider's dashboard. Push both checkpoints to the Hub.

**✅ Checkpoint:** two checkpoints on the Hub and both cost rows filled in. Every run should have stopped at the budget rather than at convergence; record which runs were still improving when they stopped, because that observation feeds the half-budget analysis in Exercise 5.

## Exercise 4 — Evaluate both models under the protocol [Predict → Run]

With two adapted models, you now run the evaluation you committed to in Exercise 2 and compare the outcome with the predictions you wrote there. The predictions were made before training, so the comparison tells you how well your understanding of the two architectures anticipated their behaviour.

1. Run the pre-registered evaluation identically for each model: the same seeds, the same episode counts, and the same suite, using the benchmark's Docker image on the cloud machine with `MUJOCO_GL=egl` set.
2. Measure inference latency and VRAM for both models on one fixed machine. Note that milliseconds per chunk is not the same as the achievable control rate once Lesson 16's asynchronous stack is in use, so report both the per-chunk latency and the implied maximum chunk rate.
3. Save at least ten rollout videos per model, stratified across tasks and including both successes and failures.
4. Populate the leaderboard with every pre-registered column, then build the model-by-task heatmap.
5. Reconcile the outcome with `PROTOCOL.md`: the predicted ranking against the observed ranking, and the predicted separating tasks against the observed ones.

**✅ Checkpoint:** no leaderboard cell reads "TODO", including the cost and jerk columns; the heatmap exists; and the reconciliation is written.

## Exercise 5 — Sensitivity, failure taxonomy, and the tested surprise [Diagnose]

A leaderboard states what happened; this exercise establishes why, to the extent that the evidence allows. It has three parts: a check that the ranking does not depend on the budget you happened to choose, a classification of the failures, and a test of your explanation for whichever result surprised you most.

1. Half-budget sensitivity. Evaluate each model's checkpoint from the two-hour mark. If the ranking is the same at half budget as at full budget, the conclusion is robust to the budget; if it is not, say so prominently, because it means the comparison is partly a comparison of convergence speed.
2. Failure taxonomy. Label each failure video with one of: grasp miss, wrong object, never approached, timeout dithering, catastrophic. Plot one stacked bar per model.
3. The tested surprise. Find at least one result that contradicts the predictions in `PROTOCOL.md`. Write down the mechanism you believe explains it, then design and run one cheap follow-up (at most 30 lines of code or 30 minutes of GPU time) whose outcome could show that mechanism to be wrong. Report what it showed.

**✅ Checkpoint:** the sensitivity verdict, the taxonomy bars, and a surprise accompanied by a follow-up test and its result.

## Exercise 6 — Write the leaderboard and its threats to validity [Write]

The final deliverable is `leaderboard.md`, about two pages, in the structure of an experiments section. Its most important part is the last one, which states for each threat to validity which of your conclusions survive it; a comparison that does not say this leaves the reader to guess how much to trust it.

1. Setup (a quarter page): the pair, the budget rationale from `PROTOCOL.md`, and the evaluation protocol.
2. Results (half a page): the leaderboard and the heatmap, described without interpretation.
3. Analysis (half a page): the three analyses from Exercise 5, and for each observed gap, the architectural choice you believe explains it together with the follow-up evidence.
4. Threats to validity (half a page, mandatory): the single dataset, the single budget point, the differing maturity of each model's adaptation recipe, and overfitting to the simulation benchmark. For each threat, state which of your conclusions survive it.

**✅ Checkpoint:** the threats section names at least three real threats and, for each, the conclusions that survive it.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `PROTOCOL.md` | committed before training, with predictions; git history proves it |
| 2 (+1 optional) Hub checkpoints | load and rerun via `eval/run_leaderboard_eval.py` |
| `leaderboard.md` + `plots/` | all pre-registered columns; heatmap; taxonomy bars; confidence intervals throughout; threats section |
| `RESULTS.md` | roster verification table; prediction reconciliation; sensitivity verdict; the tested surprise |

## Done when

- [ ] Two models, one dataset, one budget, one evaluation, and a git log proving the protocol with its predictions came first.
- [ ] Every leaderboard number carries its confidence interval or its measurement conditions.
- [ ] The per-task heatmap and the failure taxonomy exist.
- [ ] The surprise has a follow-up test with a result.
- [ ] The threats section names at least three threats and states, per conclusion, which survive.

## Self-check

1. Equal GPU-hours gives large models fewer steps. Name a comparison question for which equal steps would be the right protocol instead.
2. X-VLA adapts about 1% of its parameters, SmolVLA with LoRA a few percent, and a fully fine-tuned model 100%. Why can all three be fair under this protocol?
3. Two models tie on aggregate success, but their task heatmaps are complementary. What deployment strategy does that suggest, and which later lesson builds it?
4. Your latency column disagrees with the numbers reported in the papers. List three legitimate explanations to rule out before concluding that the papers misreported.
5. Which threat to validity would a second dataset address, and which would remain?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| One model's fine-tune script wants a different dataset format | model-specific transforms or keys | a per-model adapter shim, committed; never edit the shared dataset |
| Rankings flip between two evaluation suites | benchmark-specific overfitting, or a camera-convention mismatch | report both and conclude on neither alone; check camera keys per model |
| A large model runs out of memory at its documented batch size | the documentation assumes an 80 GB A100 | use gradient accumulation to match the *effective* batch; log the change in `PROTOCOL.md` as a dated amendment |
| Budget spent, loss still falling steeply | expected under equal GPU-hours | this is a finding, not a fault; report steps completed and the half-budget sensitivity |
| Latency numbers are not comparable | measured on different machines or precisions | one machine, one precision, batch size 1, median of 100 warm calls; assert these in the script |

## Going deeper

- **A third row.** Add EO-1 or Evo-1 under the same budget. With soft prompts, LoRA, and full fine-tuning all represented, the comparison covers the full range of adaptation mechanisms.
- **The 2023 baseline.** Add your Lesson 14 ACT model, trained on the same dataset under the same budget, as a row. Whether a non-VLA at equal compute matches the VLAs is the most clarifying question the table can answer.
- **A mini-paper.** Expand `leaderboard.md` into a four-page PDF in CoRL format. This is the writing the capstone report requires.

## References

- Wang et al., *X-VLA*, ICLR 2026. arXiv:2510.10274. LeRobot docs: `xvla`; Hub: `lerobot/xvla-base`.
- *Evo-1: Lightweight VLA with Preserved Semantic Alignment*, 2025. arXiv:2511.04555.
- EO-1 model card and documentation, via the LeRobot v0.6.0 release blog roster.
- Shukor et al., *SmolVLA*, 2025. arXiv:2506.01844.
- Agarwal et al., *Deep RL at the Edge of the Statistical Precipice*, NeurIPS 2021, on confidence intervals and per-task reporting.
