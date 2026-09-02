# Lesson 19 — Comparative VLA Lab

One model teaches you a recipe; two models on one yardstick teach you what matters. Fine-tune two architecturally distinct open VLAs on one dataset under one matched compute budget, evaluate under one pre-registered protocol, and write the result up with an honest threats-to-validity section.

| | |
|---|---|
| **Phase** | 5 — Generalist policies |
| **Time** | ~6–8 h desk time + ~8 GPU-hours total (2 models × 4 h budget, parallelizable across instances); +4 GPU-h if you add a third |
| **Cost** | ~$8–15 cloud GPU (more if a GR00T-class model joins) |
| **Prerequisites** | 18 (one fine-tune done properly — this is that, ×2, under discipline; SmolVLA-ft is entry #1), 14 (`eval.py`), 17 (`NOTE.md` is your model-selection input) |
| **Feeds into** | 22 (capstone options reuse the protocol), H4 (the winner is a candidate for your arm) |

## Learning objectives

After this lesson you can:

1. **Design** a matched-budget comparison protocol and defend its fairness choices in writing, before seeing any result.
2. **Adapt** two VLAs with two different adaptation mechanisms (expert fine-tune or LoRA vs soft prompts) from their own docs.
3. **Predict** a leaderboard and a per-task heatmap, then **report** the real ones with CIs, cost columns, and a failure taxonomy.
4. **Attribute** a performance gap to an architectural choice via one cheap follow-up test rather than speculation.
5. **Write** a threats-to-validity section that says which conclusions survive each threat.

## Principles

**The roster.** By mid-2026 LeRobot natively hosts a diverse VLA set, differing in backbone scale, action interface, and — most interestingly — *adaptation mechanism*:

| Model | Scale | Distinguishing bet | Adaptation path |
|---|---|---|---|
| **SmolVLA** | ~450M | efficiency: layer skip, 64 visual tokens, small expert | expert FT or LoRA (Lesson 18 — done) |
| **X-VLA** (arXiv 2510.10274) | 0.9B | soft prompts: frozen unified backbone, per-embodiment learnable prompts (~9M params, ~1%) | train prompts only — a categorically different mechanism |
| **EO-1** | ~3B | unified interleaved vision-text-action model (Qwen2.5-VL lineage) with a generative action head | LoRA or expert FT — verify on the model card |
| **Evo-1** (arXiv 2511.04555) | ~0.8B | lightweight, semantic-alignment-preserving, no robot-data pretraining, consumer-GPU friendly | full FT is actually affordable |
| **GR00T N1.7** | large | dual-system reasoning VLM + DiT expert | only if you rent ≥ 48 GB VRAM |

This lesson's pair: **SmolVLA + X-VLA** — two scales and two adaptation mechanisms (expert weights vs ~1% soft prompts). A third from {EO-1, Evo-1} is optional if budget allows. All roster claims are checkable: X-VLA has LeRobot docs and `lerobot/xvla-base` on the Hub; EO-1/Evo-1 arrived in the v0.6 wave. Exercise 1 makes verification a deliverable.

**Matched budget: the methodological core.** Fair comparison fixes a budget; there are two defensible choices. Equal *steps* isolates architecture at fixed optimization length, but a 3B model gets ~4× the FLOPs of a 0.5B one. Equal *GPU-hours* equalizes dollars — the constraint every practitioner faces. This course chooses **equal GPU-hours on identical hardware**, because the question a comparison answers for you is "what should I fine-tune with my $20?", not "what wins at infinite compute". Record steps-completed per model so readers can re-slice, and mitigate the convergence confound with a half-budget sensitivity check.

**Pre-registration.** Decisions made after seeing results are analysis, not protocol. `PROTOCOL.md` is committed before any GPU is rented; the git log is the receipt. Amendments are allowed, dated, and justified in-file.

**Aggregate success hides everything.** Two models can tie on average and disagree about *which* tasks are hard; that disagreement is the architectural signal. A per-task heatmap and a failure taxonomy are not decoration — they are where attribution starts.

**Carry forward**

- A comparison is only as good as the budget rule and the pre-registration; pick both before training.
- Equal GPU-hours penalizes large models' step counts; that is a feature when the question is "what do I fine-tune with my money".
- Report per-task, not just aggregate; report CIs everywhere; report cost.
- A surprise without a follow-up test is an anecdote.
- Threats to validity are stated per conclusion: which claims survive which threat.

| Source | Read for |
|---|---|
| Your Lesson 17 `NOTE.md` | the axes your pair should span — justify the choice in one paragraph |
| Each model's LeRobot docs page + model card | the documented adaptation recipe; VRAM needs; expected baseline numbers |
| X-VLA paper §4 | what Phase-II soft-prompt adaptation trains and freezes — you're about to run it |
| Agarwal et al. 2021 (*Statistical Precipice*) | why CIs and per-task reporting, and what aggregate scores hide |

## Exercise 1 — Verify the roster [Read]

Tests the post-cutoff-honesty principle: nothing enters the protocol unverified.

For SmolVLA, X-VLA, and (optionally) your third candidate: confirm the model card + LeRobot docs page exist; record the documented fine-tune entrypoint, the adaptation mechanism it ships with, and the VRAM requirement. Drop anything unverifiable.

**✅ Checkpoint:** a 2–3 row verification table in `RESULTS.md`, every cell sourced.

## Exercise 2 — Pre-register the protocol [Write]

Tests objective 1. Commit `PROTOCOL.md` before any training run:

1. Pair (+ optional third) with a one-paragraph justification from `NOTE.md`'s axes.
2. Dataset: Lesson 18's benchmark-paired dataset (your SmolVLA numbers transfer as entry #1; rerun under this lesson's budget if you want strict comparability).
3. Budget: 4 A100-hours per model, one instance type for all.
4. Adaptation per model: the documented default — you are comparing *models-as-shipped*, not your tuning skill.
5. Eval: suite, seed list, ≥ 50 episodes per model, success definition (Lesson 14's `evaluate()`).
6. Metrics: success ± Wilson CI, ms/chunk (batch 1, fixed hardware, median + p95), peak inference VRAM, trainable/total params, steps completed in budget, $ spent, mean squared jerk.
7. **Predictions** (this is the [Predict → Run] setup for Exercise 4): the ranking you expect and why; which tasks you expect to separate the models; the trainable-parameter fraction you expect for X-VLA.

**✅ Checkpoint:** `PROTOCOL.md` is committed and the git log proves it predates every training run.

## Exercise 3 — Two adaptations under one budget [Build]

Tests objective 2: two mechanisms, one dataset, one clock.

1. Launch each model's documented fine-tune with a hard wall-clock stop at the budget (`timeout 4h lerobot-train ...` or the trainer's step-time-calibrated step cap). Same GPU type for both — mixing a 4090 and an A100 voids the comparison.
2. For X-VLA, follow its LeRobot docs for Phase-II adaptation: backbone frozen, soft prompts trained. Log what fraction of parameters that is.
3. If a model's fine-tune script wants a different dataset format, spec an adapter shim for an AI tool (per-model, committed); never edit the shared dataset.
4. Capture per model: W&B curves, steps completed, peak training VRAM, $ from the provider dashboard. Push checkpoints to the Hub.

**✅ Checkpoint:** two Hub checkpoints; cost rows filled; every run stopped at budget, not at convergence (record which arms were still improving — that's data).

## Exercise 4 — One protocol, two evaluations [Predict → Run]

Tests objective 3 against the predictions you committed in Exercise 2.

1. Run the pre-registered eval identically per model: same seeds, episode counts, suite. Benchmark Docker image on the cloud box; `MUJOCO_GL=egl`.
2. Measure inference latency and VRAM on one fixed machine for both (ms/chunk ≠ control-rate ceiling once Lesson 16's async stack is in play; report both ms/chunk and implied max chunk rate).
3. Save ≥ 10 rollout videos per model, stratified across tasks: successes and failures.
4. Populate the leaderboard with every pre-registered column, then the model × task heatmap.
5. Reconcile: ranking predicted vs observed; separating tasks predicted vs observed.

**✅ Checkpoint:** no leaderboard cell says "TODO", including cost and jerk; the heatmap exists; the reconciliation is written.

## Exercise 5 — Sensitivity, taxonomy, and the tested surprise [Diagnose]

Tests objective 4: attribution by evidence.

1. **Half-budget sensitivity:** evaluate each model's mid-training checkpoint (2 h mark). If the ranking is stable from half to full budget, the conclusion is budget-robust; if not, say so prominently.
2. **Failure taxonomy:** label the failure videos (grasp miss / wrong object / never approached / timeout dithering / catastrophic). One stacked bar per model.
3. **The surprise:** find at least one result that contradicts your Exercise 2 predictions. Write the mechanism you think explains it, then run one cheap follow-up (≤ 30 lines or ≤ 30 min GPU) that could falsify that mechanism. Report what it showed.

**✅ Checkpoint:** sensitivity verdict, taxonomy bars, and a surprise with a tested hypothesis — not just a paragraph.

## Exercise 6 — Leaderboard and threats [Write]

Tests objective 5. `leaderboard.md`, ~2 pages:

1. Setup (¼ page): pair, budget rationale (from `PROTOCOL.md`), eval protocol.
2. Results (½ page): leaderboard + heatmap, described neutrally.
3. Analysis (½ page): the three Exercise 5 analyses; which architectural choice explains each observed gap, with the follow-up evidence.
4. Threats to validity (½ page, mandatory): single dataset, single budget point, adaptation-recipe maturity varies by model, sim-benchmark overfitting. For each threat, state which conclusions survive it.

**✅ Checkpoint:** the threats section names ≥ 3 real threats and, for each, the conclusions that survive.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `PROTOCOL.md` | committed before training, with predictions; git history proves it |
| 2 (+1 optional) Hub checkpoints | load and rerun via `eval/run_leaderboard_eval.py` |
| `leaderboard.md` + `plots/` | all pre-registered columns; heatmap; taxonomy bars; CIs everywhere; threats section |
| `RESULTS.md` | roster verification table; prediction reconciliation; sensitivity verdict; the tested surprise |

## Done when

- [ ] Two models, one dataset, one budget, one eval — and the git log proves the protocol (with predictions) came first.
- [ ] Every leaderboard number carries its CI or measurement conditions.
- [ ] The per-task heatmap and failure taxonomy exist.
- [ ] The surprise has a follow-up test with a result.
- [ ] The threats section names ≥ 3 threats and their blast radius per conclusion.

## Self-check

1. Equal GPU-hours penalizes large models' step counts. Name the comparison question for which equal *steps* would be the right protocol.
2. X-VLA adapts ~1% of parameters; SmolVLA-LoRA a few %; a full-FT model 100%. Why can all be "fair" under this protocol?
3. Two models tie on aggregate success but their task heatmaps are complementary. What deployment strategy does that suggest, and which later lesson builds it?
4. Your latency column disagrees with the papers' claims. List three legitimate reasons before "they lied".
5. Which threat to validity would another dataset choice fix, and which would survive it?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| One model's fine-tune script wants a different dataset format | model-specific transforms/keys | adapter shim per model, committed; never edit the shared dataset |
| Rankings flip between two eval suites | benchmark-specific overfitting or camera-convention mismatch | report both, conclude on neither alone; check camera keys per model |
| A large model OOMs at its documented batch size | docs assume 80 GB A100 | grad-accum to match *effective* batch; log the change in `PROTOCOL.md` as a dated amendment |
| Budget spent, loss still falling steeply | expected under equal-hours | that IS the finding — report steps completed + half-budget sensitivity |
| Latency numbers not comparable | measured on different machines/precisions | one machine, one precision, batch 1, median of 100 warm calls — assert in the script |

## Going deeper

- **A third row.** Add EO-1 or Evo-1 under the same budget; the soft-prompt vs LoRA vs full-FT triangle is the full mechanism comparison.
- **The 2023 baseline.** Add your Lesson 14 ACT (same dataset, same budget) as a row. "Does a non-VLA at equal compute embarrass anyone?" is the most clarifying row in the table.
- **Mini-paper.** Expand `leaderboard.md` into a 4-page CoRL-format PDF — the writing muscle the capstone report uses.

## References

- Wang et al., *X-VLA*, ICLR 2026. arXiv:2510.10274. LeRobot docs: `xvla`; Hub: `lerobot/xvla-base`.
- *Evo-1: Lightweight VLA with Preserved Semantic Alignment*, 2025. arXiv:2511.04555.
- EO-1 model card + docs via the LeRobot v0.6.0 release blog roster.
- Shukor et al., *SmolVLA*, 2025. arXiv:2506.01844.
- Agarwal et al., *Deep RL at the Edge of the Statistical Precipice*, NeurIPS 2021 — the case for CIs and per-task reporting.
