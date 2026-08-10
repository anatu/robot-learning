# Lesson 19 — Comparative VLA Lab

Fine-tune three architecturally distinct open VLAs on one dataset under one matched compute budget, evaluate them under one protocol, and write the result up as a mini-paper. One model teaches you a recipe; three models on one yardstick teach you what actually matters.

| | |
|---|---|
| **Phase** | 5 — Generalist policies |
| **Time** | ~2 sessions desk time + ~12 GPU-hours total (3 models × 4 h budget, parallelizable across instances) |
| **Cost** | ~$10–25 cloud GPU depending on instance choice and whether GR00T-class models join |
| **Prerequisites** | 18 (one fine-tune done properly — this lesson is that, ×3, under discipline), 14 (the eval harness), 17 (`NOTE.md` is your model-selection input) |
| **Feeds into** | 22 (capstone options reuse the protocol), H4 (the winner is a candidate for your arm) |

## Learning objectives

After this lesson you can:

1. **Design** a matched-budget comparison protocol and defend its fairness choices in writing, before seeing any results.
2. **Adapt** three different VLAs — including at least two different adaptation mechanisms (LoRA, soft prompts, full/expert fine-tune) — from their own docs.
3. **Report** a leaderboard with confidence intervals, cost columns, and a failure taxonomy, not just a success column.
4. **Attribute** performance differences to architectural choices via targeted follow-up analysis rather than speculation.
5. **Write** a 2–4 page mini-paper with an honest threats-to-validity section.

## Background

By mid-2026 LeRobot natively hosts a genuinely diverse VLA roster — differing in backbone scale, action interface, and (most interestingly) *adaptation mechanism*. Candidates, with what each represents:

| Model | Scale | Distinguishing bet | Adaptation path |
|---|---|---|---|
| **SmolVLA** | ~450M | efficiency: layer skip, 64 visual tokens, small expert | expert FT or LoRA (Lesson 18 — done) |
| **X-VLA** (arXiv 2510.10274) | 0.9B | soft prompts: frozen unified backbone, per-embodiment learnable prompts (~9M params, ~1%) | train prompts only — a categorically different mechanism |
| **EO-1** | ~3B | unified interleaved vision-text-action model (Qwen2.5-VL lineage) with a generative action head | LoRA or expert FT — verify on the model card |
| **Evo-1** (arXiv 2511.04555) | ~0.8B | lightweight, semantic-alignment-preserving, no robot-data pretraining, consumer-GPU friendly | full FT is actually affordable |
| **GR00T N1.7** | large | dual-system reasoning VLM + DiT expert | only if you rent ≥ 48 GB VRAM |

Recommended trio: **SmolVLA + X-VLA + one of {EO-1, Evo-1}** — it spans small/medium scale and three adaptation mechanisms. All roster claims above are checkable: X-VLA has LeRobot docs and `lerobot/xvla-base` on the Hub; EO-1/Evo-1 arrived in the v0.6 wave. Part 0 makes verification a deliverable rather than an assumption.

**The methodological core.** Fair comparison requires fixing a budget, and there are two defensible choices: equal *steps* (isolates architecture at fixed optimization length, but a 3B model gets ~4× the FLOPs of a 0.5B one) or equal *GPU-hours* (equal dollars — the constraint every practitioner actually faces). This course chooses **equal GPU-hours on identical hardware**, because the question a comparison answers for you is "what should I fine-tune with my $20?", not "what wins at infinite compute". Record steps-completed per model so readers can re-slice, and mitigate the convergence confound with the half-budget sensitivity check in Part 3.

| Source | Read for |
|---|---|
| Your Lesson 17 `NOTE.md` | the axes your three picks should span — justify the trio in one paragraph |
| Each model's LeRobot docs page + model card | the documented adaptation recipe; VRAM needs; expected baseline numbers |
| X-VLA paper §4 | what Phase-II soft-prompt adaptation trains and freezes — you're about to run it |
| A CoRL paper you rate highly | the shape of a good experiments section — steal its table design |

## Part 0 — Protocol pre-registration (~2 h, before any GPU is rented)

The discipline *is* the lesson: decisions made after seeing results are analysis, not protocol.

1. Verify the roster: for each candidate, confirm the model card + docs page exist and note the documented fine-tune entrypoint and VRAM requirement. Drop anything unverifiable; pick your trio.
2. Fix the dataset: reuse Lesson 18's benchmark-paired dataset (your SmolVLA numbers then transfer as leaderboard entry #1 — budget permitting, rerun under this lesson's budget for strict comparability).
3. Write `PROTOCOL.md` and commit it before training: trio + justification; budget (recommend 4 A100-hours/model, one instance type for all); adaptation mechanism per model (the documented default for each — you are comparing *models-as-shipped*, not your tuning skill); eval suite, seed list, episode count (≥ 50/model), success definition; the full metric list: success ± Wilson CI, ms/chunk (batch 1, fixed hardware, median + p95), peak inference VRAM, trainable/total params, steps completed in budget, $ spent, mean squared jerk.

**✅ Checkpoint:** `PROTOCOL.md` is committed and the git log proves it predates every training run.

## Part 1 — Three adaptations (~3 × 4 GPU-hours, parallelizable)

1. Launch each model's documented fine-tune on the fixed dataset with a hard wall-clock stop at the budget (`timeout 4h lerobot-train ...` or the trainer's step-time-calibrated step cap). Same GPU type for all three — mixing a 4090 and an A100 voids the comparison.
2. For X-VLA, follow its LeRobot docs for Phase-II adaptation: backbone frozen, soft prompts trained. Log what fraction of parameters that is — the leaderboard's most interesting column.
3. Capture per-model: W&B curves, steps completed, peak training VRAM, $ from the provider dashboard. Push all checkpoints to the Hub.

**✅ Checkpoint:** three Hub checkpoints; three cost rows filled; every run stopped at budget, not at convergence (say which arms were still improving — that's data).

## Part 2 — One protocol, three evaluations (~2 h)

1. Run the pre-registered eval identically per model: same seeds, same episode counts, same suite. Use the benchmark's Docker image on the cloud box; `MUJOCO_GL=egl`.
2. Measure inference latency and VRAM on one fixed machine for all three (the cloud box; note that ms/chunk ≠ control-rate ceiling once Lesson 16's async stack is in play — report both ms/chunk and implied max chunk rate).
3. Save ≥ 10 rollout videos per model, stratified across tasks: successes and failures.

**✅ Checkpoint:** the leaderboard table is fully populated — no cell says "TODO", including the cost and jerk columns.

## Part 3 — Analysis (~3–4 h)

1. **Leaderboard** with all pre-registered columns, CIs everywhere.
2. **Per-task breakdown:** aggregate success hides everything; a heatmap (model × task) usually reveals that models disagree about *which* tasks are hard — that disagreement is the architectural signal.
3. **Half-budget sensitivity:** evaluate each model's mid-training checkpoint (2 h mark). If the ranking is stable from half to full budget, your conclusion is budget-robust; if not, say so prominently.
4. **Failure taxonomy:** label the failure videos (grasp miss / wrong object / never approached / timeout dithering / catastrophic). One stacked bar per model.
5. **The surprise:** find at least one result that contradicts your Part 0 expectations, then run one cheap follow-up (≤ 30 lines or ≤ 30 min GPU) that tests your explanation. A surprise without a follow-up test is an anecdote.

**✅ Checkpoint:** heatmap + taxonomy exist; the surprise has a tested hypothesis, not just a paragraph.

## Part 4 — The mini-paper (~3 h)

2–4 pages, CoRL-ish structure — this is the writing muscle the capstone report needs:

1. Setup (½ page): trio, budget rationale (steal from `PROTOCOL.md`), eval protocol.
2. Results (1 page): leaderboard + heatmap, described neutrally.
3. Analysis (1 page): the three analyses above; what architectural choice explains each observed gap — with the follow-up evidence.
4. Threats to validity (½ page, mandatory): single dataset, single budget point, adaptation-recipe quality varies by model maturity, sim-benchmark overfitting. State which conclusions survive each threat.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `PROTOCOL.md` | committed before training; git history proves it |
| 3 Hub checkpoints | load and rerun via committed eval scripts |
| `leaderboard.md` + `plots/` | all pre-registered columns; heatmap; taxonomy bars; CIs everywhere |
| `paper.pdf` | 2–4 pages, all four sections, ≥ 1 tested surprise |

## Done when

- [ ] Three models, one dataset, one budget, one eval — and the git log proves the protocol came first.
- [ ] Every leaderboard number carries its CI or measurement conditions.
- [ ] The per-task heatmap and failure taxonomy exist.
- [ ] The mini-paper's threats section names ≥ 3 real threats and their blast radius.

## Self-check

1. Equal GPU-hours penalizes large models' steps count. Name the comparison question for which equal *steps* would be the right protocol.
2. X-VLA adapts ~1% of parameters; SmolVLA-LoRA a few %; Evo-1 full-FT is 100%. Why can all three be "fair" under this protocol?
3. Two models tie on aggregate success but their task heatmaps are complementary. What deployment strategy does that suggest, and which later lesson builds it?
4. Your latency column disagrees with the papers' claims. List three legitimate reasons before "they lied".
5. Which threat to validity would another dataset choice fix, and which would survive it?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| One model's fine-tune script wants a different dataset format | model-specific transforms/keys | write an adapter shim, never edit the shared dataset; commit the shim per-model |
| Rankings flip between your two eval suites | benchmark-specific overfitting or camera-convention mismatch | report both, conclude on neither alone; check camera keys per model |
| A 3B model OOMs at its documented batch size on your instance | docs assume 80 GB A100 | grad-accum to match *effective* batch; log the change in PROTOCOL.md as an amendment with a date |
| Budget spent, loss still falling steeply on the big model | expected under equal-hours | that IS the finding — report steps completed + half-budget sensitivity |
| Latency numbers not comparable | measured on different machines/precisions | one machine, one precision, batch 1, median of 100 warm calls — assert in the script |

## Stretch

Add your Lesson 14 ACT (trained on the same dataset, same budget) as a fourth row. "Does a 2023 non-VLA at equal compute embarrass anyone?" is the most clarifying — and most publishable — row in the table.

## References

- Wang et al., *X-VLA*, ICLR 2026. arXiv:2510.10274. LeRobot docs: `xvla`; Hub: `lerobot/xvla-base`.
- *Evo-1: Lightweight VLA with Preserved Semantic Alignment*, 2025. arXiv:2511.04555.
- EO-1 model card + docs via the LeRobot v0.6.0 release blog roster.
- Shukor et al., *SmolVLA*, 2025. arXiv:2506.01844.
- Agarwal et al., *Deep RL at the Edge of the Statistical Precipice*, NeurIPS 2021 — the case for CIs and per-task reporting.
