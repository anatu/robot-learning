# H4 — VLAs on Your Arm

Generalist policies meet your $400 robot: a foundation model zero-shot, your own SmolVLA fine-tune head-to-head against H3's specialists, a language-generalization probe, and one full DAgger correction loop — the budget version of how frontier labs actually improve deployed policies.

| | |
|---|---|
| **Phase** | Hardware track |
| **Time** | ~1 h zero-shot setup + trials, ~4 h cloud fine-tune (wall-clock), 2 × ~1.5 h eval sessions, ~2 h DAgger session + ~1 h retrain launch |
| **Cost** | ~$6–14 cloud GPU (SmolVLA LoRA fine-tune ~$3–8, DAgger retrain ~$3–6) |
| **Prerequisites** | H2 (dataset + task), H3 (baselines + the pre-registered protocol you'll reuse verbatim), 18 (SmolVLA fine-tune recipe), 16 (async/RTC concepts) |
| **Feeds into** | 22 capstone option 1 (this DAgger loop, iterated), 21's hardware echo |

## Learning objectives

After this lesson you can:

1. **Interpret** a pretrained VLA's zero-shot behavior on hardware it wasn't tuned for as a statement about its pretraining distribution.
2. **Predict and then measure** whether SmolVLA fine-tuned on 50 real episodes beats single-task specialists on their home turf, and where.
3. **Quantify** what the VLM backbone buys — paraphrase, attributes, distractors — against an ACT control.
4. **Run** a DAgger-style correction loop end-to-end and judge, with CIs, whether it moved the targeted failure mode.
5. **Defend** a claim about when a generalist beats a specialist at the 50-episode scale, with your own table as evidence.

## Principles

**Why zero-shot might work at all.** MolmoAct 2 (Ai2, May 2026) trains on community SO-101 data from the Hub — your robot's embodiment, camera conventions, and task family are *in its pretraining distribution*. Zero-shot success on your rig is a statement about that distribution, not magic. It's also post-cutoff for this course's primary sources: before Exercise 1, pull the current model card and LeRobot integration notes (the scaffold expectation: it runs via `lerobot-rollout` with built-in calibration correction — verify the exact flags against the card; don't trust this README's memory of them).

**Fine-tune vs specialist.** H3's ACT saw 50 episodes of one task from scratch. SmolVLA-ft sees the same 50 episodes on top of web-scale vision-language pretraining plus community robot data. The bet: pretraining buys robustness (OOD positions, distractors, paraphrase) more than peak ID success. Your table tests the bet. Latency is the tax: SmolVLA inference is heavier than ACT's, so deployment uses Lesson 16's machinery — `--inference.type=rtc` (real-time chunking) is the documented path for smooth VLA execution.

**The DAgger loop.** BC fails on compounding drift into states demos never covered (Lesson 12). DAgger's fix: collect *on the learner's own distribution* — run the policy, intervene at incipient failure, record the recovery, fine-tune on combined data. LeRobot ships this as `lerobot-rollout --strategy.type=dagger` with the SO-101 leader as the intervention device: **pause** (policy freezes, leader motors to match follower pose) → **take over** (leader torque off, you demonstrate recovery + correction, frames recorded) → **return control** (policy resumes mid-episode). Recovery *then* correction, per RaC (Hu et al. 2025); π*0.6/RECAP is this loop at industrial scale — Lesson 20 gives it the full treatment.

**What N=20 can and can't tell you.** A 95% Wilson interval on 14/20 spans roughly 48–85%. Aggregate ID success will rarely separate two decent policies; the *pattern* across conditions (ID vs OOD, paraphrase vs attribute) and the failure taxonomy will. Write predictions per cell, not per policy.

**Carry forward**

- Zero-shot success is a measurement of pretraining coverage, not of generalization in the abstract.
- Pretraining is expected to show up as robustness across conditions before it shows up as peak ID success.
- Intervene at *incipient* failure so the dataset contains the states the policy actually reaches, then a clean recovery from them.
- At N=20 per cell, claim improvement only when the targeted failure class shrank *and* nothing else regressed.

| Source | Read for |
|---|---|
| MolmoAct 2 model card + LeRobot integration notes (current) | exact deploy invocation; what SO-101 community data it saw; prompt format |
| [HIL data collection docs](https://huggingface.co/docs/lerobot/hil_data_collection) | the dagger strategy protocol, control bindings, combined-dataset fine-tune flow |
| Lesson 18 `RESULTS.md` (yours) | your LoRA config, VRAM budget, and eval deltas — the fine-tune recipe to transfer |
| Kelly et al. 2019 (HG-DAgger); Hu et al. 2025 (RaC) | why *gated human* intervention beats naive DAgger; recovery/correction decomposition |

## Exercise 1 — Zero-shot MolmoAct 2 [Predict → Run]

Tests objective 1: what community pretraining covers.

1. Research pass (30 min, before touching the robot): current model card, LeRobot deploy path, prompt/instruction format, calibration-correction mechanism. Write the working invocation into `RESULTS.md` *before* running it.
2. **Write first:** your predicted ID success out of 10, and whether failures will be systematic (consistent offset, wrong grasp height) or random. State why.
3. Smoke rollout on the H2 task, e-stop in reach. Watch three episodes before judging — zero-shot failure modes are often systematic rather than random, and the difference matters for the writeup.
4. Formal trials: **10 trials, ID condition, H3's protocol verbatim** (same grid sequence, same success sentence, same taxonomy + video per failure).
5. Novel-task probe: **3 trials** on a task you never demonstrated (e.g. "push the cube to the left edge"). Any success here is pretraining generalization, worth documenting carefully.

**✅ Checkpoint:** 10 protocol trials + 3 novel-task trials logged with videos; the invocation that actually worked recorded verbatim; prediction and outcome side by side.

## Exercise 2 — Fine-tune SmolVLA on your data [Build]

Tests objective 2: the pretraining-plus-50-episodes bet. Cloud, ~4 h wall-clock.

1. Lesson 18's LoRA recipe pointed at your dataset:
   ```bash
   lerobot-train \
     --dataset.repo_id=<you>/so101_pickplace_50ep \
     --policy.type=smolvla --policy.pretrained_path=lerobot/smolvla_base \
     --output_dir=outputs/train/smolvla_h2 --job_name=smolvla_h2 \
     --policy.device=cuda --wandb.enable=true \
     --policy.repo_id=<you>/smolvla_so101_pickplace
   ```
   plus your Lesson 18 LoRA flags (your `RESULTS.md` there is the source of truth for what worked; `lerobot-train --help` for current names). A100/L4, ~$3–8.
2. Sanity-eval in the loop: after ~half the steps, pull the intermediate checkpoint and run 3 physical trials. A fine-tune that's learning shows task-directed behavior by then; one that isn't saves you the second half of the GPU bill.
3. Deploy via async/RTC:
   ```bash
   lerobot-rollout --strategy.type=base --inference.type=rtc \
     --policy.path=<you>/smolvla_so101_pickplace \
     --robot.type=so101_follower --robot.port=<f-port> --robot.id=H1_follower \
     --robot.cameras="{front: {...}, wrist: {...}}" \
     --task="Pick up the cube and place it in the bin" --duration=60
   ```
   Log the achieved control rate; if the Mac can't hold it, serve inference from a cloud GPU per Lesson 16 and note the setup in the trial sheet.

**✅ Checkpoint:** fine-tuned checkpoint on Hub; mid-training sanity trials logged; deployment runs at a recorded control rate.

## Exercise 3 — The four-way table [Predict → Run]

Tests objective 2 as a measurement.

1. **Write first**, per cell of the table MolmoAct2-zs / SmolVLA-ft / ACT / DP × {ID, OOD-position}: predicted success out of 20 (10 for zero-shot), and the one condition where you expect the generalist and the specialists to *diverge*. H3's ACT/DP numbers are already known — predict only the new rows, and predict the ID-vs-OOD gap for SmolVLA-ft relative to ACT's.
2. Formal trials: **20 ID + 20 OOD-position**, H3 protocol verbatim, interleaved A/B with nothing (H3 numbers stand as recorded — but rig must match H3's witness marks; preflight proves it).
3. Fill the table with Wilson CIs; reconcile against your predictions cell by cell. A cell that surprised you gets a mechanism hypothesis.

**✅ Checkpoint:** 40 protocol trials logged; the four-way table complete with CIs; predictions and reconciliation in `RESULTS.md`.

## Exercise 4 — Language & robustness probes [Predict → Run]

Tests objective 3: what the VLM backbone buys. **3 conditions × 5 trials per cell**, SmolVLA-ft vs ACT (ACT ignores language — it's the control).

1. **Write first:** for each condition, which policy you expect to be hurt and why (language conditioning vs visual features vs neither).
2. **Paraphrase:** "put the block in the container", "grab the cube and drop it in the bin", one deliberately odd phrasing.
3. **Attribute change:** swap cube color (unseen color, same shape/size).
4. **Distractor:** H3's OOD-distractor condition, reused.
5. Score against the same success criterion; every run on video. Report per-cell success + a one-line mechanism hypothesis per surprising cell (e.g. paraphrase-robust but color-brittle → language conditioning works, visual features overfit to training palette).

**✅ Checkpoint:** probe matrix (3 conditions × 2 policies) complete, 5 trials per cell, videos labeled; predictions reconciled.

## Exercise 5 — One DAgger iteration [Predict → Run]

Tests objective 4: targeted on-policy data moves a specific failure mode. ~3 h + retrain.

1. Pick the target: your fine-tune's dominant failure class from Exercise 3's taxonomy histogram. The loop aims at *that*, not at general data volume. **Write first:** the failure class, its current count out of 20, and the count you expect after one iteration — plus whether you expect any *other* class to grow.
2. Collect ~15 intervention episodes:
   ```bash
   lerobot-rollout --strategy.type=dagger --inference.type=rtc \
     --policy.path=<you>/smolvla_so101_pickplace \
     --robot.type=so101_follower --robot.port=<f-port> --robot.id=H1_follower \
     --robot.cameras="{front: {...}, wrist: {...}}" \
     --teleop.type=so101_leader --teleop.port=<l-port> --teleop.id=H1_leader \
     --dataset.repo_id=<you>/h4_dagger_r1 \
     --dataset.single_task="Pick up the cube and place it in the bin" \
     --strategy.num_episodes=15
   ```
   The script prints its control bindings at launch — those are authoritative over any doc. Protocol per intervention: let it fail *incipiently* (not fully), pause, take over, **recover to a good state, then correct cleanly**, return control. Log interventions per episode.
3. Fine-tune from the Exercise 2 checkpoint on the combined data (base 50 + dagger 15; the docs' flow — `--policy.pretrained_path=<exercise-2 checkpoint>`, fewer steps, e.g. ~20k). ~$3–6.
4. Re-evaluate: **20 ID trials**, protocol verbatim, plus 10 trials concentrated on the targeted failure condition. Before/after with CIs — at this N, claim improvement only if the targeted failure class shrank *and* overall ID didn't regress.

**✅ Checkpoint:** one full loop closed: policy → interventions → retrain → re-eval, with before/after numbers on the targeted failure mode and the prediction beside them.

## Exercise 6 — The verdict [Decide]

Tests objective 5. In `RESULTS.md`, state the generalist-vs-specialist claim you now hold at the 50-episode scale, with the table rows that support it and the rows that cut against it. Name the confound at N=20 that your claim is most exposed to.

**✅ Checkpoint:** one claim, supporting rows, opposing rows, one named confound.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| Four-way comparison table | MolmoAct2-zs / SmolVLA-ft / ACT / DP × ID (+ OOD where run), Wilson CIs, N stated per cell |
| Probe matrix + videos | Exercise 4's condition × policy grid, mechanism hypotheses included |
| `<you>/h4_dagger_r1` dataset + retrained checkpoint | intervention episodes visible in the visualizer with autonomous + human segments |
| Before/after DAgger table | targeted failure class + overall ID, both with CIs; honest verdict |
| `RESULTS.md` | predictions vs outcomes for Exercises 1, 3, 4, 5; the Exercise 6 claim with its supporting and opposing rows |

## Done when

- [ ] Four-way ID table complete at ≥ 10 trials/cell (20 for the fine-tune), H3 protocol throughout.
- [ ] Language/robustness probes quantify 3 conditions against the ACT control.
- [ ] One DAgger iteration measurably moved the targeted failure mode, or the writeup explains why not with video evidence.
- [ ] Every [Predict → Run] has its prediction written before the trials and reconciled after.
- [ ] Zero-shot behavior documented well enough that a reader learns something about community-pretraining coverage.

## Self-check

1. What would flat-equal ID success between SmolVLA-ft and ACT, but a large OOD gap in SmolVLA's favor, tell you? What's the confound at N=20?
2. Why intervene at *incipient* failure rather than after full failure — trace it through what states enter the dataset.
3. Why recovery *then* correction (RaC's decomposition) instead of just demonstrating the right behavior from wherever the arm is?
4. MolmoAct 2 zero-shot succeeds at your H2 task but fails the novel task. Three candidate explanations, and which probe distinguishes them?
5. Where does this DAgger loop stop and RECAP begin? (Lesson 20 preview — answer in one sentence about the learning signal.)

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| SmolVLA rollout jerky/stalling | inference latency vs control rate, RTC not enabled | `--inference.type=rtc`; else serve from cloud GPU (Lesson 16 stack) |
| Zero-shot arm moves confidently to wrong workspace region | calibration/frame convention mismatch with pretraining data | that's the "calibration correction" mechanism's job — recheck the model card's setup steps before concluding "can't generalize" |
| DAgger takeover feels fighty | pausing then grabbing before leader finishes motoring to match pose | wait for the printed "takeover ready" state; the pause→match→takeover sequence is deliberate |
| Retrained policy worse overall | correction data swamped base behavior (too many steps / too high LR on 15 episodes) | fewer steps, keep base 50 in the mix (combined dataset, per docs), re-check LoRA rank |
| Four-way table contradicts itself across sessions | rig drift between H3's and H4's sessions | preflight + witness marks; if drift is real, rerun the stale baseline rather than footnoting it |
| Any post-cutoff invocation fails | this README's expectations, not the tool | model card / `--help` / LeRobot release notes are authoritative; record what actually worked |

## Going deeper

- **Second DAgger iteration.** Repeat Exercise 5 targeting the new dominant failure class; plot success per iteration. This is Capstone option 1's spine — two points on the curve tell you whether it saturates.
- **Full probe matrix.** Extend Exercise 4 to 5 trials × {paraphrase ×3, attribute, distractor, lighting} and add MolmoAct2-zs as a third column.

## References

- [HIL data collection (dagger) docs](https://huggingface.co/docs/lerobot/hil_data_collection) — protocol + commands verified Aug 2026; teleop bindings printed at launch are authoritative.
- MolmoAct 2 (Ai2, May 2026) — model card at release; post-cutoff, verify everything.
- Ross et al. 2011 (DAgger); Kelly et al. 2019 (HG-DAgger); Hu et al. 2025 (RaC, arXiv:2509.07953); PI π*0.6/RECAP (2025).
- Your Lesson 18 recipe + H3 `PROTOCOL.md`.
