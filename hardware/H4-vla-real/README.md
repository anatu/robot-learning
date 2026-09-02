# H4 — VLAs on Your Arm

This lesson brings generalist policies to the SO-101 you built in H1 and the dataset you recorded in H2. You will run a pretrained vision-language-action model zero-shot on your rig, fine-tune SmolVLA on your fifty episodes and compare it against the single-task ACT and Diffusion Policy specialists from H3 under the same protocol, probe what the language backbone actually buys, and then close one full DAgger correction loop: run the policy, intervene when it begins to fail, retrain on the corrections, and re-evaluate. Taken together these steps are a small-scale version of how frontier labs improve deployed policies, and they produce the evidence you need to hold a defensible opinion about generalists versus specialists at the fifty-episode scale.

| | |
|---|---|
| **Phase** | Hardware track |
| **Time** | ~1 h zero-shot setup + trials, ~4 h cloud fine-tune (wall-clock), 2 × ~1.5 h eval sessions, ~2 h DAgger session + ~1 h retrain launch |
| **Cost** | ~$6–14 cloud GPU (SmolVLA LoRA fine-tune ~$3–8, DAgger retrain ~$3–6) |
| **Prerequisites** | H2 (dataset + task), H3 (baselines + the pre-registered protocol you'll reuse verbatim), 18 (SmolVLA fine-tune recipe), 16 (async/RTC concepts) |
| **Feeds into** | 22 capstone option 1 (this DAgger loop, iterated), 21's hardware echo |

## Learning objectives

After this lesson you can:

1. **Interpret** a pretrained VLA's zero-shot behaviour on hardware it was not tuned for as a statement about what its pretraining data covered.
2. **Predict and then measure** whether SmolVLA fine-tuned on fifty real episodes beats the single-task specialists on the task they were trained for, and under which conditions the two diverge.
3. **Quantify** what the vision-language backbone contributes, measured on paraphrased instructions, changed object attributes, and distractors, against an ACT control that ignores language.
4. **Run** a DAgger-style correction loop end to end and judge, with confidence intervals, whether it moved the failure mode you targeted.
5. **Defend** a claim about when a generalist beats a specialist at the fifty-episode scale, using your own table as the evidence.

## Principles

### Why a zero-shot policy might work at all

MolmoAct 2 (Ai2, May 2026) is trained in part on community SO-101 data from the Hub. Your robot's embodiment, its camera conventions, and its task family are therefore inside the model's pretraining distribution, and any zero-shot success you observe on your rig is a measurement of how well that distribution covers your setup rather than evidence of generalization in the abstract. This matters for how you write up the result: a policy that succeeds because it has seen many rigs like yours is telling you something about the data, not about the architecture.

MolmoAct 2 is also newer than this course's primary sources, so the specifics of running it cannot be trusted from memory. Before Exercise 1, read the current model card and the LeRobot integration notes. The expectation at the time of writing is that it runs through `lerobot-rollout` with a built-in calibration-correction step; verify the exact flags against the card rather than against this README.

### Fine-tuned generalist versus trained-from-scratch specialist

H3's ACT policy saw fifty episodes of one task and nothing else. A SmolVLA fine-tune sees the same fifty episodes, but on top of web-scale vision-language pretraining and a body of community robot data. The hypothesis this lesson tests is that the pretraining shows up first as robustness rather than as peak success: the fine-tuned generalist should tolerate out-of-distribution start positions, distractors, and paraphrased instructions better than the specialist, even if its success rate on the demonstrated conditions is no higher. Your four-way table in Exercise 3 and the probe matrix in Exercise 4 are the evidence for or against that hypothesis.

The cost of the generalist is inference latency. SmolVLA is a much heavier model than ACT, so it is deployed through the asynchronous machinery from Lesson 16. The documented path for smooth VLA execution on hardware is real-time chunking, enabled with `--inference.type=rtc`.

### The DAgger correction loop

Behaviour cloning fails by compounding drift: the policy makes a small error, reaches a state the demonstrations never covered, and has no signal for what to do there (Lesson 12). DAgger's remedy is to collect data on the learner's own state distribution. You run the policy, take over when it begins to fail, record your recovery, and fine-tune on the combined data, so that the states the policy actually reaches are now represented in the training set. LeRobot ships this as `lerobot-rollout --strategy.type=dagger`, with the SO-101 leader arm as the intervention device. The interaction has three phases: **pause**, in which the policy freezes and the leader arm motors itself to match the follower's pose; **take over**, in which the leader's torque is released and you demonstrate the recovery and the correction while frames are recorded; and **return control**, in which the policy resumes mid-episode.

The order within a takeover matters. Following the recovery-and-correction decomposition of Hu et al. (2025), you first bring the arm back to a good state and only then perform the correct action, so that the dataset contains both how to escape a bad state and what to do once escaped. The same loop, operated at industrial scale with a learned value function in place of a human, is what π*0.6 and RECAP do; Lesson 20 treats that generalization in full.

### What twenty trials can and cannot tell you

With twenty trials per condition, a 95% Wilson interval on fourteen successes spans roughly 48% to 85%. Aggregate in-distribution success will therefore rarely separate two competent policies. What can separate them is the pattern across conditions (in-distribution against out-of-distribution, paraphrase against attribute change) and the distribution of failure types. For that reason the predictions in this lesson are written per cell of the table, not per policy.

**Carry forward**

- Zero-shot success on your rig measures how well the model's pretraining data covers your embodiment, cameras, and task; it is not evidence of generalization in the abstract.
- Pretraining is expected to show up as robustness across conditions before it shows up as higher success on the demonstrated conditions, so a comparison must include out-of-distribution cells to be informative.
- Interventions should begin at incipient failure rather than after full failure, because the purpose of DAgger data is to cover the states the policy actually reaches, and then to show a clean recovery from them.
- At twenty trials per cell, an improvement claim is only defensible when the targeted failure class shrank and no other class grew, since the interval on any single cell is too wide to support more.

| Source | Read for |
|---|---|
| MolmoAct 2 model card + LeRobot integration notes (current) | exact deploy invocation; what SO-101 community data it saw; prompt format |
| [HIL data collection docs](https://huggingface.co/docs/lerobot/hil_data_collection) | the dagger strategy protocol, control bindings, combined-dataset fine-tune flow |
| Lesson 18 `RESULTS.md` (yours) | your LoRA config, VRAM budget, and eval deltas — the fine-tune recipe to transfer |
| Kelly et al. 2019 (HG-DAgger); Hu et al. 2025 (RaC) | why *gated human* intervention beats naive DAgger; recovery/correction decomposition |

## Exercise 1 — Run MolmoAct 2 zero-shot [Predict → Run]

In this exercise you deploy a pretrained generalist on your rig without any fine-tuning and measure it under H3's protocol. The exercise tests objective 1: what you observe is a measurement of how well community pretraining covers your setup, and the interesting part is whether the failures are systematic or random. Because the model is newer than the course's sources, the first step is research rather than robot time.

1. Spend about thirty minutes, before touching the robot, on the current model card, the LeRobot deployment path, the prompt and instruction format, and the calibration-correction mechanism. Write the invocation you intend to use into `RESULTS.md` before you run it.
2. Before running, write down your predicted success out of ten on the in-distribution condition, and whether you expect failures to be systematic (a consistent offset, a wrong grasp height) or random. Give a reason for each.
3. Do a smoke rollout on the H2 task with the e-stop within reach. Watch three episodes before forming a judgment; zero-shot failures are often systematic rather than random, and telling the two apart is what makes the write-up useful.
4. Run the formal trials: **ten trials in the in-distribution condition, following H3's protocol exactly** (the same grid sequence, the same success sentence, the same taxonomy, and a video for every failure).
5. Run a novel-task probe of **three trials** on a task you never demonstrated, such as "push the cube to the left edge". Any success here is generalization from pretraining and should be documented carefully.

**✅ Checkpoint:** ten protocol trials and three novel-task trials are logged with videos; the invocation that actually worked is recorded verbatim; the prediction and the outcome sit side by side in `RESULTS.md`.

## Exercise 2 — Fine-tune SmolVLA on your dataset [Build]

Here you fine-tune SmolVLA on the fifty episodes from H2 using the LoRA recipe you settled in Lesson 18, and deploy it through the real-time chunking path. This exercise produces the generalist row of the comparison and tests the pretraining-plus-fifty-episodes hypothesis in the Principles section. The run takes about four hours of cloud wall-clock.

1. Launch Lesson 18's LoRA recipe pointed at your dataset:
   ```bash
   lerobot-train \
     --dataset.repo_id=<you>/so101_pickplace_50ep \
     --policy.type=smolvla --policy.pretrained_path=lerobot/smolvla_base \
     --output_dir=outputs/train/smolvla_h2 --job_name=smolvla_h2 \
     --policy.device=cuda --wandb.enable=true \
     --policy.repo_id=<you>/smolvla_so101_pickplace
   ```
   Add your Lesson 18 LoRA flags. Your `RESULTS.md` from that lesson records what worked, and `lerobot-train --help` gives the current flag names. An A100 or L4 costs about $3–8 for this run.
2. Check the run part-way through rather than waiting for it to finish. After roughly half the steps, pull the intermediate checkpoint and run three physical trials. A fine-tune that is learning shows task-directed behaviour by that point, and one that is not will save you the second half of the GPU bill.
3. Deploy through the asynchronous, real-time-chunking path:
   ```bash
   lerobot-rollout --strategy.type=base --inference.type=rtc \
     --policy.path=<you>/smolvla_so101_pickplace \
     --robot.type=so101_follower --robot.port=<f-port> --robot.id=H1_follower \
     --robot.cameras="{front: {...}, wrist: {...}}" \
     --task="Pick up the cube and place it in the bin" --duration=60
   ```
   Log the control rate you actually achieve. If the Mac cannot hold it, serve inference from a cloud GPU as in Lesson 16 and record that arrangement in the trial sheet.

**✅ Checkpoint:** the fine-tuned checkpoint is on the Hub; the mid-training sanity trials are logged; deployment runs at a recorded control rate.

## Exercise 3 — Fill the four-way comparison table [Predict → Run]

This exercise produces the central measurement of the lesson: MolmoAct 2 zero-shot, SmolVLA fine-tuned, ACT, and Diffusion Policy, each under the in-distribution and out-of-distribution-position conditions of H3's protocol. It tests objective 2. The prediction step is written per cell because, as the Principles section explains, the aggregate numbers are unlikely to separate the policies while the pattern across conditions can.

1. Before any trial, write your predicted success out of twenty (out of ten for the zero-shot row) for each new cell of the table, and name the one condition under which you expect the generalist and the specialists to diverge. H3's ACT and Diffusion Policy numbers are already known, so predict only the new rows, and in addition predict the size of SmolVLA's in-distribution-to-out-of-distribution gap relative to ACT's.
2. Run the formal trials: **twenty in-distribution and twenty out-of-distribution-position**, following H3's protocol exactly. There is no interleaving with another policy this time, because H3's numbers stand as recorded; the rig must nonetheless match H3's witness marks, and the preflight checklist is how you prove that.
3. Fill the table with Wilson intervals and reconcile it against your predictions cell by cell. For any cell that surprised you, write a hypothesis about the mechanism.

**✅ Checkpoint:** forty protocol trials are logged; the four-way table is complete with confidence intervals; predictions and reconciliation are in `RESULTS.md`.

## Exercise 4 — Probe language and visual robustness [Predict → Run]

The fine-tuned generalist and the ACT specialist differ in one architectural respect that this exercise isolates: SmolVLA conditions on a language instruction and on features from a pretrained vision-language backbone, while ACT ignores language entirely and learns its visual features from fifty episodes. By perturbing the instruction, the object's appearance, and the scene, you measure what the backbone contributes. The design is **three conditions with five trials per cell**, SmolVLA fine-tuned against ACT, with ACT serving as the control.

1. Before running, write for each condition which policy you expect to be hurt and why, attributing the expected effect to language conditioning, to visual features, or to neither.
2. **Paraphrase.** Use "put the block in the container", "grab the cube and drop it in the bin", and one deliberately odd phrasing.
3. **Attribute change.** Swap the cube for one of an unseen colour with the same shape and size.
4. **Distractor.** Reuse H3's out-of-distribution-distractor condition.
5. Score every trial against the same success criterion, with every run on video. Report success per cell, and for any cell that surprised you add a one-line mechanism hypothesis. For example, a policy that is robust to paraphrase but brittle to colour suggests that the language conditioning works while the visual features are overfitted to the training palette.

**✅ Checkpoint:** the probe matrix (three conditions by two policies) is complete with five trials per cell; the videos are labelled; the predictions are reconciled.

## Exercise 5 — Close one DAgger iteration [Predict → Run]

In this exercise you run the full correction loop once: choose a failure mode, collect intervention episodes aimed at it, retrain, and re-evaluate. The exercise tests objective 4, and the principle it exercises is that on-policy correction data moves a specific failure mode, which is a narrower and more testable claim than "more data helps". Budget about three hours plus the retraining run.

1. Choose the target from the fine-tuned policy's dominant failure class in the Exercise 3 taxonomy histogram. The loop aims at that class rather than at data volume in general. Before collecting anything, write down the class, its current count out of twenty, the count you expect after one iteration, and whether you expect any other class to grow as a side effect.
2. Collect about fifteen intervention episodes:
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
   The script prints its control bindings at launch, and those bindings take precedence over any documentation. For each intervention, let the policy begin to fail without letting it fail completely, pause, take over, **recover to a good state and then perform the correct action cleanly**, and return control. Log the interventions per episode.
3. Fine-tune from the Exercise 2 checkpoint on the combined data (the base fifty episodes plus the fifteen intervention episodes), following the documented flow with `--policy.pretrained_path=<exercise-2 checkpoint>` and fewer steps, on the order of 20k. Expect about $3–6.
4. Re-evaluate with **twenty in-distribution trials** under the protocol, plus ten trials concentrated on the targeted failure condition. Compare before and after with confidence intervals. At this sample size, claim an improvement only if the targeted failure class shrank and overall in-distribution success did not regress.

**✅ Checkpoint:** one full loop is closed, from policy to interventions to retraining to re-evaluation, with before-and-after numbers on the targeted failure mode and your prediction beside them.

## Exercise 6 — State the verdict [Decide]

This final exercise tests objective 5. In `RESULTS.md`, state the claim you now hold about generalists versus specialists at the fifty-episode scale, cite the rows of your table that support it and the rows that cut against it, and name the confound at twenty trials per cell to which your claim is most exposed.

**✅ Checkpoint:** one claim, its supporting rows, its opposing rows, and one named confound.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| Four-way comparison table | MolmoAct2-zs / SmolVLA-ft / ACT / DP × ID (+ OOD where run), Wilson CIs, N stated per cell |
| Probe matrix + videos | Exercise 4's condition × policy grid, mechanism hypotheses included |
| `<you>/h4_dagger_r1` dataset + retrained checkpoint | intervention episodes visible in the visualizer with autonomous + human segments |
| Before/after DAgger table | targeted failure class + overall ID, both with CIs; honest verdict |
| `RESULTS.md` | predictions vs outcomes for Exercises 1, 3, 4, 5; the Exercise 6 claim with its supporting and opposing rows |

## Done when

- [ ] The four-way in-distribution table is complete at ten or more trials per cell (twenty for the fine-tune), under H3's protocol throughout.
- [ ] The language and robustness probes quantify three conditions against the ACT control.
- [ ] One DAgger iteration measurably moved the targeted failure mode, or the write-up explains why it did not, with video evidence.
- [ ] Every [Predict → Run] exercise has its prediction written before the trials and reconciled after them.
- [ ] The zero-shot behaviour is documented well enough that a reader learns something about what community pretraining covers.

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
| Zero-shot arm moves confidently to wrong workspace region | calibration/frame convention mismatch with pretraining data | this is what the model card's calibration-correction setup addresses; recheck those steps before concluding the model cannot generalize |
| DAgger takeover feels fighty | pausing then grabbing before leader finishes motoring to match pose | wait for the printed "takeover ready" state; the pause→match→takeover sequence is deliberate |
| Retrained policy worse overall | correction data swamped base behavior (too many steps / too high LR on 15 episodes) | fewer steps, keep base 50 in the mix (combined dataset, per docs), re-check LoRA rank |
| Four-way table contradicts itself across sessions | rig drift between H3's and H4's sessions | preflight + witness marks; if drift is real, rerun the stale baseline rather than footnoting it |
| Any post-cutoff invocation fails | this README's expectations, not the tool | model card / `--help` / LeRobot release notes are authoritative; record what actually worked |

## Going deeper

- **A second DAgger iteration.** Repeat Exercise 5 targeting the new dominant failure class and plot success per iteration. Two points on that curve tell you whether the loop saturates, and the curve is the backbone of Capstone option 1.
- **A full probe matrix.** Extend Exercise 4 to five trials across three paraphrases, an attribute change, a distractor, and a lighting change, and add MolmoAct 2 zero-shot as a third column.

## References

- [HIL data collection (dagger) docs](https://huggingface.co/docs/lerobot/hil_data_collection) — protocol + commands verified Aug 2026; teleop bindings printed at launch are authoritative.
- MolmoAct 2 (Ai2, May 2026) — model card at release; post-cutoff, verify everything.
- Ross et al. 2011 (DAgger); Kelly et al. 2019 (HG-DAgger); Hu et al. 2025 (RaC, arXiv:2509.07953); PI π*0.6/RECAP (2025).
- Your Lesson 18 recipe + H3 `PROTOCOL.md`.
