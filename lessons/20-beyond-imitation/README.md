# Lesson 20 — Beyond Imitation: RL-from-Experience and World Models

A policy trained by imitation cannot be better than the demonstrations it was trained on, and by mid-2026 the field has produced several distinct answers to that limit. This lesson organises those answers by the source of the learning signal, so that you can place a new method into the scheme on first reading, and then tests two of them with your own numbers: you run a world-model policy from LeRobot v0.6 against the fine-tuned SmolVLA you already have, and you measure whether a learned reward model is well enough calibrated to be trusted as an evaluator. The lesson closes with an audit of one frontier announcement, Skild's S1, as practice in reading claims that come without weights, a paper, or an independent evaluation.

| | |
|---|---|
| **Phase** | 6 — Frontier |
| **Time** | ~1.5 sessions for the survey note, ~1 session of hands-on work including one cloud evaluation run, and ~2 h for the S1 claims audit |
| **Cost** | ~$3–6 of cloud GPU for the world-model-policy evaluation; reward-model inference runs on the Mac |
| **Prerequisites** | 10 (you have calibrated a reward classifier), 17–18 (VLA mechanics, and a fine-tuned policy whose rollouts you will reuse), 09 (intuitions about RL from offline data) |
| **Feeds into** | 21 (verification is the same problem at the planner level), 22 (capstone option 1 runs this lesson's loop on hardware), H4 (the DAgger iteration) |

## Learning objectives

After this lesson you can:

1. **Taxonomize** post-imitation methods by their signal source, meaning who or what says that one outcome was better than another, and place a new paper into the taxonomy on first reading.
2. **Explain** RECAP's advantage conditioning, RL-token bootstrapping, and π0.7's distillation as three answers to the question of how experience improves a generalist, and explain why VLA-JEPA's world model costs nothing at inference.
3. **Predict** where a world-model policy lands relative to your fine-tuned SmolVLA, and then verify the claim that its world model is free at inference by inspecting the actual inference graph.
4. **Quantify** a reward model's trustworthiness in terms of precision, recall, calibration, and expected calibration error, and **decide** the threshold at which it may replace ground-truth evaluation.
5. **Audit** a frontier announcement by separating verified, verifiable-but-unpublished, and unverifiable claims, and commit to a dated, falsifiable expectation.

## Principles

### Why behaviour cloning plateaus

Three independent limits cap the performance of a policy trained purely by imitation. The first is compounding error: the policy visits states that the demonstrations never covered, and in those states it has no training signal at all, so small errors accumulate into failures (Lesson 12 demonstrated this experimentally). The second is the demonstrator ceiling: copying a teleoperator cannot produce a policy more skilled than the teleoperator. The third is the absence of any notion of better: the behaviour-cloning loss treats a barely successful trajectory and a fast, clean one identically, because both are simply data to be reproduced. Every method in this lesson is a way of injecting a preference signal that demonstrations alone do not carry.

### A taxonomy by signal source

The most useful way to organise post-imitation methods is by asking where the preference signal comes from: who or what grades the experience. The table below is the spine of the survey note you will write in Exercise 1.

| Family | Signal source | Exemplar |
|---|---|---|
| Corrections | human takes over when the policy errs; the takeover states are exactly the compounding-error states | HG-DAgger; `lerobot-rollout`'s intervention mode (H4) |
| Advantage conditioning | a value function scores *all* experience (demos, corrections, autonomous rollouts); the policy learns success- and failure-conditioned behavior and is steered to the good side at inference | **RECAP / π*0.6** (PI, Nov 2025): condition on advantage bins, deploy conditioned on high advantage — "RL without ever writing a policy-gradient step" |
| Online RL via VLA priors | classic online RL made sample-feasible by bootstrapping from a VLA's learned representations | **RL Tokens** (PI, Mar 2026, arXiv 2604.23073): extract an RL-amenable token interface from the VLA; precision tasks (an M3 screw insertion) mastered in ~15 minutes of experience |
| Distillation back to generalist | RL-trained specialists + strategy metadata distilled into one steerable model | **π0.7** (Apr 2026): matches or beats the specialists on espresso/box-folding/laundry throughput while staying a single generalist |
| World-model supervision | the future itself: predict consequences of your own actions | **VLA-JEPA** (LeRobot v0.6): Qwen3-VL backbone + V-JEPA2 video world model + flow-matching DiT head — the world model shapes representations *during training* and is deleted at inference; **FastWAM**: ~5B video-generation expert paired with a compact action expert, asking whether test-time imagination is needed at all; **LingBot-VA**: autoregressive video+action prediction, chunk by chunk |
| Learned evaluators | a reward model watches rollouts and scores success | **Robometer / TOPReward** (LeRobot v0.6 reward-models API); TOPReward is the quality gate that filtered MolmoAct2's 38k-episode community corpus |

Two trends run through the table from top to bottom. The first is that the signal keeps getting cheaper: a human demonstration costs more per unit of information than a human correction, which costs more than the policy's own experience, which costs more than a predicted future. The second is that evaluation is itself becoming a learned model. The last row of the table is a reward model that scores rollouts, and once such a model is used to filter training data or to report success rates, its calibration determines whether the numbers built on it can be trusted. That is why the calibration study in Exercise 3 is central to the lesson rather than an aside.

### A second axis: how the task is specified

The taxonomy above sorts methods by who grades the experience. A separate and orthogonal question is how the intent enters the policy at inference time. Every VLA in Lessons 17 to 19 takes a language instruction. Alternatives include a goal image, a trajectory sketch drawn over the scene (RT-Trajectory), and a video demonstration that the policy consumes in context with no weight update (Vid2Robot, ICRT, and, at the largest scale, Skild's S1 in 2026). S1's proposition is to pretrain on episodic data in which the task is specified only by an in-context demonstration, so that the model learns to read a demonstration as a statement of intent rather than as a trajectory to copy. Its announcement claims 96% in-distribution success, an out-of-distribution advantage over language-conditioned VLAs that widens with scale (66% versus 9% at roughly 100k pretraining hours), and that one demonstration is worth about 380 post-training episodes. The weights are closed, there is no paper, and there is no independent evaluation, which is exactly why S1 is the subject of the audit in Exercise 5 rather than something you run.

**Carry forward**

- A post-imitation method is best classified by who grades the experience, because that choice predicts both what the method costs in data and where its gains appear.
- Signal source and task specification are orthogonal axes, and a method should be placed on both before it is compared with anything.
- A world model can pay for itself entirely at training time, as in VLA-JEPA, or be retained at test time, as in FastWAM; the difference is a bet about whether the policy needs to imagine in order to act.
- A learned evaluator can be used at three levels of trust, as a data filter, as a replacement for ground-truth evaluation, and as a reward for RL, and the three demand increasing precision because the cost of a false positive rises at each level.
- A frontier claim without weights, a paper, or a baseline specification is a hypothesis with a date attached, and the useful response is to write down what evidence would change your mind.

| Source | Read for |
|---|---|
| PI: π*0.6/RECAP post (pi.website, Nov 2025) | the three data streams, and what the value function is actually fit on |
| PI: *Precise Manipulation with Efficient Online RL* (pi.website/research/rlt) + arXiv 2604.23073 | what an "RL token" exposes that raw actions do not |
| PI: π0.7 post (pi.website/blog/pi07) | what metadata makes distilled skills steerable |
| LeRobot v0.6.0 release blog + `vla_jepa` docs page | the training and evaluation interfaces you will run in Exercise 2 |
| ETH Robot Learning lecture 8 (World Models, YouTube) | the latent-versus-pixel prediction trade-off, for the note's world-model section |
| Levine's "imitation vs RL" framing (CS 285 lecture 2) | the compounding-error argument stated precisely, for the first ceiling |
| Skild S1 blog (skild.ai/blogs/s1, 2026) | the in-context-learning scaling claim you will audit; read it twice, once accepting every claim and once challenging each |
| RT-Trajectory (arXiv 2311.01977) · Vid2Robot (arXiv 2403.12943) · ICRT (arXiv 2408.15980) | the open, small-scale end of the task-specification spectrum; skim for the conditioning interface rather than the results |

## Exercise 1 — Write the survey note [Write]

The survey note, `NOTE.md`, is where you build the taxonomy and take a position on it. It is two to three pages, and it is a position piece organised around the taxonomy rather than an annotated bibliography: every paragraph should either fill a cell of the table or advance the argument about what you would do with your own hardware.

1. Build the taxonomy table first, with one row per family and columns for the signal source, the data cost per unit of improvement, where the improvement shows up (robustness, peak skill, or throughput), and the sharpest open failure mode. Verify every post-cutoff row against its primary source as you fill it in, and write "?" in any cell you cannot verify.
2. Add a half-page timeline from October 2025 to August 2026 (the tutorial's release, then π*0.6, RL Tokens, π0.7, MolmoAct2, and LeRobot v0.6), annotated with the family each release advances.
3. Write the position. The question is what closes the loop after behaviour cloning for a lab with a $500 robot. Commit to an ordering of the families by expected value at your scale and defend it. Advantage conditioning needs a value function, world-model supervision needs video compute, and corrections need only you and a leader arm, so the answer depends on scale; say how.
4. Add a half-page section on task specification: the spectrum from language to goal image to sketch to in-context demonstration, with one sentence per point on data cost versus test-time flexibility, and a committed answer to what changes for a $500-robot lab if conditioning shifts toward demonstrations. Your leader arm makes demonstrations nearly free while language annotation is not, so say what that implies for your Phase 1 data practices.

**✅ Checkpoint:** the table has at most four "?" cells; the position names a first, second, and third choice for your own hardware track with reasons; and the task-specification section commits to an answer.

## Exercise 2 — Run a world-model policy [Predict → Run]

LeRobot v0.6 ships VLA-JEPA with the claim that its world model improves the policy during training and costs nothing at inference. This exercise checks both halves of that claim: you evaluate the published checkpoint under the same protocol as your Lesson 18 models, and you inspect what actually executes at test time. Predicting the outcome first gives you a stated belief about what world-model supervision is worth, which the evaluation then tests.

1. Before running, write down where you expect the published VLA-JEPA baseline to land on your Lesson 18 subset relative to your SmolVLA fine-tune and the zero-shot model, with a reason; what you expect to find in the inference graph, specifically whether the V-JEPA2 branch is present; and the ratio of milliseconds per chunk you expect relative to SmolVLA.
2. Read the `vla_jepa` documentation page for your installed LeRobot, and note the training and evaluation entry points and the published baseline checkpoint. One checkpoint is published per benchmark family and smoke-tested in continuous integration; start from it rather than training from scratch.
3. Evaluate the published baseline on the same benchmark subset and seeds you used in Lesson 18. This gives you a three-way comparison under one protocol: SmolVLA fine-tuned, VLA-JEPA, and zero-shot.
4. Inspect the checkpoint or the policy class and record what runs in the inference graph. Measure milliseconds per chunk against SmolVLA on the same hardware.
5. If your budget permits, you may instead fine-tune VLA-JEPA on your Lesson 18 dataset following its documentation and report the same table. Either path satisfies the lesson.
6. Reconcile the results with your predictions from step 1.

**✅ Checkpoint:** the three-way table exists; the inference-graph note states what actually runs at test time with the latency numbers beside it; and the prediction is reconciled.

## Exercise 3 — Calibrate a reward model [Predict → Run]

A reward model is only useful if its scores mean what they appear to mean, and you are in an unusually good position to check, because the evaluation rollouts from Lessons 14 and 18 each carry a ground-truth success label from the environment. This exercise scores those rollouts with a v0.6 reward model and measures how well its confidence tracks the truth.

1. Assemble at least 60 episodes with their videos and ground-truth labels, balanced between success and failure where possible, and record the class ratio.
2. Before scoring, write down the precision and recall you expect at the default threshold, and whether you expect the model to be over-confident or under-confident, which corresponds to a reliability curve below or above the diagonal.
3. Write a specification for `reward_calibration/score_episodes.py` and have an AI tool draft it: run one v0.6 reward model (Robometer or TOPReward, following the reward-models API documentation, and checking which input format each expects, since one may take a video file and another a frame sequence) over every episode, and write a per-episode CSV of score and ground-truth label. The check is that the CSV has one row per episode and every score lies in $[0, 1]$.
4. Analyse the scores: the confusion matrix at the default threshold; precision, recall, and F1; a reliability diagram with five bins at this sample size, showing the count per bin, and the expected calibration error; and the full threshold sweep as a precision-recall curve.
5. Build a failure gallery of the five highest-confidence false positives and the five highest-confidence false negatives, with one frame and one sentence each on what misled the model.
6. Reconcile the analysis with your predictions from step 2.

**✅ Checkpoint:** the reliability diagram and expected calibration error are computed, the precision-recall sweep is plotted, the gallery is written, and the prediction is reconciled.

## Exercise 4 — Decide the three-tier verdict [Decide]

A reward model can be deployed in three roles, and each role tolerates a different error rate. Using the precision-recall curve from Exercise 3, state the precision at which you would accept this model as (a) a filter that removes low-quality episodes from a training set, which is the role TOPReward played in MolmoAct2's data pipeline; (b) a replacement for ground-truth evaluation; and (c) a reward signal for reinforcement learning. Explain why the three bars differ by saying what a false positive and a false negative each cost in each role, and state where your measured curve clears each bar.

**✅ Checkpoint:** three bars, each with a reason and a verdict drawn from your numbers.

## Exercise 5 — Audit the S1 claims [Write]

Reading a frontier announcement that has no published weights, no paper, and no baselines is a skill, and S1's announcement is the exercise. The deliverable is `s1_claims_audit.md`, and the discipline is to classify every claim before drawing any conclusion, so that each judgment can be traced to a specific artifact that is present or missing.

1. Extract every quantifiable claim into a table with columns for the claim, its number, what it was measured on as stated, and what is missing to reproduce it. The minimum set is: 96% in-distribution success; 66% versus 9% out-of-distribution at roughly 100k hours; one demonstration equivalent to about 380 post-training episodes, or 50 to 100 hours of teleoperation; the L1 to L5 perturbation ladder, on which the language baseline degrades about three times more; the statement that the same model weights produced every example; and the ten-minute unseen long-horizon tasks.
2. Classify each claim as independently verifiable today, verifiable in principle but unpublished, or unverifiable as stated. For every claim in the second and third categories, name the single artifact (a baseline specification, a task list, trial counts, confidence intervals, a seed policy) whose release would move it up one category.
3. Examine the 66%-versus-9% figure as a case study. List at least three benign explanations that would shrink the gap without the headline being false, such as an under-tuned language baseline, a task distribution chosen to favour demonstrations, or an asymmetric evaluation protocol, and for each one name the published evidence that would rule it out.
4. Write one dated, falsifiable expectation: if the in-context-learning scaling claim is real, what should a peer-reviewed or open replication show within twelve months? You will revisit it at capstone time.

**✅ Checkpoint:** every extracted claim is classified; the case study lists at least three alternative explanations, each paired with the evidence that would discriminate it; and the expectation is dated and falsifiable.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `NOTE.md` | taxonomy table, timeline, task-specification section, and a defended position; post-cutoff cells cited or marked "?" |
| `wm_eval/run_three_way_eval.py` + JSON | the three-way comparison, rerunnable from one command |
| `reward_calibration/` | scoring script, per-episode CSV, reliability diagram, precision-recall sweep, failure gallery |
| `s1_claims_audit.md` | every claim classified; at least three benign alternatives for the headline figure; one dated falsifiable expectation |
| `RESULTS.md` | predictions and reconciliations for Exercises 2 and 3; the three-way table; the inference-graph finding; the three-tier verdict |

## Done when

- [ ] The three-way table (SmolVLA fine-tuned, VLA-JEPA, zero-shot) under one protocol exists.
- [ ] The claim that the world model is free at inference is verified or refuted from the actual inference graph and latency.
- [ ] The reward-model calibration covers at least 60 episodes with a reliability diagram, expected calibration error, and a precision-recall sweep.
- [ ] The three-tier verdict is stated with numbers.
- [ ] The note's position passes this test: a skeptical reader can find your reasons, your evidence, and your uncertainty, each labelled as such.
- [ ] The S1 audit classifies every claim and commits to a dated, falsifiable expectation.

## Self-check

1. RECAP and DAgger both learn from mistakes. What is the difference in signal source, and which one scales beyond the limits of human patience?
2. Why can VLA-JEPA discard its world model at inference while FastWAM keeps its video expert? What different bet is each design making?
3. TOPReward gated MolmoAct2's training corpus. Why does data filtering tolerate a less well-calibrated reward model than RL training does?
4. π0.7 distils RL specialists together with "strategy metadata". What failure of naive distillation does the metadata prevent?
5. Your reward model shows an expected calibration error of 0.15, with high recall at low precision. Which of the three tiers is it fit for, if any?
6. Signal source and task specification are orthogonal axes. Place S1 on both, and name one method from your taxonomy that shares its position on one axis but not the other.
7. S1 reads a demonstration as intent rather than as a trajectory to replay. Which failure mode of plain behaviour cloning does that reframing attack, and which of the three ceilings does it leave untouched?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `vla_jepa` entry points don't match this README | v0.6.x API drift | the documentation page for your installed version wins; record the difference in `RESULTS.md` |
| Reward model scores everything near 0.9 | wrong input format (frames versus video file versus resolution) | check the API's expected input schema; re-encode before concluding miscalibration |
| Reliability diagram is unreadable | 60 episodes across 10 bins is about 6 per bin | use 5 bins, or bootstrap confidence intervals per bin; report the bin counts |
| Three-way comparison shows VLA-JEPA near 0% | checkpoint and benchmark family mismatched | use the checkpoint published for that benchmark family; smoke-test 5 episodes before the full run |
| Survey note grows past 3 pages | recounting papers instead of placing them | every paragraph must fill a taxonomy cell or advance the position; delete the rest |
| Claims audit reads as advocacy or as a takedown | conclusions written before classification | classify first and conclude last; every judgment cites a specific present or missing artifact |

## Going deeper

- **Close a small loop.** Use your calibrated reward model to filter the worst 20% of a training dataset, as TOPReward does, refine your Lesson 18 fine-tune on the filtered set, and re-evaluate. The result is one number: whether learned-reward data curation helped at your scale.
- **Swap the conditioning.** If your installed LeRobot exposes a non-language conditioning interface, such as a goal image or a demonstration prompt (verify this against the documentation for your version rather than assuming), run the smallest possible swap on your Lesson 18 setup and report the change under the same evaluation protocol.
- **Extend the calibration set.** Push to at least 100 episodes and 10 bins, and compare the stability of the expected calibration error against the 60-episode run.

## References

- Physical Intelligence: π*0.6/RECAP (Nov 2025), *Precise Manipulation with Efficient Online RL* (Mar 2026, arXiv 2604.23073), π0.7 (Apr 2026). pi.website/blog.
- LeRobot v0.6.0: release blog, `vla_jepa` documentation, and the reward-models API documentation (Robometer, TOPReward).
- Fang et al., *MolmoAct2*, 2026, arXiv 2605.02881; the data-pipeline section describes TOPReward's gating role.
- Assran et al., V-JEPA2, 2025, the self-supervised video backbone inside VLA-JEPA.
- Kelly et al., *HG-DAgger*, 2019. arXiv:1810.02890.
- Guo et al., *On Calibration of Modern Neural Networks*, 2017, for expected calibration error and reliability diagrams.
- Skild AI, *S1* blog, 2026. skild.ai/blogs/s1 (blog only; no paper or weights as of September 2026).
- Gu et al., *RT-Trajectory*, 2023. arXiv:2311.01977.
- Jain et al., *Vid2Robot*, 2024. arXiv:2403.12943.
- Fu et al., *In-Context Imitation Learning via Next-Token Prediction* (ICRT), 2024. arXiv:2408.15980.
