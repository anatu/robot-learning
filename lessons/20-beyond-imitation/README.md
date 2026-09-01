# Lesson 20 — Beyond Imitation: RL-from-Experience & World Models

Imitation plateaus at the quality of its demonstrations. Map the field's answers as of mid-2026 — advantage conditioning, RL tokens, distilled specialists, world-model policies, reward models — then run two of them yourself and measure whether a learned reward model can be trusted as an evaluator. A second axis runs through the survey: how intent enters the policy at all — language, goal image, or an in-context demonstration (Skild S1) — and Part 4 audits S1's claims as an exercise in reading frontier lab announcements.

| | |
|---|---|
| **Phase** | 6 — Frontier |
| **Time** | ~2 sessions for the survey note + ~1 session hands-on (plus one cloud eval run) + ~2 h for the S1 claims audit |
| **Cost** | ~$3–6 cloud GPU for the world-model-policy eval; reward-model inference is Mac-feasible |
| **Prerequisites** | 10 (you've trained a reward classifier and know what miscalibration costs), 17–18 (VLA mechanics + a fine-tuned policy whose rollouts you'll reuse), 09 (offline-data RL intuitions) |
| **Feeds into** | 21 (verification is the same problem at the planner level), 22 (capstone option 1 is this lesson's loop on hardware), H4 (the DAgger iteration) |

## Learning objectives

After this lesson you can:

1. **Taxonomize** post-imitation methods by their *signal source* — who or what says "this was better" — and place any new paper into the taxonomy on a first read.
2. **Explain** RECAP's advantage conditioning, RL-token bootstrapping, and π0.7's distillation as three different answers to "how does experience improve a generalist".
3. **Explain** why VLA-JEPA's world model costs nothing at inference, and what question FastWAM's design poses about test-time imagination.
4. **Run** a world-model policy and a reward model from LeRobot v0.6 against a benchmark you already have numbers for.
5. **Quantify** a reward model's trustworthiness — precision/recall, calibration, and the threshold at which you'd let it replace ground-truth evaluation.
6. **Place** a policy's task-specification interface — language, goal image, trajectory sketch, in-context demo — on a spectrum, and state what each point costs in data and buys in test-time flexibility.
7. **Audit** a frontier lab announcement: separate verified, verifiable-but-unpublished, and unverifiable claims, and write down what evidence would change your mind.

## Background

**Why BC plateaus.** Three independent ceilings: (1) compounding error — the policy visits states demos never covered and has no signal there; (2) the demonstrator ceiling — you cannot exceed the teleoperator's skill by copying them; (3) no notion of *better* — BC's loss is indifferent between a barely-successful and a crisp trajectory. Everything below is a way of injecting a preference signal that demos alone lack.

**The taxonomy, by signal source** (your survey note's spine):

| Family | Signal source | Exemplar |
|---|---|---|
| Corrections | human takes over when the policy errs; the takeover states are exactly the compounding-error states | HG-DAgger; `lerobot-rollout`'s intervention mode (H4) |
| Advantage conditioning | a value function scores *all* experience (demos, corrections, autonomous rollouts); the policy learns success- and failure-conditioned behavior and is steered to the good side at inference | **RECAP / π*0.6** (PI, Nov 2025): condition on advantage bins, deploy conditioned on high advantage — "RL without ever writing a policy-gradient step" |
| Online RL via VLA priors | classic online RL made sample-feasible by bootstrapping from a VLA's learned representations | **RL Tokens** (PI, Mar 2026, arXiv 2604.23073): extract an RL-amenable token interface from the VLA; precision tasks (an M3 screw insertion) mastered in ~15 minutes of experience |
| Distillation back to generalist | RL-trained specialists + strategy metadata distilled into one steerable model | **π0.7** (Apr 2026): matches or beats the specialists on espresso/box-folding/laundry throughput while staying a single generalist |
| World-model supervision | the future itself: predict consequences of your own actions | **VLA-JEPA** (LeRobot v0.6): Qwen3-VL backbone + V-JEPA2 video world model + flow-matching DiT head — the world model shapes representations *during training* and is deleted at inference; **FastWAM**: ~5B video-generation expert paired with a compact action expert, asking whether test-time imagination is needed at all; **LingBot-VA**: autoregressive video+action prediction, chunk by chunk |
| Learned evaluators | a reward model watches rollouts and scores success | **Robometer / TOPReward** (LeRobot v0.6 reward-models API); TOPReward is the quality gate that filtered MolmoAct2's 38k-episode community corpus |

Two through-lines to carry into the note: *the signal keeps getting cheaper* (human demo → human correction → self-experience → predicted futures), and *evaluation is becoming a model too* — which is why the calibration study in Part 3 is not a side quest. A reward model you can't calibrate is a benchmark you can't trust.

**The second axis — task specification.** The taxonomy above sorts methods by who grades experience. An orthogonal axis is how intent enters the policy at inference: a language instruction (every VLA in Lessons 17–19), a goal image, a trajectory sketch (**RT-Trajectory**), or a video demonstration consumed in context with no weight update (**Vid2Robot**, **ICRT**, and — at the largest scale — **Skild S1**, 2026). S1's bet: pre-train on episodic data where the task is specified *only* by an in-context demo, so the demo is read as intent, not as a trajectory to copy. Its blog claims 96% in-distribution success, a widening OOD gap over language-conditioned VLAs with scale (66% vs 9% at ~100k pre-training hours), and one demo worth ~380 post-training episodes. Closed weights, no paper, no independent eval — which is exactly why it is Part 4's audit subject rather than something you run.

| Source | Read for |
|---|---|
| PI: π*0.6/RECAP post (pi.website, Nov 2025) | the three data streams and what the value function is actually fit on |
| PI: *Precise Manipulation with Efficient Online RL* (pi.website/research/rlt) + arXiv 2604.23073 | what an "RL token" exposes that raw actions don't |
| PI: π0.7 post (pi.website/blog/pi07) | what metadata makes distilled skills *steerable* |
| LeRobot v0.6.0 release blog + `vla_jepa` docs page | the concrete training/eval interfaces you'll run in Part 2 |
| ETH Robot Learning lecture 8 (World Models, YouTube) | the latent-vs-pixel prediction trade-off, for the note's world-model section |
| Levine's "imitation vs RL" framing (CS 285 lecture 2) | the compounding-error argument stated precisely, for ceiling (1) |
| Skild S1 blog (skild.ai/blogs/s1, 2026) | the ICL-vs-VLA scaling claim you'll audit in Part 4 — read it twice: once credulous, once adversarial |
| RT-Trajectory (arXiv 2311.01977) · Vid2Robot (arXiv 2403.12943) · ICRT (arXiv 2408.15980) | the open, small-scale end of the task-specification spectrum — skim for the conditioning interface, not the results |

## Part 1 — The survey note (~4–6 h)

Produces `NOTE.md`, 3–4 pages. This is a *position piece with a taxonomy*, not an annotated bibliography.

1. Build the taxonomy table first: one row per method family, columns = signal source, data cost per unit of improvement, where the improvement shows up (robustness vs peak skill vs throughput), sharpest open failure mode. Verify every post-cutoff row against its primary source as you fill it; unverifiable cells get "?".
2. Add a half-page timeline figure, Oct 2025 → Aug 2026 (tutorial ships → π*0.6 → RLT → π0.7 → MolmoAct2 → LeRobot v0.6), annotated with which taxonomy family each release advances.
3. Write the position: answer "what closes the loop after BC for a $500-robot lab?" — commit to an ordering of the families by expected value *at your scale*, and defend it. (Advantage conditioning needs a value function; world-model supervision needs video compute; corrections need only you and a leader arm. Scale changes the answer — say how.)
4. Add a half-page task-specification section: the language → goal image → sketch → in-context demo spectrum, one sentence per point on data cost vs test-time flexibility, and a committed answer to "what changes for a $500-robot lab if conditioning shifts demo-ward?" (Your leader arm makes demos nearly free; language annotation isn't. Say what that does to your Phase-1 data practices.)

**✅ Checkpoint:** the table has ≤ 4 "?" cells; the position section names a first, second, and third choice for your own hardware track, with reasons; the spec-spectrum section commits to an answer, not a survey shrug.

## Part 2 — Run a world-model policy (~1 session + 1 cloud run, ~$3–6)

The v0.6 claim is that world-model supervision is free at inference. Check the story end-to-end.

1. Read the `vla_jepa` docs page in your installed LeRobot version; note the training/eval entrypoints and the published baseline checkpoint (the release shipped one per benchmark family, CI-smoke-tested — start from that checkpoint, do not train from scratch).
2. Evaluate the published VLA-JEPA baseline on the same benchmark subset + seeds you used in Lesson 18. You now have a three-way, same-protocol comparison for free: SmolVLA-ft vs VLA-JEPA vs zero-shot.
3. Confirm the "free at inference" claim mechanically: inspect the checkpoint or the policy class and record what runs in the inference graph (the V-JEPA2 branch should be absent); measure ms/chunk vs SmolVLA on the same hardware.
4. If your budget allows one training run instead: fine-tune VLA-JEPA on your Lesson 18 dataset per its docs and report the same table. Either path satisfies the lesson.

**✅ Checkpoint:** the three-way table exists; the inference-graph note states what you found actually runs at test time, with the latency numbers beside it.

## Part 3 — Reward-model calibration (~3–4 h, mostly Mac)

You already own the perfect test set: your Lessons 14/18 eval rollouts, each with a ground-truth success label from the environment. Turn them on the reward models.

1. Assemble ≥ 100 episodes (videos + GT labels), balanced across success/failure if possible; note the actual class ratio.
2. Run one v0.6 reward model (Robometer or TOPReward — per the reward-models API docs; check which input format each expects, video vs frame sequence) over every episode; collect per-episode scores.
3. Analyze: confusion matrix at the default threshold; precision/recall/F1; a reliability diagram (10 score bins vs empirical success rate) with ECE; the full threshold sweep (precision–recall curve).
4. Failure gallery: the 5 highest-confidence false positives and false negatives, one frame each, one sentence on what fooled the model.
5. Write the verdict: at what precision would you accept the reward model as (a) a data-filtering gate (TOPReward's actual job in MolmoAct2's pipeline), (b) a replacement for ground-truth eval, (c) an RL reward. These bars are different — say why, and state where your measured curve clears them.

**✅ Checkpoint:** reliability diagram + ECE computed; the three-tier verdict is written with numbers, not vibes.

## Part 4 — The S1 claims audit (~2 h, desk-only)

Reading a frontier lab announcement without published weights, paper, or baselines is a skill. S1's blog is the exercise. Deliverable: `s1_claims_audit.md`.

1. Extract every quantifiable claim into a table: claim, number, what it was measured on (as stated), what's missing to reproduce it. Minimum set: 96% ID success; 66% vs 9% OOD at ~100k hours; 1 demo ≈ 380 post-training episodes ≈ 50–100 h of teleop; the L1–L5 perturbation ladder with the language baseline degrading ~3× more; "the same model weights produced every example"; 10-minute unseen long-horizon tasks.
2. Classify each claim: independently verifiable today / verifiable in principle but unpublished / unverifiable as stated. For every claim in the last two bins, name the single artifact (baseline spec, task list, trial counts, CIs, seed policy) whose release would move it up a bin.
3. Case-study the 66%-vs-9% figure: list ≥ 3 benign explanations that would shrink the gap without the headline being false (under-tuned language baseline, task distribution chosen demo-side, eval-protocol asymmetry, ...), and for each, the published evidence that would rule it out.
4. Write one dated, falsifiable expectation: if the ICL-scaling claim is real, what should a peer-reviewed or open replication show within 12 months? Revisit at capstone time.

**✅ Checkpoint:** every extracted claim is classified; the case study has ≥ 3 alternatives each paired with discriminating evidence; the expectation is dated and falsifiable.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `NOTE.md` | taxonomy table + timeline + task-specification section + a defended position; post-cutoff cells cited or "?" |
| `wm_eval/` | scripts + JSON for the three-way comparison; rerunnable from one command |
| `reward_calibration/` | scoring script, per-episode CSV, reliability diagram, threshold sweep, failure gallery |
| `RESULTS.md` | the three-way table; the inference-graph finding; the three-tier reward-model verdict |
| `s1_claims_audit.md` | every claim classified; ≥ 3 benign alternatives for the headline figure; one dated falsifiable expectation |

## Done when

- [ ] The three-way (SmolVLA-ft / VLA-JEPA / zero-shot) same-protocol table exists.
- [ ] The "free at inference" claim is verified or refuted from the actual inference graph + latency.
- [ ] Reward-model calibration covers ≥ 100 episodes with reliability diagram, ECE, and threshold sweep.
- [ ] The note's position section survives this test: a skeptical reader can find your reasons, your evidence, and your uncertainty, each labeled as such.
- [ ] The S1 audit classifies every claim and commits to a dated, falsifiable expectation.

## Self-check

1. RECAP and DAgger both learn from mistakes. What's the signal-source difference, and which scales past human patience?
2. Why can VLA-JEPA delete its world model at inference while FastWAM keeps its video expert? What different bet is each making?
3. TOPReward gated MolmoAct2's training corpus. Why does *data filtering* tolerate a worse-calibrated reward model than *RL training* does?
4. π0.7 distills RL specialists with "strategy metadata". What failure of naive distillation is the metadata preventing?
5. Your reward model shows ECE of 0.15 with high recall at low precision. Which of the three deployment tiers (filter/eval/RL-reward) is it fit for, if any?
6. Signal source and task specification are orthogonal axes. Place S1 on both, and name one method from your taxonomy that shares its position on one axis but not the other.
7. S1 reads a demo as intent, not a trajectory to replay. Which failure mode of plain BC does that reframing attack, and which of the three BC ceilings does it leave untouched?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `vla_jepa` entrypoints don't match this README | v0.6.x API drift | the docs page for your installed version wins; record the delta in RESULTS.md |
| Reward model scores everything ~0.9 | wrong input format (frames vs video file vs resolution) | check the API's expected input schema; re-encode before concluding miscalibration |
| Reliability diagram is jagged nonsense | 100 episodes across 10 bins = ~10/bin | use 5 bins, or bootstrap CIs per bin; report bin counts |
| Three-way comparison shows VLA-JEPA at ~0% | checkpoint/benchmark pairing mismatch | use the checkpoint published *for that benchmark family*; smoke-test 5 episodes before the full run |
| Survey note balloons past 4 pages | recounting papers instead of placing them | every paragraph must either fill a taxonomy cell or advance the position; delete the rest |
| Claims audit reads as a takedown (or a fan letter) | conclusions written before classification | classify first, conclude last; every judgment must cite a specific missing or present artifact |

## Stretch

Close a micro-loop: use your calibrated reward model to filter the worst 20% of a training dataset (TOPReward-style), refine your Lesson 18 fine-tune on the filtered set, and re-evaluate. One number — did learned-reward data curation help at your scale?

Second stretch: if your installed LeRobot exposes any non-language conditioning interface (goal image or demonstration prompt) — verify against the docs for your version, don't assume one exists — run the smallest possible conditioning swap on your Lesson 18 setup and report the delta under the same eval protocol.

## References

- Physical Intelligence: π*0.6/RECAP (Nov 2025), *Precise Manipulation with Efficient Online RL* (Mar 2026, arXiv 2604.23073), π0.7 (Apr 2026) — pi.website/blog.
- LeRobot v0.6.0: release blog + `vla_jepa` docs + reward-models API docs (Robometer, TOPReward).
- Fang et al., *MolmoAct2*, 2026, arXiv 2605.02881 — §data pipeline for TOPReward's gating role.
- Assran et al., V-JEPA2, 2025 — the self-supervised video backbone inside VLA-JEPA.
- Kelly et al., *HG-DAgger*, 2019. arXiv:1810.02890.
- Guo et al., *On Calibration of Modern Neural Networks*, 2017 — ECE, reliability diagrams.
- Skild AI, *S1* blog, 2026 — skild.ai/blogs/s1 (blog only; no paper or weights as of Sep 2026).
- Gu et al., *RT-Trajectory: Robotic Task Generalization via Hindsight Trajectory Sketches*, 2023. arXiv:2311.01977.
- Jain et al., *Vid2Robot: End-to-End Video-Conditioned Policy Learning*, 2024. arXiv:2403.12943.
- Fu et al., *In-Context Imitation Learning via Next-Token Prediction* (ICRT), 2024. arXiv:2408.15980.
