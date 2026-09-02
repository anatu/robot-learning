# Lesson 20 — Beyond Imitation: RL-from-Experience & World Models

Imitation plateaus at the quality of its demonstrations. Map the field's answers as of mid-2026 by *signal source*, run a world-model policy and a reward model from LeRobot v0.6 against numbers you already own, measure whether the reward model can be trusted as an evaluator, and audit one frontier announcement (Skild S1) as an exercise in reading claims without weights.

| | |
|---|---|
| **Phase** | 6 — Frontier |
| **Time** | ~1.5 sessions for the survey note + ~1 session hands-on (one cloud eval run) + ~2 h for the S1 claims audit |
| **Cost** | ~$3–6 cloud GPU for the world-model-policy eval; reward-model inference is Mac-feasible |
| **Prerequisites** | 10 (you've calibrated a reward classifier), 17–18 (VLA mechanics + a fine-tuned policy whose rollouts you'll reuse), 09 (offline-data RL intuitions) |
| **Feeds into** | 21 (verification is the same problem at the planner level), 22 (capstone option 1 is this lesson's loop on hardware), H4 (the DAgger iteration) |

## Learning objectives

After this lesson you can:

1. **Taxonomize** post-imitation methods by signal source — who or what says "this was better" — and place a new paper into it on first read.
2. **Explain** RECAP's advantage conditioning, RL-token bootstrapping, and π0.7's distillation as three answers to "how does experience improve a generalist", and why VLA-JEPA's world model costs nothing at inference.
3. **Predict** where a world-model policy lands against your fine-tuned SmolVLA, then verify the "free at inference" claim from the actual inference graph.
4. **Quantify** a reward model's trustworthiness — precision/recall, calibration, ECE — and **decide** the threshold at which it may replace ground-truth evaluation.
5. **Audit** a frontier announcement: separate verified, verifiable-but-unpublished, and unverifiable claims, and commit to a dated, falsifiable expectation.

## Principles

**Why BC plateaus.** Three independent ceilings: (1) compounding error — the policy visits states demos never covered and has no signal there; (2) the demonstrator ceiling — you cannot exceed the teleoperator's skill by copying them; (3) no notion of *better* — BC's loss is indifferent between a barely-successful and a crisp trajectory. Everything below injects a preference signal that demos alone lack.

**The taxonomy, by signal source** (the survey note's spine):

| Family | Signal source | Exemplar |
|---|---|---|
| Corrections | human takes over when the policy errs; the takeover states are exactly the compounding-error states | HG-DAgger; `lerobot-rollout`'s intervention mode (H4) |
| Advantage conditioning | a value function scores *all* experience (demos, corrections, autonomous rollouts); the policy learns success- and failure-conditioned behavior and is steered to the good side at inference | **RECAP / π*0.6** (PI, Nov 2025): condition on advantage bins, deploy conditioned on high advantage — "RL without ever writing a policy-gradient step" |
| Online RL via VLA priors | classic online RL made sample-feasible by bootstrapping from a VLA's learned representations | **RL Tokens** (PI, Mar 2026, arXiv 2604.23073): extract an RL-amenable token interface from the VLA; precision tasks (an M3 screw insertion) mastered in ~15 minutes of experience |
| Distillation back to generalist | RL-trained specialists + strategy metadata distilled into one steerable model | **π0.7** (Apr 2026): matches or beats the specialists on espresso/box-folding/laundry throughput while staying a single generalist |
| World-model supervision | the future itself: predict consequences of your own actions | **VLA-JEPA** (LeRobot v0.6): Qwen3-VL backbone + V-JEPA2 video world model + flow-matching DiT head — the world model shapes representations *during training* and is deleted at inference; **FastWAM**: ~5B video-generation expert paired with a compact action expert, asking whether test-time imagination is needed at all; **LingBot-VA**: autoregressive video+action prediction, chunk by chunk |
| Learned evaluators | a reward model watches rollouts and scores success | **Robometer / TOPReward** (LeRobot v0.6 reward-models API); TOPReward is the quality gate that filtered MolmoAct2's 38k-episode community corpus |

Two through-lines: *the signal keeps getting cheaper* (human demo → human correction → self-experience → predicted futures), and *evaluation is becoming a model too* — which is why the calibration study is not a side quest. A reward model you can't calibrate is a benchmark you can't trust.

**The second axis — task specification.** The taxonomy sorts methods by who grades experience. An orthogonal axis is how intent enters the policy at inference: a language instruction (every VLA in Lessons 17–19), a goal image, a trajectory sketch (**RT-Trajectory**), or a video demonstration consumed in context with no weight update (**Vid2Robot**, **ICRT**, and — at the largest scale — **Skild S1**, 2026). S1's bet: pre-train on episodic data where the task is specified *only* by an in-context demo, so the demo is read as intent, not as a trajectory to copy. Its blog claims 96% in-distribution success, a widening OOD gap over language-conditioned VLAs with scale (66% vs 9% at ~100k pre-training hours), and one demo worth ~380 post-training episodes. Closed weights, no paper, no independent eval — which is exactly why it is Exercise 5's audit subject rather than something you run.

**Carry forward**

- Sort a post-imitation method by *who grades the experience*; that column predicts its data cost and where its gains show up.
- Signal source and task specification are orthogonal axes; place a method on both.
- A world model can pay off entirely at training time (VLA-JEPA) or be kept at test time (FastWAM); the choice is a bet about whether imagination is needed to act.
- A learned evaluator is usable at three bars — data filter, eval replacement, RL reward — and they are ordered by the precision they demand.
- A frontier claim without weights, paper, or baseline spec is a hypothesis with a date; write down what would change your mind.

| Source | Read for |
|---|---|
| PI: π*0.6/RECAP post (pi.website, Nov 2025) | the three data streams and what the value function is actually fit on |
| PI: *Precise Manipulation with Efficient Online RL* (pi.website/research/rlt) + arXiv 2604.23073 | what an "RL token" exposes that raw actions don't |
| PI: π0.7 post (pi.website/blog/pi07) | what metadata makes distilled skills *steerable* |
| LeRobot v0.6.0 release blog + `vla_jepa` docs page | the concrete training/eval interfaces you'll run in Exercise 2 |
| ETH Robot Learning lecture 8 (World Models, YouTube) | the latent-vs-pixel prediction trade-off, for the note's world-model section |
| Levine's "imitation vs RL" framing (CS 285 lecture 2) | the compounding-error argument stated precisely, for ceiling (1) |
| Skild S1 blog (skild.ai/blogs/s1, 2026) | the ICL-vs-VLA scaling claim you'll audit — read it twice: once credulous, once adversarial |
| RT-Trajectory (arXiv 2311.01977) · Vid2Robot (arXiv 2403.12943) · ICRT (arXiv 2408.15980) | the open, small-scale end of the task-specification spectrum — skim for the conditioning interface, not the results |

## Exercise 1 — The survey note [Write]

Tests objectives 1–2. `NOTE.md`, 2–3 pages: a *position piece with a taxonomy*, not an annotated bibliography.

1. Taxonomy table first: one row per family, columns = signal source, data cost per unit of improvement, where the improvement shows up (robustness vs peak skill vs throughput), sharpest open failure mode. Verify every post-cutoff row against its primary source as you fill it; unverifiable cells get "?".
2. A half-page timeline, Oct 2025 → Aug 2026 (tutorial ships → π*0.6 → RLT → π0.7 → MolmoAct2 → LeRobot v0.6), annotated with which family each release advances.
3. The position: answer "what closes the loop after BC for a $500-robot lab?" — commit to an ordering of the families by expected value *at your scale* and defend it. (Advantage conditioning needs a value function; world-model supervision needs video compute; corrections need only you and a leader arm. Scale changes the answer — say how.)
4. A half-page task-specification section: the language → goal image → sketch → in-context demo spectrum, one sentence per point on data cost vs test-time flexibility, and a committed answer to "what changes for a $500-robot lab if conditioning shifts demo-ward?" (Your leader arm makes demos nearly free; language annotation isn't. Say what that does to your Phase-1 data practices.)

**✅ Checkpoint:** ≤ 4 "?" cells; the position names a first, second, and third choice for your own hardware track, with reasons; the spec-spectrum section commits to an answer.

## Exercise 2 — Run a world-model policy [Predict → Run]

Tests objective 3: the v0.6 claim that world-model supervision is free at inference.

1. **Write first:** where you expect the published VLA-JEPA baseline to land on your Lesson 18 subset relative to SmolVLA-ft and zero-shot, and why; and what you expect to find in the inference graph (is the V-JEPA2 branch present?) plus the ms/chunk ratio vs SmolVLA.
2. Read the `vla_jepa` docs page in your installed LeRobot; note the training/eval entrypoints and the published baseline checkpoint (one per benchmark family, CI-smoke-tested — start from it, do not train from scratch).
3. Evaluate the published baseline on the same benchmark subset + seeds you used in Lesson 18. That gives a three-way, same-protocol comparison: SmolVLA-ft vs VLA-JEPA vs zero-shot.
4. Inspect the checkpoint or policy class and record what runs in the inference graph; measure ms/chunk vs SmolVLA on the same hardware.
5. (Alternative, budget permitting: fine-tune VLA-JEPA on your Lesson 18 dataset per its docs and report the same table. Either path satisfies the lesson.)
6. Reconcile against step 1.

**✅ Checkpoint:** the three-way table exists; the inference-graph note states what actually runs at test time, with latency numbers beside it; the prediction is reconciled.

## Exercise 3 — Reward-model calibration [Predict → Run]

Tests objective 4. You already own the test set: Lesson 14/18 eval rollouts, each with a ground-truth success label from the environment.

1. Assemble ≥ 60 episodes (videos + GT labels), balanced across success/failure if possible; note the class ratio.
2. **Write first:** the precision and recall you expect at the default threshold, and whether you expect the model to be over- or under-confident (reliability curve above or below the diagonal).
3. Spec `reward_calibration/score_episodes.py` for an AI tool: run one v0.6 reward model (Robometer or TOPReward, per the reward-models API docs — check which input format each expects, video vs frame sequence) over every episode; write a per-episode CSV of score + GT label. The check: the CSV has one row per episode and scores are in [0, 1].
4. Analyze: confusion matrix at the default threshold; precision/recall/F1; a reliability diagram (5 bins at this N, with bin counts) and ECE; the full threshold sweep (PR curve).
5. Failure gallery: the 5 highest-confidence false positives and false negatives, one frame each, one sentence on what fooled the model.
6. Reconcile against step 2.

**✅ Checkpoint:** reliability diagram + ECE computed; PR sweep plotted; gallery written; prediction reconciled.

## Exercise 4 — The three-tier verdict [Decide]

Tests objective 4's decision. From Exercise 3's curve: at what precision would you accept the reward model as (a) a data-filtering gate (TOPReward's actual job in MolmoAct2's pipeline), (b) a replacement for ground-truth eval, (c) an RL reward? These bars differ — say why (what does each error type cost in each role?), and state where your measured curve clears them.

**✅ Checkpoint:** three bars, each with a reason and a verdict from your numbers.

## Exercise 5 — The S1 claims audit [Write]

Tests objective 5. Reading a frontier announcement without published weights, paper, or baselines is a skill. `s1_claims_audit.md`:

1. Extract every quantifiable claim into a table: claim, number, what it was measured on (as stated), what's missing to reproduce it. Minimum set: 96% ID success; 66% vs 9% OOD at ~100k hours; 1 demo ≈ 380 post-training episodes ≈ 50–100 h of teleop; the L1–L5 perturbation ladder with the language baseline degrading ~3× more; "the same model weights produced every example"; 10-minute unseen long-horizon tasks.
2. Classify each claim: independently verifiable today / verifiable in principle but unpublished / unverifiable as stated. For every claim in the last two bins, name the single artifact (baseline spec, task list, trial counts, CIs, seed policy) whose release would move it up a bin.
3. Case-study the 66%-vs-9% figure: list ≥ 3 benign explanations that would shrink the gap without the headline being false (under-tuned language baseline, task distribution chosen demo-side, eval-protocol asymmetry, ...), and for each, the published evidence that would rule it out.
4. Write one dated, falsifiable expectation: if the ICL-scaling claim is real, what should a peer-reviewed or open replication show within 12 months? Revisit at capstone time.

**✅ Checkpoint:** every extracted claim is classified; the case study has ≥ 3 alternatives each paired with discriminating evidence; the expectation is dated and falsifiable.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `NOTE.md` | taxonomy table + timeline + task-specification section + a defended position; post-cutoff cells cited or "?" |
| `wm_eval/run_three_way_eval.py` + JSON | the three-way comparison, rerunnable from one command |
| `reward_calibration/` | scoring script, per-episode CSV, reliability diagram, PR sweep, failure gallery |
| `s1_claims_audit.md` | every claim classified; ≥ 3 benign alternatives for the headline figure; one dated falsifiable expectation |
| `RESULTS.md` | Exercise 2/3 predictions with reconciliations; the three-way table; the inference-graph finding; the three-tier verdict |

## Done when

- [ ] The three-way (SmolVLA-ft / VLA-JEPA / zero-shot) same-protocol table exists.
- [ ] The "free at inference" claim is verified or refuted from the actual inference graph + latency.
- [ ] Reward-model calibration covers ≥ 60 episodes with reliability diagram, ECE, and PR sweep.
- [ ] The three-tier verdict is stated with numbers.
- [ ] The note's position survives this test: a skeptical reader can find your reasons, your evidence, and your uncertainty, each labeled.
- [ ] The S1 audit classifies every claim and commits to a dated, falsifiable expectation.

## Self-check

1. RECAP and DAgger both learn from mistakes. What's the signal-source difference, and which scales past human patience?
2. Why can VLA-JEPA delete its world model at inference while FastWAM keeps its video expert? What different bet is each making?
3. TOPReward gated MolmoAct2's training corpus. Why does *data filtering* tolerate a worse-calibrated reward model than *RL training* does?
4. π0.7 distills RL specialists with "strategy metadata". What failure of naive distillation is the metadata preventing?
5. Your reward model shows ECE of 0.15 with high recall at low precision. Which of the three tiers is it fit for, if any?
6. Signal source and task specification are orthogonal axes. Place S1 on both, and name one method from your taxonomy that shares its position on one axis but not the other.
7. S1 reads a demo as intent, not a trajectory to replay. Which failure mode of plain BC does that reframing attack, and which of the three BC ceilings does it leave untouched?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `vla_jepa` entrypoints don't match this README | v0.6.x API drift | the docs page for your installed version wins; record the delta in `RESULTS.md` |
| Reward model scores everything ~0.9 | wrong input format (frames vs video file vs resolution) | check the API's expected input schema; re-encode before concluding miscalibration |
| Reliability diagram is jagged nonsense | 60 episodes across 10 bins = ~6/bin | 5 bins, or bootstrap CIs per bin; report bin counts |
| Three-way comparison shows VLA-JEPA at ~0% | checkpoint/benchmark pairing mismatch | use the checkpoint published *for that benchmark family*; smoke-test 5 episodes before the full run |
| Survey note balloons past 3 pages | recounting papers instead of placing them | every paragraph must fill a taxonomy cell or advance the position; delete the rest |
| Claims audit reads as a takedown (or a fan letter) | conclusions written before classification | classify first, conclude last; every judgment cites a specific missing or present artifact |

## Going deeper

- **Close a micro-loop.** Use your calibrated reward model to filter the worst 20% of a training dataset (TOPReward-style), refine your Lesson 18 fine-tune on the filtered set, and re-evaluate. One number: did learned-reward data curation help at your scale?
- **Conditioning swap.** If your installed LeRobot exposes any non-language conditioning interface (goal image or demonstration prompt — verify against the docs for your version, don't assume), run the smallest possible swap on your Lesson 18 setup and report the delta under the same eval protocol.
- **Full calibration set.** Push to ≥ 100 episodes and 10 bins; compare ECE stability against the 60-episode run.

## References

- Physical Intelligence: π*0.6/RECAP (Nov 2025), *Precise Manipulation with Efficient Online RL* (Mar 2026, arXiv 2604.23073), π0.7 (Apr 2026) — pi.website/blog.
- LeRobot v0.6.0: release blog + `vla_jepa` docs + reward-models API docs (Robometer, TOPReward).
- Fang et al., *MolmoAct2*, 2026, arXiv 2605.02881 — §data pipeline for TOPReward's gating role.
- Assran et al., V-JEPA2, 2025 — the self-supervised video backbone inside VLA-JEPA.
- Kelly et al., *HG-DAgger*, 2019. arXiv:1810.02890.
- Guo et al., *On Calibration of Modern Neural Networks*, 2017 — ECE, reliability diagrams.
- Skild AI, *S1* blog, 2026 — skild.ai/blogs/s1 (blog only; no paper or weights as of Sep 2026).
- Gu et al., *RT-Trajectory*, 2023. arXiv:2311.01977.
- Jain et al., *Vid2Robot*, 2024. arXiv:2403.12943.
- Fu et al., *In-Context Imitation Learning via Next-Token Prediction* (ICRT), 2024. arXiv:2408.15980.
