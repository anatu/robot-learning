# Lesson 20 — Beyond Imitation: RL-from-Experience & World Models

Imitation plateaus at the quality of its demonstrations. Map the field's answers as of mid-2026 — advantage conditioning, RL tokens, distilled specialists, world-model policies, reward models — then run two of them yourself and measure whether a learned reward model can be trusted as an evaluator.

| | |
|---|---|
| **Phase** | 6 — Frontier |
| **Time** | ~2 sessions for the survey note + ~1 session hands-on (plus one cloud eval run) |
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

| Source | Read for |
|---|---|
| PI: π*0.6/RECAP post (pi.website, Nov 2025) | the three data streams and what the value function is actually fit on |
| PI: *Precise Manipulation with Efficient Online RL* (pi.website/research/rlt) + arXiv 2604.23073 | what an "RL token" exposes that raw actions don't |
| PI: π0.7 post (pi.website/blog/pi07) | what metadata makes distilled skills *steerable* |
| LeRobot v0.6.0 release blog + `vla_jepa` docs page | the concrete training/eval interfaces you'll run in Part 2 |
| ETH Robot Learning lecture 8 (World Models, YouTube) | the latent-vs-pixel prediction trade-off, for the note's world-model section |
| Levine's "imitation vs RL" framing (CS 285 lecture 2) | the compounding-error argument stated precisely, for ceiling (1) |

## Part 1 — The survey note (~4–6 h)

Produces `NOTE.md`, 3–4 pages. This is a *position piece with a taxonomy*, not an annotated bibliography.

1. Build the taxonomy table first: one row per method family, columns = signal source, data cost per unit of improvement, where the improvement shows up (robustness vs peak skill vs throughput), sharpest open failure mode. Verify every post-cutoff row against its primary source as you fill it; unverifiable cells get "?".
2. Add a half-page timeline figure, Oct 2025 → Aug 2026 (tutorial ships → π*0.6 → RLT → π0.7 → MolmoAct2 → LeRobot v0.6), annotated with which taxonomy family each release advances.
3. Write the position: answer "what closes the loop after BC for a $500-robot lab?" — commit to an ordering of the families by expected value *at your scale*, and defend it. (Advantage conditioning needs a value function; world-model supervision needs video compute; corrections need only you and a leader arm. Scale changes the answer — say how.)

**✅ Checkpoint:** the table has ≤ 4 "?" cells; the position section names a first, second, and third choice for your own hardware track, with reasons.

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

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `NOTE.md` | taxonomy table + timeline + a defended position; post-cutoff cells cited or "?" |
| `wm_eval/` | scripts + JSON for the three-way comparison; rerunnable from one command |
| `reward_calibration/` | scoring script, per-episode CSV, reliability diagram, threshold sweep, failure gallery |
| `RESULTS.md` | the three-way table; the inference-graph finding; the three-tier reward-model verdict |

## Done when

- [ ] The three-way (SmolVLA-ft / VLA-JEPA / zero-shot) same-protocol table exists.
- [ ] The "free at inference" claim is verified or refuted from the actual inference graph + latency.
- [ ] Reward-model calibration covers ≥ 100 episodes with reliability diagram, ECE, and threshold sweep.
- [ ] The note's position section survives this test: a skeptical reader can find your reasons, your evidence, and your uncertainty, each labeled as such.

## Self-check

1. RECAP and DAgger both learn from mistakes. What's the signal-source difference, and which scales past human patience?
2. Why can VLA-JEPA delete its world model at inference while FastWAM keeps its video expert? What different bet is each making?
3. TOPReward gated MolmoAct2's training corpus. Why does *data filtering* tolerate a worse-calibrated reward model than *RL training* does?
4. π0.7 distills RL specialists with "strategy metadata". What failure of naive distillation is the metadata preventing?
5. Your reward model shows ECE of 0.15 with high recall at low precision. Which of the three deployment tiers (filter/eval/RL-reward) is it fit for, if any?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `vla_jepa` entrypoints don't match this README | v0.6.x API drift | the docs page for your installed version wins; record the delta in RESULTS.md |
| Reward model scores everything ~0.9 | wrong input format (frames vs video file vs resolution) | check the API's expected input schema; re-encode before concluding miscalibration |
| Reliability diagram is jagged nonsense | 100 episodes across 10 bins = ~10/bin | use 5 bins, or bootstrap CIs per bin; report bin counts |
| Three-way comparison shows VLA-JEPA at ~0% | checkpoint/benchmark pairing mismatch | use the checkpoint published *for that benchmark family*; smoke-test 5 episodes before the full run |
| Survey note balloons past 4 pages | recounting papers instead of placing them | every paragraph must either fill a taxonomy cell or advance the position; delete the rest |

## Stretch

Close a micro-loop: use your calibrated reward model to filter the worst 20% of a training dataset (TOPReward-style), refine your Lesson 18 fine-tune on the filtered set, and re-evaluate. One number — did learned-reward data curation help at your scale?

## References

- Physical Intelligence: π*0.6/RECAP (Nov 2025), *Precise Manipulation with Efficient Online RL* (Mar 2026, arXiv 2604.23073), π0.7 (Apr 2026) — pi.website/blog.
- LeRobot v0.6.0: release blog + `vla_jepa` docs + reward-models API docs (Robometer, TOPReward).
- Fang et al., *MolmoAct2*, 2026, arXiv 2605.02881 — §data pipeline for TOPReward's gating role.
- Assran et al., V-JEPA2, 2025 — the self-supervised video backbone inside VLA-JEPA.
- Kelly et al., *HG-DAgger*, 2019. arXiv:1810.02890.
- Guo et al., *On Calibration of Modern Neural Networks*, 2017 — ECE, reliability diagrams.
