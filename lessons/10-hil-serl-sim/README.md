# Lesson 10 — HIL-SERL in Simulation

Run the tutorial's centerpiece RL system end-to-end in `gym-hil` — reward classifier, decoupled actor/learner, human interventions — and predict each moving part's behavior before observing it, so that when H5 puts this on the real arm nothing in the architecture is new to you.

| | |
|---|---|
| **Phase** | 3 — Reinforcement learning |
| **Time** | ~2 sessions (7–9 h desk, much of it *interactive* — you are the human in the loop) + a few GPU-hours |
| **Cost** | ~$2–4 (learner runs fine on one 4090; actor+env run on the Mac or the same box) |
| **Prerequisites** | 09 (two-buffer machinery; the demos-and-config workflow), 08 (SAC internals for reading learner logs) |
| **Feeds into** | H5 (identical pipeline, physical arm), 20 (reward models are this classifier's descendants) |

## Learning objectives

After this lesson you can:

1. **Calibrate** a binary reward classifier, choose its operating point by a stated rule, and quantify — in wasted robot-steps — what a false positive costs downstream RL.
2. **Draw** LeRobot's decoupled actor/learner architecture from its source: two processes, two queues (transitions up, parameters down), two buffers, one classifier.
3. **Predict** what the actor does when the learner dies, and what the queues do when it returns.
4. **Measure** the effect of human intervention budget on sample efficiency at fixed environment steps.
5. **Derive** why an intervention transition's membership in *both* buffers changes its effective sampling probability, and check the arithmetic against logged buffer sizes.

## Principles

**Why this architecture exists.** Real-robot RL has two constraints sim RL doesn't: the control loop cannot block (a robot mid-motion doesn't wait for a gradient step), and nobody will hand-engineer a reward function over pixels. HIL-SERL (Luo et al. 2024) answers with three components:

1. *Reward classifier.* A binary success detector trained on labeled frames replaces the reward function. It runs in the env loop, so its calibration is not a nicety: every false positive mints fake reward and teaches the policy to exploit the classifier, not solve the task. You build the precision/recall analysis *before* any RL consumes it.
2. *Decoupled actor/learner.* Two processes. The actor rolls the policy in the env at fixed rate and streams transitions to the learner; the learner (RLPD-style SAC from Lesson 09 — demo buffer, 50/50 sampling) trains and streams fresh parameters back. The coupling is asynchronous: the actor acts on slightly stale weights, and the two queues (transition queue, parameter queue) absorb the rate mismatch. Off-policy SAC tolerates the staleness; an on-policy method would not.
3. *Human interventions.* While the actor rolls, a human can seize control (gamepad RB / keyboard toggle in `gym-hil`; `info["is_intervention"]` marks the frames). Intervention segments are on-policy corrections at exactly the states where the policy is wrong — the highest-value data in the system. They enter the demo buffer *and* the online buffer, so under 50/50 sampling an intervened transition is sampled from both streams: its effective probability is $\frac{0.5}{|\mathcal{D}_{\text{demo}}|} + \frac{0.5}{|\mathcal{D}_{\text{online}}|}$ — roughly double-counted early, demo-weighted late. The paper's claimed result of near-perfect policies in 1–2 h of real-robot time rests on this triad.

**The run loop:** record demos → train classifier → launch learner → launch actor → intervene when it flails → watch intervention rate decay as the policy stops needing you.

**Carry forward**

- Classifier precision is the binding constraint: a false positive is a reward the policy will learn to farm.
- Actor and learner are decoupled by two queues; staleness is bounded by the parameter-push rate and tolerated because SAC is off-policy.
- Interventions are on-policy corrections at the policy's own failure states, and they land in both buffers.
- The intervention-rate curve decaying toward zero is the system working; a rising reward with flat true success is the classifier being exploited.

| Source | Read for |
|---|---|
| Tutorial §3.2.1, Codes 3–6 | the reference pseudocode for classifier, actor, learner, orchestration — map each Code to the v0.6 module that replaced it |
| Luo et al. 2024, arXiv:2410.21845 | the claims under test: 1–2 h to high success; intervention-driven sample efficiency |
| LeRobot "Train RL in Simulation" docs | verbatim commands and JSON configs (quoted below); the `lerobot.rl.*` module layout |
| Lesson 09's `RESULTS.md` | your own baseline: pick-cube sample efficiency *without* interventions |

## Exercise 1 — Classifier data and training [Build]

Produces a success detector to calibrate. Coding is LeRobot's; your spec is the labeling.

1. Collect labeled frames with the `gym-hil` recorder: ~20 successful and ~20 failed episodes. Use the env's success/failure labeling controls (gamepad Y = success, A = failure per the gym-hil docs) so labels land in the dataset rather than in your memory. Config (Lesson 09's, retargeted):
   ```json
   { "env": { "type": "gym_manipulator", "name": "gym_hil",
              "task": "PandaPickCubeKeyboard-v0", "fps": 10 },
     "dataset": { "repo_id": "<you>/gymhil_classifier_frames_v1",
                  "task": "pick_cube", "num_episodes_to_record": 40,
                  "push_to_hub": true },
     "mode": "record" }
   ```
2. Train a binary classifier on terminal-window frames (positives: last ~10 frames of successes; negatives: everything else + failure episodes). LeRobot ships a reward-classifier trainer in its RL stack — check the current docs/`lerobot.rl` module listing for the entry point and config; if your installed version lacks one, spec a ~60-line PyTorch script over a frozen ResNet-18 torso with a linear head (what the tutorial's Code 3 sketches) and have an AI tool draft it.
3. Hold out 5 episodes.

**✅ Checkpoint:** a trained classifier and a held-out set with ground-truth labels; per-frame scores saved to CSV.

## Exercise 2 — The operating point [Decide]

Tests objective 1: the threshold is a decision with a stated cost model.

1. From the held-out scores: PR curve and threshold sweep.
2. Build the *false-positive gallery* — the actual frames the classifier calls success at candidate thresholds. Look at them; they are previews of what your policy would learn to exploit.
3. Write the rule (e.g. precision ≥ 0.95 at max recall), the resulting threshold, and one sentence on the cost of each error type in robot-steps: a false negative wastes an episode; a false positive corrupts training. Freeze the threshold in the config.

**✅ Checkpoint:** held-out precision ≥ 0.95 at your threshold; the FP gallery is in `RESULTS.md` with one sentence each on *why* the classifier was fooled.

## Exercise 3 — The system, from source [Read the kernel]

Tests objective 2: you can draw the architecture because you read it, not because the paper has a figure.

1. Pull LeRobot's example `rl/gym_hil/train_config.json` from `lerobot/config_examples`; point it at your demo dataset (Lesson 09's 30 demos), your classifier, and the `PandaPickCubeKeyboard-v0` (or gamepad) task.
2. Read `lerobot.rl.actor` and `lerobot.rl.learner` (installed source). Draw the diagram: processes, the transition queue, the parameter queue, both buffers, where the classifier runs, where `info["is_intervention"]` is consumed. Mark on it what the parameter-push frequency and the UTD setting each control.
3. Launch, two terminals:
   ```bash
   python -m lerobot.rl.learner --config_path configs/train_gym_hil.json
   python -m lerobot.rl.actor   --config_path configs/train_gym_hil.json   # mjpython on Mac
   ```
   Confirm on W&B which of the diagram's quantities the stack already logs (buffer sizes, push events); note which it doesn't.

**✅ Checkpoint:** `plots/system_diagram.png` (hand-drawn is fine) with every box traceable to a module/function name; both processes running and logging.

## Exercise 4 — Kill the learner [Predict → Run]

Tests objective 3: decoupling has observable consequences.

1. **Write first:** when the learner process dies mid-run, does the actor stop, stall, or keep acting? On what weights? What happens to the transition queue while the learner is down, and what happens to both queues in the first seconds after it returns?
2. Run: start both, let it train 5 minutes, kill the learner, wait 2 minutes, restart it. Watch the actor's terminal and W&B.
3. Reconcile in `RESULTS.md`.

**✅ Checkpoint:** the actor's behavior during and after the outage is documented against your prediction; you can state where the transitions from the outage window went.

## Exercise 5 — Train with your own interventions [Predict → Run]

Produces the headline policy and the intervention-decay curve — the lesson's signature figure.

1. **Write first:** sketch the intervention-fraction-per-episode curve you expect over a 150k-step run (early plateau, decay, where it reaches ~0), and the episode-length curve beside it.
2. Full run at your Lesson 09 step budget (150k actor steps or until stable success): intervene by feel — grab control when the policy is clearly failing, release when recovered. Every intervention is logged via `info["is_intervention"]`.
3. Log per episode: success (classifier *and* env ground truth — keep both), intervention fraction of steps, episode length. Stop intervening entirely for the last ~20% of the run.
4. Evaluate: 50 scripted episodes with interventions disabled, ground-truth success.

**✅ Checkpoint:** final policy ≥ 80% success over 50 episodes *with interventions disabled*; the intervention-fraction curve decays from its early plateau to near zero. Save the curve.

## Exercise 6 — Intervention budget [Predict → Run]

Tests objective 4 with the only intervener you have — you — at two deliberate budgets.

1. **Write first:** at a fixed 100k-step budget, how much sooner does a generous intervener (intervene in ≤ 60% of episodes) reach 80% success than a sparse one (≤ 20%)? Name the mechanism: which states enter the buffers that otherwise wouldn't.
2. Two runs, same seed, same budget: sparse then generous (or the reverse — say which, and note that you improve as an intervener between runs). Log the realized intervention fraction per episode so the budgets are auditable. Lesson 09's `rlpd` arm is the zero-intervention reference.
3. Plot success-vs-steps for the three; tabulate steps-to-80%.

**✅ Checkpoint:** generous ≥ sparse ≥ none in steps-to-80%; the realized intervention fractions are reported next to the intended budgets. One seed each: this supports an ordering and a mechanism, not an effect size — say so.

## Exercise 7 — The both-buffers arithmetic [Derive]

Tests objective 5 against real buffer sizes.

1. From the logged buffer sizes of Exercise 5's run, at three points in training (early, mid, late): compute the expected sampling probability of an intervention transition (in both buffers) vs a plain online transition vs a plain demo transition.
2. State at which buffer-size ratio the demo-buffer term stops dominating, and whether your run crossed it.
3. If the stack exposes per-source sampling counts, confirm against them; otherwise derive from buffer sizes alone and say so.

**✅ Checkpoint:** the three-point table is in `RESULTS.md` with the double-counting effect annotated.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| Hub: classifier + final policy checkpoints | classifier card states threshold + held-out PR; policy card links the eval protocol |
| `configs/` | full system relaunchable from committed files |
| `plots/` | PR curve + FP gallery, system diagram, intervention-decay curve, 3-run budget comparison |
| `RESULTS.md` | operating-point rule + cost model; Exercises 4–6 predictions with reconciliations; the both-buffers table; ≤ 12 sentences of interpretation |

## Done when

- [ ] Classifier shipped with measured precision ≥ 0.95 and its FP gallery reviewed.
- [ ] Policy ≥ 80% success (interventions off, 50 episodes) trained with your own interventions.
- [ ] Intervention-decay curve exists and actually decays.
- [ ] The kill-recovery prediction and the budget prediction are each reconciled in writing.
- [ ] You can sketch the full system (processes, queues, buffers, classifier) from memory on paper.

## Self-check

1. Why must the reward classifier's *precision* be prioritized over recall for RL, and what does the policy learn when it isn't?
2. What does parameter staleness cost, concretely, in policy-gradient terms — and why does off-policy SAC tolerate it where PPO wouldn't?
3. Compute the effective sampling probability of an intervention transition when $|\mathcal{D}_{\text{demo}}| = 5\text{k}$ and $|\mathcal{D}_{\text{online}}| = 50\text{k}$. Which buffer dominates, and when does that flip?
4. Interventions are on-policy corrections; demos are off-policy. Why does the *same* two-buffer machinery handle both without modification?
5. Which of the three components (classifier, decoupling, interventions) do you expect to be hardest on the real arm in H5, and why?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Policy success climbs, then collapses to reward-hacking behavior | classifier false positives (e.g. gripper occludes cube near goal) | retrain classifier with the exploit frames as negatives; re-check FP gallery |
| Parameter staleness grows without bound | learner slower than actor (GPU contention or UTD too high) | put learner on its own GPU; drop UTD; check transition queue isn't also backing up |
| Actor crashes on launch, learner fine | env rendering in a headless context | `MUJOCO_GL=egl` on Linux; `mjpython` (not `python`) on the Mac |
| Interventions not landing in the demo buffer | `is_intervention` flag lost in a wrapper | assert on `info["is_intervention"]` at the actor and log per-episode intervention step counts |
| `lerobot.rl` module paths differ from tutorial Codes 3–6 | v0.6 rebuilt the RL stack (`sac` → `gaussian_actor`) | trust `python -m lerobot.rl.<tab>` and the current docs page over the paper's snippets |
| Success rate differs between classifier and env ground truth at eval | threshold drift or distribution shift late in training | always report both; ground truth is the eval metric, classifier is only the training signal |

## Going deeper

- **Queue instrumentation.** Patch your local checkout to log transition-queue depth, parameter-queue depth, and parameter staleness (actor env-steps between weight refreshes); keep the diff in the lesson directory. Staleness should sit at a small steady-state value, not grow.
- **Scripted oracle.** Replace yourself with a scripted intervener (seize control when episode step > N with cube ungrasped, or EE leaves a workspace box) and rerun Exercise 6 at 3 budgets × 2 seeds — the reproducible version of the human-feel result.
- **Classifier robustness loop.** Add your trained policy's near-success failures to the classifier's negative set, retrain, and measure whether the reward-hacking margin drops — a miniature of the reward-model iteration loop that reappears in Lesson 20.

## References

- Luo, Xu, Wu, Levine. *Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning* (HIL-SERL), arXiv:2410.21845.
- Luo et al., *SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning*, arXiv:2401.16013 — the engineering lineage.
- LeRobot team, *Robot Learning: A Tutorial*, §3.2.1 Codes 3–6. arXiv:2510.12403.
- LeRobot "Train RL in Simulation" docs (`lerobot.rl.actor` / `lerobot.rl.learner` commands + config examples at `huggingface.co/datasets/lerobot/config_examples`).
