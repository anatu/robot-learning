# Lesson 10 — HIL-SERL in Simulation

This lesson runs the tutorial's centrepiece reinforcement-learning system, HIL-SERL, end to end in the `gym-hil` simulator. The system has three parts, a learned reward classifier, a decoupled actor and learner, and a human who intervenes during training, and you will build or configure each of them, predict how it behaves, and then observe it. The purpose is that when H5 puts the same system on the physical arm, nothing about the architecture is new; the only new variable will be reality.

| | |
|---|---|
| **Phase** | 3 — Reinforcement learning |
| **Time** | ~2 sessions (7–9 h desk time, much of it interactive, because you are the human in the loop) + a few GPU-hours |
| **Cost** | ~$2–4 (the learner runs on one 4090; the actor and environment run on the Mac or the same box) |
| **Prerequisites** | 09 (the two-buffer machinery and the demonstration-and-config workflow), 08 (SAC internals, for reading the learner's logs) |
| **Feeds into** | H5 (the identical pipeline on the physical arm), 20 (reward models are the descendants of this lesson's classifier) |

## Learning objectives

After this lesson you can:

1. **Calibrate** a binary reward classifier, choose its operating point by a stated rule, and quantify, in wasted robot steps, what a false positive costs the downstream reinforcement learning.
2. **Draw** LeRobot's decoupled actor and learner architecture from its source: two processes, two queues (transitions up, parameters down), two buffers, and one classifier.
3. **Predict** what the actor does when the learner dies, and what the queues do when it returns.
4. **Measure** the effect of the human intervention budget on sample efficiency at a fixed number of environment steps.
5. **Derive** why an intervention transition's membership in both buffers changes its effective sampling probability, and check the arithmetic against logged buffer sizes.

## Principles

### Why this architecture exists

Reinforcement learning on a real robot faces two constraints that simulation does not. The control loop cannot block, because a robot in mid-motion does not wait for a gradient step to finish; and nobody will hand-engineer a reward function over camera pixels. HIL-SERL (Luo et al. 2024) answers the two constraints with three components, and the paper's claim that near-perfect policies can be trained in one to two hours of real-robot time rests on all three working together.

### The reward classifier

A binary success detector, trained on labelled frames, replaces the reward function. It runs inside the environment loop, which means its calibration is not a matter of tidiness: every false positive mints a reward the task did not earn, and a policy trained on such rewards learns to reproduce whatever fooled the classifier rather than to solve the task. For that reason the precision-recall analysis is done before any reinforcement learning consumes the classifier's output, and the operating threshold is chosen for precision first.

### The decoupled actor and learner

The system runs as two processes. The actor rolls out the current policy in the environment at a fixed control rate and streams transitions to the learner. The learner, which is the RLPD-style SAC of Lesson 09 with a demonstration buffer and symmetric sampling, trains on those transitions and streams fresh parameters back. The two are coupled asynchronously through two queues, one carrying transitions upward and one carrying parameters downward, and those queues absorb the mismatch between the actor's rate and the learner's. The actor therefore acts on parameters that are slightly stale. An off-policy method such as SAC tolerates that staleness, because it never assumed the data came from the current policy; an on-policy method would not.

### Human interventions and the two buffers

While the actor is rolling out, a human can take control at any time (with the gamepad's RB button or the keyboard toggle in `gym-hil`), and the environment marks the affected frames with `info["is_intervention"]`. An intervention segment is a correction demonstrated at exactly the state where the policy was going wrong, which makes it the most informative data the system receives. Intervention transitions enter both the demonstration buffer and the online buffer. Under symmetric 50/50 sampling a transition present in both is sampled from both streams, so its effective per-batch probability is $\frac{0.5}{|\mathcal{D}_{\text{demo}}|} + \frac{0.5}{|\mathcal{D}_{\text{online}}|}$. Early in training, when the two buffers are of similar size, this roughly doubles its weight; later, when the online buffer is much larger, the demonstration-buffer term dominates. Exercise 7 computes this from real buffer sizes.

### The run loop

The sequence of operations is fixed: record demonstrations, train the classifier, launch the learner, launch the actor, intervene when the policy flails, and watch the intervention rate decay as the policy stops needing you. That decay is the observable that tells you the system is working.

**Carry forward**

- The classifier's precision is the binding constraint on the whole system, because a false positive is a reward the policy will learn to farm.
- The actor and learner are decoupled by two queues; the staleness of the actor's parameters is bounded by the parameter-push rate and is tolerated because SAC is off-policy.
- Interventions are on-policy corrections at the policy's own failure states, and they enter both buffers, which raises their sampling weight under symmetric sampling.
- An intervention-rate curve that decays toward zero is the sign of a working system, whereas a rising reward alongside a flat true success rate is the sign of a classifier being exploited.

| Source | Read for |
|---|---|
| Tutorial §3.2.1, Codes 3–6 | the reference pseudocode for the classifier, actor, learner and orchestration; map each Code to the v0.6 module that replaced it |
| Luo et al. 2024, arXiv:2410.21845 | the claims under test: one to two hours to high success, and intervention-driven sample efficiency |
| LeRobot "Train RL in Simulation" docs | the verbatim commands and JSON configs quoted below, and the `lerobot.rl.*` module layout |
| Lesson 09's `RESULTS.md` | your own baseline: pick-cube sample efficiency without interventions |

## Exercise 1 — Collect labelled frames and train the classifier [Build]

In this exercise you produce the success detector that Exercise 2 will calibrate. The training code is LeRobot's; your contribution is the labelling, which determines what the classifier can learn.

1. Collect labelled frames with the `gym-hil` recorder: roughly 20 successful and 20 failed episodes. Use the environment's labelling controls (gamepad Y for success and A for failure, per the gym-hil docs) so that the labels are stored in the dataset rather than in your memory. The config is Lesson 09's, retargeted:
   ```json
   { "env": { "type": "gym_manipulator", "name": "gym_hil",
              "task": "PandaPickCubeKeyboard-v0", "fps": 10 },
     "dataset": { "repo_id": "<you>/gymhil_classifier_frames_v1",
                  "task": "pick_cube", "num_episodes_to_record": 40,
                  "push_to_hub": true },
     "mode": "record" }
   ```
2. Train a binary classifier on terminal-window frames, with the last ten or so frames of each successful episode as positives and everything else, including the failed episodes, as negatives. LeRobot ships a reward-classifier trainer in its RL stack; check the current docs and the `lerobot.rl` module listing for the entry point and config. If your installed version lacks one, specify a PyTorch script of about sixty lines with a frozen ResNet-18 torso and a linear head, which is what the tutorial's Code 3 sketches, and have an AI tool draft it.
3. Hold out five episodes for evaluation.

**✅ Checkpoint:** a trained classifier, a held-out set with ground-truth labels, and per-frame scores saved to a CSV file.

## Exercise 2 — Choose the operating threshold [Decide]

This exercise tests objective 1. The threshold is a decision with a cost model behind it, and the exercise asks you to state the model and then choose accordingly.

1. From the held-out scores, plot the precision-recall curve and sweep the threshold.
2. Build the false-positive gallery: the actual frames that the classifier calls successes at each candidate threshold. Look at them carefully, because they are previews of what the policy would learn to exploit.
3. Write the rule you are applying (for example, precision of at least 0.95 at the maximum recall that permits), the resulting threshold, and one sentence on the cost of each error type in robot steps: a false negative wastes an episode, whereas a false positive corrupts training. Freeze the threshold in the config.

**✅ Checkpoint:** held-out precision of at least 0.95 at your threshold, and the false-positive gallery in `RESULTS.md` with one sentence per frame on why the classifier was fooled.

## Exercise 3 — Draw the system from its source [Read the kernel]

This exercise tests objective 2. You will be able to draw the architecture because you have read the code that implements it, rather than because the paper contains a figure.

1. Pull LeRobot's example `rl/gym_hil/train_config.json` from `lerobot/config_examples` and point it at your demonstration dataset (Lesson 09's 30 episodes), your classifier, and the `PandaPickCubeKeyboard-v0` task or its gamepad variant.
2. Read `lerobot.rl.actor` and `lerobot.rl.learner` in the installed source. Draw the diagram: the two processes, the transition queue, the parameter queue, both buffers, where the classifier runs, and where `info["is_intervention"]` is consumed. Mark on the diagram what the parameter-push frequency and the UTD setting each control.
3. Launch the two processes in separate terminals:
   ```bash
   python -m lerobot.rl.learner --config_path configs/train_gym_hil.json
   python -m lerobot.rl.actor   --config_path configs/train_gym_hil.json   # mjpython on Mac
   ```
   Confirm on W&B which of the quantities in your diagram the stack already logs (buffer sizes, push events) and note which it does not.

**✅ Checkpoint:** `plots/system_diagram.png` (a hand drawing is fine) with every box traceable to a module or function name, and both processes running and logging.

## Exercise 4 — Stop the learner mid-run [Predict → Run]

This exercise tests objective 3. Decoupling the two processes has consequences that are easy to state and worth seeing, and the simplest way to see them is to remove one process while the other is running.

1. Before running, write down what happens when the learner process dies mid-run: does the actor stop, stall, or keep acting, and on what parameters? What happens to the transition queue while the learner is down, and what happens to both queues in the first seconds after it returns?
2. Start both processes, let them train for five minutes, kill the learner, wait two minutes, and restart it. Watch the actor's terminal and W&B throughout.
3. Reconcile your prediction with what you observed in `RESULTS.md`.

**✅ Checkpoint:** the actor's behaviour during and after the outage is documented against your prediction, and you can state where the transitions generated during the outage went.

## Exercise 5 — Train with your own interventions [Predict → Run]

This exercise produces the lesson's main result, a policy trained with human interventions, and the curve that shows the interventions becoming unnecessary. Sketching that curve first makes it a prediction rather than a picture.

1. Before running, sketch the intervention fraction per episode that you expect over a 150k-step run: an early plateau, a decay, and the point at which it reaches roughly zero. Sketch the episode-length curve beside it.
2. Run at your Lesson 09 step budget (150k actor steps, or until success is stable). Intervene by judgement: take control when the policy is clearly failing and release it when the situation is recovered. Every intervention is logged through `info["is_intervention"]`.
3. Log per episode the success according to the classifier and according to the environment's ground truth (keep both), the fraction of steps that were interventions, and the episode length. Stop intervening entirely for the last twenty percent of the run.
4. Evaluate with 50 scripted episodes, interventions disabled, scored by ground truth.

**✅ Checkpoint:** the final policy succeeds on at least 80 percent of 50 episodes with interventions disabled, and the intervention-fraction curve decays from its early plateau to near zero. Save the curve.

## Exercise 6 — Compare two intervention budgets [Predict → Run]

This exercise tests objective 4 with the only intervener available, which is you, at two deliberately different budgets. The result supports an ordering and a mechanism rather than an effect size, and the writeup should say so.

1. Before running, write down how much sooner you expect a generous intervener (intervening in at most 60 percent of episodes) to reach 80 percent success than a sparse one (at most 20 percent), at a fixed budget of 100k steps. Name the mechanism: which states enter the buffers under the generous budget that would not otherwise?
2. Run twice with the same seed and the same budget, sparse and then generous or the reverse; say which order you used, and note that you will have improved as an intervener between the two runs. Log the realized intervention fraction per episode so that the budgets are auditable. Lesson 09's `rlpd` arm is the zero-intervention reference.
3. Plot success against steps for all three and tabulate steps-to-80%.

**✅ Checkpoint:** generous ≥ sparse ≥ none in steps-to-80%, with the realized intervention fractions reported next to the intended budgets.

## Exercise 7 — Compute the sampling probabilities from logged buffer sizes [Derive]

This exercise tests objective 5 against the real buffer sizes from your training run rather than against hypothetical ones.

1. From the logged buffer sizes of Exercise 5's run, at three points in training (early, middle, late), compute the expected per-batch sampling probability of an intervention transition (which is in both buffers), of a plain online transition, and of a plain demonstration transition.
2. State the buffer-size ratio at which the demonstration-buffer term stops dominating, and whether your run crossed it.
3. If the stack exposes per-source sampling counts, confirm your numbers against them; otherwise derive them from the buffer sizes alone and say so.

**✅ Checkpoint:** the three-point table is in `RESULTS.md` with the double-counting effect annotated.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| Hub: the classifier and the final policy checkpoints | the classifier card states the threshold and held-out precision and recall; the policy card links the evaluation protocol |
| `configs/` | the full system can be relaunched from the committed files |
| `plots/` | the precision-recall curve and false-positive gallery, the system diagram, the intervention-decay curve, the three-run budget comparison |
| `RESULTS.md` | the operating-point rule and cost model; the predictions for Exercises 4–6 with their reconciliations; the both-buffers table; at most 12 sentences of interpretation |

## Done when

- [ ] The classifier ships with a measured precision of at least 0.95 and its false-positive gallery has been reviewed.
- [ ] The policy reaches at least 80 percent success (interventions off, 50 episodes), trained with your own interventions.
- [ ] The intervention-decay curve exists and decays.
- [ ] The learner-outage prediction and the budget prediction are each reconciled in writing.
- [ ] You can sketch the full system (processes, queues, buffers, classifier) from memory.

## Self-check

1. Why must the reward classifier's precision be prioritized over its recall for reinforcement learning, and what does the policy learn when it is not?
2. What does parameter staleness cost, concretely, in policy-gradient terms, and why does off-policy SAC tolerate it where PPO would not?
3. Compute the effective sampling probability of an intervention transition when $|\mathcal{D}_{\text{demo}}| = 5\text{k}$ and $|\mathcal{D}_{\text{online}}| = 50\text{k}$. Which buffer dominates, and when does that change?
4. Interventions are on-policy corrections and demonstrations are off-policy. Why does the same two-buffer machinery handle both without modification?
5. Which of the three components (classifier, decoupling, interventions) do you expect to be hardest on the real arm in H5, and why?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Policy success climbs, then collapses into reward-hacking behaviour | classifier false positives (for example, the gripper occluding the cube near the goal) | retrain the classifier with the exploit frames as negatives; re-check the false-positive gallery |
| Parameter staleness grows without bound | the learner is slower than the actor (GPU contention, or the UTD is too high) | put the learner on its own GPU; lower the UTD; check that the transition queue is not also backing up |
| Actor crashes on launch while the learner is fine | environment rendering in a headless context | `MUJOCO_GL=egl` on Linux; `mjpython` rather than `python` on the Mac |
| Interventions not landing in the demonstration buffer | the `is_intervention` flag is lost in a wrapper | assert on `info["is_intervention"]` at the actor and log per-episode intervention step counts |
| `lerobot.rl` module paths differ from the tutorial's Codes 3–6 | v0.6 rebuilt the RL stack (`sac` became `gaussian_actor`) | trust `python -m lerobot.rl.<tab>` and the current docs page over the paper's snippets |
| Success rate differs between the classifier and the environment's ground truth at evaluation | threshold drift or distribution shift late in training | report both; ground truth is the evaluation metric and the classifier is only the training signal |

## Going deeper

- **Queue instrumentation.** Patch your local checkout to log the transition-queue depth, the parameter-queue depth, and the parameter staleness measured in actor steps between weight refreshes; keep the diff in the lesson directory. Staleness should settle at a small steady-state value rather than grow.
- **A scripted intervener.** Replace yourself with a scripted policy that seizes control when the episode step exceeds N with the cube ungrasped, or when the end effector leaves a workspace box, and rerun Exercise 6 at three budgets and two seeds. This is the reproducible version of the human-judgement result.
- **A classifier robustness loop.** Add your trained policy's near-success failures to the classifier's negative set, retrain, and measure whether the margin by which failures can be scored as successes drops. This is a miniature of the reward-model iteration loop that reappears in Lesson 20.

## References

- Luo, Xu, Wu, Levine. *Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning* (HIL-SERL), arXiv:2410.21845.
- Luo et al., *SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning*, arXiv:2401.16013, for the engineering lineage.
- LeRobot team, *Robot Learning: A Tutorial*, §3.2.1 Codes 3–6. arXiv:2510.12403.
- LeRobot "Train RL in Simulation" docs (the `lerobot.rl.actor` and `lerobot.rl.learner` commands, and the config examples at `huggingface.co/datasets/lerobot/config_examples`).
