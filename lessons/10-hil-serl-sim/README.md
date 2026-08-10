# Lesson 10 — HIL-SERL in Simulation

Run the tutorial's centerpiece RL system end-to-end in `gym-hil` — reward classifier, decoupled actor/learner, human interventions — and instrument every moving part, so that when H5 puts this on the real arm nothing in the architecture is new to you.

| | |
|---|---|
| **Phase** | 3 — Reinforcement learning |
| **Time** | ~3 sessions (10–12 h desk, much of it *interactive* — you are the human in the loop) + a few GPU-hours |
| **Cost** | ~$3–5 (learner runs fine on one 4090; actor+env run on the Mac or the same box) |
| **Prerequisites** | 09 (two-buffer machinery; the demos-and-config workflow), 08 (SAC internals for reading learner logs) |
| **Feeds into** | H5 (identical pipeline, physical arm), 20 (reward models are this classifier's descendants) |

## Learning objectives

After this lesson you can:

1. **Train and calibrate** a binary reward classifier and quantify — in wasted robot-steps — what its false-positive rate costs downstream RL.
2. **Operate** LeRobot's decoupled actor/learner architecture and read its two queues (transitions up, parameters down) as the system's vital signs.
3. **Measure** the effect of human interventions on sample efficiency at fixed environment-step budget.
4. **Explain** why an intervention transition's membership in *both* buffers changes its effective sampling probability, and verify the arithmetic in logs.
5. **Predict**, before H5, which parts of this system get harder on real hardware and which transfer unchanged.

## Background

**Why this architecture exists.** Real-robot RL has two constraints sim RL doesn't: the control loop cannot block (a robot mid-motion doesn't wait for a gradient step), and nobody will hand-engineer a reward function over pixels. HIL-SERL (Luo et al. 2024) answers with three components:

1. *Reward classifier.* A binary success detector trained on labeled frames replaces the reward function. It runs in the env loop, so its calibration is not a nicety: every false positive mints fake reward and teaches the policy to exploit the classifier, not solve the task. You will build the precision/recall analysis *before* any RL consumes it.
2. *Decoupled actor/learner.* Two processes. The actor rolls the policy in the env at fixed rate and streams transitions to the learner; the learner (RLPD-style SAC from Lesson 09 — demo buffer, 50/50 sampling) trains and streams fresh parameters back. The coupling is asynchronous: the actor acts on slightly stale weights, and the two queues (transition queue, parameter queue) absorb the rate mismatch. Depth and staleness of those queues are the observables you'll instrument.
3. *Human interventions.* While the actor rolls, a human can seize control (gamepad RB / keyboard toggle in `gym-hil`; `info["is_intervention"]` marks the frames). Intervention segments are on-policy corrections at exactly the states where the policy is wrong — the highest-value data in the system. They enter the demo buffer *and* the online buffer, so under 50/50 sampling a intervened transition is sampled from both streams: its effective probability is $\frac{0.5}{|\mathcal{D}_{\text{demo}}|} + \frac{0.5}{|\mathcal{D}_{\text{online}}|}$ — roughly double-counted early, demo-weighted late. The paper's claimed result of near-perfect policies in 1–2 h of real-robot time rests on this triad.

**The run loop you're building toward:** record demos → train classifier → launch learner → launch actor → intervene when it flails → watch intervention rate decay as the policy stops needing you.

| Source | Read for |
|---|---|
| Tutorial §3.2.1, Codes 3–6 | the reference pseudocode for classifier, actor, learner, orchestration — map each Code to the v0.6 module that replaced it |
| Luo et al. 2024, arXiv:2410.21845 | the claims under test: 1–2 h to high success; intervention-driven sample efficiency |
| LeRobot "Train RL in Simulation" docs | verbatim commands and JSON configs (quoted below); the `lerobot.rl.*` module layout |
| Lesson 09's `RESULTS.md` | your own baseline: pick-cube sample efficiency *without* interventions |

## Part 1 — Reward classifier, calibrated before trusted (Mac + short GPU, ~3 h)

Produces a success detector with a measured operating point.

1. Collect labeled frames with the `gym-hil` recorder: ~20 successful and ~20 failed episodes. Use the env's success/failure labeling controls (gamepad Y = success, A = failure per the gym-hil docs) so labels land in the dataset rather than in your memory. Config (Lesson 09's, retargeted):
   ```json
   { "env": { "type": "gym_manipulator", "name": "gym_hil",
              "task": "PandaPickCubeKeyboard-v0", "fps": 10 },
     "dataset": { "repo_id": "<you>/gymhil_classifier_frames_v1",
                  "task": "pick_cube", "num_episodes_to_record": 40,
                  "push_to_hub": true },
     "mode": "record" }
   ```
2. Train a binary classifier on terminal-window frames (positives: last ~10 frames of successes; negatives: everything else + failure episodes). LeRobot ships a reward-classifier trainer in its RL stack — check the current docs/`lerobot.rl` module listing for the entry point and config; if your installed version lacks one, a ~60-line PyTorch script over a frozen ResNet-18 torso with a linear head is the fallback (and is what the tutorial's Code 3 sketches).
3. Evaluate on 5 held-out episodes: PR curve, threshold sweep, and a *false-positive gallery* — the actual frames the classifier calls success. Look at them; they are previews of what your policy would learn to exploit.
4. Pick the operating threshold by a stated rule (e.g. precision ≥ 0.95 at max recall) and freeze it in the config.

**✅ Checkpoint:** held-out precision ≥ 0.95 at your threshold; the FP gallery exists in `RESULTS.md` with one sentence each on *why* the classifier was fooled.

## Part 2 — Actor/learner bring-up, instrumented (Mac + GPU, ~2 h)

Produces a running system with its vitals on a dashboard before any experiment.

1. Pull LeRobot's example `rl/gym_hil/train_config.json` from `lerobot/config_examples`; point it at your demo dataset (Lesson 09's 30 demos), your classifier, and the `PandaPickCubeKeyboard-v0` (or gamepad) task.
2. Launch, two terminals:
   ```bash
   python -m lerobot.rl.learner --config_path configs/train_gym_hil.json
   python -m lerobot.rl.actor   --config_path configs/train_gym_hil.json   # mjpython on Mac
   ```
3. Instrument (log alongside W&B): transition-queue depth, parameter-queue depth, and parameter staleness (actor env-steps between weight refreshes). If the stack doesn't expose them directly, wrap the queue objects — this is allowed to be a small patch to your local checkout; keep it as a diff in the lesson directory.
4. Kill the learner mid-run; confirm the actor keeps acting on stale weights and recovers when the learner returns. Note what happened to the queues.

**✅ Checkpoint:** dashboard shows all three vitals live; the kill-recovery behavior is documented. Staleness should sit at a small steady-state value (a few env steps), not grow unboundedly — if it grows, see Pitfalls.

## Part 3 — Train with interventions (interactive, the fun part, ~2 h)

Produces the headline policy and the intervention-decay curve.

1. Full run at your Lesson 09 step budget (150k actor steps or until stable success): intervene by feel — grab control when the policy is clearly failing, release when recovered. Every intervention is logged via `info["is_intervention"]`.
2. Log per-episode: success (classifier *and* env ground truth — keep both), intervention fraction of steps, episode length.
3. Watch for the signature dynamic: interventions frequent early, decaying as the policy improves. Stop intervening entirely for the last ~20% of the run.

**✅ Checkpoint:** final policy ≥ 80% success over 50 scripted evaluation episodes *with interventions disabled*; the intervention-fraction curve decays from its early plateau to near zero. Save the curve — it's the lesson's signature figure.

## Part 4 — The intervention-budget experiment (scripted, ~3 h GPU)

Human feel doesn't replicate; script the human to make the claim quantitative.

1. Build a scripted oracle intervener from your demo policy or a hand-coded pick routine: it seizes control when (a) episode step > N with cube ungrasped, or (b) EE strays outside a workspace box — crude is fine, consistent is the point.
2. Three arms × 2 seeds at *fixed total env steps* (e.g. 100k): no interventions (= Lesson 09's RLPD arm, reuse it), sparse oracle (intervene in ≤ 20% of episodes), generous oracle (≤ 60%).
3. Plot success-vs-steps for the three arms; tabulate steps-to-80%.
4. Verify the both-buffers arithmetic: from buffer-size logs, compute the expected sampling probability of an intervention transition vs a plain online transition at three points in training; confirm against observed sampling counts if the stack exposes them (else derive from buffer sizes alone and say so).

**✅ Checkpoint:** generous ≥ sparse ≥ none in sample efficiency, visible across both seeds; the sampling-probability table appears in `RESULTS.md` with the doubling effect annotated.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| Hub: classifier + final policy checkpoints | classifier card states threshold + held-out PR; policy card links the eval protocol |
| `configs/` (+ any instrumentation diff) | full system relaunchable from committed files |
| `plots/` | PR curve + FP gallery, queue/staleness dashboard export, intervention-decay curve, 3-arm budget comparison |
| `RESULTS.md` | operating-point rationale; kill-recovery note; budget table; sampling-probability arithmetic; ≤ 12 sentences |

## Done when

- [ ] Classifier shipped with measured precision ≥ 0.95 and its FP gallery reviewed.
- [ ] Policy ≥ 80% success (interventions off, 50 episodes) trained with your own interventions.
- [ ] Intervention-decay curve exists and actually decays.
- [ ] Scripted-oracle study shows the intervention arms dominating at equal step budget.
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

## Stretch

Close the loop on classifier robustness: take your trained policy's near-success failures, add them to the classifier's negative set, retrain, and measure whether the reward-hacking margin (classifier score on failures) drops. This is a miniature of the reward-model iteration loop that reappears in Lesson 20.

## References

- Luo, Xu, Wu, Levine. *Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning* (HIL-SERL), arXiv:2410.21845.
- Luo et al., *SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning*, arXiv:2401.16013 — the engineering lineage.
- LeRobot team, *Robot Learning: A Tutorial*, §3.2.1 Codes 3–6. arXiv:2510.12403.
- LeRobot "Train RL in Simulation" docs (`lerobot.rl.actor` / `lerobot.rl.learner` commands + config examples at `huggingface.co/datasets/lerobot/config_examples`).
