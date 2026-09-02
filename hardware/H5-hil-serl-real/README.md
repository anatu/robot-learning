# H5 — Real-Robot RL: HIL-SERL (Stretch)

This lesson runs the tutorial's centrepiece reinforcement-learning system on your physical arm: a learned reward classifier in place of a reward function, a decoupled actor and learner, and live interventions from the leader arm. The claim under test is the one made by Luo et al. (2024), that this combination reaches near-perfect success on a real manipulation task in one to two hours of robot time. Every component was rehearsed in simulation in Lessons 09 and 10, and the purpose here is to find out what changes when the environment is real. It is the most operationally demanding lesson in the course, because a robot exploring on its own initiative needs an engineered safety envelope and an attentive operator throughout.

| | |
|---|---|
| **Phase** | Hardware track (stretch) |
| **Time** | ~2 h workspace/config setup, ~2 h demos + classifier train/validation, one supervised 2–4 h RL session (you are present and attentive *throughout*), ~1 h eval + writeup |
| **Cost** | $0–4 (learner runs fine locally with a GPU; a cloud learner in networked mode adds pennies) |
| **Prerequisites** | 09–10 (RLPD + HIL-SERL in sim — non-negotiable; every concept here was rehearsed there), H1–H2 (hardware fluency, rig discipline), H3 (deployment debugging reflexes) |
| **Feeds into** | 20 (you'll have a live opinion on RL-from-experience), 22 capstone |

## Learning objectives

After this lesson you can:

1. **Constrain** a real-robot RL problem so that exploration is both safe and learnable, using end-effector-space actions, workspace bounds, region-of-interest crops, and short horizons.
2. **Decide** a reward classifier's operating point from held-out data, and explain why precision rather than recall is the constraint that binds.
3. **Predict** the shape of a live actor/learner session from your Lesson 10 run, then operate one: intervene productively, read the intervention-rate curve, and recognise when to stop.
4. **Evaluate** the trained policy against a bar you registered beforehand, and diagnose where the real session diverged from the simulated one.
5. **Judge** the one-to-two-hour claim from your own evidence.

## Principles

### Why this works when naive real-robot RL does not

Reinforcement learning from scratch on a physical robot fails for a simple reason: random exploration in joint space almost never produces a rewarding outcome, and every sample costs seconds of wall-clock time and wear on the hardware. HIL-SERL makes the problem tractable with four constraints, each of which you configure explicitly. The first is end-effector-space actions: the policy commands small $x$, $y$, $z$ displacements and an inverse-kinematics layer converts them to joint targets. The LeRobot documentation notes that some tasks are close to unlearnable in joint space and learnable in end-effector space, because the action dimensions then correspond directly to task-relevant motion. The second is workspace bounds: a box in end-effector space that exploration cannot leave, enforced by the `EEBoundsAndSafety` processor. This box, and not the human operator, is the primary safety mechanism; a human cannot react fast enough to an arm that decides to swing. The third is demonstrations in the replay buffer: the RLPD machinery from Lesson 09 seeds off-policy learning with successful trajectories, so the policy does not have to discover the task by chance. The fourth is human gating: you take over at incipient failure, and the intervention lands in the buffer as corrective experience. Lesson 10 showed how an intervened transition's presence in both buffers changes its effective sampling probability.

### The reward classifier is the foundation

The reinforcement learner will optimise whatever the classifier says is success, whether or not that matches what you intended. A classifier with a meaningful false-positive rate gets exploited: the policy finds the arm pose that triggers the classifier without completing the task, and the robot-hours you spent have purchased a reward hack. The order of operations therefore has to be fixed. You validate the classifier against a written precision bar on held-out data before the first reinforcement-learning step is taken. The documentation notes that manual keyboard annotation of success is possible instead of a classifier; build the classifier anyway, because a session without one skips the most instructive failure mode of the whole method.

### The shape of a session

Everything is driven by one configuration file, a `GymManipulatorConfig` with sections for `env.robot`, `env.teleop`, `env.processor.*`, and `dataset`. The learner process (`python -m lerobot.rl.learner`) holds the replay buffers and performs gradient steps. The actor process (`python -m lerobot.rl.actor`) drives the robot and streams transitions to the learner over gRPC. You stand by with the leader arm, and the `space` key toggles takeover. Episodes are short, with `control_time_s` around 20 seconds and a task horizon of 5 to 10 seconds as the documentation recommends, and the control rate is 10 fps. Short episodes matter for sample efficiency in two ways: more attempts fit into an hour, and each attempt's reward arrives sooner after the actions that earned it.

### Safety doctrine, extended for autonomous exploration

Everything from H1 applies, and the following additions apply because the arm now moves on its own initiative. The e-stop (the power cut) must be within reach and must be tested at the start of every session. The end-effector bounds must be verified empirically before any reinforcement learning, by driving the arm to each face of the box via teleoperation and confirming that the processor clamps. Nothing fragile may be inside or near the bounding box. You never leave the loop: a bathroom break means training paused and torque off. Wear nothing dangling near a robot that moves on its own initiative.

**Carry forward**

- The bounds box is the safety mechanism and the human is the learning mechanism; a human reaction is too slow to stop an exploring arm, but a human intervention is exactly the corrective data the learner needs.
- Classifier precision gates everything downstream, because a false positive is a reward the policy will learn to farm, and a session spent farming it is wasted.
- Short episodes and end-effector-space actions are what make the one-to-two-hour claim plausible at all; both are sample-efficiency levers rather than conveniences.
- The Lesson 10 curves are predictions for this lesson, and the places where reality deviates from them are what the lesson teaches.

| Source | Read for |
|---|---|
| [HIL-SERL real-robot docs](https://huggingface.co/docs/lerobot/hilserl) | the full workflow this lesson instantiates: config schema, bounds-finding, classifier, actor/learner — keep open throughout |
| Tutorial §3.2.1 + Luo et al. 2024 | what the orchestration you're about to run is implementing; the 1–2 h claim's original evidence |
| Your Lesson 10 `RESULTS.md` | your sim intervention-decay curve, classifier calibration, queue behavior — the predictions this lesson tests |

## Exercise 1 — Define the task, the bounds, and the configuration [Build]

In this exercise you set up the four constraints from the Principles section and verify the safety envelope before any learning takes place. The exercise tests objective 1, and its product is a configuration file in which every safety-relevant value has been checked against the physical arm. Budget about two hours with the robot powered and reinforcement learning switched off.

1. Choose the task: a reach-and-place or a push-to-region with a horizon of 5 to 10 seconds and a binary visual success condition, such as the object ending inside a marked zone. Avoid pick-and-place with a grasp for the first session, because grasping adds a contact-rich failure mode to a session that already has enough novelty. Write the success criterion as a single sentence.
2. Find the workspace bounds by teleoperating the follower through the whole useful task region while the limit finder records:
   ```bash
   lerobot-find-joint-limits \
     --robot.type=so101_follower --robot.port=<f-port> --robot.id=H1_follower \
     --teleop.type=so101_leader --teleop.port=<l-port> --teleop.id=H1_leader
   ```
   Take the printed minimum and maximum end-effector positions, shrink each face by about 1 cm of margin, and enter them in `env.processor.inverse_kinematics.end_effector_bounds`. Start with step sizes of about 0.01–0.02 m per axis.
3. Write the configuration JSON, starting from the documentation's example `env_config`: `env.robot` is your follower with both cameras; `env.teleop` is `so101_leader` with `control_mode: "leader"`; `fps: 10`; `reset.fixed_reset_joint_positions` is your H2 home pose; `control_time_s: 20`. The inverse-kinematics section needs the SO-101 URDF path and the end-effector frame name; see `lerobot/model/kinematics.py` and the SO-ARM100 repository for the URDF.
4. Verify the safety envelope empirically. With the processor pipeline live (record mode is sufficient), drive the leader hard toward each face of the bounding box; the follower must stop at the wall on all six faces. Then confirm that `space` engages and disengages takeover cleanly and that the reset pose is free of collisions.

**✅ Checkpoint:** all six bound faces clamp; takeover toggles cleanly; the configuration is committed to the repository; the e-stop reach test has been done with the arm moving.

## Exercise 2 — Record demonstrations and classifier data [Build]

Here you record the two datasets the system needs: demonstrations that seed the replay buffer, and a labelled dataset for training the reward classifier. The exercise exercises the third constraint from the Principles section and prepares the fourth. Budget about an hour.

1. Record roughly 20–30 demonstration episodes in record mode with leader teleoperation:
   ```bash
   python -m lerobot.rl.gym_manipulator --config_path configs/h5_record.json
   ```
   Annotate as the documentation describes, with `s` for success and `esc` for failure. Keep the demonstrations inside the shrunken bounds by construction.
2. Record a separate classifier dataset with `terminate_on_success: false`, roughly 15–20 episodes, so that frames after the success moment accumulate as positives; the documentation is explicit that this asymmetry matters. Include deliberate near-misses, such as the object adjacent to the zone or the gripper hovering above it, because these hard negatives are what keep the classifier from rewarding almost-success.
3. Determine the camera crops interactively and write them into every configuration file (`crop_params_dict`, with `resize_size: [128, 128]`):
   ```bash
   python -m lerobot.rl.crop_dataset_roi --repo-id <you>/h5_classifier_data
   ```
   The region of interest should cover the task zone and the gripper, and exclude you, the leader arm, and the rest of the room.

**✅ Checkpoint:** both datasets are on the Hub; the crops are chosen; the success and failure label counts are roughly balanced, with at least 200 positive frames.

## Exercise 3 — Train and validate the reward classifier [Decide]

This exercise tests objective 2 and contains the one hard stop in the course: you do not proceed to reinforcement learning until the classifier has passed a precision bar that you wrote down before seeing its performance. The reason is given in the Principles section, and it bears repeating: the policy will optimise whatever the classifier rewards, so a classifier that rewards near-misses produces a policy that performs near-misses. Budget about an hour.

1. Train with the documentation's recipe, using the `helper2424/resnet10` backbone at 128×128 on both cameras:
   ```bash
   lerobot-train --config_path configs/h5_reward_classifier.json
   ```
2. Apply the pre-registered validation gate on a held-out episode split: precision of at least 0.95 at your operating `success_threshold`, recall of at least 0.8, and a manual review of the false-positive gallery. Sweep the threshold using the machinery from Lesson 10, and choose the operating point for precision at the expense of recall. A missed success costs one wasted episode, whereas a false positive corrupts the training signal. Write the threshold and the rule that selected it in `RESULTS.md`.
3. Test the classifier live. Run the environment with `reward_classifier.pretrained_path` set and `terminate_on_success: true`, and stage five successes and five near-misses by teleoperation. Every near-miss that scores at or above the threshold must be investigated; the usual causes are a crop that is too loose or too few hard negatives, and the fix is more data and a retrain.

**✅ Checkpoint:** the written gate is passed on held-out data and on the ten-episode live test, and the threshold and its numbers are in `RESULTS.md`. Do not proceed on a failed gate.

## Exercise 4 — Run the training session [Predict → Run]

This is the live session, and it tests objective 3. Your Lesson 10 simulation run gave you an intervention-decay curve, a time to first success, and a sense of which component was hardest; those results are now predictions, and the session is the experiment that tests them. The session lasts two to four hours and you must be present and attentive throughout.

1. Before starting, write down from your Lesson 10 `RESULTS.md`: the predicted wall-clock time and episode count to the first success; the predicted shape of the intervention-rate curve, including the height of its initial plateau and when it should begin to decay; and which of the three components (classifier, decoupling, interventions) you expect to be hardest on the real arm. These predictions are what the session will be reconciled against.
2. Before the session, run the H2 preflight checklist, test the e-stop, spot-check two faces of the bounds, bring up Weights & Biases, and set a phone to record a timelapse of the rig. The timelapse is worth having for the write-up.
3. Launch the learner and then the actor, from the same configuration in two terminals, with `policy.type=gaussian_actor` and `algorithm.type=sac` as the current documentation specifies. These are the renamed modules you noted in Lesson 09.
   ```bash
   python -m lerobot.rl.learner --config_path configs/h5_train.json
   python -m lerobot.rl.actor   --config_path configs/h5_train.json
   ```
   The documentation recommends `temperature_init: 1e-2` (a temperature that is too high makes your interventions ineffective), `policy_parameters_push_frequency: 2` seconds, and `storage_device: cuda` if VRAM allows.
4. Follow the intervention protocol from the documentation and from your Lesson 10 experience. Let the policy explore the first episodes uninterrupted. Then intervene briefly at incipient failure, with short corrective nudges via `space` rather than long demonstrations. As successes begin, shrink interventions to quick finishing touches, and taper them deliberately. Keep subjective session notes with timestamps, because they will explain features of the curves later.
5. Watch the run on Weights & Biases. Episodic reward should trend upward, the intervention rate should decay along the shape of your Lesson 10 curve, and episode length should shorten as successes terminate early. Two pathologies need immediate action. If reward rises while your own observation says the task is not being completed, the classifier is being exploited: stop, fix the classifier, and restart. If reward is flat and interventions are not helping after 45 minutes, check the temperature and the tightness of the bounds before spending more of the session.
6. Stop at roughly ten consecutive uninterrupted successes or at four hours of wall-clock time, whichever comes first, and checkpoint the policy either way.

**✅ Checkpoint:** a completed session with logged curves and timestamped notes; the policy checkpoint saved; no safety event (any near-miss is written up in `FAILURES.md`); the predictions from step 1 reconciled against the curves.

## Exercise 5 — Evaluate and compare against simulation [Predict → Run]

In this exercise you evaluate the trained policy against a bar registered before the evaluation, and then write the comparison between this session and your Lesson 10 simulation run. The exercise tests objectives 4 and 5; the comparison is what turns a demonstration into a lesson, because the places where reality diverged from simulation are the transferable knowledge.

1. Register the bar before evaluating: at least 90% success over **twenty consecutive** rollouts with zero interventions and the object's start position randomised within the trained zone. Write your predicted success count beside the bar. Run the actor with interventions disabled (hands off the leader), twenty episodes, all on video.
2. In `RESULTS.md`, compare the real session against the Lesson 10 simulation: wall-clock time and environment steps to first success and to plateau; intervention count and the shape of its decay; the classifier operating points; and every place where reality diverged, such as latency, reset variance, and classifier drift with lighting, with your best account of the mechanism for each.
3. Judge the claim: state whether near-perfect success in one to two hours held on your rig, with the numbers that carry the verdict.

**✅ Checkpoint:** the twenty-rollout evaluation is on video with its number, whatever it is, and at least five simulation-to-real differences are recorded, each with a mechanism.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| Configs (`h5_record/classifier/train.json`) | committed, bounds + crops + reset pose included; rerunnable |
| Reward classifier + gate report | held-out precision/recall at threshold, FP gallery, live-test log |
| Policy checkpoint on Hub | with the exact config + step count it was stopped at |
| Training-session record | W&B curves, intervention-rate plot, timestamped operator notes |
| Eval: 20-rollout sheet + videos | pre-registered bar stated; consecutive, unedited |
| `RESULTS.md` | Exercise 4/5 predictions vs outcomes; ≥ 5 concrete sim-vs-real deltas each with a mechanism hypothesis; the verdict on the 1–2 h claim |

## Done when

- [ ] The classifier passed its pre-registered gate before any reinforcement learning.
- [ ] One full supervised session was completed with a declining intervention rate, with the predictions written beforehand.
- [ ] The twenty-consecutive-rollout evaluation is recorded, at 90% or above or with a precise account of why not; the account is equally valid coursework.
- [ ] The simulation-to-real write-up would save the next person an hour of their robot's time.
- [ ] There were zero uncontrolled contacts during the lesson.

## Self-check

1. Why is classifier *precision* the binding constraint, not recall? What does the policy do with each error type?
2. The EE bounds are the primary safety mechanism, not you. Why is a human reaction insufficient against an RL explorer, and what's the human actually for?
3. Why do short episodes (`control_time_s` ≈ 20 s) matter so much for wall-clock sample efficiency here? Two mechanisms.
4. Mid-session, reward climbs while true success stagnates. What happened, why did RL find it, and what's the fix path?
5. Your intervention-rate curve decays faster than Lesson 10's sim curve. Name three candidate explanations and how you'd distinguish them.

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Actor can't reach learner | gRPC address/port mismatch in config, or firewall in networked mode | same-machine first; verify config's learner address; then go networked |
| Policy pins itself to a bound face | reward gradient points out of the box (bounds too tight for the task) | re-teleop the task, re-derive bounds with margin; restart session |
| Interventions feel ignored (policy repeats mistakes) | temperature too high (docs' explicit warning) or parameter push too slow | `temperature_init: 1e-2`, push frequency 1–2 s |
| Classifier great offline, poor live | lighting/exposure shifted since classifier data | H2 lighting discipline; retrain with session-start frames mixed in |
| Learner throughput crawls | storage on CPU, or learner sharing the Mac with cameras + actor | `storage_device: cuda`; separate the learner (cloud/desktop GPU) |
| Episode resets drift the object out of the trained zone | manual reset variance | tape the reset zone; the fixed reset pose positions the arm, and you position the object the same way every time |
| v0.6 module paths differ from these commands | RL stack refactor drift | `python -m lerobot.rl.learner --help` and current hilserl docs page are authoritative |

## Going deeper

- **A classifier-hardening loop.** Take the trained policy's near-success failures, add them to the classifier's negatives, retrain, and measure whether the reward-hacking margin (the classifier's score on failures) drops. This is Lesson 10's extension carried out on real frames.
- **A grasp task, second round.** Repeat the lesson with pick-and-place, and compare session length and intervention count against the first round to put a price on the contact-rich failure mode.

## References

- [HIL-SERL real-robot workflow docs](https://huggingface.co/docs/lerobot/hilserl) — schema + commands verified Aug 2026 (`gaussian_actor` + `algorithm.type=sac` per your Lesson 09 version note).
- Luo et al. 2024, arXiv:2410.21845 (HIL-SERL); Tutorial §3.2.1 Codes 3–6.
- Your Lesson 09/10 `RESULTS.md`: the predictions under test.
