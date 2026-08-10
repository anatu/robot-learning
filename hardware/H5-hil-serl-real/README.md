# H5 — Real-Robot RL: HIL-SERL (Stretch)

The tutorial's centerpiece pipeline on physical hardware: reward classifier, decoupled actor/learner, live interventions from the leader arm — and the "near-perfect success in 1–2 hours" claim, tested by you. This is the most operationally demanding lesson in the course; it is also the one where everything you've built converges.

| | |
|---|---|
| **Phase** | Hardware track (stretch) |
| **Time** | ~2 h workspace/config setup, ~2 h demos + classifier train/validation, one supervised 2–4 h RL session (you are present and attentive *throughout*), ~1 h eval + writeup |
| **Cost** | $0–4 (learner runs fine locally with a GPU; a cloud learner in networked mode adds pennies) |
| **Prerequisites** | 09–10 (RLPD + HIL-SERL in sim — non-negotiable; every concept here was rehearsed there), H1–H2 (hardware fluency, rig discipline), H3 (deployment debugging reflexes) |
| **Feeds into** | 20 (you'll have a live opinion on RL-from-experience), 22 capstone |

## Learning objectives

After this lesson you can:

1. **Constrain** a real-robot RL problem so exploration is safe and learnable: EE-space actions, workspace bounds, ROI crops, short horizons.
2. **Train and calibrate** a reward classifier on your own data and articulate why classifier precision is the binding constraint on everything downstream.
3. **Operate** a live actor/learner training session: intervene productively from the leader arm, read the intervention-rate curve, and know when to stop.
4. **Evaluate** the trained policy against a pre-registered bar and diagnose the sim-to-real deltas against your Lesson 10 run.
5. **Judge** the HIL-SERL claim from your own evidence.

## Background

**Why this works when naive real-robot RL doesn't.** Four constraints do the heavy lifting, and you configure every one: (1) *EE-space actions* — the policy commands x,y,z deltas, IK handles joints; per the docs, some tasks are near-unlearnable in joint space but learnable in EE space; (2) *workspace bounds* — a box in EE space that exploration cannot leave, enforced by the `EEBoundsAndSafety` processor: this is the primary safety mechanism, not the human; (3) *demos in the buffer* — your RLPD machinery from Lesson 09, seeding SAC off-policy learning; (4) *human gating* — you take over at incipient failure, and interventions land in the buffer as corrective experience (Lesson 10 taught you how their double-entry changes effective sampling).

**The reward classifier is the foundation.** RL will optimize *whatever the classifier says*. A false-positive-prone classifier gets exploited — the policy finds the arm pose that fools it, and your robot-hours purchase a reward hack. Hence the order of operations: validate the classifier to a written precision bar *before* the first RL step. (The docs note manual keyboard annotation is possible instead; do the classifier anyway — H5 without it dodges the most instructive failure mode.)

**Session shape.** Config JSON (`GymManipulatorConfig`: `env.robot`, `env.teleop`, `env.processor.*`, `dataset`) drives everything. Learner (`python -m lerobot.rl.learner`) holds replay buffers and does gradient steps; actor (`python -m lerobot.rl.actor`) drives the robot and streams transitions over gRPC; you hover with the leader arm, `space` toggles takeover. Episodes are short (`control_time_s` ~20 s, task horizon 5–10 s per the docs' recommendation), fps 10.

**Safety doctrine, extended for autonomous exploration.** Everything from H1 plus: e-stop (power cut) within reach and *tested at session start*; EE bounds verified empirically before RL (drive the arm to each bound face via teleop — the processor must clamp); nothing fragile inside or near the bounding box; you never leave the loop — bathroom break = training paused, torque off. Wear no dangling anything near a robot that moves on its own initiative.

| Source | Read for |
|---|---|
| [HIL-SERL real-robot docs](https://huggingface.co/docs/lerobot/hilserl) | the full workflow this lesson instantiates: config schema, bounds-finding, classifier, actor/learner — keep open throughout |
| Tutorial §3.2.1 + Luo et al. 2024 | what the orchestration you're about to run is implementing; the 1–2 h claim's original evidence |
| Your Lesson 10 `RESULTS.md` | your sim intervention-decay curve, classifier calibration, queue behavior — the predictions this lesson tests |

## Part 1 — Task, bounds, config (~2 h, robot on, RL off)

1. **Task:** reach-and-place or push-to-region, 5–10 s horizon, binary visual success (object in marked zone). Resist pick-and-place-with-grasp for round one — grasp adds a contact-rich failure mode to a session that has enough novelty. Write the success criterion sentence.
2. **Workspace bounds** — teleop the follower through the entire useful task region while the limit finder records:
   ```bash
   lerobot-find-joint-limits \
     --robot.type=so101_follower --robot.port=<f-port> --robot.id=H1_follower \
     --teleop.type=so101_leader --teleop.port=<l-port> --teleop.id=H1_leader
   ```
   Take the printed min/max EE positions, shrink each face by ~1 cm margin, and put them in `env.processor.inverse_kinematics.end_effector_bounds`. Step sizes ~0.01–0.02 m per axis to start.
3. **Config JSON** (start from the docs' example `env_config` and edit): `env.robot` = your follower + both cameras; `env.teleop` = `so101_leader` with `control_mode: "leader"`; `fps: 10`; `reset.fixed_reset_joint_positions` = your H2 home pose; `control_time_s: 20`; IK section needs the SO-101 URDF path + EE frame name (see `lerobot/model/kinematics.py` and the SO-ARM100 repo for the URDF).
4. **Verify the safety envelope empirically:** with the processor pipeline live (record mode is fine), drive the leader hard toward each bound face — the follower must stop at the wall, all six faces. Then verify `space` takeover engages/disengages cleanly and the reset pose is collision-free.

**✅ Checkpoint:** all six bound faces clamp; takeover toggles cleanly; config committed to the repo; e-stop reach-test done with the arm moving.

## Part 2 — Demos and classifier data (~1 h)

1. Record ~20–30 demo episodes (record mode, leader teleop):
   ```bash
   python -m lerobot.rl.gym_manipulator --config_path configs/h5_record.json
   ```
   Annotate per the docs: `s` = success, `esc` = failure. Keep demos inside the shrunken bounds by construction.
2. Record a *classifier* dataset variant with `terminate_on_success: false` (~15–20 episodes) so post-success frames accumulate positives — the docs are explicit that this asymmetry matters. Include deliberate near-misses: object adjacent to the zone, gripper hovering — the hard negatives that keep the classifier honest.
3. Determine camera crops interactively and write them into every config (`crop_params_dict`, `resize_size: [128, 128]`):
   ```bash
   python -m lerobot.rl.crop_dataset_roi --repo-id <you>/h5_classifier_data
   ```
   ROI covers task zone + gripper, excludes you, the leader arm, and the room.

**✅ Checkpoint:** two datasets on the Hub; crops chosen; success/failure label counts roughly balanced with ≥ 200 positive frames.

## Part 3 — Train and *validate* the reward classifier (~1 h)

1. Train (docs' recipe: `helper2424/resnet10` backbone, 128×128, both cameras):
   ```bash
   lerobot-train --config_path configs/h5_reward_classifier.json
   ```
2. **Validation gate, pre-registered:** on a held-out episode split — precision ≥ 0.95 at your operating `success_threshold`, recall ≥ 0.8, plus a manual false-positive gallery review. Sweep the threshold (Lesson 10's machinery); pick the operating point for precision, at recall's expense — a missed success costs a wasted episode, a false positive corrupts training.
3. Live test: run the env with `reward_classifier.pretrained_path` set and `terminate_on_success: true`; stage 5 successes and 5 near-misses by teleop. Every near-miss that scores ≥ threshold gets investigated (usually: crop too loose or hard-negative shortage — fix data, retrain).

**✅ Checkpoint:** written gate passed on held-out data *and* the 10-episode live test; threshold + numbers in `RESULTS.md`. Do not proceed on a failed gate — this is the one hard stop in the course.

## Part 4 — The training session (2–4 h, fully supervised)

1. Pre-session: H2 preflight; e-stop test; bounds spot-check (two faces); W&B up; phone timelapse of the rig running (you'll want it).
2. Launch learner, then actor (same config, two terminals; `policy.type=gaussian_actor`, `algorithm.type=sac` per current docs — your Lesson 09 rename note made flesh):
   ```bash
   python -m lerobot.rl.learner --config_path configs/h5_train.json
   python -m lerobot.rl.actor   --config_path configs/h5_train.json
   ```
   Docs-recommended knobs: `temperature_init: 1e-2` (too high makes your interventions ineffective), `policy_parameters_push_frequency: 2` s, `storage_device: cuda` if VRAM allows.
3. **Intervention protocol** (from the docs + your Lesson 10 experience): let it explore the first episodes uninterrupted; then intervene *briefly* at incipient failure — short corrective nudges via `space`, not long demonstrations; as success begins, shrink interventions to quick finishing touches; deliberately taper. Log subjective session notes with timestamps — they'll explain curve features later.
4. Watch on W&B: episodic reward trending up, intervention rate decaying (your Lesson 10 curve is the reference shape), episode length shortening as successes terminate early. Pathologies: reward up + your own eyes say "not succeeding" = classifier exploited → stop, fix classifier, restart; flat reward + interventions not helping past 45 min → check temperature and bounds tightness before burning the session.
5. Stop when: ~10 consecutive uninterrupted successes, or 4 h wall-clock, whichever first. Checkpoint the policy either way.

**✅ Checkpoint:** a completed session with logged curves + timestamped notes; policy checkpoint saved; no safety event (any near-miss gets written up in `FAILURES.md`).

## Part 5 — Evaluation and the sim-to-real delta (~1 h)

1. Pre-registered bar (write before eval): ≥ 90% over **20 consecutive** rollouts, zero interventions, object start randomized within the trained zone. Actor with interventions disabled (hands off the leader), 20 episodes, videos on.
2. `RESULTS.md`, the comparison that makes this a lesson rather than a stunt — real vs your Lesson 10 sim run: wall-clock and env-steps to first-success and to plateau; intervention count + decay shape; classifier operating points; every place reality diverged (latency, reset variance, classifier drift with lighting) and your best mechanism story for each.

**✅ Checkpoint:** 20-rollout eval on video with the number, whatever it is.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| Configs (`h5_record/classifier/train.json`) | committed, bounds + crops + reset pose included; rerunnable |
| Reward classifier + gate report | held-out precision/recall at threshold, FP gallery, live-test log |
| Policy checkpoint on Hub | with the exact config + step count it was stopped at |
| Training-session record | W&B curves, intervention-rate plot, timestamped operator notes |
| Eval: 20-rollout sheet + videos | pre-registered bar stated; consecutive, unedited |
| `RESULTS.md` sim-vs-real analysis | ≥ 5 concrete deltas vs Lesson 10, each with a mechanism hypothesis |

## Done when

- [ ] Classifier passed its pre-registered gate before any RL.
- [ ] One full supervised session completed with declining intervention rate.
- [ ] 20-consecutive-rollout eval recorded; ≥ 90% or a precise account of why not (the account is equally valid coursework).
- [ ] Sim-vs-real writeup would save the next person an hour of their robot's time.
- [ ] Zero uncontrolled contacts all lesson.

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
| Episode resets drift the object out of the trained zone | manual reset variance | tape the reset zone; fixed reset pose does the arm, *you* do the object — consistently |
| v0.6 module paths differ from these commands | RL stack refactor drift | `python -m lerobot.rl.learner --help` and current hilserl docs page are authoritative |

## References

- [HIL-SERL real-robot workflow docs](https://huggingface.co/docs/lerobot/hilserl) — schema + commands verified Aug 2026 (`gaussian_actor` + `algorithm.type=sac` per your Lesson 09 version note).
- Luo et al. 2024, arXiv:2410.21845 (HIL-SERL); Tutorial §3.2.1 Codes 3–6.
- Your Lesson 09/10 `RESULTS.md` — the predictions under test.
