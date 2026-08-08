# H5 — Real-Robot RL: HIL-SERL (Stretch)

**Goal:** the full tutorial §3 pipeline on physical hardware — reward classifier, decoupled actor/learner, live human interventions from the leader arm. The 99%-in-1-2-hours claim, tested by you.

## Prereqs
Lessons 09–10 done in sim; H1–H2 hardware fluency. Supervise the arm at all times; keep the e-stop (power cut) reachable.

## Tasks
1. Collect a success/failure image dataset for a reach-and-place or push task; train the reward classifier; validate precision/recall before any RL (a miscalibrated classifier wastes robot-hours).
2. Record ~20–50 demos into the offline buffer.
3. Run the actor on the Mac driving the arm; learner on a cloud GPU (networked mode) or locally. Intervene via the leader arm when the policy goes wrong.
4. Train 1–2 hours of wall-clock robot time; log intervention rate per episode.
5. Evaluate: ≥20 consecutive rollouts, target >90% success.

## Deliverables
- Trained SACPolicy on the Hub; intervention-decay plot; evaluation video; a writeup on where the real run diverged from your sim run (Lesson 10).

## Done when
The policy clears your success bar without interventions, or you've documented precisely why it didn't.
