# Lesson 10 — HIL-SERL in Simulation

**Goal:** the tutorial's centerpiece RL method — reward classifiers, decoupled actor/learner, human interventions — run end-to-end in `gym-hil` before ever touching hardware.

## Read
- Tutorial §3.2.1 (Codes 3–6: reward classifier, Actor, Learner, orchestration).
- Luo et al. 2024 (HIL-SERL); Luo et al. 2025 (SERL suite).
- LeRobot "Train RL in Simulation" docs (gym-hil tutorial).

## Build
1. Train a binary reward classifier (ResNet-18 backbone) on success/failure frames from gym-hil episodes; evaluate precision/recall + threshold sweep, with a false-positive gallery.
2. Run the decoupled actor/learner architecture (two processes, transition + parameter queues). Instrument queue depths and parameter staleness.
3. Train with keyboard/gamepad interventions. Log intervention rate per episode.
4. Intervention-budget study: none vs sparse vs generous scripted-oracle interventions at fixed total env steps.

## Deliverables
- Trained policy + classifier on the Hub; success-rate and intervention-decay plots; writeup on why intervention transitions entering *both* buffers changes their effective sampling probability.

## Done when
The intervention arm clearly beats no-intervention at equal steps, and you've watched the intervention rate decay as the policy improves.

## Hardware echo
H5 repeats this on the real arm. Everything you instrument here (classifier calibration, queue behavior) transfers directly.
