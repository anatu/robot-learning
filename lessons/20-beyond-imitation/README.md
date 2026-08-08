# Lesson 20 — Beyond Imitation: RL-from-Experience & World Models

**Goal:** where the frontier moved after the tutorial: imitation plateaus, and the field's answers are RL on top of VLAs and world-model policies.

## Read
- Physical Intelligence: π*0.6 + RECAP (Nov 2025) — RL with Experience and Corrections via Advantage-conditioned Policies; the "Precise manipulation via efficient online RL" post (Mar 2026).
- World-model thread: LeRobot v0.6 world-model policies (VLA-JEPA, FastWAM), GR00T N2 preview (world action models, GTC 2026), ETH Robot Learning lecture 8 (World Models, open on YouTube).
- Reward-model thread: Robometer/TOPReward in LeRobot v0.6.

## Build
1. Survey note (3–4 pages): taxonomy of post-imitation improvement — DAgger-style correction, RECAP advantage conditioning, online RL with RL tokens, world-model planning. Where does each get its signal?
2. Hands-on: run one world-model policy and one reward model from LeRobot v0.6 on a benchmark task; compare the reward model's judgments against ground-truth success on ≥100 episodes.

## Deliverables
- The note + reward-model calibration analysis (precision/recall vs ground truth).

## Done when
You can defend a position on "what closes the loop after BC" with evidence from your own runs.

## Hardware echo
H4's DAgger loop via `lerobot-rollout` is the budget version of exactly this.
