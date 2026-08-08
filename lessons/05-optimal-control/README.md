# Lesson 05 — Optimal Control Sprint: LQR → iLQR

**Goal:** the control-theoretic backbone the tutorial names but never teaches — distilled from CMU 16-745 into one lesson. This is what makes the RL chapters principled instead of recipes.

## Read/Watch
- CMU 16-745 lectures on LQR, TVLQR, and iLQR: https://optimalcontrol.ri.cmu.edu/ (2025 YouTube playlist)
- Underactuated Robotics ch. on LQR and trajectory optimization: https://underactuated.csail.mit.edu/

## Build
1. Finite-horizon discrete LQR via backward Riccati recursion; stabilize a linearized cartpole in MuJoCo (or plain numpy dynamics).
2. iLQR from scratch: cartpole swing-up. Line search, regularization, convergence plot.
3. TVLQR tracking of the iLQR trajectory under perturbed initial conditions.

## Deliverables
- `lqr.py`, `ilqr.py` with tests (Riccati fixed-point checks, cost-decrease assertions).
- Swing-up animation + cost-vs-iteration plots.

## Done when
iLQR swings up the cartpole from rest and TVLQR stabilizes it under ±10% initial-state perturbation.

## Why this matters downstream
SAC (Lesson 08) is solving the same problem with sampled gradients; world-model policies (Lesson 20) are learned MPC. Keep the connection explicit in the writeup.
