# Lesson 04 — Differential IK as Optimization + Feedback

**Goal:** the tutorial gives `q̇ = J⁺ṗ*` and a proportional correction; MIT formulates diff-IK as a constrained QP. Build both and stress-test them.

## Read
- Tutorial §2.3.1 (feedback loops).
- MIT Robotic Manipulation ch. 3 (differential IK as QP with joint/velocity limits).

## Build
1. Open-loop diff-IK tracker: trace a circle and a line with the SO-101 in MuJoCo via `q̇ = J⁺ṗ*` + forward-Euler.
2. Add proportional feedback `q̇ = J⁺(ṗ* + k_p Δp)`; sweep `k_p`.
3. Inject model mismatch (perturb link lengths 5–10%) and a moving disturbance; plot tracking error open-loop vs feedback.
4. QP version (`qpsolvers` or `cvxpy`): add joint-limit and velocity-limit constraints; compare behavior near limits vs the unconstrained pseudo-inverse.

## Deliverables
- Single reproducible script + plots (error vs time, per controller, per `k_p`).
- Writeup connecting results to the tutorial's brittleness argument (§2.4).

## Done when
The feedback controller rejects the disturbance the open-loop one can't, and the QP respects limits where `J⁺` violates them.

## Note for hardware track
This controller gets reused in H1 to drive the real arm through a line/circle trace — keep the interface clean (`q̇ = f(q, target)`).
