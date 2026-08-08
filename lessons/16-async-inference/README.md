# Lesson 16 — Async Inference

**Goal:** the tutorial's own engineering contribution — decoupled PolicyServer/RobotClient — validated empirically, including its analytic queue bound.

## Read
- Tutorial §4.4 (inference taxonomy, Algorithm 1, queue analysis, the idle-avoidance bound g ≥ (E[l_S]/Δt)/H_a).

## Build
1. Run PolicyServer (localhost first, then a cloud GPU box) and RobotClient against a sim/mock robot with the Lesson 14 or 15 policy.
2. Sweep chunk-size threshold g ∈ {0, 0.3, 0.5, 0.7, 1.0}; reproduce the queue-size-evolution plots (tutorial Fig. 33) via `visualize_action_queue_size`.
3. Measure E[l_S] and control-loop idle time per g; empirically validate the analytic idle-avoidance bound.
4. Note LeRobot v0.5's Real-Time Chunking (RTC) — run it and compare against the plain async stack.

## Deliverables
- Queue plots, latency histograms, a table validating the bound, and a sync-vs-async-vs-RTC comparison writeup.

## Done when
Your measured idle-avoidance threshold matches the analytic bound within noise.

## Hardware echo
H3/H4 redeploy this against the physical arm with the policy server on a rented GPU.
