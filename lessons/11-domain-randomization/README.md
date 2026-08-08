# Lesson 11 — Domain Randomization & the Reality Gap

**Goal:** the tutorial surveys DR (AutoDR, DORAEMON) with zero code. Make it empirical: how much randomization transfers, and when it hurts.

## Read
- Tutorial §3.2.2 (simulators, reality gap, DR formalism `D_ξ`, ξ ~ Ξ).
- Tobin et al. 2017; Akkaya et al. 2019 (AutoDR); Tiboni et al. 2024 (DORAEMON).

## Build
In a MuJoCo push or reach task:
1. Fixed-range DR over friction and mass; train SAC per randomization width.
2. AutoDR-lite: widen uniform bounds automatically on success.
3. Evaluate every policy zero-shot on a held-out grid of test dynamics.
4. Produce the transfer heatmap (train distribution × test dynamics → success).

## Deliverables
- Heatmap + curves reproducing the low-entropy-fails / over-randomization-hurts tradeoff; short writeup tying it to sim-to-real practice (why π-labs collect real data instead).

## Done when
The heatmap shows both failure modes and identifies the sweet-spot width for your task.

## Stretch
Run NVIDIA's free "Train an SO-101 From Sim-to-Real With Isaac" learning path (cloud RTX instance) as a guided sim-to-real counterpart: https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html
