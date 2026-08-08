# Lesson 15 — Diffusion Policy + Sampler Study

**Goal:** train Diffusion Policy on PushT, then measure the inference-cost story: DDPM vs DDIM vs a flow-matching head.

## Read
- Tutorial §4.3 (conditional diffusion objective, U-Net vs transformer, DDIM).
- Chi et al. 2024 (Diffusion Policy); Song et al. 2022 (DDIM).

## Build
1. Train DP on `gym-pusht` (dataset `lerobot/pusht`). Feasible locally on `mps`; faster in cloud.
2. Sampler comparison at inference: DDPM (T=100) vs DDIM (T=10) vs a self-implemented FM head with forward-Euler at {1, 2, 5, 10} steps.
3. Table: success rate vs wall-clock inference latency per sampler/step count.
4. Reproduce the tutorial's Figures 24–27 idea on real robot data: extract a 2D joint distribution from `lerobot/svla_so101_pickplace`, train a small ε-regressor and a CFM vector-field regressor, visualize denoising trajectories and path straightness.

## Deliverables
- Hub checkpoint; sampler table; animated GIFs of diffusion vs FM paths with a measured straightness ratio.

## Done when
DDIM/FM retain DDPM-level success at ~10× fewer steps on your runs — or you can explain why they didn't.
