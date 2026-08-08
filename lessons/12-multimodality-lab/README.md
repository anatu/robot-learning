# Lesson 12 — Why Generative Policies: The Multimodality Lab

**Goal:** demonstrate, on data you fully control, the two failure modes that motivate everything in Phase 4: mode-averaging and compounding error.

## Read
- Tutorial §4.0–4.1 (BC formalism, failure modes, generative-model intro).
- Florence et al. 2022 (Implicit BC) for the mode-averaging argument.

## Build
1. 2D toy BC dataset with two symmetric expert modes (go left/right around an obstacle).
2. Train four policy heads from scratch in PyTorch: MSE point regressor, CVAE, DDPM, conditional flow matching.
3. Side-by-side sampled-action scatter plots; a quantitative mode-coverage/indecision metric.
4. Rollout study: execute each policy closed-loop; show the MSE head splitting the difference into the obstacle.

## Deliverables
- Notebook with the four implementations (each <150 lines), plots, metric table.
- Writeup: which failure is distribution-shift and which is mode-averaging, and why chunking helps one but not the other.

## Done when
The MSE head visibly averages modes and at least two generative heads commit to a mode, quantified by your metric.
