# Lesson 14 — ACT: Action Chunking with Transformers

**Goal:** train the first real policy of the course, then reproduce the ablation that made ACT work: chunking.

## Read
- Tutorial §4.2 (CVAE objective, architecture, ablations).
- Zhao et al. 2023 (ALOHA/ACT).

## Build
1. Extend the tutorial's Code 7 from a 1-step placeholder to a real run: ACT on `gym-aloha` TransferCube (dataset `lerobot/aloha_sim_transfer_cube_human`), 100k steps, W&B logging. Record/eval locally on the Mac; train on a rented GPU (~1 hr on A100, ~$1–3).
2. Scripted evaluation: ≥50 seeded rollouts, success rate with a fixed protocol.
3. Ablation: chunk size H_a ∈ {1, 10, 50, 100}, with/without EMA temporal ensembling. Qualitatively reproduce Zhao et al.'s 1% vs 44% chunking effect.
4. Unit-test your standalone implementation of the overlapping-chunk EMA aggregation.

## Deliverables
- Hub checkpoint + eval script + success-rate and action-smoothness (jerk) plots vs chunk size.

## Done when
Baseline ACT clears ~70% on TransferCube and the H_a=1 arm collapses, matching the paper's story.

## Stretch
Reimplement ACT's architecture (<500 lines) and assert loss parity with `ACTPolicy` on identical batches.
