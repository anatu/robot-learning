# H3 — ACT & Diffusion Policy on Real Hardware

**Goal:** train both workhorse imitation policies on your H2 dataset and evaluate them like you mean it.

## Tasks
1. Train ACT and Diffusion Policy on the H2 dataset (cloud GPU, ~$2–6 total; `lerobot-train` works unmodified on RunPod/Vast).
2. Deploy on the arm via `mps` inference (tutorial Codes 8/10 pattern, updated for v0.6 APIs). Camera keys/resolutions must match training exactly.
3. Evaluation protocol (pre-registered in the repo before running): 20 trials per policy per condition — in-distribution starts + an OOD condition (shifted object position, new distractor). VLA-REPLICA's protocol (arXiv 2605.20774) is the reference standard.
4. Sync vs async inference comparison (Lesson 16's stack, policy server on a cloud GPU or the Mac).

## Deliverables
- Two Hub checkpoints; evaluation table (success rate ID/OOD, per-policy); rollout videos; failure taxonomy (grasp miss vs perception vs policy indecision).

## Done when
≥20 physical trials per configuration are logged with videos, and the ID/OOD gap is quantified.
