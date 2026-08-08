# Lesson 09 — Sample-Efficient, Data-Driven RL: RLPD

**Goal:** the mechanism that makes real-world RL viable — offline demonstrations in a second buffer with 50/50 sampling. Ablate it so the design choices stop being folklore.

## Read
- Tutorial §3.2 ("sample-efficient, data-driven RL").
- Ball et al. 2023, *Efficient Online RL with Offline Data* (RLPD).

## Build
In `gym-hil` (Franka pick-cube, runs on the Mac via `mjpython`; train on cloud GPU):
1. Record ~30 demos (scripted or keyboard teleop) to a LeRobotDataset.
2. Train SAC three ways: (a) online-only, (b) demos pre-loaded into a single buffer, (c) RLPD two-buffer 50/50 sampling.
3. Ablate LayerNorm in the critic on/off.
4. Plot steps-to-success-rate for all arms, ≥3 seeds each.

## Deliverables
- Configs + seeds committed; sample-efficiency curves; a writeup on why 50/50 sampling beats pre-loading (effective oversampling of expert transitions).

## Done when
The RLPD arm demonstrably dominates online-only in sample efficiency across seeds, and you can explain the LayerNorm result.

## Version note
LeRobot v0.6 renamed the `sac` policy to `gaussian_actor` and rebuilt the RL stack — check current module paths before porting tutorial snippets.
