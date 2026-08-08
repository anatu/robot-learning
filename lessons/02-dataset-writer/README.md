# Lesson 02 — Write Your Own Dataset

**Goal:** close the loop on the format — produce a valid LeRobotDataset from simulation, the same pipeline the hardware track will use with real teleop.

## Read
- Tutorial §1.3 (data collection recipe).
- SO-101 MJCF: TheRobotStudio `SO-ARM100/Simulation/SO101/so101_new_calib.xml`.

## Build
1. Load the SO-101 model in MuJoCo. Script 50 pick-or-reach trajectories (scripted IK or waypoint interpolation — Lesson 03 will do this properly; keep it simple here).
2. Render two camera views per step; serialize joint states, actions, and frames into a valid v3 dataset (correct `info.json` schema, `stats.json`, episode metadata, MP4 encoding) using `LeRobotDataset.create()` + `save_episode()`.
3. Push to the Hub; confirm it loads with `delta_timestamps` and renders in the LeRobot dataset visualizer.

## Deliverables
- Public HF dataset repo + the generation script.
- Short writeup: what the format forced you to get right (fps sync, stats computation, video encoding).

## Done when
Your Hub dataset loads cleanly in `LeRobotDataset` and displays correctly in the visualizer.
