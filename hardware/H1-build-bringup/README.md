# H1 — Build & Bring-Up the SO-101

**Goal:** assembled, calibrated leader/follower pair with cameras; the physical platform for everything downstream. Start any time after Lesson 02.

## Tasks
1. Assemble both arms (~2–4 hrs; leader and follower use *different gear ratios* — don't mix servos). Follow TheRobotStudio/LeRobot assembly videos.
2. Motor setup + calibration: `lerobot-find-port`, `lerobot-setup-motors`, `lerobot-calibrate` per arm.
3. Mount cameras (printed wrist + overhead mounts from the SO-ARM100 repo); use two different webcam models to avoid USB enumeration confusion. Verify with `lerobot-find-cameras`.
4. Teleoperate: leader drives follower with live rerun visualization.
5. Reuse Lesson 04: drive the follower through a line and circle trace with your own diff-IK controller via the low-level motor API; log commanded vs measured joint angles.

## Deliverables
- Bring-up log (gotchas, calibration values, port mapping), teleop video, diff-IK trace video + error CSV.

## Done when
Smooth teleop, both cameras streaming at 30 fps, and your controller traces a circle within a few mm of the sim prediction.
