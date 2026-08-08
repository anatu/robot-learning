# H6 — Mobile Manipulation: LeKiwi → XLeRobot (Stretch)

**Goal:** extend the arm into a mobile manipulator without abandoning the stack you know.

## Path
1. **LeKiwi** (Seeed kit $179 + Raspberry Pi 5 ~$80): 3-omniwheel holonomic base carrying your SO-101 follower. Official LeRobot support (`lekiwi` robot class) — same record/train/eval pipeline, teleop over WiFi from the Mac.
2. Fetch-and-carry task: record a mobile-manipulation dataset (base + arm actions), train ACT or SmolVLA on it, evaluate.
3. **XLeRobot** upgrade (~$250 incremental: second SO-101 arm set + IKEA RASKOG cart + battery): dual-arm mobile household robot; community-maintained sim (ManiSkill) + docs at xlerobot.readthedocs.io.

## Deliverables
- Build log, a mobile task dataset on the Hub, a trained policy with rollout video.

## Done when
The robot fetches an object from another table on command. That's the demo that ends the build-in-public series properly.

## Caveats
WiFi teleop adds latency (affects demo quality); battery logistics; XLeRobot is a community project — expect to read issues and tinker.
