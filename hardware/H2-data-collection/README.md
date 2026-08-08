# H2 — Real Teleop Data Collection

**Goal:** a real, published dataset — the raw material for H3/H4. This is the tutorial's §1.3 recipe on your own hardware.

## Tasks
1. Design one pick-place task with a repeatable reset (fixed start zone, target container). Consistent lighting (desk lamp).
2. Adapt the tutorial's recording script (Code 2) to current LeRobot: `lerobot-record` with front + wrist cameras, 30 fps, keyboard events for re-record/early-stop, 10 s reset phases.
3. Record 50+ episodes. Re-record failures deliberately or keep them labeled — decide and document your policy.
4. Push to the Hub with a proper dataset card: fps, features, episode count, camera layout photo, task description.

## Deliverables
- Public HF dataset + card; a 1-page failure-mode log (dropped frames, sync issues, calibration drift, USB quirks).

## Done when
The dataset renders correctly in the LeRobot visualizer and a stranger could reproduce your recording setup from the card.

## Quality notes
Demo quality caps policy quality: consistent grasp strategy across episodes, cover the start-position distribution you'll evaluate on, keep episodes under ~60 s.
