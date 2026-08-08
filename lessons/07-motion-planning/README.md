# Lesson 07 — Motion Planning (Optional)

**Goal:** the "plan" stage of the classical pipeline the tutorial critiques but never shows — sampling-based planning and trajectory optimization on the SO-101.

## Read
- MIT Robotic Manipulation ch. 7 (RRT, kinematic trajectory optimization, GCS overview).
- LaValle, *Planning Algorithms* (free online) for RRT/PRM foundations.

## Build
1. RRT in SO-101 joint space with MuJoCo collision checking; a cluttered tabletop scene.
2. Path shortcutting + time parameterization; execute in sim with the Lesson 04 tracker.
3. Kinematic trajectory optimization (direct transcription with collision penalties) on the same scene; compare path quality and compute time vs RRT.

## Deliverables
- Planner module + scene assets; side-by-side animation RRT vs trajopt; benchmark table (success rate, path length, wall-clock over 50 random scenes).

## Done when
Both planners reach 90%+ success on random feasible scenes and you can articulate when each wins.

## Skip criteria
Optional. The learning-based track never needs it, but it sharpens the "why end-to-end replaced this pipeline" argument — and mobile work (H6) reuses it.
