# Lesson 06 — Grasp Mechanics (Optional)

**Goal:** the analytic lineage behind learned grasping — contact models, force/form closure, grasp force optimization as a convex program. Straight from Stanford CS237B HW2, which is fully public.

## Read
- CS237B grasping lectures + HW2: https://web.stanford.edu/class/cs237b/ (public slide PDFs, homework PDF, starter code on GitHub)
- MIT Robotic Manipulation ch. 5 (clutter/grasp statics) for friction cones and antipodal grasps.

## Build
1. Implement force-closure and form-closure tests for planar and 3D contact sets.
2. Grasp force optimization with friction-cone constraints via `cvxpy`.
3. Antipodal grasp scoring on a synthetic point cloud (sampled from an SO-101-graspable object mesh).

## Deliverables
- Notebook + tests; a table of closure verdicts for a set of contact configurations with intuition for each.

## Done when
Your closure tests agree with hand-derivable cases and the cvxpy optimizer finds valid force distributions inside friction cones.

## Skip criteria
Optional. Skip if the goal is speed-to-modern-methods; do it if grasping intuition feels rusty — H3's failure analysis benefits from it.
