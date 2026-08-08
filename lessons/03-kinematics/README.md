# Lesson 03 — Kinematics From Scratch

**Goal:** rebuild FK/IK/Jacobian machinery properly — the tutorial's §2 does a 2-DOF planar case; extend to the full arm the way CS223A/MIT would.

## Read
- Tutorial §2.3 (planar FK, IK-as-optimization, differential IK).
- MIT Robotic Manipulation ch. 3 (pick-and-place kinematics): https://manipulation.csail.mit.edu/
- Lynch & Park, *Modern Robotics* ch. 4–6 for the SE(3) treatment the tutorial skips.

## Build
1. Reproduce the tutorial's 2-DOF planar SO-101: analytic FK, both IK branches (elbow-up/down), workspace visualization; unit tests asserting `FK(IK(p)) = p` to 1e-6 across sampled reachable poses.
2. Full 6-DOF: numerical FK from the MJCF kinematic tree; geometric Jacobian; numerical IK via damped least squares. Validate against MuJoCo's own FK.
3. Singularity study: behavior of `J⁺` vs damped least squares near singular configurations; condition-number maps over the workspace.
4. Constrained IK with an obstacle via `scipy.optimize` (reproduce the tutorial's Figure-6 feasible-set idea).

## Deliverables
- Notebook with animations + passing unit tests.
- Writeup: where the pseudo-inverse breaks and why damping fixes it.

## Done when
Your 6-DOF FK matches MuJoCo to numerical precision and your IK converges from random seeds across the reachable workspace.
