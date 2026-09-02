# Lesson 03 — Kinematics

Know where the SO-101's geometry breaks the naive math — singularities, the 5-DOF wall, damping's bias-for-velocity trade — by validating an FK/Jacobian/IK library against MuJoCo to numerical precision and then mapping its failure loci yourself. The tutorial's §2 does a 2-DOF planar toy; this lesson does the real arm.

| | |
|---|---|
| **Phase** | 2 — Classical core |
| **Time** | 1–2 sessions (5–6 h desk time), all Mac-local |
| **Cost** | $0 |
| **Prerequisites** | 00 (MuJoCo working, SO-ARM100 repo cloned) |
| **Feeds into** | 04 (`fk`, `jacobian` feed the diff-IK tracker), 07 (collision-checked planning uses this FK), H1 (the same library scores the real arm's trace) |

**A correction to the scaffold:** the SO-101 is a **5-DOF arm + 1-DOF gripper** (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper` in `so101_new_calib.xml`), not 6-DOF. That's not a detail — with 5 joints, arbitrary SE(3) pose targets are *generically infeasible* (you can hit a 5-dim submanifold of the 6-dim pose space). Everything below therefore poses IK as **position (3-DOF)** tasks.

## Learning objectives

After this lesson you can:

1. **Derive** the planar 2-link FK and both analytic IK branches, and state the reachability condition.
2. **Diagnose** a frame-convention bug in numerical FK from its error signature alone (which configurations, what magnitude).
3. **Predict** which IK method's joint velocity blows up at the workspace boundary, and why damping bounds it at the price of bias.
4. **Predict** where this arm's singularities lie from its joint-axis inventory, then confirm on a $\sigma_{\min}(J)$ map.
5. **Decide** a damping $\lambda$ for a real controller from a tracking-bias vs joint-velocity trade-off plot.

## Principles

**FK is composition of transforms.** Each joint $i$ contributes a transform; the end-effector pose is their product. In product-of-exponentials form (Lynch & Park, *Modern Robotics* ch. 4):

$$T(q) = e^{[\mathcal{S}_1]q_1} e^{[\mathcal{S}_2]q_2} \cdots e^{[\mathcal{S}_n]q_n} M$$

where $\mathcal{S}_i$ are the joint screw axes in the home configuration and $M$ the home pose. Numerically you read the equivalent data (body offsets `pos`/`quat`, joint axes) out of the MJCF tree and compose $T_i = T_{i-1}\cdot T_{\text{offset},i}\cdot R_{\text{axis}_i}(q_i)$ down the chain — the same math without transcription errors. MuJoCo quaternions are **wxyz**; scipy's are xyzw. Every FK bug in this lesson is a frame-convention bug, and Exercise 3 makes you find one from its symptoms.

**The geometric Jacobian** maps joint velocities to end-effector spatial velocity, $\begin{bmatrix} v \\ \omega \end{bmatrix} = J(q)\,\dot q$. For revolute joint $i$ with world-frame axis $\hat\omega_i$ through point $p_i$, the $i$-th column is (Lynch & Park ch. 5):

$$J_i = \begin{bmatrix} \hat\omega_i \times (p_e - p_i) \\ \hat\omega_i \end{bmatrix}$$

Column $i$ is the end-effector velocity you get from moving joint $i$ alone at unit rate — so you can predict which columns shrink when two axes align.

**Numerical IK is root-finding on FK.** Gauss-Newton iterates $q \leftarrow q + J^{+}(q)\,\Delta p$ with $\Delta p = p^{*} - \mathrm{FK}(q)$ (Lynch & Park ch. 6). The pseudo-inverse $J^{+} = J^\top(JJ^\top)^{-1}$ explodes when $JJ^\top$ loses rank. Damped least squares replaces it with

$$\dot q = J^\top \left( J J^\top + \lambda^2 I \right)^{-1} \Delta p,$$

which solves $\min_{\dot q} \lVert J\dot q - \Delta p \rVert^2 + \lambda^2 \lVert \dot q \rVert^2$ — bounded velocities everywhere, at the price of a bias that scales with $\lambda$.

**Singularities in one picture.** SVD $J = U\Sigma V^\top$: the smallest singular value $\sigma_{\min}$ is the gain of the hardest task direction. $J^{+}$ multiplies by $1/\sigma_{\min}$ (→ ∞ at singularity); DLS multiplies by $\sigma_i/(\sigma_i^2 + \lambda^2)$, which peaks at $1/2\lambda$. The condition number $\kappa = \sigma_{\max}/\sigma_{\min}$ maps manipulability across the workspace. Two singularity types to expect on this arm: the *boundary* one (arm fully extended, $\sigma_{\min} \to 0$ along the reach direction) and the *interior* one (two joint axes aligned — read the axis inventory in Exercise 0 to predict which pair).

**Carry forward**

- FK = compose body offsets and joint rotations down the MJCF chain; validate against `mj_forward` to 1e-10, and an error of 1e-3 is a convention bug, not numerics.
- Jacobian column $i$ = $[\hat\omega_i \times (p_e - p_i);\ \hat\omega_i]$; validate against *both* `mj_jacSite` and finite differences, because a shared frame mistake cancels in one check but not the other.
- DLS caps the velocity gain at $1/2\lambda$ and buys it with a tracking bias $\propto \lambda$; the $\lambda$-tradeoff plot is how you pick it.
- A 5-DOF arm reaches a 5-dim slice of SE(3): pose IK is generically infeasible, position IK is not.

| Source | Read for |
|---|---|
| Tutorial §2.3 | the planar 2-DOF worked example you'll reproduce in Exercise 1; IK-as-optimization framing |
| Lynch & Park ch. 4–6 (free PDF) | PoE FK, the Jacobian column formula, Newton–Raphson IK — the SE(3) treatment the tutorial skips |
| MIT *Robotic Manipulation* ch. 3 | the pick-and-place framing of differential IK you'll build on in Lesson 04 |
| MuJoCo docs: `mj_forward`, `mj_jacSite`, named access | the validation API surface (Exercises 2–3) |

## Exercise 0 — Inventory the model [Read]

Tests the prediction half of objective 4: know the axes before reimplementing anything.

1. Load `SO-ARM100/Simulation/SO101/so101_new_calib.xml` with `mujoco.MjModel.from_xml_path`. This calibration zeroes each joint mid-range, matching LeRobot's convention.
2. Print, per joint: `model.joint(name)` → axis, range (radians), parent body; `model.nq`. Identify or add a **site** at the gripper (if the model lacks one, add `<site name="ee_site" .../>` to the wrist body in a scene wrapper file — don't edit the upstream XML).
3. Open the scene in the viewer (`python -m mujoco.viewer`; see Lesson 00's mjpython pitfalls), drag each joint through its range, and **write in `RESULTS.md`** which pair of axes can align and at what configuration — that is your interior-singularity prediction for Exercise 6.

**✅ Checkpoint:** a 6-row table of joints with axes and ranges in radians; `data.site("ee_site").xpos` readable after `mj_forward`; the axis-alignment prediction is written.

## Exercise 1 — The planar warm-up [Derive]

Tests objective 1: the only place you get *exact* answers.

1. On paper: analytic FK for the 2-DOF planar arm (`shoulder_lift` + `elbow_flex` in the sagittal plane), $x = l_1 \cos q_1 + l_2\cos(q_1{+}q_2)$, $y = l_1 \sin q_1 + l_2 \sin(q_1{+}q_2)$, with $l_1, l_2$ read from the MJCF body offsets (not meshes). Both IK branches: $\cos q_2 = \frac{x^2+y^2-l_1^2-l_2^2}{2 l_1 l_2}$, elbow-up/-down via $\pm$ on $q_2$, then $q_1$ by `atan2` correction. Reachability: $|l_1 - l_2| \le \lVert p \rVert \le l_1 + l_2$. Derive the elbow-down branch from elbow-up without redoing the algebra.
2. [Build] `planar.py` (~30 lines from your formulas): `fk2(q)`, `ik2(p, branch)`, and a round-trip check on 1,000 uniformly sampled reachable targets, both branches, plus correct rejection of unreachable targets. Plot the reachable annulus colored by which branches are inside joint limits.

**✅ Checkpoint:** round-trip error < 1e-9 on both branches; the annulus shows a limit-clipped region.

## Exercise 2 — `fk` and `jacobian`, validated twice [Build]

Tests the composition principle and produces the module Lessons 04/07 and H1 import. Spec for `kinematics.py`:

- `build_chain(model)`: walk the MJCF tree once, recording per body from base to gripper (parent, body-frame offset `pos`, orientation `quat`, joint axis) as a flat chain.
- `fk(q) -> (p_ee, R_ee)`: compose $T_{i} = T_{i-1} \cdot T_{\text{offset},i} \cdot R_{\text{axis}_i}(q_i)$ down the chain to `ee_site`. Pure numpy in the math path; MuJoCo only in `build_chain`.
- `jacobian(q) -> J (6×5)`: the cross-product column formula using world-frame axes and positions from `fk`'s own intermediates.
- The check (`check_kinematics.py`, printed numbers): over 1,000 random configurations inside joint limits, `fk` vs `mj_forward` + `data.site("ee_site").xpos`/`.xmat` — max position error < 1e-10 m, max rotation error < 1e-10 (Frobenius); `jacobian` vs `mujoco.mj_jacSite(model, data, jacp, jacr, site_id)` — max abs diff < 1e-8; `jacobian` vs central finite differences of your own `fk` ($h = 10^{-6}$) — max abs diff < 1e-5.

Before running the check, write down which of the two Jacobian validations would *still pass* if `fk` had a wrong-but-self-consistent frame convention, and why.

**✅ Checkpoint:** all three validations pass at tolerance. (Anything at 1e-3–1e-2 is a convention bug — go to Exercise 3 early.)

## Exercise 3 — Plant a convention bug [Diagnose]

Tests objective 2.

1. Have your AI tool make a copy `fk_bugged` in which one body's quaternion is interpreted as xyzw instead of wxyz (don't look at which body).
2. **Predict, in writing:** the error will be zero for which configurations, nonzero for which, and of what magnitude (order 1e-3? order 0.1 m?). Which joint's motion will the error track?
3. Run the Exercise 2 validation against `fk_bugged`; tabulate max error as a function of each joint swept alone with the others at zero. Locate the bugged body from the table alone, then confirm by reading the copy.

**✅ Checkpoint:** you named the bugged body before reading the code; `RESULTS.md` has the per-joint error table and the mechanism (a rotation misread as a different rotation, so the error is configuration-dependent and appears only downstream of that body).

## Exercise 4 — IK at the boundary [Predict → Run]

Tests objective 3. Spec for `ik(p_target, q0, method) -> (q, converged, iters)` in `kinematics.py`: position targets; `"gn"` = Gauss-Newton with $J_v^{+}$ (position rows only), `"dls"` = damped least squares with fixed $\lambda$; iterate until $\lVert \Delta p \rVert < 10^{-5}$ m or 200 iterations; clip each update into `model.jnt_range`; up to 3 random restarts on failure.

1. **Write first:** for 50 targets scaled to 1.05× the workspace (just outside), which method's per-iteration $\lVert \dot q \rVert$ spikes, by what factor relative to its in-workspace median, and where each method *terminates*.
2. Run: 100 reachable targets (rejection-sampled via `fk` of random $q$) and 50 out-of-workspace targets, both methods. Report success rate, iteration histogram, max $\lVert \dot q \rVert$ per solve.
3. Reconcile against the $1/\sigma_{\min}$ vs $\sigma/(\sigma^2+\lambda^2)$ picture in Principles.

**✅ Checkpoint:** DLS ≥ 99% on reachable targets; on out-of-workspace targets DLS terminates at the boundary with bounded $\lVert \dot q \rVert$ while Gauss-Newton's spikes > 100× its median.

## Exercise 5 — Singularity atlas and the $\lambda$ trade [Predict → Run]

Tests objective 4 and produces the plot objective 5 reads.

1. **Write first:** on a (`shoulder_lift` × `elbow_flex`) grid with the other joints at zero, sketch where $\sigma_{\min}(J_v)$ is smallest — the fully-extended boundary and the interior locus you predicted in Exercise 0.
2. Render heatmaps of $\sigma_{\min}(J_v)$ and $\kappa(J_v)$ over the grid. Mark your predicted loci.
3. Near a singular configuration, command a fixed task-space step $\Delta p$ toward the lost direction; plot $\lVert \dot q \rVert$ vs distance-to-singularity for $J^{+}$ and DLS at $\lambda \in \{10^{-3}, 10^{-2}, 10^{-1}\}$.
4. The trade plot: tracking bias $\lVert J\dot q - \Delta p \rVert$ vs $\lVert \dot q \rVert$ as $\lambda$ sweeps, one curve per distance-to-singularity.

**✅ Checkpoint:** ≥ 2 distinct low-$\sigma_{\min}$ loci, at least one where you predicted; the velocity plot shows the $1/2\lambda$ ceiling.

## Exercise 6 — Pick $\lambda$ for H1 [Decide]

H1 runs Lesson 04's controller on the real arm at 50 Hz with a software joint-speed cap. From the Exercise 5 trade plot, choose $\lambda$ and defend it in `RESULTS.md`: the worst-case velocity amplification you accepted, the tracking bias you paid at the workspace center and near the interior singularity, and what you'd change if the speed cap halved.

**✅ Checkpoint:** the decision names a number and the two rows of the plot that justify it.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `kinematics.py` | **Interface contract, used by Lessons 04/07 and H1:** `fk(q) -> (p_ee, R_ee)` (numpy, radians) and `jacobian(q) -> (6×5)`; plus `ik(p_target, q0, method)`. No MuJoCo calls in the math paths |
| `planar.py`, `check_kinematics.py` | round-trip < 1e-9; FK < 1e-10 m; Jacobian < 1e-8 vs MuJoCo and < 1e-5 vs FD — all printed by one command |
| `plots/` | annulus, $\sigma_{\min}$/$\kappa$ heatmaps, velocity-vs-distance, $\lambda$-trade |
| `RESULTS.md` | Exercise 0 axis prediction; Exercise 3 per-joint error table + mechanism; Exercise 4/5 predictions and reconciliations; Exercise 6 decision |

## Done when

- [ ] All three `check_kinematics.py` validations pass at tolerance.
- [ ] The planted convention bug was located from the error signature before reading the code.
- [ ] DLS ≥ 99% on reachable targets and bounded at the boundary; Gauss-Newton's spike quantified.
- [ ] The heatmap shows a singularity where Exercise 0 predicted it; $\lambda$ for H1 is chosen and defended.

## Self-check

1. Why can't the SO-101 track an arbitrary SE(3) trajectory, and what dimension is the set of poses it *can* reach?
2. Derive the elbow-down branch from the elbow-up one without redoing the algebra.
3. What does $\sigma_{\min}(J)$ mean operationally — in units, for this arm?
4. DLS with $\lambda = 0.01$: what is the worst-case amplification of $\Delta p$ into $\dot q$, and at which $\sigma$ does it occur?
5. Why validate the Jacobian against finite differences *and* MuJoCo, when either alone would "pass"?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| FK off by a constant offset | forgot a body's `pos`/`quat` between joint frames, or measured $l_i$ from visual geometry instead of body offsets | read offsets from `model.body(name)`, never from meshes |
| FK wrong only when wrist moves | MuJoCo quaternions are **wxyz**; scipy's are xyzw | convert explicitly at the boundary, test a 90° rotation (Exercise 3 is this bug on purpose) |
| `site_xpos` stale/zero | read before `mj_forward(model, data)` | always forward after setting `qpos` |
| Jacobian matches FD but not MuJoCo | your site vs MuJoCo's body-frame Jacobian (`mj_jacBody`) | use `mj_jacSite` on the same site your FK targets |
| IK "converges" outside joint limits | unclipped Newton steps | clip into `model.jnt_range` every iteration; count clipped iterations |
| Angles look 50× too big | mixing LeRobot's normalized/degree action space with MJCF radians | this library is radians-only; conversion happens at the robot boundary (H1) |
| Menagerie model disagrees with SO-ARM100 repo | `trs_so_arm100` (menagerie) vs `so101_new_calib.xml` differ in calibration zero | this course standardizes on `so101_new_calib.xml`; note the offset if you compare |

## Going deeper

- **Constrained IK.** Position IK with a spherical keep-out constraint via `scipy.optimize.minimize` (SLSQP), reproducing the tutorial's feasible-set idea (its Fig. 6): a target whose unconstrained solution violates the sphere and whose constrained solution routes the elbow around it, keep-out satisfied to 1e-6.
- **4-DOF IK.** Position + wrist pitch (a 4th task row); note where the extra row costs convergence.
- **PoE properly.** Extract screw axes $\mathcal{S}_i$ from the home configuration, implement the matrix exponential for twists, and show agreement with chain FK to 1e-12; then derive the body Jacobian and reconcile it with the geometric one via the adjoint (Lynch & Park ch. 5.2).

## References

- Lynch & Park, *Modern Robotics: Mechanics, Planning, and Control*, ch. 4–6. Free PDF at modernrobotics.org.
- Tedrake, *Robotic Manipulation*, ch. 3 (Basic Pick and Place). manipulation.csail.mit.edu.
- LeRobot team, *Robot Learning: A Tutorial*, §2.3. arXiv:2510.12403.
- MuJoCo documentation: computation chapter (`mj_forward`, `mj_jacSite`), named access API.
- TheRobotStudio SO-ARM100 repo, `Simulation/SO101/so101_new_calib.xml`.
