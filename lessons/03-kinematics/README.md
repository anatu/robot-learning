# Lesson 03 — Kinematics

This lesson builds the kinematics library for the SO-101 that Lessons 04 and 07 and the hardware track all import: forward kinematics, the geometric Jacobian, and numerical inverse kinematics, each validated against MuJoCo to numerical precision. The tutorial's §2 treats a two-link planar arm where everything is closed-form; this lesson does that warm-up and then moves to the real five-joint arm, where the instructive behaviour is where the naive mathematics breaks down: at singularities, at the workspace boundary, and in the fact that a five-joint arm cannot reach arbitrary poses. You finish by choosing the damping parameter that the real-arm controller in H1 uses, on the basis of a trade-off plot you generate yourself.

| | |
|---|---|
| **Phase** | 2 — Classical core |
| **Time** | 1–2 sessions (5–6 h desk time), all Mac-local |
| **Cost** | $0 |
| **Prerequisites** | 00 (MuJoCo working, SO-ARM100 repo cloned) |
| **Feeds into** | 04 (`fk` and `jacobian` feed the differential-IK tracker), 07 (collision-checked planning uses this FK), H1 (the same library scores the real arm's trace) |

A note on the arm's degrees of freedom. The SO-101 is a five-degree-of-freedom arm with a one-degree-of-freedom gripper (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper` in `so101_new_calib.xml`), not a six-degree-of-freedom arm. This matters. With five joints, an arbitrary pose in SE(3) is generically unreachable, because the arm can only reach a five-dimensional submanifold of the six-dimensional pose space. Every inverse-kinematics problem in this lesson is therefore posed as a position (3-DOF) task.

## Learning objectives

After this lesson you can:

1. **Derive** the planar two-link forward kinematics and both analytic inverse-kinematics branches, and state the reachability condition.
2. **Diagnose** a frame-convention bug in numerical forward kinematics from its error signature alone: which configurations it affects and at what magnitude.
3. **Predict** which inverse-kinematics method's joint velocity becomes unbounded at the workspace boundary, and explain why damping bounds it at the price of bias.
4. **Predict** where this arm's singularities lie from its joint-axis inventory, and confirm the prediction on a map of $\sigma_{\min}(J)$.
5. **Decide** a damping parameter $\lambda$ for a real controller from a plot of tracking bias against joint velocity.

## Principles

### Forward kinematics is a composition of transforms

Each joint contributes a rigid transform that depends on its angle, and the pose of the end-effector is the product of those transforms taken in order from the base. In the product-of-exponentials form of Lynch & Park (*Modern Robotics* ch. 4) this reads

$$T(q) = e^{[\mathcal{S}_1]q_1} e^{[\mathcal{S}_2]q_2} \cdots e^{[\mathcal{S}_n]q_n} M$$

where $\mathcal{S}_i$ are the joint screw axes in the home configuration and $M$ is the home pose. You will not transcribe screw axes by hand. Instead you read the equivalent data, namely each body's offset `pos` and orientation `quat` and each joint's axis, out of the MJCF tree, and compose $T_i = T_{i-1}\cdot T_{\text{offset},i}\cdot R_{\text{axis}_i}(q_i)$ down the chain. The mathematics is the same, and the transcription errors are avoided. One convention has to be fixed at the outset: MuJoCo stores quaternions in wxyz order, whereas scipy uses xyzw. Nearly every forward-kinematics bug in this lesson is a frame-convention bug of that kind, and Exercise 3 has you diagnose one from its symptoms.

### The geometric Jacobian

The geometric Jacobian maps joint velocities to the end-effector's spatial velocity, $\begin{bmatrix} v \\ \omega \end{bmatrix} = J(q)\,\dot q$. For a revolute joint $i$ with world-frame axis $\hat\omega_i$ passing through a point $p_i$, the $i$-th column is (Lynch & Park ch. 5)

$$J_i = \begin{bmatrix} \hat\omega_i \times (p_e - p_i) \\ \hat\omega_i \end{bmatrix}$$

Column $i$ is therefore the end-effector velocity produced by moving joint $i$ alone at unit rate. This reading of the columns is what lets you predict, before computing anything, which columns shrink when two joint axes align: their contributions become parallel, and the Jacobian loses rank.

### Numerical inverse kinematics is root-finding on forward kinematics

Given a target position $p^{*}$, numerical inverse kinematics looks for a configuration $q$ such that $\mathrm{FK}(q) = p^{*}$. Gauss-Newton iterates $q \leftarrow q + J^{+}(q)\,\Delta p$ with $\Delta p = p^{*} - \mathrm{FK}(q)$ (Lynch & Park ch. 6). The pseudo-inverse $J^{+} = J^\top(JJ^\top)^{-1}$ becomes unbounded when $JJ^\top$ loses rank, which is exactly what happens at a singularity. Damped least squares replaces it with

$$\dot q = J^\top \left( J J^\top + \lambda^2 I \right)^{-1} \Delta p,$$

which is the minimiser of $\lVert J\dot q - \Delta p \rVert^2 + \lambda^2 \lVert \dot q \rVert^2$. The damping term bounds the joint velocity everywhere, and the price is a bias in the solution that grows with $\lambda$.

### Singularities through the singular value decomposition

Write $J = U\Sigma V^\top$. The smallest singular value $\sigma_{\min}$ is the gain of the task direction in which the arm can move least easily. The pseudo-inverse multiplies that direction by $1/\sigma_{\min}$, which tends to infinity at a singularity; damped least squares multiplies by $\sigma_i/(\sigma_i^2 + \lambda^2)$, which has a maximum of $1/2\lambda$ at $\sigma_i = \lambda$. The condition number $\kappa = \sigma_{\max}/\sigma_{\min}$, plotted over the workspace, is a map of manipulability. On this arm you should expect two kinds of singularity: a boundary singularity when the arm is fully extended, where $\sigma_{\min} \to 0$ along the reach direction, and an interior singularity where two joint axes align. The axis inventory you take in Exercise 0 lets you predict which pair.

**Carry forward**

- Forward kinematics is the composition of body offsets and joint rotations down the MJCF chain. Validate it against `mj_forward` to 1e-10; an error of order 1e-3 indicates a frame-convention bug rather than numerical error, because composing exact transforms has no other source of error at that magnitude.
- Column $i$ of the geometric Jacobian is $[\hat\omega_i \times (p_e - p_i);\ \hat\omega_i]$. Validate it against both `mj_jacSite` and finite differences of your own forward kinematics, because a frame mistake shared by your FK and your Jacobian cancels in one of those checks but not in the other.
- Damped least squares caps the velocity gain at $1/2\lambda$ and pays for that cap with a tracking bias proportional to $\lambda$. The plot of bias against velocity as $\lambda$ varies is the basis for choosing $\lambda$.
- A five-joint arm reaches a five-dimensional slice of SE(3), so inverse kinematics for a full pose is generically infeasible while inverse kinematics for a position is not.

| Source | Read for |
|---|---|
| Tutorial §2.3 | the planar 2-DOF worked example you reproduce in Exercise 1, and the framing of IK as optimisation |
| Lynch & Park ch. 4–6 (free PDF) | product-of-exponentials FK, the Jacobian column formula, Newton–Raphson IK: the SE(3) treatment the tutorial skips |
| MIT *Robotic Manipulation* ch. 3 | the pick-and-place framing of differential IK that Lesson 04 builds on |
| MuJoCo docs: `mj_forward`, `mj_jacSite`, named access | the validation API surface used in Exercises 2–3 |

## Exercise 0 — Inventory the model [Read]

Before reimplementing anything, you record what the model contains: each joint's axis, range and parent body, and the location of the end-effector site. The inventory also lets you make the interior-singularity prediction that Exercise 5 tests, because two joint axes that can be brought into alignment are exactly where the Jacobian loses rank.

1. Load `SO-ARM100/Simulation/SO101/so101_new_calib.xml` with `mujoco.MjModel.from_xml_path`. This calibration zeroes each joint mid-range, matching LeRobot's convention.
2. Print, per joint, `model.joint(name)`'s axis, range (in radians) and parent body, and print `model.nq`. Identify or add a **site** at the gripper. If the model lacks one, add `<site name="ee_site" .../>` to the wrist body in a scene wrapper file rather than editing the upstream XML.
3. Open the scene in the viewer (`python -m mujoco.viewer`; see Lesson 00's mjpython pitfalls), drag each joint through its range, and write in `RESULTS.md` which pair of axes can be brought into alignment and at what configuration. That is your interior-singularity prediction for Exercise 5.

**✅ Checkpoint:** a 6-row table of joints with axes and ranges in radians; `data.site("ee_site").xpos` is readable after `mj_forward`; the axis-alignment prediction is written.

## Exercise 1 — The planar two-link arm [Derive]

The two-link planar arm is the one case in this lesson where every answer is closed-form, which makes it the place to check your understanding against exact results before moving to the full arm. You derive the forward kinematics and both inverse-kinematics branches on paper, then check the derivation with a round-trip test.

1. On paper, derive the analytic forward kinematics for the two-link planar arm formed by `shoulder_lift` and `elbow_flex` in the sagittal plane: $x = l_1 \cos q_1 + l_2\cos(q_1{+}q_2)$, $y = l_1 \sin q_1 + l_2 \sin(q_1{+}q_2)$, with $l_1, l_2$ read from the MJCF body offsets rather than from the meshes. Derive both inverse-kinematics branches: $\cos q_2 = \frac{x^2+y^2-l_1^2-l_2^2}{2 l_1 l_2}$, with elbow-up and elbow-down given by the sign of $q_2$, and then $q_1$ by an `atan2` correction. State the reachability condition $|l_1 - l_2| \le \lVert p \rVert \le l_1 + l_2$. Derive the elbow-down branch from the elbow-up one without redoing the algebra.
2. [Build] `planar.py`, about 30 lines from your formulas: `fk2(q)`, `ik2(p, branch)`, and a round-trip check on 1,000 uniformly sampled reachable targets for both branches, together with correct rejection of unreachable targets. Plot the reachable annulus, coloured by which branches lie inside the joint limits.

**✅ Checkpoint:** round-trip error below 1e-9 on both branches, and the annulus shows a region clipped by the joint limits.

## Exercise 2 — Implement and validate `fk` and `jacobian` [Build]

This exercise produces the module that Lessons 04 and 07 and H1 import. You specify numerical forward kinematics and the geometric Jacobian, have an AI tool draft them, and validate both against MuJoCo. The Jacobian is validated twice, against MuJoCo and against finite differences of your own forward kinematics, for a reason the exercise asks you to state before you run the check.

Write the specification for `kinematics.py` and have an AI tool draft it:

- `build_chain(model)`: walk the MJCF tree once and record, for each body from base to gripper, its parent, its body-frame offset `pos`, its orientation `quat`, and its joint axis, as a flat chain.
- `fk(q) -> (p_ee, R_ee)`: compose $T_{i} = T_{i-1} \cdot T_{\text{offset},i} \cdot R_{\text{axis}_i}(q_i)$ down the chain to `ee_site`. The math path is pure numpy; MuJoCo is used only in `build_chain`.
- `jacobian(q) -> J (6×5)`: the cross-product column formula, using world-frame axes and positions taken from `fk`'s own intermediate transforms.
- The check, in `check_kinematics.py`, prints its numbers. Over 1,000 random configurations inside the joint limits: `fk` against `mj_forward` followed by `data.site("ee_site").xpos` and `.xmat`, with maximum position error below 1e-10 m and maximum rotation error below 1e-10 (Frobenius); `jacobian` against `mujoco.mj_jacSite(model, data, jacp, jacr, site_id)`, with maximum absolute difference below 1e-8; and `jacobian` against central finite differences of your own `fk` with $h = 10^{-6}$, with maximum absolute difference below 1e-5.

Before running the check, write down which of the two Jacobian validations would still pass if `fk` had a wrong but self-consistent frame convention, and why.

**✅ Checkpoint:** all three validations pass at tolerance. An error of order 1e-3 to 1e-2 indicates a convention bug; if you see one, go to Exercise 3 early.

## Exercise 3 — Diagnose a planted convention bug [Diagnose]

A frame-convention bug produces a characteristic error signature: zero for some configurations, nonzero for others, and tracking the motion of one particular joint. In this exercise you have an AI tool plant such a bug in a copy of `fk` without telling you where, and you locate it from the signature alone.

1. Have your AI tool make a copy `fk_bugged` in which one body's quaternion is interpreted as xyzw instead of wxyz, without telling you which body.
2. Before running anything, predict in writing: for which configurations will the error be zero and for which nonzero; what order of magnitude will it have (1e-3, or 0.1 m); and which joint's motion will the error track?
3. Run the Exercise 2 validation against `fk_bugged`, and tabulate the maximum error as a function of each joint swept alone with the others at zero. Locate the bugged body from the table alone, then confirm by reading the copy.

**✅ Checkpoint:** you named the bugged body before reading the code, and `RESULTS.md` has the per-joint error table and the mechanism: a rotation misread as a different rotation, so the error depends on configuration and appears only downstream of that body.

## Exercise 4 — Inverse kinematics at the workspace boundary [Predict → Run]

Gauss-Newton and damped least squares behave equally well inside the workspace and very differently at its edge. This exercise implements both and compares them on targets just outside the reachable set, where the pseudo-inverse's $1/\sigma_{\min}$ behaviour and the damped solver's $1/2\lambda$ cap should be visible in the joint velocities.

Write the specification for `ik(p_target, q0, method) -> (q, converged, iters)` in `kinematics.py`: position targets only; `"gn"` is Gauss-Newton with the position-row pseudo-inverse $J_v^{+}$, and `"dls"` is damped least squares with a fixed $\lambda$; iterate until $\lVert \Delta p \rVert < 10^{-5}$ m or 200 iterations; clip each update into `model.jnt_range`; allow up to 3 random restarts on failure.

1. Before running, write down what you expect for 50 targets scaled to 1.05× the workspace (just outside it): which method's per-iteration $\lVert \dot q \rVert$ spikes, by roughly what factor relative to its in-workspace median, and where each method terminates.
2. Run 100 reachable targets (rejection-sampled by applying `fk` to random $q$) and 50 out-of-workspace targets with both methods. Report the success rate, an iteration histogram, and the maximum $\lVert \dot q \rVert$ per solve.
3. Reconcile the results against the $1/\sigma_{\min}$ versus $\sigma/(\sigma^2+\lambda^2)$ picture in the Principles section.

**✅ Checkpoint:** damped least squares succeeds on at least 99% of reachable targets; on out-of-workspace targets it terminates at the boundary with bounded $\lVert \dot q \rVert$, while Gauss-Newton's $\lVert \dot q \rVert$ spikes to more than 100× its median.

## Exercise 5 — The singularity atlas and the λ trade-off [Predict → Run]

Here you map $\sigma_{\min}$ and the condition number over a two-joint grid, confirm the singularity you predicted in Exercise 0, and then generate the plot of tracking bias against joint velocity as $\lambda$ varies. That plot is what Exercise 6 reads.

1. Before running, sketch where you expect $\sigma_{\min}(J_v)$ to be smallest on a (`shoulder_lift` × `elbow_flex`) grid with the other joints at zero: the fully extended boundary, and the interior locus you predicted in Exercise 0.
2. Render heatmaps of $\sigma_{\min}(J_v)$ and $\kappa(J_v)$ over the grid, and mark your predicted loci on them.
3. Near a singular configuration, command a fixed task-space step $\Delta p$ toward the lost direction, and plot $\lVert \dot q \rVert$ against distance to the singularity for $J^{+}$ and for damped least squares at $\lambda \in \{10^{-3}, 10^{-2}, 10^{-1}\}$.
4. Produce the trade-off plot: tracking bias $\lVert J\dot q - \Delta p \rVert$ against $\lVert \dot q \rVert$ as $\lambda$ sweeps, with one curve per distance to the singularity.

**✅ Checkpoint:** at least two distinct low-$\sigma_{\min}$ loci, at least one of them where you predicted; the velocity plot shows the $1/2\lambda$ ceiling.

## Exercise 6 — Choose λ for the real-arm controller [Decide]

H1 runs Lesson 04's controller on the physical arm at 50 Hz with a software cap on joint speed. Using the trade-off plot from Exercise 5, choose the damping $\lambda$ that controller will use and defend the choice in `RESULTS.md`: state the worst-case velocity amplification you accepted, the tracking bias you pay at the workspace centre and near the interior singularity, and what you would change if the speed cap were halved.

**✅ Checkpoint:** the decision names a number and the two rows of the plot that justify it.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `kinematics.py` | **Interface contract, used by Lessons 04/07 and H1:** `fk(q) -> (p_ee, R_ee)` (numpy, radians) and `jacobian(q) -> (6×5)`; plus `ik(p_target, q0, method)`. No MuJoCo calls in the math paths |
| `planar.py`, `check_kinematics.py` | round-trip < 1e-9; FK < 1e-10 m; Jacobian < 1e-8 vs MuJoCo and < 1e-5 vs FD, all printed by one command |
| `plots/` | annulus, $\sigma_{\min}$/$\kappa$ heatmaps, velocity-vs-distance, $\lambda$ trade-off |
| `RESULTS.md` | Exercise 0 axis prediction; Exercise 3 per-joint error table + mechanism; Exercise 4/5 predictions and reconciliations; Exercise 6 decision |

## Done when

- [ ] All three `check_kinematics.py` validations pass at tolerance.
- [ ] The planted convention bug was located from the error signature before reading the code.
- [ ] Damped least squares succeeds on at least 99% of reachable targets and stays bounded at the boundary; Gauss-Newton's spike is quantified.
- [ ] The heatmap shows a singularity where Exercise 0 predicted it, and $\lambda$ for H1 is chosen and defended.

## Self-check

1. Why can't the SO-101 track an arbitrary SE(3) trajectory, and what dimension is the set of poses it *can* reach?
2. Derive the elbow-down branch from the elbow-up one without redoing the algebra.
3. What does $\sigma_{\min}(J)$ mean operationally, in units, for this arm?
4. With damped least squares at $\lambda = 0.01$, what is the worst-case amplification of $\Delta p$ into $\dot q$, and at which $\sigma$ does it occur?
5. Why validate the Jacobian against finite differences *and* MuJoCo, when either alone would pass?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| FK off by a constant offset | a body's `pos`/`quat` between joint frames forgotten, or $l_i$ measured from visual geometry instead of body offsets | read offsets from `model.body(name)`, never from meshes |
| FK wrong only when the wrist moves | MuJoCo quaternions are **wxyz**; scipy's are xyzw | convert explicitly at the boundary and test a 90° rotation (Exercise 3 plants this bug deliberately) |
| `site_xpos` stale or zero | read before `mj_forward(model, data)` | always call `mj_forward` after setting `qpos` |
| Jacobian matches FD but not MuJoCo | your site vs MuJoCo's body-frame Jacobian (`mj_jacBody`) | use `mj_jacSite` on the same site your FK targets |
| IK "converges" outside joint limits | unclipped Newton steps | clip into `model.jnt_range` every iteration; count clipped iterations |
| Angles look 50× too big | mixing LeRobot's normalized/degree action space with MJCF radians | this library is radians-only; conversion happens at the robot boundary (H1) |
| Menagerie model disagrees with SO-ARM100 repo | `trs_so_arm100` (menagerie) vs `so101_new_calib.xml` differ in calibration zero | this course standardizes on `so101_new_calib.xml`; note the offset if you compare |

## Going deeper

- **Constrained IK.** Position IK with a spherical keep-out constraint via `scipy.optimize.minimize` (SLSQP), reproducing the tutorial's feasible-set idea (its Fig. 6): a target whose unconstrained solution violates the sphere and whose constrained solution routes the elbow around it, with the keep-out satisfied to 1e-6.
- **4-DOF IK.** Position plus wrist pitch, as a fourth task row; note where the extra row costs convergence.
- **Product of exponentials properly.** Extract the screw axes $\mathcal{S}_i$ from the home configuration, implement the matrix exponential for twists, and show agreement with the chain FK to 1e-12; then derive the body Jacobian and reconcile it with the geometric one via the adjoint (Lynch & Park ch. 5.2).

## References

- Lynch & Park, *Modern Robotics: Mechanics, Planning, and Control*, ch. 4–6. Free PDF at modernrobotics.org.
- Tedrake, *Robotic Manipulation*, ch. 3 (Basic Pick and Place). manipulation.csail.mit.edu.
- LeRobot team, *Robot Learning: A Tutorial*, §2.3. arXiv:2510.12403.
- MuJoCo documentation: computation chapter (`mj_forward`, `mj_jacSite`), named access API.
- TheRobotStudio SO-ARM100 repo, `Simulation/SO101/so101_new_calib.xml`.
