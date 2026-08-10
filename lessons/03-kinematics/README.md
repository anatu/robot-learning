# Lesson 03 — Kinematics From Scratch

Build the FK/Jacobian/IK library for the SO-101 with your own hands, validate every piece against MuJoCo to numerical precision, and map where the arm's geometry breaks the naive math. The tutorial's §2 does a 2-DOF planar toy; this lesson does the real arm the way CS223A/MIT would.

| | |
|---|---|
| **Phase** | 2 — Classical core |
| **Time** | ~2 sessions (8–10 h desk time), all Mac-local |
| **Cost** | $0 |
| **Prerequisites** | 00 (MuJoCo + `mjpython` working, SO-ARM100 repo cloned) |
| **Feeds into** | 04 (the Jacobian and FK feed the diff-IK tracker), 07 (collision-checked planning uses this FK), H1 (the same library drives the real arm) |

**A correction to the scaffold:** the SO-101 is a **5-DOF arm + 1-DOF gripper** (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper` in `so101_new_calib.xml`), not 6-DOF. That's not a detail — with 5 joints, arbitrary SE(3) pose targets are *generically infeasible* (you can hit a 5-dim submanifold of the 6-dim pose space). Everything below therefore poses IK as **position (3-DOF) or position + wrist-pitch (4-DOF)** tasks, and Part 5 makes the infeasibility empirical.

## Learning objectives

After this lesson you can:

1. **Derive** the planar 2-link FK and both analytic IK branches, and state the reachability condition.
2. **Implement** numerical FK for the full SO-101 directly from the MJCF kinematic tree and validate it against MuJoCo to < 1e-10 m.
3. **Construct** the geometric Jacobian column-by-column from joint axes and verify it against both `mj_jacSite` and finite differences.
4. **Implement** Gauss-Newton and damped-least-squares IK, and quantify their convergence over the reachable workspace.
5. **Diagnose** singularities via the SVD of $J$ and explain, with your own condition-number maps, why damping trades tracking bias for bounded joint velocities.

## Background

**FK is composition of transforms.** Each joint $i$ contributes a transform; the end-effector pose is their product. In product-of-exponentials form (Lynch & Park, *Modern Robotics* ch. 4):

$$T(q) = e^{[\mathcal{S}_1]q_1} e^{[\mathcal{S}_2]q_2} \cdots e^{[\mathcal{S}_n]q_n} M$$

where $\mathcal{S}_i$ are the joint screw axes in the home configuration and $M$ the home pose. You won't hand-derive screws for the SO-101 — you'll read the equivalent data (body offsets, joint axes) out of the MJCF tree and compose transforms numerically, which is the same math without the transcription errors.

**The geometric Jacobian** maps joint velocities to end-effector spatial velocity, $\begin{bmatrix} v \\ \omega \end{bmatrix} = J(q)\,\dot q$. For revolute joint $i$ with world-frame axis $\hat\omega_i$ through point $p_i$, the $i$-th column is (Lynch & Park ch. 5):

$$J_i = \begin{bmatrix} \hat\omega_i \times (p_e - p_i) \\ \hat\omega_i \end{bmatrix}$$

**Numerical IK is root-finding on FK.** Gauss-Newton iterates $q \leftarrow q + J^{+}(q)\,\Delta p$ with $\Delta p = p^{*} - \mathrm{FK}(q)$ (Lynch & Park ch. 6). The pseudo-inverse $J^{+} = J^\top(JJ^\top)^{-1}$ explodes when $JJ^\top$ loses rank. Damped least squares replaces it with

$$\dot q = J^\top \left( J J^\top + \lambda^2 I \right)^{-1} \Delta p,$$

which solves $\min_{\dot q} \lVert J\dot q - \Delta p \rVert^2 + \lambda^2 \lVert \dot q \rVert^2$ — bounded velocities everywhere, at the price of a bias that scales with $\lambda$.

**Singularities in one picture.** SVD $J = U\Sigma V^\top$: the smallest singular value $\sigma_{\min}$ is the gain of the hardest task direction. $J^{+}$ multiplies by $1/\sigma_{\min}$ (→ ∞ at singularity); DLS multiplies by $\sigma_i/(\sigma_i^2 + \lambda^2)$, which peaks at $1/2\lambda$. The condition number $\kappa = \sigma_{\max}/\sigma_{\min}$ maps manipulability across the workspace.

| Source | Read for |
|---|---|
| Tutorial §2.3 | the planar 2-DOF worked example you'll reproduce in Part 1; IK-as-optimization framing |
| Lynch & Park ch. 4–6 (free PDF) | PoE FK, the Jacobian column formula, Newton–Raphson IK — the SE(3) treatment the tutorial skips |
| MIT *Robotic Manipulation* ch. 3 | the pick-and-place framing of differential IK you'll build on in Lesson 04 |
| MuJoCo docs: `mj_forward`, `mj_jacSite`, named access | the validation API surface (Part 2–3) |

## Part 0 — Load the model and take inventory (~30 min)

Ground truth first: know exactly what the model contains before reimplementing it.

1. Load `SO-ARM100/Simulation/SO101/so101_new_calib.xml` with `mujoco.MjModel.from_xml_path`. This calibration zeroes each joint mid-range, matching LeRobot's convention.
2. Print the inventory: for each joint, `model.joint(name)` → axis, range (radians), parent body; count `model.nq`. Identify or add a **site** at the gripper (if the model lacks one, add `<site name="ee_site" .../>` to the wrist body in a scene wrapper file — don't edit the upstream XML).
3. Open the scene in `mjpython -m mujoco.viewer`, drag each joint through its range, and note which pairs of axes can align (that's where Part 5 will find singularities).

**✅ Checkpoint:** a printed table of 6 joints with axes and ranges in radians; an `ee_site` whose world position you can read from `data.site("ee_site").xpos` after `mj_forward`.

## Part 1 — The planar warm-up, analytically (~1.5 h)

Reproduce the tutorial's 2-DOF planar arm (shoulder_lift + elbow_flex in the sagittal plane) where everything is closed-form — your only chance to test against *exact* answers.

1. Analytic FK: $x = l_1 \cos q_1 + l_2\cos(q_1{+}q_2)$, $y = l_1 \sin q_1 + l_2 \sin(q_1{+}q_2)$, with $l_1, l_2$ measured from the MJCF body offsets.
2. Analytic IK, both branches: $\cos q_2 = \frac{x^2+y^2-l_1^2-l_2^2}{2 l_1 l_2}$, elbow-up/-down via $\pm$ on $q_2$, then $q_1$ by `atan2` correction. Reachability: $|l_1 - l_2| \le \lVert p \rVert \le l_1 + l_2$.
3. Workspace plot: the reachable annulus, colored by which branches are in joint limits.
4. Tests: `FK(IK_branch(p)) == p` to 1e-9 for 1,000 uniformly sampled reachable targets, both branches; IK correctly reports unreachable targets.

**✅ Checkpoint:** both round-trip tests green at 1e-9; the annulus plot shows a limit-clipped region (the SO-101's joint ranges cut the full annulus).

## Part 2 — Numerical FK for the full arm (~1.5 h)

Your own FK, validated against MuJoCo's — this is the reference implementation everything downstream trusts.

1. Walk the MJCF tree once at load: for each body from base to gripper record (parent, body-frame offset `pos`, orientation `quat`, joint axis). Store as a flat chain.
2. `fk(q) -> (p_ee, R_ee)`: compose $T_{i} = T_{i-1} \cdot T_{\text{offset},i} \cdot R_{\text{axis}_i}(q_i)$ down the chain to the `ee_site`. Pure numpy, no MuJoCo calls.
3. Validate: 1,000 random configurations sampled inside joint limits; compare against `mj_forward` + `data.site("ee_site").xpos` / `.xmat`.

**✅ Checkpoint:** max position error < 1e-10 m, max rotation error < 1e-10 (Frobenius) over all 1,000 configs. Anything at 1e-3–1e-2 means a frame-convention bug (see Pitfalls), not "numerical error."

## Part 3 — The geometric Jacobian (~1 h)

1. `jacobian(q) -> J (6×5)`: the cross-product column formula above, using world-frame joint axes and positions from your own FK intermediates (not MuJoCo's).
2. Validate twice, independently:
   - against MuJoCo: `mujoco.mj_jacSite(model, data, jacp, jacr, site_id)` after `mj_forward`; max abs diff < 1e-8;
   - against finite differences of your own FK (central differences, $h = 10^{-6}$); max abs diff < 1e-5.
3. Plot $\sigma_{\min}(J)$ along one joint's sweep with the others fixed — your first look at a singularity approaching.

**✅ Checkpoint:** both validations pass at their tolerances. (The FD check catches errors the MuJoCo check can't — if you validated only against `mj_jacSite`, a shared frame-convention mistake could cancel.)

## Part 4 — Numerical IK (~2 h)

1. `ik(p_target, q0, method) -> (q, converged, iters)` for position targets: Gauss-Newton with $J_v^{+}$ (position rows only), and DLS with fixed $\lambda$. Iterate until $\lVert \Delta p \rVert < 10^{-5}$ m or 200 iterations; clip each update into joint limits.
2. Convergence study: 500 targets sampled from the reachable workspace (rejection-sample via FK of random $q$), random restarts on failure. Report: success rate, iteration histogram, wall-clock per solve — per method.
3. Boundary behavior: 100 targets just *outside* the workspace (scale reachable points by 1.05). Gauss-Newton should oscillate/diverge; DLS should converge to the closest reachable point. Show it.
4. 4-DOF variant: position + wrist pitch (stack a 4th task row). Note where the extra row costs you convergence.

**✅ Checkpoint:** DLS ≥ 99% success on reachable targets (with ≤ 3 restarts); on out-of-workspace targets DLS terminates at the boundary with bounded $\lVert \dot q \rVert$ while pure Gauss-Newton's $\lVert \dot q \rVert$ spikes > 100× median.

## Part 5 — Singularity atlas (~2 h)

Make the abstract concrete: where is this specific arm nearly uncontrollable, and what does damping actually buy?

1. Grid-sweep (`shoulder_lift` × `elbow_flex`) with other joints at candidate values; heatmap $\sigma_{\min}(J_v)$ and $\kappa(J_v)$ over the grid. Identify the arm-fully-extended boundary singularity and the wrist-aligned interior one you predicted in Part 0.
2. Near a singular configuration, command a fixed task-space step $\Delta p$ toward the lost direction; plot $\lVert \dot q \rVert$ vs distance-to-singularity for $J^{+}$ vs DLS at $\lambda \in \{10^{-3}, 10^{-2}, 10^{-1}\}$.
3. The tradeoff plot: tracking bias $\lVert J\dot q - \Delta p \rVert$ vs $\lVert \dot q \rVert$ as $\lambda$ sweeps — one curve per distance-to-singularity. This plot *is* the answer to "how do I pick λ."
4. Constrained IK finale: position IK with a spherical keep-out constraint via `scipy.optimize.minimize` (SLSQP), reproducing the tutorial's feasible-set idea (its Fig. 6). Show a target whose unconstrained solution violates the sphere and whose constrained solution routes the elbow around it.

**✅ Checkpoint:** the heatmap shows ≥ 2 distinct low-$\sigma_{\min}$ loci; the $\lambda$-tradeoff plot shows the $1/2\lambda$ velocity ceiling; the constrained solve satisfies the keep-out to 1e-6.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `kinematics.py` | `fk`, `jacobian`, `ik` with the signatures above; no MuJoCo imports in the math paths; this exact module is imported by Lessons 04/07 and H1 |
| `tests/` | Parts 1–4 checkpoints as `pytest` (round-trip, FK parity, double Jacobian validation, IK convergence stats) |
| `notebook.ipynb` | workspace annulus, singularity heatmaps, an animated IK convergence + one animated near-singularity trajectory |
| `RESULTS.md` | the λ-tradeoff reading; where the SO-101's singularities live in joint space; the 5-DOF feasibility note with the Part 4.4 evidence |

## Done when

- [ ] FK matches MuJoCo < 1e-10 m over 1,000 random configs.
- [ ] Jacobian passes both independent validations.
- [ ] DLS-IK ≥ 99% convergence on reachable targets, graceful at the boundary.
- [ ] Singularity heatmaps + λ-tradeoff plot exist and are interpreted in `RESULTS.md`.
- [ ] `pytest` green from a clean clone.

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
| FK wrong only when wrist moves | MuJoCo quaternions are **wxyz**; scipy's are xyzw | convert explicitly at the boundary, test a 90° rotation |
| `site_xpos` stale/zero | read before `mj_forward(model, data)` | always forward after setting `qpos` |
| Jacobian matches FD but not MuJoCo | your site vs MuJoCo's body-frame Jacobian (`mj_jacBody`) | use `mj_jacSite` on the same site your FK targets |
| IK "converges" outside joint limits | unclipped Newton steps | clip into `model.jnt_range` every iteration; count clipped iterations |
| Angles look 50× too big | mixing LeRobot's normalized/degree action space with MJCF radians | this library is radians-only; conversion happens at the robot boundary (H1) |
| Menagerie model disagrees with SO-ARM100 repo | `trs_so_arm100` (menagerie) vs `so101_new_calib.xml` differ in calibration zero | this course standardizes on `so101_new_calib.xml`; note the offset if you compare |

## Stretch

Implement PoE FK properly: extract screw axes $\mathcal{S}_i$ from the home configuration, implement the matrix exponential for twists, and show it agrees with your chain FK to 1e-12. Then derive the analytic *body* Jacobian and reconcile it with the geometric one via the adjoint map (Lynch & Park ch. 5.2).

## References

- Lynch & Park, *Modern Robotics: Mechanics, Planning, and Control*, ch. 4–6. Free PDF at modernrobotics.org.
- Tedrake, *Robotic Manipulation*, ch. 3 (Basic Pick and Place). manipulation.csail.mit.edu.
- LeRobot team, *Robot Learning: A Tutorial*, §2.3. arXiv:2510.12403.
- MuJoCo documentation: computation chapter (`mj_forward`, `mj_jacSite`), named access API.
- TheRobotStudio SO-ARM100 repo, `Simulation/SO101/so101_new_calib.xml`.
