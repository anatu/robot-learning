# Lesson 04 — Differential IK as Optimization + Feedback

Feedback covers for model error — up to a point. Turn Lesson 03's static IK into a running tracker, watch open loop drift and a proportional term fix it, break it with model mismatch, then rebuild the tracker as a constrained QP the way MIT's manipulation course does. This is the controller H1 runs on the physical arm.

| | |
|---|---|
| **Phase** | 2 — Classical core |
| **Time** | 1 session (3–4 h desk time), all Mac-local |
| **Cost** | $0 |
| **Prerequisites** | 03 (`kinematics.py`: `fk`, `jacobian`) |
| **Feeds into** | 07 (executes planned paths), H1 (this exact controller traces on the real arm), and conceptually 08+ (this is the "brittle pipeline" RL and BC replace) |

## Learning objectives

After this lesson you can:

1. **Predict** how open-loop task-space tracking drifts with a perfect model and why it can never recover an initial offset.
2. **Explain** where the practical proportional-gain ceiling comes from, predict it from the control period, and account for the gap to what you measure.
3. **Quantify** the closed-loop advantage under model mismatch and state where feedback stops helping.
4. **Explain** why a constrained QP beats a clipped pseudo-inverse when a joint limit binds, and predict the shape of clipping's failure.
5. **Decide** the gain, damping, and limits you'd ship to a real arm, with the rows that justify them.

## Principles

**From IK to control.** A trajectory tracker doesn't re-solve full IK each tick; it maps desired task-space *velocity* to joint velocity through the Jacobian and integrates:

$$\dot q_k = J^{+}(q_k)\,\dot p^{*}_k, \qquad q_{k+1} = q_k + \dot q_k \,\Delta t .$$

This is **open-loop** in task space: integration error, model mismatch, and disturbances accumulate because nothing measures where the end-effector actually is. Two drift sources even with a perfect model: the explicit-Euler integrator, and the Jacobian being evaluated at the *current* $q$ while the step is finite.

**Closing the loop.** Add proportional feedback on task-space error $e_k = p^{*}_k - \mathrm{FK}(q_k^{\text{meas}})$:

$$\dot q_k = J^{+}(q_k)\left(\dot p^{*}_k + K_p\, e_k\right).$$

The continuous-time error dynamics become $\dot e = -K_p e$ (exponentially stable for $K_p > 0$, time constant $1/K_p$); discretization caps the usable gain near $K_p \lesssim 2/\Delta t$ before oscillation. You'll find the practical limit well below the textbook bound, because $J$ varies along the trace and the "plant" the gain sees is not the constant one the bound assumes.

**The QP formulation** (MIT *Robotic Manipulation* ch. 3, "Differential inverse kinematics with constraints"). The pseudo-inverse is the *unconstrained* minimizer of $\lVert J\dot q - v \rVert^2$. Real arms have limits, so solve the constrained problem instead:

$$\min_{\dot q}\; \lVert J(q)\dot q - v \rVert_2^2 \quad \text{s.t.} \quad \dot q_{\min} \le \dot q \le \dot q_{\max}, \qquad q_{\min} \le q + \dot q\,\Delta t \le q_{\max}$$

with $v = \dot p^{*} + K_p e$. The second constraint is a one-step joint-limit guard (a velocity-damper variant scales the bound by distance-to-limit; the simple one first). Add $\epsilon \lVert \dot q \rVert^2$ ($\epsilon \sim 10^{-6}$) for strict convexity — which is Lesson 03's DLS wearing a QP costume. The point: when a limit binds, the QP *re-optimizes the remaining freedom*; clipping the pseudo-inverse solution does not, and the direction of motion goes wrong.

**Carry forward**

- Open loop integrates error; feedback makes the error dynamics $\dot e = -K_p e$ and rejects anything the model got wrong in *magnitude* — but not in *direction* (25% mismatch shows you the limit).
- The gain ceiling is set by the control period ($\sim 2/\Delta t$) and eroded by how much $J$ changes along the trace.
- Clipping a solution and constraining an optimization are different operations; only the second keeps the task direction when a limit binds.
- $\epsilon \lVert \dot q \rVert^2$ in a QP is DLS; velocity bounds in a QP do structurally what $\lambda$ did heuristically near singularities.

| Source | Read for |
|---|---|
| Tutorial §2.3.1 | the feedback-loop framing and the notation this lesson extends |
| MIT *Robotic Manipulation* ch. 3, diff-IK-with-constraints section | the QP formulation and the velocity-damper constraint idea |
| Tutorial §2.4 | the brittleness argument Exercise 4 turns into data |
| `qpsolvers` docs (OSQP backend) | the 5-line solve API |

## Exercise 1 — Harness and controllers [Build]

Produces the rig every experiment below runs through, and the controller H1 imports. Spec:

- `trajectories.py`: generators returning $(p^{*}(t), \dot p^{*}(t))$ analytically (never finite-difference your own reference): a **line** (10 cm, 5 s) and a **circle** (r = 6 cm, in the y–z plane, centered mid-workspace via Lesson 03's workspace sampling, 10 s), both with trapezoidal speed profiles (ramp 20% of duration). Check: the circle's $\dot p^{*}$ agrees with FD of $p^{*}$ to 1e-8.
- `harness.py`: simulation loop at $\Delta t = 20$ ms (50 Hz — the rate H1 uses): controller → integrate → write `qpos` → `mj_forward` → measure. Kinematic stepping, not `mj_step`: this lesson is about the kinematics-level controller. Per run: RMS and max tracking error (measured via `mj_forward` on the written `qpos`, never against the integrated $q$), $\max\lVert\dot q\rVert$, error-vs-time CSV, 3D trace overlay. Optional model-mismatch hook: the controller gets its own copy of the kinematics with scaled link lengths; MuJoCo keeps the true model.
- `controllers.py`: `OpenLoopDiffIK`, `FeedbackDiffIK(K_p)`, `QPDiffIK(K_p, qdot_max, eps)` behind one **interface contract, imported unmodified by H1 and Lesson 07:** `qdot = controller.step(q_meas, t)` — the controller owns the trajectory; the caller owns state and integration. Open-loop and feedback variants use Lesson 03's DLS with small $\lambda$; the QP uses `qpsolvers` (OSQP) with velocity limits from the MJCF actuator ranges (or a conservative 2 rad/s), the one-step position guard, and $\epsilon$-regularization.
- `run_experiments.py`: one command reproduces every table and plot below.

**✅ Checkpoint:** a do-nothing controller runs through the harness and produces the plots/CSV; the trajectory FD check passes; the QP's output matches DLS to ~1e-6 on the plain circle when no constraint is active (it's the same optimization — assert exactly this).

## Exercise 2 — Open-loop drift [Predict → Run]

Tests objective 1.

1. **Write first:** starting exactly on the circle, does open-loop error stay flat, grow monotonically, or oscillate — and roughly how large after one lap (µm? mm? cm?). Starting 1 cm off the trajectory head: what is the error at the end of the trace?
2. Run `OpenLoopDiffIK` on line and circle, on-trajectory and with the 1 cm offset. Both error curves on one plot.
3. Reconcile with the two drift sources in Principles.

**✅ Checkpoint:** on-trajectory error grows monotonically (sub-mm over one circle is typical); the 1 cm offset persists for the whole trace.

## Exercise 3 — Gain sweep [Predict → Run]

Tests objective 2.

1. **Write first:** the textbook bound is $2/\Delta t = 100$ s⁻¹. Predict the gain at which you'll first see ringing on the circle, and why it's below 100. Predict the settling time for the 1 cm offset at $K_p = 10$ (hint: $\sim 3/K_p$).
2. Sweep $K_p \in \{0, 1, 2, 5, 10, 20, 50\}$ s⁻¹ on the circle with the offset start. Plot error-vs-time per gain, then steady-state RMS vs $K_p$ and overshoot vs $K_p$.
3. Reconcile: where did instability actually start, and what about the trace explains the gap to 100?

**✅ Checkpoint:** the offset is absorbed within ~$3/K_p$ s for moderate gains; some tested gain rings or diverges; the sweep reads sluggish → crisp → ringing.

## Exercise 4 — Mismatch and disturbance [Predict → Run]

Tests objective 3 — the experiment that justifies the rest of the course.

1. **Write first:** a 2×3 table (open-loop, $K_p = 10$) × (0%, +5%, +10% link-length error in the controller's model) with your predicted RMS *ratios* relative to the 0% cell. Then: at 25% mismatch, does closed-loop tracking still converge to the reference, and if not, why can't any gain fix it?
2. Run the table. Then inject a disturbance: at $t = 4$ s, add a 2° offset on `shoulder_lift` for 1 s into the measured-state path (a crude collision stand-in); plot recovery open-loop vs feedback.
3. Push mismatch to 25% closed-loop and plot the residual.
4. Reconcile in `RESULTS.md`: feedback corrects magnitude along the directions $J$ proposes; at large mismatch the directions themselves are wrong.

**✅ Checkpoint:** at 10% mismatch closed-loop RMS is ≥ 10× lower than open-loop; the disturbance is rejected closed-loop within ~$3/K_p$ s and never open-loop; the 25% run shows visible residual error.

## Exercise 5 — Read the QP [Read the kernel]

Tests objective 4's mechanism before you see it fail. Annotate `QPDiffIK.step` line by line: the objective and what $v$ contains; the velocity bounds; the one-step position guard and why it is linear in $\dot q$; the $\epsilon$ term and the Lesson 03 equation it reproduces; the solve call and what "infeasible" means physically. Commit the annotated file.

**✅ Checkpoint:** every line maps to a term in the Principles QP; the annotation names the DLS equivalence.

## Exercise 6 — QP vs clipped pseudo-inverse at a limit [Predict → Run]

Tests objective 4.

1. **Write first:** place the circle so its far arc demands a wrist pose beyond a joint limit. Sketch the trace a *clipped* pseudo-inverse produces (which way does it veer, and does it stay on the circle elsewhere?) versus the QP's. Predict which one violates `jnt_range` and by how much.
2. Run both. Then a second stress: a line passing near Lesson 03's singular region with velocity limits on — note what the QP's bounds do that $\lambda$ did.
3. Log QP solve time per tick.

**✅ Checkpoint:** the QP's max joint excursion respects `jnt_range` to machine precision while clipped-$J^{+}$ either violates it or veers off-trace (quantify both); median QP solve < 1 ms (headroom for H1's 50 Hz loop).

## Exercise 7 — What ships to H1 [Decide]

From Exercises 3, 4, and 6, choose $K_p$, $\lambda$ (from Lesson 03), the velocity bound, and whether the position guard is the simple or damper form. Defend each in `RESULTS.md` with the row or figure that constrains it, and state which of the two failures you demonstrated (mismatch-direction error, limit-binding) Phase 3 (RL) addresses and which Phase 4 (imitation) addresses.

**✅ Checkpoint:** four named values, each with its justifying row; the Phase 3/4 sentence is specific.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `controllers.py` | **Interface contract, imported unmodified by H1 and Lesson 07:** `qdot = controller.step(q_meas, t)`; three controllers behind it; QP annotated per Exercise 5 |
| `trajectories.py`, `harness.py`, `run_experiments.py` | one command reproduces every table/plot; FD and QP-DLS parity checks printed |
| `plots/` | drift plot, gain-sweep figure, mismatch table + disturbance-recovery plot, QP-vs-clipped overlay |
| `RESULTS.md` | predictions and reconciliations for Exercises 2–4 and 6; the Exercise 7 decision; ≤ 10 sentences connecting Exercises 4 and 6 to tutorial §2.4 |

## Done when

- [ ] All checkpoints pass from `python run_experiments.py`.
- [ ] The mismatch table shows the ≥ 10× closed-loop advantage at 10% and a residual at 25%, both predicted before the run.
- [ ] The QP demonstrably respects limits where clipped-$J^{+}$ fails, with the trace overlay to prove it.
- [ ] `RESULTS.md` names the shipped values and which failure each later phase addresses.

## Self-check

1. Why does open-loop drift grow even with a perfect model? (Two sources; one is the integrator.)
2. Where does the practical $K_p$ ceiling come from, and what happens to it at $\Delta t = 100$ ms?
3. Why is clipping the pseudo-inverse solution not equivalent to the QP when a limit binds?
4. What does the $\epsilon\lVert\dot q\rVert^2$ term do mathematically, and what earlier idea is it identical to?
5. At 25% model mismatch, why can't any gain fix tracking? What *would* fix it?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Great tracking that's fake | measuring error against integrated $q$ instead of MuJoCo's FK of it | always measure via `mj_forward` on the written `qpos` |
| Feedback unstable at modest $K_p$ | $e$ computed from stale FK, or $\Delta t$ mismatch between loop and integrator | one $\Delta t$ constant, one measurement point per tick |
| QP infeasible mid-trace | one-step position constraint + tight velocity bound conflict near a limit | soften the position guard (velocity-damper scaling) or add a slack variable with a large penalty |
| OSQP returns slightly different results run-to-run | warm-starting across ticks | fine for control; for the parity *check*, construct a fresh solver |
| Circle unreachable in z | workspace center guessed, not computed | pick the center via Lesson 03's workspace sampling |
| Jerky traces in the viewer | writing `qpos` while the passive viewer thread renders | step and render from the same thread (`mjpython`), or record offline |

## Going deeper

- **Nullspace posture.** With a 3-DOF position task the SO-101 has 2 redundant DOF — bias them toward mid-range via the QP cost $\lVert J\dot q - v\rVert^2 + \gamma\lVert \dot q - \dot q_{\text{posture}}\rVert^2$; show the elbow behaving sensibly on the circle where the plain QP lets it wander.
- **Velocity damper.** Replace the one-step guard with the distance-scaled bound and show the infeasibility pitfall disappearing near limits.

## References

- Tedrake, *Robotic Manipulation*, ch. 3 — differential IK with constraints. manipulation.csail.mit.edu.
- LeRobot team, *Robot Learning: A Tutorial*, §2.3.1, §2.4. arXiv:2510.12403.
- Lynch & Park, *Modern Robotics*, ch. 5–6 (velocity kinematics, numerical IK).
- `qpsolvers` + OSQP documentation.
