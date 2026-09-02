# Lesson 04 — Differential IK as Optimization + Feedback

Lesson 03 solved inverse kinematics as a static problem: given a target, find a configuration. This lesson turns that into a running controller that tracks a task-space trajectory tick by tick, and uses it to study the central fact about feedback control, which is that a proportional term can correct for errors the model got wrong, but only up to a point. You watch open-loop tracking drift, close the loop and find the gain ceiling, break the controller with deliberate model mismatch, and finally rebuild the tracker as a constrained quadratic program in the manner of MIT's manipulation course. The resulting controller is the one H1 runs on the physical arm.

| | |
|---|---|
| **Phase** | 2 — Classical core |
| **Time** | 1 session (3–4 h desk time), all Mac-local |
| **Cost** | $0 |
| **Prerequisites** | 03 (`kinematics.py`: `fk`, `jacobian`) |
| **Feeds into** | 07 (executes planned paths), H1 (this exact controller traces on the real arm), and conceptually 08 onward (this is the "brittle pipeline" that RL and behaviour cloning replace) |

## Learning objectives

After this lesson you can:

1. **Predict** how open-loop task-space tracking drifts even with a perfect model, and explain why it can never recover an initial offset.
2. **Explain** where the practical ceiling on the proportional gain comes from, predict it from the control period, and account for the gap between the prediction and what you measure.
3. **Quantify** the closed-loop advantage under model mismatch and state where feedback stops helping.
4. **Explain** why a constrained QP outperforms a clipped pseudo-inverse when a joint limit binds, and predict the shape of clipping's failure.
5. **Decide** the gain, damping, and limits you would ship to a real arm, with the rows that justify them.

## Principles

### From inverse kinematics to a tracker

A trajectory tracker does not solve the full inverse-kinematics problem at every tick. Instead it maps the desired task-space velocity to a joint velocity through the Jacobian and integrates:

$$\dot q_k = J^{+}(q_k)\,\dot p^{*}_k, \qquad q_{k+1} = q_k + \dot q_k \,\Delta t .$$

This scheme is open-loop in task space, because nothing in it measures where the end-effector actually is. Integration error, model mismatch and disturbances therefore accumulate rather than being corrected. Even with a perfect model there are two sources of drift: the explicit Euler integrator, and the fact that the Jacobian is evaluated at the current $q$ while the step it is used for is finite.

### Closing the loop

To make the tracker correct itself, add proportional feedback on the task-space error $e_k = p^{*}_k - \mathrm{FK}(q_k^{\text{meas}})$:

$$\dot q_k = J^{+}(q_k)\left(\dot p^{*}_k + K_p\, e_k\right).$$

In continuous time the error then obeys $\dot e = -K_p e$, which is exponentially stable for any positive $K_p$ with time constant $1/K_p$. Discretisation limits how large the gain can be: with a control period $\Delta t$, gains above roughly $2/\Delta t$ make the discrete error dynamics oscillate. In practice you will find the usable limit well below that textbook bound, because the bound assumes a constant plant, whereas the Jacobian, and hence the plant the gain acts on, changes along the trajectory.

### The quadratic-programming formulation

The pseudo-inverse is the unconstrained minimiser of $\lVert J\dot q - v \rVert^2$. Real arms have joint-position and joint-velocity limits, so the treatment of differential inverse kinematics with constraints in MIT's *Robotic Manipulation* (ch. 3) solves the constrained problem instead:

$$\min_{\dot q}\; \lVert J(q)\dot q - v \rVert_2^2 \quad \text{s.t.} \quad \dot q_{\min} \le \dot q \le \dot q_{\max}, \qquad q_{\min} \le q + \dot q\,\Delta t \le q_{\max}$$

with $v = \dot p^{*} + K_p e$. The second constraint is a one-step joint-limit guard. A velocity-damper variant scales the bound by the distance to the limit, but the simple form is enough to start with. Adding $\epsilon \lVert \dot q \rVert^2$ with $\epsilon$ around $10^{-6}$ makes the problem strictly convex, and that term is Lesson 03's damped least squares in a different guise. The reason to prefer the QP over clipping the pseudo-inverse solution is what happens when a limit binds: the QP re-optimises the remaining freedom so that the task direction is preserved as far as possible, whereas clipping truncates one component of the solution and sends the motion in the wrong direction.

**Carry forward**

- Open-loop tracking integrates its own error, whereas proportional feedback gives the error dynamics $\dot e = -K_p e$ and corrects anything the model got wrong in magnitude. It cannot correct errors in direction, which is why tracking degrades again at large model mismatch.
- The proportional-gain ceiling is set by the control period, at roughly $2/\Delta t$, and is lowered in practice by how much the Jacobian changes along the trajectory.
- Clipping a solution and constraining an optimisation are different operations. Only the constrained optimisation keeps the task direction when a limit binds, because it re-solves for the remaining freedom rather than truncating one component.
- The $\epsilon \lVert \dot q \rVert^2$ regulariser in the QP is damped least squares, and the QP's velocity bounds do explicitly near singularities what the damping parameter $\lambda$ did heuristically.

| Source | Read for |
|---|---|
| Tutorial §2.3.1 | the feedback-loop framing and the notation this lesson extends |
| MIT *Robotic Manipulation* ch. 3, diff-IK-with-constraints section | the QP formulation and the velocity-damper constraint |
| Tutorial §2.4 | the brittleness argument that Exercise 4 turns into data |
| `qpsolvers` docs (OSQP backend) | the 5-line solve API |

## Exercise 1 — Build the harness and controllers [Build]

Every experiment in this lesson runs through one simulation harness and three controllers that share an interface. This exercise specifies them. The interface, `qdot = controller.step(q_meas, t)`, is the contract that H1 and Lesson 07 import, so it is worth stating precisely: the controller owns the reference trajectory, and the caller owns the state and the integration.

Write the specification below and have an AI tool draft the four files:

- `trajectories.py`: generators returning $(p^{*}(t), \dot p^{*}(t))$ analytically. Never finite-difference your own reference, because the reference velocity must be exact for the tracker to be judged fairly. Provide a **line** (10 cm, 5 s) and a **circle** (radius 6 cm, in the y–z plane, centred mid-workspace using Lesson 03's workspace sampling, 10 s), both with trapezoidal speed profiles that ramp over 20% of the duration. The check: the circle's $\dot p^{*}$ agrees with a finite difference of $p^{*}$ to 1e-8.
- `harness.py`: a simulation loop at $\Delta t = 20$ ms (50 Hz, the rate H1 uses): controller → integrate → write `qpos` → `mj_forward` → measure. Use kinematic stepping rather than `mj_step`, because this lesson is about the kinematics-level controller and dynamics would only obscure it. Per run, record RMS and maximum tracking error (measured via `mj_forward` on the written `qpos`, never against the integrated $q$), $\max\lVert\dot q\rVert$, an error-versus-time CSV, and a 3D trace overlay. Include a model-mismatch hook: the controller receives its own copy of the kinematics with scaled link lengths while MuJoCo keeps the true model.
- `controllers.py`: `OpenLoopDiffIK`, `FeedbackDiffIK(K_p)`, and `QPDiffIK(K_p, qdot_max, eps)`, all behind the interface contract `qdot = controller.step(q_meas, t)`, which H1 and Lesson 07 import unmodified. The open-loop and feedback variants use Lesson 03's damped least squares with a small $\lambda$; the QP uses `qpsolvers` with the OSQP backend, velocity limits from the MJCF actuator ranges (or a conservative 2 rad/s), the one-step position guard, and $\epsilon$-regularisation.
- `run_experiments.py`: one command that reproduces every table and plot below.

**✅ Checkpoint:** a do-nothing controller runs through the harness and produces the plots and CSV; the trajectory finite-difference check passes; and on the plain circle, with no constraint active, the QP's output matches damped least squares to about 1e-6. That last check is exact in principle, because the two are the same optimisation problem, so assert it explicitly.

## Exercise 2 — Open-loop drift [Predict → Run]

An open-loop tracker started exactly on its trajectory drifts slowly, and one started off the trajectory never recovers, because nothing in it measures the error. Before running, predict both behaviours in the terms below. The prediction is worth making because the size of the drift is easy to guess wrongly by orders of magnitude, and the reason it never recovers an offset is the whole case for feedback.

1. Before running, write in `RESULTS.md`: starting exactly on the circle, does the open-loop error stay flat, grow monotonically, or oscillate, and roughly how large is it after one lap (micrometres, millimetres, or centimetres)? Starting 1 cm off the trajectory head, what is the error at the end of the trace?
2. Run `OpenLoopDiffIK` on the line and the circle, both on-trajectory and with the 1 cm offset. Put both error curves on one plot.
3. Reconcile the plot with the two drift sources named in the Principles section.

**✅ Checkpoint:** the on-trajectory error grows monotonically (sub-millimetre over one circle is typical), and the 1 cm offset persists for the whole trace.

## Exercise 3 — Gain sweep [Predict → Run]

With feedback on, the question becomes how large the gain can be. The textbook bound for this control period is $2/\Delta t = 100$ s⁻¹; you will find the practical limit lower, and the purpose of the exercise is to predict roughly where and then explain the gap.

1. Before running, write down the gain at which you expect to first see ringing on the circle, and why it is below 100. Also predict the settling time for the 1 cm offset at $K_p = 10$; the time constant $1/K_p$ suggests roughly $3/K_p$ seconds.
2. Sweep $K_p \in \{0, 1, 2, 5, 10, 20, 50\}$ s⁻¹ on the circle with the offset start. Plot error against time for each gain, then steady-state RMS against $K_p$ and overshoot against $K_p$.
3. Reconcile: where did instability actually begin, and what about the trace explains the gap to 100?

**✅ Checkpoint:** the offset is absorbed within roughly $3/K_p$ seconds for moderate gains; some tested gain rings or diverges; and the sweep reads as sluggish, then crisp, then ringing.

## Exercise 4 — Model mismatch and disturbance [Predict → Run]

This is the experiment the rest of the course rests on: feedback compensates for model error, but only up to a point. You perturb the controller's copy of the link lengths while MuJoCo keeps the true model, and compare open-loop and closed-loop tracking as the mismatch grows. The 25% case matters most, because it shows where feedback stops helping and why.

1. Before running, fill in a 2×3 table in `RESULTS.md` with predicted RMS *ratios* relative to the 0% cell: rows are open-loop and $K_p = 10$, columns are 0%, +5% and +10% link-length error in the controller's model. Then write down whether, at 25% mismatch, closed-loop tracking still converges to the reference, and if not, why no gain can fix it.
2. Run the table. Then inject a disturbance: at $t = 4$ s, add a 2° offset on `shoulder_lift` for 1 s into the measured-state path, as a crude stand-in for a collision, and plot the recovery for open-loop and feedback.
3. Push the mismatch to 25% closed-loop and plot the residual.
4. Reconcile in `RESULTS.md`: feedback corrects magnitude along the directions $J$ proposes, and at large mismatch the directions themselves are wrong.

**✅ Checkpoint:** at 10% mismatch the closed-loop RMS is at least 10× lower than open-loop; the disturbance is rejected closed-loop within roughly $3/K_p$ seconds and never open-loop; and the 25% run shows visible residual error.

## Exercise 5 — Annotate the QP controller [Read the kernel]

Before you watch the QP controller outperform the clipped pseudo-inverse, annotate its code so that every line is tied to a term in the formulation above. Annotate `QPDiffIK.step` line by line: the objective and what $v$ contains; the velocity bounds; the one-step position guard and why it is linear in $\dot q$; the $\epsilon$ term and the Lesson 03 equation it reproduces; and the solve call, including what an infeasible result means physically. Commit the annotated file.

**✅ Checkpoint:** every line maps to a term in the Principles QP, and the annotation names the equivalence to damped least squares.

## Exercise 6 — QP versus clipped pseudo-inverse at a joint limit [Predict → Run]

Here the two ways of handling a joint limit are compared on a trajectory whose far arc demands a wrist pose past a limit. Predict the shape of each controller's failure before running; the difference between truncating a component and re-optimising the remaining freedom is visible in the traces.

1. Before running, place the circle so that its far arc demands a wrist pose beyond a joint limit, and sketch the trace a clipped pseudo-inverse would produce (which way it veers, and whether it stays on the circle elsewhere) against the trace the QP would produce. Predict which one violates `jnt_range` and by how much.
2. Run both. Then apply a second stress: a line passing near Lesson 03's singular region with velocity limits on, and note what the QP's bounds do that $\lambda$ did.
3. Log the QP solve time per tick.

**✅ Checkpoint:** the QP's maximum joint excursion respects `jnt_range` to machine precision while the clipped pseudo-inverse either violates it or veers off the trace (quantify both); the median QP solve is under 1 ms, which leaves headroom for H1's 50 Hz loop.

## Exercise 7 — Choose the values that ship to H1 [Decide]

From Exercises 3, 4, and 6, choose $K_p$, $\lambda$ (from Lesson 03), the velocity bound, and whether the position guard takes the simple or the damper form. Defend each choice in `RESULTS.md` with the row or figure that constrains it, and state which of the two failures you demonstrated, mismatch in direction and limit-binding, Phase 3 (reinforcement learning) addresses and which Phase 4 (imitation) addresses.

**✅ Checkpoint:** four named values, each with its justifying row, and a specific sentence about Phases 3 and 4.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `controllers.py` | **Interface contract, imported unmodified by H1 and Lesson 07:** `qdot = controller.step(q_meas, t)`; three controllers behind it; QP annotated per Exercise 5 |
| `trajectories.py`, `harness.py`, `run_experiments.py` | one command reproduces every table and plot; FD and QP-DLS parity checks printed |
| `plots/` | drift plot, gain-sweep figure, mismatch table + disturbance-recovery plot, QP-vs-clipped overlay |
| `RESULTS.md` | predictions and reconciliations for Exercises 2–4 and 6; the Exercise 7 decision; ≤ 10 sentences connecting Exercises 4 and 6 to tutorial §2.4 |

## Done when

- [ ] All checkpoints pass from `python run_experiments.py`.
- [ ] The mismatch table shows the ≥ 10× closed-loop advantage at 10% and a residual at 25%, both predicted before the run.
- [ ] The QP demonstrably respects limits where the clipped pseudo-inverse fails, with the trace overlay to show it.
- [ ] `RESULTS.md` names the shipped values and says which failure each later phase addresses.

## Self-check

1. Why does open-loop drift grow even with a perfect model? (There are two sources; one is the integrator.)
2. Where does the practical $K_p$ ceiling come from, and what happens to it at $\Delta t = 100$ ms?
3. Why is clipping the pseudo-inverse solution not equivalent to the QP when a limit binds?
4. What does the $\epsilon\lVert\dot q\rVert^2$ term do mathematically, and which earlier idea is it identical to?
5. At 25% model mismatch, why can't any gain fix tracking? What *would* fix it?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Tracking looks perfect but is not real | error measured against the integrated $q$ instead of MuJoCo's FK of it | always measure via `mj_forward` on the written `qpos` |
| Feedback unstable at modest $K_p$ | $e$ computed from stale FK, or $\Delta t$ mismatch between loop and integrator | one $\Delta t$ constant, one measurement point per tick |
| QP infeasible mid-trace | one-step position constraint and a tight velocity bound conflict near a limit | soften the position guard (velocity-damper scaling) or add a slack variable with a large penalty |
| OSQP returns slightly different results run-to-run | warm-starting across ticks | fine for control; for the parity check, construct a fresh solver |
| Circle unreachable in z | workspace centre guessed, not computed | pick the centre via Lesson 03's workspace sampling |
| Jerky traces in the viewer | writing `qpos` while the passive viewer thread renders | step and render from the same thread (`mjpython`), or record offline |

## Going deeper

- **Nullspace posture.** With a 3-DOF position task the SO-101 has 2 redundant degrees of freedom. Bias them toward mid-range via the QP cost $\lVert J\dot q - v\rVert^2 + \gamma\lVert \dot q - \dot q_{\text{posture}}\rVert^2$, and show the elbow behaving sensibly on the circle where the plain QP lets it wander.
- **Velocity damper.** Replace the one-step guard with the distance-scaled bound and show the infeasibility pitfall disappearing near limits.

## References

- Tedrake, *Robotic Manipulation*, ch. 3 — differential IK with constraints. manipulation.csail.mit.edu.
- LeRobot team, *Robot Learning: A Tutorial*, §2.3.1, §2.4. arXiv:2510.12403.
- Lynch & Park, *Modern Robotics*, ch. 5–6 (velocity kinematics, numerical IK).
- `qpsolvers` + OSQP documentation.
