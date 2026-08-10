# Lesson 04 — Differential IK as Optimization + Feedback

Turn Lesson 03's static IK into a running controller: track task-space trajectories with the SO-101 in MuJoCo, watch open-loop control drift and feedback fix it, then rebuild the whole thing as a constrained QP the way MIT's manipulation course does. This is the controller H1 will run on the physical arm.

| | |
|---|---|
| **Phase** | 2 — Classical core |
| **Time** | 1–2 sessions (5–7 h), all Mac-local |
| **Cost** | $0 |
| **Prerequisites** | 03 (`kinematics.py`: `fk`, `jacobian`) |
| **Feeds into** | 07 (executes planned paths), H1 (this exact controller traces on the real arm), and conceptually 08+ (this is the "brittle pipeline" RL and BC replace) |

## Learning objectives

After this lesson you can:

1. **Implement** velocity-level task-space tracking $\dot q = J^{+}\dot p^{*}$ with explicit integration, and measure its drift.
2. **Explain and demonstrate** why proportional feedback $\dot q = J^{+}(\dot p^{*} + K_p e)$ rejects model mismatch and disturbances that open-loop tracking cannot.
3. **Tune** $K_p$ against the discrete-time stability limit imposed by the control period.
4. **Formulate and solve** differential IK as a QP with joint-position and velocity limits, and show where it beats the clipped pseudo-inverse.
5. **Defend** the tutorial's brittleness argument (§2.4) from your own mismatch experiments — with numbers.

## Background

**From IK to control.** A trajectory tracker doesn't re-solve full IK each tick; it maps desired task-space *velocity* to joint velocity through the Jacobian and integrates:

$$\dot q_k = J^{+}(q_k)\,\dot p^{*}_k, \qquad q_{k+1} = q_k + \dot q_k \,\Delta t .$$

This is **open-loop** in task space: integration error, model mismatch, and disturbances accumulate because nothing measures where the end-effector actually is.

**Closing the loop.** Add proportional feedback on task-space error $e_k = p^{*}_k - \mathrm{FK}(q_k^{\text{meas}})$:

$$\dot q_k = J^{+}(q_k)\left(\dot p^{*}_k + K_p\, e_k\right).$$

The continuous-time error dynamics become $\dot e = -K_p e$ (exponentially stable for $K_p > 0$); discretization caps the usable gain near $K_p \lesssim 2/\Delta t$ before oscillation — you'll find the practical limit empirically.

**The QP formulation** (MIT *Robotic Manipulation* ch. 3, "Differential inverse kinematics with constraints"). The pseudo-inverse is the *unconstrained* minimizer of $\lVert J\dot q - v \rVert^2$. Real arms have limits, so solve the constrained problem instead:

$$\min_{\dot q}\; \lVert J(q)\dot q - v \rVert_2^2 \quad \text{s.t.} \quad \dot q_{\min} \le \dot q \le \dot q_{\max}, \qquad q_{\min} \le q + \dot q\,\Delta t \le q_{\max}$$

with $v = \dot p^{*} + K_p e$. The second constraint is a one-step joint-limit guard (a velocity-damper variant scales the bound by distance-to-limit; implement the simple one first). Add $\epsilon \lVert \dot q \rVert^2$ ($\epsilon \sim 10^{-6}$) for strict convexity — which is DLS wearing a QP costume. Crucially, when a limit binds, the QP *re-optimizes the remaining freedom*; clipping the pseudo-inverse solution does not, and the direction of motion goes wrong.

| Source | Read for |
|---|---|
| Tutorial §2.3.1 | the feedback-loop framing and the notation this lesson extends |
| MIT *Robotic Manipulation* ch. 3, diff-IK-with-constraints section | the QP formulation and the velocity-damper constraint idea |
| Tutorial §2.4 | the brittleness argument Part 3 turns into data |
| `qpsolvers` docs (OSQP backend) | the 5-line solve API |

## Part 0 — Reference trajectories + harness (~45 min)

One test rig for every controller variant; every experiment below runs through it.

1. Trajectory generators returning $(p^{*}(t), \dot p^{*}(t))$ analytically (don't finite-difference your own reference): a **line** (10 cm, 5 s) and a **circle** (r = 6 cm, in the y–z plane, centered mid-workspace, 10 s), both with trapezoidal speed profiles (ramp 20% of duration).
2. Simulation loop at $\Delta t = 20$ ms (50 Hz — the rate H1 uses): controller → integrate → write `qpos` → `mj_forward` → measure. (Kinematic stepping, not `mj_step`: this lesson is about the kinematics-level controller, so bypass dynamics.)
3. Metrics + plots per run: RMS and max tracking error, $\max\lVert\dot q\rVert$, error-vs-time curve, 3D trace overlay (target vs actual).
4. **Interface contract** (H1 imports this): `qdot = controller.step(q_meas, t)` — controller owns the trajectory; caller owns state and integration.

**✅ Checkpoint:** the harness runs a do-nothing controller and produces the plots/CSV; the circle's $\dot p^{*}$ agrees with FD of $p^{*}$ to 1e-8 (test the generator, then trust it).

## Part 1 — Open-loop tracking (~45 min)

1. Run $\dot q = J^{+}\dot p^{*}$ (use DLS from Lesson 03 with small $\lambda$) on line and circle, starting exactly on the trajectory.
2. Repeat with the start pose offset 1 cm from the trajectory head — open loop can *never* recover this; show it.
3. Log RMS error for both cases.

**✅ Checkpoint:** starting on-trajectory, error stays small but *grows monotonically* (integration drift, sub-mm over one circle is typical); with the 1 cm offset, the 1 cm error persists for the whole trace. Both curves in one plot.

## Part 2 — Proportional feedback + gain sweep (~1 h)

1. Add the $K_p e$ term. Sweep $K_p \in \{0, 1, 2, 5, 10, 20, 50\}$ s⁻¹ on the circle with the offset start.
2. Plot: error-vs-time per gain (one figure), then steady-state RMS vs $K_p$ and overshoot vs $K_p$.
3. Find the instability onset empirically; relate it to $2/\Delta t = 100$ in `RESULTS.md` (you'll find trouble well before the textbook bound — say why: the Jacobian varies along the trace).

**✅ Checkpoint:** the offset is absorbed within ~$3/K_p$ seconds for moderate gains; some tested gain oscillates or diverges. The sweep figure shows the whole story: sluggish → crisp → ringing.

## Part 3 — Break it: mismatch and disturbance (~1.5 h)

The experiment that justifies the rest of the course: feedback covers for model error — up to a point.

1. **Model mismatch:** perturb the link lengths *in your controller's copy* of the kinematics by +5% and +10% (MuJoCo keeps the true model — it plays the role of reality). Run open-loop vs $K_p = 10$ on the circle; table of RMS errors, 2 controllers × 3 model conditions.
2. **Disturbance:** at $t = 4$ s, add a constant $q$-offset injection (2° on `shoulder_lift` for 1 s — a crude collision stand-in) into the measured state path. Plot recovery for open-loop vs feedback.
3. **Where feedback fails:** push mismatch to 25% — show tracking degrades even closed-loop, and note *why* (the Jacobian direction itself is now wrong; feedback corrects magnitude along the wrong directions).

**✅ Checkpoint:** at 10% mismatch, closed-loop RMS is ≥ 10× lower than open-loop; the disturbance is rejected closed-loop within ~$3/K_p$ s and never open-loop. The 25% run shows visible residual error.

## Part 4 — The QP tracker (~2 h)

1. Implement the QP above with `qpsolvers` (OSQP): velocity limits from the MJCF actuator ranges (or a conservative 2 rad/s), the one-step position-limit constraint, $\epsilon$-regularization.
2. Baseline comparison: on the plain circle, QP output must match DLS to ~1e-6 when no constraint is active (it's the same optimization — test exactly this).
3. **Stress 1 — limits:** place the circle so its far arc demands a wrist pose beyond a joint limit. Compare clipped-pseudo-inverse vs QP: trace shape, error, constraint satisfaction. Clipping deforms the whole trace; the QP degrades gracefully and stays feasible.
4. **Stress 2 — near-singularity:** run a line that passes near the Part 5 (Lesson 03) singular region with velocity limits on. The QP's velocity bounds do structurally what λ did heuristically — note the connection.
5. Runtime check: log solve time per tick (OSQP should sit well under 1 ms — headroom for the 50 Hz real-time loop in H1).

**✅ Checkpoint:** unconstrained parity test green; in Stress 1 the QP's max joint excursion respects `jnt_range` to machine precision while clipped-$J^{+}$ either violates it or veers off-trace (quantify both); median QP solve < 1 ms.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `controllers.py` | `OpenLoopDiffIK`, `FeedbackDiffIK`, `QPDiffIK` behind the `step(q_meas, t)` contract; imported unmodified by H1 |
| `run_experiments.py` | one command reproduces every table/plot below from scratch |
| `plots/` | drift plot, gain-sweep figure, mismatch table + disturbance-recovery plot, QP-vs-clipped comparison |
| `tests/` | trajectory-generator FD test, QP-DLS parity test, constraint-satisfaction test |
| `RESULTS.md` | the numbers, plus ≤ 10 sentences connecting Part 3–4 to tutorial §2.4: what exactly is brittle, and what the QP does and doesn't fix |

## Done when

- [ ] All four checkpoints pass; `pytest` green.
- [ ] The mismatch table shows the ≥ 10× closed-loop advantage at 10% mismatch.
- [ ] The QP demonstrably respects limits where clipped-$J^{+}$ fails, with the trace overlay to prove it.
- [ ] `RESULTS.md` states, in your own words, which failure Phase 3 (RL) addresses and which Phase 4 (imitation) addresses.

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
| OSQP returns slightly different results run-to-run | warm-starting across ticks | fine for control; for the parity *test*, construct a fresh solver |
| Circle unreachable in z | workspace center guessed, not computed | pick the center via Lesson 03's workspace sampling |
| Jerky traces in the viewer | writing `qpos` while the passive viewer thread renders | step and render from the same thread (`mjpython`), or record offline |

## Stretch

Add a nullspace objective: with a 3-DOF position task the SO-101 has 2 redundant DOF — bias them toward mid-range posture via the QP cost $\lVert J\dot q - v\rVert^2 + \gamma\lVert \dot q - \dot q_{\text{posture}}\rVert^2$. Show the elbow behaving sensibly on the circle where the plain QP lets it wander.

## References

- Tedrake, *Robotic Manipulation*, ch. 3 — differential IK with constraints. manipulation.csail.mit.edu.
- LeRobot team, *Robot Learning: A Tutorial*, §2.3.1, §2.4. arXiv:2510.12403.
- Lynch & Park, *Modern Robotics*, ch. 5–6 (velocity kinematics, numerical IK).
- `qpsolvers` + OSQP documentation.
