# Lesson 05 — Optimal Control Sprint: LQR → iLQR → TVLQR

Install the control-theoretic backbone the tutorial names but never teaches — exact optimal control for linear systems, iterated local optimization for nonlinear ones, and trajectory stabilization — and prove you have it by predicting what breaks when each piece of iLQR is removed, then removing it.

| | |
|---|---|
| **Phase** | 2 — Classical core |
| **Time** | 1–2 sessions (5–7 h desk time), all Mac-local (numpy; JAX/torch for autodiff; MuJoCo optional) |
| **Cost** | $0 |
| **Prerequisites** | 03–04 (Jacobians, linearization comfort); calculus of the chain-rule-heavy kind |
| **Feeds into** | 08 (the RL ladder solves the same objective model-free — the mapping table you write here is reused there verbatim), 20 (world-model policies are learned MPC over exactly this machinery) |

## Learning objectives

After this lesson you can:

1. **Derive** the finite-horizon discrete LQR backward Riccati recursion and verify its fixed point against the algebraic Riccati equation.
2. **Explain** each term of the iLQR backward pass as a Bellman-equation Taylor expansion, and predict what breaks without regularization or line search — then show it.
3. **Quantify** the robustness basin a linear policy has around upright, and why that basin is the motivation for iLQR.
4. **Predict** TVLQR's recovery rate from perturbed starts and under model mismatch, and explain why it stabilizes states iLQR never visited.
5. **Map** every LQR/iLQR object onto its RL counterpart: value function ↔ cost-to-go, policy ↔ feedback gain, Q-function ↔ the $Q$-expansions below.

## Principles

**LQR.** Linear dynamics $x_{k+1} = A x_k + B u_k$, quadratic cost $J = x_N^\top Q_f x_N + \sum_{k=0}^{N-1} \left( x_k^\top Q x_k + u_k^\top R u_k \right)$. Dynamic programming gives cost-to-go $V_k(x) = x^\top P_k x$ with the **backward Riccati recursion** ($P_N = Q_f$):

$$K_k = (R + B^\top P_{k+1} B)^{-1} B^\top P_{k+1} A$$

$$P_k = Q + A^\top P_{k+1} (A - B K_k)$$

and optimal policy $u_k = -K_k x_k$ — *linear feedback falls out of optimality; nobody put it in*. As $N \to \infty$, $P_k$ converges to the fixed point of the discrete algebraic Riccati equation (DARE); that is your check.

**iLQR** (16-745; Underactuated ch. 10). Nonlinear dynamics $x_{k+1} = f(x_k, u_k)$: iterate (1) roll out the current control sequence; (2) linearize along the rollout, $A_k = \partial f/\partial x$, $B_k = \partial f/\partial u$; (3) **backward pass** — expand the Bellman equation to second order around the rollout. With value expansion $V_x, V_{xx}$ at $k{+}1$:

$$Q_x = \ell_x + A_k^\top V_x' \qquad Q_u = \ell_u + B_k^\top V_x'$$

$$Q_{xx} = \ell_{xx} + A_k^\top V_{xx}' A_k \qquad Q_{uu} = \ell_{uu} + B_k^\top V_{xx}' B_k \qquad Q_{ux} = \ell_{ux} + B_k^\top V_{xx}' A_k$$

$$d_k = -\left(Q_{uu} + \mu I\right)^{-1} Q_u \qquad K_k = -\left(Q_{uu} + \mu I\right)^{-1} Q_{ux}$$

then $V_x = Q_x - K_k^\top Q_{uu} d_k$, $V_{xx} = Q_{xx} - K_k^\top Q_{uu} K_k$ (the standard simplified form; 16-745 derives it). $\mu$ is **regularization** — it keeps $Q_{uu}$ positive definite far from the optimum, interpolating toward gradient descent. (4) **forward pass** — roll out $u_k = \bar u_k + \alpha d_k + K_k(x_k - \bar x_k)$, backtracking $\alpha \in \{1, \tfrac12, \tfrac14, \dots\}$ until the actual cost decrease is a reasonable fraction of the predicted decrease $-\alpha Q_u^\top d - \tfrac{\alpha^2}{2} d^\top Q_{uu} d$. The feedback term $K_k(x_k - \bar x_k)$ *inside* the rollout is what separates iLQR from plain shooting and keeps long horizons from blowing up.

**TVLQR.** Given the converged trajectory $(\bar x_k, \bar u_k)$, run the LQR recursion on the time-varying linearizations $A_k, B_k$ about it. The resulting gains $K_k$ stabilize a *tube* around the trajectory against perturbations iLQR never saw.

**Why this is the hinge of the course.** SAC (Lesson 08) minimizes the same objective with the model sampled instead of given; world-model policies (Lesson 20) learn the model and run this loop inside it. The tutorial's §2.4 argument — this machinery is beautiful and still hits a wall at models and contact — only lands if you have felt the machinery work.

**Carry forward**

- Optimality with a quadratic cost and linear dynamics *produces* linear feedback; $P_k$ is the cost-to-go Hessian, and it is computed backward because cost-to-go is defined backward.
- iLQR = repeated LQR on linearizations, made safe by two guards: $\mu$ (keeps $Q_{uu} \succ 0$) and line search (keeps the second-order model honest). Remove either and the iteration diverges in a characteristic way.
- The forward pass is closed-loop ($K_k$ inside the rollout), which is why iLQR tolerates long horizons where shooting cannot.
- TVLQR gains stabilize a tube, not a point — they cover perturbations the optimizer never visited.
- $V_{xx} \leftrightarrow$ value-function curvature, $Q_u = 0 \leftrightarrow$ the policy-gradient stationarity condition: RL is this problem without $A_k, B_k$.

| Source | Read for |
|---|---|
| CMU 16-745 lectures (LQR through iLQR/DDP; 2025 playlist at optimalcontrol.ri.cmu.edu) | the Riccati and backward-pass derivations done live, with the regularization/line-search war stories |
| Underactuated ch. 8 (LQR) | finite vs infinite horizon, time-varying LQR, what $P$ *is* |
| Underactuated ch. 10 (Trajectory Optimization) | where iLQR sits among shooting/transcription methods; DDP vs iLQR distinction |
| Tutorial §2.4 | why this machinery still hits the wall (models, contact) — the course's hinge argument |

## Exercise 1 — Cartpole dynamics you can trust [Build]

Everything downstream differentiates this function; the principle is that a Jacobian you did not check is not a Jacobian. Spec for `dynamics.py`:

- Continuous cartpole dynamics in numpy — state $x = (p, \theta, \dot p, \dot\theta)$, control = cart force, standard parameters ($m_c = 1$, $m_p = 0.1$, $l = 0.5$, $g = 9.81$; equations in Underactuated ch. 3). $\theta = \pi$ is upright.
- RK4 discretization at $\Delta t = 0.02$ s: `f(x, u)`.
- Jacobians $A = \partial f/\partial x$, $B = \partial f/\partial u$ of the *discrete* map — autodiff (JAX or torch) is fine; analytic is not required.
- The check (`checks.py`, part 1): central finite differences ($h = 10^{-6}$) vs the returned Jacobians, max abs error < 1e-6, at 100 random states.
- A short animation of a hanging pole released from rest.

**✅ Checkpoint:** FD check green at 100 states; the released pole swings symmetrically.

## Exercise 2 — LQR and its fixed point [Build]

The principle: the Riccati recursion's fixed point *is* the DARE, and the infinite-horizon gain is a limit you can watch. Spec for `lqr.py`:

- `lqr(A, B, Q, R, Qf, N) -> (Ks, Ps)`: the backward recursion, verbatim from the Principles equations.
- The check (`checks.py`, part 2): iterate to convergence, then the DARE residual $P = Q + A^\top P A - A^\top P B (R + B^\top P B)^{-1} B^\top P A$ is < 1e-8, and $P$ matches `scipy.linalg.solve_discrete_are`.

**✅ Checkpoint:** DARE residual < 1e-8; scipy cross-check agrees.

## Exercise 3 — The basin of a linear policy [Predict → Run]

The principle: a linear policy is exactly optimal for the linearization and only locally valid for the plant — the picture that motivates iLQR.

1. Linearize the cartpole about upright; compute $K_\infty$.
2. **Write first:** for initial pole offsets of 5°, 20°, 40° (at rest), which recover under $u = -K_\infty x$ on the **nonlinear** plant, and why the largest fails (name the mechanism: the linearization error, or the control authority, or both).
3. Run all three; then sweep initial $(\theta_0, \dot\theta_0)$ on a grid and mark recover/fail → `plots/lqr_basin.png`.
4. Reconcile in `RESULTS.md`.

**✅ Checkpoint:** 5° and 20° recover; 40° (or thereabouts) fails; the basin plot shows a clean bounded region around upright.

## Exercise 4 — iLQR swing-up [Build]

The centerpiece: from hanging ($\theta = 0$) to upright ($\theta = \pi$), a task no linear controller can do. Spec for `ilqr.py`:

- `ilqr(x0, u_init, f, cost, N) -> (xs, us, Ks, info)`: horizon 3 s ($N = 150$), cost $Q = \mathrm{diag}(1, 10, 0.1, 0.1)$ with the angle wrapped, $R = 0.1$, $Q_f = 100\,Q$, `u_init` = small random noise.
- Backward pass with $\mu$-regularization on the Levenberg–Marquardt schedule: start $10^{-6}$, ×10 on a non-PD $Q_{uu}$ or a failed forward pass, ÷2 on success. Use Cholesky and catch failure; never `np.linalg.inv`.
- Forward pass with backtracking line search; convergence when $|\Delta J| < 10^{-6}$.
- `info` records cost per iteration, $\alpha$ accepted, $\mu$ trace, predicted-vs-actual decrease ratio.
- **The check** (`checks.py`, part 3): on a linear-quadratic problem, iLQR converges in **one iteration** to Exercise 2's solution, < 1e-10. This single test catches most backward-pass bugs. Also: cost strictly non-increasing across iterations.

You may type this kernel yourself; the annotation in Exercise 5 is the requirement either way.

**✅ Checkpoint:** swing-up converges (typically < 100 iterations from noise init), final pole angle within 1e-3 of upright, cost curve monotone; LQR-equivalence check green. Animate the swing-up and plot the $(\theta, \dot\theta)$ phase portrait — the energy-pumping swings should be legible.

## Exercise 5 — The backward pass, line by line [Read the kernel]

The principle: every line of the backward pass is a term of a second-order Bellman expansion, and you should be able to name which.

1. In `ilqr.py`, annotate every line of the backward pass with the Principles equation it implements ($Q_x, Q_u, Q_{xx}, Q_{uu}, Q_{ux}$, $d_k$, $K_k$, $V_x$, $V_{xx}$), and every line of the $\mu$ schedule with the event that triggers it.
2. Annotate the forward pass: which line is the feedforward step, which is the feedback term, which is the predicted decrease, and where the acceptance test compares predicted to actual.
3. Check shapes in the annotations: $Q_{ux} \in \mathbb{R}^{m \times n}$, $K_k \in \mathbb{R}^{m \times n}$, $d_k \in \mathbb{R}^m$.

**✅ Checkpoint:** the annotated file is committed; the LQR-equivalence check still passes after any edits the annotation prompted.

## Exercise 6 — Remove the guards [Predict → Run]

The principle: $\mu$ and the line search each prevent a specific divergence; you should be able to predict its signature.

1. **Write first**, for each ablation, the expected signature in the cost-vs-iteration curve and in `info`: (a) line search killed ($\alpha = 1$ always); (b) regularization killed ($\mu = 0$, no schedule). Name the mechanism: which quantity stops being trustworthy, and at which iteration range it bites (early, far from optimum, or late).
2. Run each ablation from the same noise init.
3. Reconcile in `RESULTS.md`: one paragraph per ablation, prediction vs outcome.

**✅ Checkpoint:** both ablation curves recorded; each paragraph names the mechanism that matched (or did not match) the prediction.

## Exercise 7 — TVLQR robustness [Predict → Run]

The principle: gains computed along a trajectory stabilize a tube around it — including states the optimizer never visited.

1. Run the LQR recursion along the converged trajectory's $A_k, B_k$ (cost weights as in Exercise 4) → gains $K_k$.
2. **Write first:** for 20 random draws with $\pm 10\%$ perturbation on each state dimension, the fraction reaching upright under (a) open-loop replay of $\bar u$ and (b) TVLQR $u_k = \bar u_k - K_k (x_k - \bar x_k)$; and how (b) changes with model mismatch ($m_p$ +20% in the *plant*, nominal in the controller).
3. Run both controllers on the 20 nominal draws and the 20 mismatched draws. Score: fraction reaching upright, final-state error distribution. Plot the tube: perturbed TVLQR rollouts overlaid on the nominal in phase space.
4. Reconcile in `RESULTS.md` with the 2×2 recovery table.

**✅ Checkpoint:** open-loop replay fails for most ±10% draws; TVLQR recovers ≥ 90% nominal; the mismatch row is filled in; the tube plot shows convergence onto the nominal trajectory rather than divergence from it. (If TVLQR diverges at the trajectory end, see Pitfalls — the handoff to infinite-horizon LQR is in Going deeper.)

## Exercise 8 — The RL mapping [Write]

The principle that carries into Phase 3: RL is this problem with $A_k, B_k$ sampled instead of given. Write the closing section of `RESULTS.md` as a table, equation-to-equation: value function ↔ cost-to-go $V_k$, policy ↔ $-K_k$, Q-function ↔ the $Q$-expansions, $Q_u = 0$ ↔ ?, $V_{xx}$ ↔ ?, line search ↔ ?, $\mu$ ↔ ?. This section is reused verbatim when Lesson 08 starts.

**✅ Checkpoint:** every row names a specific object on both sides, not a vibe.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `dynamics.py`, `lqr.py`, `ilqr.py`, `tvlqr.py` | pure numpy + autodiff, no control libraries; `ilqr.py` carries the Exercise 5 annotations |
| `checks.py` | prints FD-Jacobian error, DARE residual, LQR-equivalence error, cost-monotonicity — one command |
| `plots/` | LQR basin, cost-vs-iteration (with both ablation curves), swing-up animation (GIF), phase portrait, TVLQR tube + recovery table |
| `RESULTS.md` | Exercise 3/6/7 predictions with reconciliations; the ablation paragraphs; the robustness table; the RL-mapping table |

## Done when

- [ ] `checks.py` green: FD < 1e-6, DARE < 1e-8, LQR-equivalence < 1e-10, cost monotone.
- [ ] iLQR swings up from rest with a monotone cost curve; `ilqr.py` is annotated line by line.
- [ ] Both ablation predictions are written before the runs and reconciled after.
- [ ] TVLQR ≥ 90% recovery at ±10% perturbation, and the mismatch row is filled in.
- [ ] `RESULTS.md`'s RL-mapping table exists and is specific (equation-to-equation).

## Self-check

1. Why does the Riccati recursion run *backward*, and what does $P_k$ mean at an interior step?
2. In the backward pass, what exactly does $\mu$ regularize against, and why does large $\mu$ turn the step into gradient descent?
3. Why does the forward pass need the feedback term $K_k(x_k - \bar x_k)$ even though $d_k$ already encodes the improvement?
4. iLQR vs DDP: which second-order term does iLQR drop, and when does that matter?
5. State the exact correspondence: $V_{xx} \leftrightarrow$ ?, $Q_u = 0 \leftrightarrow$ ? in RL terms.
6. Why can TVLQR stabilize states iLQR never visited?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Swing-up converges to a spin, not a balance | angle not wrapped in the cost | penalize $1 - \cos(\theta - \pi)$ or wrap the error to $[-\pi, \pi]$ |
| Backward pass throws on `inv(Quu)` | $Q_{uu}$ not PD far from optimum | that's what $\mu$ is for; use Cholesky + catch, never `np.linalg.inv` |
| Cost decreases then explodes | no line search, or feedback term missing in forward rollout | implement both; Exercise 6 shows you the signatures |
| iLQR "works" but LQR-equivalence fails | index off-by-one in the recursion ($P_{k+1}$ vs $P_k$), or $\ell_{ux}$ transposed | the equivalence test localizes it: check shapes $Q_{ux} \in \mathbb{R}^{m\times n}$ |
| FD Jacobian check fails only for $\theta$ rows | differentiated the continuous dynamics instead of the RK4 map | differentiate through the integrator (autodiff the discrete map) |
| TVLQR diverges at trajectory end | gains from a short horizon with weak $Q_f$ | boost $Q_f$, or hand off to infinite-horizon LQR at the top (Going deeper) |

## Going deeper

- **Handoff controller.** TVLQR that hands off to Exercise 2's infinite-horizon LQR once upright ($|\theta - \pi| < 0.1$); compare recovery to plain TVLQR on the same 20 draws.
- **Extra checks.** Energy conservation of the passive RK4 rollout (drift < 0.1% over 5 s); closed-loop spectral radius $\rho(A - BK_\infty) < 1$; analytic Jacobians vs autodiff.
- **MuJoCo edition.** Swap the numpy plant for MuJoCo's cartpole (`mjx` or FD through `mj_step`), keep the solver unchanged, and report what contact-free simulator swapping does to convergence. Then read 16-745's MPC lecture and turn `ilqr.py` into a 10 Hz receding-horizon controller — ~30 extra lines that preview exactly what Lesson 20's world-model policies learn end-to-end.

## References

- CMU 16-745 *Optimal Control & Reinforcement Learning* (Manchester), LQR/iLQR/DDP lectures. optimalcontrol.ri.cmu.edu.
- Tedrake, *Underactuated Robotics*, ch. 8 (LQR), ch. 10 (Trajectory Optimization). underactuated.csail.mit.edu.
- Li & Todorov, *Iterative Linear Quadratic Regulator Design for Nonlinear Biological Movement Systems* (ILQG origin), ICINCO 2004.
- Tassa, Erez, Todorov, *Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization*, IROS 2012 — the regularization/line-search recipe above.
- LeRobot team, *Robot Learning: A Tutorial*, §2.4. arXiv:2510.12403.
