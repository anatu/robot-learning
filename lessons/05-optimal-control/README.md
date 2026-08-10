# Lesson 05 — Optimal Control Sprint: LQR → iLQR → TVLQR

Build the control-theoretic backbone the tutorial names but never teaches, distilled from CMU 16-745 into one lesson: exact optimal control for linear systems (LQR), iterated local optimization for nonlinear ones (iLQR swing-up), and trajectory stabilization (TVLQR). This is what makes Phase 3's RL principled instead of recipes — SAC is solving *this problem* with sampled gradients.

| | |
|---|---|
| **Phase** | 2 — Classical core |
| **Time** | ~2 sessions (8–10 h), all Mac-local (numpy; MuJoCo optional) |
| **Cost** | $0 |
| **Prerequisites** | 03–04 (Jacobians, linearization comfort); calculus of the chain-rule-heavy kind |
| **Feeds into** | 08 (the RL ladder solves the same objective model-free), 20 (world-model policies are learned MPC over exactly this machinery) |

## Learning objectives

After this lesson you can:

1. **Derive and implement** the finite-horizon discrete LQR backward Riccati recursion, and verify its fixed point against the algebraic Riccati equation.
2. **Implement** iLQR from scratch — backward pass with regularization, forward pass with line search — and swing up a cartpole with it.
3. **Explain** each term of the iLQR backward pass as a Bellman-equation Taylor expansion, and what breaks without regularization or line search.
4. **Stabilize** the swing-up trajectory with TVLQR and quantify its robustness basin.
5. **Map** LQR/iLQR concepts onto RL vocabulary: value function ↔ cost-to-go, policy ↔ feedback gain, Q-function ↔ the $Q$-expansions below.

## Background

**LQR.** Linear dynamics $x_{k+1} = A x_k + B u_k$, quadratic cost $J = x_N^\top Q_f x_N + \sum_{k=0}^{N-1} \left( x_k^\top Q x_k + u_k^\top R u_k \right)$. Dynamic programming gives cost-to-go $V_k(x) = x^\top P_k x$ with the **backward Riccati recursion** ($P_N = Q_f$):

$$K_k = (R + B^\top P_{k+1} B)^{-1} B^\top P_{k+1} A$$

$$P_k = Q + A^\top P_{k+1} (A - B K_k)$$

and optimal policy $u_k = -K_k x_k$ — *linear feedback falls out of optimality; nobody put it in*. As $N \to \infty$, $P_k$ converges to the fixed point of the discrete algebraic Riccati equation (DARE); that's your unit test.

**iLQR** (16-745; Underactuated ch. 10 "Iterative LQR and DDP"). Nonlinear dynamics $x_{k+1} = f(x_k, u_k)$: iterate (1) roll out the current control sequence; (2) linearize along the rollout, $A_k = \partial f/\partial x$, $B_k = \partial f/\partial u$; (3) **backward pass** — expand the Bellman equation to second order around the rollout. With value expansion $V_x, V_{xx}$ at $k{+}1$:

$$Q_x = \ell_x + A_k^\top V_x' \qquad Q_u = \ell_u + B_k^\top V_x'$$

$$Q_{xx} = \ell_{xx} + A_k^\top V_{xx}' A_k \qquad Q_{uu} = \ell_{uu} + B_k^\top V_{xx}' B_k \qquad Q_{ux} = \ell_{ux} + B_k^\top V_{xx}' A_k$$

$$d_k = -\left(Q_{uu} + \mu I\right)^{-1} Q_u \qquad K_k = -\left(Q_{uu} + \mu I\right)^{-1} Q_{ux}$$

then $V_x = Q_x - K_k^\top Q_{uu} d_k$, $V_{xx} = Q_{xx} - K_k^\top Q_{uu} K_k$ (this simplified form is standard; 16-745 derives it). $\mu$ is **regularization** — it keeps $Q_{uu}$ positive definite far from the optimum, interpolating toward gradient descent. (4) **forward pass** — roll out $u_k = \bar u_k + \alpha d_k + K_k(x_k - \bar x_k)$, backtracking $\alpha \in \{1, \tfrac12, \tfrac14, \dots\}$ until actual cost decrease is a reasonable fraction of the predicted decrease $-\alpha Q_u^\top d - \tfrac{\alpha^2}{2} d^\top Q_{uu} d$. Note the feedback term $K_k(x_k - \bar x_k)$ *inside* the rollout: that's what separates iLQR from plain shooting and keeps long horizons from blowing up.

**TVLQR.** Given the converged trajectory $(\bar x_k, \bar u_k)$, run the LQR recursion on the time-varying linearizations $A_k, B_k$ about it. The resulting gains $K_k$ stabilize a *tube* around the trajectory against perturbations iLQR never saw.

| Source | Read for |
|---|---|
| CMU 16-745 lectures (LQR through iLQR/DDP; 2025 playlist at optimalcontrol.ri.cmu.edu) | the Riccati and backward-pass derivations done live, with the regularization/line-search war stories |
| Underactuated ch. 8 (LQR) | finite vs infinite horizon, time-varying LQR, what $P$ *is* |
| Underactuated ch. 10 (Trajectory Optimization) | where iLQR sits among shooting/transcription methods; DDP vs iLQR distinction |
| Tutorial §2.4 | why this beautiful machinery still hits the wall (models, contact) — the course's hinge argument |

## Part 0 — Cartpole dynamics you can trust (~1 h)

Everything downstream differentiates this function; get it right and prove it.

1. Implement cartpole continuous dynamics in numpy — state $x = (p, \theta, \dot p, \dot\theta)$, control = cart force, standard parameters ($m_c = 1$, $m_p = 0.1$, $l = 0.5$, $g = 9.81$; the equations are in Underactuated ch. 3). $\theta = \pi$ is upright.
2. Discretize with RK4 at $\Delta t = 0.02$ s: $f(x, u)$.
3. Analytic Jacobians $A = \partial f/\partial x$, $B = \partial f/\partial u$ (autograd via JAX is allowed *if* you also pass the FD check — the check is the point).
4. Test: central finite differences ($h = 10^{-6}$) vs analytic, max abs error < 1e-6, at 100 random states; energy conservation of the passive RK4 rollout (drift < 0.1% over 5 s).

**✅ Checkpoint:** both tests green. A hanging pole released from rest swings symmetrically in your animation.

## Part 1 — LQR (~2 h)

1. `lqr(A, B, Q, R, Qf, N) -> (Ks, Ps)`: the backward recursion, verbatim from the equations above.
2. Tests:
   - DARE residual: iterate to convergence, then check $P = Q + A^\top P A - A^\top P B (R + B^\top P B)^{-1} B^\top P A$ to < 1e-8; cross-check against `scipy.linalg.solve_discrete_are`.
   - Closed-loop spectral radius $\rho(A - BK_\infty) < 1$.
3. Stabilize the cartpole *linearized about upright*: simulate the **nonlinear** system under the linear policy from initial pole offsets of 5°, 20°, 40°.
4. Map the basin: sweep initial $(\theta_0, \dot\theta_0)$ on a grid, mark recover/fail — the picture that motivates iLQR (linear policy, local validity).

**✅ Checkpoint:** DARE and spectral tests green; 5° and 20° recover, 40° (or thereabouts) fails; the basin plot shows a clean bounded region around upright.

## Part 2 — iLQR swing-up (~3 h)

The centerpiece. From hanging ($\theta = 0$) to upright ($\theta = \pi$) — a task no linear controller can do.

1. `ilqr(x0, u_init, f, cost, N) -> (xs, us, Ks, info)`: horizon 3 s ($N = 150$), cost $Q = \mathrm{diag}(1, 10, 0.1, 0.1)$ (angle wrapped!), $R = 0.1$, $Q_f = 100\,Q$, `u_init` = small random noise.
2. Implement exactly: backward pass with $\mu$-regularization (start $10^{-6}$, ×10 on a non-PD $Q_{uu}$ or failed forward pass, ÷2 on success — the Levenberg–Marquardt schedule), forward pass with backtracking line search, convergence when $|\Delta J| < 10^{-6}$.
3. Instrument `info`: cost per iteration, $\alpha$ accepted, $\mu$ trace, predicted-vs-actual decrease ratio.
4. Tests:
   - **LQR equivalence:** on a linear-quadratic problem, iLQR must converge in one iteration to your Part 1 solution (< 1e-10). This single test catches most backward-pass bugs.
   - Cost strictly non-increasing across iterations.
5. Ablate: kill the line search (α = 1 always) and report what happens; kill regularization and report what happens. One paragraph each in `RESULTS.md`.
6. Animate the swing-up; plot the $(\theta, \dot\theta)$ phase portrait — the energy-pumping swings should be legible.

**✅ Checkpoint:** swing-up converges (typically < 100 iterations from noise init), final pole angle within 1e-3 of upright, cost curve monotone; LQR-equivalence test green.

## Part 3 — TVLQR robustness (~1.5 h)

1. Run the LQR recursion along the converged trajectory's $A_k, B_k$ (cost weights as in Part 2) → gains $K_k$.
2. Execute three controllers from perturbed initial states ($\pm 10\%$ on each state dimension, 50 random draws): (a) open-loop replay of $\bar u$, (b) TVLQR $u_k = \bar u_k - K_k (x_k - \bar x_k)$, (c) TVLQR that hands off to Part 1's infinite-horizon LQR once upright ($|\theta - \pi| < 0.1$).
3. Score: fraction reaching upright, final-state error distribution. Repeat with model mismatch ($m_p$ +20% in the *plant*, nominal in the controller).
4. Plot the trajectory tube: 50 perturbed TVLQR rollouts overlaid on the nominal in phase space.

**✅ Checkpoint:** open-loop replay fails for most ±10% draws; TVLQR+handoff recovers ≥ 90%; the tube plot shows convergence onto the nominal trajectory rather than divergence from it.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `dynamics.py`, `lqr.py`, `ilqr.py`, `tvlqr.py` | pure numpy/JAX, no control libraries; docstrings cross-reference the equations above by name |
| `tests/` | FD-Jacobian, DARE fixed point, spectral radius, LQR-equivalence, cost-monotonicity — all green |
| `plots/` | LQR basin, cost-vs-iteration, swing-up animation (GIF), phase portrait, TVLQR tube + recovery table |
| `RESULTS.md` | line-search/regularization ablation paragraphs; the robustness table; a closing section mapping every LQR/iLQR object to its RL counterpart (see objective 5) — this section gets reused verbatim when Lesson 08 starts |

## Done when

- [ ] All five test families green from a clean clone.
- [ ] iLQR swings up from rest with a monotone cost curve.
- [ ] TVLQR ≥ 90% recovery at ±10% perturbation, and the mismatch row of the table is filled in.
- [ ] `RESULTS.md`'s RL-mapping section exists and is specific (equation-to-equation, not vibes).

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
| Cost decreases then explodes | no line search, or feedback term missing in forward rollout | implement both; the ablation in Part 2.5 shows you the signatures |
| iLQR "works" but LQR-equivalence fails | index off-by-one in the recursion ($P_{k+1}$ vs $P_k$), or $\ell_{ux}$ transposed | the equivalence test localizes it: check shapes $Q_{ux} \in \mathbb{R}^{m\times n}$ |
| FD Jacobian check fails only for $\theta$ rows | forgot RK4 is the thing to differentiate, differentiated the continuous dynamics instead | differentiate through the integrator (chain rule over the 4 stages), or autodiff the discrete map |
| TVLQR diverges at trajectory end | gains from a short horizon with weak $Q_f$ | boost $Q_f$ or hand off to infinite-horizon LQR at the top (Part 3.2c) |

## Stretch

MuJoCo edition: swap the numpy plant for MuJoCo's cartpole (`mjx` or FD through `mj_step`), keep your solver unchanged, and report what contact-free simulator swapping does to convergence. Then read 16-745's MPC lecture and turn `ilqr.py` into a 10 Hz receding-horizon controller — 30 extra lines that preview exactly what Lesson 20's world-model policies learn end-to-end.

## References

- CMU 16-745 *Optimal Control & Reinforcement Learning* (Manchester), LQR/iLQR/DDP lectures. optimalcontrol.ri.cmu.edu.
- Tedrake, *Underactuated Robotics*, ch. 8 (LQR), ch. 10 (Trajectory Optimization). underactuated.csail.mit.edu.
- Li & Todorov, *Iterative Linear Quadratic Regulator Design for Nonlinear Biological Movement Systems* (ILQG origin), ICINCO 2004.
- Tassa, Erez, Todorov, *Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization*, IROS 2012 — the regularization/line-search recipe above.
- LeRobot team, *Robot Learning: A Tutorial*, §2.4. arXiv:2510.12403.
