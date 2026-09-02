# Lesson 05 — Optimal Control Sprint: LQR → iLQR → TVLQR

This lesson builds the control-theoretic backbone that the tutorial names but never teaches: exact optimal control for linear systems (LQR), iterated local optimization for nonlinear systems (iLQR), and stabilization of an optimized trajectory against disturbances (time-varying LQR). You will implement each on a cartpole, prove the implementations against their fixed points and against each other, and then remove the two safeguards inside iLQR one at a time after predicting how each removal will fail. The reason this material belongs in a robot-learning course is that the reinforcement-learning algorithms of Phase 3 solve exactly this problem with the dynamics sampled rather than given, and the mapping between the two vocabularies that you write at the end of this lesson is reused verbatim when Lesson 08 begins.

| | |
|---|---|
| **Phase** | 2 — Classical core |
| **Time** | 1–2 sessions (5–7 h desk time), all Mac-local (numpy; JAX or torch for autodiff; MuJoCo optional) |
| **Cost** | $0 |
| **Prerequisites** | 03–04 (Jacobians and linearization); comfort with chain-rule-heavy calculus |
| **Feeds into** | 08 (the RL ladder solves the same objective model-free, and the mapping table you write here is reused there verbatim), 20 (world-model policies are learned model-predictive control over exactly this machinery) |

## Learning objectives

After this lesson you can:

1. **Derive** the finite-horizon discrete LQR backward Riccati recursion and verify its fixed point against the algebraic Riccati equation.
2. **Explain** each term of the iLQR backward pass as a Bellman-equation Taylor expansion, and predict what breaks without regularization or line search, then demonstrate it.
3. **Quantify** the robustness basin a linear policy has around the upright equilibrium, and explain why that basin is the motivation for iLQR.
4. **Predict** TVLQR's recovery rate from perturbed starts and under model mismatch, and explain why it stabilizes states iLQR never visited.
5. **Map** every LQR and iLQR object onto its reinforcement-learning counterpart: value function to cost-to-go, policy to feedback gain, Q-function to the $Q$-expansions below.

## Principles

### Linear-quadratic regulation

The simplest optimal control problem has linear dynamics $x_{k+1} = A x_k + B u_k$ and a quadratic cost

$$J = x_N^\top Q_f x_N + \sum_{k=0}^{N-1} \left( x_k^\top Q x_k + u_k^\top R u_k \right),$$

where $Q$ and $Q_f$ penalize state deviation, $R$ penalizes control effort, and $N$ is the horizon. Dynamic programming solves it exactly. Define the cost-to-go $V_k(x)$ as the minimum cost achievable from state $x$ at step $k$. Because the cost is quadratic and the dynamics linear, the cost-to-go is itself quadratic, $V_k(x) = x^\top P_k x$, and the matrices $P_k$ satisfy a recursion that runs backward from the terminal condition $P_N = Q_f$:

$$K_k = (R + B^\top P_{k+1} B)^{-1} B^\top P_{k+1} A$$

$$P_k = Q + A^\top P_{k+1} (A - B K_k)$$

The optimal control at each step is $u_k = -K_k x_k$. Nothing in the problem statement asked for a linear feedback law; it emerges from the structure of the cost and dynamics. The recursion runs backward because the cost-to-go at step $k$ depends on the cost-to-go at step $k+1$, which is the definition of dynamic programming. As the horizon $N$ grows, $P_k$ for early steps converges to a fixed point, which is the solution of the discrete algebraic Riccati equation (DARE). That fixed point gives the infinite-horizon gain $K_\infty$, and checking your recursion against a library DARE solver is the natural correctness test.

### Iterative LQR

Most systems of interest are nonlinear, $x_{k+1} = f(x_k, u_k)$, and for those the exact solution above is unavailable. Iterative LQR (16-745; Underactuated ch. 10) gets around this by repeatedly solving a local LQR problem. Each iteration has four steps. First, roll out the current control sequence to obtain a nominal trajectory $(\bar x_k, \bar u_k)$. Second, linearize the dynamics along that trajectory, giving $A_k = \partial f/\partial x$ and $B_k = \partial f/\partial u$ at each step. Third, run a backward pass that expands the Bellman equation to second order around the nominal trajectory. With the value expansion $V_x, V_{xx}$ available at step $k+1$ (written with a prime), the expansion of the action-value function at step $k$ is

$$Q_x = \ell_x + A_k^\top V_x' \qquad Q_u = \ell_u + B_k^\top V_x'$$

$$Q_{xx} = \ell_{xx} + A_k^\top V_{xx}' A_k \qquad Q_{uu} = \ell_{uu} + B_k^\top V_{xx}' B_k \qquad Q_{ux} = \ell_{ux} + B_k^\top V_{xx}' A_k$$

and minimizing this quadratic over the control perturbation gives a feedforward term and a feedback gain,

$$d_k = -\left(Q_{uu} + \mu I\right)^{-1} Q_u \qquad K_k = -\left(Q_{uu} + \mu I\right)^{-1} Q_{ux},$$

after which the value expansion is propagated one step back as $V_x = Q_x - K_k^\top Q_{uu} d_k$ and $V_{xx} = Q_{xx} - K_k^\top Q_{uu} K_k$. This is the standard simplified form; 16-745 derives it in full. The term $\mu$ is a regularization parameter. Far from the optimum, the quadratic model can have an indefinite $Q_{uu}$, and adding $\mu I$ keeps it positive definite; as $\mu$ grows the step shrinks toward a gradient-descent step, so $\mu$ interpolates between Newton-like and gradient-like behaviour. Fourth, run a forward pass that rolls out the new control law

$$u_k = \bar u_k + \alpha d_k + K_k(x_k - \bar x_k),$$

backtracking on $\alpha \in \{1, \tfrac12, \tfrac14, \dots\}$ until the actual cost decrease is a reasonable fraction of the decrease predicted by the quadratic model, $-\alpha Q_u^\top d - \tfrac{\alpha^2}{2} d^\top Q_{uu} d$. Two features of the forward pass deserve attention. The line search on $\alpha$ protects against the quadratic model being wrong far from the nominal trajectory. The feedback term $K_k(x_k - \bar x_k)$ inside the rollout is what distinguishes iLQR from plain shooting: because the rollout corrects itself as it deviates from the nominal, long horizons do not blow up.

### Time-varying LQR along a trajectory

Once iLQR has converged to a trajectory $(\bar x_k, \bar u_k)$, replaying the control sequence $\bar u_k$ open-loop will fail under any perturbation, because nothing corrects the deviation. The remedy is to run the LQR recursion on the time-varying linearizations $A_k, B_k$ about the converged trajectory. The resulting gains $K_k$ define a controller $u_k = \bar u_k - K_k (x_k - \bar x_k)$ that stabilizes a tube around the trajectory. The tube covers perturbations that the optimizer never saw, because the gains come from the linearized dynamics rather than from any particular rollout.

### Why this lesson sits between classical control and reinforcement learning

Soft Actor-Critic in Lesson 08 minimizes the same discounted objective, but with the dynamics sampled from experience rather than supplied as $A_k$ and $B_k$; its value function plays the role of $V_k$ and its policy the role of $-K_k$. The world-model policies of Lesson 20 learn the dynamics model and then run a loop of this shape inside it. The tutorial's argument in §2.4, that this machinery is exact and elegant and still fails at the boundary where models and contact are involved, only carries weight once you have seen the machinery work, which is why the lesson has you build it rather than read about it.

**Carry forward**

- With a quadratic cost and linear dynamics, optimality produces a linear feedback law; $P_k$ is the Hessian of the cost-to-go, and the recursion runs backward because the cost-to-go at one step is defined in terms of the cost-to-go at the next.
- iLQR is repeated LQR on local linearizations, made safe by two guards: regularization $\mu$ keeps $Q_{uu}$ positive definite, and the line search keeps the second-order model honest. Removing either produces a characteristic divergence.
- The forward pass is closed-loop because the gain $K_k$ is applied inside the rollout, which is why iLQR tolerates long horizons where plain shooting cannot.
- TVLQR gains stabilize a tube around a trajectory rather than a single point, so they handle perturbations the optimizer never visited.
- The value curvature $V_{xx}$ and the stationarity condition $Q_u = 0$ have direct counterparts in reinforcement learning, which is the same problem solved without access to $A_k$ and $B_k$.

| Source | Read for |
|---|---|
| CMU 16-745 lectures (LQR through iLQR/DDP; 2025 playlist at optimalcontrol.ri.cmu.edu) | the Riccati and backward-pass derivations worked live, including the regularization and line-search failure modes |
| Underactuated ch. 8 (LQR) | finite versus infinite horizon, time-varying LQR, and what $P$ represents |
| Underactuated ch. 10 (Trajectory Optimization) | where iLQR sits among shooting and transcription methods; the distinction between DDP and iLQR |
| Tutorial §2.4 | why this machinery still fails at models and contact, which is the argument on which the rest of the course turns |

## Exercise 1 — Cartpole dynamics with checked Jacobians [Build]

Everything downstream of this exercise differentiates the discrete dynamics function, so the function and its Jacobians must be correct before anything else is built. The principle at stake is that a Jacobian that has not been checked against finite differences cannot be trusted, no matter how it was produced. Write the specification below for `dynamics.py` and have an AI tool draft it.

- Continuous cartpole dynamics in numpy, with state $x = (p, \theta, \dot p, \dot\theta)$ and the cart force as the control. Use the standard parameters $m_c = 1$, $m_p = 0.1$, $l = 0.5$, $g = 9.81$; the equations are in Underactuated ch. 3. The convention is that $\theta = \pi$ is upright.
- An RK4 discretization at $\Delta t = 0.02$ s, exposed as `f(x, u)`.
- Jacobians $A = \partial f/\partial x$ and $B = \partial f/\partial u$ of the *discrete* map. Automatic differentiation (JAX or torch) is fine; analytic Jacobians are not required.
- The check, as part 1 of `checks.py`: central finite differences with $h = 10^{-6}$ against the returned Jacobians, with maximum absolute error below $10^{-6}$ at 100 random states.
- A short animation of a hanging pole released from rest.

**✅ Checkpoint:** the finite-difference check passes at all 100 states, and the released pole swings symmetrically in the animation.

## Exercise 2 — LQR and the algebraic Riccati fixed point [Build]

In this exercise you implement the backward Riccati recursion and confirm that its fixed point is the DARE solution. The point of the check is that the recursion, run for long enough, must reproduce what a general-purpose solver computes; if it does not, an index or a transpose is wrong. Write the specification for `lqr.py`:

- `lqr(A, B, Q, R, Qf, N) -> (Ks, Ps)`, implementing the backward recursion exactly as written in the Principles section.
- The check, as part 2 of `checks.py`: iterate the recursion to convergence, then verify that the DARE residual $P = Q + A^\top P A - A^\top P B (R + B^\top P B)^{-1} B^\top P A$ is below $10^{-8}$, and that $P$ matches `scipy.linalg.solve_discrete_are`.

**✅ Checkpoint:** the DARE residual is below $10^{-8}$ and the scipy cross-check agrees.

## Exercise 3 — The basin of attraction of a linear policy [Predict → Run]

A linear policy is exactly optimal for the linearized system and only approximately right for the nonlinear plant, so it works within some neighbourhood of the equilibrium and fails outside it. In this exercise you map that neighbourhood for the cartpole. The picture you produce is the motivation for iLQR: a linear controller can balance the pole but cannot swing it up.

1. Linearize the cartpole about the upright equilibrium and compute the infinite-horizon gain $K_\infty$.
2. Before running, write in `RESULTS.md` which of the initial pole offsets 5°, 20°, and 40° (all from rest) you expect to recover under $u = -K_\infty x$ applied to the nonlinear plant, and why the largest one fails. Name the mechanism you expect: linearization error, insufficient control authority, or both.
3. Run all three. Then sweep the initial $(\theta_0, \dot\theta_0)$ on a grid, mark each start as recovered or failed, and save the result as `plots/lqr_basin.png`.
4. Reconcile your prediction with the outcome in `RESULTS.md`.

**✅ Checkpoint:** the 5° and 20° starts recover, the 40° start (or one near it) fails, and the basin plot shows a bounded region around upright.

## Exercise 4 — iLQR swing-up [Build]

This exercise implements iLQR and uses it to swing the pole from hanging ($\theta = 0$) to upright ($\theta = \pi$), which no linear controller can do. The check that matters most is the equivalence test: on a problem that is already linear-quadratic, iLQR must recover the LQR solution in a single iteration, because the quadratic model is then exact. That one test catches most backward-pass errors. Write the specification for `ilqr.py`:

- `ilqr(x0, u_init, f, cost, N) -> (xs, us, Ks, info)`, with a horizon of 3 s ($N = 150$), cost $Q = \mathrm{diag}(1, 10, 0.1, 0.1)$ with the angle wrapped, $R = 0.1$, $Q_f = 100\,Q$, and `u_init` set to small random noise.
- A backward pass with $\mu$-regularization on the Levenberg–Marquardt schedule: start at $10^{-6}$, multiply by 10 when $Q_{uu}$ is not positive definite or the forward pass fails, and divide by 2 on success. Use a Cholesky factorization and catch its failure; do not call `np.linalg.inv`.
- A forward pass with backtracking line search, declaring convergence when $|\Delta J| < 10^{-6}$.
- An `info` record with the cost per iteration, the accepted $\alpha$, the $\mu$ trace, and the ratio of predicted to actual decrease.
- The check, as part 3 of `checks.py`: on a linear-quadratic problem, iLQR converges in **one iteration** to Exercise 2's solution, with error below $10^{-10}$; and the cost is non-increasing across iterations on the swing-up.

You may type this kernel yourself if you wish; the annotation in Exercise 5 is the requirement either way.

**✅ Checkpoint:** the swing-up converges (typically in fewer than 100 iterations from the noise initialization), the final pole angle is within $10^{-3}$ of upright, the cost curve is monotone, and the LQR-equivalence check passes. Animate the swing-up and plot the $(\theta, \dot\theta)$ phase portrait; the energy-pumping swings should be visible.

## Exercise 5 — Annotate the backward pass [Read the kernel]

Every line of the backward pass is a term of a second-order Bellman expansion, and the purpose of this exercise is to make sure you can say which. Annotating the code against the equations is how you convert an implementation that works into one you understand.

1. In `ilqr.py`, annotate every line of the backward pass with the Principles equation it implements ($Q_x$, $Q_u$, $Q_{xx}$, $Q_{uu}$, $Q_{ux}$, $d_k$, $K_k$, $V_x$, $V_{xx}$), and every line of the $\mu$ schedule with the event that triggers it.
2. Annotate the forward pass: identify the feedforward step, the feedback term, the predicted decrease, and the line where the acceptance test compares the predicted decrease to the actual one.
3. Check the shapes in your annotations: $Q_{ux} \in \mathbb{R}^{m \times n}$, $K_k \in \mathbb{R}^{m \times n}$, and $d_k \in \mathbb{R}^m$.

**✅ Checkpoint:** the annotated file is committed, and the LQR-equivalence check still passes after any edits the annotation prompted.

## Exercise 6 — Remove regularization and line search [Predict → Run]

Regularization and line search each prevent a specific kind of divergence, and you should be able to say in advance what each divergence looks like. In this exercise you switch each guard off in turn and compare the failure to your prediction.

1. Before running, write in `RESULTS.md` the pattern you expect in the cost-versus-iteration curve and in the `info` record for each ablation: (a) the line search removed, so that $\alpha = 1$ always; (b) regularization removed, so that $\mu = 0$ with no schedule. For each, name the quantity that stops being trustworthy and say whether the failure appears early (far from the optimum) or late.
2. Run each ablation from the same noise initialization as Exercise 4.
3. Reconcile in `RESULTS.md`, one paragraph per ablation, comparing the prediction to what happened.

**✅ Checkpoint:** both ablation curves are recorded, and each paragraph names the mechanism and says whether it matched the prediction.

## Exercise 7 — TVLQR robustness under perturbation and mismatch [Predict → Run]

Gains computed along a trajectory stabilize a tube around it, including states that the optimizer never visited. This exercise measures the tube by perturbing the initial state and, separately, by changing the plant's parameters without telling the controller.

1. Run the LQR recursion along the converged trajectory's $A_k, B_k$ with the same cost weights as Exercise 4, producing the gains $K_k$.
2. Before running, write in `RESULTS.md` your expected fraction of 20 random draws, each with a $\pm 10\%$ perturbation on every state dimension, that reach upright under (a) open-loop replay of $\bar u$ and (b) TVLQR, $u_k = \bar u_k - K_k (x_k - \bar x_k)$. Then predict how (b) changes when the plant's pole mass is increased by 20% while the controller keeps the nominal value.
3. Run both controllers on the 20 nominal draws and on the 20 mismatched draws. Record the fraction reaching upright and the distribution of final-state error. Plot the tube: the perturbed TVLQR rollouts overlaid on the nominal trajectory in phase space.
4. Reconcile in `RESULTS.md` with a 2×2 recovery table (controller × plant condition).

**✅ Checkpoint:** open-loop replay fails for most $\pm 10\%$ draws; TVLQR recovers at least 90% of the nominal draws; the mismatch row is filled in; and the tube plot shows rollouts converging onto the nominal trajectory rather than diverging from it. If TVLQR diverges near the end of the trajectory, see the Pitfalls table; the handoff to an infinite-horizon controller is described under Going deeper.

## Exercise 8 — The correspondence with reinforcement learning [Write]

The idea that carries into Phase 3 is that reinforcement learning solves this same problem with $A_k$ and $B_k$ sampled from experience instead of given. Write the closing section of `RESULTS.md` as a table that pairs each object here with its counterpart there, equation to equation: the value function with the cost-to-go $V_k$, the policy with $-K_k$, the Q-function with the $Q$-expansions, and then the entries for $Q_u = 0$, $V_{xx}$, the line search, and $\mu$. This section is reused verbatim when Lesson 08 starts.

**✅ Checkpoint:** every row names a specific object on both sides of the correspondence.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `dynamics.py`, `lqr.py`, `ilqr.py`, `tvlqr.py` | pure numpy plus autodiff, no control libraries; `ilqr.py` carries the Exercise 5 annotations |
| `checks.py` | prints the FD-Jacobian error, the DARE residual, the LQR-equivalence error, and the cost-monotonicity result from one command |
| `plots/` | LQR basin, cost-vs-iteration (with both ablation curves), swing-up animation (GIF), phase portrait, TVLQR tube and recovery table |
| `RESULTS.md` | Exercise 3, 6, and 7 predictions with reconciliations; the ablation paragraphs; the robustness table; the RL-mapping table |

## Done when

- [ ] `checks.py` passes: FD error below $10^{-6}$, DARE residual below $10^{-8}$, LQR-equivalence error below $10^{-10}$, cost monotone.
- [ ] iLQR swings up from rest with a monotone cost curve, and `ilqr.py` is annotated line by line.
- [ ] Both ablation predictions were written before the runs and reconciled after them.
- [ ] TVLQR recovers at least 90% of $\pm 10\%$ perturbations, and the mismatch row is filled in.
- [ ] The RL-mapping table in `RESULTS.md` exists and is specific, equation to equation.

## Self-check

1. Why does the Riccati recursion run backward, and what does $P_k$ mean at an interior step?
2. In the backward pass, what exactly does $\mu$ regularize against, and why does a large $\mu$ turn the step into gradient descent?
3. Why does the forward pass need the feedback term $K_k(x_k - \bar x_k)$ even though $d_k$ already encodes the improvement?
4. iLQR versus DDP: which second-order term does iLQR drop, and when does that matter?
5. State the exact correspondence: $V_{xx} \leftrightarrow$ ?, $Q_u = 0 \leftrightarrow$ ? in reinforcement-learning terms.
6. Why can TVLQR stabilize states that iLQR never visited?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Swing-up converges to a spin, not a balance | angle not wrapped in the cost | penalize $1 - \cos(\theta - \pi)$, or wrap the error to $[-\pi, \pi]$ |
| Backward pass throws on `inv(Quu)` | $Q_{uu}$ not positive definite far from the optimum | this is the case $\mu$ handles; use Cholesky with a caught failure, never `np.linalg.inv` |
| Cost decreases then explodes | no line search, or the feedback term missing from the forward rollout | implement both; Exercise 6 shows you each signature |
| iLQR appears to work but the LQR-equivalence check fails | index off by one in the recursion ($P_{k+1}$ vs $P_k$), or $\ell_{ux}$ transposed | the equivalence test localizes it; check the shape $Q_{ux} \in \mathbb{R}^{m\times n}$ |
| FD Jacobian check fails only for the $\theta$ rows | the continuous dynamics were differentiated instead of the RK4 map | differentiate through the integrator by autodiffing the discrete map |
| TVLQR diverges at the end of the trajectory | gains from a short horizon with a weak $Q_f$ | raise $Q_f$, or hand off to the infinite-horizon LQR near the top (Going deeper) |

## Going deeper

- **Handoff controller.** Build a TVLQR controller that hands off to Exercise 2's infinite-horizon LQR once the pole is near upright ($|\theta - \pi| < 0.1$), and compare its recovery to plain TVLQR on the same 20 draws.
- **Extra checks.** Energy conservation of the passive RK4 rollout (drift below 0.1% over 5 s); the closed-loop spectral radius $\rho(A - BK_\infty) < 1$; analytic Jacobians compared against autodiff.
- **MuJoCo edition.** Replace the numpy plant with MuJoCo's cartpole (`mjx`, or finite differences through `mj_step`), keep the solver unchanged, and report what swapping in a contact-free simulator does to convergence. Then read 16-745's MPC lecture and turn `ilqr.py` into a 10 Hz receding-horizon controller. The addition is roughly 30 lines, and it previews what Lesson 20's world-model policies learn end to end.

## References

- CMU 16-745 *Optimal Control & Reinforcement Learning* (Manchester), LQR/iLQR/DDP lectures. optimalcontrol.ri.cmu.edu.
- Tedrake, *Underactuated Robotics*, ch. 8 (LQR), ch. 10 (Trajectory Optimization). underactuated.csail.mit.edu.
- Li & Todorov, *Iterative Linear Quadratic Regulator Design for Nonlinear Biological Movement Systems* (ILQG origin), ICINCO 2004.
- Tassa, Erez, Todorov, *Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization*, IROS 2012 (the regularization and line-search recipe used above).
- LeRobot team, *Robot Learning: A Tutorial*, §2.4. arXiv:2510.12403.
