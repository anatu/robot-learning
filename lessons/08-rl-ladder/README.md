# Lesson 08 — The RL Ladder: REINFORCE → DQN → SAC

Build the three-rung algorithm ladder from scratch — ending at the exact algorithm LeRobot's HIL-SERL runs — so that the tutorial's Eqs. 11–17 stop being assertions and become code you've debugged.

| | |
|---|---|
| **Phase** | 3 — Reinforcement learning |
| **Time** | ~4 sessions desk time (12–16 h) + overnight compute for HalfCheetah (Mac `mps`) or ~1 h on a rented 4090 |
| **Cost** | $0–3 (everything but HalfCheetah runs on the Mac in minutes) |
| **Prerequisites** | 05 (you know what an optimal controller looks like when the model is known — RL is the same problem with the model sampled, not given) |
| **Feeds into** | 09 (this SAC is the engine RLPD modifies), 10 (HIL-SERL = this SAC + buffers + humans), 11 (this SAC trains every DR arm) |

## Learning objectives

After this lesson you can:

1. **Derive** the policy gradient theorem and explain why reward-to-go and a baseline reduce variance without adding bias.
2. **Implement** DQN and demonstrate — with a plot, not a citation — why the target network is load-bearing.
3. **Derive** SAC's three updates (twin critics, reparameterized actor, temperature) from the max-entropy objective, and map each to tutorial Eqs. 14–17.
4. **Reproduce** seeded learning curves that clear fixed reward thresholds, with a test suite that fails if they regress.
5. **Rank** the three algorithms on sample efficiency and explain the mechanism behind the ordering (on-policy data burn vs off-policy reuse).

## Background

**One problem, three estimators.** Everything here maximizes $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_t \gamma^t r_t]$. The rungs differ in what they estimate and how much they reuse data.

**Rung 1 — REINFORCE.** The policy gradient theorem: $\nabla_\theta J = \mathbb{E}_{\tau}\big[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\, \Psi_t\big]$. With $\Psi_t$ = full return, the estimator is unbiased and unusable (variance grows with horizon). Two surgeries, both bias-free: *reward-to-go* ($\Psi_t = \sum_{t' \ge t} \gamma^{t'-t} r_{t'}$ — actions can't cause past rewards, causality removes those terms from the variance) and a *baseline* ($\Psi_t - b(s_t)$; $\mathbb{E}[\nabla \log \pi \cdot b(s)] = 0$ because $\int \nabla \pi = \nabla \int \pi = 0$). This is CS 285 Lecture 5, and tutorial Eq. 11's ancestor.

**Rung 2 — DQN.** Value-based: fit $Q^*$ via the Bellman optimality backup, act greedily. The TD loss (tutorial Eq. 12) regresses $Q_\phi(s,a)$ onto $r + \gamma \max_{a'} Q_{\phi^-}(s',a')$. The tabular Bellman operator is a $\gamma$-contraction; add function approximation + bootstrapping + off-policy data (the deadly triad) and convergence guarantees evaporate. The two stabilizers you'll ablate: the replay buffer (decorrelates the i.i.d.-assuming SGD batches) and the target network $\phi^-$ (freezes the regression target so you're not chasing your own updates). Double-DQN decouples argmax from evaluation to cut the $\max$-induced overestimation.

**Rung 3 — SAC.** Continuous actions kill the $\max_{a'}$; max-entropy RL replaces it. Objective: $J = \sum_t \mathbb{E}[r_t + \alpha \mathcal{H}(\pi(\cdot|s_t))]$ (tutorial Eq. 14). The three updates:
- *Critics* (Eq. 15): twin $Q_{\phi_1}, Q_{\phi_2}$ regress onto $r + \gamma\big(\min_i Q_{\phi_i^-}(s', a') - \alpha \log \pi(a'|s')\big)$, $a' \sim \pi$ — the min fights the same overestimation double-DQN fights.
- *Actor* (Eq. 16): minimize $\mathbb{E}_{s}\big[\alpha \log \pi_\theta(a|s) - \min_i Q_{\phi_i}(s,a)\big]$ with the reparameterization trick ($a = \tanh(\mu_\theta(s) + \sigma_\theta(s) \epsilon)$, so the gradient flows through $Q$).
- *Temperature* (Eq. 17): adjust $\alpha$ by gradient descent on $\mathbb{E}[-\alpha(\log \pi(a|s) + \bar{\mathcal{H}})]$ with target entropy $\bar{\mathcal{H}} = -\dim(\mathcal{A})$.

Off-policy replay is why SAC is the real-robot workhorse: every transition is reused hundreds of times, and Lessons 09–10 exist because of that property.

| Source | Read for |
|---|---|
| Tutorial §3.1–3.2 (Eqs. 11–17) | the equation numbering your docstrings will cite |
| CS 285 (Fa23) Lectures 4–6 | MDP formalism, policy gradient variance analysis, actor-critic bridge |
| CS 285 (Fa23) Lectures 7–8 | value-based methods, why deep Q-learning is unstable in practice |
| Haarnoja et al. 2018 (arXiv:1801.01290) + the applications paper (1812.05905) | the three SAC updates; the auto-temperature trick is in the second paper |

**Hyperparameter reference** (the values the Parts below assume; deviations go in `RESULTS.md`):

| | REINFORCE | DQN | SAC |
|---|---|---|---|
| Env | `CartPole-v1` | `LunarLander-v3` | `Pendulum-v1` → `HalfCheetah-v5` |
| Network | 2×64 tanh | 2×256 ReLU | 2×256 ReLU (all nets) |
| lr | 1e-2 | 1e-3 → 1e-4 cosine | 3e-4 (actor, critics, $\log\alpha$) |
| Batch | 10 episodes | 128 | 256 |
| Buffer | — (on-policy) | 1e5 | 1e6 |
| $\gamma$ | 0.99 | 0.99 | 0.99 |
| Target update | — | hard, every 1k steps | Polyak $\tau{=}0.005$ per step |
| Exploration | on-policy stochastic | $\epsilon$: 1.0→0.05 over 50k | max-ent, auto-$\alpha$, $\bar{\mathcal{H}}{=}{-}\dim(\mathcal{A})$ |
| Threshold (3 seeds) | ≥ 475 / 100 eps | ≥ 200 | ≥ −200; HalfCheetah ≥ 6000 @ 500k |

## Part 1 — REINFORCE on CartPole (Mac, ~3 h desk, minutes to train)

Produces the variance-reduction ablation that justifies everything actor-critic.

1. Scaffold the shared plumbing you'll reuse all lesson: `core/` with `MLP`, `ReplayBuffer`, `Logger` (CSV + W&B), `set_seed(seed)` seeding python/numpy/torch/env.
2. Implement REINFORCE for `CartPole-v1`: 2×64 tanh MLP policy, Adam lr 1e-2, batch = 10 episodes, $\gamma = 0.99$. Three estimator variants behind one flag: (a) full-return, (b) reward-to-go, (c) reward-to-go + learned value baseline (fit by MSE on returns).
3. Run all three × 3 seeds (0, 1, 2). Log per-update gradient norm std as your variance proxy.
4. Plot mean ± std learning curves and the variance proxy.

**✅ Checkpoint:** variant (c) reaches avg return ≥ 475 (over 100 eval episodes) fastest and its variance proxy is visibly lowest; (a) is the noisiest. If (a) beats (c), your baseline is leaking bias — check it's detached from the policy loss.

## Part 2 — DQN on LunarLander (Mac, ~4 h desk, ~20 min/run)

Produces the instability ablation — the plot that explains why every later algorithm carries a target network.

1. `pip install "gymnasium[box2d]"` (needs `swig`; see Pitfalls). Env: `LunarLander-v3`.
2. Implement DQN: 2×256 ReLU MLP, replay buffer 1e5, batch 128, lr 1e-3 → 1e-4 cosine, $\gamma = 0.99$, $\epsilon$-greedy 1.0 → 0.05 over 50k steps, target sync every 1k steps.
3. Three arms × 3 seeds: (a) DQN, (b) DQN with target network *disabled* ($\phi^- = \phi$), (c) double-DQN.
4. For each arm, log both the learning curve and $\mathbb{E}[\max_a Q(s,a)]$ on a fixed probe-state batch vs the empirically observed discounted return.

**✅ Checkpoint:** arm (a) clears avg return ≥ 200; arm (b) shows Q-value blow-up or oscillation on the probe plot; arm (c)'s Q-estimates track observed returns more closely than (a)'s. That gap *is* overestimation bias — screenshot it for `RESULTS.md`.

## Part 3 — SAC on Pendulum, then HalfCheetah (Mac overnight or 4090 ~1 h)

Produces the course's reference SAC — the exact code Lessons 09–11 import.

1. Implement SAC per the Background updates and the reference table: 1 gradient step per env step, tanh-squashed Gaussian with the log-prob correction term (the $\sum \log(1 - \tanh^2)$ Jacobian — the classic silent bug). The critic target, spelled out so there's no ambiguity about where gradients flow (nothing inside `target` gets one):
   ```python
   with torch.no_grad():
       a2, logp2 = actor.sample(s2)                      # fresh action from the CURRENT actor
       q_targ = torch.min(q1_targ(s2, a2), q2_targ(s2, a2))
       target = r + gamma * (1 - done) * (q_targ - alpha * logp2)
   loss_q = F.mse_loss(q1(s, a), target) + F.mse_loss(q2(s, a), target)
   ```
2. Sanity env first: `Pendulum-v1`, 3 seeds — converges in ~30k steps.
3. Main run: `HalfCheetah-v5` (`gymnasium[mujoco]`), 500k steps × 3 seeds. On `mps` this is the overnight run; on a 4090 ~1 h.
4. Docstring every loss with its tutorial equation number (Eqs. 14–17); this is the cross-reference future-you greps for.
5. Ablation (pick one, note the other as known results): single critic vs twin; fixed $\alpha \in \{0.05, 0.2\}$ vs auto.

**✅ Checkpoint:** Pendulum avg return ≥ −200 within 30k steps, all seeds. HalfCheetah ≥ 6000 by 500k steps (published SAC lands ~10k by 1M — you're on the curve, not at its end). Auto-$\alpha$: entropy starts high and decays toward $-\dim(\mathcal{A})$; if it crashes immediately, see Pitfalls.

## Part 4 — The regression suite (~1 h)

1. `pytest` markers: `-m fast` (loss-shape and gradient-flow unit tests, seconds) and `-m slow` (thresholded training runs at fixed seeds: CartPole ≥ 475, LunarLander ≥ 200, Pendulum ≥ −200).
2. Unit tests worth having: baseline detachment (Part 1), target-net freeze (no grads flow to $\phi^-$), tanh log-prob correction against `torch.distributions.TransformedDistribution`, buffer FIFO + dtype behavior.

**✅ Checkpoint:** `pytest -m fast` green in < 30 s; `pytest -m slow` green in < 1 h on the Mac.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `reinforce/`, `dqn/`, `sac/`, `core/` | each algorithm one module; SAC importable as `from sac import SACAgent` (Lessons 09–11 contract) |
| `plots/` | variance ablation, target-net/overestimation ablation, SAC curves — all mean ± std over 3 seeds |
| `tests/` | fast + slow suites as specced, green |
| `RESULTS.md` | the three checkpoint numbers; the sample-efficiency ranking with env-steps-to-threshold; ≤ 10 sentences on *why* the ordering holds |

## Done when

- [ ] All three agents clear their thresholds on 3/3 seeds via `pytest -m slow`.
- [ ] The two ablation plots exist and show the textbook pathologies (variance, overestimation).
- [ ] Every loss function's docstring cites its tutorial equation number.
- [ ] HalfCheetah SAC ≥ 6000 @ 500k steps with curves in W&B.

## Self-check

1. Prove in three lines that subtracting $b(s_t)$ leaves the policy gradient unbiased. Where does the argument break if $b$ depends on $a_t$?
2. Name the three legs of the deadly triad and which one the target network amputates.
3. Why does SAC take a min over two critics rather than an average?
4. Where exactly does the $\log(1 - \tanh^2(u))$ term come from, and what silently goes wrong without it?
5. REINFORCE discards data after one update; SAC reuses it for ~hundreds of updates. What property of the SAC objective makes that legal?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `box2d` install fails | missing `swig` | `brew install swig` then reinstall `gymnasium[box2d]` |
| CartPole flatlines at ~20 | missing reward-to-go sign/discount bug | print $\Psi_t$ for one episode by hand and compare |
| DQN diverges even with target net | lr too high for LunarLander's reward scale | drop to 5e-4; clip grads at 10 |
| SAC entropy collapses to $\bar{\mathcal{H}}$ instantly, no exploration | temperature lr too high or log-$\alpha$ not clamped | lr 3e-4 on $\log \alpha$; init $\alpha = 0.2$ |
| SAC actor loss NaN | tanh log-prob correction at $|u| \gg 1$ | use the numerically stable form: $2(\log 2 - u - \mathrm{softplus}(-2u))$ |
| HalfCheetah stuck < 2000 | 1-step-per-env-step not actually happening, or obs not normalized | log the update/step ratio; running mean-std normalize observations |
| `mps` slower than CPU on small MLPs | kernel launch overhead dominates | batch ≥ 256 or just use `cpu` for CartPole/LunarLander |

## Stretch

Add a pixel-input DQN (frame-stack 4, CNN torso) on `ALE/Pong-v5` — the classic rite of passage — or implement n-step returns in SAC and measure the sample-efficiency delta (foreshadows the UTD discussion in Lesson 09).

## References

- Sutton & Barto ch. 13 (policy gradient theorem); Schulman et al. 2016 (GAE) for the $\Psi_t$ taxonomy.
- Mnih et al. 2015 (DQN); van Hasselt et al. 2016 (double DQN).
- Haarnoja et al. 2018, arXiv:1801.01290 + arXiv:1812.05905 (SAC + auto-temperature).
- LeRobot team, *Robot Learning: A Tutorial*, §3.1–3.2, Eqs. 11–17. arXiv:2510.12403.
