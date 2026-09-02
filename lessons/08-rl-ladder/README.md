# Lesson 08 — The RL Ladder: REINFORCE → DQN → SAC

Three estimators of one objective, read line by line and broken on purpose — so the tutorial's Eqs. 11–17 become code you can annotate, predict, and debug, ending at the exact algorithm LeRobot's HIL-SERL runs.

| | |
|---|---|
| **Phase** | 3 — Reinforcement learning |
| **Time** | ~6–8 h desk time (AI-assisted); compute: minutes for CartPole/Pendulum, ~20 min per LunarLander run on the Mac; HalfCheetah (optional) overnight on `mps` or ~1 h on a rented 4090 |
| **Cost** | $0 (≤ $3 if you run the optional HalfCheetah arm in the cloud) |
| **Prerequisites** | 05 (you know what an optimal controller looks like when the model is known — RL is the same problem with the model sampled, not given) |
| **Feeds into** | 09 (patches this lesson's `sac.py` for RLPD), 10 (HIL-SERL = this SAC + buffers + humans), 11 (this `sac.py` trains every DR arm) |

## Learning objectives

After this lesson you can:

1. **Derive** the policy gradient theorem and explain why reward-to-go and a baseline reduce variance without adding bias.
2. **Predict and demonstrate** — with a plot, not a citation — why the target network is load-bearing in DQN.
3. **Map** every line of SAC's three updates (twin critics, reparameterized actor, temperature) to tutorial Eqs. 14–17 and to the max-entropy objective they come from.
4. **Diagnose** a silent SAC bug — the missing tanh log-prob correction — from its training symptom alone.
5. **Rank** the three algorithms on sample efficiency from seeded curves and explain the mechanism behind the ordering.

## Principles

**One problem, three estimators.** Everything here maximizes $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_t \gamma^t r_t]$. The rungs differ in what they estimate and how much they reuse data.

**Rung 1 — REINFORCE.** The policy gradient theorem: $\nabla_\theta J = \mathbb{E}_{\tau}\big[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\, \Psi_t\big]$. With $\Psi_t$ = full return, the estimator is unbiased and unusable (variance grows with horizon). Two surgeries, both bias-free: *reward-to-go* ($\Psi_t = \sum_{t' \ge t} \gamma^{t'-t} r_{t'}$ — actions can't cause past rewards, so causality removes those terms from the variance) and a *baseline* ($\Psi_t - b(s_t)$; $\mathbb{E}[\nabla \log \pi \cdot b(s)] = 0$ because $\int \nabla \pi = \nabla \int \pi = 0$). This is CS 285 Lecture 5, and tutorial Eq. 11's ancestor.

**Rung 2 — DQN.** Value-based: fit $Q^*$ via the Bellman optimality backup, act greedily. The TD loss (tutorial Eq. 12) regresses $Q_\phi(s,a)$ onto $r + \gamma \max_{a'} Q_{\phi^-}(s',a')$. The tabular Bellman operator is a $\gamma$-contraction; add function approximation + bootstrapping + off-policy data (the deadly triad) and convergence guarantees evaporate. Two stabilizers: the replay buffer (decorrelates the i.i.d.-assuming SGD batches) and the target network $\phi^-$ (freezes the regression target so you're not chasing your own updates). Double-DQN decouples argmax from evaluation to cut the $\max$-induced overestimation.

**Rung 3 — SAC.** Continuous actions kill the $\max_{a'}$; max-entropy RL replaces it. Objective: $J = \sum_t \mathbb{E}[r_t + \alpha \mathcal{H}(\pi(\cdot|s_t))]$ (tutorial Eq. 14). The three updates:
- *Critics* (Eq. 15): twin $Q_{\phi_1}, Q_{\phi_2}$ regress onto $r + \gamma\big(\min_i Q_{\phi_i^-}(s', a') - \alpha \log \pi(a'|s')\big)$, $a' \sim \pi$ — the min fights the same overestimation double-DQN fights.
- *Actor* (Eq. 16): minimize $\mathbb{E}_{s}\big[\alpha \log \pi_\theta(a|s) - \min_i Q_{\phi_i}(s,a)\big]$ with the reparameterization trick ($a = \tanh(\mu_\theta(s) + \sigma_\theta(s) \epsilon)$, so the gradient flows through $Q$). The tanh squash changes the density: $\log \pi(a|s) = \log \mathcal{N}(u) - \sum_i \log(1 - \tanh^2 u_i)$ — the Jacobian term every from-scratch SAC forgets once.
- *Temperature* (Eq. 17): adjust $\alpha$ by gradient descent on $\mathbb{E}[-\alpha(\log \pi(a|s) + \bar{\mathcal{H}})]$ with target entropy $\bar{\mathcal{H}} = -\dim(\mathcal{A})$.

The critic target, spelled out so there is no ambiguity about where gradients flow (nothing inside `target` gets one):
```python
with torch.no_grad():
    a2, logp2 = actor.sample(s2)                      # fresh action from the CURRENT actor
    q_targ = torch.min(q1_targ(s2, a2), q2_targ(s2, a2))
    target = r + gamma * (1 - done) * (q_targ - alpha * logp2)
loss_q = F.mse_loss(q1(s, a), target) + F.mse_loss(q2(s, a), target)
```

Off-policy replay is why SAC is the real-robot workhorse: every transition is reused hundreds of times, and Lessons 09–10 exist because of that property.

**Why you read a reference implementation instead of writing one.** The bugs that matter in RL are semantic (a detached baseline, a frozen target, a missing Jacobian term), not structural. CleanRL's single-file scripts are the canonical readable versions of these algorithms; annotating one against the equations, then breaking it deliberately, exercises exactly the semantic layer. A from-scratch rewrite is under Going deeper.

**Carry forward**

- Policy gradient: reward-to-go and a state-only baseline cut variance and add no bias; a baseline that depends on $a_t$ does add bias.
- Deadly triad = function approximation + bootstrapping + off-policy data; the target network amputates the bootstrapping leg's feedback loop.
- SAC = twin-min critics + reparameterized tanh-Gaussian actor (with the $\log(1-\tanh^2)$ correction) + auto-temperature to a target entropy of $-\dim(\mathcal{A})$.
- Sample-efficiency ordering follows data reuse: on-policy (burn after one update) ≪ off-policy replay.

| Source | Read for |
|---|---|
| Tutorial §3.1–3.2 (Eqs. 11–17) | the equation numbering your annotations cite |
| CS 285 (Fa23) Lectures 4–6 | MDP formalism, policy gradient variance analysis, actor-critic bridge |
| CS 285 (Fa23) Lectures 7–8 | value-based methods, why deep Q-learning is unstable in practice |
| Haarnoja et al. 2018 (arXiv:1801.01290) + the applications paper (1812.05905) | the three SAC updates; the auto-temperature trick is in the second paper |
| CleanRL docs (docs.cleanrl.dev): `dqn.py`, `sac_continuous_action.py` pages | the documented defaults and benchmark curves of the files you vendor |

**Hyperparameter reference** (the values the exercises assume; deviations from the vendored files' defaults go in `RESULTS.md`):

| | REINFORCE | DQN | SAC |
|---|---|---|---|
| Env | `CartPole-v1` | `LunarLander-v3` | `Pendulum-v1` → `HalfCheetah-v5` (optional) |
| Network | 2×64 tanh | 2×256 ReLU | 2×256 ReLU (all nets) |
| lr | 1e-2 | 1e-3 → 1e-4 cosine | 3e-4 (actor, critics, $\log\alpha$) |
| Batch | 10 episodes | 128 | 256 |
| Buffer | — (on-policy) | 1e5 | 1e6 |
| $\gamma$ | 0.99 | 0.99 | 0.99 |
| Target update | — | hard, every 1k steps | Polyak $\tau{=}0.005$ per step |
| Exploration | on-policy stochastic | $\epsilon$: 1.0→0.05 over 50k | max-ent, auto-$\alpha$, $\bar{\mathcal{H}}{=}{-}\dim(\mathcal{A})$ |
| Threshold | ≥ 475 / 100 eps (3 seeds) | ≥ 200 (2 seeds) | ≥ −200 (3 seeds); HalfCheetah ≥ 6000 @ 500k (1 seed, optional) |

## Exercise 1 — Vendor the ladder [Build]

Establishes the code base every later RL lesson patches: three single-file scripts you can read end to end.

1. Copy `dqn.py` and `sac_continuous_action.py` from CleanRL (github.com/vwxyzjn/cleanrl, `cleanrl/` directory) into this lesson directory as `dqn.py` and `sac.py`, keeping their MIT license header and adding one line: the commit hash you copied from. Install what their import lines name (`gymnasium`, `stable-baselines3` for the replay buffer, `tyro`, `tensorboard`), plus `pip install "gymnasium[box2d]"` (needs `swig`, see Pitfalls) and `"gymnasium[mujoco]"`.
2. Smoke-run each for 2k steps (`python sac.py --env-id Pendulum-v1 --total-timesteps 2000`; flag names per `--help`). Add W&B logging via the scripts' `--track` flag with `--wandb-entity` set (Lesson 00's journal note).
3. Diff each script's defaults against the hyperparameter table and set the table's values on the command line (or edit the dataclass); record every deviation in `RESULTS.md`. Expect at least one lr difference in SAC.

**✅ Checkpoint:** both scripts run clean to 2k steps and log to W&B; the deviation list exists.

## Exercise 2 — Annotate SAC [Read the kernel]

Tests objective 3: every line of the update maps to an equation. This is the lesson's core; type the file yourself if you prefer, the annotation is the requirement either way.

1. In `sac.py`, annotate the update block line by line with comments naming the equation: critic target (Eq. 15; mark the `no_grad` boundary and the `min` over targets), critic loss, actor loss (Eq. 16; mark where reparameterization happens — the sampled `u`, the `tanh`, the `log_prob -= log(1 - tanh²)` correction), temperature loss (Eq. 17; mark the target entropy), Polyak update ($\tau$).
2. Answer in the file header: which tensors carry gradients into which parameters, and why `a'` in the critic target comes from the *current* actor rather than the buffer.
3. Confirm the correction numerically in a scratch REPL: sample 1,000 `u`, compare the script's `log_prob` against `torch.distributions.TransformedDistribution(Normal, TanhTransform).log_prob` — max abs diff < 1e-4.

**✅ Checkpoint:** every loss line has an equation-number comment; the `TransformedDistribution` cross-check passes.

## Exercise 3 — REINFORCE variants [Build]

Produces the script for the variance ablation. Spec for `reinforce.py` (~80 lines in CleanRL style; an AI tool drafts it):

- `CartPole-v1`, 2×64 tanh policy, Adam lr 1e-2, batch = 10 episodes, $\gamma = 0.99$, `--seed`.
- `--estimator {return, rtg, rtg-baseline}`: full return; reward-to-go; reward-to-go minus a learned state-value baseline fit by MSE on returns — the baseline is **detached** from the policy loss.
- Logs per update: mean return over the batch, and the std of the policy-gradient norm across the 10 episodes' individual gradients (the variance proxy). Eval: 100 episodes, greedy.

The check you write: with `rtg`, print $\Psi_t$ for one hand-simulated 4-step episode with rewards `[1,1,1,1]` and confirm `[3.94, 2.97, 1.99, 1.0]` at $\gamma=0.99$.

**✅ Checkpoint:** the hand-computed $\Psi_t$ matches; each estimator runs one update without error.

## Exercise 4 — Variance ablation on CartPole [Predict → Run]

Tests objective 1: the two variance surgeries, measured.

1. **Write first**, in `RESULTS.md`: the ordering of the three estimators by (a) updates-to-475 and (b) the variance proxy, and the one-line reason for each gap.
2. Run 3 estimators × seeds {0, 1, 2}. Plot mean ± std learning curves and the variance proxy.
3. Reconcile.

**✅ Checkpoint:** `rtg-baseline` reaches avg return ≥ 475 (100 eval episodes) fastest and its variance proxy is lowest; `return` is noisiest. If `return` beats `rtg-baseline`, the baseline is leaking bias — check it is detached.

## Exercise 5 — The target network, removed [Predict → Run]

Tests objective 2: the deadly triad with one leg cut.

1. Add to `dqn.py`: `--no-target-net` (use $\phi^- = \phi$ every update) and a probe: a fixed batch of 256 states saved from an initial random rollout, on which you log $\mathbb{E}[\max_a Q(s,a)]$ every 1k steps alongside the empirically observed discounted return of recent episodes. Set the table's values: 2×256 ReLU, buffer 1e5, batch 128, lr 1e-3 → 1e-4 cosine, $\epsilon$ 1.0 → 0.05 over 50k, target sync every 1k steps.
2. **Write first:** sketch the probe curve for both arms — where the no-target arm departs from observed returns and whether it diverges or oscillates.
3. Run 2 arms × seeds {0, 1} on `LunarLander-v3`.

**✅ Checkpoint:** the target-net arm clears avg return ≥ 200; the no-target arm shows Q-value blow-up or oscillation on the probe plot. That gap *is* the missing stabilizer — the plot goes in `RESULTS.md`.

## Exercise 6 — The missing Jacobian term [Diagnose]

Tests objective 4: the classic silent SAC bug, experienced rather than read about.

1. Copy `sac.py` to `sac_broken.py` and delete the `log_prob -= ...` tanh-correction line.
2. **Write first:** what the entropy estimate does without the term, what the temperature update then does to $\alpha$, and the symptom on `Pendulum-v1` within 30k steps.
3. Run one seed of each on `Pendulum-v1`; overlay return, $\alpha$, and mean `log_prob`.
4. Explain the mechanism in `RESULTS.md`: the un-corrected density is the pre-squash Gaussian's, so entropy is misreported and the temperature controller chases the wrong target.

**✅ Checkpoint:** the broken run's $\alpha$ trace and `log_prob` trace differ visibly from the correct run's; the diagnosis names the density, not just the missing line.

## Exercise 7 — SAC, seeded [Predict → Run]

Produces the reference SAC curves Lessons 09–11 are read against.

1. **Write first:** the step at which auto-$\alpha$ entropy should approach $\bar{\mathcal{H}} = -\dim(\mathcal{A})$ on Pendulum, and the return at 30k steps.
2. `Pendulum-v1`, seeds {0, 1, 2}, 1 gradient step per env step, the table's values.
3. *Optional:* `HalfCheetah-v5`, one seed, 500k steps — overnight on `mps` or ~1 h on a 4090. Published SAC lands ~10k by 1M; ≥ 6000 at 500k means you're on the curve.

**✅ Checkpoint:** Pendulum avg return ≥ −200 within 30k steps on all seeds; entropy starts high and decays toward $-\dim(\mathcal{A})$. If it crashes to the target immediately, see Pitfalls.

## Exercise 8 — The ranking [Write]

Tests objective 5. In `RESULTS.md`: a table of env-steps-to-threshold per algorithm (with the caveat that the envs differ), then ≤ 10 sentences on *why* the ordering holds — data burn vs reuse, and what property of the SAC objective makes hundreds of reuses of one transition legal.

**✅ Checkpoint:** the mechanism paragraph names on-policy data burn and off-policy reuse explicitly.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `sac.py` (annotated), `dqn.py` (+ `--no-target-net` + probe), `reinforce.py`, `sac_broken.py` | CleanRL attribution + commit hash in headers; every SAC loss line cites its equation. **Contract:** `lessons/08-rl-ladder/sac.py` is the file Lesson 09 patches and Lesson 11 reuses — keep its CLI surface |
| `plots/` | variance ablation (3 seeds), target-net probe plot (2 seeds), Pendulum SAC curves (3 seeds), broken-vs-correct SAC overlay |
| `RESULTS.md` | default-vs-table deviation list; Exercises 4–7 predictions with reconciliations; the ranking + mechanism |

## Done when

- [ ] `rtg-baseline` REINFORCE ≥ 475 on 3/3 seeds; Pendulum SAC ≥ −200 on 3/3 seeds; DQN ≥ 200 on 2/2 seeds.
- [ ] The target-net probe plot shows the pathology; the broken-SAC overlay shows the symptom you predicted (or the reconciliation explains the gap).
- [ ] `sac.py` is annotated to the equation on every loss line and the `TransformedDistribution` check passes.
- [ ] The ranking paragraph exists and is mechanistic.

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
| CartPole flatlines at ~20 | missing reward-to-go sign/discount bug | print $\Psi_t$ for one episode by hand (Exercise 3's check) |
| DQN diverges even with target net | lr too high for LunarLander's reward scale | drop to 5e-4; clip grads at 10 |
| SAC entropy collapses to $\bar{\mathcal{H}}$ instantly, no exploration | temperature lr too high or $\log\alpha$ not clamped | lr 3e-4 on $\log \alpha$; init $\alpha = 0.2$ |
| SAC actor loss NaN | tanh log-prob correction at $\lvert u\rvert \gg 1$ | numerically stable form: $2(\log 2 - u - \mathrm{softplus}(-2u))$ |
| HalfCheetah stuck < 2000 | 1-step-per-env-step not actually happening, or obs not normalized | log the update/step ratio; running mean-std normalize observations |
| `mps` slower than CPU on small MLPs | kernel launch overhead dominates | batch ≥ 256 or just use `cpu` for CartPole/LunarLander |
| Vendored script's flag names differ from this README | CleanRL revision drift | `python <script>.py --help` is authoritative; pin the commit hash in the header |

## Going deeper

- **Double DQN.** Add `--double` (argmax from $\phi$, value from $\phi^-$) and show its probe curve tracks observed returns more closely than plain DQN's — the overestimation gap made visible.
- **From scratch.** Rewrite SAC in < 200 lines with no CleanRL reference open, then diff behavior against the vendored file on Pendulum at fixed seeds.
- **Pixels / n-step.** Pixel-input DQN (frame-stack 4, CNN torso) on `ALE/Pong-v5`; or n-step returns in SAC and the sample-efficiency delta (foreshadows Lesson 09's UTD discussion).

## References

- Sutton & Barto ch. 13 (policy gradient theorem); Schulman et al. 2016 (GAE) for the $\Psi_t$ taxonomy.
- Mnih et al. 2015 (DQN); van Hasselt et al. 2016 (double DQN).
- Haarnoja et al. 2018, arXiv:1801.01290 + arXiv:1812.05905 (SAC + auto-temperature).
- Huang et al., *CleanRL: High-quality Single-file Implementations of Deep RL Algorithms*, JMLR 2022. github.com/vwxyzjn/cleanrl (MIT).
- LeRobot team, *Robot Learning: A Tutorial*, §3.1–3.2, Eqs. 11–17. arXiv:2510.12403.
