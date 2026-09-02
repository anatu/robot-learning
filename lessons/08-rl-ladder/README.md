# Lesson 08 — The RL Ladder: REINFORCE → DQN → SAC

This lesson covers the three reinforcement-learning algorithms that the rest of Phase 3 builds on: REINFORCE, DQN, and Soft Actor-Critic (SAC). Rather than implementing each from scratch, you will work from CleanRL's single-file reference implementations, annotate them line by line against the tutorial's Eqs. 11–17, predict what happens when a specific component is removed, and then remove it and watch. The lesson ends with SAC because it is the algorithm that LeRobot's HIL-SERL runs, and because Lessons 09, 10 and 11 all modify or reuse the SAC file you produce here.

| | |
|---|---|
| **Phase** | 3 — Reinforcement learning |
| **Time** | ~6–8 h desk time (AI-assisted); compute: minutes for CartPole and Pendulum, ~20 min per LunarLander run on the Mac; HalfCheetah (optional) overnight on `mps` or ~1 h on a rented 4090 |
| **Cost** | $0 (≤ $3 if you run the optional HalfCheetah arm in the cloud) |
| **Prerequisites** | 05 (you know what an optimal controller looks like when the model is known; RL solves the same problem when the model can only be sampled) |
| **Feeds into** | 09 (patches this lesson's `sac.py` for RLPD), 10 (HIL-SERL is this SAC plus buffers and a human), 11 (this `sac.py` trains every domain-randomization arm) |

## Learning objectives

After this lesson you can:

1. **Derive** the policy gradient theorem and explain why reward-to-go and a baseline reduce variance without adding bias.
2. **Predict and demonstrate**, with a plot rather than a citation, why the target network is load-bearing in DQN.
3. **Map** every line of SAC's three updates (twin critics, reparameterized actor, temperature) to tutorial Eqs. 14–17 and to the maximum-entropy objective they come from.
4. **Diagnose** a silent SAC bug, the missing tanh log-probability correction, from its training symptom alone.
5. **Rank** the three algorithms on sample efficiency from seeded curves and explain the mechanism behind the ordering.

## Principles

### One objective, three estimators

All three algorithms maximize the same quantity, the expected discounted return $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_t \gamma^t r_t]$. They differ in what they estimate in order to improve it, and in how many times they can reuse each sampled transition. REINFORCE estimates the gradient of $J$ directly from whole trajectories and discards them after one update. DQN estimates the optimal action-value function and acts greedily with respect to it, reusing transitions from a replay buffer. SAC estimates both a value function and a stochastic policy, and reuses each transition hundreds of times. That difference in data reuse is what Exercise 8 asks you to explain, so it is worth keeping in mind as you read the three methods.

### REINFORCE and the two variance reductions

The policy gradient theorem states that $\nabla_\theta J = \mathbb{E}_{\tau}\big[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\, \Psi_t\big]$, where $\Psi_t$ is some measure of how good the trajectory turned out to be. With $\Psi_t$ equal to the full return of the trajectory, the estimator is unbiased but has variance that grows with the horizon, because every action is credited with rewards it could not have influenced. Two modifications reduce the variance without introducing bias. The first is reward-to-go, $\Psi_t = \sum_{t' \ge t} \gamma^{t'-t} r_{t'}$: since an action cannot cause rewards that came before it, dropping those terms removes noise without changing the expectation. The second is a baseline, $\Psi_t - b(s_t)$: subtracting any function of the state alone leaves the gradient unbiased because $\mathbb{E}[\nabla \log \pi \cdot b(s)] = 0$, which follows from $\int \nabla \pi = \nabla \int \pi = 0$. This is the material of CS 285 Lecture 5, and the tutorial's Eq. 11 is its descendant. Exercise 4 measures both reductions.

### DQN and the deadly triad

DQN takes the value-based route: it fits the optimal action-value function $Q^*$ by regressing $Q_\phi(s,a)$ onto the Bellman target $r + \gamma \max_{a'} Q_{\phi^-}(s',a')$ (tutorial Eq. 12), and acts greedily. In the tabular setting the Bellman operator is a $\gamma$-contraction and this iteration converges. Once you combine function approximation, bootstrapping (using your own estimate as the regression target), and off-policy data, the combination known as the deadly triad, the convergence guarantee disappears. DQN stabilizes the iteration with two devices. A replay buffer decorrelates the batches, which stochastic gradient descent implicitly assumes are independent. A target network $\phi^-$, updated only periodically, freezes the regression target so that each update is not chasing a target that the same update just moved. Double DQN goes one step further and decouples the action selection from the action evaluation in the target, which reduces the overestimation that the $\max$ introduces. Exercise 5 removes the target network and shows what it was preventing.

### SAC and maximum-entropy control

Continuous action spaces make the $\max_{a'}$ in the DQN target impossible to compute, and maximum-entropy reinforcement learning is the standard replacement. The objective becomes $J = \sum_t \mathbb{E}[r_t + \alpha \mathcal{H}(\pi(\cdot|s_t))]$ (tutorial Eq. 14), which rewards return and policy entropy together, with a temperature $\alpha$ setting the trade. SAC optimizes this objective with three coupled updates.

The critic update (Eq. 15) trains twin action-value networks $Q_{\phi_1}, Q_{\phi_2}$ to regress onto $r + \gamma\big(\min_i Q_{\phi_i^-}(s', a') - \alpha \log \pi(a'|s')\big)$ with $a' \sim \pi$. Taking the minimum over the two target critics counters the same overestimation that double DQN counters.

The actor update (Eq. 16) minimizes $\mathbb{E}_{s}\big[\alpha \log \pi_\theta(a|s) - \min_i Q_{\phi_i}(s,a)\big]$ using the reparameterization trick, $a = \tanh(\mu_\theta(s) + \sigma_\theta(s) \epsilon)$, so that the gradient can flow from $Q$ back into the policy parameters. The $\tanh$ squash keeps actions within bounds but changes the density: the log-probability of a squashed action is $\log \pi(a|s) = \log \mathcal{N}(u) - \sum_i \log(1 - \tanh^2 u_i)$. The second term is a Jacobian correction, and it is the term that most from-scratch implementations omit at least once. Exercise 6 removes it deliberately.

The temperature update (Eq. 17) adjusts $\alpha$ by gradient descent on $\mathbb{E}[-\alpha(\log \pi(a|s) + \bar{\mathcal{H}})]$, driving the policy entropy toward a target $\bar{\mathcal{H}} = -\dim(\mathcal{A})$.

The critic target is the place where gradient flow is most often wrong, so here it is written out. Nothing inside `target` receives a gradient:

```python
with torch.no_grad():
    a2, logp2 = actor.sample(s2)                      # fresh action from the CURRENT actor
    q_targ = torch.min(q1_targ(s2, a2), q2_targ(s2, a2))
    target = r + gamma * (1 - done) * (q_targ - alpha * logp2)
loss_q = F.mse_loss(q1(s, a), target) + F.mse_loss(q2(s, a), target)
```

SAC is the workhorse of real-robot reinforcement learning because it is off-policy: every transition in the replay buffer can be reused for hundreds of gradient steps, which matters when each transition costs seconds of robot time. Lessons 09 and 10 exist because of that property.

### Why you read a reference implementation instead of writing one

The bugs that matter in reinforcement learning are semantic rather than structural: a baseline that is not detached from the policy loss, a target network that is accidentally updated every step, a missing Jacobian term. None of these produce an error message; they produce a curve that is somewhat worse than it should be. CleanRL's single-file scripts are readable, widely checked implementations of exactly these algorithms. Annotating one against the equations, and then breaking it on purpose, exercises the semantic layer directly, whereas typing the file from scratch mostly exercises the structural layer. A from-scratch rewrite is available under Going deeper for anyone who wants it.

**Carry forward**

- In the policy gradient, reward-to-go and a state-only baseline reduce variance without adding bias; a baseline that depends on the action does add bias, because the cancellation argument requires $b$ to be constant with respect to $a_t$.
- The deadly triad is function approximation plus bootstrapping plus off-policy data. The target network breaks the feedback loop in the bootstrapping leg by holding the regression target fixed between updates.
- SAC consists of twin critics with a minimum in the target, a reparameterized tanh-Gaussian actor with the $\log(1-\tanh^2)$ correction, and an automatically tuned temperature that drives entropy toward $-\dim(\mathcal{A})$.
- Sample efficiency follows data reuse: an on-policy method discards each trajectory after one update, whereas an off-policy method with a replay buffer reuses each transition many times.

| Source | Read for |
|---|---|
| Tutorial §3.1–3.2 (Eqs. 11–17) | the equation numbering your annotations cite |
| CS 285 (Fa23) Lectures 4–6 | the MDP formalism, the variance analysis of policy gradients, and the bridge to actor-critic methods |
| CS 285 (Fa23) Lectures 7–8 | value-based methods and why deep Q-learning is unstable in practice |
| Haarnoja et al. 2018 (arXiv:1801.01290) and the applications paper (1812.05905) | the three SAC updates; the automatic temperature is introduced in the second paper |
| CleanRL docs (docs.cleanrl.dev): the `dqn.py` and `sac_continuous_action.py` pages | the documented defaults and benchmark curves of the files you vendor |

The exercises assume the hyperparameters below. Where the vendored files' defaults differ, record the deviation in `RESULTS.md`.

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

## Exercise 1 — Vendor the reference implementations [Build]

In this exercise you set up the code base that the rest of Phase 3 will patch and reuse: three single-file scripts that you can read from top to bottom. Copying them from CleanRL, rather than installing the package, means every later modification is a visible diff in your own repository.

1. Copy `dqn.py` and `sac_continuous_action.py` from CleanRL (github.com/vwxyzjn/cleanrl, in the `cleanrl/` directory) into this lesson directory as `dqn.py` and `sac.py`. Keep their MIT license headers and add one line recording the commit hash you copied from. Install what their import lines require (`gymnasium`, `stable-baselines3` for the replay buffer, `tyro`, `tensorboard`), plus `pip install "gymnasium[box2d]"` (which needs `swig`; see Pitfalls) and `"gymnasium[mujoco]"`.
2. Run each script for 2,000 steps as a smoke test, for example `python sac.py --env-id Pendulum-v1 --total-timesteps 2000` (flag names are whatever `--help` reports). Enable W&B logging with the scripts' `--track` flag and set `--wandb-entity` explicitly, for the reason recorded in Lesson 00's journal entry.
3. Compare each script's defaults with the hyperparameter table above and set the table's values on the command line or in the script's dataclass. Record every difference in `RESULTS.md`. Expect at least one learning-rate difference in SAC.

**✅ Checkpoint:** both scripts run to 2,000 steps and log to W&B, and the list of default-versus-table deviations exists.

## Exercise 2 — Annotate the SAC update [Read the kernel]

This exercise is the core of the lesson. You will go through the update block of `sac.py` and attach to every line the equation it implements, so that the correspondence between the maximum-entropy objective and the code is explicit rather than assumed. You may type the file yourself if you prefer; the annotation is the requirement either way.

1. In `sac.py`, annotate the update block line by line with comments naming the equation each line implements: the critic target (Eq. 15; mark the `no_grad` boundary and the minimum over the two target critics), the critic loss, the actor loss (Eq. 16; mark where reparameterization happens: the sampled `u`, the `tanh`, and the `log_prob -= log(1 - tanh²)` correction), the temperature loss (Eq. 17; mark the target entropy), and the Polyak update with its $\tau$.
2. In the file header, answer two questions in prose: which tensors carry gradients into which parameters, and why the action $a'$ in the critic target is drawn from the current actor rather than read from the buffer.
3. Confirm the Jacobian correction numerically in a scratch session: sample 1,000 values of `u`, and compare the script's `log_prob` against `torch.distributions.TransformedDistribution(Normal, TanhTransform).log_prob`. The maximum absolute difference should be below 1e-4.

**✅ Checkpoint:** every loss line carries an equation-number comment, and the `TransformedDistribution` cross-check passes.

## Exercise 3 — Write the REINFORCE variants [Build]

Here you produce the script for the variance ablation in Exercise 4. It is a small script in the CleanRL style, about eighty lines, and an AI tool can draft it from the specification below; your job is to specify the three estimators precisely and to write the check that catches the most common mistake.

The specification for `reinforce.py`:

- `CartPole-v1`, a 2×64 tanh policy, Adam with learning rate 1e-2, a batch of 10 episodes per update, $\gamma = 0.99$, and a `--seed` flag.
- `--estimator {return, rtg, rtg-baseline}`: the full return; reward-to-go; and reward-to-go minus a learned state-value baseline fitted by mean-squared error on the observed returns. The baseline must be **detached** from the policy loss, so that no gradient from the policy objective flows into it.
- Per update, log the mean return over the batch and the standard deviation of the policy-gradient norm across the ten episodes' individual gradients, which serves as the variance proxy. Evaluate with 100 greedy episodes.

The check you write yourself: with the `rtg` estimator, print $\Psi_t$ for a hand-simulated four-step episode with rewards `[1,1,1,1]` and confirm it equals `[3.94, 2.97, 1.99, 1.0]` at $\gamma=0.99$.

**✅ Checkpoint:** the hand-computed $\Psi_t$ matches, and each estimator completes one update without error.

## Exercise 4 — Measure the variance reductions on CartPole [Predict → Run]

This exercise tests objective 1 by measuring the two variance reductions rather than taking them from the derivation. Writing the expected ordering down first makes the result a test of your understanding rather than a demonstration.

1. Before running, write in `RESULTS.md` the ordering you expect among the three estimators by (a) the number of updates needed to reach an average return of 475 and (b) the variance proxy, with a one-sentence reason for each gap.
2. Run the three estimators with seeds {0, 1, 2}. Plot the learning curves as mean ± standard deviation, and plot the variance proxy.
3. Reconcile the result with your prediction.

**✅ Checkpoint:** `rtg-baseline` reaches an average return of at least 475 over 100 evaluation episodes fastest and has the lowest variance proxy, while `return` is the noisiest. If `return` beats `rtg-baseline`, the baseline is leaking bias into the policy gradient; check that it is detached.

## Exercise 5 — Remove the target network [Predict → Run]

This exercise tests objective 2 by cutting one leg of the deadly triad and observing the result. The instrument is a probe: a fixed batch of states on which you track the network's own value estimate against the returns actually observed, so that overestimation and divergence are visible directly rather than inferred from the learning curve.

1. Add two things to `dqn.py`: a `--no-target-net` flag that sets $\phi^- = \phi$ on every update, and a probe consisting of a fixed batch of 256 states saved from an initial random rollout, on which you log $\mathbb{E}[\max_a Q(s,a)]$ every 1,000 steps alongside the empirically observed discounted return of recent episodes. Set the table's values: 2×256 ReLU, buffer 1e5, batch 128, learning rate 1e-3 decaying to 1e-4 on a cosine schedule, $\epsilon$ from 1.0 to 0.05 over 50k steps, and a target sync every 1,000 steps.
2. Before running, sketch the probe curve for both arms. Where does the no-target arm depart from the observed returns, and do you expect it to diverge or to oscillate?
3. Run both arms with seeds {0, 1} on `LunarLander-v3`.

**✅ Checkpoint:** the arm with the target network reaches an average return of at least 200, and the arm without it shows Q-values that blow up or oscillate on the probe plot. The gap between the two probe curves is the stabilization the target network provides; the plot belongs in `RESULTS.md`.

## Exercise 6 — Remove the tanh log-probability correction [Diagnose]

This exercise tests objective 4. The Jacobian term in SAC's log-probability is a bug that many implementations have shipped, and it is silent: training proceeds and the curves look plausible. Removing it deliberately, after predicting the symptom, is the most reliable way to learn to recognize it.

1. Copy `sac.py` to `sac_broken.py` and delete the `log_prob -= ...` line that applies the tanh correction.
2. Before running, write down what the entropy estimate does without the term, what the temperature update then does to $\alpha$, and what symptom you expect on `Pendulum-v1` within 30k steps.
3. Run one seed of each script on `Pendulum-v1` and overlay the return, $\alpha$, and mean `log_prob` traces.
4. Explain the mechanism in `RESULTS.md`. Without the correction, the reported density is that of the pre-squash Gaussian rather than the squashed action, so the entropy is misreported and the temperature controller drives $\alpha$ toward the wrong target.

**✅ Checkpoint:** the broken run's $\alpha$ and `log_prob` traces differ visibly from the correct run's, and the diagnosis names the density that is wrong rather than only the missing line.

## Exercise 7 — Train SAC with fixed seeds [Predict → Run]

This exercise produces the reference SAC curves that Lessons 09, 10 and 11 are read against. Pendulum is the required environment because it trains in minutes; HalfCheetah is optional and included for anyone who wants to see the algorithm at a scale closer to the published benchmarks.

1. Before running, write down the step at which you expect the automatically tuned entropy to approach $\bar{\mathcal{H}} = -\dim(\mathcal{A})$ on Pendulum, and the return you expect at 30k steps.
2. Run `Pendulum-v1` with seeds {0, 1, 2}, one gradient step per environment step, and the table's values.
3. Optionally, run `HalfCheetah-v5` for one seed and 500k steps, overnight on `mps` or in about an hour on a 4090. Published SAC reaches roughly 10,000 by one million steps, so a return of 6,000 or more at 500k steps indicates you are on the expected curve.

**✅ Checkpoint:** Pendulum reaches an average return of at least −200 within 30k steps on every seed, and the entropy starts high and decays toward $-\dim(\mathcal{A})$. If the entropy drops to the target almost immediately, see Pitfalls.

## Exercise 8 — Rank the algorithms by sample efficiency [Write]

This exercise tests objective 5. In `RESULTS.md`, tabulate the environment steps each algorithm needed to reach its threshold, noting that the environments differ and so the comparison is qualitative, and then write at most ten sentences explaining why the ordering holds. The explanation should name the mechanism directly: an on-policy method burns its data after one update, an off-policy method reuses it, and there is a specific property of the SAC objective that makes hundreds of reuses of one transition legitimate.

**✅ Checkpoint:** the mechanism paragraph names on-policy data burn and off-policy reuse explicitly and states the property that licenses reuse.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `sac.py` (annotated), `dqn.py` (with `--no-target-net` and the probe), `reinforce.py`, `sac_broken.py` | CleanRL attribution and commit hash in the headers; every SAC loss line cites its equation. **Contract:** `lessons/08-rl-ladder/sac.py` is the file that Lesson 09 patches and Lesson 11 reuses, so its command-line surface must be kept stable |
| `plots/` | variance ablation (3 seeds), target-net probe plot (2 seeds), Pendulum SAC curves (3 seeds), broken-versus-correct SAC overlay |
| `RESULTS.md` | the default-versus-table deviation list; the predictions for Exercises 4–7 with their reconciliations; the ranking and its mechanism paragraph |

## Done when

- [ ] `rtg-baseline` REINFORCE reaches 475 on 3/3 seeds; Pendulum SAC reaches −200 on 3/3 seeds; DQN reaches 200 on 2/2 seeds.
- [ ] The target-net probe plot shows the pathology, and the broken-SAC overlay shows the symptom you predicted, or the reconciliation explains the difference.
- [ ] `sac.py` is annotated to the equation on every loss line, and the `TransformedDistribution` check passes.
- [ ] The ranking paragraph exists and explains the ordering by mechanism.

## Self-check

1. Prove in three lines that subtracting $b(s_t)$ leaves the policy gradient unbiased. Where does the argument break if $b$ depends on $a_t$?
2. Name the three legs of the deadly triad and say which one the target network amputates.
3. Why does SAC take a minimum over two critics rather than an average?
4. Where exactly does the $\log(1 - \tanh^2(u))$ term come from, and what silently goes wrong without it?
5. REINFORCE discards data after one update; SAC reuses it for hundreds of updates. What property of the SAC objective makes that legitimate?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `box2d` install fails | missing `swig` | `brew install swig`, then reinstall `gymnasium[box2d]` |
| CartPole flatlines at ~20 | a sign or discount bug in reward-to-go | print $\Psi_t$ for one episode by hand (the Exercise 3 check) |
| DQN diverges even with the target net | learning rate too high for LunarLander's reward scale | drop to 5e-4; clip gradients at 10 |
| SAC entropy collapses to $\bar{\mathcal{H}}$ immediately, no exploration | temperature learning rate too high, or $\log\alpha$ not clamped | learning rate 3e-4 on $\log \alpha$; initialize $\alpha = 0.2$ |
| SAC actor loss becomes NaN | the tanh correction evaluated at $\lvert u\rvert \gg 1$ | use the numerically stable form $2(\log 2 - u - \mathrm{softplus}(-2u))$ |
| HalfCheetah stuck below 2000 | not actually one gradient step per environment step, or observations not normalized | log the update-to-step ratio; normalize observations with a running mean and standard deviation |
| `mps` slower than CPU on small MLPs | kernel-launch overhead dominates | use batch sizes of 256 or more, or simply use `cpu` for CartPole and LunarLander |
| Vendored script's flag names differ from this README | CleanRL revision drift | `python <script>.py --help` is authoritative; pin the commit hash in the header |

## Going deeper

- **Double DQN.** Add a `--double` flag (argmax from $\phi$, value from $\phi^-$) and show that its probe curve tracks the observed returns more closely than plain DQN's does. This makes the overestimation gap visible directly.
- **From scratch.** Rewrite SAC in under 200 lines without the CleanRL file open, then compare its behaviour against the vendored file on Pendulum at fixed seeds.
- **Pixels and n-step returns.** Train a pixel-input DQN (four stacked frames, a convolutional torso) on `ALE/Pong-v5`; or add n-step returns to SAC and measure the change in sample efficiency, which foreshadows the update-to-data discussion in Lesson 09.

## References

- Sutton & Barto ch. 13 (the policy gradient theorem); Schulman et al. 2016 (GAE) for the $\Psi_t$ taxonomy.
- Mnih et al. 2015 (DQN); van Hasselt et al. 2016 (double DQN).
- Haarnoja et al. 2018, arXiv:1801.01290 and arXiv:1812.05905 (SAC and automatic temperature).
- Huang et al., *CleanRL: High-quality Single-file Implementations of Deep RL Algorithms*, JMLR 2022. github.com/vwxyzjn/cleanrl (MIT).
- LeRobot team, *Robot Learning: A Tutorial*, §3.1–3.2, Eqs. 11–17. arXiv:2510.12403.
