# Lesson 12 — Why Generative Policies: The Multimodality Lab

This lesson builds a small two-dimensional navigation task in which you control the data-generating process completely, and uses it to demonstrate the two distinct failure modes that motivate every architecture in Phase 4. The first, mode-averaging, is a failure of the model class: a regressor trained with mean-squared error can only return the average of the expert's choices, even when that average is an action no expert ever took. The second, compounding covariate shift, is a failure of the training-versus-deployment distribution, and it afflicts any behaviour-cloned policy regardless of which head it uses. You will train four policy heads on the same data, predict how each will behave before measuring it, and separate the two failures experimentally, so that the design choices of ACT, Diffusion Policy and π0 in the following lessons have concrete referents.

| | |
|---|---|
| **Phase** | 4 — Generative imitation policies |
| **Time** | 1 session (4–5 h desk time, AI-assisted; training runs take minutes on `mps` or CPU) |
| **Cost** | $0 |
| **Prerequisites** | 08 (you can read a PyTorch training loop); Lesson 13 pairs naturally as a co-read |
| **Feeds into** | 13 (the mathematics behind the two heads that work), 14 and 15 (ACT and Diffusion Policy are these heads at scale; Lesson 15 reuses the DDPM and CFM heads directly), 17 (π0's flow head is the CFM head conditioned on a vision-language model) |

## Learning objectives

After this lesson you can:

1. **State** the behaviour-cloning objective precisely and explain why an MSE regressor is maximum likelihood under a unimodal Gaussian, and what that assumption does to multimodal data.
2. **Explain** each of the four heads (MSE, CVAE, DDPM, conditional flow matching) as a single loss equation, and annotate its code against that equation.
3. **Predict and quantify** how each head handles multimodality, using a mode-balance metric and an indecision metric that you define as formulas.
4. **Demonstrate** closed-loop compounding error and distinguish it experimentally from mode-averaging.
5. **Explain** which of the two failures action chunking mitigates, and why it cannot fix the other.

## Principles

### Behaviour cloning and the conditional mean

Behaviour cloning treats policy learning as supervised learning. Given a dataset $\mathcal{D}$ of (observation, action) pairs collected from an expert, it fits a conditional model of actions by maximum likelihood, $\min_\theta \, \mathbb{E}_{(o,a)\sim\mathcal{D}}\!\left[-\log \pi_\theta(a\mid o)\right]$. The choice of model class determines what maximum likelihood can produce. If the policy is a deterministic network $f_\theta$ trained with mean-squared error, then implicitly $\pi_\theta(a|o) = \mathcal{N}(f_\theta(o), \sigma^2 I)$, a unimodal Gaussian, and the maximum-likelihood solution is the conditional mean, $f_\theta(o) \to \mathbb{E}[a\mid o]$.

The conditional mean is the right answer when the expert's action distribution at each state is unimodal. When it is not, the mean is a poor summary of the data. If the expert steers left around an obstacle half the time and right the other half, the conditional mean at the decision point is straight ahead, which is an action no expert ever took and which may lie off the feasible set entirely. This failure is called mode-averaging. It is a property of the model class rather than of the data or the optimizer, so it persists with infinite data and perfect optimization.

### Compounding error is a different failure

The second failure has nothing to do with multimodality. A behaviour-cloned policy is trained on the distribution of states the expert visited, but at deployment it visits the states produced by its own actions. Each small error moves the state slightly away from the expert's distribution, where the policy has seen less data and is therefore less accurate, which moves the state further away still. Ross, Gordon and Bagnell (2011) formalize this argument: with a per-step error rate of $\epsilon$, the cost of the learned policy over a horizon of $T$ steps can exceed the expert's by $O(\epsilon T^2)$, whereas the same error rate in an ordinary supervised problem costs only $O(\epsilon T)$. The quadratic dependence on the horizon is what distinguishes compounding error from ordinary supervised error.

Because this argument makes no assumption about the shape of the expert's action distribution, the failure exists even for a unimodal expert, and no generative head fixes it on its own. The two failures are therefore orthogonal, and the purpose of this lab is to observe each one in isolation.

### The generative fixes, expressed as four losses

The remedy for mode-averaging is a model class that can represent a multimodal conditional distribution. Three such heads are studied here, alongside the MSE regressor as the control, and each is defined by a single loss. Exercise 3 asks you to annotate the code of each head against the corresponding row of this table.

| Head | Loss | What it models |
|---|---|---|
| MSE | $\|a - f_\theta(s)\|^2$ | the conditional mean (Gaussian maximum likelihood) |
| CVAE | $\|a - p_\theta(s, z)\|^2 + \beta\, D_{KL}\!\left(q_\phi(z|s,a)\,\|\,\mathcal{N}(0,I)\right)$, $z \sim q_\phi$ via reparameterization | $p(a|s)$ through a latent $z$ (Lesson 13 derives the ELBO) |
| DDPM | $\mathbb{E}_{t,\epsilon}\|\epsilon - \epsilon_\theta(x_t, t, s)\|^2$, $x_t = \sqrt{\bar\alpha_t}\,a + \sqrt{1-\bar\alpha_t}\,\epsilon$ | the reverse of a Gaussian corruption process (Lesson 13's simplified loss) |
| CFM | $\mathbb{E}_{t,x_0}\|v_\theta(x_t, t, s) - (x_1 - x_0)\|^2$, $x_t = (1-t)x_0 + t x_1$, $x_0 \sim \mathcal{N}(0,I)$, $x_1 = a$ | a velocity field along the optimal-transport path |

A conditional variational autoencoder (CVAE) routes the action through a latent variable $z$, so that different values of $z$ can decode to different modes; the KL term keeps the encoder's posterior close to a standard normal so that sampling $z$ from the prior at inference time produces valid actions. A denoising diffusion probabilistic model (DDPM) learns to reverse a Gaussian corruption process, and its simplified loss is a regression on the noise that was added at a random corruption level. Conditional flow matching (CFM) regresses a velocity field along the straight path between a noise sample and a data point, and generates a sample by integrating that field. All three can place probability mass on both of the expert's modes rather than on their average, which is exactly what the MSE head cannot do. None of them, by itself, changes the distribution of states the policy will visit at deployment, so none of them addresses covariate shift.

### Chunking addresses the horizon, not the mean

A different lever addresses compounding error. If the policy predicts $H$ actions per decision instead of one, and those actions are executed before the next decision is made, then the number of decision points in an episode of length $T$ falls from $T$ to $T/H$, and drift accumulates over correspondingly fewer steps. This is action chunking, and it reduces the shift failure for any head. It does nothing for the averaging failure, because a chunked MSE head still predicts the average chunk; for the two-arc expert of this lesson, the average chunk is the average path, which runs straight into the obstacle. Lessons 14 and 15 combine a generative head with chunking for precisely this reason.

**Carry forward**

- Mean-squared-error regression is maximum likelihood under a unimodal Gaussian, so on multimodal data it returns the conditional mean, which may be an action no expert ever took.
- Mode-averaging is a failure of the model class and compounding error is a failure of distribution shift; because their causes differ, their remedies differ too.
- Sampling from a generative head fixes averaging and leaves shift untouched, whereas chunking mitigates shift and leaves averaging untouched, so a practical policy needs both.
- The four losses in the table are the vocabulary of Phase 4: ACT is a CVAE with chunking, Diffusion Policy is a DDPM with chunking, and π0 is conditional flow matching with chunking on top of a vision-language model.

| Source | Read for |
|---|---|
| Tutorial §4.0–4.1 | the formal behaviour-cloning setup and how the tutorial motivates generative heads; which failure its figure attributes to which cause |
| Florence et al. 2022 (Implicit BC), §1 + Fig. 2 | the canonical picture of mode-averaging; why implicit and energy-based models were the first proposed fix |
| Ross, Gordon, Bagnell 2011 (DAgger), §2 | where the $O(\epsilon T^2)$ bound comes from; the problem setup and the statement of Theorem 2.1 suffice |

## Exercise 1 — Build the toy world [Build]

This exercise produces the environment and the expert that every later measurement depends on. The specification is exact so that the numbers you obtain are comparable across heads and across runs, and none of the constants should be changed. Have an AI tool draft `env.py` and `expert.py` from the specification below, and verify the checkpoint yourself. The property you are building in is a region of state space where the dataset contains two opposite clusters of actions, which is what a unimodal head cannot represent.

1. **Environment.** The state is $s = (x, y) \in [-1,1]^2$. The start is $s_0 = (0, -0.8)$ plus $\mathcal{N}(0, 0.02^2)$ jitter. The goal is to reach within 0.1 of $(0, 0.8)$. There is one circular obstacle with centre $(0,0)$ and radius $0.25$. The action $a \in \mathbb{R}^2$ is a displacement, clipped to $\|a\| \le 0.05$, and the dynamics are $s' = s + a$. An episode fails on obstacle penetration or after 300 steps. Implement this as a plain Python class with `reset(seed)` and `step(a)`; no gym dependency is needed.
2. **Expert.** At the start of each episode, flip a fair coin to choose a left or right arc, defined by three waypoints $(\pm 0.45, -0.5) \to (\pm 0.45, 0.5) \to (0, 0.8)$ with the sign set by the coin. The expert steers toward the current waypoint at full step size with $\mathcal{N}(0, 0.005^2)$ action noise, and advances to the next waypoint once within 0.1 of the current one. Record 500 episodes, which yields roughly 25–30k $(s, a)$ pairs.
3. Plot all expert trajectories coloured by mode, and a quiver plot of expert actions over the state space. The feature to look for is that at states near $x = 0$ and below the obstacle, the dataset contains two opposite clusters of actions.

**✅ Checkpoint:** the trajectory plot shows two clean, symmetric arcs; the expert's success rate is 100%; and at the probe state $s^\* = (0, -0.4)$, the expert actions harvested from within a radius of 0.05 of $s^\*$ form two clusters with almost no mass between them.

## Exercise 2 — Implement the four heads on one training harness [Build]

Here you produce the four policy heads as runnable code with a common interface, which is the first half of objective 2. The specification fixes the architecture, the optimizer and the sampling procedure for each head, so that the only difference between them is the loss they minimize. Give the specification to an AI tool and read the resulting code before running it.

- Provide one `train(head, dataset, seed)` function and four `nn.Module` classes in `heads/{mse,cvae,ddpm,cfm}.py`. Each head is under 150 lines including its sampling code, and each exposes `sample(s, n) -> (n, 2)`, returning `n` actions at state `s`. All four share a trunk of three 128-unit ReLU layers, and all train with Adam at learning rate 1e-3, batch size 256, for 20k steps, at seed 0.
- **MSE**: the trunk maps directly to a 2-D action.
- **CVAE**: an encoder $q_\phi(z|s,a)$ and a decoder $p_\theta(a|s,z)$ with $z \in \mathbb{R}^2$; the loss is the CVAE row of the Principles table with $\beta = 1$; sampling draws $z \sim \mathcal{N}(0,I)$ and decodes.
- **DDPM**: 100 timesteps with a cosine schedule (in the style of `squaredcos_cap_v2`), $\epsilon$-prediction, and a sinusoidal 64-dimensional timestep embedding concatenated to $(s, a_t)$; sampling is ancestral over all 100 steps.
- **CFM**: regress $v_\theta(x_t, t, s)$ onto $x_1 - x_0$ along $x_t = (1-t)x_0 + t x_1$ with $t \sim \mathcal{U}[0,1]$; sampling is Euler integration with 10 steps.
- Build all schedules as float32 tensors (see Pitfalls).

**✅ Checkpoint:** all four training losses converge. The MSE head's loss plateaus at a high value; before reading further, write down why this must happen. Sampling 1000 actions at $s^\*$ takes under 5 s per head on `mps` or CPU.

## Exercise 3 — Annotate the four losses [Read the kernel]

The second half of objective 2 is to connect each head's code to the equation it implements. In each head's file, annotate the loss computation line by line with the corresponding row of the Principles table: identify the reconstruction term, the point at which the reparameterization trick is applied in the CVAE, the line that forms $x_t$ in the DDPM and which schedule quantity it draws on, and the line that computes the target velocity in the CFM head. Then, for the MSE head, write the two-line argument that its minimizer is $\mathbb{E}[a|s]$.

**✅ Checkpoint:** every loss line carries a comment naming its term in the equation, and the MSE argument appears in `RESULTS.md`.

## Exercise 4 — Measure multimodality at the probe state [Predict → Run]

This exercise turns the qualitative claim that some heads "handle multimodality" into two numbers, which is objective 3. At the probe state $s^\*$ you draw $N = 1000$ action samples from each head and score them with two metrics. Mode membership is determined by $\operatorname{sign}(a_x)$, and $d = \| \bar a_L^{exp} - \bar a_R^{exp} \|$ denotes the distance between the means of the expert's two modes.

- **Mode balance** is $C = 1 - |p_L - p_R|$, where $p_L$ and $p_R$ are the fractions of samples in the left and right modes. The expert scores approximately 1, and a head that has collapsed onto one mode scores approximately 0.
- **Indecision mass** is $I = \frac{1}{N} \#\{ a : |a_x| < 0.2\, d \}$, the fraction of samples that fall in the valley between the two modes. The expert scores approximately 0, and the MSE head scores approximately 1 by construction, since it returns the conditional mean.

1. Before computing anything, write in `RESULTS.md` a table of predicted $C$ and $I$ for each of the four heads, with one sentence of reasoning per head. The prediction is worth making because the mechanism in the Principles section determines the answer for the MSE head exactly, and for the generative heads it determines the direction.
2. Compute $C$ and $I$ per head using `metrics.py`, with the formulas in the docstrings, at seed 0. A single seed is sufficient here because the claim being tested is a mechanism rather than a ranking, and you should say so in the writeup.
3. Scatter-plot the 1000 samples from each head over the expert's samples at $s^\*$, one panel per head.
4. Reconcile the predicted table with the measured one.

**✅ Checkpoint:** the MSE head has $I > 0.9$, and at least two generative heads have $C > 0.8$ and $I < 0.1$. If the CVAE collapses to one mode at $\beta = 1$, that is a real finding rather than a bug; record it and continue.

## Exercise 5 — Induce posterior collapse in the CVAE [Predict → Run]

The CVAE has a failure mode of its own, distinct from mode-averaging, in which the KL term pulls the encoder's posterior so close to the prior that the decoder learns to ignore $z$ altogether. This exercise induces that failure deliberately by raising $\beta$. Before retraining, predict in one sentence what $\beta = 10$ will do to $q_\phi(z|s,a)$ and therefore to the mode-balance metric $C$. Then retrain the CVAE at $\beta \in \{1, 10\}$ and report $C$ for both.

**✅ Checkpoint:** the prediction for $\beta = 10$ is reconciled against the measurement, and the mechanism is stated: the KL term pulls $q_\phi$ toward the prior, the latent carries no information about the mode, and the decoder collapses to a single output.

## Exercise 6 — Separate the two failures in closed loop [Predict → Run]

Everything so far has been measured at a single state. This exercise runs each head as a policy and shows that mode-averaging and covariate shift are separate failures, which is objectives 4 and 5. Before running anything, fill in a two-by-two prediction grid in `RESULTS.md` with rows {MSE head, one generative head} and columns {standard start distribution, start jitter widened five-fold}; in each cell predict the direction of the success rate and name the failure that drives it. Then fill in a second two-by-two grid for single-step execution versus 8-action chunks. The predictions are worth making because the Principles section implies a specific pattern: sampling should help in the standard condition, chunking should help in the widened condition, and only their combination should help in both.

1. **Rollouts.** Run 200 episodes per head from a fixed seed list, and report the success rate, the collision rate, and the mean minimum clearance from the obstacle. Overlay 50 trajectories per head on one figure. The expected pattern is that the MSE head drives into the obstacle from central starts, whereas the generative heads commit to one arc or the other.
2. **Isolate covariate shift.** Rerun with the start jitter widened by a factor of five, so that the policy begins in states the expert never visited. Success should degrade for every head, including the generative ones. This is the evidence that sampling from a generative head fixes mode-averaging but does nothing about distribution shift.
3. **Chunking probe.** Retrain the MSE head and one generative head to predict 8-action chunks that are executed open-loop, and compare single-step against chunked execution in both start conditions. The expected pattern is that chunking helps the shift failure for both heads, because there are fewer decision points at which drift can accumulate, while the chunked MSE head still averages modes, because its chunk is the average path and that path runs into the obstacle.

**✅ Checkpoint:** both two-by-two grids have a measured number in every cell alongside the predicted direction, and every disagreement between prediction and measurement is explained.

## Exercise 7 — Write the two-by-two account [Write]

In at most twelve sentences in `RESULTS.md`, describe the two failures, state which head fixes which, and explain why chunking's help is asymmetric. Then map ACT's two components (a CVAE and chunking) and Diffusion Policy's two components (a DDPM and chunking) onto the failures they address.

**✅ Checkpoint:** every claim in the paragraph cites a number from Exercise 4 or Exercise 6.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `env.py`, `expert.py` | exact specification above; seeded; expert success = 100% |
| `heads/{mse,cvae,ddpm,cfm}.py` | each under 150 lines; common `sample(s, n) -> (n, 2)` interface, **reused unchanged by Lesson 15** (the DDPM and CFM heads on a 2-D slice of real data); loss lines annotated (Exercise 3) |
| `metrics.py` | $C$ and $I$ as documented functions with the formulas in their docstrings |
| `run_all.py` | `python run_all.py --seed 0` reproduces every number and figure |
| `plots/` | expert world, four-way sample scatter, four-way rollout overlay, chunking two-by-two |
| `RESULTS.md` | predicted-versus-measured tables for Exercises 4–6, the CVAE-β finding, and the two-by-two account |

## Done when

- [ ] MSE head: $I > 0.9$ at $s^\*$ and visible obstacle collisions in rollouts.
- [ ] At least two generative heads: $C > 0.8$, $I < 0.1$, and committed single-mode rollouts.
- [ ] The widened-jitter run degrades all heads, showing that shift is independent of the head.
- [ ] The chunking probe shows the asymmetry: chunking helps shift and does not cure averaging.
- [ ] Every prediction table was written before its run and reconciled after it.
- [ ] All numbers reproduce from `python run_all.py --seed 0`.

## Self-check

1. Why does the conditional mean minimize MSE, and for which loss would the conditional median be optimal instead? Would an L1 loss fix mode-averaging?
2. The DAgger bound is $O(\epsilon T^2)$. Which experimental knob in Exercise 6 plays the role of $T$, and which plays the role of $\epsilon$?
3. Your CVAE at $\beta = 10$ collapsed to one mode. Mechanistically, what did the KL term do to $q_\phi(z|s,a)$?
4. The CFM head sampled well with 10 Euler steps, whereas the DDPM needed about 100 ancestral steps. What property of the optimal-transport path explains the gap? (Lesson 15 measures this on real data.)
5. ACT (Lesson 14) uses both a CVAE and chunking. Map each component onto the failure it addresses.

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| DDPM samples are noise blobs | too few timesteps for sample quality, or the $t$ embedding is not reaching the network | keep 100 steps; verify that the loss as a function of $t$ is roughly flat |
| CVAE ignores $z$ (both modes gone) | posterior collapse | lower $\beta$, or warm up the KL weight over the first 2k steps |
| CFM loss is fine for $t \in (0,1)$ but samples are bad | too few Euler steps, or actions not clipped | use 10 steps; clip actions to the environment bound after integration |
| Metrics unstable | $N$ too small, or the expert harvest radius too thin | use $N = 1000$ and a harvest radius of 0.05 |
| `mps` dtype errors in the DDPM | float64 defaults inherited from numpy schedules | build schedules as float32 tensors |
| MSE head appears multimodal | the trunk is memorizing jittered starts, so the mode leaks through the state | probe at $s^\*$ exactly; keep the jitter at 0.02 and the fork well below the obstacle |

## Going deeper

- **Energy-based head.** Add an implicit behaviour-cloning head (Florence et al.) with a derivative-free argmin at inference, and compare its $C$, $I$ and inference latency against the CFM head. The comparison shows the trade-off that led the field to prefer diffusion and flow models over energy-based ones.
- **Seeds.** Rerun Exercises 4 and 6 at seeds 1 and 2 and report the mean and range, which turns the single-seed mechanism claim into a ranking claim.

## References

- Florence et al. *Implicit Behavioral Cloning*, CoRL 2022. arXiv:2109.00137.
- Ross, Gordon, Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (DAgger), AISTATS 2011. arXiv:1011.0686.
- LeRobot team. *Robot Learning: A Tutorial*, §4.0–4.1. arXiv:2510.12403.
- Lipman et al. *Flow Matching for Generative Modeling*, ICLR 2023. arXiv:2210.02747.
