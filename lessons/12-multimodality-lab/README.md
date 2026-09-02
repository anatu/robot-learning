# Lesson 12 — Why Generative Policies: The Multimodality Lab

Build a toy world where you control the data-generating process completely, then demonstrate — with predictions written first and numbers second — the two distinct failure modes that motivate every architecture in Phase 4: mode-averaging and compounding covariate shift.

| | |
|---|---|
| **Phase** | 4 — Generative imitation policies |
| **Time** | 1 session (4–5 h desk time, AI-assisted; training runs are minutes on `mps`/CPU) |
| **Cost** | $0 |
| **Prerequisites** | 08 (you can read a PyTorch training loop); Lesson 13 pairs naturally as a co-read |
| **Feeds into** | 13 (the math behind the two heads that worked), 14–15 (ACT and DP are these heads scaled up; Lesson 15 reuses the DDPM and CFM heads directly), 17 (π0's flow head is the CFM head with a VLM stapled on) |

## Learning objectives

After this lesson you can:

1. **State** the BC objective precisely and explain why an MSE regressor is maximum-likelihood under a unimodal Gaussian — and what that assumption does to multimodal data.
2. **Explain** each of the four heads — MSE, CVAE, DDPM, conditional flow matching — as one loss equation, and annotate its code against that equation.
3. **Predict and quantify** multimodality handling with a mode-balance and an indecision metric defined as formulas.
4. **Demonstrate** closed-loop compounding error and distinguish it experimentally from mode-averaging.
5. **Explain** which of the two failures action chunking mitigates, and why it cannot fix the other.

## Principles

**BC is supervised learning with a trap.** Behavior cloning fits $\min_\theta \, \mathbb{E}_{(o,a)\sim\mathcal{D}}\!\left[-\log \pi_\theta(a\mid o)\right]$. With a deterministic head and MSE loss you have chosen $\pi_\theta(a|o) = \mathcal{N}(f_\theta(o), \sigma^2 I)$; maximum likelihood then forces $f_\theta(o) \to \mathbb{E}[a\mid o]$, the *conditional mean*. If the expert goes left half the time and right half the time, the conditional mean goes straight — an action no expert ever took, possibly off the feasible manifold entirely. That is **mode-averaging**: a modeling-class failure, present even with infinite data and perfect optimization.

**Compounding error is a different disease.** The policy is trained on the expert's state distribution but tested on its *own*. Each small error drifts the state slightly off-distribution, where the policy is worse, which drifts further. Ross & Bagnell bound the cost gap as $O(\epsilon T^2)$ in the horizon $T$, versus $O(\epsilon T)$ for supervised learning. This failure exists even for *unimodal* experts. The two failures are orthogonal; this lab's job is to show them separately.

**The generative fixes, as four losses.** These are the equations Exercise 3 annotates:

| Head | Loss | What it models |
|---|---|---|
| MSE | $\|a - f_\theta(s)\|^2$ | the conditional mean (Gaussian MLE) |
| CVAE | $\|a - p_\theta(s, z)\|^2 + \beta\, D_{KL}\!\left(q_\phi(z|s,a)\,\|\,\mathcal{N}(0,I)\right)$, $z \sim q_\phi$ via reparameterization | $p(a|s)$ through a latent $z$ (Lesson 13 derives the ELBO) |
| DDPM | $\mathbb{E}_{t,\epsilon}\|\epsilon - \epsilon_\theta(x_t, t, s)\|^2$, $x_t = \sqrt{\bar\alpha_t}\,a + \sqrt{1-\bar\alpha_t}\,\epsilon$ | the reverse of a Gaussian corruption process (Lesson 13's simplified loss) |
| CFM | $\mathbb{E}_{t,x_0}\|v_\theta(x_t, t, s) - (x_1 - x_0)\|^2$, $x_t = (1-t)x_0 + t x_1$, $x_0 \sim \mathcal{N}(0,I)$, $x_1 = a$ | a velocity field along the optimal-transport path |

All three generative heads can put probability mass on *both* modes instead of their average. None of them, by themselves, fixes covariate shift.

**Chunking attacks the horizon, not the mean.** Predicting $H$ actions per decision divides the number of decision points by $H$, so drift accumulates over $T/H$ steps instead of $T$. It does nothing for a head whose chunk is the *average path*.

**Carry forward**

- MSE regression is Gaussian maximum likelihood; on multimodal data it returns the conditional mean, which may be an action nobody took.
- Mode-averaging is a modeling-class failure; compounding error is a distribution-shift failure. Different diseases, different cures.
- Sampling from a generative head fixes averaging and leaves shift untouched; chunking mitigates shift and leaves averaging untouched.
- The four losses in the table are the whole Phase 4 vocabulary. ACT is CVAE + chunking; Diffusion Policy is DDPM + chunking; π0 is CFM + chunking + a VLM.

| Source | Read for |
|---|---|
| Tutorial §4.0–4.1 | the formal BC setup and how the tutorial motivates generative heads; which failure its Figure attributes to which cause |
| Florence et al. 2022 (Implicit BC), §1 + Fig. 2 | the canonical mode-averaging picture; why implicit/energy models were the first fix |
| Ross, Gordon, Bagnell 2011 (DAgger), §2 | where the $O(\epsilon T^2)$ bound comes from — the problem setup and Thm 2.1's statement |

## Exercise 1 — The toy world [Build]

Tests nothing yet; produces the controlled substrate. The spec is exact so results are comparable across heads. An AI tool drafts `env.py` and `expert.py`; you verify the checkpoint.

1. **Environment.** State $s = (x, y) \in [-1,1]^2$. Start $s_0 = (0, -0.8)$ plus $\mathcal{N}(0, 0.02^2)$ jitter. Goal: reach within 0.1 of $(0, 0.8)$. One circular obstacle, center $(0,0)$, radius $0.25$. Action $a \in \mathbb{R}^2$ is a displacement, clipped to $\|a\| \le 0.05$; dynamics $s' = s + a$. Episode fails on obstacle penetration or after 300 steps. Plain Python class with `reset(seed)` / `step(a)`; no gym dependency.
2. **Expert.** At episode start flip a fair coin for a *left* or *right* arc: three waypoints $(\pm 0.45, -0.5) \to (\pm 0.45, 0.5) \to (0, 0.8)$ (sign per coin). Steer toward the current waypoint at full step size with $\mathcal{N}(0, 0.005^2)$ action noise, advancing waypoints within 0.1. Record 500 episodes → ~25–30k $(s, a)$ pairs.
3. Plot all expert trajectories (color by mode) and a quiver of expert actions. The picture to burn in: at states near $x=0$ below the obstacle, the dataset contains two opposite action clusters.

**✅ Checkpoint:** two clean symmetric arcs; expert success rate = 100%; at probe state $s^\* = (0, -0.4)$, expert actions harvested within radius 0.05 of $s^\*$ form two clusters with near-zero mass between them.

## Exercise 2 — Four heads, one harness [Build]

Tests objective 2's first half: the four losses exist as runnable code with a common interface. Spec for the AI tool:

- One `train(head, dataset, seed)` function; four `nn.Module`s in `heads/{mse,cvae,ddpm,cfm}.py`, each < 150 lines including sampling, each exposing `sample(s, n) -> (n, 2)` actions. Shared trunk MLP 3×128, ReLU. Adam 1e-3, batch 256, 20k steps, seed 0.
- **MSE**: trunk → 2-D action.
- **CVAE**: encoder $q_\phi(z|s,a)$ and decoder $p_\theta(a|s,z)$, $z \in \mathbb{R}^2$, loss per the Principles table with $\beta = 1$. Sample via $z \sim \mathcal{N}(0,I)$.
- **DDPM**: 100 timesteps, cosine (`squaredcos_cap_v2`-style) schedule, $\epsilon$-prediction, sinusoidal 64-d timestep embedding concatenated to $(s, a_t)$. Ancestral sampling with all 100 steps.
- **CFM**: regress $v_\theta(x_t, t, s)$ onto $x_1 - x_0$ along $x_t = (1-t)x_0 + t x_1$, $t \sim \mathcal{U}[0,1]$. Sample by Euler integration, 10 steps.
- Schedules built as float32 tensors (see Pitfalls).

**✅ Checkpoint:** all four training losses converge; the MSE head's loss plateaus *high* — write down why before reading on. Sampling 1000 actions at $s^\*$ takes < 5 s per head on `mps`/CPU.

## Exercise 3 — Annotate the four losses [Read the kernel]

Tests objective 2's second half. In each head's file, annotate the loss computation line by line with the equation from the Principles table: which term is the reconstruction, where the reparameterization happens in the CVAE, which line forms $x_t$ and from which schedule quantity in the DDPM, which line is the target velocity in the CFM. Then, for the MSE head, write the two-line argument that its optimum is $\mathbb{E}[a|s]$.

**✅ Checkpoint:** every loss line carries an equation comment; the MSE argument is in `RESULTS.md`.

## Exercise 4 — Measure the multimodality [Predict → Run]

Tests objective 3. At probe state $s^\*$, draw $N = 1000$ action samples per head. With mode membership by $\operatorname{sign}(a_x)$ and $d = \| \bar a_L^{exp} - \bar a_R^{exp} \|$ the distance between expert mode means:

- **Mode balance** $C = 1 - |p_L - p_R|$, where $p_{L/R}$ are the sampled left/right fractions. Expert ≈ 1; a mode-collapsed head ≈ 0.
- **Indecision mass** $I = \frac{1}{N} \#\{ a : |a_x| < 0.2\, d \}$ — mass in the valley between modes. Expert ≈ 0; the MSE head ≈ 1 by construction.

1. **Write first**, in `RESULTS.md`: a 4×2 table of predicted $C$ and $I$ per head, with one reason each.
2. Compute $C$ and $I$ per head (`metrics.py`, formulas in docstrings), seed 0. One seed: the claim is a mechanism, not a ranking; say so.
3. Scatter-plot the 1000 samples per head over the expert samples — the lesson's signature figure.
4. Reconcile the table.

**✅ Checkpoint:** MSE head $I > 0.9$; at least two generative heads $C > 0.8$ and $I < 0.1$. A CVAE that mode-collapses at $\beta=1$ is a finding, not a bug — record it.

## Exercise 5 — Posterior collapse [Predict → Run]

Tests the CVAE's specific failure. Predict what $\beta = 10$ does to $q_\phi(z|s,a)$ and therefore to $C$, in one sentence. Retrain the CVAE at $\beta \in \{1, 10\}$, report $C$ for both.

**✅ Checkpoint:** the $\beta = 10$ prediction is reconciled; the mechanism (KL term pulls $q_\phi$ to the prior, decoder ignores $z$) is stated.

## Exercise 6 — Closed loop: two failures, separated [Predict → Run]

Tests objectives 4 and 5. Before running anything, fill a 2×2 prediction grid in `RESULTS.md`: {MSE, one generative head} × {standard start, ×5 start jitter}, predicting success rate direction and *which* failure drives each cell. Then a second 2×2 for single-step vs 8-action chunks.

1. **Rollouts.** 200 episodes per head (fixed seed list). Report success rate, collision rate, mean minimum obstacle clearance. Overlay 50 trajectories per head on one figure. Expected shape: the MSE head drives into the obstacle from center starts; generative heads commit to an arc.
2. **Isolate covariate shift.** Rerun with start jitter widened ×5 (states the expert never visited). Success degrades for *all* heads, including generative ones — evidence that sampling fixes mode-averaging but not distribution shift.
3. **Chunking probe.** Retrain the MSE head and one generative head to predict 8-action chunks, executed open-loop. Compare single-step vs chunked on both start conditions. Expected: chunking helps the *shift* failure for both heads, but the chunked MSE head still averages — its chunk is the average *path*, straight into the obstacle.

**✅ Checkpoint:** both 2×2 grids have a number in each cell and a reconciled prediction.

## Exercise 7 — The 2×2 story [Write]

In ≤ 12 sentences in `RESULTS.md`: the two failures, which head fixes which, and why chunking's help is asymmetric. Map ACT's two components (CVAE, chunking) and Diffusion Policy's (DDPM, chunking) onto the failures they address.

**✅ Checkpoint:** the paragraph names a number from Exercise 4 or 6 for every claim.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `env.py`, `expert.py` | exact spec above; seeded; expert success = 100% |
| `heads/{mse,cvae,ddpm,cfm}.py` | each < 150 lines; common `sample(s, n) -> (n, 2)` interface — **reused unchanged by Lesson 15** (the DDPM and CFM heads on a real-data 2-D slice); loss lines annotated (Exercise 3) |
| `metrics.py` | $C$ and $I$ as documented functions, formulas in docstrings |
| `run_all.py` | `python run_all.py --seed 0` reproduces every number and figure |
| `plots/` | expert world, 4-way sample scatter, 4-way rollout overlay, chunking 2×2 |
| `RESULTS.md` | predicted-vs-measured tables (Exercises 4–6), CVAE-β finding, the 2×2 story |

## Done when

- [ ] MSE head: $I > 0.9$ at $s^\*$ and visible obstacle collisions in rollouts.
- [ ] ≥ 2 generative heads: $C > 0.8$, $I < 0.1$, committed single-mode rollouts.
- [ ] Widened-jitter run degrades all heads — shift shown to be head-independent.
- [ ] Chunking probe shows the asymmetry: helps shift, does not cure averaging.
- [ ] Every prediction table was written before its run and reconciled after.
- [ ] All numbers reproduce from `python run_all.py --seed 0`.

## Self-check

1. Why does the conditional mean minimize MSE, and for which loss would the conditional *median* be optimal instead? Would L1 fix mode-averaging?
2. The DAgger bound is $O(\epsilon T^2)$. Which experimental knob in Exercise 6 is "T", and which is "$\epsilon$"?
3. Your CVAE at $\beta = 10$ collapsed to one mode. Mechanistically, what did the KL term do to $q_\phi(z|s,a)$?
4. CFM sampled well with 10 Euler steps; DDPM needed ~100 ancestral steps. What property of the OT path explains the gap? (Lesson 15 measures this on real data.)
5. ACT (Lesson 14) uses a CVAE *and* chunking. Map each component onto the failure it addresses.

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| DDPM samples are noise blobs | too few timesteps for sample quality, or $t$ embedding not reaching the net | keep 100 steps; verify loss vs $t$ is roughly flat |
| CVAE ignores $z$ (both modes gone) | posterior collapse | lower $\beta$, or KL warm-up over first 2k steps |
| CFM fine at $t{\in}(0,1)$, bad samples anyway | too few / unclipped Euler steps | 10 steps; clip actions to the env bound *after* integration |
| Metrics unstable | $N$ too small, probe-radius harvest too thin | $N = 1000$; expert harvest radius 0.05 |
| `mps` dtype errors in the DDPM | float64 defaults from numpy schedules | build schedules in float32 tensors |
| MSE head suspiciously multimodal | trunk memorizing jittered starts (mode leaks through state) | probe at $s^\*$ exactly; jitter is 0.02, keep the fork well below the obstacle |

## Going deeper

- **Energy-based head.** Add implicit BC (Florence et al.) with a derivative-free argmin at inference; compare its $C$/$I$ and inference latency against the CFM head — the trade that explains why the field went with diffusion/flow.
- **Seeds.** Rerun Exercises 4 and 6 at seeds 1 and 2 and report mean ± range; the single-seed mechanism claim becomes a ranking claim.

## References

- Florence et al. *Implicit Behavioral Cloning*, CoRL 2022. arXiv:2109.00137.
- Ross, Gordon, Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (DAgger), AISTATS 2011. arXiv:1011.0686.
- LeRobot team. *Robot Learning: A Tutorial*, §4.0–4.1. arXiv:2510.12403.
- Lipman et al. *Flow Matching for Generative Modeling*, ICLR 2023. arXiv:2210.02747.
