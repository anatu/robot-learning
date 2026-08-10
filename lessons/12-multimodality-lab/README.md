# Lesson 12 — Why Generative Policies: The Multimodality Lab

Build a toy world where you control the data-generating process completely, then demonstrate — with numbers, not folklore — the two distinct failure modes that motivate every architecture in Phase 4: mode-averaging and compounding covariate shift.

| | |
|---|---|
| **Phase** | 4 — Generative imitation policies |
| **Time** | 1 long session (5–7 h), all Mac-local |
| **Cost** | $0 |
| **Prerequisites** | 08 (comfortable writing PyTorch training loops from scratch); Lesson 13 pairs naturally as a co-read |
| **Feeds into** | 13 (the math behind the two heads that worked), 14–15 (ACT and DP are these heads scaled up), 17 (π0's flow head is Part 2's CFM head with a VLM stapled on) |

## Learning objectives

After this lesson you can:

1. **State** the BC objective precisely and explain why an MSE regressor is maximum-likelihood under a unimodal Gaussian — and what that assumption does to multimodal data.
2. **Implement** four policy heads from scratch — MSE regressor, CVAE, DDPM, conditional flow matching — each in under 150 lines.
3. **Quantify** multimodality handling with a mode-balance and an indecision metric you define as formulas, not vibes.
4. **Demonstrate** closed-loop compounding error and distinguish it experimentally from mode-averaging.
5. **Explain** which of the two failures action chunking mitigates, and why it can't fix the other.

## Background

**BC is supervised learning with a trap.** Behavior cloning fits $\min_\theta \, \mathbb{E}_{(o,a)\sim\mathcal{D}}\!\left[-\log \pi_\theta(a\mid o)\right]$. With a deterministic head and MSE loss you've chosen $\pi_\theta(a|o) = \mathcal{N}(f_\theta(o), \sigma^2 I)$ — maximum likelihood then forces $f_\theta(o) \to \mathbb{E}[a\mid o]$, the *conditional mean*. If the expert goes left half the time and right half the time, the conditional mean goes straight — an action no expert ever took, possibly off the feasible manifold entirely. That's **mode-averaging**: a modeling-class failure, present even with infinite data and perfect optimization.

**Compounding error is a different disease.** The policy is trained on the expert's state distribution but tested on its *own*. Each small error drifts the state slightly off-distribution, where the policy is worse, which drifts further. Ross & Bagnell's classic analysis bounds the cost gap as $O(\epsilon T^2)$ in the horizon $T$, versus $O(\epsilon T)$ for supervised learning — quadratic, not linear. Crucially: this failure exists even for *unimodal* experts. The two failures are orthogonal, and this lab's job is to show them separately.

**The generative fixes.** A CVAE learns $p(a|o)$ through a latent $z$ (Lesson 13 derives its ELBO); a DDPM learns to reverse a Gaussian corruption process with the simplified loss $\mathbb{E}_{t,\epsilon}\|\epsilon - \epsilon_\theta(x_t, t)\|^2$; conditional flow matching regresses a velocity field along the optimal-transport path $x_t = (1-t)x_0 + t x_1$ with target velocity $x_1 - x_0$, where $x_0\sim\mathcal{N}(0,I)$ and $x_1$ is data. All three can put probability mass on *both* modes instead of their average. None of them, by themselves, fix covariate shift.

| Source | Read for |
|---|---|
| Tutorial §4.0–4.1 | the formal BC setup and how the tutorial motivates generative heads; which failure its Figure attributes to which cause |
| Florence et al. 2022 (Implicit BC), §1 + Fig. 2 | the canonical mode-averaging picture; why implicit/energy models were the first fix |
| Ross, Gordon, Bagnell 2011 (DAgger), §2 | where the $O(\epsilon T^2)$ bound comes from — you only need the problem setup and Thm 2.1's statement |

## Part 1 — The toy world (≈1 h)

A 2D point-mass navigation task, specified exactly so results are comparable across heads.

1. **Environment.** State $s = (x, y) \in [-1,1]^2$. Start $s_0 = (0, -0.8)$ plus $\mathcal{N}(0, 0.02^2)$ jitter. Goal: reach within 0.1 of $(0, 0.8)$. One circular obstacle, center $(0,0)$, radius $0.25$. Action $a \in \mathbb{R}^2$ is a displacement, clipped to $\|a\| \le 0.05$; dynamics $s' = s + a$. Episode fails on obstacle penetration or after 300 steps. Implement as a plain Python class with `reset(seed)` / `step(a)` — no gym dependency needed.
2. **Expert.** At episode start flip a fair coin for a *left* or *right* arc: three waypoints $(\pm 0.45, -0.5) \to (\pm 0.45, 0.5) \to (0, 0.8)$ (sign per coin). The expert steers toward the current waypoint at full step size with $\mathcal{N}(0, 0.005^2)$ action noise, advancing waypoints within 0.1. Record 500 episodes → ~25–30k $(s, a)$ pairs.
3. Plot all expert trajectories (color by mode) and a quiver of expert actions. The picture to burn in: at states near $x=0$ below the obstacle, the dataset contains two opposite action clusters.

**✅ Checkpoint:** trajectory plot shows two clean symmetric arcs; expert success rate = 100%; at probe state $s^\* = (0, -0.4)$, expert actions (harvest from the dataset within radius 0.05 of $s^\*$) form two clusters with near-zero mass between them.

## Part 2 — Four heads, one training harness (≈2–3 h)

One `train(head, dataset, seed)` function; four `nn.Module`s, each < 150 lines including sampling code. Shared trunk: MLP 3×128, ReLU. Adam 1e-3, batch 256, 20k steps, seed 0 (plus 2 more seeds for the report).

1. **MSE regressor** — trunk → 2D action. The control arm.
2. **CVAE** — encoder $q_\phi(z|s,a)$ and decoder $p_\theta(a|s,z)$, $z \in \mathbb{R}^2$, loss = MSE recon + $\beta\, D_{KL}(q_\phi \| \mathcal{N}(0,I))$ with $\beta = 1$ to start. Sample via $z \sim \mathcal{N}(0,I)$.
3. **DDPM** — 100 timesteps, cosine ($\text{squaredcos\_cap\_v2}$-style) schedule, $\epsilon$-prediction, timestep embedding (sinusoidal, 64-d) concatenated to $(s, a_t)$. Ancestral sampling with all 100 steps.
4. **CFM** — regress $v_\theta(x_t, t, s)$ onto $x_1 - x_0$ along $x_t = (1-t)x_0 + t x_1$, $t \sim \mathcal{U}[0,1]$. Sample by Euler integration, 10 steps.

**✅ Checkpoint:** all four training losses converge (MSE-head loss will plateau *high* — it cannot fit two modes; write down why before reading on). Sampling 1000 actions at $s^\*$ takes < 5 s per head on `mps`/CPU.

## Part 3 — Measure the multimodality (≈1 h)

At probe state $s^\*$, draw $N = 1000$ action samples per head. Define, with mode membership by $\operatorname{sign}(a_x)$ and $d = \| \bar a_L^{exp} - \bar a_R^{exp} \|$ the distance between expert mode means:

- **Mode balance** $C = 1 - |p_L - p_R|$, where $p_{L/R}$ are the sampled left/right fractions. Expert ≈ 1; a mode-collapsed head ≈ 0.
- **Indecision mass** $I = \frac{1}{N} \#\{ a : |a_x| < 0.2\, d \}$ — mass in the valley between modes. Expert ≈ 0; the MSE head ≈ 1 by construction.

1. Compute $C$ and $I$ per head (3 seeds → mean ± range) into one table.
2. Scatter-plot the 1000 samples per head over the expert samples — the lesson's signature figure.
3. Sweep the CVAE's $\beta \in \{0.1, 1, 10\}$ and report $C$: watch posterior collapse kill mode coverage at high $\beta$.

**✅ Checkpoint:** MSE head: $I > 0.9$. At least two generative heads: $C > 0.8$ and $I < 0.1$. If the CVAE mode-collapses at $\beta=1$, that's a real finding — record it and report the best $\beta$.

## Part 4 — Closed-loop: two failures, separated (≈1–2 h)

1. **Rollouts.** 200 episodes per head (fixed seed list). Report success rate, collision rate, and mean minimum obstacle clearance. Overlay 50 trajectories per head on one figure. Expected shape: the MSE head drives into the obstacle from center starts; generative heads commit to an arc.
2. **Isolate covariate shift.** Rerun with start jitter widened ×5 (states the expert never visited). Success degrades for *all* heads — including the generative ones. This is your evidence that sampling fixes mode-averaging but not distribution shift.
3. **Chunking probe.** Retrain the MSE head and one generative head to predict 8-action chunks, executed open-loop. Compare single-step vs chunked on both the standard and widened-jitter conditions. Expected: chunking helps the *shift* failure (fewer decision points → less drift accumulation) for both heads, but the chunked MSE head still averages modes — its chunk is the average *path*, straight into the obstacle.

**✅ Checkpoint:** a 2×2 story you can defend — {MSE, generative} × {mode failure, shift failure} — with a number in each cell.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `env.py`, `expert.py` | exact spec above; seeded; `pytest` asserts expert success = 100% |
| `heads/{mse,cvae,ddpm,cfm}.py` | each < 150 lines; common `sample(s, n)` interface |
| `metrics.py` | $C$ and $I$ as documented functions with the formulas in docstrings |
| `plots/` | expert world, 4-way sample scatter, 4-way rollout overlay, chunking 2×2 |
| `RESULTS.md` | metric table (3 seeds), the 2×2 narrative, CVAE-β finding, ≤ 12 sentences |

## Done when

- [ ] MSE head: $I > 0.9$ at $s^\*$ and visible obstacle collisions in rollouts.
- [ ] ≥ 2 generative heads: $C > 0.8$, $I < 0.1$, and committed single-mode rollouts.
- [ ] Widened-jitter run degrades all heads — shift shown to be head-independent.
- [ ] Chunking probe shows the asymmetry: helps shift, doesn't cure averaging.
- [ ] All numbers reproduce from `python run_all.py --seed 0`.

## Self-check

1. Why does the conditional mean minimize MSE, and for which loss would the conditional *median* be optimal instead? Would L1 fix mode-averaging?
2. The DAgger bound is $O(\epsilon T^2)$. Which experimental knob in Part 4 is "T", and which is "$\epsilon$"?
3. Your CVAE at $\beta = 10$ likely collapsed to one mode. Mechanistically, what did the KL term do to $q_\phi(z|s,a)$?
4. CFM sampled well with 10 Euler steps; DDPM needed ~100 ancestral steps. What property of the OT path explains the gap? (Lesson 15 measures this on real data.)
5. ACT (Lesson 14) uses a CVAE *and* chunking. Map each component onto the failure it addresses.

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| DDPM samples are noise blobs | too few timesteps for sample quality, or $t$ embedding not reaching the net | keep 100 steps; verify loss vs $t$ is roughly flat |
| CVAE ignores $z$ (both modes gone) | posterior collapse | lower $\beta$, or KL warm-up over first 2k steps |
| CFM fine at $t{\in}(0,1)$, bad samples anyway | integrating with too few / unclipped Euler steps | 10 steps; clip actions to the env bound *after* integration |
| Metrics unstable across seeds | $N$ too small, probe-radius harvest too thin | $N = 1000$; widen expert harvest radius to 0.05 |
| `mps` dtype errors in the DDPM | float64 defaults from numpy schedules | build schedules in float32 tensors |
| MSE head suspiciously multimodal | trunk memorizing jittered starts (mode leaks through state) | probe at $s^\*$ exactly; jitter is 0.02, keep the fork well below the obstacle |

## Stretch

Add a fifth head: energy-based/implicit BC (Florence et al.) with a derivative-free argmin at inference. Compare its $C$/$I$ and its inference latency against the CFM head — the trade that explains why the field went with diffusion/flow instead.

## References

- Florence et al. *Implicit Behavioral Cloning*, CoRL 2022. arXiv:2109.00137.
- Ross, Gordon, Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (DAgger), AISTATS 2011. arXiv:1011.0686.
- LeRobot team. *Robot Learning: A Tutorial*, §4.0–4.1. arXiv:2510.12403.
- Lipman et al. *Flow Matching for Generative Modeling*, ICLR 2023. arXiv:2210.02747.
