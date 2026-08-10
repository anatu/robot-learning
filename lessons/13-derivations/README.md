# Lesson 13 — Derivation Dossier

The tutorial derives the VAE ELBO and the 13-step DDPM ELBO but defers the steps that actually make diffusion and flow-matching trainable. Complete them — with every identity named, and with numerical checks that prove your algebra, not just typeset it.

| | |
|---|---|
| **Phase** | 4 — Generative imitation policies |
| **Time** | 2 desk sessions (8–10 h). No compute beyond toy scripts |
| **Cost** | $0 |
| **Prerequisites** | 12 (you've *seen* these models work; now you earn them). Graduate probability: Gaussian conjugacy, KL divergences, change of variables |
| **Feeds into** | 14 (ACT's loss is your CVAE ELBO with L1 recon), 15 (the sampler study assumes you know why $\epsilon$/score/velocity are interconvertible), 17 (π0's flow head + timestep schedule) |

## Learning objectives

After this lesson you can:

1. **Derive** the DDPM simplified $\epsilon$-prediction loss from the ELBO (tutorial eq. 42 → 44) with every step justified by a named identity.
2. **Prove** the flow-matching marginal/conditional gradient equivalence and sketch why the marginal field transports the marginal path.
3. **Convert** between the three parameterizations — noise $\epsilon_\theta$, score $s_\theta$, velocity $v_\theta$ — and verify the conversions numerically.
4. **Defend** a design choice: why π0 trains a flow head with a Beta-skewed timestep distribution rather than a DDPM head with uniform timesteps.

## Background

**The rules of the game.** A derivation note is only trustworthy if it's checkable. Two disciplines make it so. First, a fixed toolbox: every step must cite one of the **allowed identities** below, so "it can be shown" never appears. Second, every closed-form you derive gets a 20-line numerical check where Gaussians are exact — if your $\tilde\mu_t$ formula is wrong, a script says so before a reader does.

**Allowed identities** (the complete toolbox; number them and cite by number in the note):

- **(I1)** Bayes' rule applied to the forward process: $q(x_{t-1} \mid x_t, x_0) = q(x_t \mid x_{t-1})\, q(x_{t-1} \mid x_0)\, /\, q(x_t \mid x_0)$ — legal because the forward chain is Markov.
- **(I2)** Gaussian algebra: products of Gaussian densities in the exponent; completing the square; the closed-form KL between two Gaussians.
- **(I3)** Reparameterization: $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$, and its inversion for $x_0$.
- **(I4)** Tower rule / Fubini: exchanging and nesting expectations over $t$, $x_1$, $x_t$.
- **(I5)** The continuity equation: $\partial_t p_t + \nabla\cdot(p_t u_t) = 0$ characterizes when a field $u_t$ transports the path $p_t$.
- **(I6)** Expanding $\|a - b\|^2$ and dropping $\theta$-free terms under $\nabla_\theta$.

**Notation contract.** Adopt the tutorial's notation ($\alpha_t$, $\bar\alpha_t$, $\beta_t$, eq. numbers 20–49) everywhere, and translate Luo's and Lipman's notation into it when you import their arguments. Half of all diffusion-math bugs are notation collisions; the note's front page carries a symbol table.

| Source | Read for |
|---|---|
| Tutorial §4.1, eqs. 20–49 | the exact starting line (eq. 42) and finish line (eq. 44) of Obligation A; the FM statement it cites without proof (Obligation B) |
| Luo 2022, *Understanding Diffusion Models* | the three-interpretations scaffolding; **read after** attempting Obligation A yourself — it's the answer key, not the exercise |
| Lipman et al. 2023 (FM) Thm 1–2; *FM Guide and Code* 2024 | the marginal/conditional equivalence in its original notation; the guide's §4 for the OT-path specialization |
| Permenter & Yuan 2024 | denoising-as-projection: the geometric view for Obligation C |
| π0 (Black et al. 2024), flow-matching appendix | the exact timestep-sampling distribution and the stated rationale — quote it, then explain it |

## Part 1 — Obligation A: ELBO → simplified loss (≈3 h)

Produce §1 of the note: tutorial eq. 42 to eq. 44, no gaps. The required waypoints, in order:

1. Split the ELBO into reconstruction, prior, and the sum of per-step KL terms $D_{KL}\!\left(q(x_{t-1}|x_t,x_0)\,\|\,p_\theta(x_{t-1}|x_t)\right)$; state which terms are dropped or absorbed and why that's harmless.
2. Via **(I1)** + **(I2)**, derive the forward posterior in closed form:
   $q(x_{t-1}|x_t,x_0) = \mathcal{N}\!\left(\tilde\mu_t(x_t,x_0),\, \tilde\beta_t I\right)$, with $\tilde\mu_t$ and $\tilde\beta_t$ explicit in $\alpha_t, \bar\alpha_t, \beta_t$. Show the completing-the-square step; don't import the result.
3. Via **(I3)**, eliminate $x_0$ from $\tilde\mu_t$ in favor of $\epsilon$; parameterize $p_\theta$'s mean the same way; reduce the per-step KL (via **(I2)**, fixed variances) to a weighted $\epsilon$-regression: $\mathbb{E}\left[ w_t \|\epsilon - \epsilon_\theta(x_t,t)\|^2 \right]$ with $w_t = \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar\alpha_t)}$.
4. State precisely what "simplified" means: setting $w_t \to 1$ changes the objective, not just its form. Explain why it's justified (Ho et al.'s reweighting argument — it up-weights high-noise timesteps where learning is useful) and note this is an *empirical* choice; the FM timestep discussion in Part 3 lands on the same design axis.
5. **Numerical check** `checks/check_posterior.py`: 1-D, $x_0 \sim \mathcal{N}(3, 0.5^2)$, linear schedule, $T=50$. Compute $q(x_{t-1}|x_t,x_0)$ two ways — your closed-form $(\tilde\mu_t, \tilde\beta_t)$ vs brute-force Bayes on a dense grid — at 5 random $(t, x_t, x_0)$ triples.

**✅ Checkpoint:** max abs error between closed-form and grid posterior moments < 1e-4; every numbered step in §1 cites an identity (I1)–(I6).

## Part 2 — Obligation B: FM marginal/conditional equivalence (≈2–3 h)

Produce §2: the theorem the tutorial states citing Lipman et al., proved in your notation.

1. Define the conditional path $p_t(x|x_1)$ and field $u_t(x|x_1)$ for the OT construction $x_t = (1-t)x_0 + t x_1$, $u_t(x|x_1) = x_1 - x_0$; define the marginal path $p_t(x) = \int p_t(x|x_1)\, q(x_1)\, dx_1$ and the marginal field as the conditional expectation $u_t(x) = \mathbb{E}[\,u_t(x|x_1) \mid x_t = x\,]$.
2. Prove $\nabla_\theta \mathcal{L}_{FM} = \nabla_\theta \mathcal{L}_{CFM}$: expand both squared norms via **(I6)**, kill the cross-term with the tower rule **(I4)** conditioning on $x_t$, and show the residual terms are $\theta$-free. Flag the one subtle move — the exchange of expectation defining $u_t(x)$ — and the integrability condition it needs.
3. Sketch (a page, not a proof) the continuity-equation argument **(I5)**: each conditional field transports its conditional path; integrating over $x_1$ and swapping $\partial_t$ with the integral shows the marginal field transports the marginal path. Say what's swept under the rug (regularity for the swap).
4. **Numerical check** `checks/check_marginal_field.py`: 1-D target $q(x_1) = \frac{1}{2}\mathcal{N}(-2, 0.3^2) + \frac{1}{2}\mathcal{N}(2, 0.3^2)$. Everything is Gaussian-mixture-closed: compute the *analytic* marginal field $u_t(x)$ on a $(t, x)$ grid. Train a tiny MLP on the CFM objective and compare on the grid.

**✅ Checkpoint:** trained field matches the analytic marginal field with max grid error < 0.05 (in regions with $p_t(x) > 10^{-3}$); the cross-term cancellation is written out, not asserted.

## Part 3 — Obligation C: three views, one model (≈2–3 h)

Produce §3, one tight page plus a verified table.

1. Derive the conversion formulas for the Gaussian path (all via **(I3)**): score from noise $s_\theta(x_t,t) = -\epsilon_\theta(x_t,t)/\sqrt{1-\bar\alpha_t}$, and the velocity/noise relation for the OT-FM path. Present as a 3×3 table: given {$\epsilon$, score, $v$}, express the other two.
2. Connect the geometry: Permenter–Yuan's denoising-as-projection (each denoise step ≈ a gradient/projection step toward the data manifold), score matching (denoiser estimates $\nabla \log p_t$), FM (regress the transport field directly). One paragraph each, one figure total.
3. **The design-choice payoff**: π0 trains a flow head and samples timesteps from a Beta-family distribution skewed toward high noise rather than $t \sim \mathcal{U}[0,1]$. Quote the paper's stated schedule and rationale, then *explain* it with Obligation A's machinery: uniform-$t$ + simplified loss already implies a particular effective weighting; skewing $t$ is the same lever. Contrast with what an action head needs (few-step sampling at 50 Hz) vs an image generator.
4. **Numerical check** `checks/check_conversions.py`: on the Part 2 mixture, train an $\epsilon$-model only; *convert* it to a score model and a velocity model via your table; compare each against directly-trained counterparts on the grid.

**✅ Checkpoint:** converted-vs-directly-trained agreement within training noise (report the grid MAE for all three pairs); the π0 paragraph cites the exact appendix passage it explains.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `derivations.pdf` (LaTeX) or `derivations.md` (MathJax) | §1–§3 as specced; symbol table on page 1; every step cites (I1)–(I6); cross-referenced to tutorial eq. numbers |
| `checks/check_posterior.py` | passes threshold, runs < 30 s, seeded |
| `checks/check_marginal_field.py` | passes threshold; produces the field-comparison heatmap |
| `checks/check_conversions.py` | passes; produces the 3-pair MAE table |
| `RESULTS.md` | the three check outputs pasted in + 5 sentences on what surprised you |

## Done when

- [ ] All three checks pass from `pytest checks/`.
- [ ] A reader with only the tutorial open can follow §1–§3 with zero external references and zero "clearly."
- [ ] The note answers, in its own words: *why is dropping $w_t$ legitimate, and where does π0 pull the same lever?*
- [ ] Every imported argument (Luo, Lipman) is translated into the tutorial's notation.

## Self-check

1. Which single identity makes the forward posterior tractable, and why does it need the Markov property?
2. $w_t \to 1$: does the argmin of the objective change? Does the training distribution over sub-problems change? (These have different answers.)
3. In the CFM proof, why does the cross-term vanish *only in the gradient*, not in the loss values themselves?
4. Convert: a perfectly trained $\epsilon$-model at $t \to 0$ has $\epsilon_\theta \to$ what, and why does this make $\epsilon$-parameterized samplers twitchy near $t=0$ (Lesson 15 will show you empirically)?
5. Give the one-sentence version of why straight (OT) paths permit few-step sampling.

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Posterior check off by a consistent factor | $\beta_t$ vs $1-\alpha_t$ vs $\bar\alpha$ notation collision | rebuild the symbol table; derive $\bar\alpha_t = \prod \alpha_s$ once, at the top |
| Marginal-field check fails only near $t \approx 1$ | comparing in regions where $p_t \approx 0$ | mask the grid by $p_t(x) > 10^{-3}$ |
| CFM proof "works" without the tower rule | you silently defined $u_t(x)$ as the thing that makes it work | state the conditional-expectation definition *first*, then prove |
| GitHub renders your MathJax wrong | `$$` blocks inside lists, `\|` in tables | keep display math top-level; use `\Vert`; or ship the PDF |
| Conversion check disagrees at large grid MAE | forgetting the path is variance-preserving in A but OT in B/C | derive conversions per path family; don't mix $\sqrt{1-\bar\alpha_t}$ into OT formulas |

## Stretch

Derive the DDIM sampler (Song et al. 2021) as the deterministic member of the non-Markovian family sharing DDPM's marginals — the missing bridge between §1 and Lesson 15's sampler study — and add `checks/check_ddim_marginals.py` verifying marginal agreement on the 1-D toy.

## References

- LeRobot team. *Robot Learning: A Tutorial*, §4.1 (eqs. 20–49). arXiv:2510.12403.
- Luo. *Understanding Diffusion Models: A Unified Perspective*, 2022. arXiv:2208.11970.
- Lipman et al. *Flow Matching for Generative Modeling*, ICLR 2023. arXiv:2210.02747. — and *Flow Matching Guide and Code*, 2024. arXiv:2412.06264.
- Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models*, NeurIPS 2020. arXiv:2006.11239.
- Permenter, Yuan. *Interpreting and Improving Diffusion Models from an Optimization Perspective*, ICML 2024. arXiv:2306.04848.
- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*, 2024. arXiv:2410.24164.
