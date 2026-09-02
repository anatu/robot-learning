# Lesson 13 — Derivation Dossier

The tutorial derives the variational autoencoder's evidence lower bound and the multi-step DDPM bound, but it stops short of the steps that make diffusion and flow matching trainable in practice: the closed-form forward posterior, the reduction of the per-step KL terms to a regression on noise, and the equivalence between the marginal and conditional flow-matching objectives. In this lesson you complete those derivations yourself, citing a fixed toolbox of identities at every step, and then verify each closed form with a small numerical check that would fail if your algebra were wrong. The result is a derivation note you can trust, and a working understanding of why the losses in Lessons 12, 14, 15 and 17 have the form they do.

| | |
|---|---|
| **Phase** | 4 — Generative imitation policies |
| **Time** | 2 desk sessions (7–9 h); the derivations are the work, and the check scripts are AI-drafted from your closed forms |
| **Cost** | $0 |
| **Prerequisites** | 12 (you have seen these models work; here you derive them). Graduate probability: Gaussian conjugacy, KL divergences, change of variables |
| **Feeds into** | 14 (ACT's loss is your CVAE ELBO with an L1 reconstruction term), 15 (the sampler study assumes you know why $\epsilon$, score and velocity are interconvertible), 17 (π0's flow head and its timestep schedule) |

## Learning objectives

After this lesson you can:

1. **Derive** the DDPM simplified $\epsilon$-prediction loss from the ELBO (tutorial eq. 42 → 44), with every step justified by a named identity.
2. **Prove** the flow-matching marginal/conditional gradient equivalence, and sketch why the marginal field transports the marginal path.
3. **Convert** between the three parameterizations of a diffusion model (noise $\epsilon_\theta$, score $s_\theta$, velocity $v_\theta$) and verify the conversions numerically.
4. **Defend** a design choice: why π0 trains a flow head with a Beta-skewed timestep distribution rather than a DDPM head with uniformly sampled timesteps.

## Principles

### Why a derivation must be checkable

A derivation note is only as trustworthy as the reader's ability to check it, and two disciplines make that possible. The first is a fixed toolbox: every step in the note must cite one of the allowed identities listed below, so that the phrase "it can be shown" never appears and every move can be traced to a named rule. The second is numerical verification: every closed form you derive is accompanied by a short script that computes the same quantity by brute force in a setting where Gaussians are exact, so that a wrong formula for $\tilde\mu_t$ is caught by a script before it is caught by a reader. The closed form is your output. The script that checks it may be drafted by an AI tool from that closed form, because the check is only as good as the formula you hand it, and a check that re-derives the formula on its own would prove nothing about yours.

### The allowed identities

The complete toolbox is the six identities below. Number them in your note and cite them by number.

- **(I1)** Bayes' rule applied to the forward process: $q(x_{t-1} \mid x_t, x_0) = q(x_t \mid x_{t-1})\, q(x_{t-1} \mid x_0)\, /\, q(x_t \mid x_0)$. This is legal because the forward chain is Markov, so $q(x_t \mid x_{t-1}, x_0) = q(x_t \mid x_{t-1})$.
- **(I2)** Gaussian algebra: products of Gaussian densities combine in the exponent; completing the square identifies the resulting mean and variance; and the KL divergence between two Gaussians has a closed form.
- **(I3)** Reparameterization: $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$, together with its inversion for $x_0$.
- **(I4)** The tower rule and Fubini's theorem, which allow expectations over $t$, $x_1$ and $x_t$ to be nested and exchanged.
- **(I5)** The continuity equation: $\partial_t p_t + \nabla\cdot(p_t u_t) = 0$ characterizes when a velocity field $u_t$ transports the density path $p_t$.
- **(I6)** Expanding $\|a - b\|^2$ and discarding terms that do not depend on $\theta$ under $\nabla_\theta$.

### The notation contract

Diffusion papers use inconsistent notation, and a large fraction of errors in this material are notation collisions: $\beta_t$ confused with $1 - \alpha_t$, or $\bar\alpha_t$ confused with $\alpha_t$. To avoid this, adopt the tutorial's notation ($\alpha_t$, $\bar\alpha_t$, $\beta_t$, and its equation numbers 20–49) throughout the note, and translate the notation of Luo and of Lipman et al. into it whenever you import one of their arguments. The first page of the note carries a symbol table for this reason.

### Three views of one model

For the Gaussian corruption path, the noise, the score and the velocity are linear reparameterizations of one another. A denoiser that predicts the added noise is, up to a scale factor, estimating the score $\nabla \log p_t$; score matching and flow matching are then regressions onto different linear functions of the same conditional expectation. This is why one trained model can serve as any of the three. Separate from the parameterization is the question of which timesteps the training objective emphasizes. The simplified DDPM loss (which sets the per-timestep weight $w_t$ to 1) and π0's skewed timestep sampling both change that emphasis, and Exercise 5 asks you to explain them as two settings of the same lever.

**Carry forward**

- The forward posterior $q(x_{t-1} \mid x_t, x_0)$ is tractable only because the forward chain is Markov, which is what licenses identity (I1); everything after that step is Gaussian algebra via (I2) and (I3).
- The "simplified" DDPM loss is a reweighting across timesteps rather than an algebraic simplification, and it changes the objective being optimized, not merely its form.
- In the conditional flow-matching proof, the cross-term vanishes in the gradient by the tower rule, which is why it is legitimate to regress onto conditional targets that you can sample even though the marginal target cannot be computed.
- The conversions among $\epsilon$, score and velocity hold within a path family, so formulas for the variance-preserving path must not be mixed with formulas for the optimal-transport path.
- Straight optimal-transport paths are the reason few-step sampling works, and Lesson 15 measures this directly.

| Source | Read for |
|---|---|
| Tutorial §4.1, eqs. 20–49 | the exact starting point (eq. 42) and finishing point (eq. 44) of Obligation A, and the flow-matching statement it cites without proof (Obligation B) |
| Luo 2022, *Understanding Diffusion Models* | the three-interpretations scaffolding; read it only after attempting Obligation A yourself, because it is the answer key rather than the exercise |
| Lipman et al. 2023 (FM), Theorems 1–2; *FM Guide and Code* 2024 | the marginal/conditional equivalence in its original notation, and the guide's §4 for the optimal-transport specialization |
| Permenter & Yuan 2024 | denoising as projection, which is the geometric view used in Obligation C |
| π0 (Black et al. 2024), flow-matching appendix | the exact timestep-sampling distribution and the stated rationale, which you quote and then explain |

## Exercise 1 — Obligation A: derive the simplified loss from the ELBO [Derive]

This exercise produces §1 of the note and tests objective 1. Starting from the tutorial's equation 42 and finishing at its equation 44, you fill in every step the tutorial omits, so that the reader can follow the chain from the evidence lower bound to a weighted regression on noise. The required waypoints, in order, are:

1. Split the ELBO into a reconstruction term, a prior term, and a sum of per-step KL terms $D_{KL}\!\left(q(x_{t-1}|x_t,x_0)\,\|\,p_\theta(x_{t-1}|x_t)\right)$. State which terms are dropped or absorbed and why doing so is harmless.
2. Using **(I1)** and **(I2)**, derive the forward posterior in closed form as $q(x_{t-1}|x_t,x_0) = \mathcal{N}\!\left(\tilde\mu_t(x_t,x_0),\, \tilde\beta_t I\right)$, with $\tilde\mu_t$ and $\tilde\beta_t$ written explicitly in terms of $\alpha_t$, $\bar\alpha_t$ and $\beta_t$. Show the completing-the-square step rather than importing the result.
3. Using **(I3)**, eliminate $x_0$ from $\tilde\mu_t$ in favour of $\epsilon$, parameterize the mean of $p_\theta$ in the same way, and reduce the per-step KL (via **(I2)**, with fixed variances) to a weighted $\epsilon$-regression $\mathbb{E}\left[ w_t \|\epsilon - \epsilon_\theta(x_t,t)\|^2 \right]$ with $w_t = \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar\alpha_t)}$.
4. State precisely what "simplified" means. Setting $w_t \to 1$ changes the objective and not merely its form. Explain why the change is justified, following Ho et al.'s reweighting argument that it up-weights the high-noise timesteps where learning is most useful, and note that this is an empirical choice rather than a derived one. Exercise 5 returns to the same design axis.

**✅ Checkpoint:** every numbered step in §1 cites an identity from (I1)–(I6), and $\tilde\mu_t$ and $\tilde\beta_t$ are written out explicitly.

## Exercise 2 — Check the forward posterior numerically [Build]

This exercise verifies the closed form from Exercise 1 in a one-dimensional setting where the exact posterior can be computed by brute force. Your closed form is the specification: give an AI tool the formulas for $\tilde\mu_t$ and $\tilde\beta_t$ exactly as they appear in your note, and have it draft `checks/check_posterior.py` to the following specification.

- The setting is one-dimensional, with $x_0 \sim \mathcal{N}(3, 0.5^2)$, a linear schedule, $T = 50$, and a fixed seed.
- At 5 random $(t, x_t, x_0)$ triples, compute $q(x_{t-1}|x_t,x_0)$ two ways: from your closed-form $(\tilde\mu_t, \tilde\beta_t)$, and by brute-force Bayes' rule on a dense grid.
- Report the maximum absolute error between the two posteriors' moments. The script should run in under 30 s.

**✅ Checkpoint:** the maximum absolute error is below 1e-4. If the error is off by a consistent factor, consult Pitfalls before touching the script, because the formula rather than the code is the likely culprit.

## Exercise 3 — Obligation B: prove the marginal/conditional equivalence [Derive]

This exercise produces §2 of the note and tests objective 2. The tutorial states, citing Lipman et al., that the flow-matching loss and the conditional flow-matching loss have the same gradient; here you prove that statement in the tutorial's notation, and you sketch the separate argument for why the marginal field transports the marginal path.

1. Define the conditional path $p_t(x|x_1)$ and the conditional field $u_t(x|x_1)$ for the optimal-transport construction $x_t = (1-t)x_0 + t x_1$ and $u_t(x|x_1) = x_1 - x_0$. Define the marginal path as $p_t(x) = \int p_t(x|x_1)\, q(x_1)\, dx_1$ and the marginal field as the conditional expectation $u_t(x) = \mathbb{E}[\,u_t(x|x_1) \mid x_t = x\,]$. State this definition before anything else, because the proof depends on it and a proof that introduces the definition afterwards is circular.
2. Prove that $\nabla_\theta \mathcal{L}_{FM} = \nabla_\theta \mathcal{L}_{CFM}$. Expand both squared norms via **(I6)**, eliminate the cross-term with the tower rule **(I4)** by conditioning on $x_t$, and show that the remaining terms do not depend on $\theta$. Flag the one subtle move, which is the exchange of expectations that defines $u_t(x)$, and state the integrability condition it requires.
3. Sketch, in about a page rather than as a full proof, the continuity-equation argument **(I5)**: each conditional field transports its own conditional path, and integrating over $x_1$ while exchanging $\partial_t$ with the integral shows that the marginal field transports the marginal path. Say explicitly what the sketch omits, namely the regularity needed for the exchange.

**✅ Checkpoint:** the cross-term cancellation is written out rather than asserted, and the conditional-expectation definition precedes the proof.

## Exercise 4 — Check the marginal field numerically [Build]

This exercise verifies Exercise 3 on a target distribution for which the marginal field can be computed analytically. A mixture of two Gaussians is closed under the Gaussian corruption path, so the analytic marginal field is available on a grid, and a small network trained on the conditional objective should recover it if the equivalence holds. The specification for `checks/check_marginal_field.py` is:

- The one-dimensional target is $q(x_1) = \frac{1}{2}\mathcal{N}(-2, 0.3^2) + \frac{1}{2}\mathcal{N}(2, 0.3^2)$. Compute the analytic marginal field $u_t(x)$ on a $(t, x)$ grid from the definition in Exercise 3.
- Train a small MLP on the CFM objective and compare it to the analytic field on the grid, masked to the region where $p_t(x) > 10^{-3}$.
- Produce a heatmap comparing the two fields.

**✅ Checkpoint:** the maximum grid error in the masked region is below 0.05.

## Exercise 5 — Obligation C: relate the three parameterizations [Derive]

This exercise produces §3 of the note and tests objectives 3 and 4. It has three parts: the conversion formulas among the three parameterizations, the geometric picture that connects them, and the design-choice question about π0's timestep schedule, which you answer using the machinery of Exercise 1.

1. Derive the conversion formulas for the Gaussian path, all via **(I3)**: the score in terms of the noise, $s_\theta(x_t,t) = -\epsilon_\theta(x_t,t)/\sqrt{1-\bar\alpha_t}$, and the relation between velocity and noise for the optimal-transport flow-matching path. Present the results as a 3×3 table in which, given any one of {$\epsilon$, score, $v$}, the other two are expressed. Derive each conversion within its own path family; the variance-preserving and optimal-transport formulas must not be mixed.
2. Connect the geometry in one paragraph per view, with a single figure for all three: Permenter and Yuan's denoising-as-projection, in which each denoising step is approximately a gradient or projection step toward the data manifold; score matching, in which the denoiser estimates $\nabla \log p_t$; and flow matching, which regresses the transport field directly.
3. **[Read]** π0 trains a flow head and samples timesteps from a Beta-family distribution skewed toward high noise rather than from $t \sim \mathcal{U}[0,1]$. Quote the paper's stated schedule and rationale, and then explain it with the machinery of Exercise 1: uniform $t$ combined with the simplified loss already implies a particular effective weighting across timesteps, and skewing the distribution of $t$ moves the same lever. Contrast what an action head needs, which is few-step sampling at 50 Hz, with what an image generator needs.

**✅ Checkpoint:** the 3×3 table is complete, and the π0 paragraph cites the exact appendix passage it explains.

## Exercise 6 — Check the conversions numerically [Build]

This exercise verifies the table from Exercise 5. If the conversions are correct, a model trained under one parameterization and converted to another should agree with a model trained directly under the second. The specification for `checks/check_conversions.py` is: on the mixture from Exercise 4, train an $\epsilon$-model only; convert it to a score model and to a velocity model using your table; compare each converted model against a directly trained counterpart on the grid; and print the grid mean absolute error for all three pairs.

**✅ Checkpoint:** the converted and directly trained models agree to within training noise, and the three mean absolute errors are reported.

## Exercise 7 — Assemble the derivation note [Write]

The final exercise assembles §1–§3 into a single document and tests all four objectives at once. Produce `derivations.md` (MathJax) or `derivations.pdf` (LaTeX) with a symbol table on the first page, the three sections as specified above, every step citing an identity from (I1)–(I6), cross-references to the tutorial's equation numbers, and every imported argument from Luo or Lipman translated into the tutorial's notation. The standard the note must meet is that a reader with only the tutorial open can follow §1–§3 with no external references and without encountering the word "clearly".

**✅ Checkpoint:** the note answers, in its own words, why dropping $w_t$ is legitimate and where π0 pulls the same lever.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `derivations.md` or `derivations.pdf` | §1–§3 as specified; symbol table; every step cites (I1)–(I6); cross-references to tutorial equation numbers |
| `checks/check_posterior.py` | passes its threshold, runs in under 30 s, seeded |
| `checks/check_marginal_field.py` | passes its threshold; produces the field-comparison heatmap |
| `checks/check_conversions.py` | passes; produces the three-pair MAE table |
| `RESULTS.md` | the three check outputs pasted in, plus five sentences on what surprised you |

## Done when

- [ ] All three checks pass from `python checks/check_*.py`.
- [ ] A reader with only the tutorial open can follow §1–§3 with no external references and no "clearly".
- [ ] The note answers why dropping $w_t$ is legitimate and where π0 pulls the same lever.
- [ ] Every imported argument (Luo, Lipman) is translated into the tutorial's notation.

## Self-check

1. Which single identity makes the forward posterior tractable, and why does it need the Markov property?
2. When $w_t \to 1$, does the argmin of the objective change? Does the training distribution over sub-problems change? (These two questions have different answers.)
3. In the CFM proof, why does the cross-term vanish only in the gradient and not in the loss values themselves?
4. A perfectly trained $\epsilon$-model at $t \to 0$ has $\epsilon_\theta \to$ what, and why does this make $\epsilon$-parameterized samplers unstable near $t = 0$? (Lesson 15 shows this empirically.)
5. Give the one-sentence version of why straight optimal-transport paths permit few-step sampling.

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Posterior check is off by a consistent factor | notation collision among $\beta_t$, $1-\alpha_t$ and $\bar\alpha_t$ | rebuild the symbol table; derive $\bar\alpha_t = \prod \alpha_s$ once, at the top |
| Marginal-field check fails only near $t \approx 1$ | comparing in regions where $p_t \approx 0$ | mask the grid to $p_t(x) > 10^{-3}$ |
| CFM proof "works" without the tower rule | $u_t(x)$ was silently defined as whatever makes the proof go through | state the conditional-expectation definition first, then prove |
| GitHub renders the MathJax wrongly | `$$` blocks inside lists, or `\|` inside tables | keep display math at top level; use `\Vert`; or ship the PDF |
| Conversion check disagrees with large grid MAE | forgetting that the path is variance-preserving in Obligation A but optimal-transport in B and C | derive conversions per path family; do not carry $\sqrt{1-\bar\alpha_t}$ into the OT formulas |
| A check script passes a wrong formula | the AI-drafted check re-derived the closed form instead of using yours | the check must take $\tilde\mu_t$, $\tilde\beta_t$ and the conversion table as literal inputs copied from your note |

## Going deeper

Derive the DDIM sampler (Song et al. 2021) as the deterministic member of the non-Markovian family that shares DDPM's marginals. This is the bridge between §1 of the note and the sampler study of Lesson 15. Add `checks/check_ddim_marginals.py` to verify marginal agreement on the one-dimensional toy problem.

## References

- LeRobot team. *Robot Learning: A Tutorial*, §4.1 (eqs. 20–49). arXiv:2510.12403.
- Luo. *Understanding Diffusion Models: A Unified Perspective*, 2022. arXiv:2208.11970.
- Lipman et al. *Flow Matching for Generative Modeling*, ICLR 2023. arXiv:2210.02747; and *Flow Matching Guide and Code*, 2024. arXiv:2412.06264.
- Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models*, NeurIPS 2020. arXiv:2006.11239.
- Permenter, Yuan. *Interpreting and Improving Diffusion Models from an Optimization Perspective*, ICML 2024. arXiv:2306.04848.
- Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*, 2024. arXiv:2410.24164.
