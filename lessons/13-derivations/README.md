# Lesson 13 — Derivation Dossier

**Goal:** the tutorial derives the VAE ELBO and the full DDPM ELBO but defers two key steps. Complete them. This is the math-teeth lesson.

## Read
- Tutorial §4.1 (eqs. 20–49): VAE ELBO, 13-step DDPM ELBO, flow matching.
- Luo 2022, *Understanding Diffusion Models* ("three equivalent interpretations").
- Lipman et al. 2024, *Flow Matching Guide and Code*.

## Write (typed note, LaTeX or Markdown+MathJax)
1. From the DDPM ELBO (tutorial eq. 42) to the simplified ε-prediction loss (eq. 44), every step justified.
2. The flow-matching marginal-vs-conditional objective equivalence (the theorem the tutorial states citing Lipman et al.), with the continuity-equation argument sketched.
3. One page connecting the three views: denoising as manifold projection (Permenter & Yuan), score matching, and FM's vector-field regression — and why π0 chose FM with Beta(1.5,1) timesteps.

## Deliverables
- `derivations.pdf` (or rendered Markdown) cross-referenced to the tutorial's equation numbers.

## Done when
A reader with the tutorial open could follow every deferred step without external references.
