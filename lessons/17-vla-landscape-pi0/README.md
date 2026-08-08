# Lesson 17 — The VLA Landscape + π0 Dissection

**Goal:** map the generalist-policy landscape (which moved fast after the tutorial), then reimplement π0's two architectural signatures.

## Read
- Tutorial §5 (RT-1 → RT-2 → OpenVLA → π0 → SmolVLA).
- Black et al. 2024 (π0); Shukor et al. 2025 (SmolVLA); Kim et al. 2024 (OpenVLA).
- Post-tutorial: Driess et al. 2025 (knowledge insulation), PI's FAST + real-time chunking posts, GR00T N1.6 writeup, MolmoAct 2 (Ai2, May 2026 — trained on community SO-101 data), π0.7 (Apr 2026).

## Build
1. Comparative note (2–3 pages): discrete action tokens vs continuous generative heads; VLM-backbone choices; how GR00T's reasoning-VLM+action-expert split and world-action models differ from π0's recipe.
2. Reimplement π0's blockwise-causal attention mask (3-block matrix over image+language / proprio / action tokens) as a standalone PyTorch module.
3. Add prefix KV caching across flow-matching denoising steps; unit tests proving cached vs uncached inference match within tolerance; latency-vs-denoising-steps benchmark.

## Deliverables
- The note + tested attention/KV-cache module + benchmark table.

## Done when
Tests pass and the note can answer: "why did everyone converge on flow-matching action experts, and what's already replacing them?"
