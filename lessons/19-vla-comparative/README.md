# Lesson 19 — Comparative VLA Lab

**Goal:** by mid-2026 the LeRobot ecosystem has many architecturally distinct open VLAs. Fine-tune 2–3 on the *same* dataset and benchmark them head-to-head — far more instructive than studying one model.

## Read
- Model cards/papers for your picks. Good spread: SmolVLA (~450M, SigLIP+SmolLM2, flow matching), EO-1 (Qwen2.5-VL-3B + flow matching), X-VLA (Florence2-based), EVO-1 (0.77B lightweight). GR00T N1.7 if GPU budget allows.

## Build
1. Fix one dataset and one eval protocol (reuse Lesson 18's `lerobot-eval` setup).
2. Fine-tune each model with matched compute budgets (LoRA where supported).
3. Compare: success rate, inference latency, VRAM, action smoothness, training cost.
4. Write the result up as a mini-paper (2–4 pages): what architectural choices actually mattered at this scale.

## Deliverables
- Checkpoints + eval harness + the comparison report with a leaderboard table.

## Done when
Three models, one protocol, one honest table — including at least one result that surprised you.
