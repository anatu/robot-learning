# H4 — VLAs on Your Arm

**Goal:** generalist policies meet your $400 robot: zero-shot foundation-model rollout, then your own fine-tune, then a correction loop.

## Tasks
1. **Zero-shot MolmoAct 2** (Ai2, trained on community SO-101 data): run via `lerobot-rollout` with built-in calibration correction. Evaluate on your H2 task and one novel task, no training.
2. Fine-tune SmolVLA (LoRA, Lesson 18's recipe) on your H2 dataset; deploy async; head-to-head vs H3's ACT/DP on the same 20-trial protocol.
3. Language generalization probe: paraphrased instructions, new object colors, distractors — quantify what the VLM backbone buys over ACT.
4. **Improvement loop:** collect corrections with `lerobot-rollout`'s DAgger-style human-in-the-loop mode, retrain, re-evaluate. One full iteration minimum.

## Deliverables
- Rollout videos, the four-way comparison table (MolmoAct2-zero-shot / SmolVLA-ft / ACT / DP), and before/after numbers for the correction loop.

## Done when
The comparison table exists with honest error bars, and the DAgger iteration measurably improved something.
