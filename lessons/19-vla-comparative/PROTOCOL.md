# PROTOCOL — Comparative VLA Lab (Lesson 19)

<!-- Commit this file BEFORE any GPU is rented or any training run starts — the git log must prove
     the protocol predates every training run. See the exercise in README.md that names this file. -->

## Pair (+ optional third) and justification

<!-- TODO: verify each candidate's model card + docs page + documented fine-tune entrypoint + VRAM
     requirement; drop anything unverifiable; pick SmolVLA + X-VLA (third model optional) and justify against Lesson 17's NOTE.md axes. -->

## Budget

<!-- TODO: recommend ~4 A100-hours/model (~8 GPU-h for the pair), one instance type for all models. -->

## Adaptation mechanism per model

<!-- TODO: the documented default per model (LoRA / soft prompts / full or expert fine-tune) — you are
     comparing models-as-shipped, not your tuning skill. -->

## Eval suite, seed list, episode count, success definition

<!-- TODO: episode count >= 50/model. -->

## Metric list (pre-registered)

- [ ] Success ± Wilson CI
- [ ] ms/chunk (batch 1, fixed hardware, median + p95)
- [ ] Peak inference VRAM
- [ ] Trainable / total params
- [ ] Steps completed in budget
- [ ] $ spent
- [ ] Mean squared jerk
