# Lesson 18 — Fine-Tune SmolVLA

**Goal:** the tutorial only shows VLA *inference*. Do what it doesn't teach: fine-tune SmolVLA and evaluate it properly on a standard benchmark.

## Read
- Tutorial §5.4 (SmolVLA architecture: layer skipping, 64 visual tokens, interleaved cross-attention).
- SmolVLA paper (Shukor et al. 2025); LeRobot LoRA/PEFT docs (added v0.5.0).

## Build
1. Fine-tune `lerobot/smolvla_base` on a Hub dataset (e.g. a Meta-World or LIBERO single-task subset) on a rented A100/L4 (~4 hrs, ~$3–8). Do it twice: full action-expert fine-tune vs LoRA; compare.
2. Evaluate with `lerobot-eval` (unified benchmark CLI, v0.6.0): Meta-World MT-subset locally on the Mac (CPU MuJoCo), LIBERO subset on a cloud Linux box (`MUJOCO_GL=egl` — the libero extra is Linux-pinned).
3. Ablate one SmolVLA efficiency trick on your hardware: VLM layer-skipping N=L/2 vs full depth (latency/memory/success Pareto).

## Deliverables
- Fine-tuned checkpoints on the Hub; eval reports; LoRA-vs-full and layer-skip Pareto tables.

## Done when
Your fine-tune beats zero-shot `smolvla_base` on the target tasks, and the eval is reproducible from a committed script.
