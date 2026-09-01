# Lesson 18 — Fine-Tune SmolVLA

The tutorial only ever *runs* a VLA; here you adapt one. Fine-tune `smolvla_base` twice — full action-expert training vs LoRA — evaluate both on a standard benchmark with `lerobot-eval`, and measure what SmolVLA's efficiency tricks actually cost.

| | |
|---|---|
| **Phase** | 5 — Generalist policies |
| **Time** | ~1 session prep/eval desk time + 2 × ~4 h GPU wall-clock (parallelizable across two cheap instances) |
| **Cost** | ~$3–8 per fine-tune on a rented A100/L4; Meta-World eval is Mac-local, LIBERO eval adds ~$1–2 of cloud time |
| **Prerequisites** | 14 (eval harness + you've read training curves before), 17 (you know where SmolVLA sits in the design space), 01 (you can inspect any dataset before training on it) |
| **Feeds into** | 19 (this fine-tune is leaderboard entry #1), H4 (same recipe pointed at your own H2 dataset), 20 (your eval rollouts become the reward-model calibration set) |

## Learning objectives

After this lesson you can:

1. **Fine-tune** a pretrained VLA on a benchmark dataset with `lerobot-train` and diagnose the run from its loss curves.
2. **Quantify** the LoRA-vs-full trade: success rate, trainable parameters, VRAM, and wall-clock, with confidence intervals.
3. **Run** `lerobot-eval` benchmarks reproducibly — Meta-World locally, LIBERO on a Linux GPU box — and explain every flag in your eval command.
4. **Measure** the Pareto cost of SmolVLA's layer-skipping (half-depth VLM) on your own hardware: latency, memory, success.
5. **Defend** a claim about when parameter-efficient adaptation is enough, grounded in your own numbers.

## Background

**SmolVLA** (Shukor et al. 2025, arXiv 2506.01844) is the counter-thesis to scale: a ~450M-param VLA trained on < 30k episodes of community LeRobotDataset data that stays competitive with models 10× larger. Its efficiency tricks are the syllabus of this lesson:

- **Backbone:** SmolVLM2 — a SigLIP vision encoder feeding a SmolLM2 language model. Images are compressed to **64 visual tokens** per frame via pixel-shuffle.
- **Layer skipping:** the action expert reads VLM features at layer $N = L/2$, not the top — the upper half of the LM never runs. Empirically the mid-depth features carry what control needs; you'll test that claim in Part 4.
- **Action expert:** a compact (~100M) transformer with **interleaved cross-attention and self-attention** blocks — cross-attend to VLM features, self-attend within the action chunk — trained by flow matching to emit chunks of ~50 actions (Lesson 13's math, Lesson 17's structure).
- **Async inference:** action prediction is decoupled from execution so the robot never idles between chunks (Lesson 16's stack; SmolVLA ships with it).

Fine-tuning is where VLAs actually become useful — `smolvla_base` zero-shot on your task is a lottery ticket; 20k steps on task data is a policy. LeRobot v0.6 turned evaluation into infrastructure: nine benchmark families (Meta-World, LIBERO, LIBERO-plus, RoboTwin 2.0, RoboCasa365, …) behind one `lerobot-eval` CLI, each with a docs page, Docker image, and a SmolVLA baseline checkpoint smoke-tested in CI. You will lean on that: the per-benchmark docs page names the paired training dataset and the exact eval invocation — treat it as authoritative over anything printed here.

| Source | Read for |
|---|---|
| SmolVLA paper §3 | the four efficiency tricks above — for each, what it saves and what it risks |
| Tutorial §5.4 | how the tutorial frames SmolVLA vs π0; the interleaved-attention diagram |
| LeRobot docs: `smolvla` page | the canonical fine-tune command (Part 1 uses it verbatim) |
| LeRobot docs: `metaworld` + `libero` benchmark pages | paired datasets, eval commands, expected baseline numbers to sanity-check against |
| LeRobot v0.6.0 release blog | which benchmarks have SmolVLA baselines — pick your target from this list |

## Part 0 — Pick the target and stare at the data (~1 h, Mac)

1. Choose the benchmark pair: **Meta-World** as the local suite (runs on Mac CPU/MPS; 50 tasks in difficulty groups — pick one group or a 5-task subset) and **LIBERO** as the cloud suite (Linux-pinned extra). Read both benchmark docs pages end-to-end first.
2. From the benchmark docs page, identify the paired training dataset on the Hub. Load it with `LeRobotDataset`, confirm: camera keys and resolutions, state/action dims, fps, episode counts per task, and the language-instruction field (a VLA without instructions is just an ACT with extra steps).
3. Record the zero-shot baseline now, before any training: run `lerobot-eval` with `--policy.path=lerobot/smolvla_base` on your chosen subset (exact flags from the benchmark docs page; on Linux boxes prefix `MUJOCO_GL=egl`). 50+ episodes, fixed seeds.

**✅ Checkpoint:** you have a zero-shot success number with a Wilson CI, and it roughly matches the ballpark the benchmark docs report for the untuned baseline. If it's 0.0 across every task, your env/policy wiring is broken — fix before spending GPU money.

## Part 1 — Full fine-tune (~4 h GPU, ~$3–8)

The LeRobot-documented recipe, verbatim shape (this is the command from the official SmolVLA docs — 20k steps ≈ 4 h on one A100):

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=<benchmark-paired-dataset> \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/smolvla_ft_full \
  --job_name=smolvla_ft_full \
  --policy.device=cuda \
  --wandb.enable=true
```

1. Before launching, run `lerobot-train --help` and record in `RESULTS.md` which parameter groups train by default (action expert only vs expert+VLM) — this determines what "full" means in your comparison and it has changed across LeRobot versions.
2. Watch the flow-matching loss: steep drop over the first ~2k steps, then a slow grind. A loss that plateaus within 500 steps usually means the dataset's instruction or camera keys don't match what the policy expects — kill early, check Part 0 again.
3. Push the checkpoint to the Hub (`hf upload`), with a model card naming dataset, steps, and this lesson.

**✅ Checkpoint:** W&B curve shows the two-phase shape; checkpoint on the Hub; trainable-parameter count logged.

## Part 2 — LoRA fine-tune (~2–4 h GPU)

1. Find the current parameter-efficient path: `lerobot-train --help | grep -iE "lora|peft|freeze"`. LeRobot grew PEFT support in the v0.5 line; flag names have drifted between releases, so the `--help` output is authoritative. If no built-in flags exist in your version, wrap the policy's action expert with `peft.LoraConfig(r=16, lora_alpha=32, target_modules=<the expert's linear projections>)` in a 20-line training shim.
2. Match Part 1 exactly: same dataset, same 20k steps, same batch size. The *only* variable is the adaptation mechanism.
3. Log trainable params (expect roughly 1–5% of the full run) and peak VRAM for both arms.

**✅ Checkpoint:** LoRA arm trains with ≥ 10× fewer trainable parameters; loss curve shape is similar to Part 1's, typically converging to a slightly higher floor.

## Part 3 — Evaluate all three arms (~1–2 h)

1. Evaluate zero-shot (done in Part 0), full-FT, and LoRA with the *same* seeds and episode counts: Meta-World subset locally, LIBERO subset on the cloud box (`MUJOCO_GL=egl`; the benchmark's Docker image sidesteps the dependency swamp — use it).
2. If your LoRA path produced adapters rather than merged weights, merge before eval (`peft`'s `merge_and_unload()`), or confirm your eval entrypoint loads adapters — silent zero-shot-with-adapters-ignored is the classic false result here.
3. Table: rows = {zero-shot, LoRA, full}, columns = success ± CI per suite, trainable params, GPU-hours, $.

**✅ Checkpoint:** both fine-tunes beat zero-shot decisively on the target tasks; the LoRA-vs-full gap is measured, not assumed. If LoRA lands more than ~10 points behind full, check which modules you targeted before concluding LoRA "doesn't work".

## Part 4 — The layer-skip Pareto (~1–2 h, Mac or GPU)

SmolVLA's boldest trick is discarding half the VLM. Price it on your hardware.

1. Locate the config knob controlling which VLM layer feeds the expert (inspect `configuration_smolvla.py` in your installed LeRobot — record the field name in `RESULTS.md`).
2. Evaluate your full-FT checkpoint at $N{=}L/2$ (default) vs full depth: success on the local suite, latency (ms per chunk, batch 1, 100 warm calls, median + p95), peak memory. On Mac, `torch.mps.synchronize()` around timers.
3. Plot the three-metric Pareto (a 2-panel plot: success-vs-latency, success-vs-memory).

**✅ Checkpoint:** half-depth roughly halves LM compute per chunk; success delta is measured. Either direction of result is publishable in `RESULTS.md` — the paper's claim tested on your task is the deliverable.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| Hub: `<you>/smolvla_ft_full`, `<you>/smolvla_ft_lora` | load via `--policy.path`; model cards state dataset + steps |
| `eval/` scripts + JSON outputs | one command per suite reruns the full table; seeds fixed |
| `RESULTS.md` | the 3-arm table with CIs; default-trainable-groups note; layer-skip field name; Pareto reading in ≤ 8 sentences |
| `plots/pareto.png` | success vs latency and vs memory, both depths marked |

## Done when

- [ ] Full-FT and LoRA both beat zero-shot with non-overlapping CIs on the target subset.
- [ ] LoRA's trainable-parameter fraction and success gap vs full are stated as numbers.
- [ ] LIBERO eval ran on Linux via the benchmark's documented path — and you can rerun it from one committed script.
- [ ] The layer-skip Pareto plot exists with latency measured on named hardware.

## Self-check

1. Why do mid-depth VLM features suffice for control when they wouldn't for VQA? What does that say about what the action expert actually reads?
2. Your LoRA targeted specific modules. Why do attention projections usually matter more than MLPs for adaptation, and what experiment in your setup would test it?
3. Fine-tuning on 50 demos beats zero-shot from < 30k pretraining episodes. Reconcile that with "pretraining matters" — what exactly did pretraining buy?
4. `lerobot-eval` success on LIBERO and success on your Meta-World subset can disagree about which arm is better. Name two mechanisms that produce that disagreement.
5. Which of SmolVLA's four efficiency tricks would you drop first if you had 10× the compute, and why?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `pip install` of the LIBERO extra fails on the Mac | the extra is Linux-pinned | run LIBERO only on the cloud box; use the benchmark's Docker image |
| `mujoco.FatalError` / black frames on the cloud box | headless GL | `export MUJOCO_GL=egl`; `apt install libegl1` on minimal images |
| OOM at batch 64 on a 4090 | 24 GB < the A100 recipe assumes | `--batch_size=32` + gradient accumulation ×2 (check `--help` for the flag) |
| LoRA arm evaluates exactly at zero-shot level | adapters never loaded/merged at eval | `merge_and_unload()` before upload, or assert adapter load in the eval log |
| Fine-tune loss flat from step 0 | camera-key/instruction mismatch between dataset and policy config | re-run Part 0 step 2; keys must match exactly |
| W&B hangs on Vast.ai | blocked egress | `WANDB_MODE=offline` + `wandb sync` |

## Stretch

Repeat Part 1 with the VLM unfrozen vs frozen (if your version's defaults let you toggle it) and probe the backbone before/after on 50 VQA prompts — a small-scale preview of the knowledge-insulation experiment that Capstone option 4 runs properly.

## References

- Shukor et al., *SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics*, 2025. arXiv:2506.01844.
- LeRobot SmolVLA docs (fine-tune recipe) + Meta-World/LIBERO benchmark pages, for your installed version.
- LeRobot v0.6.0 release blog: huggingface.co/blog/lerobot-release-v060.
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021. arXiv:2106.09685.
