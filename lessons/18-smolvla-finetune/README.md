# Lesson 18 — Fine-Tune SmolVLA

Fine-tuning is where a VLA becomes a policy: adapt `smolvla_base` two ways (full action-expert training vs LoRA), evaluate both on a standard benchmark with `lerobot-eval`, and price what SmolVLA's efficiency tricks actually cost on your hardware.

| | |
|---|---|
| **Phase** | 5 — Generalist policies |
| **Time** | ~3–4 h desk time + 2 × ~4 h GPU wall-clock (parallelizable across two cheap instances) |
| **Cost** | ~$3–8 per fine-tune on a rented A100/L4; Meta-World eval is Mac-local, LIBERO eval adds ~$1–2 of cloud time |
| **Prerequisites** | 14 (`eval.py` contract + you've read training curves), 17 (`NOTE.md`: where SmolVLA sits in the design space), 01 (you inspect a dataset before training on it) |
| **Feeds into** | 19 (this fine-tune is leaderboard entry #1), H4 (same recipe pointed at your H2 dataset), 20 (your eval rollouts become the reward-model calibration set) |

## Learning objectives

After this lesson you can:

1. **Predict** and then **diagnose** a VLA fine-tune from its loss curve: the two-phase shape, and the flat-from-step-0 signature of a key mismatch.
2. **Quantify** the LoRA-vs-full trade: success with CIs, trainable parameters, VRAM, wall-clock.
3. **Explain** every flag in a `lerobot-eval` command and run both suites reproducibly (Meta-World local, LIBERO on Linux).
4. **Measure** the Pareto cost of layer skipping on your own hardware and state which direction the result went.
5. **Decide**, from your numbers, when parameter-efficient adaptation is enough.

## Principles

**SmolVLA** (Shukor et al. 2025, arXiv 2506.01844) is the counter-thesis to scale: a ~450M-param VLA trained on < 30k episodes of community LeRobotDataset data that stays competitive with models 10× larger. Its efficiency tricks are the syllabus:

- **Backbone:** SmolVLM2 — a SigLIP vision encoder feeding a SmolLM2 language model. Images are compressed to **64 visual tokens** per frame via pixel-shuffle.
- **Layer skipping:** the action expert reads VLM features at layer $N = L/2$, not the top; the upper half of the LM never runs. The bet is that mid-depth features carry what control needs. Exercise 6 tests it.
- **Action expert:** a compact (~100M) transformer with **interleaved cross-attention and self-attention** blocks — cross-attend to VLM features, self-attend within the action chunk — trained by flow matching to emit chunks of ~50 actions (Lesson 13's math, Lesson 17's structure).
- **Async inference:** prediction decoupled from execution so the robot never idles between chunks (Lesson 16's stack; SmolVLA ships with it).

**Why fine-tune at all.** `smolvla_base` zero-shot on your task is a lottery ticket; 20k steps on task data is a policy. What pretraining buys is not the task — it is a representation from which the task is cheap to learn (Exercise 5 puts a number on "cheap").

**Full vs parameter-efficient.** "Full" fine-tuning in LeRobot has meant different parameter groups across versions (action expert only vs expert + VLM); you must find out which before comparing. LoRA (Hu et al. 2021) inserts low-rank adapters into linear projections and trains only those — typically 1–5% of parameters — at some cost in the achievable floor. The lesson's comparison holds everything else fixed: same dataset, steps, batch size; the only variable is the adaptation mechanism.

**Evaluation as infrastructure.** LeRobot v0.6 turned benchmarks into one `lerobot-eval` CLI over nine families (Meta-World, LIBERO, LIBERO-plus, RoboTwin 2.0, RoboCasa365, …), each with a docs page, Docker image, and a SmolVLA baseline checkpoint smoke-tested in CI. The per-benchmark docs page names the paired training dataset and the exact eval invocation — it is authoritative over anything printed here.

**Carry forward**

- A fine-tune's loss curve has a shape (steep for ~2k steps, then a grind); a curve without the shape means the inputs are wrong, not the model.
- "Full fine-tune" is a parameter-group list, not a word; read it out of the trainer before you compare anything.
- LoRA's cost is a slightly higher floor; its benefit is 10–100× fewer trainable parameters. Whether the floor matters is an empirical question per task.
- Mid-depth VLM features suffice for control when they wouldn't for VQA; that says the action expert reads geometry, not semantics.
- The benchmark's docs page is the API of record; a README is a snapshot.

| Source | Read for |
|---|---|
| SmolVLA paper §3 | the four efficiency tricks — for each, what it saves and what it risks |
| Tutorial §5.4 | how the tutorial frames SmolVLA vs π0; the interleaved-attention diagram |
| LeRobot docs: `smolvla` page | the canonical fine-tune command (Exercise 2 uses it verbatim) |
| LeRobot docs: `metaworld` + `libero` benchmark pages | paired datasets, eval commands, expected baseline numbers |
| LeRobot v0.6.0 release blog | which benchmarks have SmolVLA baselines — pick your target from this list |

## Exercise 1 — Pick the target and inspect the data [Read]

Tests the principle that a VLA without matching keys and instructions is just an ACT with extra steps.

1. Benchmark pair: **Meta-World** as the local suite (Mac CPU/MPS; 50 tasks in difficulty groups — pick one group or a 5-task subset) and **LIBERO** as the cloud suite (Linux-pinned extra). Read both benchmark docs pages end-to-end.
2. From the docs page, identify the paired training dataset on the Hub. Load it with `LeRobotDataset`; record in `RESULTS.md`: camera keys and resolutions, state/action dims, fps, episode counts per task, and the language-instruction field.

**✅ Checkpoint:** the feature table is in `RESULTS.md` and the instruction field is non-empty.

## Exercise 2 — Zero-shot baseline [Predict → Run]

Tests objective 5's premise: what pretraining alone buys on your task.

1. **Write first:** the success rate you expect from `lerobot/smolvla_base` on your subset, from the ballpark on the benchmark docs page, with a one-line reason.
2. Run `lerobot-eval` with `--policy.path=lerobot/smolvla_base` on your subset (exact flags from the benchmark docs page; on Linux boxes prefix `MUJOCO_GL=egl`). 50+ episodes, fixed seeds. Report success ± Wilson CI (Lesson 14's stats helper).
3. Reconcile.

**✅ Checkpoint:** a zero-shot number with CI that roughly matches the docs' untuned baseline. If it is 0.0 across every task, the env/policy wiring is broken — fix before spending GPU money.

## Exercise 3 — Full fine-tune [Predict → Run]

Tests objective 1: the loss curve as a diagnostic.

1. **Write first:** the shape you expect for the flow-matching loss over 20k steps, and the signature that would mean the dataset's instruction or camera keys don't match the policy.
2. Before launching, run `lerobot-train --help` and record in `RESULTS.md` which parameter groups train by default (action expert only vs expert + VLM) — this defines what "full" means in your comparison and has changed across LeRobot versions.
3. Launch (the LeRobot-documented recipe; 20k steps ≈ 4 h on one A100):
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
4. Watch the curve: steep drop over the first ~2k steps, then a slow grind. A plateau within 500 steps means kill early and recheck Exercise 1.
5. Push the checkpoint (`hf upload`) with a model card naming dataset, steps, and this lesson. Log the trainable-parameter count.

**✅ Checkpoint:** W&B curve shows the two-phase shape; checkpoint on the Hub; trainable-parameter count and default-groups note in `RESULTS.md`.

## Exercise 4 — LoRA fine-tune [Build]

Tests objective 2: the only variable is the adaptation mechanism.

1. Find the current parameter-efficient path: `lerobot-train --help | grep -iE "lora|peft|freeze"`. LeRobot grew PEFT support in the v0.5 line; flag names have drifted, so `--help` is authoritative.
2. If no built-in flags exist in your version, spec a 20-line training shim for an AI tool: wrap the policy's action expert with `peft.LoraConfig(r=16, lora_alpha=32, target_modules=<the expert's linear projections>)`, log trainable vs total parameters, and otherwise call the same trainer. The check: trainable parameters ≤ 10% of Exercise 3's count.
3. Match Exercise 3 exactly: same dataset, 20k steps, batch 64. Log trainable params and peak VRAM for both arms.

**✅ Checkpoint:** LoRA arm trains with ≥ 10× fewer trainable parameters; loss curve shape resembles Exercise 3's, typically converging to a slightly higher floor.

## Exercise 5 — Three arms, one protocol [Predict → Run]

Tests objective 2 and the pretraining question.

1. **Write first:** the LoRA-vs-full success gap you expect (points), and whether you expect either to beat zero-shot with non-overlapping CIs.
2. Evaluate zero-shot (Exercise 2), full-FT, and LoRA with the *same* seeds and episode counts: Meta-World subset locally, LIBERO subset on the cloud box (`MUJOCO_GL=egl`; use the benchmark's Docker image).
3. If LoRA produced adapters rather than merged weights, merge before eval (`peft`'s `merge_and_unload()`) or confirm the eval entrypoint loads adapters — silent zero-shot-with-adapters-ignored is the classic false result.
4. Table: rows = {zero-shot, LoRA, full}; columns = success ± CI per suite, trainable params, GPU-hours, $.
5. Reconcile with your prediction.

**✅ Checkpoint:** both fine-tunes beat zero-shot decisively on the target tasks; the LoRA-vs-full gap is measured. If LoRA lands > ~10 points behind full, check which modules you targeted before concluding LoRA "doesn't work".

## Exercise 6 — Layer-skip Pareto [Predict → Run]

Tests objective 4: SmolVLA's boldest trick, priced on your hardware.

1. Locate the config field controlling which VLM layer feeds the expert (inspect `configuration_smolvla.py` in your installed LeRobot; record the field name in `RESULTS.md`).
2. **Write first:** the latency ratio (half-depth vs full-depth, ms per chunk) and the direction of the success delta you expect, with a reason.
3. Evaluate your full-FT checkpoint at $N{=}L/2$ (default) vs full depth on the local suite: success, latency (ms per chunk, batch 1, 100 warm calls, median + p95; `torch.mps.synchronize()` around timers on Mac), peak memory.
4. Two-panel plot: success-vs-latency, success-vs-memory, both depths marked. Reconcile.

**✅ Checkpoint:** half-depth roughly halves LM compute per chunk; the success delta is measured. Either direction is a result — the paper's claim tested on your task is the deliverable.

## Exercise 7 — When is PEFT enough? [Decide]

From your Exercise 5 table and Exercise 6 plot: state the rule you would apply to the next task (H4's real-robot fine-tune) — LoRA or full, at which depth — and the row that justifies it. Name the condition under which you would flip the decision.

**✅ Checkpoint:** the decision, its supporting row, and its flip condition are in `RESULTS.md`.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| Hub: `<you>/smolvla_ft_full`, `<you>/smolvla_ft_lora` | load via `--policy.path`; model cards state dataset + steps |
| `eval/run_eval.py` (+ JSON outputs) | one command per suite reruns the full 3-arm table; seeds fixed |
| `plots/pareto.png` | success vs latency and vs memory, both depths marked |
| `RESULTS.md` | Exercise 2/3/5/6 predictions with reconciliations; default-trainable-groups note; layer-skip field name; the 3-arm table with CIs; the Exercise 7 decision |

## Done when

- [ ] Full-FT and LoRA both beat zero-shot with non-overlapping CIs on the target subset.
- [ ] LoRA's trainable-parameter fraction and success gap vs full are stated as numbers.
- [ ] LIBERO eval ran on Linux via the benchmark's documented path, rerunnable from one committed script.
- [ ] The layer-skip Pareto plot exists with latency measured on named hardware.
- [ ] Every [Predict → Run] has its prediction written before the run.

## Self-check

1. Why do mid-depth VLM features suffice for control when they wouldn't for VQA? What does that say about what the action expert actually reads?
2. Your LoRA targeted specific modules. Why do attention projections usually matter more than MLPs for adaptation, and what experiment in your setup would test it?
3. Fine-tuning on 50 demos beats zero-shot from < 30k pretraining episodes. Reconcile that with "pretraining matters" — what exactly did pretraining buy?
4. `lerobot-eval` success on LIBERO and on your Meta-World subset can disagree about which arm is better. Name two mechanisms.
5. Which of SmolVLA's four efficiency tricks would you drop first with 10× the compute, and why?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `pip install` of the LIBERO extra fails on the Mac | the extra is Linux-pinned | run LIBERO only on the cloud box; use the benchmark's Docker image |
| `mujoco.FatalError` / black frames on the cloud box | headless GL | `export MUJOCO_GL=egl`; `apt install libegl1` on minimal images |
| OOM at batch 64 on a 4090 | 24 GB < the A100 recipe assumes | `--batch_size=32` + gradient accumulation ×2 (check `--help` for the flag) |
| LoRA arm evaluates exactly at zero-shot level | adapters never loaded/merged at eval | `merge_and_unload()` before upload, or assert adapter load in the eval log |
| Fine-tune loss flat from step 0 | camera-key/instruction mismatch between dataset and policy config | re-run Exercise 1 step 2; keys must match exactly |
| W&B hangs on Vast.ai | blocked egress | `WANDB_MODE=offline` + `wandb sync` |

## Going deeper

- **Knowledge-insulation preview.** Repeat Exercise 3 with the VLM unfrozen vs frozen (if your version's defaults let you toggle it) and probe the backbone before/after on 50 VQA prompts — a small-scale preview of the experiment Capstone option 4 runs properly.
- **LoRA rank sweep.** r ∈ {4, 16, 64} at fixed steps: where does the floor stop moving?

## References

- Shukor et al., *SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics*, 2025. arXiv:2506.01844.
- LeRobot SmolVLA docs (fine-tune recipe) + Meta-World/LIBERO benchmark pages, for your installed version.
- LeRobot v0.6.0 release blog: huggingface.co/blog/lerobot-release-v060.
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021. arXiv:2106.09685.
