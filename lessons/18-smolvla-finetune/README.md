# Lesson 18 — Fine-Tune SmolVLA

Up to this point the course has only run pretrained vision-language-action models. This lesson is where you adapt one. You will take `smolvla_base`, fine-tune it on a benchmark dataset in two ways (training the action expert fully, and training low-rank adapters only), evaluate both against the untuned model under one protocol with `lerobot-eval`, and measure what SmolVLA's efficiency tricks cost and save on your own hardware. The recipe you settle on here is the one Lesson 19 compares against other models and the one H4 points at your own robot data.

| | |
|---|---|
| **Phase** | 5 — Generalist policies |
| **Time** | ~3–4 h desk time + 2 × ~4 h GPU wall-clock (the two fine-tunes can run in parallel on two cheap instances) |
| **Cost** | ~$3–8 per fine-tune on a rented A100/L4; the Meta-World evaluation runs on the Mac, and the LIBERO evaluation adds ~$1–2 of cloud time |
| **Prerequisites** | 14 (the `evaluate()` contract, and experience reading training curves), 17 (`NOTE.md`, which places SmolVLA in the design space), 01 (the habit of inspecting a dataset before training on it) |
| **Feeds into** | 19 (this fine-tune becomes leaderboard entry #1), H4 (the same recipe applied to your H2 dataset), 20 (your evaluation rollouts become the reward-model calibration set) |

## Learning objectives

After this lesson you can:

1. **Predict** the shape of a VLA fine-tune's loss curve and **diagnose** from a curve that lacks it that the dataset's keys or instructions do not match the policy.
2. **Quantify** the trade between LoRA and full fine-tuning in success rate with confidence intervals, trainable parameters, VRAM, and wall-clock.
3. **Explain** every flag in a `lerobot-eval` command and run both benchmark suites reproducibly, Meta-World on the Mac and LIBERO on a Linux machine.
4. **Measure** the cost of SmolVLA's layer skipping on your own hardware and state which direction the result went.
5. **Decide**, from your own numbers, when parameter-efficient adaptation is sufficient.

## Principles

### What SmolVLA is and how it stays small

SmolVLA (Shukor et al. 2025, arXiv 2506.01844) is the counter-argument to scale in the VLA literature: a model of roughly 450M parameters, trained on fewer than 30k episodes of community-contributed LeRobotDataset data, that stays competitive with models ten times its size. It achieves this through four design choices, and each of them is something this lesson either uses directly or measures.

The backbone is SmolVLM2, a SigLIP vision encoder feeding a SmolLM2 language model. Each camera frame is compressed to 64 visual tokens by pixel-shuffling before it enters the language model, which keeps the sequence length, and therefore the attention cost, small.

The action expert does not read the top of the language model. It reads the hidden states at layer $N = L/2$, and the upper half of the language model is never executed. This is called layer skipping. The claim behind it is that the features at mid-depth already carry what control needs, so the remaining layers are paying for language competence that a manipulation policy does not use. Exercise 6 tests that claim on your task.

The action expert itself is a compact transformer of about 100M parameters. Its blocks alternate between cross-attention, in which the action tokens attend to the VLM's features, and self-attention within the action chunk. It is trained by conditional flow matching to emit chunks of about 50 actions, which is the machinery of Lesson 13 arranged as in Lesson 17.

Finally, inference is asynchronous. Action prediction is decoupled from execution so that the robot never idles between chunks. This is the stack of Lesson 16, and SmolVLA ships with it.

### Why fine-tuning is the step that produces a policy

Running `smolvla_base` on a new task without adaptation occasionally works and usually does not, because the pretraining data covers a distribution of tasks, cameras, and embodiments in which yours is at best a near neighbour. Twenty thousand training steps on task data turn the same weights into a policy. What pretraining provides is not the task itself but a representation from which the task is cheap to learn: fewer demonstrations, fewer steps, and better robustness to conditions that the task data did not cover. Exercise 5 puts a number on how cheap.

### Full fine-tuning versus parameter-efficient adaptation

The phrase "full fine-tune" hides a decision that has changed across LeRobot versions: which parameter groups are actually trained. In some versions only the action expert is trained by default, while in others the VLM backbone is trained as well. Before comparing anything you must find out which groups your installed trainer updates, because the answer determines what your comparison measures.

Low-rank adaptation, or LoRA (Hu et al. 2021), takes a different approach. Instead of updating the weight matrices of the linear projections, it inserts a pair of small low-rank matrices alongside each projection and trains only those. The trainable parameter count is typically one to five percent of the full model. The cost is that the achievable loss floor is somewhat higher, because the adapter cannot represent every update that full training could. The comparison in this lesson holds everything else fixed, meaning the same dataset, the same number of steps, and the same batch size, so that the only difference between the two arms is the adaptation mechanism.

### Evaluation as infrastructure

LeRobot v0.6 consolidated benchmark evaluation into a single `lerobot-eval` command that covers nine benchmark families, among them Meta-World, LIBERO, LIBERO-plus, RoboTwin 2.0, and RoboCasa365. Each family has a documentation page, a Docker image, and a SmolVLA baseline checkpoint that is smoke-tested in continuous integration. The documentation page for a benchmark names the training dataset that pairs with it and gives the exact evaluation invocation. Because those pages track the installed version and this README does not, treat them as authoritative wherever the two disagree.

**Carry forward**

- A fine-tune's loss curve has a characteristic shape, a steep drop over roughly the first two thousand steps followed by a slow decline. A curve that is flat from the start indicates that the inputs do not match what the policy expects, not that the model cannot learn.
- "Full fine-tune" names a list of parameter groups, and that list differs between LeRobot versions, so you must read it out of the trainer before you compare adaptation methods.
- LoRA trains ten to a hundred times fewer parameters at the cost of a slightly higher loss floor; whether that floor matters for success rate is an empirical question that has to be answered per task.
- Mid-depth VLM features are sufficient for control even though they would not be sufficient for visual question answering, which tells you that the action expert is reading geometric rather than semantic content.
- The benchmark's documentation page is the interface of record, because it tracks the installed version and a README is a snapshot.

| Source | Read for |
|---|---|
| SmolVLA paper §3 | the four efficiency choices, and for each one what it saves and what it risks |
| Tutorial §5.4 | how the tutorial positions SmolVLA relative to π0; the interleaved-attention diagram |
| LeRobot docs: `smolvla` page | the canonical fine-tune command, which Exercise 3 uses verbatim |
| LeRobot docs: `metaworld` and `libero` benchmark pages | the paired datasets, evaluation commands, and expected baseline numbers |
| LeRobot v0.6.0 release blog | which benchmarks have SmolVLA baselines; pick your target from this list |

## Exercise 1 — Pick the target and inspect the data [Read]

Before training anything, you choose the benchmark pair and confirm that the training dataset contains what the policy expects. A vision-language-action model conditions on camera images under specific keys and on a language instruction; if the dataset's keys or instruction field do not match, the model degenerates into a plain visuomotor policy and the fine-tune fails silently. This exercise establishes that the inputs are right.

1. Choose the benchmark pair. Meta-World is the local suite, because it runs on the Mac's CPU or MPS; it has 50 tasks organised into difficulty groups, and you should pick one group or a subset of five tasks. LIBERO is the cloud suite, because its extra is pinned to Linux. Read both benchmark documentation pages from beginning to end.
2. From the documentation page, identify the paired training dataset on the Hub. Load it with `LeRobotDataset` and record in `RESULTS.md` the camera keys and resolutions, the state and action dimensions, the frame rate, the episode count per task, and the contents of the language-instruction field.

**✅ Checkpoint:** the feature table is in `RESULTS.md`, and the instruction field is non-empty.

## Exercise 2 — Establish the zero-shot baseline [Predict → Run]

Everything in this lesson is measured relative to what the pretrained model can do without adaptation, so that number has to be established first, and it has to be established before any training so that it cannot be influenced by what you later see. This exercise records it, and asks you to predict it first so that you have a stated expectation of what pretraining alone buys on your task.

1. Before running, write in `RESULTS.md` the success rate you expect from `lerobot/smolvla_base` on your subset, taking the untuned-baseline figure on the benchmark documentation page as your reference point, and give a one-line reason for your number.
2. Run `lerobot-eval` with `--policy.path=lerobot/smolvla_base` on your subset, using the exact flags from the benchmark documentation page; on a Linux machine, prefix the command with `MUJOCO_GL=egl`. Use at least 50 episodes with fixed seeds, and report success with a Wilson confidence interval using Lesson 14's statistics helper.
3. Compare the result with your prediction and record the comparison.

**✅ Checkpoint:** a zero-shot success rate with a confidence interval that roughly agrees with the documentation's untuned baseline. A success rate of exactly zero across every task means the environment or policy wiring is broken, and you should fix that before spending any GPU time.

## Exercise 3 — Full fine-tune [Predict → Run]

This exercise produces the first adapted checkpoint and, along the way, teaches you to read a fine-tune's loss curve as a diagnostic instrument. The curve has a shape when the inputs are right and lacks it when they are wrong, so you predict the shape before launching and check the run against it while it trains.

1. Before launching, write down the shape you expect for the flow-matching loss over 20k steps, and the curve shape that would tell you the dataset's instruction or camera keys do not match the policy.
2. Run `lerobot-train --help` and record in `RESULTS.md` which parameter groups train by default, that is, whether only the action expert is trained or the VLM as well. This defines what "full" means in your comparison, and it has changed across LeRobot versions.
3. Launch the fine-tune. This is the recipe from the official SmolVLA documentation, and 20k steps take roughly four hours on one A100:
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
4. Watch the loss curve. It should drop steeply over roughly the first two thousand steps and then decline slowly. If it plateaus within the first 500 steps, stop the run and go back to Exercise 1, because that shape almost always means a key mismatch.
5. Push the checkpoint with `hf upload`, with a model card that names the dataset, the step count, and this lesson. Log the trainable-parameter count.

**✅ Checkpoint:** the W&B curve shows the two-phase shape, the checkpoint is on the Hub, and `RESULTS.md` records both the trainable-parameter count and the default parameter groups.

## Exercise 4 — LoRA fine-tune [Build]

The second arm of the comparison trains low-rank adapters instead of the expert's full weights. For the comparison to mean anything, this arm must match the first in every respect except the adaptation mechanism, so the exercise is as much about controlling the experiment as about running it.

1. Find the current parameter-efficient training path in your installed version with `lerobot-train --help | grep -iE "lora|peft|freeze"`. LeRobot added PEFT support in the v0.5 line, but the flag names have changed between releases, so the `--help` output is authoritative.
2. If your version has no built-in flags, write a specification for a training shim of about twenty lines and have an AI tool draft it: wrap the policy's action expert with `peft.LoraConfig(r=16, lora_alpha=32, target_modules=<the expert's linear projections>)`, log the trainable and total parameter counts, and otherwise call the same trainer as Exercise 3. The check is that the trainable parameter count is at most 10% of Exercise 3's.
3. Match Exercise 3 exactly: the same dataset, 20k steps, batch size 64. Log the trainable parameter count and the peak VRAM for both arms.

**✅ Checkpoint:** the LoRA arm trains with at least ten times fewer trainable parameters, and its loss curve has the same shape as Exercise 3's, typically settling at a slightly higher floor.

## Exercise 5 — Evaluate the three arms under one protocol [Predict → Run]

With three checkpoints in hand (the untuned model, the full fine-tune, and the LoRA fine-tune), you now evaluate them under identical conditions and build the table that answers the lesson's central question. You predict the LoRA-versus-full gap before you see it because doing so forces you to commit to a belief about how much the adapter's restricted capacity costs.

1. Before evaluating, write down the success-rate gap in percentage points you expect between LoRA and full fine-tuning, and whether you expect either fine-tune to beat the zero-shot baseline with non-overlapping confidence intervals.
2. Evaluate all three arms with the same seeds and episode counts: the Meta-World subset locally, and the LIBERO subset on the cloud machine with `MUJOCO_GL=egl` set, using the benchmark's Docker image.
3. If your LoRA run produced adapter weights rather than merged weights, merge them before evaluating with `peft`'s `merge_and_unload()`, or confirm that the evaluation entry point actually loads the adapters. An evaluation that silently ignores the adapters reports the zero-shot number under the LoRA label, and that is the most common false result in this exercise.
4. Build the table: rows for zero-shot, LoRA, and full; columns for success with confidence interval per suite, trainable parameters, GPU-hours, and dollars.
5. Compare the table with your prediction and record the comparison.

**✅ Checkpoint:** both fine-tunes beat the zero-shot baseline decisively on the target tasks, and the LoRA-versus-full gap is a measured number. If LoRA lands more than about ten points behind full, check which modules the adapters targeted before concluding that LoRA is inadequate.

## Exercise 6 — Measure the layer-skip Pareto [Predict → Run]

SmolVLA's decision to discard the upper half of the language model is its most aggressive efficiency choice, and the paper's evidence for it comes from the paper's tasks. This exercise prices the choice on your task and your hardware, in latency, memory, and success, so that you know what it costs you rather than what it cost them.

1. Locate the configuration field that controls which VLM layer feeds the action expert by inspecting `configuration_smolvla.py` in your installed LeRobot, and record the field name in `RESULTS.md`.
2. Before measuring, write down the latency ratio you expect between half depth and full depth in milliseconds per chunk, and the direction you expect the success rate to move, with a reason for each.
3. Evaluate your full-fine-tune checkpoint at $N{=}L/2$, which is the default, and at full depth, on the local suite. Record success, latency (milliseconds per chunk at batch size 1, over 100 warm calls, reporting median and p95, with `torch.mps.synchronize()` around the timers on the Mac), and peak memory.
4. Make a two-panel plot of success against latency and success against memory, with both depths marked. Compare with your prediction.

**✅ Checkpoint:** half depth roughly halves the language model's compute per chunk, and the success delta is measured. Either direction of the success delta is a legitimate result; what the lesson requires is the paper's claim tested on your task.

## Exercise 7 — Decide when parameter-efficient adaptation is enough [Decide]

The tables and plots from Exercises 5 and 6 exist so that a decision can be made from them. State the rule you would apply to the next fine-tune, which is H4's fine-tune on real robot data: LoRA or full, and at which depth. Cite the row that justifies the rule, and name the condition under which you would reverse the decision.

**✅ Checkpoint:** the decision, its supporting row, and its reversal condition are in `RESULTS.md`.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| Hub: `<you>/smolvla_ft_full`, `<you>/smolvla_ft_lora` | load via `--policy.path`; model cards state dataset and step count |
| `eval/run_eval.py` (+ JSON outputs) | one command per suite reruns the full three-arm table; seeds fixed |
| `plots/pareto.png` | success against latency and against memory, both depths marked |
| `RESULTS.md` | predictions and reconciliations for Exercises 2, 3, 5, and 6; the default-parameter-groups note; the layer-skip field name; the three-arm table with confidence intervals; the Exercise 7 decision |

## Done when

- [ ] Full fine-tune and LoRA both beat zero-shot with non-overlapping confidence intervals on the target subset.
- [ ] LoRA's trainable-parameter fraction and its success gap against full fine-tuning are stated as numbers.
- [ ] The LIBERO evaluation ran on Linux through the benchmark's documented path and can be rerun from one committed script.
- [ ] The layer-skip Pareto plot exists, with latency measured on named hardware.
- [ ] Every [Predict → Run] exercise has its prediction written before the run.

## Self-check

1. Why do mid-depth VLM features suffice for control when they would not suffice for visual question answering? What does that say about what the action expert actually reads?
2. Your LoRA targeted specific modules. Why do attention projections usually matter more than MLPs for adaptation, and what experiment in your setup would test that?
3. Fine-tuning on 50 demonstrations beats zero-shot from fewer than 30k pretraining episodes. Reconcile that with the claim that pretraining matters: what exactly did pretraining buy?
4. `lerobot-eval` success on LIBERO and on your Meta-World subset can disagree about which arm is better. Name two mechanisms that would produce the disagreement.
5. Which of SmolVLA's four efficiency choices would you drop first if you had ten times the compute, and why?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `pip install` of the LIBERO extra fails on the Mac | the extra is pinned to Linux | run LIBERO only on the cloud machine; use the benchmark's Docker image |
| `mujoco.FatalError` or black frames on the cloud machine | headless OpenGL | `export MUJOCO_GL=egl`; `apt install libegl1` on minimal images |
| Out of memory at batch size 64 on a 4090 | 24 GB is less than the A100 recipe assumes | `--batch_size=32` with gradient accumulation ×2 (check `--help` for the flag) |
| LoRA arm evaluates at exactly the zero-shot level | adapters never loaded or merged at evaluation | `merge_and_unload()` before upload, or assert adapter loading in the evaluation log |
| Fine-tune loss flat from step 0 | camera-key or instruction mismatch between dataset and policy config | rerun Exercise 1 step 2; the keys must match exactly |
| W&B hangs on Vast.ai | blocked egress | `WANDB_MODE=offline`, then `wandb sync` |

## Going deeper

- **A knowledge-insulation preview.** Repeat Exercise 3 with the VLM unfrozen and frozen, if your version's defaults let you toggle it, and probe the backbone before and after on 50 VQA prompts. This is a small-scale preview of the experiment that Capstone option 4 runs properly.
- **A LoRA rank sweep.** Train at $r \in \{4, 16, 64\}$ for a fixed number of steps and find the rank at which the loss floor stops moving.

## References

- Shukor et al., *SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics*, 2025. arXiv:2506.01844.
- LeRobot SmolVLA documentation (fine-tune recipe) and the Meta-World and LIBERO benchmark pages, for your installed version.
- LeRobot v0.6.0 release blog: huggingface.co/blog/lerobot-release-v060.
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021. arXiv:2106.09685.
