# Lesson 14 — ACT: Action Chunking with Transformers

In this lesson you train the first complete policy of the course, Action Chunking with Transformers (ACT), on a simulated bimanual task, and then reproduce the ablation that made ACT work: predicting a chunk of future actions at each decision rather than a single action. Along the way you build the seeded evaluation harness with confidence intervals that every later lesson reuses, and you read LeRobot's implementation of temporal ensembling against the paper's formula rather than taking either on trust. The lesson matters for what follows because Diffusion Policy (Lesson 15), asynchronous inference (Lesson 16) and the real-robot deployments of H3 all depend on the harness and on the chunking intuition established here.

| | |
|---|---|
| **Phase** | 4 — Generative imitation policies |
| **Time** | ~1.5 sessions desk time (4–6 h, AI-assisted) plus ~4 GPU-hours wall-clock (baseline and three ablation arms, which can run in parallel) |
| **Cost** | ~$6–10 cloud GPU (one 100k-step baseline plus three 50k-step arms on a 4090 or A100) |
| **Prerequisites** | 12 (you have seen mode-averaging defeat an MSE head), 13 (the CVAE ELBO; ACT's loss is the tutorial's Eq. 29 with L1 reconstruction), 01–02 (you can read any LeRobotDataset) |
| **Feeds into** | 15 (Diffusion Policy trains and evaluates on the same harness), 16 (asynchronous inference serves this checkpoint), 19 and H3 (the same `eval.py`) |

## Learning objectives

After this lesson you can:

1. **Explain** ACT's CVAE structure: what the style variable `z` absorbs, why the encoder is discarded at inference, and what the β-weighted KL term trades off.
2. **Train** an ACT policy with `lerobot-train` on a rented GPU, predict the shape of its loss curves before launching, and publish the checkpoint to the Hub.
3. **Specify** a seeded evaluation harness whose success rates carry binomial confidence intervals, and reuse it for every policy you train afterwards.
4. **Predict and reproduce** the chunking ablation: success as a function of chunk size $H_a$, including the collapse at $H_a{=}1$.
5. **Explain** temporal ensembling from LeRobot's implementation, annotated against the paper's formula, and quantify its effect on smoothness.

## Principles

### Why single-step behaviour cloning struggles

A single-step behaviour-cloning policy running at 50 Hz must make 400 correct decisions in an eight-second episode. Lesson 12 showed that per-step errors compound into a drift away from the states the demonstrator visited. A second, subtler problem arises with human demonstrations: demonstrators pause, and a pause is a state in which the recorded action is "do nothing". A single-step policy that reaches such a state has been trained to do nothing there, and having done nothing it is in the same state again, so the pause becomes an attractor from which the policy never leaves.

ACT's answer to both problems is to predict a chunk of $H_a$ future actions, $a_{t:t+H_a}$, from the current observation. With the chunk executed before the next decision, the effective decision horizon shrinks by a factor of $H_a$: at $H_a{=}100$, the 400 decisions of an episode become 4. Drift has fewer opportunities to accumulate, and a demonstrator's pause is absorbed into a chunk that also contains the motion that followed it.

### The architecture

ACT is a conditional variational autoencoder over action chunks, with three parts.

The encoder is used only during training. It is a BERT-style transformer that takes a `[CLS]` token, the current joint positions and the ground-truth action chunk, and emits a latent variable $z \in \mathbb{R}^{32}$. Its purpose is to absorb the style of the demonstration (fast or slow, a wide arc or a tight one), so that the decoder does not have to average over styles that vary from episode to episode.

The decoder is the policy. It is a transformer encoder–decoder that conditions on ResNet18 features of each camera image, on the joint positions and on $z$, and emits all $H_a$ actions in one forward pass.

The loss is $\mathcal{L} = \| a - \hat a \|_1 + \beta \, D_{KL}(q(z|a,o) \,\|\, \mathcal{N}(0,I))$ with $\beta = 10$. The reconstruction term uses the L1 norm rather than L2 because the paper found that it produces more precise actions. At inference the encoder is discarded and $z$ is set to zero, the mean of the prior; the effect is to ask the decoder for the average style rather than the average action, which is the distinction that keeps mode-averaging (Lesson 12) at bay.

### Temporal ensembling

Executing each chunk open-loop produces a discontinuity every $H_a$ steps, at the point where one chunk ends and the next begins. Temporal ensembling removes the discontinuity by querying the policy at every step, retaining all the overlapping predictions that cover the current timestep, and averaging them with exponential weights $w_i = e^{-m \cdot i}$, normalized to sum to one, where $i{=}0$ is the oldest prediction and $m{=}0.01$. Because the weights decay slowly, older predictions carry slightly more weight, and the executed action changes smoothly. The costs are reduced reactivity, since the executed action is partly determined by predictions made up to $H_a$ steps earlier, and $H_a$ times as many inference calls.

### The paper's hyperparameters

The values from Zhao et al. (2023), appendix, are: chunk size 100, learning rate 1e-5, batch size 8, hidden dimension 512, 8 attention heads, 4 encoder and 7 decoder layers, feedforward dimension 3200, dropout 0.1, KL weight 10, and ResNet18 image backbones. It is more useful to remember the shape of this configuration than its exact digits, because Exercise 3 asks you to find where LeRobot's defaults depart from it.

**Carry forward**

- Chunking divides the number of decisions in an episode by $H_a$, which is why $H_a{=}1$ collapses and $H_a{=}100$ succeeds on exactly the same data.
- The CVAE encoder exists so that demonstration style is absorbed into $z$ rather than averaged by the decoder, and setting $z{=}0$ at inference requests the mean style rather than the mean action.
- Temporal ensembling is a convex combination of overlapping predictions with exponential weights; it trades reactivity for smoothness, and the same operation reappears in Lesson 16 as the aggregation function between processes.
- A success rate without a seed list and a confidence interval is not a measurement, and the harness built in this lesson is the yardstick used by every later lesson.

| Source | Read for |
|---|---|
| Tutorial §4.2 | how the CVAE objective specializes Lesson 13's ELBO, and which ablation rows justify chunking, ensembling, and the CVAE itself |
| Zhao et al. 2023, §IV + App. B | the exact ensembling formula, and the simulated TransferCube numbers you are about to reproduce (roughly 1% without chunking versus tens of percent with it) |
| `lerobot/policies/act/configuration_act.py` (your installed version) | every knob LeRobot exposes, and where its defaults deviate from the paper (Exercise 3) |

## Exercise 1 — Inspect the dataset [Read]

Before training on a dataset you should know what it contains, and this exercise establishes that for `lerobot/aloha_sim_transfer_cube_human`: 50 human-teleoperated episodes of bimanual cube transfer in `gym-aloha`, 400 frames per episode at 50 fps, one 480×640 top camera, and 14-dimensional joint state and action vectors. You will confirm the tensor shapes ACT will see and look for the two features of human demonstrations that the Principles section described.

1. Load the dataset with a chunk-shaped window and confirm the tensor shapes:
   ```python
   from lerobot.datasets.lerobot_dataset import LeRobotDataset
   ds = LeRobotDataset("lerobot/aloha_sim_transfer_cube_human",
                       delta_timestamps={"action": [i / 50 for i in range(100)]})
   item = ds[0]
   print({k: v.shape for k, v in item.items() if hasattr(v, "shape")})
   ```
   Expect `action: (100, 14)` and one image tensor of shape `(3, 480, 640)`.
2. Render a 3×3 grid of frames spanning one episode and look for two things: where the grasp happens, and whether the demonstrator pauses. The pauses are the states on which single-step behaviour cloning stalls.
3. Plot one joint's action trace for three episodes overlaid. The variation across episodes is the style that $z$ will absorb.

**✅ Checkpoint:** the shapes match, and you can point to the frame index at which the cube changes hands in at least one episode.

## Exercise 2 — Train the baseline and predict its loss curves [Predict → Run]

This exercise produces the course's first trained checkpoint and tests objective 2. Rent a 4090 or an A100 (Vast.ai or RunPod, with a CUDA ≥ 12 image). Before launching, you predict the shape of the two loss curves, because the KL term's behaviour tells you whether the encoder is doing its job, and a prediction made in advance is the only way to know whether you understood the mechanism or merely recognized the plot afterwards.

1. Write in `RESULTS.md` what `l1_loss` and `kld_loss` should each do over 100k steps and why. Say which of the two should remain of order one, and what it would mean if that one collapsed toward zero early.
2. On the rented machine:
   ```bash
   pip install "lerobot[training]" gym-aloha wandb
   hf auth login   # write token
   wandb login
   ```
3. Launch training with the LeRobot-documented recipe. The default of 100k steps takes about an hour on an A100:
   ```bash
   lerobot-train \
     --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
     --policy.type=act \
     --output_dir=outputs/train/act_transfercube_base \
     --job_name=act_transfercube_base \
     --policy.device=cuda \
     --wandb.enable=true
   ```
4. Reconcile your prediction against the curves. `l1_loss` should fall steeply for about 20k steps and then improve slowly, and `kld_loss` should remain of order one, which indicates that β = 10 is holding the posterior near the prior without collapsing it. If `kld_loss` collapses toward zero early, the encoder is being ignored; note it if you see it.
5. Push the checkpoint:
   ```bash
   hf upload <you>/act_transfercube_base \
     outputs/train/act_transfercube_base/checkpoints/last/pretrained_model
   ```

**✅ Checkpoint:** a W&B run with both losses logged, a checkpoint on the Hub, and the loss-curve prediction reconciled.

## Exercise 3 — Compare the paper's hyperparameters with LeRobot's defaults [Read]

Running a policy and understanding it are different things, and the difference shows up in the configuration. While the baseline trains, compare the paper's hyperparameter table against your installed `configuration_act.py`. At least one structural default differs from the paper; the decoder depth is a good place to start, and the comment in the file explains the reason for the change. Record every deviation you find in `RESULTS.md`, with the file and field it came from.

**✅ Checkpoint:** the deviation list has at least two entries, each attributed to a file and field.

## Exercise 4 — Build the evaluation harness [Build]

Success rates reported without a fixed seed list and a confidence interval cannot be compared, either between policies or between runs of the same policy. This exercise builds the harness that fixes both, which is objective 3, and the harness is reused unchanged by Lessons 15, 16 and 19 and by H3. Write the specification below and have an AI tool draft `eval.py` from it.

- The entry point is `evaluate(policy, env_id, seeds) -> EvalReport`. The environment is created via `gym.make("gym_aloha/AlohaTransferCube-v0")`, and the fixed seed list `range(1000, 1050)` is passed to `env.reset(seed=...)`.
- Success is defined as the episode reaching the environment's maximum reward of 4, which corresponds to the cube being held by the receiving arm.
- Each episode's record contains the seed, the success flag, the number of steps to success, and the executed action sequence, which Exercise 5 needs for the jerk calculation.
- `EvalReport` carries the success rate with a 95% Wilson interval, the per-episode records, and the paths to three success videos and three failure videos.
- The check is agreement with the stock evaluator: a smoke run of `lerobot-eval` must agree with `evaluate(...)` to within their confidence intervals.
  ```bash
  MUJOCO_GL=egl lerobot-eval \
    --policy.path=<you>/act_transfercube_base \
    --env.type=aloha --env.task=AlohaTransferCube-v0 \
    --eval.n_episodes=50 --eval.batch_size=10 --policy.device=cuda
  ```
  On the Mac, drop `MUJOCO_GL` and use `--policy.device=mps`. If the CLI surface has drifted, `lerobot-eval --help` is authoritative.

This interface is a contract. Lessons 15, 16 and 19 and H3 call `evaluate(policy, env_id, seeds)` unchanged, adding only an environment adapter for each task.

**✅ Checkpoint:** `evaluate` and `lerobot-eval` agree within their intervals, and the baseline succeeds on roughly 70% or more of episodes. If you are at 40–60%, train for a further 100k steps before debugging anything else, because ACT converges slowly and LeRobot's reference run is a late checkpoint.

## Exercise 5 — Reproduce the chunking ablation [Predict → Run]

This exercise tests objective 4 by reproducing the paper's central ablation. The claim under test is that without chunking, success collapses to about 1%, whereas with $H_a{=}100$ it reaches tens of percent. You are reproducing the shape of this result rather than its exact digits, and you predict that shape before training so that the mechanisms in the Principles section are tested rather than illustrated.

1. Write in `RESULTS.md` the predicted ordering of success for $H_a \in \{1, 10, 100\}$, the mechanisms behind the collapse at $H_a{=}1$ (there are two; Self-check question 1 names them), and the predicted direction of the change in jerk when ensembling is turned on.
2. Train three arms at 50k steps each. The ordering between arms is visible well before convergence, and you should say so in `RESULTS.md` when you compare their absolute numbers to the 100k-step baseline:
   ```bash
   for H in 1 10 100; do
     lerobot-train \
       --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
       --policy.type=act --policy.chunk_size=$H --policy.n_action_steps=$H \
       --steps=50000 \
       --output_dir=outputs/train/act_H$H --job_name=act_H$H \
       --policy.device=cuda --wandb.enable=true
   done
   ```
3. Evaluate every arm twice with your harness: once with open-loop chunk execution, and once with temporal ensembling. In LeRobot, ensembling is enabled at load time with `n_action_steps=1` and `temporal_ensemble_coeff=0.01`; it is an inference-time change and requires no retraining.
4. From the logged action sequences, compute the mean squared jerk (the third difference of joint positions, averaged over joints and time) for each configuration.
5. Plot success against $H_a$ as two lines, with and without ensembling, with Wilson error bars; and plot jerk against $H_a$. Reconcile both against your predictions.
6. Predict in one sentence what changing $m$ from 0.01 to 0.1 does to jerk and to reactivity, then evaluate the $H_a{=}100$ arm at both values of `temporal_ensemble_coeff` and reconcile.

**✅ Checkpoint:** success at $H_a{=}1$ is near zero, success increases with $H_a$, and ensembling visibly reduces jerk. Any surprise, such as ensembling reducing success at $H_a{=}100$, goes in `RESULTS.md` with a hypothesis.

## Exercise 6 — Annotate LeRobot's temporal ensembler [Read the kernel]

The temporal ensembler is about forty lines of code that every ACT deployment runs and few people read. This exercise, which tests objective 5, has you read LeRobot's implementation against the formula in the Principles section. Locate the class:

```bash
grep -rn "class ACTTemporalEnsembler" $(python -c 'import lerobot,os;print(os.path.dirname(lerobot.__file__))')/policies/act
```

Copy the class into `ensemble_annotated.py` and annotate every line against the paper's formula: where the buffer of live chunks is kept, which line computes $w_i = e^{-m i}$ and in which direction $i$ runs (confirm from the indexing in the code that the oldest prediction has $i{=}0$, rather than trusting the docstring), where the weights are normalized, and what the output is at $t{=}0$ before any overlap exists. Then write down the four properties the code must satisfy, namely that a constant action is returned unchanged, that the output is a convex combination of the inputs, that the warm-up output equals the first chunk's first action, and that in steady state exactly $H_a$ predictions are averaged, and check each with a short call on random tensors.

You may reimplement the class yourself instead of copying it (see Going deeper); the annotation is the requirement in either case.

**✅ Checkpoint:** the annotated file is committed, and the four property checks pass on LeRobot's class.

## Exercise 7 — Choose the configuration for H3 [Decide]

Choose the chunk size, the ensembling setting and the value of $m$ that you would deploy on the real arm in H3. Cite the rows of Exercise 5 that support the choice, and the row that argues against it.

**✅ Checkpoint:** the decision paragraph names its supporting rows and its dissenting row.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| Hub: `<you>/act_transfercube_base` (plus 3 ablation arms) | loads via `--policy.path`; the model card links this lesson |
| `eval.py` | `evaluate(policy, env_id, seeds) -> EvalReport`; seeded, Wilson intervals, video dumps; **reused unchanged by Lessons 15, 16, 19 and H3** (environment adapter only) |
| `ensemble_annotated.py` | LeRobot's ensembler annotated line by line; the four property checks pass |
| `plots/` | success against $H_a$ (with intervals, both execution modes) and jerk against $H_a$ |
| `RESULTS.md` | the loss-curve, ablation and $m$-sweep predictions with their reconciliations; the baseline number with its interval; the paper-versus-LeRobot deviations; the H3 decision; one surprise with a hypothesis |

## Done when

- [ ] Baseline ACT succeeds on at least 70% of 50 seeded episodes, with the interval reported and cross-checked against `lerobot-eval`.
- [ ] The $H_a{=}1$ arm collapses (≤ 5%), matching the paper qualitatively, and the predictions were written before the runs.
- [ ] Ensembling's smoothness effect is quantified with the jerk plot rather than asserted.
- [ ] `ensemble_annotated.py` exists and its four property checks pass.
- [ ] A stranger could rerun everything from the README of your lesson directory.

## Self-check

1. Two distinct mechanisms explain why $H_a{=}100$ beats $H_a{=}1$. Name both; one concerns compounding error and the other concerns non-Markovian demonstrators.
2. Why is the CVAE encoder discarded at inference, and what would go wrong if you sampled $z \sim \mathcal{N}(0,I)$ instead of using $z{=}0$?
3. β controls a trade-off. What breaks as β → 0? As β → ∞?
4. Ensembling averages actions across chunks. Under what task condition does that averaging become the mode-averaging failure demonstrated in Lesson 12?
5. Why does LeRobot deviate from the paper's decoder depth?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `mujoco.FatalError: gladLoadGL` on the cloud box | headless render without EGL | `export MUJOCO_GL=egl` (and `apt install libegl1` on minimal images) |
| Eval success near 0% while training loss looks fine | fps or image-size mismatch between environment and policy | assert `env.metadata["render_fps"] == 50` and check camera key and shape parity with the dataset features |
| `lerobot-train` rejects a `--policy.*` flag | v0.6 config drift relative to the docs | `lerobot-train --help`; flags mirror the field names in `configuration_act.py` |
| Loss NaNs on `mps` in local smoke runs | fp16 and float64 edge cases on MPS | smoke-test on `cpu`; train for real on CUDA |
| W&B hangs on Vast.ai | egress blocked on some hosts | `WANDB_MODE=offline`, then `wandb sync` afterwards |
| Ablation arms are all mediocre and flat | 50k steps mistaken for convergence | compare the ordering only, or extend the arms to 100k steps at twice the cost |
| The ensembler annotation disagrees with the docstring about weight direction | docstrings drift while code does not | trust the indexing in the code; test with a one-hot chunk to see which prediction dominates |

## Going deeper

- **Ensembler from scratch.** Implement `TemporalEnsembler(chunk_size, m=0.01)` with `add_chunk(t, actions)` and `get(t)`; encode the four properties as `pytest` tests; and cross-check against LeRobot's class on random tensors to a maximum absolute deviation below 1e-6.
- **ACT forward pass from scratch.** Reimplement it in under 500 lines in a single file with no LeRobot imports, and assert loss parity with `ACTPolicy` on identical batches to within 1e-5. This is the strongest available evidence that you understand the architecture, and it is good preparation for the π0 attention work in Lesson 17.

## References

- Zhao, Kumar, Levine, Finn. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT/ALOHA), RSS 2023. arXiv:2304.13705.
- LeRobot team. *Robot Learning: A Tutorial*, §4.2. arXiv:2510.12403.
- LeRobot ACT docs, and `configuration_act.py` / `modeling_act.py` for your installed version (`pip show lerobot`).
