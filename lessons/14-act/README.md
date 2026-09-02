# Lesson 14 — ACT: Action Chunking with Transformers

Train the first full policy of the course on a bimanual sim task, predict and then reproduce the ablation that made ACT work — chunking — and understand temporal ensembling from LeRobot's own code rather than from a citation.

| | |
|---|---|
| **Phase** | 4 — Generative imitation policies |
| **Time** | ~1.5 sessions desk time (4–6 h, AI-assisted) + ~4 GPU-hours wall-clock (baseline + three ablation arms, parallelizable) |
| **Cost** | ~$6–10 cloud GPU (one 100k-step baseline + three 50k-step arms on a 4090/A100) |
| **Prerequisites** | 12 (you have seen mode-averaging kill an MSE head), 13 (CVAE ELBO — ACT's loss is Eq. 29 of the tutorial with L1 reconstruction), 01–02 (you can read any LeRobotDataset) |
| **Feeds into** | 15 (Diffusion Policy trains on the same harness), 16 (async inference serves this checkpoint), 19 and H3 (same `eval.py`) |

## Learning objectives

After this lesson you can:

1. **Explain** ACT's CVAE structure: what the style variable `z` absorbs, why the encoder is discarded at inference, and what the β-weighted KL term trades off.
2. **Train** an ACT policy with `lerobot-train` on a rented GPU, predict its loss-curve shape before launch, and publish the checkpoint to the Hub.
3. **Specify** a seeded evaluation harness whose success rates carry binomial confidence intervals, and reuse it for every policy you train after this.
4. **Predict and reproduce** the chunking ablation: success as a function of chunk size $H_a$ and the $H_a{=}1$ collapse.
5. **Explain** temporal ensembling from LeRobot's implementation, annotated against the paper's formula, and quantify its smoothness effect.

## Principles

**The problem.** A single-step BC policy at 50 Hz must make 400 correct decisions per 8-second episode; per-step error compounds (Lesson 12's shift failure) and small pauses in the demos become attractors: the policy sees a state where the demonstrator hesitated and hesitates forever. ACT's answer is to predict a *chunk* of $H_a$ future actions $a_{t:t+H_a}$ from the current observation, cutting the effective decision horizon by $H_a{\times}$ — 400 decisions become 4 at $H_a{=}100$.

**The architecture.** ACT is a conditional VAE over action chunks:

- *Encoder (training only):* a BERT-style transformer takes `[CLS]` + joint positions + the ground-truth action chunk and emits a latent $z \in \mathbb{R}^{32}$. Its only job is to absorb demonstration *style* (fast/slow, wide/tight arcs) so the decoder does not have to average over it.
- *Decoder (the policy):* a transformer encoder–decoder conditions on ResNet18 features of each camera, joint positions, and $z$, and emits all $H_a$ actions in one forward pass.
- *Loss:* $\mathcal{L} = \| a - \hat a \|_1 + \beta \, D_{KL}(q(z|a,o) \,\|\, \mathcal{N}(0,I))$ with $\beta = 10$. L1, not L2 — the paper found it gives more precise actions. At inference $z = 0$ (the prior mean): you ask for the *average style*, not the average action.

**Temporal ensembling.** Executing chunks open-loop causes a discontinuity every $H_a$ steps. Instead, query the policy every step, keep the overlapping predictions for the current timestep, and average them with exponential weights $w_i = e^{-m \cdot i}$ ($i{=}0$ is the *oldest* prediction, $m{=}0.01$), normalized. Older predictions dominate slightly → smooth actions, at the price of reactivity and $H_a{\times}$ more inference calls.

**Paper hyperparameters** (Zhao et al. 2023, appendix — memorize the shape, not the digits): chunk 100, lr 1e-5, batch 8, hidden 512, 8 heads, 4 encoder / 7 decoder layers, feedforward 3200, dropout 0.1, KL weight 10, ResNet18 backbones.

**Carry forward**

- Chunking divides the decision horizon; that is why $H_a{=}1$ collapses and $H_a{=}100$ works on the same data.
- The CVAE encoder exists to absorb style so the decoder need not average it; $z{=}0$ at inference asks for the mean style.
- Ensembling is a convex combination of overlapping predictions with exponential weights; it trades reactivity for smoothness and reappears as Lesson 16's aggregation function.
- Success rates without seeds and intervals are noise. The harness built here is the yardstick for the rest of the course.

| Source | Read for |
|---|---|
| Tutorial §4.2 | how the CVAE objective specializes Lesson 13's ELBO; which ablation rows justify chunking vs ensembling vs the CVAE itself |
| Zhao et al. 2023, §IV + App. B | the exact ensembling formula; the sim TransferCube numbers you are about to reproduce (~1% without chunking vs tens-of-% with) |
| `lerobot/policies/act/configuration_act.py` (your installed version) | every knob LeRobot exposes, and where its defaults deviate from the paper — Exercise 3 |

## Exercise 1 — Know your data [Read]

Tests nothing yet; never train on a dataset you have not inspected. The dataset is `lerobot/aloha_sim_transfer_cube_human`: 50 human-teleoperated episodes of bimanual cube transfer in `gym-aloha`, 400 frames/episode at 50 fps, one 480×640 top camera, 14-D joint state/action.

1. Load it with a chunk-shaped window and confirm the tensor shapes ACT will see:
   ```python
   from lerobot.datasets.lerobot_dataset import LeRobotDataset
   ds = LeRobotDataset("lerobot/aloha_sim_transfer_cube_human",
                       delta_timestamps={"action": [i / 50 for i in range(100)]})
   item = ds[0]
   print({k: v.shape for k, v in item.items() if hasattr(v, "shape")})
   ```
   Expect `action: (100, 14)` and one image tensor `(3, 480, 640)`.
2. Render a 3×3 grid of frames across one episode and eyeball two things: where the grasp happens, and whether the demonstrator pauses (those pauses are exactly what single-step BC trips on).
3. Plot one joint's action trace for 3 episodes overlaid. Note the multi-modality across episodes — this is what `z` will absorb.

**✅ Checkpoint:** shapes match; you can point at the frame index where the cube changes hands in at least one episode.

## Exercise 2 — Train the baseline [Predict → Run]

Tests objective 2. Rent a 4090 or A100 (Vast.ai/RunPod, CUDA ≥ 12 image).

1. **Write first**, in `RESULTS.md`: what `l1_loss` and `kld_loss` should each do over 100k steps and why (which one should stay order-1, and what it means if it collapses to ~0 early).
2. On the box:
   ```bash
   pip install "lerobot[training]" gym-aloha wandb
   hf auth login   # write token
   wandb login
   ```
3. Launch (the LeRobot-documented recipe; 100k steps is the default and takes ~1 h on an A100):
   ```bash
   lerobot-train \
     --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
     --policy.type=act \
     --output_dir=outputs/train/act_transfercube_base \
     --job_name=act_transfercube_base \
     --policy.device=cuda \
     --wandb.enable=true
   ```
4. Reconcile against the curves: `l1_loss` should fall steeply for ~20k steps then grind; `kld_loss` should stay order-1 (β=10 is doing its job). A `kld_loss` collapsing to ~0 early means the encoder is being ignored — note it if you see it.
5. Push the checkpoint:
   ```bash
   hf upload <you>/act_transfercube_base \
     outputs/train/act_transfercube_base/checkpoints/last/pretrained_model
   ```

**✅ Checkpoint:** W&B run with both losses logged; checkpoint on the Hub; loss-curve prediction reconciled.

## Exercise 3 — Running vs knowing [Read]

Tests objective 1. While the baseline trains, diff the paper's hyperparameter table against your installed `configuration_act.py`. At least one structural default differs from the paper (look at the decoder depth and read the comment explaining why). Record every deviation in `RESULTS.md` — this is the difference between *running* a policy and *knowing* it.

**✅ Checkpoint:** the deviation list has ≥ 2 entries, each with the file/field it came from.

## Exercise 4 — The evaluation harness [Build]

Tests objective 3. Spec for `eval.py`, drafted by an AI tool from this contract:

- `evaluate(policy, env_id, seeds) -> EvalReport`. Env via `gym.make("gym_aloha/AlohaTransferCube-v0")`; fixed seed list `range(1000, 1050)` passed to `env.reset(seed=...)`.
- Success = the episode reaches the environment's max reward (4 = cube held by the receiving arm).
- Per-episode record: seed, success, steps-to-success, and the executed action sequence (needed for jerk in Exercise 5).
- `EvalReport` carries success rate with a 95% Wilson interval, the per-episode records, and paths to 3 success + 3 failure videos.
- The check: a smoke run with the stock evaluator must agree with `evaluate(...)` within their intervals:
  ```bash
  MUJOCO_GL=egl lerobot-eval \
    --policy.path=<you>/act_transfercube_base \
    --env.type=aloha --env.task=AlohaTransferCube-v0 \
    --eval.n_episodes=50 --eval.batch_size=10 --policy.device=cuda
  ```
  (On the Mac drop `MUJOCO_GL` and use `--policy.device=mps`; if the CLI surface has drifted, `lerobot-eval --help` is authoritative.)

**This interface is a contract:** Lessons 15, 16, 19 and H3 call `evaluate(policy, env_id, seeds)` unchanged, with only an env adapter per task.

**✅ Checkpoint:** `evaluate` and `lerobot-eval` agree within intervals; baseline success ≥ ~70%. If you are at 40–60%, train 100k more steps before debugging anything else — ACT converges slowly and LeRobot's reference run is a late checkpoint.

## Exercise 5 — The chunking ablation [Predict → Run]

Tests objective 4. The claim under test (paper ablation): without chunking, success collapses to ~1%; with $H_a{=}100$, tens of percent. Reproduce the *shape*, not the digits.

1. **Write first**: predicted success ordering for $H_a \in \{1, 10, 100\}$, the mechanism behind $H_a{=}1$'s collapse (two mechanisms — see Self-check 1), and the predicted direction of jerk with ensembling on.
2. Train three arms at 50k steps each (the ordering is visible well before convergence — say so in `RESULTS.md` when comparing to the baseline):
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
3. Evaluate every arm twice with your harness: open-loop chunk execution vs temporal ensembling (in LeRobot: `n_action_steps=1` + `temporal_ensemble_coeff=0.01` at load time — ensembling is inference-time only, no retraining).
4. From the logged action sequences, compute mean squared jerk (third difference of joint positions, averaged over joints and time) per configuration.
5. Plot: success vs $H_a$ (two lines: with/without ensembling, Wilson error bars) and jerk vs $H_a$. Reconcile.
6. Weight-decay direction: predict in one sentence what $m{=}0.1$ vs $m{=}0.01$ does to jerk and reactivity, then evaluate the $H_a{=}100$ arm at both `temporal_ensemble_coeff` values and reconcile.

**✅ Checkpoint:** $H_a{=}1$ is near-zero; success increases with $H_a$; ensembling visibly reduces jerk. Any surprise (e.g. ensembling *hurting* at $H_a{=}100$) goes in `RESULTS.md` with a hypothesis.

## Exercise 6 — The ensembler [Read the kernel]

Tests objective 5. The ensembler is ~40 lines that everyone uses and nobody reads. Locate LeRobot's implementation:

```bash
grep -rn "class ACTTemporalEnsembler" $(python -c 'import lerobot,os;print(os.path.dirname(lerobot.__file__))')/policies/act
```

Copy the class into `ensemble_annotated.py` and annotate every line against the paper's formula: where the ring buffer of live chunks lives, which line computes $w_i = e^{-m i}$ and in which direction $i$ runs (oldest first — confirm from the code, not the docstring), where normalization happens, and what the output is at $t{=}0$ before any overlap exists. Then write the four properties the code must satisfy (constant-action invariance; convex combination; warm-up equals the first chunk's first action; steady state averages exactly $H_a$ predictions) and check each with a 5-line call on random tensors.

You may reimplement the class yourself instead of copying it (see Going deeper); the annotation is the requirement either way.

**✅ Checkpoint:** the annotated file is committed; the four property checks pass on LeRobot's class.

## Exercise 7 — What ships to H3 [Decide]

Choose the (chunk size, ensembling on/off, $m$) configuration you would deploy on the real arm, citing the rows from Exercise 5 that support it and the row that cuts against it.

**✅ Checkpoint:** the decision paragraph names its supporting rows.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| Hub: `<you>/act_transfercube_base` (+ 3 ablation arms) | loads via `--policy.path`, model card links this lesson |
| `eval.py` | `evaluate(policy, env_id, seeds) -> EvalReport`; seeded, Wilson CIs, video dumps; **reused unchanged by Lessons 15, 16, 19 and H3** (env adapter only) |
| `ensemble_annotated.py` | LeRobot's ensembler annotated line by line; four property checks pass |
| `plots/` | success-vs-$H_a$ (±CI, both execution modes), jerk-vs-$H_a$ |
| `RESULTS.md` | loss-curve, ablation, and $m$-sweep predictions with reconciliations; baseline number with CI; paper-vs-LeRobot deviations; the H3 decision; one surprise + hypothesis |

## Done when

- [ ] Baseline ACT ≥ 70% on TransferCube over 50 seeded episodes, CI reported, cross-checked against `lerobot-eval`.
- [ ] The $H_a{=}1$ arm collapses (≤ 5%), matching the paper's story qualitatively; predictions were written before the runs.
- [ ] Ensembling's smoothness effect is quantified (jerk plot), not asserted.
- [ ] `ensemble_annotated.py` exists and its four property checks pass.
- [ ] A stranger could rerun everything from the README of your lesson directory.

## Self-check

1. Two distinct mechanisms explain why $H_a{=}100$ beats $H_a{=}1$. Name both (hint: one is about compounding, one is about non-Markovian demonstrators).
2. Why is the CVAE encoder thrown away at inference, and what would go wrong if you sampled $z \sim \mathcal{N}(0,I)$ instead of using $z{=}0$?
3. β controls a trade-off. What breaks at β→0? At β→∞?
4. Ensembling averages *actions* across chunks. Under what task condition does that averaging become the very mode-averaging failure Lesson 12 demonstrated?
5. Why does LeRobot deviate from the paper's decoder depth?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `mujoco.FatalError: gladLoadGL` on the cloud box | headless render without EGL | `export MUJOCO_GL=egl` (and `apt install libegl1` on minimal images) |
| Eval success ~0% but training loss fine | env/policy fps or image-size mismatch | assert `env.metadata["render_fps"] == 50` and camera key/shape parity with the dataset features |
| `lerobot-train` rejects a `--policy.*` flag | v0.6 config drift vs docs | `lerobot-train --help`; flags mirror `configuration_act.py` field names |
| Loss NaNs on `mps` in local smoke runs | fp16/float64 edge cases on MPS | smoke-test on `cpu`, train for real on CUDA |
| W&B hangs on Vast.ai | egress blocked on some hosts | `WANDB_MODE=offline`, `wandb sync` after |
| Ablation arms all mediocre and flat | 50k steps read as converged | compare *ordering* only, or extend arms to 100k (2× cost) |
| Ensembler annotation disagrees with the docstring on weight direction | docstrings drift; code does not | trust the indexing in the code; test with a one-hot chunk to see which prediction wins |

## Going deeper

- **Ensembler from scratch.** Implement `TemporalEnsembler(chunk_size, m=0.01)` with `add_chunk(t, actions)` / `get(t)`; the four properties as `pytest` tests; cross-check against LeRobot's class on random tensors to max abs deviation < 1e-6.
- **ACT forward pass from scratch.** Reimplement it in < 500 lines (single file, no LeRobot imports) and assert loss parity with `ACTPolicy` on identical batches to 1e-5 — the strongest evidence you know the architecture, and the warm-up for Lesson 17's π0 attention work.

## References

- Zhao, Kumar, Levine, Finn. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT/ALOHA), RSS 2023. arXiv:2304.13705.
- LeRobot team. *Robot Learning: A Tutorial*, §4.2. arXiv:2510.12403.
- LeRobot ACT docs + `configuration_act.py` / `modeling_act.py` for your installed version (`pip show lerobot`).
