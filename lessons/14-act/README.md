# Lesson 14 — ACT: Action Chunking with Transformers

Train the first full policy of the course on a bimanual sim task, then reproduce the ablation that made ACT work — chunking — and prove to yourself *why* it works by implementing and testing the temporal-ensembling machinery by hand.

| | |
|---|---|
| **Phase** | 4 — Generative imitation policies |
| **Time** | ~2 sessions desk time (6–8 h) + ~5 GPU-hours wall-clock (baseline + ablation arms, parallelizable) |
| **Cost** | ~$8–12 cloud GPU (one 100k-step baseline + four 50k-step ablation arms on a 4090/A100) |
| **Prerequisites** | 12 (you've seen mode-averaging kill an MSE head), 13 (CVAE ELBO — ACT's loss is Eq. 29 of the tutorial with L1 reconstruction), 01–02 (you can read any LeRobotDataset) |
| **Feeds into** | 15 (Diffusion Policy trains on the same harness), 16 (async inference serves this checkpoint), H3 (same recipe, real robot) |

## Learning objectives

After this lesson you can:

1. **Explain** ACT's CVAE structure: what the style variable `z` absorbs, why the encoder is discarded at inference, and what the β-weighted KL term trades off.
2. **Train** an ACT policy with `lerobot-train` on a rented GPU, read its loss curves, and publish the checkpoint to the Hub.
3. **Build** a seeded evaluation harness whose success rates carry binomial confidence intervals, and reuse it for every policy you train after this.
4. **Reproduce** the chunking ablation: quantify success as a function of chunk size $H_a$ and show the $H_a{=}1$ collapse.
5. **Implement** overlapping-chunk temporal ensembling from the paper's formula and verify it against LeRobot's implementation.

## Background

**The problem.** A single-step BC policy at 50 Hz must make 400 correct decisions per 8-second episode; per-step error compounds and small pauses in the demos become attractors (the policy sees a state where the demonstrator hesitated and hesitates forever). ACT's answer is to predict a *chunk* of $H_a$ future actions $a_{t:t+H_a}$ from the current observation, cutting the effective decision horizon by $H_a{\times}$ — 400 decisions become 4 at $H_a{=}100$.

**The architecture.** ACT is a conditional VAE over action chunks:

- *Encoder (training only):* a BERT-style transformer takes `[CLS]` + joint positions + the ground-truth action chunk and emits a latent $z \in \mathbb{R}^{32}$. Its only job is to absorb demonstration *style* (fast/slow, wide/tight arcs) so the decoder doesn't have to average over it.
- *Decoder (the policy):* a transformer encoder–decoder conditions on ResNet18 features of each camera, joint positions, and $z$, and emits all $H_a$ actions in one forward pass.
- *Loss:* $\mathcal{L} = \| a - \hat a \|_1 + \beta \, D_{KL}(q(z|a,o) \,\|\, \mathcal{N}(0,I))$ with $\beta = 10$. L1, not L2 — the paper found it gives more precise actions. At inference $z = 0$ (the prior mean): you ask for the *average style*, not the average action.

**Temporal ensembling.** Executing chunks open-loop causes a discontinuity every $H_a$ steps. Instead, query the policy every step, keep the overlapping predictions for the current timestep, and average them with exponential weights $w_i = e^{-m \cdot i}$ ($i{=}0$ is the *oldest* prediction, $m{=}0.01$), normalized. Older predictions dominate slightly → smooth actions, at the price of reactivity and $H_a{\times}$ more inference calls.

**Paper hyperparameters** (Zhao et al. 2023, appendix — memorize the shape, not the digits): chunk 100, lr 1e-5, batch 8, hidden 512, 8 heads, 4 encoder / 7 decoder layers, feedforward 3200, dropout 0.1, KL weight 10, ResNet18 backbones.

| Source | Read for |
|---|---|
| Tutorial §4.2 | how the CVAE objective specializes Lesson 13's ELBO; which ablation rows justify chunking vs ensembling vs the CVAE itself |
| Zhao et al. 2023, §IV + App. B | the exact ensembling formula; the sim TransferCube numbers you're about to reproduce (~1% without chunking vs tens-of-% with) |
| `lerobot/policies/act/configuration_act.py` (your installed version) | every knob LeRobot exposes, and where its defaults deviate from the paper — see Part 1 step 3 |

## Part 0 — Know your data (Mac, ~30 min)

Never train on a dataset you haven't inspected. The dataset is `lerobot/aloha_sim_transfer_cube_human`: 50 human-teleoperated episodes of bimanual cube transfer in `gym-aloha`, 400 frames/episode at 50 fps, one 480×640 top camera, 14-D joint state/action.

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

**✅ Checkpoint:** shapes match the above; you can point at the frame index where the cube changes hands in at least one episode.

## Part 1 — Train the baseline (cloud GPU, ~1.5 h wall-clock)

Produces the course's first checkpoint. Rent a 4090 or A100 (Vast.ai/RunPod, CUDA ≥ 12 image).

1. On the box:
   ```bash
   pip install "lerobot[training]" gym-aloha wandb
   huggingface-cli login   # write token
   wandb login
   ```
2. Launch (this is the LeRobot-documented recipe; 100k steps is the default and takes ~1 h on an A100):
   ```bash
   lerobot-train \
     --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
     --policy.type=act \
     --output_dir=outputs/train/act_transfercube_base \
     --job_name=act_transfercube_base \
     --policy.device=cuda \
     --wandb.enable=true
   ```
3. While it trains, diff the paper's hyperparameter table against your installed `configuration_act.py`. At least one structural default differs from the paper (look at the decoder depth and read the comment explaining why). Record every deviation in `RESULTS.md` — this is the difference between *running* a policy and *knowing* it.
4. Read the curves: `l1_loss` should fall steeply for ~20k steps then grind; `kld_loss` should stay order-1 (β=10 is doing its job). A `kld_loss` collapsing to ~0 early means the encoder is being ignored — note it if you see it.
5. Push the checkpoint:
   ```bash
   huggingface-cli upload <you>/act_transfercube_base \
     outputs/train/act_transfercube_base/checkpoints/last/pretrained_model
   ```

**✅ Checkpoint:** W&B run with both losses logged; checkpoint on the Hub; deviation list has ≥ 2 entries.

## Part 2 — The evaluation harness you'll reuse all course (Mac, ~1 h desk + ~30 min compute)

Success-rate numbers without seeds and intervals are noise. Build the harness once, properly.

1. Quick smoke test with the stock evaluator (50 episodes, batched):
   ```bash
   MUJOCO_GL=egl lerobot-eval \
     --policy.path=<you>/act_transfercube_base \
     --env.type=aloha --env.task=AlohaTransferCube-v0 \
     --eval.n_episodes=50 --eval.batch_size=10 --policy.device=cuda
   ```
   (On the Mac drop `MUJOCO_GL` and use `--policy.device=mps`; if the CLI surface has drifted, `lerobot-eval --help` is authoritative.)
2. Write your own `eval.py` on top of `gym.make("gym_aloha/AlohaTransferCube-v0")`:
   - fixed seed list `range(1000, 1050)` passed to `env.reset(seed=...)`;
   - success = the episode reaches the environment's max reward (4 = cube held by the receiving arm);
   - per-episode record: seed, success, steps-to-success, and the executed action sequence (you need it for jerk in Part 3);
   - report success rate with a 95% Wilson interval; save 3 success and 3 failure videos.
   Interface contract (Lessons 15/16/19 and H3 call this): `evaluate(policy, env_id, seeds) -> EvalReport`.
3. Run it on the baseline.

**✅ Checkpoint:** your `eval.py` and `lerobot-eval` agree within their intervals; baseline success ≥ ~70%. If you're at 40–60%, train 100k more steps before debugging anything else — ACT converges slowly and LeRobot's reference run is a late checkpoint.

## Part 3 — The chunking ablation (cloud, 4 × ~45 min, parallelizable)

The claim under test (paper Table/ablation): without chunking, success collapses to ~1%; with $H_a{=}100$, tens of percent. Reproduce the *shape*, not the exact digits.

1. Train four arms at 50k steps each (the ordering is visible well before full convergence — say so in `RESULTS.md` when comparing absolute numbers to the baseline):
   ```bash
   for H in 1 10 50 100; do
     lerobot-train \
       --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
       --policy.type=act --policy.chunk_size=$H --policy.n_action_steps=$H \
       --steps=50000 \
       --output_dir=outputs/train/act_H$H --job_name=act_H$H \
       --policy.device=cuda --wandb.enable=true
   done
   ```
2. Evaluate every arm twice with your harness: open-loop chunk execution vs temporal ensembling (in LeRobot: `n_action_steps=1` + `temporal_ensemble_coeff=0.01` at load time — ensembling is inference-time only, no retraining).
3. From the logged action sequences, compute mean squared jerk (third difference of joint positions, averaged over joints and time) per configuration.
4. Plot: success vs $H_a$ (two lines: with/without ensembling, Wilson error bars) and jerk vs $H_a$.

**✅ Checkpoint:** $H_a{=}1$ is near-zero; success increases with $H_a$; ensembling visibly reduces jerk. Any surprise (e.g. ensembling *hurting* at $H_a{=}100$) goes in `RESULTS.md` with a hypothesis.

## Part 4 — Temporal ensembling by hand (Mac, ~1 h)

The ensembler is 40 lines that everyone uses and nobody tests. You will.

1. Implement `TemporalEnsembler(chunk_size, m=0.01)` with `add_chunk(t, actions)` and `get(t) -> action`, using a ring buffer of live chunks and $w_i = e^{-m i}$ over however many predictions currently cover $t$.
2. `pytest` properties:
   - constant-action invariance: if every chunk predicts the same constant, output equals it exactly;
   - weights normalize: output is a convex combination (test with one-hot chunks);
   - warm-up: at $t{=}0$ output equals the first chunk's first action;
   - steady state: after $H_a$ steps, exactly $H_a$ predictions are being averaged.
3. Cross-check on random tensors against LeRobot's `ACTTemporalEnsembler` — max abs deviation < 1e-6.

**✅ Checkpoint:** all tests green, cross-check passes.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| Hub: `<you>/act_transfercube_base` (+ 4 ablation arms) | loads via `--policy.path`, model card links this lesson |
| `eval.py` | seeded, Wilson CIs, video dumps; the interface contract above; reused unchanged in Lesson 15 |
| `ensemble.py` + `tests/` | the four properties + LeRobot cross-check, all green in CI |
| `plots/` | success-vs-$H_a$ (±CI, both execution modes), jerk-vs-$H_a$ |
| `RESULTS.md` | baseline number with CI; paper-vs-LeRobot hyperparameter deviations; ablation reading (≤ 10 sentences); one surprise + hypothesis |

## Done when

- [ ] Baseline ACT ≥ 70% on TransferCube over 50 seeded episodes, CI reported.
- [ ] The $H_a{=}1$ arm collapses (≤ 5%), matching the paper's story qualitatively.
- [ ] Ensembling's smoothness effect is quantified (jerk plot), not asserted.
- [ ] `pytest` green; ensembler matches LeRobot to 1e-6.
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

## Stretch

Reimplement ACT's forward pass in < 500 lines (single file, no LeRobot imports) and assert loss parity with `ACTPolicy` on identical batches to 1e-5. This is the strongest possible evidence you actually know the architecture — and the warm-up for Lesson 17's π0 attention work.

## References

- Zhao, Kumar, Levine, Finn. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT/ALOHA), RSS 2023. arXiv:2304.13705.
- LeRobot team. *Robot Learning: A Tutorial*, §4.2. arXiv:2510.12403.
- LeRobot ACT docs + `configuration_act.py` for your installed version (`pip show lerobot`).
