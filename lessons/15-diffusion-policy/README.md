# Lesson 15 — Diffusion Policy + Sampler Study

Train Diffusion Policy on PushT, then answer the question that decides deployability: how few denoising steps can you pay at 50 Hz, and why do flow paths pay fewer — measured on the same weights, one harness, one Pareto plot.

| | |
|---|---|
| **Phase** | 4 — Generative imitation policies |
| **Time** | 4–6 h desk + ~1 GPU-h cloud for the baseline (or an overnight `mps` run); the 2-D study runs on the Mac in minutes |
| **Cost** | ~$2–3 cloud; $0 if trained locally overnight |
| **Prerequisites** | 13 (DDPM loss, DDIM as shared-marginal sampler, ε/score/velocity conversions), 14 (`evaluate(policy, env_id, seeds) -> EvalReport` — reused unchanged), 12 (`heads/{ddpm,cfm}.py` with `sample(s, n)` — reused in Exercise 6) |
| **Feeds into** | 16 (this checkpoint gets served async; DP's latency is why async exists), H3 (same recipe on the real arm) |

## Learning objectives

After this lesson you can:

1. **Explain** Diffusion Policy's receding-horizon design — predict $T_p$, execute $T_a$, condition on $T_o$ — and what each horizon buys.
2. **Predict** which sampler configurations retain success and which degrade, before running them, and reconcile against measured success and latency.
3. **Quantify** the sampler trade honestly: success ± CI vs synchronized wall-clock per decision and per control step.
4. **Demonstrate** on real robot data why flow paths tolerate few-step sampling: path straightness and few-step sample quality, side by side.
5. **Decide** a sampler/step-count for a deployment constraint and defend it with your own numbers.

## Principles

**The architecture.** DP is a conditional generative model over *action sequences*: at each control step, sample a chunk $A_t \in \mathbb{R}^{T_p \times d_a}$ conditioned on the last $T_o$ observations, execute the first $T_a$ actions, repeat. Paper settings: $T_o = 2$, $T_p = 16$, $T_a = 8$. Conditioning enters a 1-D temporal CNN U-Net through FiLM layers (a transformer head is the paper's alternative; the U-Net is the robust default). Training is exactly Lesson 13's simplified loss with $K = 100$ diffusion steps; the paper leans on an EMA copy of the weights for stability. Receding horizon is the mirror image of Lesson 14's ensembling: $T_a < T_p$ keeps the policy reactive; $T_a = T_p$ is open-loop; $T_a = 1$ throws away the chunk's decision-horizon benefit.

**Why samplers matter here and nowhere else so far.** ACT pays one forward pass per chunk. DP pays one per *denoising step* per chunk — 100 ancestral DDPM steps can mean tens-to-hundreds of ms per decision. Two escapes, in increasing radicalism:

- **DDIM** (Song et al. 2021): a non-Markovian family sharing DDPM's marginals; its deterministic member permits *strided* sampling. Same trained ε-network, sub-schedule of {10, 5, 1} steps, no retraining. Legal because DDIM only needs the marginals $q(x_t|x_0)$ the ε-network was trained under.
- **Flow matching**: retrain the head to regress a velocity field along the straight OT path. Straight paths ⇒ Euler integration with very few steps is nearly exact (Lesson 13 proved the machinery; here you measure it).

The ε-parameterization has a known weak spot at few steps near $t \to 0$ (Lesson 13 self-check #4): a perfect ε-model's output stops carrying information about $x_0$ there, so large strides at the low-noise end overshoot. Expect DDIM-1 to saturate the action bounds.

**Task facts** (verify against the installed package; these are stable): env `gym_pusht/PushT-v0` — push a T-block onto a target zone; success = block ≥ 95% inside the goal zone; reward = coverage; action = target agent position in $[0,512]^2$; obs types include `pixels_agent_pos` (96×96 RGB + 2-D agent position) and `environment_state_agent_pos` (16-D keypoints + agent pos). Dataset `lerobot/pusht`: 206 episodes, 25,650 frames, 10 fps, 96×96 images. PushT is deliberately multimodal — the block can be approached from either side — which is why it is *the* DP benchmark.

**Carry forward**

- $T_o / T_p / T_a$: condition on a little, predict a chunk, execute a fraction. Each horizon is a knob with a failure at both ends.
- DDIM reuses DDPM weights because it shares the marginals; it changes the sampler, not the objective. Flow matching changes the objective.
- Latency is only real when the device is synchronized before both timestamps.
- Few-step tolerance is a property of path straightness, and straightness is measurable.

| Source | Read for |
|---|---|
| Chi et al. 2023/2024 (Diffusion Policy) §3, §4.4 | receding-horizon rationale; why position control + chunking; the ablations behind $T_o{=}2, T_p{=}16, T_a{=}8$; where EMA matters |
| Tutorial §4.3 | how the tutorial frames DP after ACT: what the CVAE couldn't do that diffusion can |
| Song et al. 2021 (DDIM) §4 | the sub-sequence sampling trick you use verbatim; where determinism comes from |
| `configuration_diffusion.py` (installed) | every knob for Exercise 2; the fields you sweep in Exercise 4: `noise_scheduler_type`, `num_train_timesteps`, `num_inference_steps` |

## Exercise 1 — Know the data [Read]

Tests objective 1: the multimodality DP must not average.

1. Load `lerobot/pusht`, print feature shapes; confirm 206 episodes / 10 fps / `action: (2,)` in $[0,512]$.
2. Plot 20 episode trajectories in the 2-D action plane, colored by episode. Find a pair of episodes that solve a similar block pose from opposite sides.
3. Load with DP-shaped windows: `delta_timestamps` giving 2 obs steps back and 16 action steps forward; confirm shapes `(2, ...)` / `(16, 2)`.

**✅ Checkpoint:** shapes as stated; you can point at two opposite-strategy episodes.

## Exercise 2 — Train the baseline [Read]

Tests objective 1: knowing the config you ran, not just running it.

1. Diff the paper's horizons against your installed defaults (`configuration_diffusion.py`). Recent LeRobot mains default to a *longer* horizon (e.g. `horizon=64`, `n_action_steps=32`) than the paper's 16/8. Record the defaults in `RESULTS.md`, then pin the paper values:
   ```bash
   lerobot-train \
     --dataset.repo_id=lerobot/pusht \
     --policy.type=diffusion \
     --policy.horizon=16 --policy.n_action_steps=8 --policy.n_obs_steps=2 \
     --output_dir=outputs/train/dp_pusht_base --job_name=dp_pusht_base \
     --policy.device=cuda --wandb.enable=true
   ```
   (~100k steps: ~1 h on an A100, overnight on `mps`.)
2. Write down, before looking: what the ε-MSE curve can tell you and what it cannot. Then read it: falls fast for ~10k steps, then improves slowly; there is no success signal in the loss — coverage-at-eval is the only truth.
3. Push the checkpoint to `<you>/dp_pusht_base`.

**✅ Checkpoint:** W&B run live; checkpoint on Hub; deviation list recorded (≥ 2 entries expected).

## Exercise 3 — Evaluate with the Lesson 14 harness [Predict → Run]

Tests objective 3's foundation: a number with an interval, cross-checked.

1. **Write first:** the strict-success band and the mean max coverage you expect, given the reference band in step 4.
2. Point Lesson 14's `evaluate()` at PushT: 50 seeds (`range(2000, 2050)`), success = env's solved flag (≥ 95% coverage), also record final coverage and steps-to-success; dump 3 success + 3 failure videos.
3. Cross-check one number against the stock evaluator:
   ```bash
   MUJOCO_GL=egl lerobot-eval --policy.path=<you>/dp_pusht_base \
     --env.type=pusht --eval.n_episodes=50 --eval.batch_size=10 --policy.device=cuda
   ```
4. Report success ± Wilson CI and mean final coverage. Reference reproductions land in the ~60–70% strict-success band with high (> 0.90) mean max coverage — treat < 50% as a bug (see Pitfalls), not a result.

**✅ Checkpoint:** your harness and `lerobot-eval` agree within CIs; numbers in the expected band.

## Exercise 4 — Samplers on the same weights [Predict → Run]

Tests objective 2: DDIM changes the sampler, not the model.

1. **Write first**, per config in {DDPM-100, DDIM-10, DDIM-5, DDIM-1}: predicted success relative to the baseline (within CI / degraded / collapsed) and the mechanism. Name the config you expect to saturate the action bounds and why (Principles: ε-parameterization near $t \to 0$).
2. On the *same* `dp_pusht_base` weights, evaluate DDPM at $K{=}100$ (the training config), then DDIM at `num_inference_steps` ∈ {10, 5, 1} — `--policy.noise_scheduler_type=DDIM` plus `--policy.num_inference_steps=<n>` at load/eval time (flags mirror `configuration_diffusion.py` field names; `lerobot-eval --help` is authoritative). DDIM reuses the ε-network on a strided sub-schedule — the Lesson 13 stretch made real. 50 seeds each, the harness from Exercise 3.
3. Reconcile per config.

**✅ Checkpoint (expected shape):** DDIM-10 retains DDPM-100 success within CI; DDIM-1 degrades visibly. Any deviation → a hypothesis in `RESULTS.md`.

## Exercise 5 — Latency, measured honestly [Diagnose]

Tests objective 3: the measurement mistake that makes latency tables lie.

1. **Predict:** time 100 policy calls *without* synchronizing the device, then *with* `torch.cuda.synchronize()` / `torch.mps.synchronize()` before each timestamp. Write down the direction and rough magnitude of the discrepancy, and which measurement is wrong.
2. Run both on `mps` (deployment reality for H3) and on the cloud GPU: median over 100 calls after 10 warm-up calls. Explain the mechanism (asynchronous dispatch returns before compute finishes).
3. With the synchronized numbers, build the Pareto table and plot: success (± CI) vs median latency per (sampler, steps, device). Report per-*decision* latency (one chunk) and per-*control-step* latency (divide by $T_a$).

**✅ Checkpoint:** the unsynchronized numbers are visibly too small and you can say why; the Pareto plot has one point per config × device.

## Exercise 6 — Why straight paths win, on real data [Predict → Run]

Tests objective 4: the mechanism (straightness) and the consequence (few-step quality) in one report, on robot data instead of the tutorial's toy blobs (its Figs. 24–27).

1. From `lerobot/svla_so101_pickplace`, extract a 2-D slice of the action distribution (two joints, e.g. shoulder-lift vs elbow) — a few thousand points; hold out 20% as a reference set.
2. **Build** (spec): reuse Lesson 12's `heads/ddpm.py` (ε-MLP, 100 steps) and `heads/cfm.py` (velocity MLP) on this 2-D distribution, conditioned on a constant. Add a strided deterministic DDIM path to the DDPM head, `sample(s, n, steps, method="ddim")`. Check: DDIM-100 and ancestral DDPM-100 give the same energy distance to the reference set within noise.
3. **Write first:** predicted mean straightness ratio per head; predicted quality-vs-steps curves for DDIM ∈ {1, 2, 5, 10} and Euler ∈ {1, 2, 5, 10} (which flattens first, and why).
4. From a shared set of noise seeds, integrate both samplers and record full denoising trajectories. Compute **straightness ratio** = path length / chord length per trajectory; plot side by side; animate one pair as the lesson GIF.
5. For each (sampler, steps), draw 2,000 samples and compute the energy distance (or an RBF-kernel MMD) to the reference set. Plot quality vs steps, both heads.

**✅ Checkpoint:** FM straightness ≈ 1.0–1.1, DDPM well above; the CFM quality curve is flat by Euler-{2, 5} while DDIM degrades below ~5 steps — consistent with Exercise 4's shape. Where the two disagree, the reconciliation is in `RESULTS.md`.

## Exercise 7 — The H3 deployment choice [Decide]

Tests objective 5. From the Pareto table: for the SO-101 at 30 Hz on `mps`, choose the sampler and step count you would ship, state the per-control-step latency budget it clears, and name the row that supports it. Then state what would make you switch to a flow-matching head (Lesson 18's SmolVLA already is one).

**✅ Checkpoint:** the decision, its supporting row, and the switch condition are written in `RESULTS.md`.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| Hub: `<you>/dp_pusht_base` | loads via `--policy.path`; model card links this lesson |
| `run_sampler_study.py` | reproduces the Pareto table and plot (both devices) from one command, seeds fixed |
| `slice_study.py` | reproduces the straightness figure, quality-vs-steps plot, and the GIF from one command |
| `plots/` | Pareto plot, straightness figure + GIF, quality-vs-steps |
| `RESULTS.md` | config-deviation list; Exercise 3, 4, and 6 predictions with reconciliations; sampler table with CIs and synchronized latencies; the H3 decision with its row; ≤ 12 sentences of interpretation |

## Done when

- [ ] Baseline DP in the expected band on 50 seeded episodes, CI reported, cross-checked against `lerobot-eval`.
- [ ] Four-config sampler table complete with synchronized latency on both devices.
- [ ] DDIM-10 ≈ DDPM-100 within CI (or a documented investigation of why not).
- [ ] Straightness and few-step quality measured on real data; predictions written before the runs.
- [ ] The H3 deployment choice is written down with its supporting row.

## Self-check

1. Why execute only $T_a{=}8$ of $T_p{=}16$ predicted actions? What breaks at $T_a{=}T_p$ and at $T_a{=}1$? (Lesson 14's ensembling is the mirror image.)
2. DDIM reuses DDPM's weights. Exactly what property of the two processes makes that legal?
3. Your ε-parameterized sampler misbehaves worst at few steps near $t \to 0$. Which Lesson 13 self-check predicted this, and what is the mechanism?
4. Which failure from Lesson 12 would reappear if you replaced DP's sampler with "just output the conditional mean of the denoiser at $t{=}T$"?
5. PushT trains at 10 fps but H3's arm runs at 30 fps. List the two places in the config where fps assumptions hide.

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Success ≪ 50% but loss looks fine | obs-type mismatch (policy trained on pixels, env giving keypoints, or vice versa) | pin `obs_type` end-to-end; check the env features the config resolved |
| Eval crashes on cloud: GL context | headless MuJoCo/pygame rendering | `MUJOCO_GL=egl`; for gym-pusht specifically ensure a virtual display or `render_mode="rgb_array"` |
| DDIM-1 outputs saturate the action bounds | ε-parameterization instability at tiny step counts | expected — report it; clip actions; cite Lesson 13 self-check #4 |
| CFM head much worse than DDPM head on the slice | unequal training budget or trunk mismatch | identical trunk, steps, lr; only the objective/sampler differ |
| Latency numbers absurdly small | measuring async dispatch, not compute | synchronize the device before both timestamps; discard warm-up (Exercise 5) |
| `mps` NaNs in U-Net GroupNorm | fp16/float64 edge cases | force float32; smoke-test 100 steps on `cpu` first |
| My eval disagrees with `lerobot-eval` by > CI | different success definition (strict solve vs max-coverage) | log both; compare like with like |

## Going deeper

- **A flow-matching DP.** LeRobot's DP head ships DDPM/DDIM schedulers only — CFM changes the *training* objective, not just sampling, so there is no drop-in scheduler. Fork the DP model into a standalone `dp_cfm/` (keypoint obs `environment_state_agent_pos` to keep it Mac-trainable; same U-Net, conditioning, horizons), replace the objective with Lesson 12's CFM loss, train ~100k steps, and evaluate at Euler ∈ {1, 2, 5, 10} against a keypoint-obs DDPM twin with the same trunk. This adds the full-task FM row to the Pareto plot.
- **Fancy samplers vs retraining.** Distill the DDPM baseline into a 1-step student via consistency distillation, *or* implement DPM-Solver++ on the trained weights, and add the point. One extra point on that plot tells you whether sampler engineering beats retraining with FM.

## References

- Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023 / IJRR 2024. arXiv:2303.04137.
- Song, Meng, Ermon. *Denoising Diffusion Implicit Models*, ICLR 2021. arXiv:2010.02502.
- LeRobot team. *Robot Learning: A Tutorial*, §4.3 (and Figs. 24–27). arXiv:2510.12403.
- Lipman et al. *Flow Matching Guide and Code*, 2024. arXiv:2412.06264.
- gym-pusht: github.com/huggingface/gym-pusht; dataset card: huggingface.co/datasets/lerobot/pusht.
