# Lesson 15 — Diffusion Policy + Sampler Study

Train Diffusion Policy on PushT, then attack the question that decides real-world deployability: how few denoising steps can you pay at 50 Hz? DDPM vs DDIM vs a flow-matching head, on one task, one harness, one Pareto plot.

| | |
|---|---|
| **Phase** | 4 — Generative imitation policies |
| **Time** | 6–8 h desk + ~2 GPU-h cloud (or an overnight `mps` run for the baseline) |
| **Cost** | ~$3–5 cloud (baseline + FM arm); $0 if trained locally overnight |
| **Prerequisites** | 13 (DDPM loss, DDIM as shared-marginal sampler, ε/score/velocity conversions), 14 (the eval-harness contract `evaluate(policy, env_id, seeds) -> EvalReport` — reused unchanged), 12 (why the action head is generative at all) |
| **Feeds into** | 16 (this checkpoint gets served async; DP's inference latency is exactly why async exists), H3 (same recipe on the real arm) |

## Learning objectives

After this lesson you can:

1. **Explain** Diffusion Policy's receding-horizon design — predict $T_p$, execute $T_a$, condition on $T_o$ — and what each horizon buys.
2. **Train** DP on `gym-pusht` with LeRobot and diagnose its loss curves and eval behavior.
3. **Quantify** the sampler trade: success rate vs wall-clock inference latency for DDPM-100, DDIM-{10, 5, 1}, and a CFM head at Euler-{1, 2, 5, 10}.
4. **Demonstrate**, on real robot data, why flow paths tolerate few-step sampling: measure path straightness of diffusion vs FM denoising trajectories.
5. **Choose** a sampler/step-count for a deployment constraint (e.g. "10 ms budget on `mps`") and defend the choice with your own numbers.

## Background

**The architecture.** DP is a conditional generative model over *action sequences*: at each control step, sample an action chunk $A_t \in \mathbb{R}^{T_p \times d_a}$ conditioned on the last $T_o$ observations, execute the first $T_a$ actions, repeat. Paper settings: $T_o = 2$, $T_p = 16$, $T_a = 8$. Conditioning enters a 1-D temporal CNN U-Net via FiLM layers (a transformer head is the paper's alternative; the U-Net is the robust default). Training is exactly Lesson 13's simplified loss with $K = 100$ diffusion steps; the paper leans on an EMA copy of the weights for stability.

**Why samplers matter here and nowhere else in the course so far.** ACT pays one forward pass per chunk. DP pays one per *denoising step* per chunk — 100 ancestral DDPM steps can mean tens-to-hundreds of ms per decision. The escape hatches, in increasing radicalism:
- **DDIM** (Song et al. 2021): a non-Markovian family sharing DDPM's marginals; its deterministic member permits *strided* sampling — reuse the same trained ε-network, sample on a sub-schedule of {10, 5, 1} steps. No retraining.
- **Flow matching**: retrain the head to regress a velocity field along the straight OT path. Straight paths ⇒ Euler integration with very few steps is nearly exact (you proved the machinery in Lesson 13; here you measure it).

**Task facts** (verify against the installed package, but these are stable): env `gym_pusht/PushT-v0` — push a T-block onto a target zone; success = block ≥ 95% inside the goal zone; reward = coverage; action = target agent position in $[0,512]^2$; obs types include `pixels_agent_pos` (96×96 RGB + 2-D agent position) and `environment_state_agent_pos` (16-D keypoints + agent pos). Dataset `lerobot/pusht`: 206 episodes, 25,650 frames, 10 fps, 96×96 images. PushT is deliberately multimodal — the block can be approached from either side — which is why it's *the* DP benchmark.

| Source | Read for |
|---|---|
| Chi et al. 2023/2024 (Diffusion Policy) §3, §4.4 | receding-horizon rationale; why position control + chunking; the ablations behind $T_o{=}2, T_p{=}16, T_a{=}8$; where EMA matters |
| Tutorial §4.3 | how the tutorial frames DP after ACT: what the CVAE couldn't do that diffusion can |
| Song et al. 2021 (DDIM) §4 | the sub-sequence sampling trick you'll use verbatim; where determinism comes from |
| `configuration_diffusion.py` (installed) | every knob for Part 1; the fields you'll sweep in Part 3: `noise_scheduler_type`, `num_train_timesteps`, `num_inference_steps` |

## Part 0 — Know your data (Mac, ~30 min)

1. Load `lerobot/pusht`, print feature shapes; confirm 206 episodes / 10 fps / `action: (2,)` in $[0,512]$.
2. Plot 20 episode trajectories in the 2-D action plane, colored by episode. Find a pair of episodes that solve a similar block pose from opposite sides — that's the multimodality DP must not average.
3. Load with DP-shaped windows: `delta_timestamps` giving 2 obs steps back and 16 action steps forward; confirm shapes `(2, ...)` / `(16, 2)`.

**✅ Checkpoint:** shapes as stated; you can point at two opposite-strategy episodes.

## Part 1 — Train the baseline (cloud ~1 h on A100, or overnight `mps`, ~100k steps)

1. Launch:
   ```bash
   lerobot-train \
     --dataset.repo_id=lerobot/pusht \
     --policy.type=diffusion \
     --output_dir=outputs/train/dp_pusht_base --job_name=dp_pusht_base \
     --policy.device=cuda --wandb.enable=true
   ```
2. Before launching, diff the paper's horizons against your installed defaults (`configuration_diffusion.py`). Recent LeRobot mains default to a *longer* horizon (e.g. `horizon=64`, `n_action_steps=32`) than the paper's 16/8. Pin the paper values for comparability: `--policy.horizon=16 --policy.n_action_steps=8 --policy.n_obs_steps=2`. Record what the defaults were in `RESULTS.md`.
3. Loss reading: the ε-MSE falls fast for ~10k steps, then improves slowly; there is no "success" signal in the loss — coverage-at-eval is the only truth. Push the checkpoint to `<you>/dp_pusht_base`.

**✅ Checkpoint:** W&B run live; checkpoint on Hub; deviation list recorded (≥ 2 entries expected).

## Part 2 — Evaluate with the Lesson 14 harness (Mac or cloud, ~30 min)

1. Point the harness at PushT: 50 seeds (`range(2000, 2050)`), success = env's solved flag (≥ 95% coverage), also record final coverage and steps-to-success; dump 3 success + 3 failure videos.
2. Sanity-cross-check one number against the stock evaluator:
   ```bash
   MUJOCO_GL=egl lerobot-eval --policy.path=<you>/dp_pusht_base \
     --env.type=pusht --eval.n_episodes=50 --eval.batch_size=10 --policy.device=cuda
   ```
3. Report success ± Wilson CI and mean final coverage. Reference reproductions land in the ~60–70% strict-success band with high (> 0.90) mean max coverage — treat < 50% as a bug (see Pitfalls), not a result.

**✅ Checkpoint:** your harness and `lerobot-eval` agree within CIs; numbers in the expected band.

## Part 3 — The sampler study (the point of the lesson; ~2–3 h)

Two model arms, six sampler configs, one table.

1. **DDIM needs no retraining.** On the *same* `dp_pusht_base` weights, evaluate: DDPM at $K{=}100$ (the training config), then DDIM at `num_inference_steps` ∈ {10, 5, 1} (`--policy.noise_scheduler_type=DDIM` at load/eval time — DDIM reuses the ε-network on a strided sub-schedule; that's the Lesson 13 stretch made real). 50 seeds each.
2. **The FM arm retrains.** LeRobot's DP head ships DDPM/DDIM schedulers only — there is no drop-in CFM scheduler, because CFM changes the *training* objective, not just sampling. So: fork the DP model file into a standalone `dp_cfm/` (keypoint obs `environment_state_agent_pos` to keep it Mac-trainable — same U-Net, conditioning, horizons), replace the DDPM objective with Lesson 12's CFM loss, train ~100k steps, evaluate at Euler steps ∈ {1, 2, 5, 10}. Train a keypoint-obs DDPM twin with the same trunk so the FM-vs-DDPM comparison is apples-to-apples.
3. **Latency, measured honestly:** median over 100 policy calls after 10 warm-up calls, with `torch.cuda.synchronize()` / MPS sync before each timestamp; measure on both `mps` (deployment reality for H3) and the cloud GPU. Report per-*decision* latency (one chunk) and per-*control-step* latency (divide by $T_a$).
4. Produce the Pareto table and plot: success (± CI) vs median latency, one point per (sampler, steps, device).

**✅ Checkpoint:** DDIM-10 retains DDPM-100 success within CI at ~10× less compute; DDIM-1 degrades visibly; the CFM head at Euler-{2, 5} sits at-or-near its DDPM twin's success at a fraction of the steps. Any deviation from this shape → a hypothesis in `RESULTS.md` (Lesson 13 self-check #4 is the usual suspect at very low step counts).

## Part 4 — Why straight paths win, on real data (Mac, ~1–2 h)

Reproduce the tutorial's Figures 24–27 idea on real robot data instead of toy blobs.

1. From `lerobot/svla_so101_pickplace`, extract a 2-D slice of the action distribution (two joints, e.g. shoulder-lift vs elbow) — a few thousand points.
2. Train a small ε-MLP (DDPM, 100 steps) and a CFM velocity MLP on this 2-D distribution (Lesson 12 code, reused).
3. From a shared set of noise seeds, integrate both samplers and record full denoising trajectories. Plot side-by-side; compute **straightness ratio** = path length / chord length per trajectory.
4. Report mean straightness: expect FM ≈ 1.0–1.1 and DDPM well above it. Animate one pair of trajectories as the lesson GIF.

**✅ Checkpoint:** measured straightness gap consistent with the Part 3 few-step results — the mechanism and the consequence now sit in the same report.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| Hub: `<you>/dp_pusht_base` (+ keypoint DDPM/CFM twins) | load via `--policy.path`; model cards link this lesson |
| `eval.py` usage (unchanged from L14) | PushT adapter only; seeds/CI/videos as specced |
| `dp_cfm/` | standalone, < 500 lines, trains on `mps`; README one-paragraph diff vs LeRobot's DP |
| `plots/` | Pareto plot (all arms, both devices), straightness figure + GIF |
| `RESULTS.md` | config-deviation list; sampler table with CIs and latencies; straightness numbers; a stated deployment choice for H3 ("on `mps`, I'd ship X at N steps because...") ≤ 12 sentences |

## Done when

- [ ] Baseline DP in the expected band on 50 seeded episodes, CI reported, cross-checked against `lerobot-eval`.
- [ ] Six-config sampler table complete with honest latency methodology.
- [ ] DDIM-10 ≈ DDPM-100 within CI (or a documented investigation of why not).
- [ ] Straightness measured on real data; number appears next to the few-step claim it explains.
- [ ] The H3 deployment choice is written down with its supporting row from the table.

## Self-check

1. Why execute only $T_a{=}8$ of $T_p{=}16$ predicted actions? What breaks at $T_a{=}T_p$ and at $T_a{=}1$? (Lesson 14's ensembling discussion is the mirror image.)
2. DDIM reuses DDPM's weights. Exactly what property of the two processes makes that legal?
3. Your ε-parameterized sampler misbehaves worst at few steps near $t \to 0$. Which Lesson 13 self-check predicted this, and what's the mechanism?
4. Which failure from Lesson 12 would reappear if you replaced DP's sampler with "just output the conditional mean of the denoiser at $t{=}T$"?
5. PushT trains at 10 fps but H3's arm runs at 30 fps. List the two places in the config where fps assumptions hide.

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Success ≪ 50% but loss looks fine | obs-type mismatch (policy trained on pixels, env giving keypoints, or vice versa) | pin `obs_type` end-to-end; check the env features the config resolved |
| Eval crashes on cloud: GL context | headless MuJoCo/pygame rendering | `MUJOCO_GL=egl`; for gym-pusht specifically ensure a virtual display or `render_mode="rgb_array"` |
| DDIM-1 outputs saturate the action bounds | ε-parameterization instability at tiny step counts | expected — report it; clip actions; cite Lesson 13 self-check #4 |
| CFM twin much worse than DDPM twin | unequal training budget or trunk mismatch | identical trunk, steps, lr; only the objective/sampler differ |
| Latency numbers absurdly small | measuring async dispatch, not compute | synchronize the device before both timestamps; discard warm-up |
| `mps` NaNs in U-Net GroupNorm | fp16/float64 edge cases | force float32; smoke-test 100 steps on `cpu` first |
| My eval disagrees with `lerobot-eval` by > CI | different success definition (strict solve vs max-coverage) | log both; compare like with like |

## Stretch

Distill the DDPM baseline into a 1-step student via consistency distillation *or* implement DPM-Solver++ on the trained weights, and add the point to the Pareto plot. One extra point on that plot is worth more than a paragraph of reading — it tells you whether fancy samplers beat retraining with FM.

## References

- Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023 / IJRR 2024. arXiv:2303.04137.
- Song, Meng, Ermon. *Denoising Diffusion Implicit Models*, ICLR 2021. arXiv:2010.02502.
- LeRobot team. *Robot Learning: A Tutorial*, §4.3 (and Figs. 24–27). arXiv:2510.12403.
- Lipman et al. *Flow Matching Guide and Code*, 2024. arXiv:2412.06264.
- gym-pusht: github.com/huggingface/gym-pusht; dataset card: huggingface.co/datasets/lerobot/pusht.
