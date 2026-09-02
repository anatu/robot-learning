# Lesson 15 — Diffusion Policy and the Sampler Study

This lesson trains Diffusion Policy on the PushT task and then studies the question that decides whether a diffusion policy can be deployed at all: how many denoising steps must be paid for each decision, and why a flow-matching head can pay fewer. You will train the baseline with LeRobot, evaluate it with the harness from Lesson 14, compare samplers on one set of trained weights, measure latency in a way that does not lie, and then reproduce the straight-path argument from Lesson 13 on real robot action data. The deployment decision you make at the end is the one H3 uses on the physical arm.

| | |
|---|---|
| **Phase** | 4 — Generative imitation policies |
| **Time** | 4–6 h desk + ~1 GPU-h cloud for the baseline (or an overnight `mps` run); the 2-D study runs on the Mac in minutes |
| **Cost** | ~$2–3 cloud; $0 if trained locally overnight |
| **Prerequisites** | 13 (the DDPM loss, DDIM as a shared-marginal sampler, the ε/score/velocity conversions), 14 (`evaluate(policy, env_id, seeds) -> EvalReport`, reused unchanged), 12 (`heads/{ddpm,cfm}.py` with `sample(s, n)`, reused in Exercise 6) |
| **Feeds into** | 16 (this checkpoint is served asynchronously; DP's latency is the reason async inference exists), H3 (the same recipe on the real arm) |

## Learning objectives

After this lesson you can:

1. **Explain** Diffusion Policy's receding-horizon design, in which the policy conditions on $T_o$ observations, predicts $T_p$ actions, and executes $T_a$ of them, and say what each horizon buys and what it costs.
2. **Predict** which sampler configurations retain the baseline's success and which degrade, before running them, and reconcile the prediction against measured success and latency.
3. **Quantify** the sampler trade-off correctly: success with a confidence interval against synchronized wall-clock time per decision and per control step.
4. **Demonstrate** on real robot data why flow-matching paths tolerate few-step sampling, by measuring path straightness and few-step sample quality side by side.
5. **Decide** on a sampler and step count for a stated deployment constraint and defend the choice with your own numbers.

## Principles

### The receding-horizon architecture

Diffusion Policy is a conditional generative model over action sequences rather than over single actions. At each control step it samples a chunk $A_t \in \mathbb{R}^{T_p \times d_a}$ conditioned on the last $T_o$ observations, executes the first $T_a$ of those actions, and then repeats with a fresh observation. The paper's settings are $T_o = 2$, $T_p = 16$ and $T_a = 8$. The observation conditioning enters a one-dimensional temporal CNN U-Net through FiLM layers; the paper also describes a transformer variant, but the U-Net is the more robust default and the one LeRobot ships. Training uses exactly the simplified loss derived in Lesson 13 with $K = 100$ diffusion steps, and the paper keeps an exponential moving average of the weights, which stabilizes the samples.

The three horizons interact in a way that is worth thinking through before touching the code. Predicting a chunk of $T_p$ actions gives the policy the decision-horizon benefit that Lesson 14 demonstrated for ACT. Executing only the first $T_a$ of them keeps the policy reactive, because a new observation is consulted every $T_a$ steps rather than every $T_p$. If $T_a$ were raised to $T_p$, the policy would run open-loop for a full chunk; if it were lowered to 1, the policy would re-plan at every step and lose most of the benefit of chunking. Lesson 14's temporal ensembling addresses the same trade-off from the other side, by smoothing between overlapping chunks rather than by choosing how many actions to commit to.

### Why the sampler matters for this policy and not for ACT

ACT pays one forward pass per chunk. Diffusion Policy pays one forward pass per denoising step per chunk, so a chunk produced by 100 ancestral DDPM steps costs 100 forward passes, which on the hardware used in this course can mean tens to hundreds of milliseconds per decision. That cost is what makes inference latency a first-class design constraint for diffusion policies, and it is the reason Lesson 16 exists. There are two ways to reduce it, and they differ in how much of the system they change.

The first is DDIM (Song et al. 2021). DDIM defines a family of non-Markovian processes that share DDPM's marginal distributions $q(x_t \mid x_0)$. Its deterministic member allows strided sampling, in which the same trained $\epsilon$-network is evaluated on a sub-schedule of, for example, 10, 5, or even 1 of the 100 training timesteps. No retraining is needed, because the network was trained to denoise samples drawn from exactly those marginals, and DDIM asks it to do nothing else.

The second is flow matching. Here the head is retrained to regress a velocity field along the straight optimal-transport path between noise and data. Because the path is straight, Euler integration with very few steps is nearly exact. Lesson 13 established the machinery; this lesson measures its consequence.

The $\epsilon$-parameterization has a known weak spot that you should expect to see at very small step counts. As $t \to 0$ the noise level vanishes, and the output of even a perfectly trained $\epsilon$-model stops carrying information about $x_0$ (Lesson 13, self-check question 4). A large stride at the low-noise end of the schedule therefore overshoots, and the practical symptom is that DDIM with a single step produces actions that saturate the action bounds.

### The task

The environment is `gym_pusht/PushT-v0`, in which an agent pushes a T-shaped block onto a target zone. Success is defined as the block covering at least 95% of the goal zone; the reward is the coverage fraction; the action is a target agent position in $[0, 512]^2$. Two observation types are available: `pixels_agent_pos` (a 96×96 RGB image plus the 2-D agent position) and `environment_state_agent_pos` (16-D keypoints plus the agent position). The dataset `lerobot/pusht` contains 206 episodes and 25,650 frames at 10 fps with 96×96 images. PushT is deliberately multimodal, because the block can be approached from either side, and that is why it became the standard benchmark for Diffusion Policy: a policy that averages the two approaches fails. These facts have been stable, but verify them against the installed package.

**Carry forward**

- Diffusion Policy conditions on $T_o$ observations, predicts a chunk of $T_p$ actions, and executes $T_a$ of them; each horizon has a characteristic failure at both extremes, so the settings are design decisions rather than defaults to accept.
- DDIM can reuse DDPM's trained weights because the two processes share the same marginals; it changes only the sampler. Flow matching changes the training objective, so it requires retraining.
- A latency measurement is only meaningful if the accelerator is synchronized before both timestamps, because GPU and MPS dispatch return before the computation has finished.
- The number of steps a sampler can skip is governed by how straight its denoising paths are, and straightness is a quantity you can measure directly.

| Source | Read for |
|---|---|
| Chi et al. 2023/2024 (Diffusion Policy) §3, §4.4 | the receding-horizon rationale; why position control and chunking; the ablations behind $T_o{=}2, T_p{=}16, T_a{=}8$; where EMA matters |
| Tutorial §4.3 | how the tutorial frames DP after ACT, and what the CVAE could not do that diffusion can |
| Song et al. 2021 (DDIM) §4 | the sub-sequence sampling trick you use verbatim, and where its determinism comes from |
| `configuration_diffusion.py` (installed) | every knob for Exercise 2, and the fields you sweep in Exercise 4: `noise_scheduler_type`, `num_train_timesteps`, `num_inference_steps` |

## Exercise 1 — Inspect the dataset [Read]

Before training, look at the data to see the multimodality that Diffusion Policy exists to handle. The purpose of this exercise is to find, in the real dataset, a pair of demonstrations that solve a similar block configuration from opposite sides, since that is exactly the situation in which a regression policy would average the two and fail.

1. Load `lerobot/pusht` and print the feature shapes; confirm 206 episodes, 10 fps, and `action: (2,)` in $[0,512]$.
2. Plot 20 episode trajectories in the 2-D action plane, colored by episode, and find a pair of episodes that solve a similar block pose from opposite sides.
3. Load the dataset with DP-shaped windows, using `delta_timestamps` to request 2 observation steps back and 16 action steps forward, and confirm the shapes `(2, ...)` and `(16, 2)`.

**✅ Checkpoint:** the shapes are as stated, and you can point at two opposite-strategy episodes.

## Exercise 2 — Train the baseline [Read]

In this exercise you launch LeRobot's training recipe, but the learning objective is to know which configuration you actually ran rather than merely to run it. LeRobot's defaults have drifted from the paper's horizons, and a comparison against the paper is meaningful only if the horizons match.

1. Diff the paper's horizons against the installed defaults in `configuration_diffusion.py`. Recent LeRobot versions default to a longer horizon (for example `horizon=64` and `n_action_steps=32`) than the paper's 16 and 8. Record the defaults in `RESULTS.md`, then pin the paper's values:
   ```bash
   lerobot-train \
     --dataset.repo_id=lerobot/pusht \
     --policy.type=diffusion \
     --policy.horizon=16 --policy.n_action_steps=8 --policy.n_obs_steps=2 \
     --output_dir=outputs/train/dp_pusht_base --job_name=dp_pusht_base \
     --policy.device=cuda --wandb.enable=true
   ```
   The run is about 100k steps, which takes roughly an hour on an A100 or overnight on `mps`.
2. Before looking at the loss curve, write down what the $\epsilon$-MSE curve can tell you and what it cannot. Then read it. It falls quickly for about 10k steps and improves slowly afterwards, and it contains no success signal at all, because the loss measures denoising accuracy rather than task performance. Coverage at evaluation time is the only measure of success.
3. Push the checkpoint to `<you>/dp_pusht_base`.

**✅ Checkpoint:** the W&B run is live, the checkpoint is on the Hub, and the deviation list has at least two entries.

## Exercise 3 — Evaluate with the Lesson 14 harness [Predict → Run]

A success rate without a seed list and a confidence interval cannot be compared against anything. In this exercise you evaluate the baseline with the harness from Lesson 14 and cross-check it against LeRobot's own evaluator, so that every later comparison in this lesson rests on a number you trust.

1. Before running, write down the strict-success band and the mean maximum coverage you expect, using the reference band given in step 4.
2. Point Lesson 14's `evaluate()` at PushT with 50 seeds (`range(2000, 2050)`). Success is the environment's solved flag (coverage of at least 95%); also record the final coverage and the steps to success, and save 3 success and 3 failure videos.
3. Cross-check one number against the stock evaluator:
   ```bash
   MUJOCO_GL=egl lerobot-eval --policy.path=<you>/dp_pusht_base \
     --env.type=pusht --eval.n_episodes=50 --eval.batch_size=10 --policy.device=cuda
   ```
4. Report success with a 95% Wilson interval and the mean final coverage. Reference reproductions land in the 60–70% strict-success band with a mean maximum coverage above 0.90. A result below 50% should be treated as a bug (see Pitfalls) rather than as a finding.

**✅ Checkpoint:** your harness and `lerobot-eval` agree within their intervals, and the numbers are in the expected band.

## Exercise 4 — Compare samplers on the same weights [Predict → Run]

DDIM's claim is that the sampler can be changed without touching the model. This exercise tests that claim directly: the same checkpoint is evaluated under DDPM with 100 steps and under DDIM with 10, 5, and 1 steps. Before running, predict which configurations retain the baseline's success and which degrade, using the argument about the $\epsilon$-parameterization near $t \to 0$ from the Principles section.

1. For each configuration in {DDPM-100, DDIM-10, DDIM-5, DDIM-1}, write down the predicted success relative to the baseline (within the interval, degraded, or collapsed) and the mechanism behind the prediction. Name the configuration you expect to saturate the action bounds, and say why.
2. On the same `dp_pusht_base` weights, evaluate DDPM at $K = 100$ (the training configuration), then DDIM at `num_inference_steps` of 10, 5 and 1, using `--policy.noise_scheduler_type=DDIM` together with `--policy.num_inference_steps=<n>` at load or evaluation time. The flag names mirror the fields of `configuration_diffusion.py`, and `lerobot-eval --help` is the authority if they have moved. DDIM here reuses the $\epsilon$-network on a strided sub-schedule, which is the Lesson 13 stretch exercise carried out on real weights. Use 50 seeds for each configuration and the harness from Exercise 3.
3. Reconcile each prediction against its measured result.

**✅ Checkpoint:** DDIM-10 retains DDPM-100's success within the confidence interval, and DDIM-1 degrades visibly. Any deviation from that shape needs a hypothesis in `RESULTS.md`.

## Exercise 5 — Measure latency correctly [Diagnose]

Latency tables in papers and READMEs are often wrong in the same way: they time the call that dispatches work to the accelerator rather than the work itself. In this exercise you make that mistake deliberately, measure the size of the error, and then build the Pareto table from the correct numbers.

1. Predict first. You will time 100 policy calls without synchronizing the device, and then again with `torch.cuda.synchronize()` or `torch.mps.synchronize()` before each timestamp. Write down the direction and rough magnitude of the discrepancy, and which of the two measurements is wrong.
2. Run both measurements on `mps` (which is the deployment device for H3) and on the cloud GPU, taking the median over 100 calls after 10 warm-up calls. Explain the mechanism: asynchronous dispatch returns to Python before the computation has finished, so an unsynchronized timer measures the cost of enqueueing work rather than doing it.
3. Using the synchronized numbers, build the Pareto table and plot: success with its confidence interval against median latency, one point per (sampler, steps, device). Report both per-decision latency (one chunk) and per-control-step latency (the per-decision figure divided by $T_a$).

**✅ Checkpoint:** the unsynchronized numbers are visibly too small and you can explain why, and the Pareto plot has one point per configuration and device.

## Exercise 6 — Measure straightness and few-step quality on real action data [Predict → Run]

Lesson 13 argued that flow matching tolerates few-step sampling because its denoising paths are straight. This exercise measures that property on a two-dimensional slice of real robot actions and, in the same experiment, measures how sample quality degrades with step count for each head, so that the mechanism and its consequence appear in one report. The tutorial illustrates the same idea on toy distributions in its Figures 24–27; here the data are real.

1. From `lerobot/svla_so101_pickplace`, extract a 2-D slice of the action distribution, for example shoulder-lift against elbow, amounting to a few thousand points. Hold out 20% as a reference set.
2. [Build] Write the specification and have an AI tool draft the code: reuse Lesson 12's `heads/ddpm.py` (an $\epsilon$-MLP with 100 steps) and `heads/cfm.py` (a velocity MLP) on this 2-D distribution, conditioned on a constant. Add a strided deterministic DDIM path to the DDPM head, `sample(s, n, steps, method="ddim")`. The check is that DDIM-100 and ancestral DDPM-100 give the same energy distance to the reference set within noise, which confirms that the DDIM path is implemented correctly before it is used with fewer steps.
3. Before running, write down the predicted mean straightness ratio for each head, and the predicted quality-versus-steps curves for DDIM at 1, 2, 5, and 10 steps and Euler at 1, 2, 5, and 10 steps. State which curve you expect to flatten first, and why.
4. From a shared set of noise seeds, integrate both samplers and record the full denoising trajectories. Compute the straightness ratio, defined as path length divided by chord length, for each trajectory. Plot the trajectories side by side and animate one pair as a GIF.
5. For each (sampler, steps) pair, draw 2,000 samples and compute the energy distance (or an RBF-kernel MMD) to the reference set. Plot quality against steps for both heads.

**✅ Checkpoint:** the flow-matching straightness ratio is about 1.0–1.1 and the DDPM ratio is well above it; the CFM quality curve is flat by 2–5 Euler steps while the DDIM curve degrades below about 5 steps, which is consistent with the shape found in Exercise 4. Where the two experiments disagree, the reconciliation belongs in `RESULTS.md`.

## Exercise 7 — Choose the H3 deployment configuration [Decide]

The Pareto table exists so that a deployment decision can be made from evidence rather than habit. For the SO-101 running at 30 Hz on `mps`, choose the sampler and step count you would ship, state the per-control-step latency budget that the choice clears, and name the table row that supports it. Then state the condition under which you would switch to a flow-matching head instead; note that SmolVLA, which you fine-tune in Lesson 18, already uses one.

**✅ Checkpoint:** the decision, its supporting row, and the switch condition are written in `RESULTS.md`.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| Hub: `<you>/dp_pusht_base` | loads via `--policy.path`; the model card links this lesson |
| `run_sampler_study.py` | reproduces the Pareto table and plot on both devices from one command, with fixed seeds |
| `slice_study.py` | reproduces the straightness figure, the quality-versus-steps plot, and the GIF from one command |
| `plots/` | the Pareto plot; the straightness figure and GIF; the quality-versus-steps plot |
| `RESULTS.md` | the configuration-deviation list; the Exercise 3, 4 and 6 predictions with their reconciliations; the sampler table with confidence intervals and synchronized latencies; the H3 decision with its supporting row; at most 12 sentences of interpretation |

## Done when

- [ ] The baseline is in the expected band on 50 seeded episodes, with a confidence interval, and cross-checked against `lerobot-eval`.
- [ ] The four-configuration sampler table is complete, with synchronized latency on both devices.
- [ ] DDIM-10 matches DDPM-100 within the confidence interval, or there is a documented investigation of why not.
- [ ] Straightness and few-step quality are measured on real data, with predictions written before the runs.
- [ ] The H3 deployment choice is written down with its supporting row.

## Self-check

1. Why execute only $T_a = 8$ of the $T_p = 16$ predicted actions? What breaks at $T_a = T_p$, and what breaks at $T_a = 1$? Lesson 14's ensembling is the mirror image of this question.
2. DDIM reuses DDPM's weights. Exactly which property of the two processes makes that legitimate?
3. Your $\epsilon$-parameterized sampler misbehaves worst at few steps near $t \to 0$. Which Lesson 13 self-check predicted this, and what is the mechanism?
4. Which failure from Lesson 12 would reappear if you replaced DP's sampler with the conditional mean of the denoiser at $t = T$?
5. PushT trains at 10 fps but H3's arm runs at 30 fps. Where are the two places in the configuration in which fps assumptions hide?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Success far below 50% but the loss looks fine | observation-type mismatch (policy trained on pixels while the env gives keypoints, or vice versa) | pin `obs_type` end to end; check which env features the config resolved |
| Evaluation crashes on the cloud box with a GL-context error | headless MuJoCo or pygame rendering | `MUJOCO_GL=egl`; for gym-pusht specifically, ensure a virtual display or `render_mode="rgb_array"` |
| DDIM-1 outputs saturate the action bounds | $\epsilon$-parameterization instability at very small step counts | expected; report it, clip the actions, and cite Lesson 13 self-check 4 |
| The CFM head is much worse than the DDPM head on the slice | unequal training budget or a trunk mismatch | identical trunk, steps and learning rate; only the objective and sampler may differ |
| Latency numbers are absurdly small | timing asynchronous dispatch rather than compute | synchronize the device before both timestamps and discard the warm-up calls (Exercise 5) |
| `mps` NaNs in the U-Net's GroupNorm | fp16 or float64 edge cases | force float32; smoke-test 100 steps on `cpu` first |
| Your evaluation disagrees with `lerobot-eval` by more than the interval | different success definitions (strict solve versus maximum coverage) | log both and compare like with like |

## Going deeper

- **A flow-matching Diffusion Policy.** LeRobot's DP head ships only DDPM and DDIM schedulers, and there is no drop-in CFM scheduler because flow matching changes the training objective rather than the sampler. Fork the DP model into a standalone `dp_cfm/` package, using the keypoint observation `environment_state_agent_pos` so that it trains on a Mac and keeping the same U-Net, conditioning, and horizons; replace the objective with Lesson 12's CFM loss; train for about 100k steps; and evaluate at 1, 2, 5, and 10 Euler steps against a keypoint-observation DDPM twin with the same trunk. This adds the full-task flow-matching row to the Pareto plot.
- **Sampler engineering versus retraining.** Distill the DDPM baseline into a one-step student by consistency distillation, or implement DPM-Solver++ on the trained weights, and add the resulting point to the Pareto plot. That one point answers whether a better sampler can match what retraining with flow matching buys.

## References

- Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023 / IJRR 2024. arXiv:2303.04137.
- Song, Meng, Ermon. *Denoising Diffusion Implicit Models*, ICLR 2021. arXiv:2010.02502.
- LeRobot team. *Robot Learning: A Tutorial*, §4.3 (and Figs. 24–27). arXiv:2510.12403.
- Lipman et al. *Flow Matching Guide and Code*, 2024. arXiv:2412.06264.
- gym-pusht: github.com/huggingface/gym-pusht; dataset card: huggingface.co/datasets/lerobot/pusht.
