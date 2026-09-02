# Lesson 09 — Sample-Efficient, Data-Driven RL: RLPD

This lesson is about the mechanism that makes reinforcement learning practical on a real robot: keeping a small set of demonstrations in a second replay buffer and sampling every gradient batch half from the demonstrations and half from online experience. You will first derive how much that arrangement over-weights a demonstration relative to simply adding the demonstrations to the online buffer, then record demonstrations in simulation, patch Lesson 08's SAC file to support both arrangements, and run a small grid that shows one of them losing its advantage as training proceeds and the other keeping it. Lesson 10 adds a reward classifier and human interventions to exactly this machinery, and H5 runs it on the physical arm.

| | |
|---|---|
| **Phase** | 3 — Reinforcement learning |
| **Time** | ~4–5 h desk time (AI-assisted) + ~2–3 h compute for the run grid (Mac for demos, cloud GPU for the grid) |
| **Cost** | ~$2–3 (8 short training runs; batchable on one 4090) |
| **Prerequisites** | 08 (`sac.py` is the file being patched), 02 (you can read a LeRobotDataset) |
| **Feeds into** | 10 (HIL-SERL adds a classifier and humans to this machinery), H5 (the same recipe on the real arm) |

## Learning objectives

After this lesson you can:

1. **Derive** the effective oversampling factor of a demonstration transition under symmetric 50/50 sampling, and predict how it evolves as the online buffer grows.
2. **Quantify** the sample-efficiency gap between online-only SAC, demonstration preloading, and RLPD on the same task with matched seeds.
3. **Diagnose** critic divergence under off-distribution actions and show what LayerNorm changes.
4. **Record** demonstrations in `gym-hil` through LeRobot's config-driven recorder into a Hub-hosted LeRobotDataset.
5. **Defend** why RLPD deliberately drops the trust-region and behaviour-cloning-penalty machinery of earlier offline-to-online methods.

## Principles

### Why preloading demonstrations is not enough

Online SAC trained from scratch spends most of its early samples discovering that random arm motion does not pick up cubes. In simulation that is merely slow; on hardware each sample costs seconds of wall-clock time and wear on the gripper, so the number of samples spent flailing is the quantity that decides whether the method is usable at all. The obvious remedy is to put $N$ demonstration transitions into the replay buffer before training starts, an arrangement usually called preloading. Preloading helps at first and then stops helping, for a simple reason: the replay buffer samples uniformly, so once the online buffer holds 100k transitions, 1k demonstration transitions are sampled with probability one percent, and whatever signal they carried is drowned by the policy's own early experience.

### Symmetric sampling and the oversampling factor

RLPD (Ball et al. 2023) keeps the demonstrations in their own buffer $\mathcal{D}_{\text{demo}}$ and builds every gradient batch as fifty percent $\mathcal{D}_{\text{demo}}$ and fifty percent $\mathcal{D}_{\text{online}}$, for the whole of training. Under this rule a demonstration transition is sampled per batch with probability $\frac{0.5}{|\mathcal{D}_{\text{demo}}|}$, whereas an online transition is sampled with probability $\frac{0.5}{|\mathcal{D}_{\text{online}}|}$. The ratio of the two, $|\mathcal{D}_{\text{online}}|/|\mathcal{D}_{\text{demo}}|$, is the effective oversampling factor of a demonstration, and it grows throughout training because the online buffer grows while the demonstration buffer does not. This is the entire mechanism: the expert signal never dilutes, because its share of every batch is fixed by construction rather than by chance.

### The supporting components

Three further design choices accompany symmetric sampling, and each is there for a reason you can measure. The first is LayerNorm in the critic networks. Early in training, demonstration actions are far from anything the policy would produce, and an unnormalized critic can extrapolate arbitrarily large Q-values for such off-distribution actions; the actor then chases those values, and the critic diverges. LayerNorm bounds the extrapolation. This is the ablation you will run. The second is a high update-to-data (UTD) ratio: RLPD takes up to twenty gradient steps per environment step in order to extract as much as possible from each sample, and pairs that with critic ensembles to keep the estimates honest. You will use a UTD of one and note the full recipe in your writeup. The third is the absence of any pessimism term or behaviour-cloning penalty. Earlier offline-to-online methods kept the policy close to the demonstrations with an explicit constraint; RLPD's position is that symmetric sampling together with normalization provides enough stabilization on its own, and that the simpler method wins once training is online. Your curves test that position at small scale.

### The task

The environment is `gym-hil`'s Franka pick-cube task (the `PandaPickCubeBase-v0` family): end-effector delta actions resolved through the environment's built-in inverse kinematics, a sparse success reward, and 10 Hz control, following LeRobot's example configurations. Check the reward specification against the gym-hil source for your installed version, because whether the reward is sparse or shaped changes where in an episode the demonstrations matter most.

**Carry forward**

- Preloading demonstrations into the online buffer dilutes them as the buffer grows, because uniform sampling gives each transition a share proportional to its count; symmetric sampling fixes the demonstration share at half of every batch regardless of buffer sizes.
- The effective oversampling factor of a demonstration transition under symmetric sampling is $|\mathcal{D}_{\text{online}}|/|\mathcal{D}_{\text{demo}}|$, and it grows throughout training.
- LayerNorm in the critic bounds Q-value extrapolation on off-distribution actions, and demonstrations are the earliest source of such actions, so they trigger the failure first.
- Most failures of RLPD in practice are in the conversion of demonstrations to transitions, in particular the placement of the reward and the frame of the actions, rather than in the algorithm.

| Source | Read for |
|---|---|
| Tutorial §3.2 ("sample-efficient, data-driven RL") | how the tutorial positions RLPD as the bridge from simulated RL to HIL-SERL |
| Ball et al. 2023, arXiv:2302.02948, §4 and the ablation tables | which components the authors found necessary (LayerNorm, symmetric sampling) and which optional (ensembles at low UTD) |
| gym-hil README and the LeRobot RL-in-sim docs page | environment IDs, the JSON config schema, and the intervention and recording mechanics you reuse in Lesson 10 |

## Exercise 1 — Derive the oversampling arithmetic [Derive]

This exercise tests objective 1 before any training run. The number you compute here is the one the whole lesson turns on, and it is worth having it on paper before you see a curve.

1. With 30 demonstrations (about 3k transitions) and a batch size of 256, compute a demonstration transition's per-batch sampling probability under `preload` and under `rlpd` at $|\mathcal{D}_{\text{online}}| = 0$, 30k, and 150k. Tabulate the six values.
2. Compute the fraction of each batch that is expert data under `preload` at the same three points.
3. Write one sentence predicting which arm's advantage shrinks late in training and by roughly how much.

**✅ Checkpoint:** the six-cell table is in `RESULTS.md` before Exercise 5 starts, and `rlpd`'s expert fraction is 0.5 in every cell.

## Exercise 2 — Record 30 demonstrations [Build]

Here you record the demonstrations that both `preload` and `rlpd` will use, producing `<you>/gymhil_pickcube_demos_v1` on the Hub. The recording workflow is the same one H5 uses on the physical arm, so this is also a rehearsal.

1. Run `pip install -e ".[hilserl]"` in your LeRobot checkout, which installs `gym_hil`.
2. Copy LeRobot's example environment config (`huggingface.co/datasets/lerobot/config_examples`, `rl/gym_hil/env_config.json`) and set:
   ```json
   "env":  { "type": "gym_manipulator", "name": "gym_hil",
             "task": "PandaPickCubeKeyboard-v0", "fps": 10 },
   "dataset": { "repo_id": "<you>/gymhil_pickcube_demos_v1", "task": "pick_cube",
                "num_episodes_to_record": 30, "push_to_hub": true },
   "mode": "record"
   ```
   If you have a gamepad, use the gamepad task variant; it is considerably easier to control than the keyboard.
3. Run `mjpython -m lerobot.rl.gym_manipulator --config_path <cfg>` and record 30 successful episodes, discarding failures at record time.
4. Inspect the dataset in the LeRobot visualizer. Confirm the action space (end-effector deltas plus gripper) and that episode lengths fall in the expected range of roughly 50–150 steps.

**✅ Checkpoint:** all 30 episodes end in success, the dataset loads through `LeRobotDataset`, and the action percentiles fall inside `env.action_space` (print both to compare).

## Exercise 3 — Patch `sac.py` for two buffers [Build]

In this exercise you add the demonstration buffer and the batch-composition switch to Lesson 08's `sac.py`, producing the three training arms as flags on one file. An AI tool can draft the patch from the specification below; you should read every line of the sampling path, because that is where the mechanism lives.

The specification:

- `demo_buffer.py` loads the Hub dataset and converts its episodes into $(s, a, r, s', d)$ transitions from state observations. The reward is zero everywhere except the terminal success frame; if the dataset lacks rewards, document how you derived them. The conversion is deliberately explicit:
  ```python
  ds = LeRobotDataset("<you>/gymhil_pickcube_demos_v1")
  for ep in range(ds.num_episodes):
      frames = load_episode_frames(ds, ep)              # ordered by frame_index
      for t in range(len(frames) - 1):
          demo_buffer.add(s=frames[t]["observation.state"],
                          a=frames[t]["action"],
                          r=terminal_reward if t == len(frames) - 2 else 0.0,
                          s2=frames[t + 1]["observation.state"],
                          d=(t == len(frames) - 2))
  ```
- `sac.py` gains a `--batch-composition {online, preload, rlpd}` flag. Under `online`, batches come entirely from the online buffer. Under `preload`, the demonstrations are inserted into the single online buffer at the start of training and are subject to the same first-in-first-out eviction as everything else. Under `rlpd`, each batch is 128 transitions sampled from the demonstration buffer plus 128 sampled from the online buffer, concatenated after sampling; the two buffers are never merged. The file also gains `--critic-layernorm` (LayerNorm after each critic hidden layer) and per-step logging of both buffer sizes.
- `--env-id` points at the state-observation gym-hil task. Flatten the dictionary observation in `make_env` if needed, and verify the environment ID and observation shape against the gym-hil README for your version.
- The check: under `rlpd`, assert that every batch is exactly 128 plus 128, and that under a fixed seed the sampled indices are identical across two runs.

**✅ Checkpoint:** the composition assertion holds, and a 5k-step smoke run of each arm executes on `cpu` without shape errors.

## Exercise 4 — Demonstrations in the wrong action frame [Diagnose]

This exercise tests the conversion rather than the algorithm. When RLPD appears not to work in practice, the cause is usually that the demonstration actions are expressed differently from the environment's actions, so the critic is being trained on actions the policy could never produce. You will plant that mistake and predict its symptom.

1. In a copy of `demo_buffer.py`, convert the demonstration actions as absolute end-effector positions (or scale them by ten) instead of the environment's deltas.
2. Before running, write down the symptom you expect on a 20k-step `rlpd` run compared with `online`, and which single printout would have caught the problem before training started.
3. Run both for 20k steps with one seed. Compare the demonstration action percentiles against `env.action_space` for the broken and the correct conversion.

**✅ Checkpoint:** the broken run's `rlpd` curve is no better than `online`, and possibly worse; the percentile comparison is the diagnostic you name in `RESULTS.md`.

## Exercise 5 — Run the composition and LayerNorm grid [Predict → Run]

This exercise tests objectives 2 and 3: the sample-efficiency ordering of the three compositions, and the effect of LayerNorm. It is eight runs of 150k environment steps at a UTD of one.

1. Fix the metric before launching: environment steps until the rolling success rate over the last 20 episodes reaches 80 percent, plus the final success rate at 150k steps.
2. Before running, write down the ordering of `online`, `preload`, and `rlpd` by steps-to-80%; where you expect `preload`'s curve to bend toward `online`'s, using the numbers from Exercise 1; and what you expect `rlpd` without LayerNorm to do to the critic loss.
3. The arms are {`online`, `preload`, `rlpd`} × seeds {0, 1} with `--critic-layernorm`, plus `rlpd` without LayerNorm × seeds {0, 1}. Launch them from a bash loop over configs, with the W&B group set to the arm name.
4. Plot success against steps (mean with min and max over the two seeds, one panel per LayerNorm setting) and tabulate steps-to-80%, using ∞ for arms that never reach it.

**✅ Checkpoint:** `rlpd` reaches 80 percent in the fewest steps on both seeds; `preload` starts faster than `online` but its advantage shrinks late, which is the dilution crossover and should be annotated; and the arm without LayerNorm shows either a critic-loss blow-up or a collapse in success. If `preload` and `rlpd` are indistinguishable at your scale, say so and offer a hypothesis; 30 demonstrations against 150k steps may be too small a ratio for full dilution to appear. Two seeds support an ordering but not an effect size, and the writeup should say so.

## Exercise 6 — Plot the mechanism [Predict → Run]

This exercise tests objective 1 against the logs: the arithmetic from Exercise 1, drawn from what actually happened during training. Sketching the curves before plotting them turns the figure into a check on your derivation.

1. Before plotting, sketch the effective oversampling factor $|\mathcal{D}_{\text{online}}|/|\mathcal{D}_{\text{demo}}|$ over training, and beneath it the expert fraction of each batch for `preload` and for `rlpd`.
2. Plot both from the logged buffer sizes. For the LayerNorm ablation, also plot the critic's Q-value percentiles (p50 and p99) over training for both settings; divergence appears in the p99 curve well before the success curve dies.
3. Reconcile the figures with the curves from Exercise 5. The arm whose expert fraction diluted should be the arm that slowed.

**✅ Checkpoint:** the two mechanism figures agree with Exercise 1's table at the three tabulated points, and the Q-percentile figure explains the no-LayerNorm arm's curve.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| Hub dataset `<you>/gymhil_pickcube_demos_v1` | 30 successful episodes; the card states the control mode and fps |
| `demo_buffer.py` and the patched `sac.py` (from Lesson 08) | the 50/50 composition assertion; seeded; buffer sizes logged |
| `run_grid.sh` and `configs/` | reruns the 8-run grid from one command |
| `plots/` | efficiency curves (2 panels), the steps-to-80% table, the oversampling-factor and expert-fraction figure, the Q-percentile figure |
| `RESULTS.md` | the Exercise 1 table; the predictions for Exercises 4–6 with their reconciliations; the dilution crossover; the LayerNorm reading; the UTD caveat; at most 10 sentences of interpretation |

## Done when

- [ ] All 8 runs are complete and logged, and the steps-to-80% table has no empty cells (∞ is allowed).
- [ ] `rlpd` dominates `online` on both seeds by the pre-registered metric.
- [ ] The oversampling-factor figure exists, and `RESULTS.md` uses it to explain `preload`'s decay.
- [ ] You can state, with your own numbers, what LayerNorm bought, and the wrong-frame diagnosis names its printout.

## Self-check

1. For a demonstration transition at t=0 and at t=150k, compute its per-batch sampling probability under `preload` and under `rlpd` (30 demonstrations, about 3k transitions, online buffer of 150k), without looking at Exercise 1's table.
2. Why does RLPD not need a behaviour-cloning loss term to stay near the demonstrations early in training?
3. What failure mode does LayerNorm in the critic suppress, and why do demonstrations in particular trigger it early?
4. The paper runs a UTD of 20 with critic ensembles; you ran a UTD of 1 without them. What breaks if you raise the UTD without ensembles?
5. Where does this machinery reappear in Lesson 10, and what third data source joins the two buffers there?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Recorder starts but the robot won't move | intervention or control not engaged | keyboard: hold the enable key (spacebar per the docs); gamepad: hold RB |
| Demonstration transitions have reward 0 everywhere | the reward is sparse and appears only at the terminal success frame | verify against the gym-hil source; recompute the terminal reward when converting episodes |
| `rlpd` arm no better than `online` | demonstration actions in a different frame from the environment's (end-effector deltas versus absolutes) | compare the demonstration action percentiles with `env.action_space`; fix the conversion, not the algorithm (Exercise 4) |
| Critic loss explodes in every arm | the UTD is effectively above 1 (multiple updates per step by accident) | log the update-to-step counter; it should be exactly 1 |
| Runs not reproducible across machines | the environment's physics timestep differs between MuJoCo versions | pin `mujoco` and `gym-hil` versions in the config committed with the run |
| `sac` module paths from the tutorial don't exist | LeRobot v0.6 renamed `sac` to `gaussian_actor` and rebuilt the RL stack | you are using Lesson 08's `sac.py`; for any LeRobot import, check what `python -c "import lerobot.rl"` exposes first |

## Going deeper

- **REDQ.** Raise the UTD to 4 and replace the two critics with an ensemble of five sampled at random in each update, which is the technique RLPD borrows. Measure whether the reduction in steps-to-80% pays for the extra compute.
- **Pixels.** Swap the state observations for images (`image_obs=True`) and report how the ranking changes when the critic must also learn to see.
- **Three seeds.** Rerun the grid with three seeds and report whether the two-seed ordering survived.

## References

- Ball, Smith, Kostrikov, Levine. *Efficient Online Reinforcement Learning with Offline Data*, ICML 2023. arXiv:2302.02948.
- LeRobot team, *Robot Learning: A Tutorial*, §3.2. arXiv:2510.12403.
- gym-hil: github.com/huggingface/gym-hil; LeRobot "Train RL in Simulation" docs (the config schema quoted above).
- Chen et al. 2021 (REDQ), arXiv:2101.05982, for the UTD and ensemble machinery in Going deeper.
