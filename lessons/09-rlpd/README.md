# Lesson 09 — Sample-Efficient, Data-Driven RL: RLPD

Measure the mechanism that makes RL viable on real robots — a second buffer of demonstrations sampled 50/50 with online data — by deriving its oversampling arithmetic first, then watching preloading dilute and symmetric sampling not.

| | |
|---|---|
| **Phase** | 3 — Reinforcement learning |
| **Time** | ~4–5 h desk time (AI-assisted) + ~2–3 h compute for the run grid (Mac for demos, cloud GPU for the grid) |
| **Cost** | ~$2–3 (8 short training runs; batchable on one 4090) |
| **Prerequisites** | 08 (`sac.py` is the file being patched), 02 (you can read a LeRobotDataset) |
| **Feeds into** | 10 (HIL-SERL adds a classifier and humans to exactly this machinery), H5 (same recipe on the real arm) |

## Learning objectives

After this lesson you can:

1. **Derive** the effective oversampling factor of a demo transition under symmetric 50/50 sampling, and predict how it evolves as the online buffer grows.
2. **Quantify** the sample-efficiency gap between online-only SAC, demo-preloading, and RLPD on the same task with matched seeds.
3. **Diagnose** critic divergence under off-distribution actions and show what LayerNorm changes.
4. **Record** demonstrations in `gym-hil` through LeRobot's config-driven recorder into a Hub-hosted LeRobotDataset.
5. **Defend** why RLPD deliberately drops the trust-region/BC-penalty machinery of prior offline-to-online methods.

## Principles

**The problem.** Online SAC from scratch spends most of its samples discovering that random flailing doesn't pick up cubes. On hardware, samples are wall-clock minutes and gripper wear. The obvious fix — put $N$ demos in the replay buffer before training ("preloading") — decays: once the online buffer holds 100k transitions, 1k demo transitions are sampled with probability 1%, and the demos' signal drowns.

**RLPD's answer** (Ball et al. 2023): keep demos in their own buffer $\mathcal{D}_{\text{demo}}$ and build every gradient batch as 50% $\mathcal{D}_{\text{demo}}$, 50% $\mathcal{D}_{\text{online}}$, forever. A demo transition's per-batch sampling probability is $\frac{0.5}{|\mathcal{D}_{\text{demo}}|}$ vs $\frac{0.5}{|\mathcal{D}_{\text{online}}|}$ for an online one — an effective oversampling factor of $|\mathcal{D}_{\text{online}}|/|\mathcal{D}_{\text{demo}}|$ that *grows* through training. That's the whole trick: the expert signal never dilutes.

**The supporting cast**, each there for a measurable reason:
- *LayerNorm in the critics.* Off-distribution actions (and demos are off-distribution for an early policy) let an unnormalized critic extrapolate arbitrarily large Q-values; the actor then chases them and the critic diverges. LayerNorm bounds the extrapolation. This is the ablation you'll run.
- *High UTD (update-to-data) ratio.* RLPD runs up to 20 gradient steps per env step to wring the buffer dry — with ensembles to keep the critics honest. You'll use UTD 1 and note the full recipe.
- *No pessimism, no BC penalty.* Unlike offline-RL hybrids, RLPD trusts symmetric sampling + normalization to do the stabilizing. The paper's bet is that simplicity wins online — your curves test it at small scale.

**Task.** `gym-hil`'s Franka pick-cube (`PandaPickCubeBase-v0` family): end-effector delta actions via built-in IK, sparse success reward, 10 Hz control (per LeRobot's example configs). Verify the reward spec against the gym-hil source for your installed version — sparse-vs-shaped changes where the demos matter most.

**Carry forward**

- Preloading dilutes as $|\mathcal{D}_{\text{online}}|$ grows; symmetric sampling keeps the demo share fixed at 50% of every batch.
- Effective oversampling of a demo transition = $|\mathcal{D}_{\text{online}}|/|\mathcal{D}_{\text{demo}}|$, growing through training.
- Critic LayerNorm bounds Q-extrapolation on off-distribution actions; demos trigger that failure earliest.
- Demo-to-transition conversion (reward placement, action frame) is where the silent bugs live, not the algorithm.

| Source | Read for |
|---|---|
| Tutorial §3.2 ("sample-efficient, data-driven RL") | how the tutorial frames RLPD as the bridge from sim RL to HIL-SERL |
| Ball et al. 2023, arXiv:2302.02948, §4 + ablation tables | which components the authors found load-bearing (LayerNorm, symmetric sampling) vs optional (ensembles at low UTD) |
| gym-hil README + LeRobot RL-in-sim docs page | env IDs, the JSON config schema, intervention/record mechanics you'll reuse in Lesson 10 |

## Exercise 1 — The oversampling arithmetic [Derive]

Tests objective 1, before any run: the number the whole lesson turns on.

1. With 30 demos ≈ 3k transitions and batch 256: compute a demo transition's per-batch sampling probability under `preload` and under `rlpd` at $|\mathcal{D}_{\text{online}}| = 0$, 30k, and 150k. Tabulate.
2. Compute the fraction of each batch that is expert data under `preload` at the same three points.
3. Write one sentence predicting which arm's advantage shrinks late and by how much.

**✅ Checkpoint:** the six-cell table is in `RESULTS.md` before Exercise 5 starts; `rlpd`'s expert fraction is 0.5 in every cell.

## Exercise 2 — Record 30 demos [Build]

Produces `<you>/gymhil_pickcube_demos_v1` on the Hub; also H5's rehearsal.

1. `pip install -e ".[hilserl]"` in your LeRobot checkout (installs `gym_hil`).
2. Copy LeRobot's example env config (`huggingface.co/datasets/lerobot/config_examples`, `rl/gym_hil/env_config.json`) and set:
   ```json
   "env":  { "type": "gym_manipulator", "name": "gym_hil",
             "task": "PandaPickCubeKeyboard-v0", "fps": 10 },
   "dataset": { "repo_id": "<you>/gymhil_pickcube_demos_v1", "task": "pick_cube",
                "num_episodes_to_record": 30, "push_to_hub": true },
   "mode": "record"
   ```
   (Gamepad task variant if you have one — markedly easier than keyboard.)
3. Run `mjpython -m lerobot.rl.gym_manipulator --config_path <cfg>` and record 30 *successful* episodes; discard failures at record time.
4. Inspect in the LeRobot visualizer; confirm action space (EE deltas + gripper) and episode lengths (~50–150 steps).

**✅ Checkpoint:** 30/30 episodes end in success; the dataset loads via `LeRobotDataset` and action percentiles sit inside `env.action_space` (print both).

## Exercise 3 — Patch `sac.py` for two buffers [Build]

Produces the three training arms as flags on Lesson 08's file. Spec for the patch (an AI tool drafts it; you read every line of the sampling path):

- `demo_buffer.py`: loads the Hub dataset and converts episodes to $(s, a, r, s', d)$ transitions from state observations. Reward: 0 except the terminal success frame; document what you did if the dataset lacks rewards. The conversion, made boring and explicit:
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
- `sac.py` gains `--batch-composition {online, preload, rlpd}`: `online` = 100% online buffer; `preload` = demos inserted into the single online buffer at t=0 (FIFO-evictable); `rlpd` = each batch is 128 from the demo buffer + 128 from the online buffer, concatenated *after* sampling — never merged into one buffer. Also `--critic-layernorm` (LayerNorm after each critic hidden layer) and per-step logging of both buffer sizes.
- `--env-id` pointing at the state-observation gym-hil task (flatten the dict observation in `make_env` if needed; verify the env ID and obs shape against the gym-hil README for your version).
- The check: under `rlpd`, assert every batch is exactly 128 + 128; under a fixed seed, sampled indices are identical across two runs.

**✅ Checkpoint:** the composition assert holds; one 5k-step smoke run per arm executes on `cpu` without shape errors.

## Exercise 4 — Demos in the wrong frame [Diagnose]

Tests the conversion, not the algorithm — where RLPD "doesn't work" in practice.

1. In a copy of `demo_buffer.py`, convert demo actions as absolute EE positions (or scale them ×10) instead of the env's deltas.
2. **Write first:** the symptom on a 20k-step `rlpd` run vs `online` — and which single printout would have caught it before training.
3. Run both for 20k steps, one seed. Compare demo action percentiles to `env.action_space` for the broken and correct conversions.

**✅ Checkpoint:** the broken run's `rlpd` curve is no better than `online` (or worse); the percentile comparison is the diagnostic you name in `RESULTS.md`.

## Exercise 5 — The grid [Predict → Run]

Tests objectives 2–3: the efficiency ordering and the LayerNorm effect. Eight runs, 150k env steps each, UTD 1.

1. Metric, fixed before launch: env steps until rolling success rate (last 20 episodes) ≥ 80%, plus final success at 150k.
2. **Write first:** the ordering of `online`, `preload`, `rlpd` by steps-to-80%; where `preload`'s curve crosses toward `online`'s (from Exercise 1); and what `rlpd` without LayerNorm does to the critic loss.
3. Arms: {`online`, `preload`, `rlpd`} × seeds {0, 1} with `--critic-layernorm`, plus `rlpd` without LayerNorm × seeds {0, 1}. A bash loop over configs; W&B group = arm name.
4. Plot success-vs-steps (mean ± min/max over 2 seeds, one panel per LayerNorm setting); tabulate steps-to-80% (∞ if never reached).

**✅ Checkpoint (expected shape):** `rlpd` reaches 80% in the fewest steps on both seeds; `preload` starts faster than `online` but its advantage shrinks late (the dilution crossover — annotate it); the no-LayerNorm arm shows critic-loss blow-up or a success collapse. If `preload` ≈ `rlpd` at your scale, say so and hypothesize why (30 demos vs 150k steps may be too small a ratio for full dilution). Two seeds support an ordering, not an effect size — state that.

## Exercise 6 — The mechanism figure [Predict → Run]

Tests objective 1 against the logs: Exercise 1's arithmetic, plotted from what actually happened.

1. **Write first:** sketch the effective oversampling factor $|\mathcal{D}_{\text{online}}|/|\mathcal{D}_{\text{demo}}|$ over training, and under it the expert fraction of each batch for `preload` vs `rlpd`.
2. Plot both from the logged buffer sizes. For the LayerNorm ablation, plot critic Q-value percentiles (p50/p99) over training, both settings — divergence shows in p99 long before the success curve dies.
3. Reconcile: the arm that diluted is the arm that slowed.

**✅ Checkpoint:** the two mechanism figures match Exercise 1's table at the three points; the Q-percentile figure explains the no-LayerNorm arm's curve.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| Hub dataset `<you>/gymhil_pickcube_demos_v1` | 30 successful episodes; card states control mode + fps |
| `demo_buffer.py` + patched `sac.py` (from Lesson 08) | 50/50 composition assert; seeded; buffer sizes logged |
| `run_grid.sh` + `configs/` | reruns the 8-run grid with one command |
| `plots/` | efficiency curves (2 panels), steps-to-80% table, oversampling-factor + expert-fraction figure, Q-percentile figure |
| `RESULTS.md` | Exercise 1 table; Exercises 4–6 predictions with reconciliations; the dilution crossover; the LayerNorm reading; UTD caveat stated; ≤ 10 sentences of interpretation |

## Done when

- [ ] 8/8 runs completed and logged; steps-to-80% table has no empty cells (∞ allowed).
- [ ] `rlpd` dominates `online` on both seeds by the pre-registered metric.
- [ ] The oversampling-factor figure exists and `RESULTS.md` uses it to explain `preload`'s decay.
- [ ] You can state, with your own numbers, what LayerNorm bought — and the wrong-frame diagnosis names its printout.

## Self-check

1. A demo transition at t=0 vs t=150k: compute its per-batch sampling probability under `preload` and under `rlpd` (30 demos ≈ 3k transitions, online buffer 150k). Do it without Exercise 1's table.
2. Why does RLPD *not* need a BC loss term to stay near the demos early in training?
3. What failure mode does LayerNorm in the critic suppress, and why do *demos* specifically trigger it early?
4. The paper runs UTD 20 with critic ensembles; you ran UTD 1 without. What breaks if you raise UTD without ensembles?
5. Where does this exact machinery reappear in Lesson 10, and what third data source joins the two buffers there?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Recorder starts but robot won't move | intervention/control not engaged | keyboard: hold the enable key (spacebar per docs); gamepad: hold RB |
| Demo transitions have reward 0 everywhere | sparse reward only at terminal success frame | verify against gym-hil source; recompute terminal reward when converting episodes |
| `rlpd` arm no better than `online` | demo actions in a different frame than env actions (EE deltas vs absolutes) | compare demo action percentiles to `env.action_space`; fix the conversion, not the algorithm (Exercise 4) |
| Critic loss explodes in *all* arms | UTD effectively > 1 (multiple updates per step by accident) | log the update:step counter; it should be exactly 1 |
| Runs not reproducible across machines | env physics timestep differs with MuJoCo version | pin `mujoco` and `gym-hil` versions in the config committed with the run |
| `sac` module paths from the tutorial don't exist | LeRobot v0.6 renamed `sac` → `gaussian_actor` and rebuilt the RL stack | you're using Lesson 08's `sac.py` anyway; for any LeRobot import, check `python -c "import lerobot.rl"` surfaces first |

## Going deeper

- **REDQ.** Raise UTD to 4 and add a 2-critic → 5-critic ensemble with random subsetting (the trick RLPD borrows); measure whether steps-to-80% drops enough to pay for the compute.
- **Pixels.** Swap state observations for pixels (`image_obs=True`) and report how the ranking changes when the critic must also learn to see.
- **Three seeds.** Rerun the grid at 3 seeds and report whether the 2-seed ordering survived.

## References

- Ball, Smith, Kostrikov, Levine. *Efficient Online Reinforcement Learning with Offline Data*, ICML 2023. arXiv:2302.02948.
- LeRobot team, *Robot Learning: A Tutorial*, §3.2. arXiv:2510.12403.
- gym-hil: github.com/huggingface/gym-hil; LeRobot "Train RL in Simulation" docs (config schema quoted above).
- Chen et al. 2021 (REDQ), arXiv:2101.05982 — the UTD/ensemble machinery (Going deeper).
