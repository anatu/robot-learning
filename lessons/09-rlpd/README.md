# Lesson 09 — Sample-Efficient, Data-Driven RL: RLPD

Ablate the mechanism that makes RL viable on real robots — a second buffer of demonstrations sampled 50/50 with online data — so the design choices in every SERL-style system stop being folklore and become measured effects.

| | |
|---|---|
| **Phase** | 3 — Reinforcement learning |
| **Time** | ~2 sessions desk time (6–8 h) + ~4–6 h total compute for the run grid (Mac for demos, cloud GPU for the grid) |
| **Cost** | ~$3–6 (18 short training runs; batchable on one 4090) |
| **Prerequisites** | 08 (`SACAgent` is the engine being modified), 02 (you can read/write LeRobotDataset) |
| **Feeds into** | 10 (HIL-SERL adds a classifier and humans to exactly this machinery), H5 (same recipe on the real arm) |

## Learning objectives

After this lesson you can:

1. **Explain** symmetric 50/50 sampling as an importance-weighting scheme, and compute the effective oversampling factor of a demo transition as buffers grow.
2. **Quantify** the sample-efficiency gap between online-only SAC, demo-preloading, and RLPD on the same task with matched seeds.
3. **Diagnose** critic divergence under off-distribution actions and show what LayerNorm changes.
4. **Record** demonstrations in `gym-hil` through LeRobot's config-driven recorder into a Hub-hosted LeRobotDataset.
5. **Defend** why RLPD deliberately drops the trust-region/BC-penalty machinery of prior offline-to-online methods.

## Background

**The problem.** Online SAC from scratch spends most of its samples discovering that random flailing doesn't pick up cubes. On hardware, samples are wall-clock minutes and gripper wear. The obvious fix — put $N$ demos in the replay buffer before training ("preloading") — decays: once the online buffer holds 100k transitions, 1k demo transitions are sampled with probability 1%, and the demos' signal drowns.

**RLPD's answer** (Ball et al. 2023): keep demos in their own buffer $\mathcal{D}_{\text{demo}}$ and build every gradient batch as 50% $\mathcal{D}_{\text{demo}}$, 50% $\mathcal{D}_{\text{online}}$, forever. A demo transition's per-batch sampling probability is $\frac{0.5}{|\mathcal{D}_{\text{demo}}|}$ vs $\frac{0.5}{|\mathcal{D}_{\text{online}}|}$ for an online one — an effective oversampling factor of $|\mathcal{D}_{\text{online}}|/|\mathcal{D}_{\text{demo}}|$ that *grows* through training. That's the whole trick: the expert signal never dilutes.

**The supporting cast**, each there for a measurable reason:
- *LayerNorm in the critics.* Off-distribution actions (and demos are off-distribution for an early policy) let an unnormalized critic extrapolate arbitrarily large Q-values; the actor then chases them and the critic diverges. LayerNorm bounds the extrapolation. This is the ablation you'll run.
- *High UTD (update-to-data) ratio.* RLPD runs up to 20 gradient steps per env step to wring the buffer dry — with ensembles to keep the critics honest. You'll use a modest UTD and note the full recipe.
- *No pessimism, no BC penalty.* Unlike offline-RL hybrids, RLPD trusts symmetric sampling + normalization to do the stabilizing. The paper's bet is that simplicity wins online — your curves will test it at small scale.

**Task.** `gym-hil`'s Franka pick-cube (`PandaPickCubeBase-v0` family): end-effector delta actions via built-in IK, sparse success reward, 10 Hz control (per LeRobot's example configs). Verify the reward spec against the gym-hil source for your installed version — sparse-vs-shaped changes where the demos matter most.

| Source | Read for |
|---|---|
| Tutorial §3.2 ("sample-efficient, data-driven RL") | how the tutorial frames RLPD as the bridge from sim RL to HIL-SERL |
| Ball et al. 2023, arXiv:2302.02948, §4 + ablation tables | which components the authors found load-bearing (LayerNorm, symmetric sampling) vs optional (ensembles at low UTD) |
| gym-hil README + LeRobot RL-in-sim docs page | env IDs, the JSON config schema, intervention/record mechanics you'll reuse in Lesson 10 |

## Part 1 — Record 30 demos (Mac, ~1 h)

Produces `<you>/gymhil_pickcube_demos_v1` on the Hub.

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
4. Inspect the dataset in the LeRobot visualizer; confirm action space (EE deltas + gripper) and episode lengths (~50–150 steps).

**✅ Checkpoint:** 30/30 episodes end in success; dataset loads via `LeRobotDataset` and actions are in the range the env expects (print `env.action_space` and compare percentiles).

## Part 2 — Wire demos into SAC (Mac, ~3 h)

Produces the three training arms as one flag on Lesson 08's agent.

1. Write `DemoBuffer`: loads the Hub dataset, converts episodes to $(s, a, r, s', d)$ transitions (state observations; re-derive rewards from the env's success criterion if the dataset lacks them — document what you did). The conversion is where most silent bugs live; make it boring and explicit:
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
2. Extend `SACAgent.update()` with a `batch_composition` switch: `online` (100% online), `preload` (demos inserted into the single online buffer at t=0), `rlpd` (each batch: 128 sampled from the demo buffer + 128 from the online buffer, concatenated *after* sampling — never merged into one buffer).
3. Unit tests: batch composition is exactly 50/50 under `rlpd`; preloaded transitions are FIFO-evictable; seeds fix the sampled indices.

**✅ Checkpoint:** tests green; one 5k-step smoke run per arm executes without shape errors on `cpu`.

## Part 3 — The grid (cloud GPU, ~4 h wall-clock)

The experiment: 3 compositions × LayerNorm {on, off} × 3 seeds = 18 runs, 150k env steps each, UTD 1 (note in `RESULTS.md` that the paper's full recipe is UTD 20 + ensembles).

1. Metric, fixed before launch: env steps until rolling success rate (last 20 episodes) ≥ 80%, plus final success at 150k.
2. Run the grid (a simple bash loop over configs; log to W&B with group = arm name).
3. Plot success-vs-steps, one panel per LayerNorm setting, three curves each (mean ± min/max over seeds).
4. Compute and tabulate steps-to-80% per arm (∞ if never reached).

**✅ Checkpoint (expected shape):** `rlpd` reaches 80% in the fewest steps on every seed; `preload` starts faster than `online` but its advantage shrinks late (that's dilution — annotate the crossover); with LayerNorm off, at least one arm shows critic loss blow-up or a success collapse. If `preload` ≈ `rlpd` at your scale, say so honestly and hypothesize why (30 demos vs 150k steps may be too small a ratio for full dilution).

## Part 4 — Read the mechanism (desk, ~1 h)

1. From logged buffer sizes, plot the *effective oversampling factor* $|\mathcal{D}_{\text{online}}|/|\mathcal{D}_{\text{demo}}|$ over training, and under it the fraction of each batch that is expert data for `preload` vs `rlpd`. This pair of curves is the lesson's thesis in one figure.
2. For the LayerNorm ablation: plot critic Q-value percentiles (p50/p99) over training, both settings. Divergence, if you got it, is visible in p99 long before the success curve dies.

**✅ Checkpoint:** the two mechanism figures exist and match the curves from Part 3 (the arm that diluted is the arm that slowed).

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| Hub dataset `<you>/gymhil_pickcube_demos_v1` | 30 successful episodes, card states control mode + fps |
| `demo_buffer.py` + tests | 50/50 composition property-tested; seeded |
| `configs/` + `run_grid.sh` | reruns the full 18-run grid with one command |
| `plots/` | efficiency curves (2 panels), steps-to-80% table, oversampling-factor figure, Q-percentile figure |
| `RESULTS.md` | the table; the dilution crossover; the LayerNorm reading; ≤ 10 sentences; UTD caveat stated |

## Done when

- [ ] 18/18 runs completed and logged; steps-to-80% table has no empty cells (∞ allowed).
- [ ] RLPD dominates online-only on all 3 seeds by the pre-registered metric.
- [ ] The oversampling-factor figure exists and `RESULTS.md` uses it to explain preload's decay.
- [ ] You can state, with your own numbers, what LayerNorm bought.

## Self-check

1. A demo transition at t=0 vs t=150k: compute its per-batch sampling probability under `preload` and under `rlpd` (30 demos ≈ 3k transitions, online buffer 150k).
2. Why does RLPD *not* need a BC loss term to stay near the demos early in training?
3. What failure mode does LayerNorm in the critic suppress, and why do *demos* specifically trigger it early?
4. The paper runs UTD 20 with critic ensembles; you ran UTD 1 without. What breaks if you raise UTD without ensembles?
5. Where does this exact machinery reappear in Lesson 10, and what third data source joins the two buffers there?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Recorder starts but robot won't move | intervention/control not engaged | keyboard: hold the enable key (spacebar per docs); gamepad: hold RB |
| Demo transitions have reward 0 everywhere | sparse reward only at terminal success frame | verify against gym-hil source; recompute terminal reward when converting episodes |
| `rlpd` arm no better than `online` | demo actions in a different frame than env actions (EE deltas vs absolutes) | compare demo action percentiles to `env.action_space`; fix the conversion, not the algorithm |
| Critic loss explodes in *all* arms | UTD effectively > 1 (multiple updates per step by accident) | log the update:step counter; it should be exactly 1 |
| Runs not reproducible across machines | env physics timestep differs with MuJoCo version | pin `mujoco` and `gym-hil` versions in the config committed with the run |
| `sac` module paths from the tutorial don't exist | LeRobot v0.6 renamed `sac` → `gaussian_actor` and rebuilt the RL stack | you're using your own SAC anyway; for any LeRobot import, check `python -c "import lerobot.rl"` surfaces first |

## Stretch

Raise UTD to 4 and add a 2-critic → 5-critic ensemble with random subsetting (the REDQ trick RLPD borrows); measure whether steps-to-80% drops enough to pay for the compute. Or swap state observations for pixels (`image_obs=True`) and report how the ranking changes when the critic must also learn to see.

## References

- Ball, Smith, Kostrikov, Levine. *Efficient Online Reinforcement Learning with Offline Data*, ICML 2023. arXiv:2302.02948.
- LeRobot team, *Robot Learning: A Tutorial*, §3.2. arXiv:2510.12403.
- gym-hil: github.com/huggingface/gym-hil; LeRobot "Train RL in Simulation" docs (config schema quoted above).
- Chen et al. 2021 (REDQ), arXiv:2101.05982 — the UTD/ensemble machinery (stretch).
