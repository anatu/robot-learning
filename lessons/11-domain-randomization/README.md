# Lesson 11 — Domain Randomization & the Reality Gap

Make the tutorial's zero-code survey of domain randomization empirical: sketch the transfer heatmaps before training, train policies across randomization widths, and see both canonical failure modes in your own data — too little entropy fails to transfer, too much hurts the nominal task.

| | |
|---|---|
| **Phase** | 3 — Reinforcement learning |
| **Time** | ~4–5 h desk time (AI-assisted) + ~3–4 GPU-hours for the sweep |
| **Cost** | ~$3–5 (3 widths × 2 seeds, short runs; evaluation is cheap) |
| **Prerequisites** | 08 (`sac.py` trains every arm unchanged) |
| **Feeds into** | 18–19 (why sim benchmarks and real transfer diverge), H3 (interpreting the ID/OOD gap on the real arm), 22 (the sim-to-real capstone option) |

## Learning objectives

After this lesson you can:

1. **Formalize** DR as training on $p(s'|s,a;\xi)$, $\xi \sim \Xi$, and state precisely what distribution the resulting policy is optimal for.
2. **Diagnose** the compounding-multiplier bug from its drift signature, and state the invariant a randomization wrapper must hold across resets.
3. **Predict** the shape of a train-width × test-dynamics transfer heatmap before training, then produce it with enough seeds to trust its ordering.
4. **Demonstrate** both failure modes — no-transfer at zero width, nominal-performance loss at excessive width — in your own data.
5. **Decide** a randomization width from the nominal-vs-off-nominal tradeoff, and explain why frontier labs largely collect real data instead.

## Principles

**The formalism** (tutorial §3.2.2). A simulator is a dynamics family $\mathcal{D}_\xi$ indexed by physical parameters $\xi$ (masses, frictions, damping, delays). DR trains one policy on the mixture: maximize $\mathbb{E}_{\xi \sim \Xi}\,\mathbb{E}_{\tau \sim \pi, \mathcal{D}_\xi}[R(\tau)]$. The bet: if $\Xi$ is wide enough to contain (or surround) the real world's $\xi^*$, a policy robust across $\Xi$ transfers. The cost: the policy optimizes an *average* over dynamics, so width is not free — at high width the optimum is a conservative behavior that underperforms everywhere, and with a memoryless policy the best it can do is play the average dynamics rather than identify the current ones.

**What the knobs mean.** A width-$w$ arm samples each randomized parameter log-uniformly in $[\xi_{\text{nom}}/w, \; \xi_{\text{nom}} \cdot w]$ per episode. $w{=}1$ is no randomization; $w{=}4$ spans a 16× range. Log-uniform, because mass and friction act multiplicatively on dynamics.

**The invariant.** Multipliers are drawn per episode and applied to the *pristine* model values, never to the current ones. Apply them to current values and the dynamics random-walk across episodes — the classic silent bug, and Exercise 2's subject.

**The automation layer.** Fixed widths are guesses, so the field automated them: AutoDR (Akkaya et al. 2019, the Rubik's-cube system) samples at the boundary of the current range per parameter and *expands* a bound when boundary performance clears a threshold, shrinks when it fails — curriculum via bracketing. DORAEMON (Tiboni et al. 2024) states the objective cleanly: maximize the *entropy* of $\Xi$ subject to a success-rate constraint. Going deeper has the minimal version of both.

**The punchline:** DR buys robustness to the parameters you randomize, priced in nominal performance and training compute — and it does nothing for the parameters you didn't think to randomize. That's the argument (tutorial §3.2.2 → §5) for why π-style labs spend on real data collection instead of ever-wider $\Xi$.

**Carry forward**

- DR optimizes an average over $\Xi$; the width $w$ trades nominal performance for coverage.
- Randomize from pristine values, per episode, by name (`mj_name2id`), and log $\xi$.
- The heatmap's two failure modes: a bright island at $w{=}1$ that dies off-nominal; a flat map at large $w$ whose nominal cell has dropped.
- DR is blind to parameters you didn't randomize; real data isn't.

| Source | Read for |
|---|---|
| Tutorial §3.2.2 | the $\mathcal{D}_\xi$, $\xi \sim \Xi$ formalism; where DR sits in the sim-to-real argument |
| Tobin et al. 2017, arXiv:1703.06907 | visual DR origins — note it randomizes *appearance*, you randomize *dynamics*; keep the distinction crisp |
| Akkaya et al. 2019, arXiv:1910.07113, §5 | ADR's expand/shrink rule and boundary sampling |
| Tiboni et al. 2024 (DORAEMON) | the entropy-maximization framing; read the objective, skim the rest |

## Exercise 1 — The randomization wrapper [Build]

Produces the controlled experimental substrate. Env: `Pusher-v5` (`gymnasium[mujoco]`) — 7-DOF arm pushes a cylinder to a goal; success is metric and objective, no classifier needed. Success = final object-to-goal distance < 0.05 m; also log return.

Spec for `dr_wrapper.py` (`DynamicsRandomizationWrapper(env, width, seed)`; an AI tool drafts it):
- on each `reset()`, draw multipliers log-uniformly in $[1/w, w]$ and apply to (a) the object's `body_mass` and (b) the object–table sliding friction (`geom_friction[:, 0]` of the relevant geoms);
- resolve body/geom IDs by name via `mj_name2id` — never hardcode indices; print the model's name table first, the body name below is illustrative;
- restore pristine model values before each draw;
- expose the drawn $\xi$ in `info["dynamics"]`;
- a `pin(xi)` method that fixes $\xi$ for evaluation instead of sampling.

The intended structure:
```python
class DynamicsRandomizationWrapper(gym.Wrapper):
    def __init__(self, env, width, seed):
        super().__init__(env)
        m = env.unwrapped.model
        self._obj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
        self._nominal_mass = m.body_mass[self._obj].copy()      # pristine snapshot
        self._rng = np.random.default_rng(seed)
        self._w = width

    def reset(self, **kw):
        m = self.env.unwrapped.model
        mult = np.exp(self._rng.uniform(-np.log(self._w), np.log(self._w)))
        m.body_mass[self._obj] = self._nominal_mass * mult      # from pristine, not current
        obs, info = self.env.reset(**kw)
        info["dynamics"] = {"mass_mult": float(mult)}           # friction analogous
        return obs, info
```
The check (`check_wrapper.py`): multipliers within $[1/w, w]$ over 100 resets; identical draws under the same seed; $w{=}1$ is bit-identical to the raw env; `pin()` holds across resets.

Then calibrate difficulty: wrap into Lesson 08's `sac.py` `make_env` and run $w{=}1$ briefly (Pusher solves in ~200–300k steps).

**✅ Checkpoint:** `check_wrapper.py` passes; a 10-reset log shows $\xi$ draws spanning the intended range; nominal SAC shows clear learning progress by 100k steps.

## Exercise 2 — Compounding multipliers [Diagnose]

Tests objective 2: the bug that makes "DR didn't help" a wrapper error.

1. In a copy of the wrapper, apply each draw to the *current* `body_mass` instead of the pristine snapshot.
2. **Write first:** the trajectory of the effective mass multiplier over 10 resets at $w{=}2$ — its expected value, its spread, and why (what random walk is this in log space?).
3. Run 10 resets with each wrapper and plot the effective multiplier per reset; then state the check in `check_wrapper.py` that catches the bug (two consecutive resets must not compound).

**✅ Checkpoint:** the broken wrapper's log-multiplier random-walks (variance growing with reset count) while the correct one stays i.i.d. in $[1/w, w]$; the catching assert is in the check script.

## Exercise 3 — Pre-register the evaluation grid [Write]

Tests the discipline: the test set exists before the training runs, and git proves it.

1. Held-out grid: mass-multiplier × friction-multiplier, each 5 log-spaced points in $[0.25, 4]$ → 25 cells. For every trained policy: 10 episodes per cell with the wrapper pinned to that cell's $\xi$, success rate per cell.
2. Write `configs/eval_grid.yaml` with the cell list, episode count, and seed list; commit it.
3. Write the two scalar summaries you will extract across $w$: nominal-cell success (the center cell) and mean off-nominal success (all cells outside $[1/2, 2]$ on either axis).

**✅ Checkpoint:** `configs/eval_grid.yaml` is committed, and its commit predates every training run in the git log.

## Exercise 4 — The sweep and the heatmaps [Predict → Run]

Tests objectives 3–4: the two failure modes, sketched first.

1. **Write first**, in `RESULTS.md`: a hand sketch of the three heatmaps ($w \in \{1.0, 2.0, 4.0\}$) — where the bright region sits, how it widens, and whether the nominal cell at $w{=}4$ is dimmer than at $w{=}1$. Then sketch the two-curve plot (nominal-cell success and mean off-nominal success vs $w$) and mark where you expect them to cross.
2. Arms: $w \in \{1.0, 2.0, 4.0\}$ × seeds {0, 1}, identical SAC config (Lesson 08's `sac.py`, `--env-id Pusher-v5`, the wrapper injected in `make_env`), identical step budget (300k). Log $\xi$ per episode. `run_sweep.sh` launches all six and then the pinned evaluation over `configs/eval_grid.yaml`.
3. Render one heatmap per arm (average the 2 seeds; keep per-seed maps in an appendix folder). Same color scale across arms — the comparison *is* the figure.
4. Extract the two scalar curves and plot both against $w$ on one axis pair.
5. Reconcile against the sketch.

**✅ Checkpoint (expected shape):** the $w{=}1$ map is a bright island at nominal that dies off-diagonal (low-entropy fails); maps widen with $w$; at $w{=}4$ the map is flatter but the nominal cell has measurably dropped from its $w{=}1$ peak (over-randomization hurts). The two-curve plot crosses. If no arm shows the nominal drop, your task is too easy for the widths chosen; add $w{=}6$ and say so. Ten episodes per cell is a binomial with ±~15% at p=0.5 — read the ordering of maps, not single cells.

## Exercise 5 — The width you'd ship [Decide]

Tests objective 5: a number with a defense.

1. From the two-curve plot, pick the $w$ you would train at for a real-arm deployment whose true mass and friction you can estimate to within 2×. State the nominal-performance price you pay and the off-nominal coverage you buy, with the numbers from your table.
2. One paragraph on the blind spot: name two parameters of a real pusher you did *not* randomize and what your heatmap can say about them (nothing). Tie to §5: when is widening $\Xi$ the right call versus collecting real data? Cost per real episode is the hinge.

**✅ Checkpoint:** the decision names a $w$, both prices, and the two un-randomized parameters.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `dr_wrapper.py` + `check_wrapper.py` | the four properties + the compounding assert, passing |
| `configs/eval_grid.yaml` + `run_sweep.sh` | full sweep + eval grid reproducible in one command; grid committed before training (git history is the proof) |
| `plots/` | 3 heatmaps on a shared scale; the two-curve tradeoff plot; the compounding-multiplier random-walk figure |
| `RESULTS.md` | the heatmap sketch and its reconciliation; both failure modes annotated on your own figures; the width decision with numbers; the "why real data instead" paragraph; ≤ 12 sentences |

## Done when

- [ ] The transfer heatmap set exists with 2 seeds averaged and a shared color scale.
- [ ] Both failure modes are visible in *your* data and annotated against your sketch.
- [ ] The eval grid predates the training runs in git history.
- [ ] The compounding-multiplier bug is diagnosed by mechanism and caught by a committed check.
- [ ] A width is chosen and defended with numbers.

## Self-check

1. Write the DR training objective and mark exactly where the width $w$ enters it.
2. Why log-uniform rather than uniform for mass/friction multipliers?
3. A memoryless policy trained at $w{=}4$ behaves conservatively. What architectural change would let a policy *adapt* to the episode's dynamics instead, and what information does it exploit?
4. Your heatmap shows transfer along the mass axis but a cliff along friction. What does that tell you about which parameter the task is actually sensitive to — and about DR's blind spot for parameters you didn't randomize?
5. Steelman the opposite bet: when is widening $\Xi$ the *right* call versus collecting real data?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Dynamics drift monotonically over episodes | multipliers applied to already-multiplied model | snapshot pristine `model.body_mass`/`geom_friction` at wrapper init; restore before every draw (Exercise 2) |
| Heatmaps differ wildly across seeds | 10 eps/cell too few near the success cliff | it's a binomial: 10 eps → ±~15% at p=0.5; average seeds, read map ordering, don't over-read single cells |
| $w{=}4$ arm never learns anything | hardest draws dominate early replay | it's real (that's the failure mode) — but confirm nominal-pinned eval also fails before claiming it; consider a warmup at $w{=}1$ noted as a deviation |
| `mj_name2id` returns −1 | body/geom names differ across Gymnasium/MuJoCo versions | print the model's name lists once; pin `gymnasium` and `mujoco` versions in the committed config |
| Cloud eval renders fail | headless GL | `MUJOCO_GL=egl`; state-based obs means you only render for videos anyway |

## Going deeper

- **AutoDR-lite.** Start from $w{=}1.1$ bounds per parameter; every 10k steps evaluate 10 episodes pinned at each parameter's current upper and lower bound (one parameter at a time, others nominal); boundary success ≥ 70% → push that bound out 10%, < 30% → pull in 10%. Same budget as the fixed arms; overlay the learned range on its heatmap and place it on the two-curve plot. One paragraph on what DORAEMON's entropy objective would do differently. Bounds should grow then stabilize — hitting the cap means the thresholds are too lax.
- **Three seeds + $w{=}1.5$.** Fill in the width axis and report whether the crossing moved.
- **Isaac path.** NVIDIA's "Train an SO-101 From Sim-to-Real With Isaac" learning path (docs.nvidia.com/learning/physical-ai/sim-to-real-so-101) on a cloud RTX instance: GPU-parallel DR at scale; compare its randomization schedule to the width your heatmap chose. If the hardware track is running, deploy your $w$ intuition as a prediction about which H3 OOD conditions will transfer, and check it.

## References

- Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*, IROS 2017. arXiv:1703.06907.
- Akkaya et al., *Solving Rubik's Cube with a Robot Hand*, 2019. arXiv:1910.07113 (§5: ADR).
- Tiboni et al., *Domain Randomization via Entropy Maximization* (DORAEMON), ICLR 2024. arXiv:2311.01885.
- LeRobot team, *Robot Learning: A Tutorial*, §3.2.2. arXiv:2510.12403.
