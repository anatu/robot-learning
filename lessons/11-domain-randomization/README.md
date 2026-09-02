# Lesson 11 — Domain Randomization and the Reality Gap

This lesson makes the tutorial's survey of domain randomization empirical. You will build a wrapper that randomizes the physical parameters of a simulated pushing task, train policies at several randomization widths, and evaluate each on a held-out grid of test dynamics that you commit to before training. The result is a set of transfer heatmaps that show the two characteristic failure modes of domain randomization: a policy trained without randomization fails as soon as the dynamics change, and a policy trained with too much randomization loses performance on the nominal task. Both matter later, because they explain why simulation benchmarks and real-robot results diverge (Lessons 18 and 19) and how to read the in-distribution versus out-of-distribution gap on the physical arm (H3).

| | |
|---|---|
| **Phase** | 3 — Reinforcement learning |
| **Time** | ~4–5 h desk time (AI-assisted) + ~3–4 GPU-hours for the sweep |
| **Cost** | ~$3–5 (3 widths × 2 seeds, short runs; evaluation is cheap) |
| **Prerequisites** | 08 (`sac.py` trains every arm unchanged) |
| **Feeds into** | 18–19 (why simulation benchmarks and real transfer diverge), H3 (interpreting the ID/OOD gap on the real arm), 22 (the sim-to-real capstone option) |

## Learning objectives

After this lesson you can:

1. **Formalize** domain randomization as training on $p(s'|s,a;\xi)$ with $\xi \sim \Xi$, and state precisely what distribution the resulting policy is optimal for.
2. **Diagnose** the compounding-multiplier bug from its drift pattern, and state the invariant a randomization wrapper must maintain across resets.
3. **Predict** the shape of a train-width × test-dynamics transfer heatmap before training, then produce it with enough seeds to trust its ordering.
4. **Demonstrate** both failure modes, no transfer at zero width and nominal-performance loss at excessive width, in your own data.
5. **Decide** a randomization width from the nominal-versus-off-nominal trade-off, and explain why frontier laboratories largely collect real data instead.

## Principles

### The formalism

A simulator is not a single dynamics model but a family of them, $\mathcal{D}_\xi$, indexed by physical parameters $\xi$ such as masses, friction coefficients, damping, and actuation delays (tutorial §3.2.2). Domain randomization trains one policy on a mixture over that family: it maximizes $\mathbb{E}_{\xi \sim \Xi}\,\mathbb{E}_{\tau \sim \pi, \mathcal{D}_\xi}[R(\tau)]$, where $\Xi$ is a distribution over parameters chosen by the designer. The bet is that if $\Xi$ is wide enough to contain, or at least surround, the real world's parameters $\xi^*$, then a policy that is robust across $\Xi$ will transfer. The cost is that the policy is now optimizing an average over dynamics rather than the dynamics it will actually face. At large widths the optimum of that average is a conservative behaviour that underperforms in every individual environment, and a policy without memory can do no better than play the average dynamics, because it has no way to identify which member of the family it is currently in.

### What the width controls

A width-$w$ arm samples each randomized parameter log-uniformly in $[\xi_{\text{nom}}/w, \; \xi_{\text{nom}} \cdot w]$ once per episode. A width of one is no randomization at all, and a width of four spans a sixteen-fold range. The distribution is log-uniform rather than uniform because mass and friction enter the dynamics multiplicatively, so a factor of two above nominal should be as likely as a factor of two below.

### The invariant a wrapper must maintain

The multipliers are drawn per episode and must be applied to the pristine model values, never to whatever the model currently holds. If a wrapper multiplies the current value each time, the effective multiplier becomes the product of every draw so far and performs a random walk across episodes, so that the dynamics drift steadily away from nominal without any error being raised. This is a common silent bug in randomization code, and Exercise 2 has you plant it and observe the drift.

### Automated randomization

A fixed width is a guess, and the field has developed methods that adjust it during training. Automatic Domain Randomization (Akkaya et al. 2019, the Rubik's-cube system) samples at the boundary of the current range for one parameter at a time, and expands that bound when performance at the boundary exceeds a threshold or shrinks it when performance falls below one; the result is a curriculum by bracketing. DORAEMON (Tiboni et al. 2024) states the objective more cleanly: maximize the entropy of $\Xi$ subject to a constraint on the success rate. A minimal version of both ideas is described under Going deeper.

### What randomization can and cannot buy

Domain randomization buys robustness to the parameters you chose to randomize, and it pays for that robustness in nominal performance and training compute. It does nothing for the parameters you did not think to randomize, because the policy has never seen them vary. This limitation is the argument, running from tutorial §3.2.2 to §5, for why laboratories such as Physical Intelligence spend their budget on real-data collection rather than on ever-wider $\Xi$: real data varies along every axis at once, including the ones nobody enumerated.

**Carry forward**

- Domain randomization optimizes an average over the parameter distribution $\Xi$, so the width $w$ trades nominal performance for coverage of off-nominal dynamics.
- A randomization wrapper must draw its multipliers per episode, apply them to the pristine model values, resolve bodies and geoms by name through `mj_name2id`, and log the drawn $\xi$; applying multipliers to current values makes the dynamics random-walk.
- The transfer heatmap has two failure modes: at $w{=}1$ a bright island at nominal that dies off-nominal, and at large $w$ a flat map whose nominal cell has dropped.
- Domain randomization is blind to parameters that were not randomized, whereas real data varies along every axis, which is why real data is preferred when it can be afforded.

| Source | Read for |
|---|---|
| Tutorial §3.2.2 | the $\mathcal{D}_\xi$, $\xi \sim \Xi$ formalism, and where domain randomization sits in the sim-to-real argument |
| Tobin et al. 2017, arXiv:1703.06907 | the origins of visual domain randomization; note that it randomizes appearance whereas you randomize dynamics, and keep the distinction clear |
| Akkaya et al. 2019, arXiv:1910.07113, §5 | ADR's expand-and-shrink rule and boundary sampling |
| Tiboni et al. 2024 (DORAEMON) | the entropy-maximization framing; read the objective and skim the rest |

## Exercise 1 — Build the randomization wrapper [Build]

In this exercise you build the controlled experimental substrate that everything else rests on. The environment is `Pusher-v5` from `gymnasium[mujoco]`, in which a 7-DOF arm pushes a cylinder to a goal. Success is a metric quantity (a final object-to-goal distance below 0.05 m), so no classifier is needed; log the return as well.

The specification for `dr_wrapper.py`, which defines `DynamicsRandomizationWrapper(env, width, seed)` and which an AI tool can draft:

- On each `reset()`, draw multipliers log-uniformly in $[1/w, w]$ and apply them to the object's `body_mass` and to the object-table sliding friction (`geom_friction[:, 0]` of the relevant geoms).
- Resolve body and geom IDs by name with `mj_name2id`, never by hard-coded index. Print the model's name table first; the body name in the skeleton below is illustrative.
- Restore the pristine model values before each draw.
- Expose the drawn $\xi$ in `info["dynamics"]`.
- Provide a `pin(xi)` method that fixes $\xi$ for evaluation instead of sampling it.

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
The check, in `check_wrapper.py`: the multipliers stay within $[1/w, w]$ over 100 resets; the draws are identical under the same seed; $w{=}1$ is bit-identical to the raw environment; and `pin()` holds its value across resets.

Once the wrapper passes, calibrate the task's difficulty: inject the wrapper into Lesson 08's `sac.py` through `make_env` and run briefly at $w{=}1$. Pusher is solved in roughly 200–300k steps.

**✅ Checkpoint:** `check_wrapper.py` passes; a log of ten resets shows $\xi$ draws spanning the intended range; and nominal SAC shows clear learning progress by 100k steps.

## Exercise 2 — Compounding multipliers [Diagnose]

This exercise tests objective 2. The compounding bug is the most common reason that a randomization study appears to show that randomization did not help, and its pattern is easy to recognize once you have seen it.

1. In a copy of the wrapper, apply each draw to the current `body_mass` instead of to the pristine snapshot.
2. Before running, write down the trajectory you expect for the effective mass multiplier over ten resets at $w{=}2$: its expected value, its spread, and the reason, which is a question about what kind of random walk this is in log space.
3. Run ten resets with each wrapper and plot the effective multiplier per reset. Then write the check for `check_wrapper.py` that catches the bug, which is that two consecutive resets must not compound.

**✅ Checkpoint:** the broken wrapper's log-multiplier performs a random walk whose variance grows with the reset count, whereas the correct wrapper's draws are independent and identically distributed in $[1/w, w]$; the catching assertion is in the check script.

## Exercise 3 — Pre-register the evaluation grid [Write]

This exercise is about experimental discipline: the test set exists before any training run, and the git history proves it. Deciding the evaluation grid after seeing the policies would let the grid be tuned, consciously or not, to the results.

1. Define the held-out grid as mass multiplier × friction multiplier, each with five log-spaced points in $[0.25, 4]$, giving 25 cells. For every trained policy you will run ten episodes per cell with the wrapper pinned to that cell's $\xi$ and record the success rate per cell.
2. Write `configs/eval_grid.yaml` with the cell list, the episode count, and the seed list, and commit it.
3. Write down the two scalar summaries you will extract across widths: success in the nominal cell (the centre of the grid) and mean success over the off-nominal cells (every cell outside $[1/2, 2]$ on either axis).

**✅ Checkpoint:** `configs/eval_grid.yaml` is committed, and its commit predates every training run in the git log.

## Exercise 4 — Run the sweep and read the heatmaps [Predict → Run]

This exercise tests objectives 3 and 4. Sketching the heatmaps before training turns them from illustrations into predictions, and the two failure modes should be visible in your own data.

1. Before training, draw in `RESULTS.md` a hand sketch of the three heatmaps for $w \in \{1.0, 2.0, 4.0\}$: where the bright region sits, how it widens with $w$, and whether the nominal cell at $w{=}4$ is dimmer than at $w{=}1$. Then sketch the two-curve plot of nominal-cell success and mean off-nominal success against $w$, and mark where you expect the curves to cross.
2. The arms are $w \in \{1.0, 2.0, 4.0\}$ × seeds {0, 1}, with an identical SAC configuration (Lesson 08's `sac.py` with `--env-id Pusher-v5` and the wrapper injected in `make_env`) and an identical budget of 300k steps. Log $\xi$ per episode. `run_sweep.sh` launches all six runs and then the pinned evaluation over `configs/eval_grid.yaml`.
3. Render one heatmap per arm, averaging the two seeds and keeping the per-seed maps in an appendix folder. Use the same colour scale across arms, because the comparison between arms is what the figure is for.
4. Extract the two scalar curves and plot both against $w$ on one pair of axes.
5. Reconcile the result with your sketch.

**✅ Checkpoint:** the $w{=}1$ map is a bright island at nominal that fades away from the diagonal, which is the low-entropy failure; the maps widen with $w$; and at $w{=}4$ the map is flatter but the nominal cell has measurably dropped from its $w{=}1$ value, which is the over-randomization failure. The two curves cross. If no arm shows the nominal drop, the task is too easy for the widths chosen; add $w{=}6$ and say so. Ten episodes per cell is a binomial estimate with a spread of roughly ±15 percentage points at p=0.5, so read the ordering of the maps rather than individual cells.

## Exercise 5 — Choose a width to ship [Decide]

This exercise tests objective 5. It asks for a number and a defence of it, using the trade-off you have just measured.

1. From the two-curve plot, choose the width you would train at for a real-arm deployment whose true mass and friction you can estimate to within a factor of two. State the nominal-performance price you pay and the off-nominal coverage you buy, with the numbers from your table.
2. Write one paragraph on the blind spot: name two parameters of a real pusher that you did not randomize, and say what your heatmap can tell you about them, which is nothing. Connect this to §5 of the tutorial: when is widening $\Xi$ the right choice, and when is collecting real data the right choice? The cost per real episode is the deciding quantity.

**✅ Checkpoint:** the decision names a width, both prices, and the two un-randomized parameters.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `dr_wrapper.py` and `check_wrapper.py` | the four wrapper properties plus the compounding assertion, all passing |
| `configs/eval_grid.yaml` and `run_sweep.sh` | the full sweep and evaluation grid reproducible from one command; the grid committed before training, with the git history as proof |
| `plots/` | three heatmaps on a shared scale; the two-curve trade-off plot; the compounding-multiplier random-walk figure |
| `RESULTS.md` | the heatmap sketch and its reconciliation; both failure modes annotated on your own figures; the width decision with numbers; the paragraph on why real data is preferred; at most 12 sentences |

## Done when

- [ ] The set of transfer heatmaps exists, with two seeds averaged and a shared colour scale.
- [ ] Both failure modes are visible in your data and annotated against your sketch.
- [ ] The evaluation grid predates the training runs in the git history.
- [ ] The compounding-multiplier bug is diagnosed by mechanism and caught by a committed check.
- [ ] A width is chosen and defended with numbers.

## Self-check

1. Write the domain-randomization training objective and mark exactly where the width $w$ enters it.
2. Why are the mass and friction multipliers drawn log-uniformly rather than uniformly?
3. A memoryless policy trained at $w{=}4$ behaves conservatively. What architectural change would let a policy adapt to the episode's dynamics instead, and what information would it exploit?
4. Your heatmap shows transfer along the mass axis but a cliff along the friction axis. What does that tell you about which parameter the task is sensitive to, and what does it illustrate about the blind spot for parameters you did not randomize?
5. Make the opposite case: when is widening $\Xi$ the right choice rather than collecting real data?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Dynamics drift monotonically over episodes | multipliers applied to an already-multiplied model | snapshot the pristine `model.body_mass` and `geom_friction` at wrapper init and restore them before every draw (Exercise 2) |
| Heatmaps differ wildly across seeds | 10 episodes per cell is too few near the success cliff | it is a binomial estimate, roughly ±15 points at p=0.5; average the seeds, read the ordering of the maps, and don't over-read single cells |
| The $w{=}4$ arm never learns anything | the hardest draws dominate early replay | this is the failure mode itself, but confirm that the nominal-pinned evaluation also fails before claiming it; a warm-up at $w{=}1$ is acceptable if noted as a deviation |
| `mj_name2id` returns −1 | body and geom names differ across Gymnasium and MuJoCo versions | print the model's name lists once; pin `gymnasium` and `mujoco` versions in the committed config |
| Cloud evaluation renders fail | headless GL | `MUJOCO_GL=egl`; with state-based observations you only render for videos anyway |

## Going deeper

- **A minimal AutoDR.** Start from bounds of $w{=}1.1$ per parameter. Every 10k steps, evaluate ten episodes pinned at each parameter's current upper and lower bound, one parameter at a time with the others at nominal; if boundary success is at least 70 percent, push that bound outward by 10 percent, and if it is below 30 percent, pull it inward by 10 percent. Use the same budget as the fixed arms, overlay the learned range on its heatmap, and place the arm on the two-curve plot. Write one paragraph on what DORAEMON's entropy objective would do differently. The bounds should grow and then stabilize; if they reach the cap, the thresholds are too lax.
- **Three seeds and $w{=}1.5$.** Fill in the width axis and report whether the crossing moved.
- **The Isaac path.** Run NVIDIA's "Train an SO-101 From Sim-to-Real With Isaac" learning path (docs.nvidia.com/learning/physical-ai/sim-to-real-so-101) on a cloud RTX instance to see GPU-parallel randomization at scale, and compare its randomization schedule with the width your heatmap chose. If the hardware track is running, state your chosen width as a prediction about which H3 out-of-distribution conditions will transfer, and check it.

## References

- Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*, IROS 2017. arXiv:1703.06907.
- Akkaya et al., *Solving Rubik's Cube with a Robot Hand*, 2019. arXiv:1910.07113 (§5: ADR).
- Tiboni et al., *Domain Randomization via Entropy Maximization* (DORAEMON), ICLR 2024. arXiv:2311.01885.
- LeRobot team, *Robot Learning: A Tutorial*, §3.2.2. arXiv:2510.12403.
