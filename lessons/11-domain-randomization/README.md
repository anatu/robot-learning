# Lesson 11 — Domain Randomization & the Reality Gap

Make the tutorial's zero-code survey of domain randomization empirical: train policies across randomization widths, evaluate them on a held-out grid of test dynamics, and produce the transfer heatmap that shows both canonical failure modes — too little entropy fails to transfer, too much hurts the nominal task.

| | |
|---|---|
| **Phase** | 3 — Reinforcement learning |
| **Time** | ~2 sessions desk time (6–8 h) + ~4–6 GPU-hours for the training sweep |
| **Cost** | ~$4–8 (5 training arms × 3 seeds, short runs; evaluation is cheap) |
| **Prerequisites** | 08 (`SACAgent` trains every arm unchanged) |
| **Feeds into** | 18–19 (why sim benchmarks and real transfer diverge), H3 (interpreting the ID/OOD gap on the real arm), 22 (the sim-to-real capstone option) |

## Learning objectives

After this lesson you can:

1. **Formalize** DR as training on $p(s'|s,a;\xi)$, $\xi \sim \Xi$, and state precisely what distribution the resulting policy is optimal for.
2. **Implement** dynamics randomization as an env wrapper that survives resets and version pins.
3. **Produce** a train-width × test-dynamics transfer heatmap with enough seeds to trust its shape.
4. **Demonstrate** both failure modes — no-transfer at zero width, nominal-performance loss at excessive width — in your own data.
5. **Explain** what AutoDR and DORAEMON automate, and why frontier labs largely collect real data instead.

## Background

**The formalism** (tutorial §3.2.2). A simulator is a dynamics family $\mathcal{D}_\xi$ indexed by physical parameters $\xi$ (masses, frictions, damping, delays). DR trains one policy on the mixture: maximize $\mathbb{E}_{\xi \sim \Xi}\,\mathbb{E}_{\tau \sim \pi, \mathcal{D}_\xi}[R(\tau)]$. The bet: if $\Xi$ is wide enough to contain (or surround) the real world's $\xi^*$, a policy robust across $\Xi$ transfers. The cost: the policy optimizes an *average* over dynamics, so width is not free — at high width the optimum is a conservative behavior that underperforms everywhere, and with a memoryless policy the best it can do is play the average dynamics rather than identify the current ones.

**What the knobs mean.** A width-$w$ arm samples each randomized parameter log-uniformly in $[\xi_{\text{nom}}/w, \; \xi_{\text{nom}} \cdot w]$ per episode. $w{=}1$ is no randomization; $w{=}4$ spans a 16× range. Log-uniform, because mass and friction act multiplicatively on dynamics.

**The automation layer.** Fixed widths are guesses, so the field automated them: AutoDR (Akkaya et al. 2019, the Rubik's-cube system) samples at the boundary of the current range per-parameter and *expands* a bound when boundary performance clears a threshold, shrinks when it fails — curriculum via bracketing. DORAEMON (Tiboni et al. 2024) states the objective cleanly: maximize the *entropy* of $\Xi$ subject to a success-rate constraint. Your AutoDR-lite in Part 3 is the minimal version of both ideas.

**The punchline you should carry out of this lesson:** DR buys robustness to the parameters you randomize, priced in nominal performance and training compute — and it does nothing for the parameters you didn't think to randomize. That's the argument (tutorial §3.2.2 → §5) for why π-style labs spend on real data collection instead of ever-wider $\Xi$.

| Source | Read for |
|---|---|
| Tutorial §3.2.2 | the $\mathcal{D}_\xi$, $\xi \sim \Xi$ formalism; where DR sits in the sim-to-real argument |
| Tobin et al. 2017, arXiv:1703.06907 | visual DR origins — note it randomizes *appearance*, you randomize *dynamics*; keep the distinction crisp |
| Akkaya et al. 2019, arXiv:1910.07113, §5 | ADR's expand/shrink rule and boundary sampling — the exact loop you're miniaturizing |
| Tiboni et al. 2024 (DORAEMON) | the entropy-maximization framing; read the objective, skim the rest |

## Part 1 — The randomization wrapper (Mac, ~2 h)

Produces the controlled experimental substrate everything else stands on.

1. Env: `Pusher-v5` (`gymnasium[mujoco]`) — 7-DOF arm pushes a cylinder to a goal; success is metric and objective, no classifier needed. Define success = final object-to-goal distance < 0.05 m; also log return.
2. Write `DynamicsRandomizationWrapper(env, width, params, seed)`:
   - on each `reset()`, draw multipliers log-uniformly in $[1/w, w]$ and apply to (a) the object's `body_mass` and (b) the object–table sliding friction (`geom_friction[:, 0]` of the relevant geoms);
   - resolve body/geom IDs by name via `mj_name2id` — never hardcode indices;
   - restore pristine model values before each draw (multipliers compose otherwise — the classic silent bug);
   - expose the drawn $\xi$ in `info["dynamics"]` for logging.

   The skeleton, to fix the intended structure:
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
   (Body name `"object"` is illustrative — step 2's `mj_name2id` discipline means you print the model's actual name table first.)
3. Tests: multipliers within bounds; deterministic under seed; two consecutive resets don't compound; $w{=}1$ is bit-identical to the raw env.
4. Calibrate difficulty: run your Lesson 08 SAC briefly on $w{=}1$ to confirm the nominal task trains at all (Pusher solves in ~200–300k steps).

**✅ Checkpoint:** tests green; a 10-reset log shows $\xi$ draws spanning the intended range; nominal SAC shows clear learning progress by 100k steps.

## Part 2 — The sweep and the heatmap (cloud GPU, ~4 h wall-clock)

The core experiment. Pre-register everything before launching.

1. Arms: $w \in \{1.0, 1.5, 2.5, 4.0\}$ × 3 seeds, identical SAC config, identical step budget (300k). Log $\xi$ per episode.
2. Held-out evaluation grid, fixed in the repo *before* training finishes: mass-multiplier × friction-multiplier, each 7 log-spaced points in $[0.25, 4]$ → 49 cells. For every trained policy: 20 episodes per cell with the wrapper pinned to that cell's $\xi$ (no sampling), success rate per cell.
3. Render one heatmap per arm (average the 3 seeds; keep per-seed maps in an appendix folder). Same color scale across arms — the comparison *is* the figure.
4. Extract two scalar curves across $w$: nominal-cell success (the center cell) and mean off-nominal success (all cells outside the training range of $w{=}1.5$). Plot both against $w$ on one axis pair.

**✅ Checkpoint (expected shape):** the $w{=}1$ map is a bright island at nominal that dies off-diagonal (low-entropy fails); maps widen with $w$; at $w{=}4.0$ the map is flatter but the nominal cell has measurably dropped from its $w{=}1$ peak (over-randomization hurts). The two-curve plot crosses — the crossing region is your sweet spot. If no arm shows the nominal drop, your task is too easy for the widths chosen; push to $w{=}6$ and say so in `RESULTS.md`.

## Part 3 — AutoDR-lite (GPU, ~1.5 h)

One more arm, with the width *learned* instead of guessed.

1. Start from $w{=}1.1$ bounds per parameter. Every 10k steps, evaluate 20 episodes pinned at each parameter's current *upper and lower bound* (boundary sampling, one parameter at a time, others at nominal). Rule: boundary success ≥ 70% → push that bound outward by 10%; < 30% → pull inward by 10%.
2. Train with the same budget as Part 2's arms; log the bounds trajectory.
3. Evaluate the final policy on the same 49-cell grid; overlay the final learned range as a rectangle on its heatmap.
4. In `RESULTS.md`, place the AutoDR-lite arm on Part 2's two-curve plot and compare against the best fixed $w$. One paragraph: what DORAEMON's entropy objective would do differently from your expand/shrink rule.

**✅ Checkpoint:** the bounds trajectory grows and then stabilizes (not monotone to the cap — if it hits the cap, your thresholds are too lax); final transfer is competitive with the best fixed-width arm without you having chosen the width.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `dr_wrapper.py` + tests | the four properties in Part 1 step 3, green |
| `configs/` + `run_sweep.sh` | full sweep + eval grid reproducible in one command; eval grid committed before training (git history is the proof) |
| `plots/` | 4+1 heatmaps on a shared scale; the two-curve tradeoff plot; AutoDR bounds trajectory |
| `RESULTS.md` | both failure modes annotated on your own figures; sweet-spot statement with numbers; the "why real data instead" paragraph tying to §5; ≤ 12 sentences |

## Done when

- [ ] The transfer heatmap set exists with ≥ 3 seeds averaged and a shared color scale.
- [ ] Both failure modes are visible in *your* data and annotated.
- [ ] AutoDR-lite matched or beat fixed widths without a hand-picked $w$.
- [ ] The eval grid predates the training runs in git history.

## Self-check

1. Write the DR training objective and mark exactly where the width $w$ enters it.
2. Why log-uniform rather than uniform for mass/friction multipliers?
3. A memoryless policy trained at $w{=}4$ behaves conservatively. What architectural change would let a policy *adapt* to the episode's dynamics instead, and what information does it exploit?
4. Your heatmap shows transfer along the mass axis but a cliff along friction. What does that tell you about which parameter the task is actually sensitive to — and about DR's blind spot for parameters you didn't randomize?
5. Steelman the opposite bet: when is widening $\Xi$ the *right* call versus collecting real data? (Cost per real episode is the hinge.)

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Dynamics drift monotonically over episodes | multipliers applied to already-multiplied model | snapshot pristine `model.body_mass`/`geom_friction` at wrapper init; restore before every draw |
| Heatmaps differ wildly across seeds | 20 eps/cell too few near the success cliff | it's a binomial: 20 eps → ±~11% at p=0.5; average seeds, don't over-read single cells |
| $w{=}4$ arm never learns anything | hardest draws dominate early replay | it's real (that's the failure mode) — but confirm nominal-pinned eval also fails before claiming it; consider a warmup at $w{=}1$ noted as a deviation |
| `mj_name2id` returns −1 | body/geom names differ across Gymnasium/MuJoCo versions | print the model's name lists once; pin `gymnasium` and `mujoco` versions in the committed config |
| Boundary evals dominate AutoDR wall-clock | 2 params × 2 bounds × 20 eps every 10k steps | drop to 10 eps per boundary probe; the rule only needs a coarse signal |
| Cloud eval renders fail | headless GL | `MUJOCO_GL=egl`; state-based obs means you only render for videos anyway |

## Stretch

Run NVIDIA's free "Train an SO-101 From Sim-to-Real With Isaac" learning path (docs.nvidia.com/learning/physical-ai/sim-to-real-so-101) on a cloud RTX instance as the guided industrial-strength counterpart: GPU-parallel DR at scale, then compare its randomization schedule to the one your heatmap says is optimal. If you have the hardware track running, close the loop: deploy your best fixed-$w$ Pusher intuition as a prediction about which H3 OOD conditions will transfer, and check it.

## References

- Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*, IROS 2017. arXiv:1703.06907.
- Akkaya et al., *Solving Rubik's Cube with a Robot Hand*, 2019. arXiv:1910.07113 (§5: ADR).
- Tiboni et al., *Domain Randomization via Entropy Maximization* (DORAEMON), ICLR 2024. arXiv:2311.01885.
- LeRobot team, *Robot Learning: A Tutorial*, §3.2.2. arXiv:2510.12403.
