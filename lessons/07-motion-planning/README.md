# Lesson 07 — Motion Planning *(optional)*

The "plan" stage of the classical pipeline the tutorial critiques but never shows: a joint-space RRT and a kinematic trajectory optimizer for the SO-101 in clutter, raced on the same problems — so that "sample for topology, optimize for quality" is a result in your table, not folklore. (Correction to the scaffold: this material is MIT *Robotic Manipulation* **ch. 6**, "Motion Planning.")

| | |
|---|---|
| **Phase** | 2 — Classical core (optional) |
| **Time** | 1 session (4–6 h desk time), all Mac-local |
| **Cost** | $0 |
| **Prerequisites** | 03 (`kinematics.py`: `fk`, `jacobian`), 04 (`controllers.py`, for Going deeper only) |
| **Feeds into** | H6 (mobile bases replan constantly); the Phase 4 argument for why end-to-end replaced sense-plan-act |
| **Skip criteria** | The learning-based track never needs it — but it sharpens the "why end-to-end won" argument, and H6 reuses it |

## Learning objectives

After this lesson you can:

1. **Explain** why RRT is probabilistically complete but arbitrarily suboptimal, and what shortcutting buys back.
2. **Predict** the effect of a weighted joint-space metric on tree growth, and confirm it on your own problems.
3. **Diagnose** an optimizer that reports convergence while the path collides — both ways it happens.
4. **Decide** when each planner wins from a benchmark you ran, and articulate the completeness-vs-quality tradeoff with rows from your table.

## Principles

**Configuration space.** Planning happens in $\mathcal{C} = $ joint space, not task space: obstacles map to $\mathcal{C}_{\text{obs}}$ (weirdly shaped, never computed explicitly), and a collision checker answering "is $q$ free?" is the only interface to it. For the SO-101, $\mathcal{C} \subset \mathbb{R}^5$ boxed by joint limits. Everything depends on a trustworthy, fast `is_free(q)`.

**RRT** (LaValle; MIT ch. 6). Grow a tree from $q_{\text{start}}$:

```
repeat:
  q_rand ← goal with prob. p_goal, else Uniform(C)
  q_near ← nearest tree node to q_rand
  q_new  ← q_near + η · (q_rand − q_near)/‖·‖     # steer, step size η
  if edge (q_near → q_new) collision-free (checked at resolution δ):
      add q_new; if ‖q_new − q_goal‖ < η and edge free: done
```

Probabilistically complete (finds a path if one exists, eventually), zero optimality: raw paths jerk through free space, hence **shortcutting** — repeatedly pick two random points on the path, replace the segment between them with a straight edge if collision-free. The nearest-neighbor metric matters: a radian of `shoulder_pan` moves the gripper much farther than a radian of `wrist_roll`, so an unweighted Euclidean metric explores the wrong directions first.

**Kinematic trajectory optimization** (MIT ch. 6). Decision variables = waypoints $q_{1..N}$; solve

$$\min_{q_{1..N}} \sum_k \lVert q_{k+1} - q_k \rVert^2 \quad \text{s.t.} \quad q_1 = q_{\text{start}},\; q_N = q_{\text{goal}},\; q_k \in \text{limits}, \; d(q_k) \ge d_{\text{safe}}$$

where $d(q)$ is scene clearance. With penalty-form collisions, plain `scipy.optimize.minimize` (SLSQP/L-BFGS) works at this scale. Smooth, locally optimal, fast **when it converges** — but the landscape is nonconvex: a straight-line initialization through an obstacle converges to an infeasible local minimum. The standard cure: seed it with the RRT path. That hybrid — sample for topology, optimize for quality — is the punchline. (GCS, MIT ch. 6's finale, gets global optimality through convex decomposition; read the section, skip the implementation.)

**An optimizer's "success" is not feasibility.** Penalty-form collisions trade clearance for smoothness when the weight is low; waypoint-only checks miss collisions between waypoints. Verify every "converged" result with the collision oracle at execution resolution.

**Carry forward**

- Planning lives in $\mathcal{C}$; the collision oracle is the only window into $\mathcal{C}_{\text{obs}}$, and its speed sets the planning budget.
- RRT: complete but not optimal; shortcut afterwards. RRT*'s one-line rewiring change buys asymptotic optimality at a per-iteration cost.
- Trajopt: locally optimal and smooth, blind to topology; seed it with a sample-based path.
- "Converged" ≠ "collision-free" — always re-verify with the oracle.
- The whole pipeline assumes a perfect scene model; a 2 cm pose error breaks it in three places, and Phase 4 exists because of that.

| Source | Read for |
|---|---|
| MIT *Robotic Manipulation* ch. 6 | RRT variants, kinematic trajectory optimization, and the GCS overview — the lesson's spine |
| LaValle, *Planning Algorithms* (free at lavalle.pl/planning) ch. 5 | sampling-based planning foundations; why nearest-neighbor metrics matter |
| Tutorial §2.4 | the pipeline-brittleness argument you're about to experience firsthand |

## Exercise 1 — The scene and the collision oracle [Build]

The principle: a fast, exact `is_free(q)` is the interface to $\mathcal{C}_{\text{obs}}$; everything else is built on it. Spec:

- `scenes/tabletop.xml`: the SO-101 on a table with 4–5 box/cylinder obstacles (a shelf edge, a mug, a wall segment) placed to create at least one narrow passage in workspace. `<include>` `so101_new_calib.xml`, don't fork it. Obstacle poses parameterized so Exercise 7 can randomize them. Commit a rendered scene image.
- `planner/collision.py`: `is_free(q) -> bool` — set `qpos`, `mj_forward`, inspect contacts (`data.ncon`), filtering intentional/adjacent-link pairs via an exclusion list built once from the home config's contacts; `edge_free(q1, q2, delta)` discretized at $\delta = 0.02$ rad (max joint-space step).
- The check (`checks.py`): home config free; a config driving the wrist into the table not free; symmetric ± wrist configs agree; zero false collisions over 1,000 random *visually verified* free configs (spot-check 20 in the viewer); calls/second printed.

**✅ Checkpoint:** oracle ≥ 5,000 calls/s (expect order 10⁴/s — it's `mj_forward`, not physics); the exclusion-list test passes.

## Exercise 2 — The endpoints-only bug [Diagnose]

The principle: an edge check that only tests its endpoints is a planner that walks through walls, and you should be able to predict exactly how.

1. **Write first:** if `edge_free` checked only `q1` and `q2`, what would RRT paths look like near thin obstacles (the shelf edge), and which of the two path-quality numbers (success rate, path length) would *improve*?
2. Have the AI tool produce a variant `edge_free_endpoints_only`; run RRT (Exercise 3) with it on 5 problems that pass near the shelf edge; verify each returned path at $\delta = 0.02$ with the correct oracle.
3. Record in `RESULTS.md`: how many "successful" paths collide, and where along the path.

**✅ Checkpoint:** at least one endpoints-only path passes through the shelf; the reconciliation names the mechanism.

## Exercise 3 — RRT and shortcutting [Build]

The principle: probabilistic completeness from the algorithm, path quality from post-processing. Spec:

- `planner/rrt.py`: the pseudocode above with $\eta = 0.15$ rad, $p_{\text{goal}} = 0.1$, $\delta = 0.02$ rad, joint-limit-aware sampling, max 20,000 nodes. Brute-force nearest-neighbor (profile before reaching for a KD-tree — and KD-trees are awkward under per-joint scaling anyway). A `metric_weights` argument: `None` for Euclidean, or a diagonal $W$.
- `planner/shortcut.py`: 200 iterations of random-pair rewiring with `edge_free`; returns the path and its length before/after.
- A problem generator: random collision-free start/goal pairs with minimum task-space separation 25 cm (rejection sampling via Lesson 03's `fk`); mark problems whose straight-line joint-space segment collides as narrow-passage candidates.
- Record per solve: success (within node budget), nodes expanded, planning time, raw and shortcut path length.

**✅ Checkpoint:** ≥ 90% success on 20 random feasible problems within budget; shortcutting cuts median path length ≥ 30%; one animation of a solve showing the tree exploring *around* an obstacle, not through it.

## Exercise 4 — Weight the metric [Predict → Run]

The principle: the nearest-neighbor metric decides which directions the tree explores first, and the right weights come from the Jacobian.

1. Build $W$ from average task-space displacement per joint: Lesson 03's `jacobian` at 100 random configs, mean column norm per joint.
2. **Write first:** the ratio of median nodes-expanded, weighted / unweighted, on the same 20 problems — direction and rough magnitude — and which joint's weight dominates.
3. Run both metrics on the 20 problems with the same seeds. Table: success, median nodes, median time, per metric.
4. Reconcile in `RESULTS.md`.

**✅ Checkpoint:** the weighted metric beats unweighted on median nodes-expanded; the measured ratio and your predicted ratio are both in the report.

## Exercise 5 — Trajectory optimization [Build]

The principle: direct transcription yields smooth, locally optimal paths — when it converges to a feasible one. Spec for `planner/trajopt.py`:

- $N = 30$ waypoints; smoothness objective; penalty-form clearance $\max(0, d_{\text{safe}} - d(q_k))^2$ with $d_{\text{safe}} = 2$ cm, clearance from MuJoCo contact distances (`mj_forward` + nearest signed distance per waypoint; FD gradients are acceptable — budget for it); joint-limit bounds; `scipy.optimize.minimize` (SLSQP or L-BFGS).
- `optimize(q_init_path) -> (path, converged)`; initializations: straight line in joint space, or an RRT path resampled to $N$ waypoints.
- Post-verification is mandatory and built in: every returned path is re-checked with `edge_free` at $\delta = 0.02$ and the result reported separately from the optimizer's `converged` flag.

**✅ Checkpoint:** on an obstacle-free problem, trajopt from a straight line converges and passes verification; the two flags (`converged`, `verified`) are both logged.

## Exercise 6 — The local minimum and its cure [Predict → Run]

The principle: trajopt is blind to topology; a sample-based seed supplies exactly what restarts cannot.

1. Pick one narrow-passage problem from Exercise 3's generator.
2. **Write first:** what trajopt from a straight-line init does on it (converges? verified? where does the path sit relative to the obstacle?); whether 3 random restarts fix it; and what the RRT→trajopt hybrid returns instead.
3. Run the three arms: (a) straight line, (b) straight line + 3 random restarts, (c) RRT path as seed. Side-by-side plot or animation of (a) vs (c).
4. Reconcile in `RESULTS.md`: what the seed provides that restarts don't.

**✅ Checkpoint:** (a) converges into the obstacle (converged = true, verified = false) or fails; (c) is verified and shorter/smoother than the raw RRT path.

## Exercise 7 — The benchmark [Predict → Run]

The principle: completeness vs quality is a tradeoff you measure, not assert.

1. **Write first:** for 20 problems × 3 planners (RRT+shortcut, trajopt-straight, RRT→trajopt hybrid), the expected ordering on success %, median path length, and median wall-clock; and on which problem subset trajopt-straight fails.
2. `benchmark.py`: one command runs the 20×3 grid (narrow-passage problems marked as a subset) and prints the table.
3. Reconcile in `RESULTS.md`.

**✅ Checkpoint:** trajopt-straight fails on a nontrivial fraction, concentrated on narrow-passage problems; the hybrid matches RRT's success rate at meaningfully shorter, smoother paths.

## Exercise 8 — When each wins [Decide]

From your own table: in ≤ 10 sentences, when you would run RRT+shortcut alone, when trajopt alone, when the hybrid — each claim pointing at a row. Then the closing paragraph: list the three places this pipeline breaks when the mug's pose estimate is off by 2 cm, and which Phase 4 method sidesteps each.

**✅ Checkpoint:** every "when" claim cites a row; the 2 cm paragraph names three breakpoints.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `planner/` (`collision.py`, `rrt.py`, `shortcut.py`, `trajopt.py`) | importable; `trajopt.py` reports `converged` and `verified` separately |
| `scenes/tabletop.xml` + generator | randomizable obstacle poses; committed with a rendered scene image |
| `checks.py`, `benchmark.py` | oracle checks + the 20×3 table from one command each |
| `media/` | one tree-growth animation; the local-minimum vs hybrid side-by-side |
| `RESULTS.md` | Exercise 2/4/6/7 predictions with reconciliations; the Exercise 8 decision; the 2 cm paragraph |

## Done when

- [ ] Oracle ≥ 5,000 calls/s with an exact exclusion list.
- [ ] RRT ≥ 90% success on 20 random feasible problems; hybrid matches it with better paths.
- [ ] The endpoints-only bug and the local-minimum failure are both demonstrated and explained.
- [ ] All predictions written before their runs and reconciled after.
- [ ] `RESULTS.md` answers "when each wins" with rows from your own table.

## Self-check

1. Why is RRT probabilistically complete but arbitrarily suboptimal? What one-line change makes it asymptotically optimal, and what does it cost?
2. Why does the joint-space metric need weighting, and what geometric object would the *right* metric use?
3. Your optimizer reports convergence but the path collides. Name the two distinct ways that happens.
4. Why does seeding trajopt with an RRT path fix the local-minimum problem — what exactly does the seed provide that restarts don't?
5. This whole pipeline assumed a perfect scene model. List the three places it breaks when the mug's pose estimate is off by 2 cm — and which Phase 4 method sidesteps each.

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| RRT finds paths through obstacles | edge checked only at endpoints | discretize at $\delta$; Exercise 2 makes you experience this on purpose |
| Every config "collides" | self-collision pairs (adjacent links, gripper pads) not excluded | build the exclusion list from the home config's contacts; test it |
| RRT slow despite fast oracle | nearest-neighbor is $O(n)$ per iteration and you have 20k nodes | it's fine to be $O(n)$ — but cap nodes, and profile before optimizing |
| Trajopt "succeeds" everywhere including through walls | penalty weight too low — collisions traded for smoothness | the built-in verification catches it; raise the penalty or use a hard constraint with SLSQP |
| Narrow passage never solved | $\eta$ too large to enter, $\delta$ too coarse inside | halve both near failure cases; note the cost — this tuning pain is a *result*, write it down |

## Going deeper

- **Execute the plan.** Time-parameterize shortcut paths with per-segment trapezoidal profiles honoring $\dot q_{\max} = 1.5$ rad/s, $\ddot q_{\max} = 4$ rad/s² per joint (zero boundary velocities per segment, or blend — state which); sample $q^{*}(t)$ at 50 Hz and execute through Lesson 04's QP tracker in the full MuJoCo scene. Verify zero collisions during *execution* — the tracker can cut corners the planner didn't (fix: tighten $d_{\text{safe}}$ or densify waypoints). Expect ≥ 45/50 collision-free with $\max \lvert \dot q \rvert \le 1.5$ rad/s.
- **RRT-Connect and RRT*.** Add the bidirectional variant and the rewiring step; re-run the benchmark.
- **GCS.** Read MIT ch. 6's Graphs of Convex Sets section and write a one-page note: what GCS guarantees that neither of your planners can, what it requires (convex decomposition of free space), and why that requirement is hard for the SO-101's 5-D C-space.

## References

- Tedrake, *Robotic Manipulation*, ch. 6 — Motion Planning (RRT, kinematic trajectory optimization, GCS). manipulation.csail.mit.edu.
- LaValle, *Planning Algorithms*, ch. 5 — free online at lavalle.pl/planning.
- Kuffner & LaValle, *RRT-Connect*, ICRA 2000 (the bidirectional variant worth knowing exists).
- LeRobot team, *Robot Learning: A Tutorial*, §2.4. arXiv:2510.12403.
