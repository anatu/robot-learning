# Lesson 07 — Motion Planning *(optional)*

The "plan" stage of the classical pipeline the tutorial critiques but never shows: build a joint-space RRT and a kinematic trajectory optimizer for the SO-101 in a cluttered MuJoCo scene, race them on 50 random problems, and learn exactly when each one wins. (Correction to the scaffold: this material is MIT *Robotic Manipulation* **ch. 6**, "Motion Planning.")

| | |
|---|---|
| **Phase** | 2 — Classical core (optional) |
| **Time** | 1–2 sessions (6–9 h), all Mac-local |
| **Cost** | $0 |
| **Prerequisites** | 03 (`kinematics.py`), 04 (the tracker that executes plans) |
| **Feeds into** | H6 (mobile bases replan constantly); the Phase 4 argument for why end-to-end replaced sense-plan-act |
| **Skip criteria** | The learning-based track never needs it — but it sharpens the "why end-to-end won" argument, and H6 reuses it |

## Learning objectives

After this lesson you can:

1. **Implement** RRT in the SO-101's 5-D joint space with MuJoCo collision checking, with goal biasing and tuned step size.
2. **Post-process** raw RRT paths with shortcutting and time-parameterize them against velocity/acceleration limits for execution by the Lesson 04 tracker.
3. **Formulate** kinematic trajectory optimization as direct transcription with collision penalties, and demonstrate its local-minima failure mode and its cure.
4. **Benchmark** sampling vs optimization honestly — success rate, path quality, wall-clock — and articulate the completeness-vs-quality tradeoff.

## Background

**Configuration space.** Planning happens in $\mathcal{C} = $ joint space, not task space: obstacles map to $\mathcal{C}_{\text{obs}}$ (weirdly shaped, never computed explicitly), and a collision checker answering "is $q$ free?" is the only interface to it. For the SO-101, $\mathcal{C} \subset \mathbb{R}^5$ boxed by joint limits.

**RRT** (LaValle; MIT ch. 6). Grow a tree from $q_{\text{start}}$:

```
repeat:
  q_rand ← goal with prob. p_goal, else Uniform(C)
  q_near ← nearest tree node to q_rand
  q_new  ← q_near + η · (q_rand − q_near)/‖·‖     # steer, step size η
  if edge (q_near → q_new) collision-free (checked at resolution δ):
      add q_new; if ‖q_new − q_goal‖ < η and edge free: done
```

Probabilistically complete (finds a path if one exists, eventually), zero optimality: raw paths jerk through free space, hence **shortcutting** — repeatedly pick two random points on the path, replace the segment between them with a straight edge if collision-free. Then **time parameterization**: assign timestamps so per-joint velocity/acceleration limits hold (per-segment trapezoidal profiles are enough here; TOPP-RA is the industrial-strength version).

**Kinematic trajectory optimization** (MIT ch. 6). Decision variables = waypoints $q_{1..N}$; solve

$$\min_{q_{1..N}} \sum_k \lVert q_{k+1} - q_k \rVert^2 \quad \text{s.t.} \quad q_1 = q_{\text{start}},\; q_N = q_{\text{goal}},\; q_k \in \text{limits}, \; d(q_k) \ge d_{\text{safe}}$$

where $d(q)$ is scene clearance. With penalty-form collisions, plain `scipy.optimize.minimize` (SLSQP/L-BFGS) works at this scale. Smooth, locally optimal, fast **when it converges** — but the landscape is nonconvex: a straight-line initialization through an obstacle converges to an infeasible local minimum. The standard cure: seed it with the RRT path. That hybrid — sample for topology, optimize for quality — is the punchline of the lesson. (GCS, MIT ch. 6's finale, gets global optimality through convex decomposition; read the section, skip the implementation.)

| Source | Read for |
|---|---|
| MIT *Robotic Manipulation* ch. 6 | RRT variants, kinematic trajectory optimization, and the GCS overview — the lesson's spine |
| LaValle, *Planning Algorithms* (free at lavalle.pl/planning) ch. 5 | sampling-based planning foundations; why nearest-neighbor metrics matter |
| Tutorial §2.4 | the pipeline-brittleness argument you're about to experience firsthand |

## Part 0 — The scene and the collision oracle (~1.5 h)

Everything depends on a trustworthy, fast `is_free(q)`.

1. Author `scenes/tabletop.xml`: the SO-101 on a table with 4–5 box/cylinder obstacles (a shelf edge, a mug, a wall segment) placed to create at least one narrow passage in workspace. Include a `<include>` of `so101_new_calib.xml`, don't fork it. Parameterize obstacle poses so Part 3's benchmark can randomize them.
2. `is_free(q) -> bool`: set `qpos`, `mj_forward`, inspect contacts (`data.ncon`, filtering out intentional/adjacent-link pairs via the contact geom IDs — build the exclusion list once and test it). Also `edge_free(q1, q2, delta)`: discretize at resolution $\delta = 0.02$ rad (max joint-space step) and check each point.
3. Benchmark the oracle: calls/second (expect order 10⁴/s — it's `mj_forward`, not physics). This number sets your RRT budget.
4. Sanity tests: home config free; a config driving the wrist into the table not free; symmetric ± wrist configs agree.

**✅ Checkpoint:** oracle ≥ 5,000 calls/s; the exclusion list is exact (zero false collisions over 1,000 random *visually verified* free configs — spot-check 20 in the viewer).

## Part 1 — RRT (~2 h)

1. Implement the pseudocode above: $\eta = 0.15$ rad, $p_{\text{goal}} = 0.1$, $\delta = 0.02$ rad, joint-limit-aware sampling, max 20,000 nodes. Brute-force nearest-neighbor is fine at this scale (profile before reaching for a KD-tree — and note KD-trees are awkward under per-joint scaling anyway).
2. Weight the joint-space metric: a radian of `shoulder_pan` moves the gripper much farther than a radian of `wrist_roll`. Use $\lVert \Delta q \rVert_W$ with $W$ from average task-space displacement per joint (estimate it with your Lesson 03 Jacobian at 100 random configs). Run with and without $W$ — keep both numbers for the benchmark.
3. Problem generator: random collision-free start/goal pairs with a minimum task-space separation of 25 cm (rejection sampling; verify goal reachability with Lesson 03 IK where task-space goals are used).
4. Run on 50 problems; record success (within node budget), nodes expanded, planning time, raw path length.
5. Animate one solve in the viewer: tree edges as ghost traces, then the found path.

**✅ Checkpoint:** ≥ 90% success on 50 random feasible problems within budget; the weighted metric beats unweighted on median nodes-expanded (report the ratio); the animation shows the tree actually exploring around an obstacle, not through it.

## Part 2 — Shortcut, time-parameterize, execute (~1.5 h)

1. Shortcutting: 200 iterations of random-pair rewiring with `edge_free`. Record path length before/after.
2. Time parameterization: per-segment trapezoidal profiles honoring $\dot q_{\max} = 1.5$ rad/s, $\ddot q_{\max} = 4$ rad/s² per joint (conservative SO-101-ish numbers; the exact values matter less than respecting *some*). Output: $q^{*}(t)$ sampled at 50 Hz.
3. Execute $q^{*}(t)$ through the Lesson 04 QP tracker in the full MuJoCo scene; verify zero collisions during execution (the tracker can cut corners the planner didn't — check, don't assume; if it happens, that's a finding for `RESULTS.md`, fixed by tightening $d_{\text{safe}}$ or densifying waypoints).
4. Plot joint trajectories before/after shortcutting: raw RRT (jagged) vs shortcut (clean) vs time-parameterized velocity profiles (trapezoids visible, limits respected).

**✅ Checkpoint:** shortcutting cuts median path length ≥ 30%; executed trajectories respect velocity limits (max measured $\lvert \dot q \rvert \le 1.5$ rad/s) and finish collision-free on ≥ 45/50 problems.

## Part 3 — Trajectory optimization and the hybrid (~2.5 h)

1. Direct transcription as above: $N = 30$ waypoints, smoothness objective, penalty-form clearance $\max(0, d_{\text{safe}} - d(q_k))^2$ with $d_{\text{safe}} = 2$ cm, clearance from MuJoCo contact distances (`mj_forward` + nearest signed distance per waypoint; FD gradients are acceptable — budget for it).
2. Three initialization arms on the same 50 problems: (a) straight line in joint space, (b) RRT path resampled to $N$ waypoints, (c) straight line + 3 random restarts.
3. Verify every "converged" result with the collision oracle at execution resolution — an optimizer's "success" claim is not feasibility (this distinction is the lesson).
4. The benchmark table, 50 problems × 5 planners (RRT, RRT+shortcut, trajopt-straight, trajopt-restarts, **RRT→trajopt hybrid**): success %, median path length, median wall-clock. Include the narrow-passage problems as a marked subset.
5. Side-by-side animation on one narrow-passage problem: trajopt-straight converging into the obstacle (infeasible minimum) vs the hybrid sailing through.

**✅ Checkpoint:** trajopt-straight fails on a nontrivial fraction (expect it concentrated on narrow-passage problems); the hybrid matches RRT's success rate at meaningfully shorter, smoother paths; the table makes the sample-for-topology/optimize-for-quality story legible at a glance.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `planner/` (`rrt.py`, `shortcut.py`, `timing.py`, `trajopt.py`, `collision.py`) | each importable and unit-tested; the collision oracle exclusion-list test included |
| `scenes/tabletop.xml` + generator | randomizable obstacle poses; committed with a rendered scene image |
| `benchmark.py` + `RESULTS.md` | one command regenerates the 50×5 table; `RESULTS.md` reads it: when does each planner win, in ≤ 10 sentences |
| `media/` | tree-growth animation, before/after shortcut plot, the local-minimum vs hybrid side-by-side |

## Done when

- [ ] RRT ≥ 90% success on 50 random feasible problems; hybrid matches it with better paths.
- [ ] Executed (not just planned) trajectories are collision-free and limit-respecting.
- [ ] The local-minimum failure is demonstrated on video and cured by RRT initialization.
- [ ] `RESULTS.md` answers "when each wins" with rows from your own table, not folklore.

## Self-check

1. Why is RRT probabilistically complete but arbitrarily suboptimal? What one-line change makes it asymptotically optimal, and what does it cost?
2. Why does the joint-space metric need weighting, and what geometric object would the *right* metric use?
3. Your optimizer reports convergence but the path collides. Name the two distinct ways that happens.
4. Why does seeding trajopt with an RRT path fix the local-minimum problem — what exactly does the seed provide that restarts don't?
5. This whole pipeline assumed a perfect scene model. List the three places it breaks when the mug's pose estimate is off by 2 cm — and which Phase 4 method sidesteps each.

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| RRT finds paths through obstacles | edge checked only at endpoints | discretize at $\delta$; test `edge_free` on a known-colliding midpoint |
| Every config "collides" | self-collision pairs (adjacent links, gripper pads) not excluded | build the exclusion list from the home config's contacts; test it |
| RRT slow despite fast oracle | nearest-neighbor is $O(n)$ per iteration and you have 20k nodes | it's fine to be $O(n)$ — but cap nodes, and profile before optimizing |
| Trajopt "succeeds" everywhere including through walls | penalty weight too low — collisions traded for smoothness | verify with the oracle (Part 3.3); raise the penalty or use a hard constraint with SLSQP |
| Trapezoidal profiles violate accel limits at segment joints | per-segment profiles don't blend | zero boundary velocities per segment (simple, slower) or blend; state which you chose |
| Narrow passage never solved | $\eta$ too large to enter, $\delta$ too coarse inside | halve both near failure cases; note the cost — this tuning pain is a *result*, write it down |

## Stretch

Read MIT ch. 6's Graphs of Convex Sets section and write a one-page note: what GCS guarantees that neither of your planners can, what it requires (convex decomposition of free space), and why that requirement is hard for the SO-101's 5-D C-space. No implementation — the point is knowing what "solved" looks like for this problem.

## References

- Tedrake, *Robotic Manipulation*, ch. 6 — Motion Planning (RRT, kinematic trajectory optimization, GCS). manipulation.csail.mit.edu.
- LaValle, *Planning Algorithms*, ch. 5 — free online at lavalle.pl/planning.
- Kuffner & LaValle, *RRT-Connect*, ICRA 2000 (the bidirectional variant worth knowing exists).
- LeRobot team, *Robot Learning: A Tutorial*, §2.4. arXiv:2510.12403.
