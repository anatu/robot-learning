# Lesson 07 — Motion Planning *(optional)*

This optional lesson covers the planning stage of the classical sense-plan-act pipeline, which the tutorial critiques but never shows. You will build a joint-space rapidly-exploring random tree (RRT) and a kinematic trajectory optimizer for the SO-101 in a cluttered MuJoCo scene, run both on the same set of problems, and measure where each succeeds and fails. The result you are working toward is the standard division of labour, in which a sampling-based planner supplies the topology of a path and an optimizer supplies its quality, established from rows in your own benchmark table rather than taken on authority. (A correction to the original scaffold: this material is MIT *Robotic Manipulation* **ch. 6**, "Motion Planning.")

| | |
|---|---|
| **Phase** | 2 — Classical core (optional) |
| **Time** | 1 session (4–6 h desk time), all Mac-local |
| **Cost** | $0 |
| **Prerequisites** | 03 (`kinematics.py`: `fk`, `jacobian`), 04 (`controllers.py`, needed only for Going deeper) |
| **Feeds into** | H6 (mobile bases replan constantly); the Phase 4 argument for why end-to-end learning replaced sense-plan-act |
| **Skip criteria** | The learning-based track never needs this lesson, but it sharpens the argument for why end-to-end methods won, and H6 reuses it |

## Learning objectives

After this lesson you can:

1. **Explain** why RRT is probabilistically complete but arbitrarily suboptimal, and what shortcutting buys back.
2. **Predict** the effect of a weighted joint-space metric on tree growth, and confirm it on your own problems.
3. **Diagnose** an optimizer that reports convergence while its path collides, in both of the ways that can happen.
4. **Decide** when each planner wins from a benchmark you ran, and state the completeness-versus-quality tradeoff with reference to rows in your table.

## Principles

### Configuration space and the collision oracle

Motion planning takes place in configuration space $\mathcal{C}$, which for a manipulator is joint space, rather than in the task space where obstacles are described. For the SO-101, $\mathcal{C}$ is a box in $\mathbb{R}^5$ bounded by the joint limits. Obstacles in the workspace map to a region $\mathcal{C}_{\text{obs}}$ of configurations in which some part of the arm intersects an obstacle. That region has a complicated shape and is never computed explicitly. Instead, a planner interacts with it only through a collision checker, a function `is_free(q)` that answers whether a single configuration is collision-free. Because every planner below makes thousands of such queries, the speed and correctness of the oracle set the budget for everything else.

### Rapidly-exploring random trees

RRT (LaValle; MIT ch. 6) explores configuration space by growing a tree from the start configuration:

```
repeat:
  q_rand ← goal with prob. p_goal, else Uniform(C)
  q_near ← nearest tree node to q_rand
  q_new  ← q_near + η · (q_rand − q_near)/‖·‖     # steer, step size η
  if edge (q_near → q_new) collision-free (checked at resolution δ):
      add q_new; if ‖q_new − q_goal‖ < η and edge free: done
```

The algorithm is probabilistically complete, meaning that if a path exists it will be found eventually with probability one, but it makes no attempt at optimality: raw RRT paths wander through free space with unnecessary detours. The standard remedy is shortcutting, which repeatedly picks two random points on the path and replaces the segment between them with a straight edge whenever that edge is collision-free. A second detail that matters more than it first appears is the metric used to find the nearest tree node. A radian of `shoulder_pan` moves the gripper much farther than a radian of `wrist_roll`, so an unweighted Euclidean metric in joint space measures distance in the wrong units and explores the wrong directions first.

### Kinematic trajectory optimization

The alternative to sampling is to treat the path as a decision variable and optimize it. In kinematic trajectory optimization (MIT ch. 6), the path is represented by waypoints $q_{1..N}$ and the problem is

$$\min_{q_{1..N}} \sum_k \lVert q_{k+1} - q_k \rVert^2 \quad \text{s.t.} \quad q_1 = q_{\text{start}},\; q_N = q_{\text{goal}},\; q_k \in \text{limits}, \; d(q_k) \ge d_{\text{safe}},$$

where $d(q)$ is the clearance between the arm and the scene. With collisions handled as penalties rather than hard constraints, an ordinary solver such as `scipy.optimize.minimize` (SLSQP or L-BFGS) is adequate at this scale. When it converges to a feasible solution the result is smooth and locally optimal, and it is fast. The difficulty is that the landscape is nonconvex. A straight-line initialization that passes through an obstacle will typically converge to a local minimum that is still inside the obstacle, because the penalty gradient pushes the path outward but the smoothness objective pulls it back and the optimizer has no way to know that a different homotopy class exists. The standard cure is to seed the optimizer with an RRT path: sampling supplies the topology, and optimization supplies the quality. Graphs of Convex Sets, the final topic of MIT ch. 6, achieves global optimality through a convex decomposition of free space; read that section, but do not implement it here.

### Convergence is not feasibility

An optimizer's report that it converged is not evidence that its path is collision-free, for two distinct reasons. With penalty-form collision terms, a low penalty weight lets the optimizer trade clearance for smoothness and settle on a path that touches or crosses an obstacle. And because the clearance constraint is evaluated only at waypoints, a path can be clear at every waypoint and still collide between them. For both reasons, every result that an optimizer returns must be re-verified with the collision oracle at execution resolution, and the two flags, converged and verified, should be reported separately.

**Carry forward**

- Planning lives in configuration space, and the collision oracle is the only window into $\mathcal{C}_{\text{obs}}$; its speed sets the planning budget.
- RRT is probabilistically complete but not optimal, so its paths are shortcut afterwards; the rewiring step of RRT* buys asymptotic optimality at a per-iteration cost.
- Trajectory optimization is locally optimal and smooth but blind to topology, so it should be seeded with a sample-based path.
- An optimizer's convergence flag does not establish that a path is collision-free, because the penalty may be too weak or the constraint may hold only at waypoints; re-verify with the oracle.
- The whole pipeline assumes a perfect scene model, and a 2 cm error in an object's pose breaks it in three places; the learned methods of Phase 4 exist largely because of that fragility.

| Source | Read for |
|---|---|
| MIT *Robotic Manipulation* ch. 6 | RRT variants, kinematic trajectory optimization, and the GCS overview, which together form the spine of this lesson |
| LaValle, *Planning Algorithms* (free at lavalle.pl/planning) ch. 5 | the foundations of sampling-based planning, and why nearest-neighbour metrics matter |
| Tutorial §2.4 | the pipeline-brittleness argument you are about to experience firsthand |

## Exercise 1 — Build the scene and the collision oracle [Build]

A fast and exact `is_free(q)` is the interface to $\mathcal{C}_{\text{obs}}$, and everything else in the lesson is built on it. In this exercise you author the scene and the oracle, and you verify the oracle's exclusion list, because an oracle that reports false collisions between adjacent links will make every configuration appear blocked. Write the specification and have an AI tool draft the code.

- `scenes/tabletop.xml`: the SO-101 on a table with four or five box or cylinder obstacles (a shelf edge, a mug, a wall segment) placed so that at least one narrow passage exists in the workspace. Use `<include>` for `so101_new_calib.xml` rather than forking it. Parameterize the obstacle poses so that Exercise 7 can randomize them, and commit a rendered image of the scene.
- `planner/collision.py`: `is_free(q) -> bool`, which sets `qpos`, calls `mj_forward`, and inspects the contacts in `data.ncon`, filtering out intentional and adjacent-link pairs through an exclusion list built once from the contacts present in the home configuration; and `edge_free(q1, q2, delta)`, which discretizes the segment at $\delta = 0.02$ rad of maximum joint-space step and checks each point.
- The check, in `checks.py`: the home configuration is free; a configuration that drives the wrist into the table is not; symmetric positive and negative wrist configurations agree; there are zero false collisions over 1,000 random configurations that you have visually verified as free (spot-check 20 of them in the viewer); and the number of oracle calls per second is printed.

**✅ Checkpoint:** the oracle sustains at least 5,000 calls per second (on the order of $10^4$ is typical, since each call is `mj_forward` rather than a physics step), and the exclusion-list check passes.

## Exercise 2 — Check edges only at their endpoints [Diagnose]

An edge check that tests only its two endpoints produces a planner that walks through thin obstacles, and you should be able to predict exactly how the resulting paths will look and which metrics they will improve. This exercise introduces that bug deliberately, in a copy of the oracle, so that you recognize it when it appears in your own code later.

1. Before running, write in `RESULTS.md` what RRT paths would look like near thin obstacles such as the shelf edge if `edge_free` checked only `q1` and `q2`, and which of the two path-quality numbers, success rate or path length, would appear to improve as a result.
2. Have the AI tool produce a variant `edge_free_endpoints_only`. Run the RRT from Exercise 3 with it on five problems whose solutions pass near the shelf edge, then verify each returned path at $\delta = 0.02$ with the correct oracle.
3. Record in `RESULTS.md` how many of the apparently successful paths collide and where along each path the collision occurs.

**✅ Checkpoint:** at least one endpoints-only path passes through the shelf, and the reconciliation in `RESULTS.md` names the mechanism.

## Exercise 3 — Implement RRT and shortcutting [Build]

This exercise builds the sampling-based planner. The algorithm provides probabilistic completeness, and the post-processing step provides path quality; keeping the two separate makes it possible to measure what each contributes. Write the specification and have an AI tool draft the code.

- `planner/rrt.py`: the pseudocode from the Principles section with $\eta = 0.15$ rad, $p_{\text{goal}} = 0.1$, $\delta = 0.02$ rad, joint-limit-aware sampling, and a cap of 20,000 nodes. Use brute-force nearest-neighbour search; profile before reaching for a KD-tree, which is in any case awkward under per-joint scaling. Accept a `metric_weights` argument that is `None` for the Euclidean metric or a diagonal $W$.
- `planner/shortcut.py`: 200 iterations of random-pair rewiring using `edge_free`, returning the path and its length before and after.
- A problem generator that draws random collision-free start and goal pairs with a minimum task-space separation of 25 cm (rejection sampling through Lesson 03's `fk`), and marks as narrow-passage candidates those problems whose straight-line joint-space segment collides.
- Per solve, record success within the node budget, nodes expanded, planning time, and the raw and shortcut path lengths.

**✅ Checkpoint:** at least 90% success on 20 random feasible problems within the node budget; shortcutting reduces the median path length by at least 30%; and one animation shows the tree exploring around an obstacle rather than through it.

## Exercise 4 — Weight the joint-space metric [Predict → Run]

The nearest-neighbour metric determines which directions the tree explores first, and the appropriate weights can be derived from the Jacobian, since the Jacobian relates joint motion to end-effector motion. In this exercise you build a weighted metric and predict its effect on tree growth before measuring it.

1. Build $W$ from the average task-space displacement per joint: evaluate Lesson 03's `jacobian` at 100 random configurations and take the mean column norm for each joint.
2. Before running, write in `RESULTS.md` your expected ratio of median nodes expanded, weighted to unweighted, on the same 20 problems, giving both its direction and its rough magnitude, and say which joint's weight you expect to dominate.
3. Run both metrics on the 20 problems with the same random seeds, and tabulate success, median nodes, and median time for each.
4. Reconcile the table with your prediction in `RESULTS.md`.

**✅ Checkpoint:** the weighted metric beats the unweighted one on median nodes expanded, and both the measured ratio and your predicted ratio appear in the report.

## Exercise 5 — Implement kinematic trajectory optimization [Build]

Direct transcription produces smooth, locally optimal paths when it converges to a feasible solution, and this exercise builds the optimizer with post-verification built in so that the two outcomes are never confused. Write the specification for `planner/trajopt.py` and have an AI tool draft it.

- $N = 30$ waypoints; the smoothness objective from the Principles section; a penalty-form clearance term $\max(0, d_{\text{safe}} - d(q_k))^2$ with $d_{\text{safe}} = 2$ cm, where the clearance comes from MuJoCo contact distances (`mj_forward` and the nearest signed distance per waypoint; finite-difference gradients are acceptable, but budget for their cost); joint-limit bounds; and `scipy.optimize.minimize` with SLSQP or L-BFGS.
- `optimize(q_init_path) -> (path, converged)`, with two supported initializations: a straight line in joint space, or an RRT path resampled to $N$ waypoints.
- Mandatory post-verification: every returned path is re-checked with `edge_free` at $\delta = 0.02$, and the result is reported separately from the optimizer's `converged` flag.

**✅ Checkpoint:** on an obstacle-free problem, the optimizer converges from a straight-line initialization and passes verification, and both flags (`converged`, `verified`) appear in the log.

## Exercise 6 — Demonstrate the local minimum and its cure [Predict → Run]

Trajectory optimization cannot discover a different homotopy class on its own, and a sample-based seed supplies exactly the information that random restarts cannot. In this exercise you pick a problem where the straight-line initialization passes through an obstacle and compare three ways of initializing the optimizer.

1. Pick one narrow-passage problem from Exercise 3's generator.
2. Before running, write in `RESULTS.md` what you expect the optimizer to do from a straight-line initialization: whether it converges, whether the result verifies, and where the path sits relative to the obstacle. Then write whether three random restarts should fix it, and what the RRT-seeded run should return instead.
3. Run the three arms: (a) the straight line; (b) the straight line plus three random restarts; (c) the RRT path as the seed. Produce a side-by-side plot or animation of (a) against (c).
4. Reconcile in `RESULTS.md`, and state what the seed provides that restarts do not.

**✅ Checkpoint:** arm (a) either converges into the obstacle (converged true, verified false) or fails outright, and arm (c) is verified and is shorter and smoother than the raw RRT path.

## Exercise 7 — Run the benchmark [Predict → Run]

The tradeoff between completeness and path quality is something to measure rather than assert, and this exercise measures it on a fixed set of problems. Writing down the expected ordering first makes the table a test of your understanding rather than a description of it.

1. Before running, write in `RESULTS.md` the ordering you expect on success rate, median path length, and median wall-clock time across the three planners (RRT with shortcutting, trajectory optimization from a straight line, and the RRT-seeded hybrid) on 20 problems, and say on which subset of problems you expect the straight-line optimizer to fail.
2. Write `benchmark.py` so that one command runs the 20 × 3 grid, with the narrow-passage problems marked as a subset, and prints the table.
3. Reconcile the table with your prediction in `RESULTS.md`.

**✅ Checkpoint:** the straight-line optimizer fails on a nontrivial fraction of problems, concentrated on the narrow-passage subset, and the hybrid matches RRT's success rate with meaningfully shorter and smoother paths.

## Exercise 8 — Decide when each planner wins [Decide]

Using your own table, write at most ten sentences in `RESULTS.md` saying when you would run RRT with shortcutting alone, when you would run trajectory optimization alone, and when you would run the hybrid, with each claim pointing to a row of the table. Then write a closing paragraph listing the three places in this pipeline that break when the mug's pose estimate is off by 2 cm, and naming the Phase 4 method that sidesteps each.

**✅ Checkpoint:** every claim about when a planner wins cites a row, and the 2 cm paragraph names three points of failure.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `planner/` (`collision.py`, `rrt.py`, `shortcut.py`, `trajopt.py`) | importable; `trajopt.py` reports `converged` and `verified` separately |
| `scenes/tabletop.xml` and the problem generator | randomizable obstacle poses; committed with a rendered scene image |
| `checks.py`, `benchmark.py` | the oracle checks and the 20 × 3 table, each from one command |
| `media/` | one tree-growth animation; the local-minimum versus hybrid side-by-side |
| `RESULTS.md` | Exercise 2, 4, 6, and 7 predictions with reconciliations; the Exercise 8 decision; the 2 cm paragraph |

## Done when

- [ ] The oracle sustains at least 5,000 calls per second with an exact exclusion list.
- [ ] RRT achieves at least 90% success on 20 random feasible problems, and the hybrid matches it with better paths.
- [ ] The endpoints-only bug and the local-minimum failure are both demonstrated and explained.
- [ ] All predictions were written before their runs and reconciled afterwards.
- [ ] `RESULTS.md` answers when each planner wins with rows from your own table.

## Self-check

1. Why is RRT probabilistically complete but arbitrarily suboptimal? What one-line change makes it asymptotically optimal, and what does that change cost?
2. Why does the joint-space metric need weighting, and what geometric object would the right metric use?
3. Your optimizer reports convergence but the path collides. Name the two distinct ways that happens.
4. Why does seeding trajectory optimization with an RRT path fix the local-minimum problem? What exactly does the seed provide that restarts do not?
5. This whole pipeline assumed a perfect scene model. List the three places it breaks when the mug's pose estimate is off by 2 cm, and name the Phase 4 method that sidesteps each.

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| RRT finds paths through obstacles | edges checked only at their endpoints | discretize at $\delta$; Exercise 2 has you experience this deliberately |
| Every configuration appears to collide | self-collision pairs (adjacent links, gripper pads) not excluded | build the exclusion list from the home configuration's contacts, and test it |
| RRT slow despite a fast oracle | nearest-neighbour search is $O(n)$ per iteration with 20k nodes | $O(n)$ is acceptable here; cap the node count and profile before optimizing |
| The optimizer succeeds everywhere, including through walls | penalty weight too low, so collisions are traded for smoothness | the built-in verification catches it; raise the penalty, or use a hard constraint with SLSQP |
| A narrow passage is never solved | $\eta$ too large to enter it, $\delta$ too coarse inside it | halve both near the failing cases, and record the tuning cost in `RESULTS.md`, because it is itself a finding |

## Going deeper

- **Execute the plan.** Time-parameterize the shortcut paths with per-segment trapezoidal profiles that honour $\dot q_{\max} = 1.5$ rad/s and $\ddot q_{\max} = 4$ rad/s² per joint (either zero boundary velocities per segment or blended segments; state which), sample $q^{*}(t)$ at 50 Hz, and execute it through Lesson 04's QP tracker in the full MuJoCo scene. Verify that execution, not just planning, is collision-free, since the tracker can cut corners the planner did not; if it does, tighten $d_{\text{safe}}$ or densify the waypoints. Expect at least 45 of 50 problems to execute collision-free with $\max \lvert \dot q \rvert \le 1.5$ rad/s.
- **RRT-Connect and RRT*.** Add the bidirectional variant and the rewiring step, then re-run the benchmark.
- **Graphs of Convex Sets.** Read MIT ch. 6's section on GCS and write a one-page note on what it guarantees that neither of your planners can, what it requires (a convex decomposition of free space), and why that requirement is difficult for the SO-101's five-dimensional configuration space.

## References

- Tedrake, *Robotic Manipulation*, ch. 6 — Motion Planning (RRT, kinematic trajectory optimization, GCS). manipulation.csail.mit.edu.
- LaValle, *Planning Algorithms*, ch. 5 — free online at lavalle.pl/planning.
- Kuffner & LaValle, *RRT-Connect*, ICRA 2000 (the bidirectional variant).
- LeRobot team, *Robot Learning: A Tutorial*, §2.4. arXiv:2510.12403.
