# Lesson 06 — Grasp Mechanics *(optional)*

The analytic lineage behind learned grasping — friction cones, the grasp map, closure as feasibility, force optimization as a convex program — proven by deriving one threshold by hand and watching your code flip exactly there. When H3's policies drop objects, this is how you know *which way* they failed.

| | |
|---|---|
| **Phase** | 2 — Classical core (optional) |
| **Time** | 1 session (3–4 h desk time), all Mac-local |
| **Cost** | $0 |
| **Prerequisites** | 03 (frames/wrenches comfort); basic convex optimization (`cvxpy`) |
| **Feeds into** | H3's failure taxonomy (grasp-miss vs slip vs perception); Lesson 07's scene design |
| **Skip criteria** | Skip if the goal is speed-to-modern-methods; do it if grasping intuition feels rusty |

## Learning objectives

After this lesson you can:

1. **Explain** what the grasp map $G$ is and what lives in its column space, and state your torque sign convention from memory.
2. **Derive** the friction threshold for the antipodal case by hand, and predict from which side a linearized cone test converges to it.
3. **Predict** where a pinch grasp's required normal force peaks as an external disturbance rotates, and confirm it with an SOCP.
4. **Decide**, for an infeasible load, which of the three physically distinct fixes applies — and which one a learned policy implicitly chooses.

## Principles

**Contact models.** A contact transmits force to the object through a model: *frictionless point* (normal force only, 1-D), *point contact with friction* (PCWF: force anywhere in the friction cone, 3-D in 3D space), *soft finger* (adds torsion about the normal). This course uses PCWF.

**The friction cone.** Coulomb friction at contact $i$ with normal $\hat n_i$ and coefficient $\mu$: the transmissible force satisfies

$$\lVert f_{t,i} \rVert_2 \le \mu\, f_{n,i}, \qquad f_{n,i} \ge 0$$

— a second-order (quadratic) cone $\mathcal{FC}_i$. Linearizing it as an $m$-edge pyramid turns cone constraints into linear ones (LP-friendly); keeping it exact keeps you in SOCP-land. The pyramid is *inscribed*, so linearized tests are conservative — Exercise 3 makes you predict which way.

**The grasp map.** Stack the contacts: contact $i$ at point $p_i$ (object frame) with rotation $R_i$ mapping contact frame to object frame contributes a wrench $w_i = \begin{bmatrix} R_i f_i \\ p_i \times R_i f_i \end{bmatrix}$. Collecting columns gives $G \in \mathbb{R}^{6 \times 3k}$ with total object wrench $w = G f$. Everything below is a statement about $\{Gf : f_i \in \mathcal{FC}_i\}$. Every bug in this lesson is a sign bug in the $p_i \times$ term; fix the convention once, in a docstring.

**Form closure** (geometry only, $\mu = 0$): the frictionless normals alone can resist any wrench ⇔ the wrench cone spans $\mathbb{R}^6$ positively ⇔ the origin is in the *interior* of the convex hull of the contact normal wrenches. Testable as an LP: maximize $\delta$ s.t. $G_n \lambda = 0$, $\lambda \ge \delta \mathbf{1}$, $\mathbf{1}^\top \lambda \le k$; form closure ⇔ optimal $\delta > 0$ (Lynch & Park ch. 12 give this exact program). Needs ≥ 7 frictionless contacts in 3D (≥ 4 in the plane).

**Force closure** (with friction): $\{Gf : f_i \in \mathcal{FC}_i\} = \mathbb{R}^6$ — the grasp can resist any external wrench with some admissible contact forces. With linearized cones this reduces to form-closure-style LP tests on the cone edges. The classic hand-derivable case: **two antipodal PCWF contacts achieve force closure iff the line between them lies inside both friction cones** — which gives a closed-form minimum $\mu$ for a given geometry. That derivation is your ground truth (it is also CS237B HW2's warm-up).

**Grasp force optimization** (CS237B HW2, Problem 2). Given an external wrench $w_{\text{ext}}$ (gravity, inertial load), find contact forces that resist it with minimal effort:

$$\min_{f}\; \max_i f_{n,i} \quad \text{(or } \lVert f \rVert_2\text{)} \qquad \text{s.t.} \qquad G f = -w_{\text{ext}}, \qquad \lVert f_{t,i} \rVert_2 \le \mu f_{n,i} \;\; \forall i$$

— an SOCP, 15 lines of `cvxpy`. Infeasibility *is* the answer "this grasp cannot hold this load."

**Carry forward**

- The grasp map turns contact forces into an object wrench; closure is a statement about whether the image of the friction cones under $G$ fills wrench space.
- Form closure is geometry (7 contacts in 3D); force closure is friction (2 antipodal contacts suffice iff the line between them is inside both cones).
- Linearized cones are conservative: an $m$-edge pyramid is inscribed in the true cone.
- Holding a load is a feasibility question; the SOCP's infeasibility is a physical verdict, and the three ways out are more friction, more squeeze, or different contacts.
- None of this sees deformables, uncertainty, or dynamics — the reason Phase 4 learns instead.

| Source | Read for |
|---|---|
| CS237B W25 HW2 PDF (web.stanford.edu/class/cs237b — public) | the exact antipodal-μ derivation and the SOCP formulation with starter/test code |
| MIT *Robotic Manipulation* ch. 5 (Bin Picking) | friction cones, the contact wrench cone, and antipodal grasp heuristics in the wild |
| Lynch & Park ch. 12 | form/force closure formalism and the LP test |

## Exercise 1 — The antipodal threshold and the sign convention [Derive]

The principle: closure for two contacts is a geometric condition you can solve in closed form, and every later number is checked against it.

1. On paper: two antipodal PCWF contacts on a disk of radius $r$, contact line offset $c$ from the center. Derive $\mu_{\min}$ such that the connecting line lies inside both friction cones. Mind that $\tan^{-1}\mu$ is the cone *half*-angle.
2. On paper: three hand-check wrenches — a single contact at the origin (pure force, zero torque); two opposed contacts on a unit-width square (their normal wrenches cancel); a contact offset by $d$ in $x$ with force $f_y$ (torque $\tau_z = \pm f_y d$ — pick the sign, write it down, it becomes the module docstring).
3. Write all three results plus $\mu_{\min}(r, c)$ into `RESULTS.md` before any code exists.

**✅ Checkpoint:** $\mu_{\min}$ is a formula in $r$ and $c$; the torque sign convention is written in one sentence.

## Exercise 2 — Grasp map, closure tests, force SOCP [Build]

The principle: closure and load-holding are both convex feasibility problems over the same object $\{Gf : f_i \in \mathcal{FC}_i\}$. Spec for `grasp.py`:

- `contact_wrench(p, R, f) -> w` and `grasp_map(contacts) -> G` for planar (3-D wrenches) and spatial (6-D) cases; torque convention from Exercise 1 in the docstring.
- `is_form_closure(contacts) -> (bool, delta)`: the LP above via `cvxpy`.
- `is_force_closure(contacts, mu, m_edges) -> bool`: linearize each cone into $m$ edges, positive-span test on edge wrenches.
- `optimize_forces(contacts, mu, w_ext, objective) -> f | None`: the SOCP, both objectives (min-max normal force via the epigraph form `min t` s.t. $f_{n,i} \le t$; min total $\lVert f \rVert_2$). Solve with ECOS or Clarabel; return `None` on infeasibility.
- The check (`checks.py`): the three Exercise 1 hand-check wrenches reproduce; 4 orthogonal frictionless contacts on a planar square → form closure, 3 → not; translating all contact points *and* the external wrench reference by the same offset leaves solvability invariant; every feasible SOCP solve has strictly positive cone slack.

**✅ Checkpoint:** `checks.py` green; `prob.solver_stats` confirms an SOCP-capable solver.

## Exercise 3 — The μ-flip and the side it converges from [Predict → Run]

The principle: linearization is conservative, and you should know from which side before you plot it.

1. **Write first:** at what $\mu$ (from Exercise 1) does `is_force_closure` flip for the antipodal disk case; and for $m \in \{4, 8, 16, 64\}$ edges, does the numerical flip point sit *above* or *below* the analytic value, and does it approach monotonically? Name the reason (inscribed pyramid).
2. Sweep $\mu$ in steps of $10^{-3}$ for each $m$; record the flip point.
3. Plot flip point vs $m$ with the analytic line; reconcile in `RESULTS.md`.

**✅ Checkpoint:** at $m = 64$ the flip is within one step ($10^{-3}$) of your derivation; the convergence is monotone from the side you predicted.

## Exercise 4 — Holding a box against a rotating disturbance [Predict → Run]

The principle: required grasp force is a function of disturbance direction with peaks physics can locate before the solver does.

1. Scenario: a 200 g box held by a 2-contact pinch (SO-101 gripper geometry: contact patches ~1 cm apart per jaw face, $\mu \approx 0.6$ rubber-on-cardboard). External wrench = gravity + a 1 N disturbance force rotated through 360° in the vertical plane.
2. **Write first:** the disturbance angles at which the min-max normal force peaks, and whether any angular range is infeasible at $\mu = 0.6$. Then, for $\mu \in \{0.2, 0.4, 0.8\}$, whether the infeasible range grows or shrinks and roughly by how much.
3. Solve the SOCP over the angle sweep at each $\mu$; polar plot of required max normal force vs angle, infeasible ranges in red; contact-force arrows inside their cones for 3 representative angles.
4. Reconcile in `RESULTS.md`.

**✅ Checkpoint:** the polar plot is smooth with peaks where you predicted (disturbance orthogonal to the grasp axis); the infeasible range grows as μ shrinks, quantified.

## Exercise 5 — Three ways out [Decide]

The principle: an infeasible SOCP has exactly three physical remedies, and they cost different things.

1. Take the widest infeasible range from Exercise 4. For each remedy — raise $\mu$ (surface material), raise the allowed normal force (squeeze harder), move the contacts (regrasp) — compute the minimum change that makes the whole sweep feasible.
2. Decide which one you would specify for the SO-101 gripper on cardboard boxes, and defend it in `RESULTS.md` with the three numbers.
3. State which remedy a learned pick policy chooses implicitly, and why it cannot choose the others.

**✅ Checkpoint:** three quantified remedies, one decision, one sentence on the learned-policy analogue.

## Exercise 6 — The verdict table and the limits [Write]

In `RESULTS.md`: a table of ≥ 6 contact configurations (planar + 3D) with closure verdicts and a one-line intuition each; the μ-sensitivity reading from Exercise 4; and 5 sentences on what this machinery *cannot* see (deformables, uncertainty, dynamics) — the reason Phase 4 learns instead.

**✅ Checkpoint:** the table has ≥ 6 rows; the limits paragraph names all three blind spots.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `grasp.py` | `grasp_map`, closure tests, `optimize_forces` — importable, torque convention in the docstring |
| `checks.py` | hand-derived cases, translation-invariance property, cone-slack check — one command, green |
| `plots/` | flip-point-vs-$m$ plot, polar disturbance plot (all μ), cone-arrow renders |
| `RESULTS.md` | Exercise 1 derivation; Exercise 3/4 predictions with reconciliations; the Exercise 5 decision; the verdict table; the limits paragraph |

## Done when

- [ ] `checks.py` green, including the analytic μ threshold at $m = 64$.
- [ ] Exercise 3 and 4 predictions are written before the runs and reconciled after.
- [ ] The SOCP finds valid in-cone force distributions and correctly reports infeasibility.
- [ ] `RESULTS.md` closes with the decision and the limits-of-analytic-grasping paragraph.

## Self-check

1. Why does form closure need ≥ 7 contacts in 3D but force closure only 2?
2. What convex object does the set of admissible contact forces form, and why does linearizing it make closure tests *conservative*?
3. Your SOCP is infeasible for some disturbance. What are the three physically distinct fixes, and which does a learned policy implicitly choose?
4. Where exactly does the $p_i \times$ in the grasp map come from?
5. H3's policy drops a box during transport but never during lift. Which analysis from this lesson explains the difference?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Everything is force-closed | normals flipped inward, or forgot $f_n \ge 0$ | render the normals; assert nonnegativity constraints are present |
| Nothing is force-closed | wrench torque reference point inconsistent between contacts | one object-frame origin for all torques (the translation-invariance check catches this) |
| μ-flip off by 2× | cone half-angle vs full-angle confusion ($\tan^{-1}\mu$ is the *half*-angle) | rederive with a picture; test at μ where you did the algebra |
| `cvxpy` says SOCP is unbounded | objective missing, or min-max not epigraph-formulated | `min t` s.t. $f_{n,i} \le t$ — check the epigraph transform |
| Solver flips LP/SOCP verdicts near the threshold | numerical tolerance at the boundary | treat $\delta$ within ±1e-6 as "marginal," report it as such |

## Going deeper

- **Antipodal grasp scoring on a point cloud.** Sample a mug or box mesh (`trimesh.sample.sample_surface`, scaled to the SO-101's ~6 cm max opening); generate random point pairs with distance ≤ opening, anti-aligned normals, and the connecting line inside both cones (Exercise 2's test, reused); score each by the SOCP's min-max normal force under gravity + a disturbance set plus a robustness score (fraction of ±5 mm contact perturbations that stay force-closed); render top/bottom-10. Expect ≥ 100 valid candidates and Spearman ρ > 0.5 between the two scores; top grasps on the rim edge only means curvature-blind sampling — a real limitation, note it.
- **Ferrari–Canny $\epsilon$-metric.** Radius of the largest origin-centered ball inside the unit grasp wrench space, via `scipy.spatial.ConvexHull` on the edge wrenches; re-rank the point-cloud grasps and find one where the metrics disagree, explained geometrically.

## References

- Stanford CS237B *Principles of Robot Autonomy II*, W25 Problem Set 2 (public PDF + starter code). web.stanford.edu/class/cs237b.
- Tedrake, *Robotic Manipulation*, ch. 5 — friction cones, contact wrench cone. manipulation.csail.mit.edu.
- Lynch & Park, *Modern Robotics*, ch. 12 (Grasping and Manipulation) — closure formalism and LP tests.
- Ferrari & Canny, *Planning Optimal Grasps*, ICRA 1992 — the $\epsilon$-metric (Going deeper).
