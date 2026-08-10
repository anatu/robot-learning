# Lesson 06 — Grasp Mechanics *(optional)*

The analytic lineage behind learned grasping: contact models, friction cones, force/form closure as feasibility problems, and grasp force optimization as a second-order cone program. Built on Stanford CS237B's public HW2, which poses exactly these problems with starter code. When H3's policies drop objects, this lesson is why you'll know *which way* they failed.

| | |
|---|---|
| **Phase** | 2 — Classical core (optional) |
| **Time** | 1 session (4–6 h), all Mac-local |
| **Cost** | $0 |
| **Prerequisites** | 03 (frames/wrenches comfort); basic convex optimization (`cvxpy`) |
| **Feeds into** | H3's failure taxonomy (grasp-miss vs slip vs perception); Lesson 07's scene design |
| **Skip criteria** | Skip if the goal is speed-to-modern-methods; do it if grasping intuition feels rusty |

## Learning objectives

After this lesson you can:

1. **Build** the grasp map $G$ for arbitrary contact sets and state precisely what lives in its column space.
2. **Test** form closure (an LP) and force closure (a cone feasibility problem) and derive the friction threshold for the antipodal case by hand.
3. **Formulate and solve** grasp force optimization as an SOCP with friction-cone and equilibrium constraints.
4. **Score** candidate antipodal grasps on a point cloud and defend the ranking with the closure machinery.

## Background

**Contact models.** A contact transmits force to the object through a model: *frictionless point* (normal force only, 1-D), *point contact with friction* (PCWF: force anywhere in the friction cone, 3-D in 3D space), *soft finger* (adds torsion about the normal). This course uses PCWF.

**The friction cone.** Coulomb friction at contact $i$ with normal $\hat n_i$ and coefficient $\mu$: the transmissible force satisfies

$$\lVert f_{t,i} \rVert_2 \le \mu\, f_{n,i}, \qquad f_{n,i} \ge 0$$

— a second-order (quadratic) cone $\mathcal{FC}_i$. Linearizing it as an $m$-edge pyramid turns cone constraints into linear ones (LP-friendly); keeping it exact keeps you in SOCP-land. Do both; compare.

**The grasp map.** Stack the contacts: contact $i$ at point $p_i$ (object frame) with rotation $R_i$ mapping contact frame to object frame contributes a wrench $w_i = \begin{bmatrix} R_i f_i \\ p_i \times R_i f_i \end{bmatrix}$. Collecting columns gives $G \in \mathbb{R}^{6 \times 3k}$ with total object wrench $w = G f$. Everything below is a statement about $\{Gf : f_i \in \mathcal{FC}_i\}$.

**Form closure** (geometry only, $\mu = 0$): the frictionless normals alone can resist any wrench ⇔ the wrench cone spans $\mathbb{R}^6$ positively ⇔ the origin is in the *interior* of the convex hull of the contact normal wrenches. Testable as an LP: maximize $\delta$ s.t. $G_n \lambda = 0$, $\lambda \ge \delta \mathbf{1}$, $\mathbf{1}^\top \lambda \le k$; form closure ⇔ optimal $\delta > 0$ (Lynch & Park ch. 12 give this exact program). Needs ≥ 7 frictionless contacts in 3D (≥ 4 in the plane).

**Force closure** (with friction): $\{Gf : f_i \in \mathcal{FC}_i\} = \mathbb{R}^6$ — the grasp can resist any external wrench with some admissible contact forces. With linearized cones this reduces to form-closure-style LP tests on the cone edges. The classic hand-derivable case: **two antipodal PCWF contacts achieve force closure iff the line between them lies inside both friction cones** — which gives a closed-form minimum $\mu$ for a given geometry. That derivation is your ground truth (it's also CS237B HW2's warm-up).

**Grasp force optimization** (CS237B HW2, Problem 2). Given an external wrench $w_{\text{ext}}$ (gravity, inertial load), find contact forces that resist it with minimal effort:

$$\min_{f}\; \max_i f_{n,i} \quad \text{(or } \lVert f \rVert_2\text{)} \qquad \text{s.t.} \qquad G f = -w_{\text{ext}}, \qquad \lVert f_{t,i} \rVert_2 \le \mu f_{n,i} \;\; \forall i$$

— an SOCP, 15 lines of `cvxpy`. Infeasibility *is* the answer "this grasp cannot hold this load."

| Source | Read for |
|---|---|
| CS237B W25 HW2 PDF (web.stanford.edu/class/cs237b — public) | the exact antipodal-μ derivation and the SOCP formulation with starter/test code |
| MIT *Robotic Manipulation* ch. 5 (Bin Picking) | friction cones, the contact wrench cone, and antipodal grasp heuristics in the wild |
| Lynch & Park ch. 12 | form/force closure formalism and the LP test |

## Part 0 — Wrenches and the grasp map (~1 h)

1. `contact_wrench(p, R, f) -> w` and `grasp_map(contacts) -> G` for planar (3-D wrenches) and spatial (6-D) cases.
2. Hand-check tests, all derivable on paper: a single contact at the origin (pure force, zero torque); two opposed contacts on a unit-width square (their normal wrenches cancel); a contact offset in $x$ producing exactly the expected $\tau_z = -f_y \cdot d$... wait, check your own sign convention and encode it in a test.
3. Property test: translating all contact points *and* the external wrench reference by the same offset leaves solvability invariant.

**✅ Checkpoint:** hand-check tests green; you can state your torque sign convention from memory (write it in the module docstring — every bug in this lesson is a sign bug).

## Part 1 — Closure tests (~1.5 h)

1. `is_form_closure(contacts) -> (bool, delta)`: the LP above via `cvxpy`.
2. `is_force_closure(contacts, mu, m_edges) -> bool`: linearize each cone into $m$ edges, then run the positive-span test on edge wrenches.
3. Validate against paper: (a) 4 orthogonal frictionless contacts on a planar square → form closure; 3 → not; (b) two antipodal contacts on a disk of radius $r$, contact-line offset $c$ from center: derive $\mu_{\min}$ by hand, then confirm your code flips exactly at it (sweep $\mu$ in steps of $10^{-3}$, assert the flip is within one step of the derived value); (c) $m$-edge sensitivity — the flip point moves with $m \in \{4, 8, 16, 64\}$; plot convergence toward the analytic value.
4. A table of ≥ 6 contact configurations (planar + 3D) with closure verdicts and one-line intuitions each.

**✅ Checkpoint:** the μ-flip matches your derivation to 1e-3; the $m$-convergence plot approaches the analytic threshold monotonically from the conservative side (note *which* side, and why linearization is conservative).

## Part 2 — Grasp force optimization (~1.5 h)

1. The SOCP above in `cvxpy` (ECOS/Clarabel solve SOCPs natively — check `prob.solver_stats`). Both objectives: min-max normal force, min total $\lVert f \rVert_2$.
2. Scenario: a 200 g box held by a 2-contact pinch (SO-101 gripper geometry: contact patches ~1 cm apart per jaw face, $\mu \approx 0.6$ rubber-on-cardboard). Solve for gravity + a disturbance wrench sweep: rotate a 1 N disturbance force through 360° in the vertical plane, plot required max normal force vs disturbance angle.
3. Visualize: contact forces as arrows inside their friction cones for 3 representative disturbance angles; mark the infeasible angular range (if any) in red.
4. Sensitivity: repeat at $\mu \in \{0.2, 0.4, 0.8\}$; the infeasible range should grow as μ shrinks — quantify.

**✅ Checkpoint:** feasible solves put every force strictly inside its cone (check the constraint slack); the polar plot of required force vs disturbance angle is smooth with peaks where physics says they should be (disturbance orthogonal to the grasp axis).

## Part 3 — Antipodal grasp scoring on a point cloud (~1.5 h)

Close the loop to perception-era grasping: score grasps the way GPD/Dex-Net-lineage systems seed theirs.

1. Sample a point cloud with normals from a graspable mesh (`trimesh.sample.sample_surface`; use a mug or box mesh scaled to the SO-101's ~6 cm max gripper opening).
2. Antipodal candidate generation: random point pairs with (a) distance ≤ gripper opening, (b) normals anti-aligned within a tolerance angle, (c) the connecting line inside both friction cones (your Part 1 test, reused).
3. Score each candidate: the Part 2 SOCP's min-max normal force under gravity + a fixed disturbance set (lower = better), plus a simple robustness score (fraction of ±5 mm contact perturbations that stay force-closed).
4. Render the top-10 and bottom-10 grasps on the cloud. The top ones should look like where a human would pinch it. If they don't, your normals are inward/outward flipped — the classic.

**✅ Checkpoint:** ≥ 100 valid candidates on the mug; top-10 concentrate on nearly-parallel opposing surfaces; the robustness score and the SOCP score broadly agree (Spearman ρ > 0.5 — compute it).

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `grasp.py` | `grasp_map`, closure tests, `optimize_forces`, antipodal sampler — importable, docstringed with conventions |
| `tests/` | hand-derived cases, the μ-flip test, translation-invariance property — green |
| `notebook.ipynb` | cone visualizations, the polar disturbance plot, top/bottom-10 grasp renders |
| `RESULTS.md` | the closure-verdict table with intuitions; the μ-sensitivity reading; 5 sentences on what this machinery *can't* see (deformables, uncertainty, dynamics) — i.e. why Phase 4 learns instead |

## Done when

- [ ] Closure tests agree with every hand-derivable case, including the analytic μ threshold.
- [ ] The SOCP finds valid in-cone force distributions and correctly reports infeasibility.
- [ ] Grasp ranking on the point cloud is visibly sensible and quantitatively consistent.
- [ ] `RESULTS.md` closes with the limits-of-analytic-grasping paragraph.

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
| Nothing is force-closed | wrench torque reference point inconsistent between contacts | one object-frame origin for all torques (Part 0 property test catches this) |
| μ-flip off by 2× | cone half-angle vs full-angle confusion ($\tan^{-1}\mu$ is the *half*-angle) | rederive with a picture; test at μ where you did the algebra |
| `cvxpy` says SOCP is unbounded | objective missing, or min-max not epigraph-formulated | `min t` s.t. $f_{n,i} \le t$ — check the epigraph transform |
| Solver flips LP/SOCP verdicts near the threshold | numerical tolerance at the boundary | treat $\delta$ within ±1e-6 as "marginal," report it as such |
| Top grasps on the mug rim edge only | curvature-blind point sampling | fine — note it; that's a real limitation of point-pair antipodal sampling |

## Stretch

Implement the Ferrari–Canny $\epsilon$-metric (radius of the largest origin-centered ball inside the unit grasp wrench space, via `scipy.spatial.ConvexHull` on the edge wrenches) and re-rank Part 3's grasps with it. Compare rankings; find one grasp where the metrics disagree and explain the disagreement geometrically.

## References

- Stanford CS237B *Principles of Robot Autonomy II*, W25 Problem Set 2 (public PDF + starter code). web.stanford.edu/class/cs237b.
- Tedrake, *Robotic Manipulation*, ch. 5 — friction cones, contact wrench cone. manipulation.csail.mit.edu.
- Lynch & Park, *Modern Robotics*, ch. 12 (Grasping and Manipulation) — closure formalism and LP tests.
- Ferrari & Canny, *Planning Optimal Grasps*, ICRA 1992 — the $\epsilon$-metric (stretch).
