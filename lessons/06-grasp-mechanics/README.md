# Lesson 06 — Grasp Mechanics *(optional)*

This optional lesson covers the analytic theory of grasping that preceded, and still underlies, learned grasping: contact models, friction cones, the grasp map, closure conditions posed as feasibility problems, and grasp force optimization posed as a convex program. You will derive one threshold by hand, the minimum friction coefficient at which two antipodal contacts achieve force closure, and then watch a numerical closure test flip at exactly that value. The practical payoff comes later, in H3: when a learned policy drops an object, this lesson is what lets you say whether the grasp was geometrically unsound, whether the load exceeded what friction could hold, or whether the failure was one of perception.

| | |
|---|---|
| **Phase** | 2 — Classical core (optional) |
| **Time** | 1 session (3–4 h desk time), all Mac-local |
| **Cost** | $0 |
| **Prerequisites** | 03 (frames and wrenches); basic convex optimization with `cvxpy` |
| **Feeds into** | H3's failure taxonomy (grasp miss versus slip versus perception); Lesson 07's scene design |
| **Skip criteria** | Skip if the goal is speed to modern methods; do it if grasping intuition feels rusty |

## Learning objectives

After this lesson you can:

1. **Explain** what the grasp map $G$ is and what lives in its column space, and state your torque sign convention from memory.
2. **Derive** the friction threshold for the antipodal case by hand, and predict from which side a linearized cone test converges to it.
3. **Predict** where a pinch grasp's required normal force peaks as an external disturbance rotates, and confirm it with a second-order cone program.
4. **Decide**, for an infeasible load, which of the three physically distinct remedies applies, and say which one a learned policy chooses implicitly.

## Principles

### Contact models and the friction cone

A contact transmits force from a finger to an object, and a contact model specifies which forces it can transmit. The three standard models are the frictionless point contact, which transmits only a normal force (one dimension); the point contact with friction (PCWF), which transmits any force inside a friction cone about the normal (three dimensions in 3-D space); and the soft-finger contact, which adds a torsional moment about the normal. This lesson uses the point contact with friction throughout.

The friction cone comes from Coulomb's law. At contact $i$ with unit normal $\hat n_i$ and friction coefficient $\mu$, a force with normal component $f_{n,i}$ and tangential component $f_{t,i}$ can be transmitted without slipping if and only if

$$\lVert f_{t,i} \rVert_2 \le \mu\, f_{n,i}, \qquad f_{n,i} \ge 0.$$

The set of admissible forces is therefore a second-order cone, written $\mathcal{FC}_i$. The cone can be handled exactly, which places any optimization over contact forces in the class of second-order cone programs (SOCPs), or it can be approximated by an $m$-sided pyramid, which turns every cone constraint into $m$ linear ones and makes closure tests into linear programs. Because the pyramid is inscribed in the true cone, any test based on the linearization is conservative: it may declare a grasp infeasible that the exact cone would admit, but never the reverse. Exercise 3 asks you to predict this direction before observing it.

### The grasp map

To reason about several contacts at once, their forces are collected into a single object wrench. Contact $i$ sits at point $p_i$ in the object frame, and a rotation $R_i$ maps its contact frame into the object frame. A contact force $f_i$ expressed in the contact frame therefore contributes the wrench

$$w_i = \begin{bmatrix} R_i f_i \\ p_i \times R_i f_i \end{bmatrix},$$

a force and the moment that force produces about the object origin. Stacking these contributions as columns gives the grasp map $G \in \mathbb{R}^{6 \times 3k}$ for $k$ contacts, so that the total wrench on the object is $w = G f$. Every statement in the rest of the lesson is a statement about the set $\{Gf : f_i \in \mathcal{FC}_i\}$, the wrenches the grasp can apply while respecting friction. Almost every implementation error in this material is a sign error in the $p_i \times$ term, so the sign convention for moments should be decided once, written into the module docstring, and tested against a case you can check by hand.

### Form closure and force closure

Form closure is a purely geometric property. It asks whether the frictionless normal forces alone, with $\mu = 0$, can resist any external wrench. This holds if and only if the contact normal wrenches positively span $\mathbb{R}^6$, which is equivalent to the origin lying in the interior of their convex hull. Lynch & Park (ch. 12) give an exact linear-programming test: maximize $\delta$ subject to $G_n \lambda = 0$, $\lambda \ge \delta \mathbf{1}$, and $\mathbf{1}^\top \lambda \le k$; the grasp is form-closed if and only if the optimal $\delta$ is positive. Because six wrench dimensions must be spanned positively, form closure requires at least seven frictionless contacts in 3-D, or four in the plane.

Force closure relaxes the geometry by admitting friction. The condition is that $\{Gf : f_i \in \mathcal{FC}_i\} = \mathbb{R}^6$, meaning that for any external wrench there exist admissible contact forces that cancel it. With linearized cones this reduces to a form-closure-style test applied to the pyramid edge wrenches. The case that can be solved by hand, and that serves as the ground truth for this lesson, is two antipodal point contacts with friction: they achieve force closure if and only if the line connecting them lies inside both friction cones. That condition yields a closed-form minimum $\mu$ for a given geometry, and it is also the warm-up problem of CS237B's second homework.

### Grasp force optimization as a convex program

Closure says whether some set of contact forces can resist any load; it does not say how large those forces must be. Given a specific external wrench $w_{\text{ext}}$, such as gravity plus an inertial load, grasp force optimization (CS237B HW2, Problem 2) asks for the admissible contact forces that resist it with the least effort:

$$\min_{f}\; \max_i f_{n,i} \quad \text{(or } \lVert f \rVert_2\text{)} \qquad \text{s.t.} \qquad G f = -w_{\text{ext}}, \qquad \lVert f_{t,i} \rVert_2 \le \mu f_{n,i} \;\; \forall i.$$

This is an SOCP and takes about fifteen lines of `cvxpy`. When the program is infeasible, that infeasibility is itself the answer: this grasp cannot hold this load with this friction, and something physical must change.

**Carry forward**

- The grasp map turns contact forces into an object wrench, and closure is a statement about whether the image of the friction cones under $G$ fills wrench space.
- Form closure is a geometric property that needs at least seven contacts in 3-D, whereas force closure relies on friction and is achieved by two antipodal contacts whenever the line between them lies inside both cones.
- A linearized friction cone is an inscribed pyramid, so every test built on it is conservative.
- Holding a load is a feasibility question; the SOCP's infeasibility is a physical verdict, and the three ways to restore feasibility are more friction, more normal force, or different contact locations.
- None of this analysis sees deformable objects, uncertainty in contact location, or dynamics, which is the reason Phase 4 turns to learned policies.

| Source | Read for |
|---|---|
| CS237B W25 HW2 PDF (web.stanford.edu/class/cs237b, public) | the exact antipodal-$\mu$ derivation and the SOCP formulation, with starter and test code |
| MIT *Robotic Manipulation* ch. 5 (Bin Picking) | friction cones, the contact wrench cone, and antipodal grasp heuristics as they are used in practice |
| Lynch & Park ch. 12 | the form-closure and force-closure formalism and the LP test |

## Exercise 1 — Derive the antipodal friction threshold and fix the sign convention [Derive]

For two contacts, force closure is a geometric condition with a closed-form answer, and every numerical result later in the lesson is checked against it. In this exercise you derive that answer on paper and settle the moment sign convention that the code will use, before any code exists.

1. On paper, consider two antipodal point contacts with friction on a disk of radius $r$, with the line between the contacts offset by $c$ from the disk's center. Derive $\mu_{\min}$, the smallest friction coefficient for which the connecting line lies inside both friction cones. Note that $\tan^{-1}\mu$ is the cone's half-angle, not its full angle.
2. On paper, work out three wrenches that you can verify without code: a single contact at the origin, which produces a pure force and zero moment; two opposed contacts on a unit-width square, whose normal wrenches cancel; and a contact offset by $d$ along $x$ carrying a force $f_y$, which produces a moment $\tau_z = \pm f_y d$. Pick the sign, write it down, and adopt it as the module docstring's convention.
3. Write all three results and the formula $\mu_{\min}(r, c)$ into `RESULTS.md` before writing any code.

**✅ Checkpoint:** $\mu_{\min}$ is a formula in $r$ and $c$, and the moment sign convention is stated in one sentence.

## Exercise 2 — Implement the grasp map, closure tests, and force SOCP [Build]

Closure and load-holding are both convex feasibility problems over the same set of admissible wrenches, and this exercise builds the code that poses them. Write the specification for `grasp.py` and have an AI tool draft it; the checks below are what make the draft trustworthy.

- `contact_wrench(p, R, f) -> w` and `grasp_map(contacts) -> G`, for both the planar case (3-D wrenches) and the spatial case (6-D wrenches), with the moment convention from Exercise 1 in the docstring.
- `is_form_closure(contacts) -> (bool, delta)`, implementing the LP from the Principles section with `cvxpy`.
- `is_force_closure(contacts, mu, m_edges) -> bool`, which linearizes each cone into $m$ edges and runs the positive-span test on the edge wrenches.
- `optimize_forces(contacts, mu, w_ext, objective) -> f | None`, the SOCP with both objectives: the min-max normal force via the epigraph form (`min t` subject to $f_{n,i} \le t$), and the minimum total $\lVert f \rVert_2$. Solve with ECOS or Clarabel and return `None` on infeasibility.
- The check, in `checks.py`: the three hand-derived wrenches from Exercise 1 are reproduced; four orthogonal frictionless contacts on a planar square give form closure and three do not; translating every contact point and the external wrench's reference point by the same offset leaves solvability unchanged; and every feasible SOCP solution has strictly positive slack in each cone constraint.

**✅ Checkpoint:** `checks.py` passes, and `prob.solver_stats` confirms that an SOCP-capable solver was used.

## Exercise 3 — Locate the friction threshold numerically [Predict → Run]

The linearized closure test should flip from infeasible to feasible near the $\mu_{\min}$ you derived, and because the pyramid is inscribed in the cone it should approach that value from a particular side as the number of edges grows. In this exercise you predict both the flip point and the side, then sweep $\mu$ to check.

1. Before running, write in `RESULTS.md` the $\mu$ at which `is_force_closure` should flip for the antipodal disk case, using your Exercise 1 formula. Then, for $m \in \{4, 8, 16, 64\}$ edges, write whether you expect the numerical flip point to sit above or below the analytic value, whether it approaches monotonically, and why the inscribed pyramid implies that direction.
2. Sweep $\mu$ in steps of $10^{-3}$ for each $m$ and record the flip point.
3. Plot the flip point against $m$ with the analytic value as a horizontal line, and reconcile the plot with your prediction in `RESULTS.md`.

**✅ Checkpoint:** at $m = 64$ the flip is within one step ($10^{-3}$) of your derivation, and the convergence is monotone from the side you predicted.

## Exercise 4 — Hold a box against a rotating disturbance [Predict → Run]

The normal force a grasp needs depends on the direction of the load, and the directions where it peaks can be located by physical reasoning before any solver runs. This exercise sweeps a disturbance force through a full rotation and compares the SOCP's required force profile to your prediction.

1. The scenario is a 200 g box held by a two-contact pinch with SO-101 gripper geometry: contact patches roughly 1 cm apart on each jaw face, and $\mu \approx 0.6$ for rubber on cardboard. The external wrench is gravity plus a 1 N disturbance force whose direction rotates through 360° in the vertical plane.
2. Before running, write in `RESULTS.md` the disturbance angles at which you expect the min-max normal force to peak, and whether you expect any angular range to be infeasible at $\mu = 0.6$. Then, for $\mu \in \{0.2, 0.4, 0.8\}$, write whether the infeasible range should grow or shrink and by roughly how much.
3. Solve the SOCP over the angle sweep at each $\mu$. Produce a polar plot of the required maximum normal force against disturbance angle, with infeasible ranges marked in red, and render the contact forces as arrows inside their friction cones for three representative angles.
4. Reconcile the plots with your prediction in `RESULTS.md`.

**✅ Checkpoint:** the polar plot is smooth with peaks where you predicted, which is where the disturbance is orthogonal to the grasp axis, and the infeasible range grows as $\mu$ shrinks, with the growth quantified.

## Exercise 5 — Choose a remedy for an infeasible load [Decide]

An infeasible SOCP admits exactly three physical remedies, and they cost different things. This exercise quantifies each remedy for the worst case found in Exercise 4 and asks you to choose one.

1. Take the widest infeasible range from Exercise 4. For each remedy, compute the minimum change that makes the whole sweep feasible: raising $\mu$ (a change of surface material), raising the allowed normal force (squeezing harder), or moving the contacts (regrasping).
2. Decide which remedy you would specify for the SO-101 gripper handling cardboard boxes, and defend the choice in `RESULTS.md` using the three numbers.
3. State which remedy a learned pick policy chooses implicitly, and explain why it cannot choose the other two.

**✅ Checkpoint:** three quantified remedies, one defended decision, and one sentence on the learned-policy analogue.

## Exercise 6 — Write the verdict table and the limits of the analysis [Write]

In `RESULTS.md`, produce a table of at least six contact configurations, planar and spatial, each with its closure verdict and a one-line explanation of why. Add the $\mu$-sensitivity reading from Exercise 4. Close with five sentences on what this machinery cannot see, namely deformable objects, uncertainty in contact placement, and dynamics, and why those blind spots are the reason Phase 4 learns grasps rather than computing them.

**✅ Checkpoint:** the table has at least six rows, and the closing paragraph names all three blind spots.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `grasp.py` | `grasp_map`, the closure tests, and `optimize_forces`, importable, with the moment convention in the docstring |
| `checks.py` | hand-derived cases, the translation-invariance property, and the cone-slack check, passing from one command |
| `plots/` | flip point versus $m$, the polar disturbance plot at every $\mu$, and the cone-arrow renders |
| `RESULTS.md` | the Exercise 1 derivation; Exercise 3 and 4 predictions with reconciliations; the Exercise 5 decision; the verdict table; the limits paragraph |

## Done when

- [ ] `checks.py` passes, including the analytic $\mu$ threshold at $m = 64$.
- [ ] The Exercise 3 and 4 predictions were written before the runs and reconciled after them.
- [ ] The SOCP finds valid in-cone force distributions and correctly reports infeasibility.
- [ ] `RESULTS.md` closes with the decision and the paragraph on the limits of analytic grasping.

## Self-check

1. Why does form closure need at least seven contacts in 3-D while force closure needs only two?
2. What convex object does the set of admissible contact forces form, and why does linearizing it make closure tests conservative?
3. Your SOCP is infeasible for some disturbance. What are the three physically distinct fixes, and which does a learned policy choose implicitly?
4. Where exactly does the $p_i \times$ term in the grasp map come from?
5. H3's policy drops a box during transport but never during lift. Which analysis from this lesson explains the difference?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Everything is force-closed | normals flipped inward, or the $f_n \ge 0$ constraint omitted | render the normals; assert that the nonnegativity constraints are present |
| Nothing is force-closed | moment reference point inconsistent between contacts | use one object-frame origin for all moments; the translation-invariance check catches this |
| $\mu$-flip off by a factor of 2 | cone half-angle confused with the full angle ($\tan^{-1}\mu$ is the half-angle) | rederive with a picture; test at the $\mu$ where you did the algebra |
| `cvxpy` reports the SOCP as unbounded | objective missing, or the min-max not written in epigraph form | `min t` subject to $f_{n,i} \le t$; check the epigraph transform |
| Solver flips verdicts near the threshold | numerical tolerance at the boundary | treat $\delta$ within $\pm 10^{-6}$ as marginal and report it as such |

## Going deeper

- **Antipodal grasp scoring on a point cloud.** Sample a mug or box mesh with `trimesh.sample.sample_surface`, scaled to the SO-101's roughly 6 cm maximum opening. Generate random point pairs whose distance is at most the opening, whose normals are anti-aligned, and whose connecting line lies inside both cones (Exercise 2's test, reused). Score each candidate by the SOCP's min-max normal force under gravity plus a disturbance set, and by a robustness score defined as the fraction of $\pm 5$ mm contact perturbations that remain force-closed; render the top and bottom ten. Expect at least 100 valid candidates and a Spearman correlation above 0.5 between the two scores. If the top grasps cluster on the rim edge only, that reflects curvature-blind point sampling, which is a genuine limitation worth noting.
- **The Ferrari–Canny $\epsilon$-metric.** Compute the radius of the largest origin-centred ball inside the unit grasp wrench space via `scipy.spatial.ConvexHull` on the edge wrenches, re-rank the point-cloud grasps with it, and find one grasp where the two metrics disagree and explain the disagreement geometrically.

## References

- Stanford CS237B *Principles of Robot Autonomy II*, W25 Problem Set 2 (public PDF and starter code). web.stanford.edu/class/cs237b.
- Tedrake, *Robotic Manipulation*, ch. 5 — friction cones and the contact wrench cone. manipulation.csail.mit.edu.
- Lynch & Park, *Modern Robotics*, ch. 12 (Grasping and Manipulation) — closure formalism and LP tests.
- Ferrari & Canny, *Planning Optimal Grasps*, ICRA 1992 (the $\epsilon$-metric, Going deeper).
