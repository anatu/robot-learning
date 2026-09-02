"""Lesson 07 Part 3 stub — spec in README.md "Part 3 — Trajectory optimization and the hybrid".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from typing import Callable


def trajopt(
    q_start: "np.ndarray",
    q_goal: "np.ndarray",
    clearance_fn: Callable,
    n_waypoints: int = 30,
    d_safe: float = 0.02,
    q_init: "list | None" = None,
) -> "list | None":
    """Direct transcription: minimize sum_k ||q_{k+1} - q_k||^2 over waypoints q_1..q_N subject to
    q_1=q_start, q_N=q_goal, joint limits, and penalty-form clearance max(0, d_safe - clearance_fn(q_k))^2.
    q_init seeds the optimizer (straight line, an RRT path resampled to n_waypoints, or None for random
    restarts) via scipy.optimize.minimize (SLSQP/L-BFGS).

    Verified by: Part 3 checkpoint — every "converged" result must be re-verified with the collision
    oracle at execution resolution (convergence is not feasibility); trajopt-straight fails on a
    nontrivial, narrow-passage-concentrated fraction of the 50 problems; seeding with an RRT path (the
    hybrid) matches RRT's success rate at shorter, smoother paths.
    """
    raise NotImplementedError
