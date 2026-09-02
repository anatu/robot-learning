"""Lesson 05 Part 3 stub — spec in README.md "Part 3 — TVLQR robustness".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from typing import Callable


def tvlqr(
    xs_bar: "np.ndarray",
    us_bar: "np.ndarray",
    f: Callable,
    Q: "np.ndarray",
    R: "np.ndarray",
    Qf: "np.ndarray",
) -> list:
    """Time-varying LQR: linearize f along the converged trajectory (xs_bar, us_bar) at each timestep to
    get A_k, B_k, then run the backward Riccati recursion (lqr.lqr's math, time-varying inputs) to
    produce gains K_k for u_k = ubar_k - K_k (x_k - xbar_k).

    Verified by: Part 3 checkpoint (open-loop replay fails for most +/-10% perturbed draws; TVLQR with
    handoff to infinite-horizon LQR near upright recovers >= 90% of 50 draws; the tube plot converges
    onto the nominal trajectory).
    """
    raise NotImplementedError
