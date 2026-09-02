"""Lesson 05 Part 2 stub — spec in README.md "Part 2 — iLQR swing-up".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from typing import Callable


def ilqr(
    x0: "np.ndarray",
    u_init: "np.ndarray",
    f: Callable,
    cost: Callable,
    N: int,
) -> tuple:
    """Iterative LQR: backward pass with Levenberg-Marquardt regularization (mu, start 1e-6, x10 on a
    non-PD Q_uu or failed forward pass, /2 on success) and forward pass with backtracking line search
    (alpha in {1, 1/2, 1/4, ...}) until predicted-vs-actual cost decrease is acceptable; converges when
    |delta J| < 1e-6. Returns (xs, us, Ks, info) where info carries cost-per-iteration, accepted alpha,
    the mu trace, and the predicted-vs-actual decrease ratio.

    Verified by: Part 2 checkpoint (swing-up converges in < 100 iterations from noise init, final pole
    angle within 1e-3 of upright, cost curve monotone) and the LQR-equivalence test (on a
    linear-quadratic problem, converges in one iteration to lqr.lqr's solution, < 1e-10).
    """
    raise NotImplementedError
