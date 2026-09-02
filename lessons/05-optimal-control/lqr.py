"""Lesson 05 Part 1 stub — spec in README.md "Part 1 — LQR".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations


def lqr(
    A: "np.ndarray",
    B: "np.ndarray",
    Q: "np.ndarray",
    R: "np.ndarray",
    Qf: "np.ndarray",
    N: int,
) -> tuple:
    """Finite-horizon discrete backward Riccati recursion; returns (Ks, Ps), the per-step gain and
    cost-to-go matrices for k = 0..N-1 with optimal policy u_k = -K_k x_k.

    Verified by: Part 1 checkpoint (DARE residual < 1e-8 at convergence, cross-checked against
    scipy.linalg.solve_discrete_are; closed-loop spectral radius rho(A - B K_inf) < 1).
    """
    raise NotImplementedError
