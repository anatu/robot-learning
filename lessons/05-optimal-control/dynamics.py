"""Lesson 05 Part 0 stub — spec in README.md "Part 0 — Cartpole dynamics you can trust".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from typing import Callable


def cartpole_continuous(
    x: "np.ndarray",
    u: float,
    mc: float = 1.0,
    mp: float = 0.1,
    l: float = 0.5,
    g: float = 9.81,
) -> "np.ndarray":
    """Continuous cartpole dynamics xdot = f(x, u); state x = (p, theta, pdot, thetadot), theta = pi is upright.

    Verified by: Part 0 checkpoint (hanging-pole rollout swings symmetrically; energy drift < 0.1% over 5 s).
    """
    raise NotImplementedError


def rk4_step(f: Callable, x: "np.ndarray", u: float, dt: float = 0.02) -> "np.ndarray":
    """One RK4 integration step of continuous dynamics f at step size dt.

    Verified by: Part 0 checkpoint (energy conservation of the passive RK4 rollout).
    """
    raise NotImplementedError


def cartpole_discrete(x: "np.ndarray", u: float, dt: float = 0.02) -> "np.ndarray":
    """Discrete-time map x_{k+1} = f(x_k, u_k): RK4 of cartpole_continuous at dt.

    This is the f(x, u) consumed by lqr.py, ilqr.py, and tvlqr.py.
    Verified by: Part 0 checkpoint; indirectly by every downstream test that calls f(x, u).
    """
    raise NotImplementedError


def cartpole_jacobians(x: "np.ndarray", u: float, dt: float = 0.02) -> tuple:
    """Analytic Jacobians A = df/dx, B = df/du of the discrete map cartpole_discrete (i.e. differentiate
    through the RK4 integrator, not the continuous dynamics).

    Verified by: Part 0's finite-difference check — central differences (h=1e-6) vs analytic,
    max abs error < 1e-6 at 100 random states.
    """
    raise NotImplementedError
