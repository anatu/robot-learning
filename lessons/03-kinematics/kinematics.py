"""
Lesson 03 Parts 2-5 stub — spec in README.md "Part 2 — Numerical FK for the full arm" / "Part 3 — The geometric Jacobian" / "Part 4 — Numerical IK" / "Part 5 — Singularity atlas".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""
from __future__ import annotations


def build_chain(model: "mujoco.MjModel") -> list:
    """Walk the MJCF tree once at load: for each body from base to gripper, record (parent, body-frame offset pos, orientation quat, joint axis) as a flat chain. Verified by: Part 2 checkpoint (fk() matches MuJoCo)."""
    raise NotImplementedError


def fk(q: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    """Compose T_i = T_{i-1} . T_offset_i . R_axis_i(q_i) down the chain to the ee_site, pure numpy, no MuJoCo calls. Returns (p_ee, R_ee). Verified by: Part 2 checkpoint (max position error < 1e-10 m, max rotation error < 1e-10 Frobenius vs mj_forward, over 1,000 random configs)."""
    raise NotImplementedError


def jacobian(q: "np.ndarray") -> "np.ndarray":
    """Geometric Jacobian J (6x5): column i = [axis_i x (p_ee - p_i); axis_i], using world-frame joint axes/positions from this module's own fk() intermediates. Verified by: Part 3 checkpoint (max abs diff < 1e-8 vs mj_jacSite, and < 1e-5 vs central finite differences of fk())."""
    raise NotImplementedError


def ik(p_target: "np.ndarray", q0: "np.ndarray", method: str = "dls") -> "tuple[np.ndarray, bool, int]":
    """Numerical IK for position targets: Gauss-Newton (method="gn", position-rows pseudo-inverse) or damped least squares (method="dls", fixed lambda). Iterate until ||delta_p|| < 1e-5 m or 200 iterations; clip each update into joint limits. Returns (q, converged, iters). Verified by: Part 4 checkpoint (DLS >= 99% success on reachable targets with <= 3 restarts; graceful boundary behavior on out-of-workspace targets)."""
    raise NotImplementedError


def ik_constrained(p_target: "np.ndarray", q0: "np.ndarray", keepout_center: "np.ndarray", keepout_radius: float) -> "tuple[np.ndarray, bool]":
    """Position IK with a spherical keep-out constraint, via scipy.optimize.minimize (SLSQP). Verified by: Part 5 checkpoint (constrained solve satisfies the keep-out to 1e-6; elbow routes around the sphere where the unconstrained solution would violate it)."""
    raise NotImplementedError
