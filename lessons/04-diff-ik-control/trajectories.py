"""
Lesson 04 Part 0 stub — spec in README.md "Part 0 — Reference trajectories + harness".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""
from __future__ import annotations


def line_trajectory(t: float, length: float = 0.10, duration: float = 5.0, ramp_frac: float = 0.2) -> "tuple[np.ndarray, np.ndarray]":
    """Analytic 10cm line reference (p*(t), pdot*(t)) with a trapezoidal speed profile (ramp = 20% of duration). Computed analytically, never finite-differenced. Verified by: Part 0 checkpoint (pdot* agrees with FD of p* to 1e-8)."""
    raise NotImplementedError


def circle_trajectory(t: float, radius: float = 0.06, duration: float = 10.0, ramp_frac: float = 0.2,
                       center: "np.ndarray | None" = None) -> "tuple[np.ndarray, np.ndarray]":
    """Analytic r=6cm circle in the y-z plane, centered mid-workspace, with a trapezoidal speed profile (ramp = 20% of duration). Verified by: Part 0 checkpoint (pdot* agrees with FD of p* to 1e-8)."""
    raise NotImplementedError
