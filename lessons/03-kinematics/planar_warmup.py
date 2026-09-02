"""
Lesson 03 Part 1 stub — spec in README.md "Part 1 — The planar warm-up, analytically".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""
from __future__ import annotations


def fk_planar(q1: float, q2: float, l1: float, l2: float) -> "tuple[float, float]":
    """Analytic 2-link planar FK: x = l1*cos(q1) + l2*cos(q1+q2), y = l1*sin(q1) + l2*sin(q1+q2). Verified by: Part 1 checkpoint (FK(IK_branch(p)) == p to 1e-9)."""
    raise NotImplementedError


def ik_planar(x: float, y: float, l1: float, l2: float, branch: str = "up") -> "tuple[float, float] | None":
    """Analytic 2-link planar IK: cos(q2) from the law of cosines, elbow-up/elbow-down via +/- on q2, then q1 by atan2 correction. Returns None when |l1-l2| <= ||p|| <= l1+l2 fails (unreachable). Verified by: Part 1 checkpoint (round-trip at 1e-9 for both branches; unreachable targets correctly reported)."""
    raise NotImplementedError
