"""Lesson 07 Part 2 stub — spec in README.md "Part 2 — Shortcut, time-parameterize, execute".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations


def time_parameterize(
    path: list,
    qdot_max: float = 1.5,
    qddot_max: float = 4.0,
    hz: float = 50.0,
) -> tuple:
    """Per-segment trapezoidal time parameterization of a shortcut joint-space path honoring per-joint
    qdot_max (rad/s) and qddot_max (rad/s^2). Returns (ts, qs): q*(t) sampled at hz.

    Verified by: Part 2 checkpoint (executed trajectories respect max measured |qdot| <= 1.5 rad/s and
    finish collision-free on >= 45/50 problems through the Lesson 04 tracker).
    """
    raise NotImplementedError
