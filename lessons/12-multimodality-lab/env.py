"""Lesson 12 Part 1 stub — spec in README.md "Part 1 — The toy world".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class StepResult:
    """TODO: student design. Holds the next state, whether the episode terminated (goal reached
    or obstacle penetration), and whether it truncated (300-step timeout)."""


class PointMassEnv:
    """2D point-mass navigation task, no gym dependency needed. State s=(x,y) in [-1,1]^2, start
    (0,-0.8) + N(0, 0.02^2) jitter, goal within 0.1 of (0, 0.8), one circular obstacle centered at
    (0,0) with radius 0.25. Action a in R^2 is a displacement clipped to ||a|| <= 0.05; dynamics
    s' = s + a. Episode fails on obstacle penetration or after 300 steps.

    Verified by: Part 1 checkpoint (trajectory plot shows two clean symmetric arcs when driven by
    expert.py; expert success rate = 100%).
    """

    def reset(self, seed: int | None = None) -> Any:
        """Sample s0 = (0, -0.8) + N(0, 0.02^2) jitter, seeded. Verified by: Part 1 checkpoint."""
        raise NotImplementedError

    def step(self, a: Any) -> StepResult:
        """Clip `a` to ||a|| <= 0.05, apply s' = s + a, and check goal / obstacle-penetration /
        300-step timeout termination.

        Verified by: Part 1 checkpoint; Part 4 (rollout success rate, collision rate, minimum
        obstacle clearance).
        """
        raise NotImplementedError
