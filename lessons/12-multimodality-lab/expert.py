"""Lesson 12 Part 1 stub — spec in README.md "Part 1 — The toy world".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from typing import Any, Literal


def choose_mode(seed: int | None = None) -> Literal["left", "right"]:
    """Flip a fair coin at episode start for the arc's sign: left waypoints at x=-0.45, right at
    x=+0.45.

    Verified by: Part 1 checkpoint (two clean symmetric arcs, colored by mode).
    """
    raise NotImplementedError


def expert_action(state: Any, mode: Literal["left", "right"], waypoint_index: int) -> Any:
    """Steer toward the current waypoint at full step size with N(0, 0.005^2) action noise.
    Waypoints, sign per `mode`: (+-0.45, -0.5) -> (+-0.45, 0.5) -> (0, 0.8), advancing to the next
    waypoint once within 0.1 of the current one.

    Verified by: Part 1 checkpoint (expert success rate = 100%).
    """
    raise NotImplementedError


def record_expert_episodes(env: Any, num_episodes: int, seed: int | None = None) -> Any:
    """Roll `num_episodes` expert episodes on `env` (500 -> ~25-30k (s, a) pairs per the README),
    returning the recorded trajectories for Part 1's plot and Part 2's training dataset.

    Verified by: Part 1 checkpoint (trajectory plot; probe-state s*=(0,-0.4) cluster check) and
    test_expert.py.
    """
    raise NotImplementedError
