"""Lesson 07 Parts 1-3 stub — spec in README.md "Part 3 — Trajectory optimization and the hybrid"
(the 50x5 benchmark table integrating Parts 1-3's planners).
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations


def run_benchmark(n_problems: int = 50, seed: int = 0) -> "pandas.DataFrame":
    """Run all five planners — RRT, RRT+shortcut, trajopt-straight, trajopt-restarts, and the
    RRT->trajopt hybrid — on n_problems random problems (planner.rrt.generate_problem); return the
    50x5 table of success %, median path length, and median wall-clock, with narrow-passage problems
    marked as a subset.

    Verified by: Part 3 checkpoint (one command regenerates the table; it makes the
    sample-for-topology/optimize-for-quality story legible at a glance).
    """
    raise NotImplementedError
