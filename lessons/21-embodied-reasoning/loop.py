"""Lesson 21 Part 3 stub — spec in README.md "Part 3 — The planner-executor loop".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from typing import Any


def observe() -> Any:
    """Capture the current frame from the sim scene (or camera) for the ER client to consume. Verified by Part 3 checkpoint (per-layer wall-clock logged)."""
    raise NotImplementedError


def run_loop(goal: str, max_replans: int = 2) -> Any:
    """Run the planner-executor loop (README Part 3 pseudocode): plan, then per-subtask ground/execute/verify with bounded replanning, returning SUCCESS or FAIL(sub). `executor` follows the Lesson 14 harness interface or a scripted pick/place controller (Lesson 02's machinery). Verified by Part 3 checkpoint (hierarchical completes >= 1 task the flat baseline can't; per-layer time budget logged)."""
    raise NotImplementedError


def run_flat_baseline(goal: str, max_steps: int) -> Any:
    """Run the same language-conditioned policy on the raw goal with no ER in the loop, matched episode/timeout budget to `run_loop`. Verified by Part 3 checkpoint (flat baseline comparison under a matched budget)."""
    raise NotImplementedError
