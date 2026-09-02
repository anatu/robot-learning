"""Lesson 21 Part 3 stub — spec in README.md "Part 3 — The planner-executor loop".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TaskSpec:
    """TODO: fields are the student's design — see README Part 3 step 2 (goal text, object-placement randomization, seed list)."""


def get_tasks() -> list["TaskSpec"]:
    """Return the 3 multi-step tasks (sequential pair, order-constrained stack, negation-filtered move) with 10 randomized-placement seeded episodes each. Verified by Part 3 checkpoint (3 tasks x 10 seeded episodes rerunnable)."""
    raise NotImplementedError
