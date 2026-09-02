"""Lesson 14 Part 2 stub — spec in README.md "Part 2 — The evaluation harness you'll reuse all course".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class EpisodeResult:
    """TODO: student-designed fields — e.g. seed, success, steps_to_success, executed actions (Part 2)."""


@dataclass
class EvalReport:
    """TODO: student-designed fields — e.g. success rate, Wilson CI, per-episode EpisodeResults, video paths (Part 2)."""


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> "tuple[float, float]":
    """95% Wilson score interval for a binomial success rate. Verified by the Part 2 checkpoint
    (harness success rate matches lerobot-eval within CI)."""
    raise NotImplementedError


def evaluate(policy: Any, env_id: str, seeds: Sequence[int]) -> EvalReport:
    """Run policy over the given seeds on env_id, fixed-seed reset, Wilson CI on success rate.
    Interface contract reused unchanged in Lesson 15 and cited by Lessons 16/19/H3.
    Verified by the Part 2 checkpoint."""
    raise NotImplementedError


def mean_squared_jerk(actions: Any) -> float:
    """Mean squared third difference of joint positions, averaged over joints and time,
    computed from an episode's executed action sequence. Verified by the Part 3 checkpoint
    (jerk vs H_a plot; ensembling visibly reduces jerk)."""
    raise NotImplementedError
