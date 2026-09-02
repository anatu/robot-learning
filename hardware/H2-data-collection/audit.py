"""H2 Part 4 stub — spec in README.md "Part 4 — Audit, card, publish".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from typing import Any


def check_episode_count(dataset: Any, min_episodes: int = 50) -> bool:
    """Assert the dataset has at least `min_episodes` episodes. Verified by Part 4 checkpoint (audit script passes)."""
    raise NotImplementedError


def duration_histogram(dataset: Any) -> Any:
    """Compute the per-episode duration histogram; flag anything sitting at the 60 s cap. Verified by Part 4 checkpoint (median 15-35 s, none at the cap)."""
    raise NotImplementedError


def per_cell_counts(dataset: Any, session_log: Any) -> dict[str, int]:
    """Compute per-start-cell episode counts and compare against the planned grid from TASK.md. Verified by Part 4 checkpoint (counts match the plan)."""
    raise NotImplementedError


def frame_timestamp_gaps(dataset: Any, max_gap_multiple: float = 2.0) -> list[Any]:
    """Flag frame-timestamp gaps exceeding `max_gap_multiple` x the nominal frame period (dropped frames). Verified by Part 4 checkpoint (audit script passes)."""
    raise NotImplementedError


def joint_delta_outliers(dataset: Any) -> dict[str, float]:
    """Compute mean per-step joint delta per episode and flag outliers (jerky teleop). Verified by Part 4 checkpoint (audit script passes)."""
    raise NotImplementedError


def run_audit(repo_id: str) -> None:
    """Run all automated checks against the published Hub dataset `repo_id` and print a pass/fail summary. Verified by Part 4 checkpoint (audit script passes; run against the published repo)."""
    raise NotImplementedError
