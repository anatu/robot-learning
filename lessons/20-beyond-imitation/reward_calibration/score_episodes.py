"""Lesson 20 Part 3 stub — spec in README.md "Part 3 — Reward-model calibration".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from typing import Any


def score_episodes(
    reward_model_name: str,
    episodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run one v0.6 reward model (Robometer or TOPReward) over every episode; collect per-episode scores.

    Check which input format the chosen model expects (video vs frame sequence) per the
    reward-models API docs before calling this.
    Verifies: Part 3 checkpoint. Spec: README.md "Part 3", steps 1-2. Feeds the per-episode CSV.
    """
    raise NotImplementedError


def confusion_matrix_at_threshold(
    scores: list[float],
    labels: list[int],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Confusion matrix + precision/recall/F1 at the default threshold.

    Spec: README.md "Part 3", step 3.
    """
    raise NotImplementedError


def reliability_diagram(
    scores: list[float],
    labels: list[int],
    num_bins: int = 10,
) -> dict[str, Any]:
    """Bin scores into `num_bins` and compute empirical success rate per bin.

    With ~100 episodes across 10 bins expect ~10/bin — use 5 bins or bootstrap per-bin CIs if jagged
    (see Pitfalls table).
    Spec: README.md "Part 3", step 3.
    """
    raise NotImplementedError


def expected_calibration_error(
    scores: list[float],
    labels: list[int],
    num_bins: int = 10,
) -> float:
    """ECE computed over the reliability_diagram binning.

    Spec: README.md "Part 3", step 3.
    """
    raise NotImplementedError


def threshold_sweep(scores: list[float], labels: list[int]) -> list[dict[str, float]]:
    """Full precision-recall curve across thresholds.

    Spec: README.md "Part 3", step 3.
    """
    raise NotImplementedError


def failure_gallery(
    scores: list[float],
    labels: list[int],
    episodes: list[dict[str, Any]],
    k: int = 5,
) -> dict[str, Any]:
    """The k highest-confidence false positives and false negatives, one representative frame each.

    Spec: README.md "Part 3", step 4.
    """
    raise NotImplementedError
