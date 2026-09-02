"""Lesson 10 Part 1 stub — spec in README.md "Part 1 — Reward classifier, calibrated before trusted".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ClassifierEvalResult:
    """TODO: student design. Holds the PR curve, the threshold sweep, and the false-positive
    gallery (frames the classifier calls success on held-out failure episodes)."""


class RewardClassifier:
    """Fallback binary success detector (frozen ResNet-18 torso + linear head), used only if the
    installed LeRobot's `lerobot.rl` stack lacks a reward-classifier trainer. Real implementation
    subclasses `torch.nn.Module`; this stub omits that base class to stay import-light.

    Verified by: Part 1 checkpoint (held-out precision >= 0.95 at the chosen threshold).
    """

    def __init__(self) -> None:
        """Build the frozen ResNet-18 torso with a linear head, per the README's fallback spec."""
        raise NotImplementedError

    def forward(self, frame: Any) -> float:
        """Return the success score in [0, 1] for one frame. Verified by: Part 1 checkpoint
        (PR curve)."""
        raise NotImplementedError


def train_classifier(repo_id: str, seed: int = 0) -> RewardClassifier:
    """Train on terminal-window frames from the labeled dataset `repo_id`: positives are the last
    ~10 frames of successes, negatives are everything else plus failure episodes.

    Verified by: Part 1 checkpoint (held-out precision >= 0.95 at the chosen threshold).
    """
    raise NotImplementedError


def evaluate_classifier(classifier: RewardClassifier, held_out_repo_id: str) -> ClassifierEvalResult:
    """Compute the PR curve, threshold sweep, and false-positive gallery on 5 held-out episodes;
    pick the operating threshold by a stated rule (e.g. precision >= 0.95 at max recall).

    Verified by: Part 1 checkpoint (precision >= 0.95; FP gallery reviewed in RESULTS.md, one
    sentence each on why the classifier was fooled).
    """
    raise NotImplementedError
