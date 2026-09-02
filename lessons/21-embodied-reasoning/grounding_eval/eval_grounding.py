"""Lesson 21 Part 1 stub — spec in README.md "Part 1 — Pointing accuracy on your own images".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from typing import Any


def load_ground_truth(labels_path: str) -> dict[str, Any]:
    """Load hand-labeled polygons/boxes per object per image (labelme JSON or a matplotlib-clicker's output). Verified by Part 1 checkpoint."""
    raise NotImplementedError


def point_in_mask(point: tuple[float, float], polygon: Any) -> bool:
    """Return whether a normalized ER point (y, x, 0-1000) falls inside the ground-truth polygon of the named object. Verified by Part 1 checkpoint (hit-rate metric)."""
    raise NotImplementedError


def miss_distance(point: tuple[float, float], polygon: Any, image_diagonal_px: float) -> float:
    """Compute the miss distance in px, normalized by image diagonal, for a point outside its GT polygon. Verified by Part 1 checkpoint (miss distances recorded for failed points)."""
    raise NotImplementedError


def hit_rate(points: Any, ground_truth: dict[str, Any]) -> dict[str, float]:
    """Compute point-in-mask hit rate, broken down per image source (sim/real) and per object category. Verified by Part 1 checkpoint (hit rate >= ~80% on sim images for unambiguous objects)."""
    raise NotImplementedError


def prompt_sensitivity_ablation(images: Any, objects: Any) -> dict[str, float]:
    """Run the bare-noun / descriptive / functional phrasing ablation (3 phrasings x object, same images/metric) and return the three hit-rate numbers. Verified by Part 1 checkpoint (ablation table shows a measurable phrasing effect)."""
    raise NotImplementedError
