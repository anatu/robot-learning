"""Lesson 14 Part 4 stub — spec in README.md "Part 4 — Temporal ensembling by hand".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from typing import Any


class TemporalEnsembler:
    """Overlapping-chunk temporal ensembling with exponential weights w_i = e^{-m i}
    (i=0 is the oldest live prediction), over a ring buffer of not-yet-consumed chunks.
    Verified by the Part 4 checkpoint (four pytest properties + cross-check against
    LeRobot's ACTTemporalEnsembler, max abs deviation < 1e-6)."""

    def __init__(self, chunk_size: int, m: float = 0.01) -> None:
        raise NotImplementedError

    def add_chunk(self, t: int, actions: Any) -> None:
        """Register a newly predicted H_a-length action chunk starting at timestep t.
        Verified by the Part 4 checkpoint."""
        raise NotImplementedError

    def get(self, t: int) -> Any:
        """Return the weighted-average ensembled action for timestep t over every live
        chunk that covers it. Verified by the Part 4 checkpoint."""
        raise NotImplementedError
