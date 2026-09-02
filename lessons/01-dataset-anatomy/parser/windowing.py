"""
Lesson 01 Part 3 stub — spec in README.md "Part 3 — Windowing".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""
from __future__ import annotations


def window(query_idx: int, ep_start: int, ep_end: int,
           deltas: list[float], fps: float, tolerance_s: float = 1e-4
           ) -> tuple[list[int], list[bool]]:
    """Absolute frame indices to gather, and is_pad per position. Validate deltas as multiples of 1/fps within tolerance_s, clamp into [ep_start, ep_end), emit pad mask. Verified by: Part 3 checkpoint (parity on sample #100 + both boundaries of >= 3 episodes) and Part 4's property suite."""
    raise NotImplementedError
