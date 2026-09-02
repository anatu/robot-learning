"""
Lesson 01 Part 3 stub — spec in README.md "Part 3 — Windowing".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""
from __future__ import annotations


class ParsedDataset:
    """TODO: constructor args and internal fields are the student's design. Composes meta/tabular/video/windowing into the same per-sample dict LeRobotDataset.__getitem__ returns, including {key}_is_pad tensors."""

    def __init__(self, root: str, delta_timestamps: "dict[str, list[float]] | None" = None) -> None:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        """Return the sample dict for global index idx, windowed per delta_timestamps. Verified by: Part 3 checkpoint (parity vs LeRobotDataset on sample #100 + episode boundaries)."""
        raise NotImplementedError
