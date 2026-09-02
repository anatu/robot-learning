"""
Lesson 01 Part 2 stub — spec in README.md "Part 2 — Byte-level readers".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""
from __future__ import annotations


def read_frame(root: str, ep_idx: int, frame_idx: int) -> dict:
    """Resolve (ep_idx, frame_idx) via episode offsets into the right file-XXXX.parquet (pyarrow, memory-mapped) and return state/action/timestamp for that row. Verified by: Part 2 checkpoint (torch.equal vs dataset[global_idx] for 20 random pairs)."""
    raise NotImplementedError
