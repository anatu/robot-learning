"""
Lesson 01 Part 2 stub — spec in README.md "Part 2 — Byte-level readers".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""
from __future__ import annotations


def read_image(root: str, camera_key: str, ep_idx: int, frame_idx: int) -> "np.ndarray":
    """Locate the MP4 shard + in-file timestamp from metadata; seek to the nearest keyframe at or before the target with av/pyav, decode forward to the exact frame; return CHW float32 in [0,1]. Verified by: Part 2 checkpoint (max abs diff <= 1/255 vs dataset[global_idx] for 20 random pairs)."""
    raise NotImplementedError
