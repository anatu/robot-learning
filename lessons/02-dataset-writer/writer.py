"""
Lesson 02 Part 3 stub — spec in README.md "Part 3 — Serialize".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""
from __future__ import annotations


def write_dataset(episodes: "list[Episode]", repo_id: str, fps: int = 30, root: "str | None" = None) -> None:
    """Drive LeRobotDataset.create() -> add_frame() per timestep -> save_episode() per episode -> finalize() once at the end, before any push_to_hub(). Verified by: Part 3 checkpoint (local load-back works; episode count and frame totals match generation logs exactly)."""
    raise NotImplementedError


def load_back_check(repo_id: str, root: "str | None" = None) -> dict:
    """Load the written dataset locally, window it with delta_timestamps={"action": [i/30 for i in range(10)]}, and check meta/stats.json exists with sane per-feature mean/std. Verified by: Part 3 checkpoint (windowed action comes back shape (10, 6))."""
    raise NotImplementedError
