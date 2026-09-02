"""
Lesson 01 Part 1 stub — spec in README.md "Part 1 — Metadata parsers".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Info:
    """TODO: fields are the student's design. Parsed meta/info.json: feature dict (name -> dtype/shape/names), fps, codebase_version, path templates for data/video shards."""


@dataclass
class EpisodeMeta:
    """TODO: fields are the student's design. Per-episode length, task id, and data/video offsets, parsed from meta/episodes/ chunked parquet."""


def load_info(root: str) -> Info:
    """Parse meta/info.json into an Info record. Verified by: Part 1 checkpoint (tree matches layout table; can state file+offset for episode 30)."""
    raise NotImplementedError


def load_episodes(root: str) -> list[EpisodeMeta]:
    """Parse meta/episodes/ chunked parquet into per-episode length/task/offsets. Verified by: Part 1 checkpoint (sum(ep.length for ep in episodes) == 11_939)."""
    raise NotImplementedError


def load_stats(root: str):
    """Parse meta/stats.json (per-feature mean/std/min/max). Verified by: Part 1 checkpoint (every feature in info.json has a stats entry)."""
    raise NotImplementedError


def load_tasks(root: str):
    """Parse meta/tasks.jsonl (natural-language task -> integer id). Verified by: Part 1 checkpoint (task table has exactly 1 task)."""
    raise NotImplementedError
