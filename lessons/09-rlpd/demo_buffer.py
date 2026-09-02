"""Lesson 09 Part 2 stub — spec in README.md "Part 2 — Wire demos into SAC".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class DemoTransition:
    """TODO: student design. One (s, a, r, s', done) transition converted from a recorded demo
    episode frame pair."""


class DemoBuffer:
    """Loads the Hub LeRobotDataset of recorded demos and stores them as (s, a, r, s', d)
    transitions, one buffer separate from the online replay buffer (never merged).

    Verified by: Part 1 checkpoint (dataset loads via LeRobotDataset); Part 2 checkpoint (tests
    green; 5k-step smoke run per arm executes without shape errors on cpu).
    """

    def __init__(self, repo_id: str) -> None:
        """Load `repo_id` from the Hub and populate transitions via `add`, per the README's
        per-episode conversion loop (terminal reward at t == len(frames) - 2, else 0.0)."""
        raise NotImplementedError

    def add(self, s: Any, a: Any, r: float, s2: Any, d: bool) -> None:
        """Append one (s, a, r, s', done) transition. Verified by: Part 2 checkpoint (smoke run
        executes without shape errors)."""
        raise NotImplementedError

    def sample(self, batch_size: int, seed: int | None = None) -> Any:
        """Sample `batch_size` transitions uniformly at random, seeded. Verified by: Part 2 step 3
        (seeds fix the sampled indices)."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Number of stored transitions — the |D_demo| term in the oversampling factor.

        Verified by: Part 4 checkpoint (oversampling-factor figure)."""
        raise NotImplementedError


def compose_batch(
    demo_buffer: DemoBuffer,
    online_buffer: Any,
    batch_size: int,
    composition: Literal["online", "preload", "rlpd"],
    seed: int | None = None,
) -> Any:
    """Build one training batch under `SACAgent.update()`'s `batch_composition` switch.

    `online`: 100% online_buffer. `preload`: demos inserted into online_buffer at t=0, FIFO
    thereafter (a single buffer — no split at sample time). `rlpd`: `batch_size // 2` sampled
    from `demo_buffer` concatenated *after* sampling with `batch_size // 2` from `online_buffer`
    (never merged into one buffer).

    Verified by: Part 2 checkpoint (batch composition is exactly 50/50 under `rlpd`; preloaded
    transitions are FIFO-evictable; seeds fix the sampled indices).
    """
    raise NotImplementedError
