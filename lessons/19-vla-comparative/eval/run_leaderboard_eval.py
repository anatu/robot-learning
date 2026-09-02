"""Lesson 19 Part 2 stub — spec in README.md "Part 2 — One protocol, three evaluations".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from typing import Any


def evaluate_model(
    checkpoint_path: str,
    suite: str,
    seeds: list[int],
    num_episodes: int = 50,
    output_json: str = "eval/results.json",
) -> dict[str, Any]:
    """Run the pre-registered eval for one model checkpoint: same seeds/episode count/suite as the
    other two models. Use the benchmark's Docker image on the cloud box; MUJOCO_GL=egl.

    Verifies: Part 2 checkpoint (leaderboard table fully populated, no cell says "TODO").
    Spec: README.md "Part 2", steps 1-2; metric list from PROTOCOL.md.
    """
    raise NotImplementedError


def measure_latency_and_vram(checkpoint_path: str, batch_size: int = 1) -> dict[str, float]:
    """ms/chunk (median, p95) and peak inference VRAM on one fixed machine, comparable across all
    three models. Note ms/chunk != control-rate ceiling once the async stack (Lesson 16) is in play —
    report both ms/chunk and implied max chunk rate.

    Spec: README.md "Part 2", step 2.
    """
    raise NotImplementedError
