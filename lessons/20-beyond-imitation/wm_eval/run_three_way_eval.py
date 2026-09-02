"""Lesson 20 Part 2 stub — spec in README.md "Part 2 — Run a world-model policy".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from typing import Any


def run_eval_arm(
    policy_path: str,
    dataset_repo_id: str,
    suite: str,
    seeds: list[int],
    num_episodes: int,
) -> dict[str, Any]:
    """Evaluate one arm (SmolVLA-ft / VLA-JEPA / zero-shot) on the same benchmark subset + seeds
    used in Lesson 18, producing one row of the three-way comparison table.

    Start from the published VLA-JEPA baseline checkpoint for the target benchmark family; do not
    train from scratch unless budget allows the Part 2 step-4 alternative.
    Verifies: Part 2 checkpoint (three-way table exists). Spec: README.md "Part 2", steps 1-2, 4.
    """
    raise NotImplementedError


def inspect_inference_graph(policy_path: str) -> dict[str, Any]:
    """Inspect the checkpoint or policy class and record what actually runs in the inference graph
    (confirm the V-JEPA2 world-model branch is absent at inference).

    Verifies: Part 2 checkpoint (inference-graph note stating what actually runs at test time).
    Spec: README.md "Part 2", step 3.
    """
    raise NotImplementedError


def measure_latency_ms_per_chunk(
    policy_path: str,
    batch_size: int = 1,
    num_warm_calls: int = 100,
) -> dict[str, float]:
    """Median ms/chunk on the same hardware as the SmolVLA-ft comparison point.

    Spec: README.md "Part 2", step 3.
    """
    raise NotImplementedError
