"""Lesson 18 Part 0/3 stub — spec in README.md "Part 0 — Pick the target and stare at the data" and "Part 3 — Evaluate all three arms".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from typing import Any


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial success rate.

    Verifies: Part 0 checkpoint (zero-shot number has a Wilson CI) and Part 3 checkpoint
    (success ± CI per arm). Spec: README.md "Part 0", step 3; "Part 3", step 3.
    """
    raise NotImplementedError


def run_eval(
    policy_path: str,
    dataset_repo_id: str,
    suite: str,
    num_episodes: int = 50,
    seeds: list[int] | None = None,
    output_json: str = "eval/results.json",
) -> dict[str, Any]:
    """Run lerobot-eval for one policy/suite pair, seeded, and write a JSON result with success + Wilson CI.

    One command per suite must rerun the full table (Deliverables acceptance criteria: "one command
    per suite reruns the full table; seeds fixed"). On Linux boxes, prefix MUJOCO_GL=egl.
    Verifies: Part 0 checkpoint (zero-shot baseline) and Part 3 checkpoint (zero-shot/LoRA/full arms).
    Spec: README.md "Part 0", step 3; "Part 3", steps 1-3.
    """
    raise NotImplementedError
