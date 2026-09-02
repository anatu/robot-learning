"""Lesson 15 Part 3 stub — spec in README.md "Part 3 — The sampler study".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations


def train(dataset_repo_id: str = "lerobot/pusht", steps: int = 100_000) -> None:
    """Train DiffusionPolicyCFM on keypoint-obs (environment_state_agent_pos) PushT for
    ~100k steps on mps, with a matched-trunk keypoint DDPM twin for apples-to-apples
    comparison. Verified by the Part 3 checkpoint."""
    raise NotImplementedError


def main() -> None:
    """CLI entrypoint: python -m dp_cfm.train. Verified by the Part 3 checkpoint."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
