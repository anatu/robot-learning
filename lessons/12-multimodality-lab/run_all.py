"""Lesson 12 stub — spec in README.md "Part 2 — Four heads, one training harness" and "Done when"
(`python run_all.py --seed 0` reproduces all numbers).
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from typing import Any, Literal


def train(
    head: Literal["mse", "cvae", "ddpm", "cfm"],
    dataset: Any,
    seed: int,
) -> Any:
    """Train one head on the expert dataset: Adam 1e-3, batch 256, 20k steps.

    Verified by: Part 2 checkpoint (all four training losses converge; MSE plateaus high).
    """
    raise NotImplementedError


def main() -> None:
    """CLI entry point: train all four heads across 3 seeds, compute metrics.py's C/I at the
    probe state s*=(0,-0.4) (Part 3), run the Part 4 rollouts (standard, widened-jitter x5, and
    the chunking probe), and regenerate plots/ and RESULTS.md's tables.

    Verified by: Done when checklist ("All numbers reproduce from `python run_all.py --seed 0`").
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
