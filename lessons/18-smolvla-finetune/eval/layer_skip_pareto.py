"""Lesson 18 Part 4 stub — spec in README.md "Part 4 — The layer-skip Pareto".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from typing import Any


def measure_latency(
    policy_path: str,
    batch_size: int = 1,
    num_warm_calls: int = 100,
) -> dict[str, float]:
    """Median + p95 latency in ms per chunk, batch 1, over `num_warm_calls` warm calls.

    On Mac, call torch.mps.synchronize() around timers.
    Verifies: Part 4 checkpoint (half-depth roughly halves LM compute per chunk).
    Spec: README.md "Part 4", step 2.
    """
    raise NotImplementedError


def run_layer_skip_comparison(
    checkpoint_path: str,
    default_layer_n: int | None = None,
    full_depth_layer_n: int | None = None,
) -> dict[str, Any]:
    """Evaluate the full-FT checkpoint at N=L/2 (default) vs full depth: success, latency, peak memory.

    Locate the config knob controlling which VLM layer feeds the expert (inspect
    configuration_smolvla.py in the installed LeRobot) before calling this.
    Verifies: Part 4 checkpoint. Spec: README.md "Part 4", steps 1-2. Feeds `plots/pareto.png`.
    """
    raise NotImplementedError
