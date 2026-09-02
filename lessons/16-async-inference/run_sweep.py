"""Lesson 16 Part 3 stub — spec in README.md "Part 3 — The sweep, and the bound".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from typing import Any, Sequence


def sweep(g_values: Sequence[float], regimes: Sequence[str], n_episodes: int = 20) -> Any:
    """Sweep chunk_size_threshold g in {0, 0.3, 0.5, 0.7, 1.0} x policy regime in
    {ACT, DP-DDIM-10, DP-DDPM-100}, >= 20 fixed-seed episodes each; records idle fraction,
    observation-send rate, task success, and the queue trace. Verified by the Part 3
    checkpoint (measured crossover within noise of g* for >= 2 regimes)."""
    raise NotImplementedError


def main() -> None:
    """CLI entrypoint reproducing the full sweep and its plots from one command
    (Done-when: a stranger reruns everything via `python run_sweep.py`)."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
