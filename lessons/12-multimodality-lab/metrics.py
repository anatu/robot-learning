"""Lesson 12 Part 3 stub — spec in README.md "Part 3 — Measure the multimodality".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from typing import Any


def mode_balance(samples: Any, expert_mode_means: Any) -> float:
    """Mode balance C = 1 - |p_L - p_R|, mode membership by sign(a_x), p_L/p_R the sampled
    left/right fractions. Expert ~= 1; a mode-collapsed head ~= 0.

    Verified by: Part 3 checkpoint (>= 2 generative heads score C > 0.8).
    """
    raise NotImplementedError


def indecision_mass(samples: Any, expert_mode_means: Any) -> float:
    """Indecision mass I = (1/N) #{a : |a_x| < 0.2 * d}, where d = ||mean(left expert actions) -
    mean(right expert actions)||. Expert ~= 0; the MSE head ~= 1 by construction.

    Verified by: Part 3 checkpoint (MSE head I > 0.9; >= 2 generative heads I < 0.1).
    """
    raise NotImplementedError
