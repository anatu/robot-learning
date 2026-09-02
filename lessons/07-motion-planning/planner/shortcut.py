"""Lesson 07 Part 2 stub — spec in README.md "Part 2 — Shortcut, time-parameterize, execute".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from typing import Callable


def shortcut_path(path: list, edge_free: Callable, iterations: int = 200) -> list:
    """Random-pair rewiring shortcutting: iterations times, pick two random points on the path and
    replace the segment between them with a straight edge if edge_free.

    Verified by: Part 2 checkpoint (shortcutting cuts median path length by >= 30% vs. the raw RRT path).
    """
    raise NotImplementedError
