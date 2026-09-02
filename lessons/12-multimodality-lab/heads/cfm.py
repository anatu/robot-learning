"""Lesson 12 Part 2 stub — spec in README.md "Part 2 — Four heads, one training harness" (CFM).
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from typing import Any


class CFMHead:
    """Conditional flow matching: regress a velocity field v_theta(x_t, t, s) onto x1 - x0 along
    the optimal-transport path x_t = (1-t) x0 + t x1, t ~ U[0,1], x0 ~ N(0,I), x1 = the expert
    action. Sample by Euler integration, 10 steps. Real implementation subclasses
    `torch.nn.Module`; this stub omits that base class to stay import-light. Common `sample(s, n)`
    interface shared with the other three heads.

    Verified by: Part 2 checkpoint (loss converges; sampling 1000 actions takes < 5s on mps/CPU);
    Part 3 (C > 0.8, I < 0.1 at the probe state).
    """

    def __init__(self) -> None:
        raise NotImplementedError

    def loss(self, s: Any, a: Any) -> Any:
        """Regress v_theta(x_t, t, s) onto (x1 - x0). Verified by: Part 2 checkpoint."""
        raise NotImplementedError

    def sample(self, s: Any, n: int) -> Any:
        """Euler-integrate the learned velocity field for 10 steps, clipping actions to the env
        bound after integration, to draw `n` actions at state `s`.

        Verified by: Part 3 checkpoint; Self-check Q4 (10 Euler steps vs DDPM's ~100).
        """
        raise NotImplementedError
