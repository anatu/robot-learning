"""Lesson 12 Part 2 stub — spec in README.md "Part 2 — Four heads, one training harness" (MSE regressor).
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from typing import Any


class MSEHead:
    """The control arm: shared MLP trunk (3x128, ReLU) -> 2D action, trained by MSE regression.
    Real implementation subclasses `torch.nn.Module`; this stub omits that base class to stay
    import-light. Common `sample(s, n)` interface shared with the other three heads.

    Verified by: Part 2 checkpoint (loss converges but plateaus high — cannot fit two modes);
    Part 3 (indecision mass I > 0.9 at the probe state).
    """

    def __init__(self) -> None:
        raise NotImplementedError

    def loss(self, s: Any, a: Any) -> Any:
        """MSE between the predicted and expert action. Verified by: Part 2 checkpoint."""
        raise NotImplementedError

    def sample(self, s: Any, n: int) -> Any:
        """Return `n` action samples at state `s` (deterministic — all `n` are identical, since
        the MSE head has no sampling randomness).

        Verified by: Part 3 checkpoint (mode-balance/indecision metrics via metrics.py).
        """
        raise NotImplementedError
