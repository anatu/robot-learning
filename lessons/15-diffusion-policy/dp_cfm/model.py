"""Lesson 15 Part 3 stub — spec in README.md "Part 3 — The sampler study".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from typing import Any


class DiffusionPolicyCFM:
    """Standalone fork of LeRobot's Diffusion Policy U-Net (keypoint obs, same conditioning
    and horizons) trained with a conditional flow-matching velocity-regression loss instead
    of the DDPM epsilon-prediction loss. Verified by the Part 3 checkpoint (Euler-{2,5}
    success sits at-or-near its DDPM twin's success at a fraction of the steps)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def forward(self, obs: Any, noisy_actions: Any, t: Any) -> Any:
        """Predict the velocity field at noise level t, conditioned on obs. Verified by the
        Part 3 checkpoint."""
        raise NotImplementedError

    def sample(self, obs: Any, num_steps: int) -> Any:
        """Euler-integrate the learned velocity field for num_steps to produce an action
        chunk. Verified by the Part 3 checkpoint (evaluated at Euler steps in {1, 2, 5, 10})."""
        raise NotImplementedError


def cfm_loss(velocity_pred: Any, velocity_target: Any) -> Any:
    """Lesson 12/13's conditional flow-matching regression loss on the OT path. Verified by
    the Part 3 checkpoint."""
    raise NotImplementedError
