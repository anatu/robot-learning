"""Lesson 12 Part 2 stub — spec in README.md "Part 2 — Four heads, one training harness" (DDPM).
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from typing import Any


class DDPMHead:
    """100-timestep DDPM, cosine (squaredcos_cap_v2-style) schedule, epsilon-prediction, sinusoidal
    timestep embedding (64-d) concatenated to (s, a_t). Ancestral sampling over all 100 steps.
    Real implementation subclasses `torch.nn.Module`; this stub omits that base class to stay
    import-light. Common `sample(s, n)` interface shared with the other three heads.

    Verified by: Part 2 checkpoint (loss converges; loss vs t roughly flat; sampling 1000 actions
    takes < 5s on mps/CPU); Part 3 (C > 0.8, I < 0.1 at the probe state).
    """

    def __init__(self, num_timesteps: int = 100) -> None:
        raise NotImplementedError

    def loss(self, s: Any, a: Any) -> Any:
        """E_{t,eps} ||eps - eps_theta(x_t, t, s)||^2. Verified by: Part 2 checkpoint."""
        raise NotImplementedError

    def sample(self, s: Any, n: int) -> Any:
        """Ancestral sampling over all 100 timesteps to draw `n` actions at state `s`.

        Verified by: Part 3 checkpoint; Self-check Q4 (steps-to-quality vs the CFM head).
        """
        raise NotImplementedError
