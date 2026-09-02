"""Lesson 08 Part 1 stub — spec in README.md "Part 1 — REINFORCE on CartPole" (shared plumbing reused
by reinforce/, dqn/, sac/).
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations


class MLP:
    """Feedforward network trunk shared by all three algorithms' policies/critics/value functions.

    TODO(student): choose the constructor signature (input/output/hidden sizes, activation, init).
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        raise NotImplementedError


class ReplayBuffer:
    """FIFO transition replay buffer used by DQN (capacity 1e5) and SAC (capacity 1e6).

    TODO(student): choose the constructor signature (capacity, observation/action shapes, dtype).
    Verified by: Part 4 unit test (buffer FIFO eviction + consistent dtype behavior on sample()).
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def push(self, *args, **kwargs) -> None:
        """Add one transition, evicting the oldest at capacity (FIFO)."""
        raise NotImplementedError

    def sample(self, batch_size: int):
        """Uniformly sample a batch of transitions."""
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class Logger:
    """CSV + W&B logging shared across REINFORCE, DQN, and SAC runs.

    TODO(student): choose the constructor signature (run name, log dir, W&B project/entity).
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def log(self, step: int, **metrics) -> None:
        raise NotImplementedError


def set_seed(seed: int) -> None:
    """Seed python's random, numpy, torch, and the gymnasium env for a reproducible run.

    Verified by: every 3-seed checkpoint in Parts 1-3 depends on this being deterministic per seed.
    """
    raise NotImplementedError
