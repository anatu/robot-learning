"""Lesson 08 Part 1 stub — spec in README.md "Part 1 — REINFORCE on CartPole".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations


def train_reinforce(
    env_id: str = "CartPole-v1",
    variant: str = "reward_to_go_baseline",
    seed: int = 0,
    **hparams,
) -> dict:
    """Train REINFORCE on env_id (2x64 tanh MLP policy, Adam lr 1e-2, batch = 10 episodes, gamma = 0.99).
    variant selects the policy-gradient estimator Psi_t: "full_return", "reward_to_go", or
    "reward_to_go_baseline" (adds a learned value baseline fit by MSE on returns, detached from the
    policy loss). Returns per-update logs including gradient-norm std as the variance proxy.

    Verified by: Part 1 checkpoint (tutorial Eq. 11's ancestor; "reward_to_go_baseline" reaches avg
    return >= 475 over 100 eval episodes fastest with the lowest variance proxy; "full_return" is
    noisiest).
    """
    raise NotImplementedError
