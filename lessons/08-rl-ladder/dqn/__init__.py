"""Lesson 08 Part 2 stub — spec in README.md "Part 2 — DQN on LunarLander".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations


def train_dqn(
    env_id: str = "LunarLander-v3",
    target_network: bool = True,
    double_dqn: bool = False,
    seed: int = 0,
    **hparams,
) -> dict:
    """Train DQN (2x256 ReLU MLP, ReplayBuffer 1e5, batch 128, lr 1e-3 -> 1e-4 cosine, gamma = 0.99,
    epsilon-greedy 1.0 -> 0.05 over 50k steps, target sync every 1k steps). target_network=False
    disables the target net (phi- = phi, arm b); double_dqn=True decouples argmax from evaluation
    (arm c). Logs both the learning curve and E[max_a Q(s,a)] on a fixed probe-state batch, per
    tutorial Eq. 12.

    Verified by: Part 2 checkpoint (arm a clears avg return >= 200; arm b shows Q-value blow-up or
    oscillation on the probe plot; arm c's Q-estimates track observed returns more closely than arm a's
    — the overestimation-bias gap).
    """
    raise NotImplementedError
