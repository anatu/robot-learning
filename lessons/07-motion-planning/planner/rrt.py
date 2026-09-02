"""Lesson 07 Part 1 stub — spec in README.md "Part 1 — RRT".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from typing import Callable


def joint_space_weights(jacobian_fn: Callable, n_samples: int = 100) -> "np.ndarray":
    """Per-joint metric weights W, estimated from average task-space displacement per joint via the
    Lesson 03 Jacobian evaluated at n_samples random configs.

    Verified by: Part 1 checkpoint (the weighted ||dq||_W metric beats the unweighted metric on median
    nodes-expanded; report the ratio).
    """
    raise NotImplementedError


def rrt(
    q_start: "np.ndarray",
    q_goal: "np.ndarray",
    is_free: Callable,
    edge_free: Callable,
    eta: float = 0.15,
    p_goal: float = 0.1,
    delta: float = 0.02,
    max_nodes: int = 20000,
    weights: "np.ndarray | None" = None,
) -> "list | None":
    """Grow a tree from q_start toward q_goal with goal-biased sampling (probability p_goal), steer step
    size eta, and edge collision checking at resolution delta. Returns the joint-space path as a list of
    configs, or None if max_nodes is exhausted before reaching the goal.

    Verified by: Part 1 checkpoint (>= 90% success on 50 random feasible problems within budget; the
    animation shows the tree exploring around obstacles, not through them).
    """
    raise NotImplementedError


def generate_problem(is_free: Callable, min_separation: float = 0.25) -> tuple:
    """Rejection-sample a random collision-free (q_start, q_goal) pair with >= min_separation (m)
    task-space distance; verify goal reachability with Lesson 03 IK where task-space goals are used.

    Verified by: Part 1 checkpoint (the 50-problem benchmark set feeding Parts 1-3).
    """
    raise NotImplementedError
