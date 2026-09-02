"""Lesson 07 Part 0 stub — spec in README.md "Part 0 — The scene and the collision oracle".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations


def build_exclusion_list(model: "mujoco.MjModel") -> set:
    """Build the set of self-collision geom-ID pairs (adjacent links, gripper pads) to ignore in
    is_free/edge_free, derived once from the home configuration's contacts.

    Verified by: Part 0 checkpoint (the exclusion list is exact: zero false collisions over 1,000
    random, visually-verified-free configs).
    """
    raise NotImplementedError


def is_free(q: "np.ndarray") -> bool:
    """True iff joint configuration q is collision-free: set qpos, mj_forward, inspect data.ncon
    filtered through the exclusion list.

    Verified by: Part 0 checkpoint (oracle throughput >= 5,000 calls/s; home config free, a
    wrist-into-table config not free, symmetric +/- wrist configs agree).
    """
    raise NotImplementedError


def edge_free(q1: "np.ndarray", q2: "np.ndarray", delta: float = 0.02) -> bool:
    """True iff the straight-line joint-space edge q1->q2 is collision-free, discretized at resolution
    delta (rad, the max joint-space step).

    Verified by: Part 0 checkpoint; reused by rrt.py's steering and shortcut.py's rewiring.
    """
    raise NotImplementedError


def clearance(q: "np.ndarray") -> float:
    """Scene clearance d(q): nearest signed distance among MuJoCo contacts at configuration q (mj_forward
    + nearest contact distance; finite-difference gradients are acceptable).

    Verified by: Part 3 checkpoint (trajopt.py's penalty-form clearance constraint d(q_k) >= d_safe).
    """
    raise NotImplementedError
