"""
Lesson 02 Part 2 stub — spec in README.md "Part 2 — Scripted trajectories".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Episode:
    """TODO: fields are the student's design. One scripted episode: per-timestep qpos (state), ctrl (action), and up/side camera renders, recorded every n_sub physics steps."""


def make_waypoints(target_xy: "tuple[float, float]", home_q: "np.ndarray") -> "list[np.ndarray]":
    """Reach-and-return primitive: home -> hover above target -> descend -> close gripper -> return, as 4-6 joint-space waypoints. Verified by: Part 2 checkpoint (smooth traces, all within jnt_range)."""
    raise NotImplementedError


def cosine_interpolate(q_a: "np.ndarray", q_b: "np.ndarray", max_joint_vel: float = 1.5, dt: float = 0.002) -> "np.ndarray":
    """Minimum-jerk-ish cosine interpolation q(s) = q_a + (q_b - q_a) * (1 - cos(pi*s)) / 2 between two waypoints, segment length sized to respect max_joint_vel. Verified by: Part 2 checkpoint (no steps/spikes in joint traces)."""
    raise NotImplementedError


def randomize_episode_params(rng: "np.random.Generator") -> "tuple[tuple[float, float], np.ndarray]":
    """Sample a target position uniformly over the 10x10 cm table zone and a slightly randomized home pose. Verified by: Part 2 checkpoint (50 episodes, each 5-10s / 150-300 frames)."""
    raise NotImplementedError


def generate_episode(model: "mujoco.MjModel", data: "mujoco.MjData", rng: "np.random.Generator", fps: int = 30) -> Episode:
    """Drive data.ctrl through the interpolated waypoint profile; record (qpos -> state, ctrl -> action, renders) every n_sub physics steps. Verified by: Part 2 checkpoint (commanded-vs-measured traces show a small tracking lag — state != action evidence)."""
    raise NotImplementedError


def generate_dataset(n_episodes: int = 50, seed: int = 0) -> "list[Episode]":
    """Generate n_episodes scripted reach-and-return episodes, seeded and deterministic. Verified by: Deliverables (one command regenerates the full dataset deterministically)."""
    raise NotImplementedError
