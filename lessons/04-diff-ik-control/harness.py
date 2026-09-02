"""
Lesson 04 Part 0 stub — spec in README.md "Part 0 — Reference trajectories + harness".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""
from __future__ import annotations


def simulate(controller: "object", model: "mujoco.MjModel", data: "mujoco.MjData", dt: float = 0.02, duration: float = 5.0) -> dict:
    """Kinematic-stepping sim loop at dt=20ms (50 Hz): controller.step -> integrate -> write qpos -> mj_forward -> measure (no mj_step; this is the kinematics-level controller). Returns a log dict consumed by compute_metrics(). Verified by: Part 0 checkpoint (runs a do-nothing controller and produces plots/CSV)."""
    raise NotImplementedError


def compute_metrics(log: dict) -> dict:
    """RMS and max tracking error, max ||qdot||, error-vs-time curve, from a simulate() log. Verified by: Parts 1-4 checkpoints (the numbers and plots reported in RESULTS.md)."""
    raise NotImplementedError
