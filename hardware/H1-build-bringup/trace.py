"""H1 Part 6 stub — spec in README.md "Part 6 — Close the loop with your own controller".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TraceLog:
    """TODO: fields are the student's design — see README Part 6 step 3 (per-step timestep, commanded q, measured q)."""


def line_target(length_cm: float = 10.0, speed_cm_s: float = 3.0) -> Any:
    """Generate the 10 cm horizontal-line reference trajectory (Lesson 04 targets). Verified by Part 6 checkpoint (RMS EE error, single-digit mm)."""
    raise NotImplementedError


def circle_target(radius_cm: float = 6.0, speed_cm_s: float = 3.0) -> Any:
    """Generate the 6 cm-radius circle reference trajectory. Verified by Part 6 checkpoint (circle visibly circular)."""
    raise NotImplementedError


def run_trace(
    target: Any,
    robot_port: str,
    robot_id: str = "H1_follower",
    hz: float = 30.0,
    max_step_deg: float = 2.0,
) -> "TraceLog":
    """Drive the follower along `target` via Lesson 04's diff-IK controller (q̇ = f(q, target)) at `hz`, capping per-step joint deltas at `max_step_deg` as a software speed limit. Connects via `SO101Follower(SO101FollowerConfig(port=robot_port, id=robot_id))`. Verified by Part 6 checkpoint (commanded + measured q logged at >= 30 Hz)."""
    raise NotImplementedError


def plot_trace(log: "TraceLog") -> None:
    """Plot per-joint commanded-vs-measured overlay and EE path (reference/sim/real, EE from Lesson 03 fk(q) on measured angles), and report RMS + max EE error in mm. Verified by Part 6 checkpoint (RMS EE error single-digit mm; error source nameable from the plots)."""
    raise NotImplementedError
