"""
Lesson 04 Parts 0-4 stub — spec in README.md "Part 1 — Open-loop tracking" / "Part 2 — Proportional feedback + gain sweep" / "Part 3 — Break it: mismatch and disturbance" / "Part 4 — The QP tracker".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""
from __future__ import annotations


def run_do_nothing_check() -> None:
    """Run the harness with a do-nothing controller and produce plots/CSV. Verified by: Part 0 checkpoint."""
    raise NotImplementedError


def run_open_loop_experiment() -> None:
    """Run OpenLoopDiffIK on line and circle, once on-trajectory and once with a 1cm offset start; log RMS error for both in one plot. Verified by: Part 1 checkpoint."""
    raise NotImplementedError


def run_gain_sweep(gains: "list[float] | None" = None) -> None:
    """Sweep Kp in {0,1,2,5,10,20,50} s^-1 on the circle with the offset start; plot error-vs-time per gain, steady-state RMS vs Kp, and overshoot vs Kp. Verified by: Part 2 checkpoint."""
    raise NotImplementedError


def run_mismatch_and_disturbance_experiment() -> None:
    """Model-mismatch table (open-loop vs Kp=10 closed-loop, at +5%/+10%/+25% controller-side link-length perturbation) and a disturbance-recovery plot (2deg shoulder_lift offset injected at t=4s for 1s). Verified by: Part 3 checkpoint."""
    raise NotImplementedError


def run_qp_experiments() -> None:
    """QP-vs-DLS unconstrained parity check, joint-limit stress test vs clipped pseudo-inverse, near-singularity velocity-limit stress test, and per-tick solve-time logging. Verified by: Part 4 checkpoint."""
    raise NotImplementedError


def main() -> None:
    """Run every experiment above and regenerate every table/plot in the lesson from scratch, in one command. Verified by: Deliverables (run_experiments.py acceptance criteria)."""
    raise NotImplementedError
