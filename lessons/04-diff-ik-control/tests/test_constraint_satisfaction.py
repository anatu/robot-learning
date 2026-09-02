import pytest

pytest.skip("Lesson 04 Part 4 — not implemented yet", allow_module_level=True)


def test_qp_respects_joint_position_limits():
    """On the joint-limit stress trace (circle's far arc demands a wrist pose beyond a joint limit), QPDiffIK's commanded q + qdot*dt stays within jnt_range to machine precision. Verified by: Part 4 checkpoint (Stress 1)."""
    raise NotImplementedError


def test_clipped_pseudo_inverse_violates_limits_or_deforms_trace():
    """Clipped pseudo-inverse either violates jnt_range or veers off-trace on the same stress trace where the QP stays feasible. Verified by: Part 4 checkpoint (Stress 1 comparison)."""
    raise NotImplementedError


def test_qp_respects_velocity_limits_near_singularity():
    """Near the Lesson 03 singular region, QPDiffIK's ||qdot|| stays within the configured velocity bound. Verified by: Part 4 checkpoint (Stress 2)."""
    raise NotImplementedError
