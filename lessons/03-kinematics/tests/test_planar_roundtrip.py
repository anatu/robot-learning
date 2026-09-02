import pytest

pytest.skip("Lesson 03 Part 1 — not implemented yet", allow_module_level=True)


def test_fk_ik_roundtrip_elbow_up():
    """FK(IK_planar(p, branch="up")) == p to 1e-9 for 1,000 uniformly sampled reachable targets. Verified by: Part 1 checkpoint."""
    raise NotImplementedError


def test_fk_ik_roundtrip_elbow_down():
    """FK(IK_planar(p, branch="down")) == p to 1e-9 for 1,000 uniformly sampled reachable targets. Verified by: Part 1 checkpoint."""
    raise NotImplementedError


def test_unreachable_target_reported():
    """ik_planar reports unreachable targets (outside [|l1-l2|, l1+l2]) instead of returning a bogus solution. Verified by: Part 1 checkpoint."""
    raise NotImplementedError
