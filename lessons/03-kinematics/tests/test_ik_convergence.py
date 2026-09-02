import pytest

pytest.skip("Lesson 03 Part 4 — not implemented yet", allow_module_level=True)


def test_dls_success_rate_on_reachable_targets():
    """DLS ik() converges on >= 99% of 500 reachable targets (rejection-sampled via fk of random q), with <= 3 random restarts on failure. Verified by: Part 4 checkpoint."""
    raise NotImplementedError


def test_dls_bounded_velocity_outside_workspace():
    """On 100 out-of-workspace targets (reachable points scaled by 1.05), DLS terminates at the boundary with bounded ||qdot||, while pure Gauss-Newton's ||qdot|| spikes > 100x the median. Verified by: Part 4 checkpoint."""
    raise NotImplementedError
