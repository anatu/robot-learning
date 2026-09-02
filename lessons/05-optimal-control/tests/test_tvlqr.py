import pytest

pytest.skip("Lesson 05 Part 3 — not implemented yet", allow_module_level=True)


def test_open_loop_replay_fails_under_perturbation():
    """Open-loop replay of the nominal control sequence fails for most of 50 +/-10% perturbed initial
    draws — the contrast case that motivates TVLQR."""
    raise NotImplementedError


def test_tvlqr_handoff_recovery_rate():
    """TVLQR, handing off to the infinite-horizon LQR of lqr.py once |theta - pi| < 0.1, recovers >= 90%
    of 50 +/-10% perturbed initial draws."""
    raise NotImplementedError


def test_tvlqr_robust_to_mass_mismatch():
    """TVLQR gains computed on the nominal plant still stabilize a plant with m_p +20%, quantified by
    the same recovery-rate metric."""
    raise NotImplementedError
