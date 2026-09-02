import pytest

pytest.skip("Lesson 05 Part 2 — not implemented yet", allow_module_level=True)


def test_lqr_equivalence():
    """On a linear-quadratic problem, ilqr() must converge in one iteration to lqr.lqr's solution,
    agreeing to < 1e-10. This single test localizes most backward-pass bugs."""
    raise NotImplementedError


def test_cost_nonincreasing_across_iterations():
    """The iLQR cost sequence must be strictly non-increasing across iterations."""
    raise NotImplementedError


def test_swingup_converges_to_upright():
    """From a small-random-noise u_init, iLQR swings the cartpole from hanging to upright: converges in
    < 100 iterations, final pole angle within 1e-3 of upright."""
    raise NotImplementedError
