import pytest

pytest.skip("Lesson 05 Part 1 — not implemented yet", allow_module_level=True)


def test_dare_residual_below_tolerance():
    """The Riccati recursion iterated to convergence satisfies the discrete algebraic Riccati equation
    residual < 1e-8, and matches scipy.linalg.solve_discrete_are."""
    raise NotImplementedError


def test_closed_loop_spectral_radius_stable():
    """rho(A - B K_inf) < 1 for the converged infinite-horizon gain."""
    raise NotImplementedError


def test_basin_of_attraction_thresholds():
    """The nonlinear cartpole under the linearized-about-upright LQR policy recovers from 5deg and 20deg
    initial pole offsets, and fails around 40deg."""
    raise NotImplementedError
