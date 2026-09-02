import pytest

pytest.skip("Lesson 05 Part 0 — not implemented yet", allow_module_level=True)


def test_jacobian_matches_finite_difference():
    """Central finite differences (h=1e-6) vs cartpole_jacobians must agree to max abs error < 1e-6
    across 100 random states."""
    raise NotImplementedError


def test_energy_conservation_passive_rollout():
    """A passive (u=0) RK4 rollout from a hanging release must conserve energy: drift < 0.1% over 5 s."""
    raise NotImplementedError
