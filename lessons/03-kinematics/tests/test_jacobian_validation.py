import pytest

pytest.skip("Lesson 03 Part 3 — not implemented yet", allow_module_level=True)


def test_jacobian_matches_mj_jacsite():
    """jacobian(q) matches mujoco.mj_jacSite after mj_forward to max abs diff < 1e-8. Verified by: Part 3 checkpoint."""
    raise NotImplementedError


def test_jacobian_matches_finite_differences():
    """jacobian(q) matches central finite differences (h = 1e-6) of fk() to max abs diff < 1e-5. Verified by: Part 3 checkpoint (the FD check catches errors a shared frame-convention bug with MuJoCo cannot)."""
    raise NotImplementedError
