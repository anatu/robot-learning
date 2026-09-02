import pytest

pytest.skip("Lesson 03 Part 2 — not implemented yet", allow_module_level=True)


def test_fk_position_matches_mujoco():
    """fk(q)'s position matches mj_forward + data.site("ee_site").xpos to < 1e-10 m over 1,000 random in-limit configs. Verified by: Part 2 checkpoint."""
    raise NotImplementedError


def test_fk_rotation_matches_mujoco():
    """fk(q)'s rotation matches data.site("ee_site").xmat to < 1e-10 (Frobenius norm) over 1,000 random in-limit configs. Verified by: Part 2 checkpoint."""
    raise NotImplementedError
