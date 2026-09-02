import pytest

pytest.skip("Lesson 07 Part 0 — not implemented yet", allow_module_level=True)


def test_home_config_is_free():
    """The SO-101 home configuration reports collision-free."""
    raise NotImplementedError


def test_wrist_into_table_not_free():
    """A configuration driving the wrist into the table reports a collision."""
    raise NotImplementedError


def test_symmetric_wrist_configs_agree():
    """Mirror-symmetric +/- wrist configurations agree on collision status."""
    raise NotImplementedError


def test_exclusion_list_zero_false_collisions():
    """Over 1,000 random configs, the exclusion list produces zero false-positive self-collisions
    (spot-checked against 20 visually verified in the viewer)."""
    raise NotImplementedError


def test_oracle_throughput():
    """is_free sustains >= 5,000 calls/s (it's mj_forward, not physics)."""
    raise NotImplementedError
