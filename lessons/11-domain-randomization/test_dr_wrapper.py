import pytest

pytest.skip("Lesson 11 Part 1 — not implemented yet", allow_module_level=True)


def test_multipliers_within_bounds() -> None:
    """Drawn mass/friction multipliers stay within [1/width, width] across many resets."""
    raise NotImplementedError


def test_deterministic_under_seed() -> None:
    """The same seed reproduces the same sequence of drawn multipliers."""
    raise NotImplementedError


def test_consecutive_resets_do_not_compound() -> None:
    """A second reset's multiplier is drawn from the pristine model value, not the previous draw."""
    raise NotImplementedError


def test_width_one_matches_raw_env() -> None:
    """width=1 produces dynamics bit-identical to the unwrapped env."""
    raise NotImplementedError
