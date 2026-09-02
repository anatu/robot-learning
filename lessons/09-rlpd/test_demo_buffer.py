import pytest

pytest.skip("Lesson 09 Part 2 — not implemented yet", allow_module_level=True)


def test_batch_composition_5050_under_rlpd() -> None:
    """Each `rlpd`-composed batch is exactly half demo_buffer, half online_buffer samples."""
    raise NotImplementedError


def test_preload_transitions_fifo_evictable() -> None:
    """Demo transitions inserted under `preload` are evicted FIFO like any online transition."""
    raise NotImplementedError


def test_batch_composition_seeded_indices_reproducible() -> None:
    """The same seed reproduces the same sampled indices across `compose_batch` calls."""
    raise NotImplementedError
