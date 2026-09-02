import pytest

pytest.skip("Lesson 01 Part 4 — not implemented yet", allow_module_level=True)


def test_output_length_matches_deltas():
    """window()'s returned index/pad lists have length == len(deltas). Verified by: Part 4 invariant (a)."""
    raise NotImplementedError


def test_pad_mask_matches_unclamped_out_of_range():
    """is_pad[j] is True iff the unclamped index for delta j fell outside [ep_start, ep_end). Verified by: Part 4 invariant (b)."""
    raise NotImplementedError


def test_no_cross_episode_leakage():
    """All returned indices stay within [ep_start, ep_end) — never a neighboring episode's frames. Verified by: Part 4 invariant (c)."""
    raise NotImplementedError


def test_zero_deltas_returns_query_idx_with_no_padding():
    """With all-zero deltas, output is [query_idx] repeated for every position, and no position is padded. Verified by: Part 4 invariant (d)."""
    raise NotImplementedError


def test_monotone_deltas_give_monotone_indices():
    """Monotone deltas produce monotone (clamped) output indices. Verified by: Part 4 invariant (e)."""
    raise NotImplementedError


def test_oracle_matches_lerobot_dataset():
    """The five invariants hold against real LeRobotDataset outputs for a seeded sample of 200 (query, delta-set) pairs over non-image keys. Verified by: Part 4 checkpoint (oracle test)."""
    raise NotImplementedError


def test_regression_pinned_example():
    """Pinned @example(...) regression for a bug hypothesis found during fuzzing. Verified by: Part 4 checkpoint (at least one pinned regression example)."""
    raise NotImplementedError
