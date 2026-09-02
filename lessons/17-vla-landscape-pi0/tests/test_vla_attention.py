import pytest

pytest.skip("Lesson 17 Part 2-3 — not implemented yet", allow_module_level=True)


def test_mask_shape() -> None:
    """Mask has shape (T, T) with T = prefix_len + state_len + action_len. Spec: Part 2, step 2."""
    raise NotImplementedError


def test_mask_prefix_rows_attend_only_prefix() -> None:
    """Prefix rows are False on all state/action columns. Spec: Part 2, step 2."""
    raise NotImplementedError


def test_mask_state_rows_attend_prefix_and_state_only() -> None:
    """State rows are False on action columns. Spec: Part 2, step 2."""
    raise NotImplementedError


def test_mask_blocks_bidirectional_on_diagonal() -> None:
    """Every block's diagonal sub-square is all-True (bidirectional within block). Spec: Part 2, step 2."""
    raise NotImplementedError


def test_mask_last_action_row_attends_everything() -> None:
    """The last action row attends to all T positions. Spec: Part 2, step 2."""
    raise NotImplementedError


def test_mask_matches_block_diag_construction() -> None:
    """Mask equals the block-lower-triangular matrix built independently via torch.block_diag + ones
    (two constructions, one truth). Spec: Part 2, step 3.
    """
    raise NotImplementedError


def test_sdpa_mask_convention() -> None:
    """Hand-built 3-token example asserts the True = attend convention used by
    F.scaled_dot_product_attention (not nn.Transformer's True = masked). Spec: Pitfalls table.
    """
    raise NotImplementedError


def test_kv_cache_equivalence() -> None:
    """max |cached - uncached| < 1e-5 in fp32 on the final action chunk, over 10 random seeds.
    Verifies: Part 3 checkpoint. Spec: Part 3, step 4.
    """
    raise NotImplementedError
