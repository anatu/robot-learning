import pytest

pytest.skip("Lesson 14 Part 4 — not implemented yet", allow_module_level=True)


def test_constant_action_invariance():
    """If every live chunk predicts the same constant action, the ensembled output equals
    it exactly (Part 4 checkpoint)."""
    raise NotImplementedError


def test_weights_normalize_to_convex_combination():
    """The ensembled output is a convex combination of the covering predictions; verified
    with one-hot chunks (Part 4 checkpoint)."""
    raise NotImplementedError


def test_warm_up_returns_first_chunk_first_action():
    """At t=0, before any overlap has accumulated, the output equals the first chunk's
    first action (Part 4 checkpoint)."""
    raise NotImplementedError


def test_steady_state_averages_exactly_chunk_size_predictions():
    """After H_a steps of steady-state operation, exactly H_a predictions are being
    averaged for the current timestep (Part 4 checkpoint)."""
    raise NotImplementedError


def test_cross_check_against_lerobot_act_temporal_ensembler():
    """On random tensors, TemporalEnsembler matches LeRobot's ACTTemporalEnsembler with
    max abs deviation < 1e-6 (Part 4 checkpoint)."""
    raise NotImplementedError
