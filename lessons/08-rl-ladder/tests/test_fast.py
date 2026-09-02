import pytest

pytest.skip("Lesson 08 Part 4 — not implemented yet", allow_module_level=True)


@pytest.mark.fast
def test_baseline_gradient_is_detached():
    """REINFORCE's learned value baseline (variant reward_to_go_baseline) is detached from the policy
    loss graph — no bias leak into the policy gradient."""
    raise NotImplementedError


@pytest.mark.fast
def test_target_network_receives_no_gradients():
    """No gradients flow into DQN's/SAC's target network parameters (phi-) during an update."""
    raise NotImplementedError


@pytest.mark.fast
def test_tanh_log_prob_correction_matches_reference():
    """SAC's tanh-squashing log-prob correction matches
    torch.distributions.TransformedDistribution's log_prob for the same samples."""
    raise NotImplementedError


@pytest.mark.fast
def test_replay_buffer_fifo_and_dtype():
    """ReplayBuffer evicts oldest transitions FIFO at capacity and returns consistent dtypes from
    sample()."""
    raise NotImplementedError
