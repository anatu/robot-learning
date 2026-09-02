import pytest

pytest.skip("Lesson 08 Part 4 — not implemented yet", allow_module_level=True)


@pytest.mark.slow
def test_reinforce_cartpole_threshold():
    """REINFORCE (reward_to_go_baseline) clears avg return >= 475 over 100 eval episodes on
    CartPole-v1, 3/3 fixed seeds."""
    raise NotImplementedError


@pytest.mark.slow
def test_dqn_lunarlander_threshold():
    """DQN clears avg return >= 200 on LunarLander-v3, 3/3 fixed seeds."""
    raise NotImplementedError


@pytest.mark.slow
def test_sac_pendulum_threshold():
    """SAC clears avg return >= -200 on Pendulum-v1 within 30k steps, 3/3 fixed seeds."""
    raise NotImplementedError
