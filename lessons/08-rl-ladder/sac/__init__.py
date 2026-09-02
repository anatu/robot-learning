"""Lesson 08 Part 3 stub — spec in README.md "Part 3 — SAC on Pendulum, then HalfCheetah".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations


class SACAgent:
    """Soft Actor-Critic: twin critics (Eq. 15), reparameterized tanh-squashed Gaussian actor (Eq. 16),
    auto-temperature (Eq. 17), 2x256 ReLU nets, lr 3e-4, Polyak tau = 0.005, one gradient step per env
    step. Importable as `from sac import SACAgent` — the contract Lessons 09-11 build on.

    TODO(student): choose the constructor signature (obs_dim, act_dim, hidden sizes, lr, gamma, tau,
    target_entropy = -dim(A), init alpha = 0.2).
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def act(self, obs: "np.ndarray", deterministic: bool = False) -> "np.ndarray":
        """Sample an action from the current policy (or its tanh(mu) mean if deterministic)."""
        raise NotImplementedError

    def update(self, batch) -> dict:
        """One SAC gradient step: critic update (Eq. 15, target computed with torch.no_grad() and a
        fresh action from the current actor), actor update (Eq. 16, reparameterized), temperature
        update (Eq. 17). Returns a dict of loss values for logging, each docstring-traceable to its
        tutorial equation.

        Verified by: Part 3 checkpoint (Pendulum avg return >= -200 within 30k steps, 3/3 seeds;
        HalfCheetah avg return >= 6000 by 500k steps, 3/3 seeds; auto-alpha entropy starts high and
        decays toward -dim(A)).
        """
        raise NotImplementedError
