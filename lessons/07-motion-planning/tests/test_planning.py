import pytest

pytest.skip("Lesson 07 Part 1-3 — not implemented yet", allow_module_level=True)


def test_rrt_success_rate():
    """RRT succeeds on >= 90% of 50 random feasible problems within the 20,000-node budget."""
    raise NotImplementedError


def test_weighted_metric_beats_unweighted():
    """The Jacobian-weighted joint-space metric beats the unweighted metric on median nodes-expanded."""
    raise NotImplementedError


def test_shortcut_reduces_path_length():
    """Shortcutting cuts median path length by >= 30% relative to the raw RRT path."""
    raise NotImplementedError


def test_time_parameterization_respects_limits():
    """Time-parameterized trajectories, executed through the Lesson 04 tracker, respect
    max |qdot| <= 1.5 rad/s and finish collision-free on >= 45/50 problems."""
    raise NotImplementedError


def test_trajopt_convergence_is_not_feasibility():
    """A trajopt "converged" result on a narrow-passage problem can still collide; the collision oracle
    must be the final arbiter, not the optimizer's convergence flag."""
    raise NotImplementedError


def test_hybrid_matches_rrt_success_with_shorter_paths():
    """The RRT->trajopt hybrid matches RRT's success rate on the 50-problem benchmark with shorter,
    smoother paths than raw RRT."""
    raise NotImplementedError
