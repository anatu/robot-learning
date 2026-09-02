import pytest

pytest.skip("Lesson 13 Part 1 — not implemented yet", allow_module_level=True)


def test_posterior_matches_closed_form():
    """Closed-form q(x_{t-1}|x_t,x_0) = N(mu_tilde_t, beta_tilde_t I) matches brute-force
    Bayes on a dense grid at 5 random (t, x_t, x_0) triples; max abs error < 1e-4 (Part 1 checkpoint)."""
    raise NotImplementedError
