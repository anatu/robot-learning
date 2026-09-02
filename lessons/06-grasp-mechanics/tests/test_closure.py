import pytest

pytest.skip("Lesson 06 Part 1 — not implemented yet", allow_module_level=True)


def test_four_orthogonal_contacts_form_closure():
    """4 orthogonal frictionless contacts on a planar square achieve form closure; 3 do not."""
    raise NotImplementedError


def test_antipodal_mu_flip_matches_hand_derivation():
    """Sweeping mu in 1e-3 steps for two antipodal contacts on a disk (radius r, contact-line offset c),
    is_force_closure flips from False to True within one step of the hand-derived mu_min."""
    raise NotImplementedError


def test_m_edge_convergence_is_monotone_and_conservative():
    """As m_edges grows over {4, 8, 16, 64}, the mu-flip point converges monotonically toward the
    analytic threshold from the conservative (higher-mu) side."""
    raise NotImplementedError
