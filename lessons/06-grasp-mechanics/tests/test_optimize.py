import pytest

pytest.skip("Lesson 06 Part 2-3 — not implemented yet", allow_module_level=True)


def test_feasible_forces_strictly_inside_friction_cones():
    """Every optimize_forces solution has zero constraint slack violation: ||f_t,i|| < mu f_n,i for
    all i on a feasible solve."""
    raise NotImplementedError


def test_infeasibility_grows_as_mu_shrinks():
    """The infeasible range of the 360-degree disturbance sweep grows as mu decreases over
    {0.2, 0.4, 0.8}."""
    raise NotImplementedError


def test_antipodal_candidate_count_on_mesh():
    """sample_antipodal_candidates finds >= 100 valid candidates on the test mug/box mesh."""
    raise NotImplementedError


def test_score_correlation_spearman():
    """The SOCP min-max score and the robustness score from score_grasp correlate with Spearman
    rho > 0.5 across sampled candidates."""
    raise NotImplementedError
