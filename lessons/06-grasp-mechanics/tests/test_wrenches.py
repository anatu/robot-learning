import pytest

pytest.skip("Lesson 06 Part 0 — not implemented yet", allow_module_level=True)


def test_single_contact_at_origin_is_pure_force():
    """A single contact at the origin contributes zero torque."""
    raise NotImplementedError


def test_two_opposed_contacts_cancel():
    """Two opposed contacts on a unit-width square produce normal wrenches that cancel."""
    raise NotImplementedError


def test_offset_contact_matches_sign_convention():
    """A contact offset in x produces the torque predicted by the module's documented sign convention."""
    raise NotImplementedError


def test_translation_invariance_of_solvability():
    """Translating all contact points and the external wrench reference by the same offset leaves
    grasp-map solvability (form/force closure, force optimization feasibility) invariant."""
    raise NotImplementedError
