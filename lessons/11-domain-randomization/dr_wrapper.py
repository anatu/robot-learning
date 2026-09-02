"""Lesson 11 Part 1 stub — spec in README.md "Part 1 — The randomization wrapper".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from typing import Any


class DynamicsRandomizationWrapper:
    """Dynamics-randomizing env wrapper for `Pusher-v5`. Per the README's skeleton, the real
    implementation subclasses `gymnasium.Wrapper`; this stub omits that base class to stay
    import-light.

    On each `reset()`, draws multipliers log-uniformly in [1/width, width] and applies them to
    (a) the object's `body_mass` and (b) the object-table sliding friction (`geom_friction[:, 0]`),
    resolving body/geom IDs by name via `mj_name2id` (never hardcoded indices), always restoring
    pristine model values before each draw (multipliers must not compose across resets).

    Verified by: Part 1 checkpoint (tests green; a 10-reset log shows draws spanning the intended
    range; nominal SAC at width=1 shows clear learning progress by 100k steps).
    """

    def __init__(self, env: Any, width: float, seed: int) -> None:
        """Snapshot pristine `body_mass`/`geom_friction` values before any draw. Verified by:
        Part 1 test (width=1 is bit-identical to the raw env)."""
        raise NotImplementedError

    def reset(self, **kwargs: Any) -> Any:
        """Draw fresh multipliers from the pristine snapshot, apply them, and expose the draw in
        `info["dynamics"]`.

        Verified by: Part 1 checkpoint (multipliers within bounds; deterministic under seed;
        consecutive resets don't compound).
        """
        raise NotImplementedError
