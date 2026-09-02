"""
Lesson 04 Parts 1-4 stub — spec in README.md "Part 1 — Open-loop tracking" / "Part 2 — Proportional feedback + gain sweep" / "Part 4 — The QP tracker".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""
from __future__ import annotations


class OpenLoopDiffIK:
    """Velocity-level task-space tracker qdot = J^+ * pdot*, no feedback: qdot_k = J^+(q_k) pdot*_k. Verified by: Part 1 checkpoint (on-trajectory error grows monotonically; a 1cm offset start never recovers)."""

    def __init__(self, trajectory: "Callable", fk_fn: "Callable | None" = None, jacobian_fn: "Callable | None" = None) -> None:
        raise NotImplementedError

    def step(self, q_meas: "np.ndarray", t: float) -> "np.ndarray":
        """Return qdot for the current tick. Interface contract (H1 imports this): controller owns the trajectory; caller owns state and integration."""
        raise NotImplementedError


class FeedbackDiffIK:
    """Task-space tracker with proportional feedback: qdot_k = J^+(q_k) (pdot*_k + Kp e_k), e_k = p*_k - FK(q_meas). Verified by: Part 2 checkpoint (gain sweep: sluggish -> crisp -> ringing) and Part 3 (>= 10x lower RMS than open-loop at 10% model mismatch; disturbance rejected within ~3/Kp s)."""

    def __init__(self, trajectory: "Callable", kp: float = 0.0, fk_fn: "Callable | None" = None, jacobian_fn: "Callable | None" = None) -> None:
        raise NotImplementedError

    def step(self, q_meas: "np.ndarray", t: float) -> "np.ndarray":
        """Return qdot for the current tick. Interface contract (H1 imports this): controller owns the trajectory; caller owns state and integration."""
        raise NotImplementedError


class QPDiffIK:
    """Constrained differential IK: min ||J qdot - v||^2 + eps ||qdot||^2 s.t. qdot_min <= qdot <= qdot_max, q_min <= q + qdot*dt <= q_max, v = pdot* + Kp e. Verified by: Part 4 checkpoint (matches DLS to ~1e-6 when unconstrained; respects jnt_range where clipped-J+ fails; median solve time < 1 ms)."""

    def __init__(self, trajectory: "Callable", kp: float = 0.0,
                 qdot_limits: "tuple[np.ndarray, np.ndarray] | None" = None,
                 q_limits: "tuple[np.ndarray, np.ndarray] | None" = None,
                 dt: float = 0.02, epsilon: float = 1e-6,
                 fk_fn: "Callable | None" = None, jacobian_fn: "Callable | None" = None) -> None:
        raise NotImplementedError

    def step(self, q_meas: "np.ndarray", t: float) -> "np.ndarray":
        """Return qdot for the current tick, solved via qpsolvers (OSQP backend). Interface contract (H1 imports this): controller owns the trajectory; caller owns state and integration."""
        raise NotImplementedError
