"""Lesson 10 Part 4 stub — spec in README.md "Part 4 — The intervention-budget experiment".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class OracleConfig:
    """TODO: student design. Fields for the two seizure rules: the step-count threshold N (cube
    still ungrasped) and the workspace box bounds the EE must stay inside."""


class ScriptedOracle:
    """Scripted intervener standing in for the human, so the intervention-budget experiment is
    reproducible: seizes control when (a) episode step > N with the cube ungrasped, or (b) the EE
    strays outside a workspace box.

    Verified by: Part 4 checkpoint (sparse oracle intervenes in <= 20% of episodes, generous in
    <= 60%; generous >= sparse >= none in sample efficiency across both seeds).
    """

    def __init__(self, config: OracleConfig) -> None:
        raise NotImplementedError

    def should_intervene(self, obs: Any, info: dict[str, Any], step: int) -> bool:
        """Return True if the oracle seizes control at this step, per the two rules above.

        Verified by: Part 4 checkpoint (intervention fraction matches the sparse/generous targets).
        """
        raise NotImplementedError
