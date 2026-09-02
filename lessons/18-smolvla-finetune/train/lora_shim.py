"""Lesson 18 Part 2 stub — spec in README.md "Part 2 — LoRA fine-tune".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from typing import Any


def wrap_action_expert_with_lora(
    policy: Any,
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: list[str] | None = None,
) -> Any:
    """Wrap the action expert's linear projections with peft.LoraConfig.

    Only needed if `lerobot-train --help | grep -iE "lora|peft|freeze"` shows no built-in
    parameter-efficient flags for the installed LeRobot version — check that first.
    Verifies: Part 2 checkpoint (LoRA arm trains with >= 10x fewer trainable parameters than Part 1).
    Spec: README.md "Part 2", step 1.
    """
    raise NotImplementedError
