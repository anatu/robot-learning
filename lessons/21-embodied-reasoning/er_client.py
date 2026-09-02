"""Lesson 21 Part 0 stub — spec in README.md "Part 0 — Access + first grounded call".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Subtask:
    """TODO: fields are the student's design — see README Part 2 schema block (action: pick/place/push/move_to, target, destination, success_check)."""


@dataclass
class Plan:
    """TODO: fields are the student's design — see README Part 2 schema block (goal, subtasks: list[Subtask])."""


class ERClient:
    """Thin wrapper around the Gemini Robotics-ER API.

    Must provide: retries + exponential backoff on rate limits, JSON-schema validation of
    every response (pydantic), a per-call log (prompt, image hash, response, latency), and a
    running call counter. No raw `client.interactions.create` calls anywhere else in the
    codebase — see README Part 0 step 3 and the `er_client.py` Deliverables acceptance criteria.
    """

    def __init__(self, model: str = "gemini-robotics-er-2-preview") -> None:
        """Construct the client against `model`. Verified by Part 0 checkpoint (annotated image with points on the right objects; malformed-JSON path exercised and handled by one retry)."""
        raise NotImplementedError

    def point(self, image: Any, prompt: str) -> list[dict[str, Any]]:
        """Ground `prompt` in `image`; return [{"point": [y, x], "label": ...}] in 0-1000 normalized, y-first coords. Verified by Part 1 checkpoint (hit rate >= ~80% on sim images for unambiguous objects)."""
        raise NotImplementedError

    def plan(self, goal: str, image: Any) -> "Plan":
        """Request a schema-valid Plan decomposing `goal` against `image` (README Part 2 schema). Verified by Part 2 checkpoint (5/5 goals produce schema-valid plans with <= 1 retry each)."""
        raise NotImplementedError

    def verify(self, success_check: str, image: Any) -> bool:
        """Ask ER whether `success_check` holds true of `image`. Verified by Part 2 checkpoint (verification precision/recall computed on >= 40 before/after pairs)."""
        raise NotImplementedError

    def replan(self, goal: str, image: Any, history: list[Any]) -> "Plan":
        """Request a revised Plan given prior subtask `history`. Verified by Part 3 checkpoint (loop replans bounded by max_replans, logged)."""
        raise NotImplementedError
