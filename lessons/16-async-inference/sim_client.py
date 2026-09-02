"""Lesson 16 Part 1 stub — spec in README.md "Part 1 — Stand the stack up".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.

Assumption: scaffolds route (a) from the README ("Robot-adapter" — implement LeRobot's
Robot interface around the gym env and reuse RobotClient unchanged), since the README
calls it the less-code, maintained-path option. If you choose route (b) instead, replace
this adapter with your own minimal client loop and say so in RESULTS.md.
"""

from __future__ import annotations

from typing import Any


class PushTRobotAdapter:
    """Wraps gym_pusht/PushT-v0 (or the Lesson 14 aloha env) behind LeRobot's Robot
    interface (connect/get_observation/send_action) so the stock RobotClient can drive it
    unchanged. Verified by the Part 1 checkpoint (5 clean episodes; success consistent
    with Lesson 14's synchronous eval)."""

    def connect(self) -> None:
        """Robot interface: establish the (simulated) connection. Verified by the Part 1
        checkpoint."""
        raise NotImplementedError

    def get_observation(self) -> Any:
        """Robot interface: return the current env observation. Verified by the Part 1
        checkpoint."""
        raise NotImplementedError

    def send_action(self, action: Any) -> None:
        """Robot interface: step the env with the given action. Verified by the Part 1
        checkpoint."""
        raise NotImplementedError


def main() -> None:
    """Build the RobotClientConfig around PushTRobotAdapter and run control_loop() against
    the policy server at the dataset's control rate. Verified by the Part 1 checkpoint
    (debug_visualize_queue_size produces a sawtooth queue plot)."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
