"""Lesson 17 Parts 2-3 stub — spec in README.md "Part 2 — The blockwise-causal mask" and "Part 3 — Prefix KV cache across denoising steps".
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from typing import Any


def make_blockwise_causal_mask(
    prefix_len: int,
    state_len: int,
    action_len: int,
    device: Any = None,
) -> torch.BoolTensor:
    """(T, T) mask, T = prefix_len + state_len + action_len. True = query may attend to key.

    Full attention within each block; block i attends to blocks j <= i.
    Verifies: Part 2 checkpoint (mask tests green on cpu and mps).
    Spec: README.md "Part 2 — The blockwise-causal mask", step 1.
    """
    raise NotImplementedError


class ToyPi0Denoiser:
    """Toy 4-layer, d=256, 8-head transformer using the Part 2 mask, with a flow-matching velocity head.

    TODO(student): design the layer stack, the prefix(300)/state(1)/action(50) token embeddings, and
    the velocity-prediction head over the action block.
    Spec: README.md "Part 3 — Prefix KV cache across denoising steps", step 1.
    """

    def __init__(
        self,
        prefix_len: int = 300,
        state_len: int = 1,
        action_len: int = 50,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
    ) -> None:
        """Verifies: Part 3 checkpoint (equivalence test green)."""
        raise NotImplementedError

    def forward_full(self, prefix: Any, state: Any, actions: Any, timestep: Any) -> Any:
        """Re-encode the full prefix+state+action sequence for one Euler step (uncached baseline).

        Spec: README.md "Part 3", step 2.
        """
        raise NotImplementedError

    def forward_prefix(self, prefix: Any, state: Any) -> Any:
        """One forward pass over prefix+state; returns per-layer K/V to cache.

        Spec: README.md "Part 3", step 3.
        """
        raise NotImplementedError

    def forward_action_step(self, actions: Any, timestep: Any, cache: Any) -> Any:
        """One cached Euler step: forward only the action block, attending to cached prefix+state KV
        concatenated with the action block's own KV.

        Spec: README.md "Part 3", step 3.
        """
        raise NotImplementedError


def sample_uncached(
    model: ToyPi0Denoiser,
    batch_size: int,
    num_steps: int = 10,
    seed: int | None = None,
) -> Any:
    """Integrate `num_steps` Euler steps from x_0 ~ N(0, I), re-encoding the full sequence each step.

    Verifies: Part 3 checkpoint (uncached baseline for the equivalence + latency comparisons).
    Spec: README.md "Part 3", step 2.
    """
    raise NotImplementedError


def sample_cached(
    model: ToyPi0Denoiser,
    batch_size: int,
    num_steps: int = 10,
    seed: int | None = None,
) -> Any:
    """Integrate `num_steps` Euler steps via one prefix forward pass + cached per-step action forwards.

    Verifies: Part 3 checkpoint (`max |cached - uncached| < 1e-5` in fp32, over 10 random seeds).
    Spec: README.md "Part 3", steps 3-4.
    """
    raise NotImplementedError


def benchmark_latency(
    model: ToyPi0Denoiser,
    step_counts: list[int] | None = None,
    prefix_lengths: list[int] | None = None,
) -> dict[str, Any]:
    """Wall-clock per action chunk, cached vs uncached, across denoising step counts and prefix lengths.

    Defaults per README: step_counts=[1, 2, 5, 10, 20], prefix_lengths=[100, 300, 1000].
    On mps, call torch.mps.synchronize() around timers; time steady-state chunks only.
    Verifies: Part 3 checkpoint (uncached cost grows ~linearly in steps; cached flattens; speedup at
    10 steps exceeds 2x and grows with prefix length). Spec: README.md "Part 3", step 5.
    Feeds `plots/latency.png`.
    """
    raise NotImplementedError
