"""Lesson 12 Part 2 stub — spec in README.md "Part 2 — Four heads, one training harness" (CVAE).
Implement yourself. Claude scaffolds, reviews, and verifies only — see TEMPLATE.md execution contract.
"""

from __future__ import annotations

from typing import Any


class CVAEHead:
    """Conditional VAE: encoder q_phi(z|s,a), decoder p_theta(a|s,z), z in R^2. Loss = MSE
    reconstruction + beta * KL(q_phi(z|s,a) || N(0,I)), beta=1 to start. Samples via z ~ N(0,I)
    then decode. Real implementation subclasses `torch.nn.Module`; this stub omits that base
    class to stay import-light. Common `sample(s, n)` interface shared with the other three heads.

    Verified by: Part 2 checkpoint (loss converges); Part 3 (C > 0.8, I < 0.1 at a good beta; the
    beta in {0.1, 1, 10} sweep shows posterior collapse — falling C — at high beta).
    """

    def __init__(self, beta: float = 1.0) -> None:
        raise NotImplementedError

    def loss(self, s: Any, a: Any) -> Any:
        """Reconstruction MSE + beta * KL(q_phi(z|s,a) || N(0,I)). Verified by: Part 2 checkpoint."""
        raise NotImplementedError

    def sample(self, s: Any, n: int) -> Any:
        """Draw `n` actions at state `s` via z ~ N(0,I), decoded through p_theta(a|s,z).

        Verified by: Part 3 checkpoint (mode-balance/indecision metrics; beta sweep table).
        """
        raise NotImplementedError
