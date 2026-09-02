# S1 Claims Audit (Lesson 20, Part 4)

<!-- Desk-only. See README.md "Part 4 — The S1 claims audit". Classify first, conclude last —
     every judgment must cite a specific missing or present artifact. -->

## Claims table

| Claim | Number | Measured on (as stated) | What's missing to reproduce | Classification | Artifact that would move it up a bin |
|---|---|---|---|---|---|
| ID success | 96% | | | | |
| OOD success, language-conditioned baseline (~100k pre-training hours) | 9% | | | | |
| OOD success, S1 (~100k pre-training hours) | 66% | | | | |
| Demo-to-post-training-episode equivalence | 1 demo ≈ 380 post-training episodes ≈ 50-100 h teleop | | | | |
| L1-L5 perturbation ladder | language baseline degrades ~3x more | | | | |
| Weight consistency | "the same model weights produced every example" | | | | |
| Long-horizon unseen tasks | 10-minute tasks | | | | |

<!-- Classification per claim: independently verifiable today / verifiable in principle but
     unpublished / unverifiable as stated. -->

## Case study: the 66% vs 9% OOD figure

<!-- >= 3 benign explanations that would shrink the gap without the headline being false. -->

| Benign explanation | Published evidence that would rule it out |
|---|---|
| | |
| | |
| | |

## Dated, falsifiable expectation

<!-- TODO: if the ICL-scaling claim is real, what should a peer-reviewed or open replication show
     within 12 months? Revisit at capstone time. -->
