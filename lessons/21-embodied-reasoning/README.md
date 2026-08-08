# Lesson 21 — Embodied Reasoning as a Planning Layer

**Goal:** the "ER plans / VLA acts" architecture, hands-on: an embodied-reasoning VLM as high-level planner over a low-level policy.

## Read
- Gemini Robotics-ER 1.5 developer docs (available via plain Gemini API key) — pointing, spatial grounding, task planning, success detection.
- Gemini Robotics 2 announcement (Jul 2026) and PI's Hi Robot post for the hierarchical-reasoning framing.

## Build
1. API-only warm-up (no robot needed): feed ER 1.5 scene images; extract 2D points/boxes for referenced objects, multi-step plans for a tabletop task, and success/failure judgments. Quantify pointing accuracy on ~30 hand-labeled images.
2. Planner-executor loop in sim: ER 1.5 decomposes a language goal into subtasks; your Lesson 14/15 policy (or a scripted controller) executes each; ER verifies completion and replans on failure.
3. Compare against a flat language-conditioned policy on multi-step tasks.

## Deliverables
- The planner-executor harness + evaluation on ≥3 multi-step tasks + a writeup on failure modes (grounding errors vs execution errors).

## Done when
The hierarchical system completes multi-step tasks the flat policy can't, and you can attribute failures to the correct layer.

## Hardware echo
Swap the sim executor for the real SO-101 policy from H3/H4 — this is the natural H-track finale short of the capstone.
