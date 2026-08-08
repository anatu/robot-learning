# Lesson 22 — Capstone

**Goal:** one open-ended project that closes a full loop and produces a portfolio artifact. MIT Robotic Manipulation's project format is the template: 1-page proposal → check-ins → 3–5 min video → short report.

## Pick one (or propose your own)
1. **Collect → train → deploy → improve** (recommended, needs SO-101): record a task dataset, fine-tune SmolVLA, deploy async, collect corrections with `lerobot-rollout`'s DAgger-style human-in-the-loop mode, retrain, and quantify the improvement loop over ≥2 iterations.
2. **Sim-to-real:** NVIDIA's SO-101 Isaac Lab path end-to-end — teleop in sim (LeIsaac), GPU-parallel training, GR00T N1.6 post-training, deploy to your physical arm; measure the reality gap explicitly.
3. **Build a benchmark:** a VLA-REPLICA-subset rig (SO-101 + light box + AprilTags, arXiv 2605.20774) with ID/OOD splits; benchmark 3 policies from earlier lessons; publish the protocol so others can reproduce it.
4. **Research extension:** the knowledge-insulation experiment (fine-tune with frozen vs unfrozen VLM backbone; probe VQA degradation vs policy success — extends Driess et al. 2025 at small scale).

## Deliverables
- Proposal (1 page), code, evaluation with a pre-registered protocol, 3–5 minute video, 4-page report (CoRL format), closing blog post for the build-in-public series.

## Done when
Someone else could reproduce your headline number from the repo alone.
