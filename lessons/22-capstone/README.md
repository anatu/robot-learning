# Lesson 22 — Capstone

One open-ended project that closes a full loop — data, training, deployment, evaluation, iteration — and produces a portfolio artifact someone else could reproduce. The format follows MIT Robotic Manipulation's project structure: 1-page proposal → check-ins → 3–5 min video → short report.

| | |
|---|---|
| **Phase** | 6 — Capstone |
| **Time** | ~4 weeks part-time (the week-by-week sketches below assume ~8–10 h/week) |
| **Cost** | option-dependent: $10–40 cloud GPU; option 2 adds Isaac-capable RTX cloud time; options 1/3 need the SO-101 |
| **Prerequisites** | the track you've built: 14/15 (policies + eval harness), 18 (fine-tuning), 16 (async deploy); per-option extras listed in each brief |
| **Feeds into** | the build-in-public series finale; your portfolio |

## Learning objectives

After the capstone you can:

1. **Scope** a research-shaped project to a fixed time and dollar budget, with pre-registered success criteria and a fallback.
2. **Execute** a full loop where the evaluation protocol was committed before the headline experiment ran.
3. **Communicate** the result three ways — a reproducible repo, a 4-page report, a 3–5 minute video — each carrying the same honest headline number.
4. **Defend** the result against the strongest objection you can construct (the report's limitations section is graded like a rebuttal).

## The rules (all options)

- **Pre-registration is binding.** `PROPOSAL.md` and `PROTOCOL.md` merge before the headline experiment runs; the git log is the receipt. Protocol amendments are allowed but must be dated and justified in-file.
- **One headline number.** Every deliverable leads with the same claim (e.g. "correction loop: 42% → 71% ± CI over two iterations"). If you can't state the headline as one sentence with a number, the project isn't scoped yet.
- **Reproduction bar.** A stranger with the repo, the listed hardware, and the README reproduces the headline number without messaging you. That's the "done" test for the whole course.
- **Failure is publishable.** A negative result with a clean protocol and a diagnosed cause meets the bar; a positive result with a leaky protocol does not.

## Option 1 — Collect → train → deploy → improve *(recommended; needs SO-101)*

**Brief.** The full modern loop on your own hardware: record a task dataset (H2 recipe), fine-tune SmolVLA (Lesson 18 recipe), deploy async (Lesson 16), then run ≥ 2 DAgger-style correction iterations with `lerobot-rollout`'s human-in-the-loop mode and quantify the improvement curve. This is the course's thesis — *policies improve with experience, not just data* — executed end-to-end by one person on a $400 robot.

**Headline shape.** Success rate per iteration on a fixed 20-trial pre-registered protocol: baseline → +corrections₁ → +corrections₂, with CIs.

**Risks.** Correction quality is the hidden variable (sloppy takeovers teach sloppy recovery — decide your intervention policy in advance and log every takeover); improvement can saturate after iteration 1 (that's a finding — diagnose *which* failure modes the corrections fixed via H3's taxonomy).

**Milestones.** W1: dataset + fine-tune + pre-registered eval, baseline number locked. W2: async deployment + first correction round (≥ 20 interventions) + retrain. W3: second round + retrain + final eval. W4: report, video, blog post.

## Option 2 — Sim-to-real: the Isaac path end-to-end

**Brief.** NVIDIA's SO-101 sim-to-real learning path as a measured experiment rather than a guided tour: teleop in sim (LeIsaac), GPU-parallel training in Isaac Lab, GR00T post-training per the path's current docs, deploy to your physical arm — and quantify the reality gap explicitly with matched sim and real protocols. Lesson 11's transfer-heatmap discipline, at production scale.

**Headline shape.** Matched-protocol success: sim X% vs real Y% on identical task/reset distributions, plus the gap decomposition (perception vs dynamics — swap real camera frames into sim inference to isolate).

**Risks.** Toolchain weight is the schedule-killer (Isaac needs an RTX cloud instance; budget W1 entirely for bring-up); the NVIDIA path's model versions move — pin what you actually used. Verify the current path docs before proposing (docs.nvidia.com, "Sim-to-Real with SO-101").

**Milestones.** W1: Isaac bring-up + LeIsaac teleop working. W2: train in sim, sim eval locked. W3: deploy real, real eval, gap decomposition. W4: writeups.

## Option 3 — Build a benchmark: a VLA-REPLICA-subset rig

**Brief.** Replicate a subset of VLA-REPLICA (arXiv 2605.20774: SO-101 + controlled light box + off-the-shelf parts, assembleable in under an hour, with ID/OOD protocols), then benchmark three policies from your own course history — ACT (H3), Diffusion Policy (H3), SmolVLA-ft (H4) — under its protocol, and publish your rig spec + results so another builder can reproduce them. The contrarian bet: in a field drowning in models, the scarce artifact is a trustworthy yardstick.

**Headline shape.** A 3-policy × ID/OOD table under a published, replicable protocol — plus a replication delta: your numbers vs any the paper reports for comparable settings.

**Risks.** Physical reproducibility is the whole game — lighting drift, camera pose, reset discipline; instrument all three (light-box readings, AprilTag-anchored camera pose, reset jigs). Related work to cite and differentiate: the SO-101 failure/recovery benchmarking study (arXiv 2606.08881).

**Milestones.** W1: rig build + protocol doc + reset jig. W2: run all three policies ID. W3: OOD conditions + re-runs for variance. W4: publish rig spec, protocol, table, writeups.

## Option 4 — Research extension: knowledge insulation at small scale

**Brief.** Driess et al. 2025 showed naive action fine-tuning erodes a VLA's language/VQA competence, and gradient insulation prevents it. Test whether the effect and the cure replicate at SmolVLA scale: fine-tune with frozen vs unfrozen VLM backbone (× with/without stop-gradient if the codebase permits), probe VQA capability before/after on a fixed prompt set, and measure policy success simultaneously. Small-scale, sim-only, cheapest option — and the most research-shaped.

**Headline shape.** A 2×2 (or 2×1) grid: Δ VQA score vs Δ policy success per regime, ≥ 3 seeds each, with the claim "KI's trade-off does/doesn't appear at 450M scale".

**Risks.** Effect size may be small at this scale — power it: 3 seeds minimum, a VQA probe set big enough (≥ 200 prompts) that a 5-point swing is outside noise; decide the probe metric (exact match vs LLM-judged) in advance. A null result here is genuinely informative — say what it bounds.

**Milestones.** W1: probe set + baseline probes + PROTOCOL.md. W2: the fine-tune grid (cloud, ~$10–20). W3: probes + evals + seeds. W4: writeups.

**Propose your own** instead if something from Lessons 20–21 pulled harder — same rules, same rubric, and the proposal must name which option above it most resembles and why it beats it.

## Deliverables and rubrics

| Artifact | Acceptance criteria |
|---|---|
| `PROPOSAL.md` (1 page) | passes the proposal rubric below; merged before any headline work |
| `PROTOCOL.md` | eval spec exact enough that a stranger runs the same experiment; committed before the headline run |
| Code + configs + seeds | one documented entry point per pipeline stage; reproduces the headline number |
| `report.pdf` (4 pages, CoRL format) | structure below; limitations written as a rebuttal |
| Video (3–5 min) | structure below; includes ≥ 1 failure clip |
| Closing blog post | the build-in-public series finale: what the course claimed, what your numbers showed |

**Proposal rubric** — each row must be unambiguously present:

| Section | Bar |
|---|---|
| Problem + headline claim | one sentence, one number-to-be, falsifiable |
| Protocol sketch | trials, conditions, success definition, CI plan |
| Schedule | 4 weekly milestones, each with a checkable artifact |
| Risks + fallback | top 2 risks, and the salvage plan if W2's milestone slips |
| Budget | $ and hours, itemized |

**Report structure (4 pages).** §1 Problem + claim (¼ p). §2 System: what you built, one figure (1 p). §3 Experiments: protocol, headline table with CIs, one ablation or decomposition (1.5 p). §4 Limitations-as-rebuttal: the three strongest objections and your honest response (¾ p). §5 What you'd do with another month (¼ p).

**Video structure.** 0:00–0:30 the task and the claim, stated over real footage; middle: the system working *and failing*, narrated causally ("it misses here because…"); final 30 s: the headline number on screen, and the one thing you'd tell someone starting the same build.

## Done when

- [ ] Git history shows proposal → protocol → results, in that order.
- [ ] The headline number appears identically in repo README, report, and video — with its CI.
- [ ] A named stranger (or a clean-machine run) reproduced the pipeline from the repo alone.
- [ ] All six artifacts shipped; the blog post links the lot.

## Self-check

1. Your headline improved when you re-ran with a "small fix" after seeing results. What's the legitimate way to report this, and what's the illegitimate one?
2. For your chosen option: which single measurement, if it came out badly, would falsify the project's premise — and does your W1 milestone surface it early?
3. Why does the rubric grade limitations as a *rebuttal* rather than a list?
4. What makes 20 pre-registered trials with CIs more publishable than 100 exploratory ones without?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| W1 slips into W2 | bring-up underestimated (esp. options 1–2) | W1 milestone is deliberately the riskiest item; if it slips, invoke the proposal's fallback *now*, not W3 |
| Headline number drifts across drafts | re-runs cherry-picked | the number comes from one committed script's output artifact; drafts cite the artifact |
| "Reproducible" repo fails on a clean machine | environment rot | pin everything; CI job or a clean cloud-instance dry run in W4 |
| Video runs long and shows only successes | montage instinct | script it to the structure; the failure clip is mandatory |
| Protocol amendment temptation mid-run | OOD condition turned out too hard | amend with date + rationale in PROTOCOL.md and report both pre/post-amendment results |

## References

- MIT Robotic Manipulation — final-project format (manipulation.csail.mit.edu).
- VLA-REPLICA, arXiv 2605.20774; SO-101 failure/recovery benchmark, arXiv 2606.08881 (option 3).
- Driess et al., *Knowledge Insulating VLA Models*, 2025, arXiv:2505.23705 (option 4).
- NVIDIA Sim-to-Real SO-101 learning path — verify current docs (option 2).
- CoRL author kit (report format).
