# Lesson 22 — Capstone

The capstone is one open-ended project that closes a full loop, from data through training, deployment, and evaluation to iteration, under an evaluation protocol that you commit to before the headline experiment runs. Its product is a single headline number, reported with a confidence interval, that a stranger can reproduce from your repository without contacting you. The format follows the final-project structure of MIT's Robotic Manipulation course: a one-page proposal, weekly check-ins, a three-to-five-minute video, and a short report.

| | |
|---|---|
| **Phase** | 6 — Capstone |
| **Time** | ~4 weeks part-time (the week-by-week sketches below assume ~6–8 h/week with AI-assisted coding) |
| **Cost** | option-dependent: $10–40 cloud GPU; option 2 adds Isaac-capable RTX cloud time; options 1 and 3 need the SO-101 |
| **Prerequisites** | the track you have built: 14/15 (policies and the `evaluate(...)` harness), 18 (fine-tuning), 16 (async deployment); per-option extras are listed in each brief |
| **Feeds into** | your portfolio; the optional build-in-public finale |

## Learning objectives

After the capstone you can:

1. **Scope** a research-shaped project to a fixed time and dollar budget, with pre-registered success criteria and a fallback.
2. **Execute** a full loop in which the evaluation protocol was committed before the headline experiment ran.
3. **Communicate** the result in three forms, a reproducible repository, a two-to-three-page report, and a three-to-five-minute video, each carrying the same honest headline number.
4. **Defend** the result against the strongest objection you can construct; the report's limitations section is graded as a rebuttal.

## The rules (all options)

The six rules below apply whichever option you choose. They are the methodological content of the capstone, and each exists to close a specific way in which a project can produce a number that looks better than it is.

### Rule 1 — Pre-registration is binding

`PROPOSAL.md` and `PROTOCOL.md` are merged before the headline experiment runs, and the git log serves as the receipt. Committing the protocol first removes the freedom to adjust conditions after seeing results, which is the main way an honest experimenter ends up with an inflated number. Amendments to the protocol are allowed, but each must be dated and justified in the file itself.

### Rule 2 — One headline number

Every deliverable leads with the same claim, stated in one sentence with one number, for example "correction loop: 42% → 71% ± CI over two iterations". If you cannot state the headline in that form, the project is not yet scoped, because there is no single measurement that would settle it.

### Rule 3 — The reproduction bar

A stranger with the repository, the listed hardware, and the README must be able to reproduce the headline number without messaging you. This is the test of whether the project is finished, and it is also the test of whether the course as a whole has succeeded, since every earlier lesson was built toward artifacts that meet this bar.

### Rule 4 — Failure is publishable

A negative result with a clean protocol and a diagnosed cause meets the bar. A positive result obtained through a leaky protocol does not. The capstone grades the method, not the direction of the outcome.

### Rule 5 — Code is AI-assisted; the science is yours

Specifications, drafts, and plumbing may come from an AI tool. The proposal, the protocol, every prediction written before a run, the analysis, and the limitations-as-rebuttal are non-delegable, because those are the parts in which the understanding is exercised. They are the capstone.

### Rule 6 — Predict before you measure

`PROPOSAL.md` states the headline number you expect and the reason you expect it. The report then reconciles the expectation against the outcome, whichever direction the difference went. A prediction written in advance is what turns a measurement into a test of your understanding rather than a description of what happened.

## Option 1 — Collect, train, deploy, improve *(recommended; needs the SO-101)*

This option runs the full modern loop on your own hardware. You record a task dataset using the H2 recipe, fine-tune SmolVLA using the Lesson 18 recipe, deploy it asynchronously using the Lesson 16 stack, and then run at least two DAgger-style correction iterations using the human-in-the-loop mode of `lerobot-rollout`, quantifying the improvement curve across iterations. This is the course's central claim, that policies improve with experience and not only with more data, executed end to end by one person on a $400 robot.

The headline result is the success rate per iteration on a fixed, pre-registered 20-trial protocol: the baseline, then after the first round of corrections, then after the second, each with a confidence interval.

The main risks are two. Correction quality is the hidden variable, because sloppy takeovers teach sloppy recovery; decide your intervention policy in advance and log every takeover. Improvement may also saturate after the first iteration. If it does, that is a finding rather than a failure, and the analysis should diagnose which failure modes the corrections fixed using the taxonomy from H3.

Milestones: W1, dataset, fine-tune, and pre-registered evaluation, with the baseline number locked. W2, asynchronous deployment, the first correction round with at least 20 interventions, and a retrain. W3, the second round, retrain, and final evaluation. W4, the report and video.

## Option 2 — Sim-to-real through the Isaac path, end to end

This option treats NVIDIA's SO-101 sim-to-real learning path as a measured experiment rather than a guided tour. You teleoperate in simulation with LeIsaac, train with GPU-parallel simulation in Isaac Lab, post-train GR00T according to the path's current documentation, deploy to your physical arm, and quantify the reality gap explicitly using matched simulated and real protocols. It is the transfer-heatmap discipline of Lesson 11 applied at production scale.

The headline result is matched-protocol success, sim X% versus real Y% on identical task and reset distributions, together with a decomposition of the gap into perception and dynamics components. The decomposition is obtained by swapping real camera frames into the simulated inference pipeline, which isolates the perceptual part of the gap.

The main risk is the weight of the toolchain. Isaac requires an RTX-class cloud instance, and bring-up can consume a week on its own, so W1 should be budgeted entirely for it. The model versions in NVIDIA's path also move; pin what you actually used, and verify the current path documentation (docs.nvidia.com, "Sim-to-Real with SO-101") before writing the proposal.

Milestones: W1, Isaac bring-up and LeIsaac teleoperation working. W2, training in simulation with the simulated evaluation locked. W3, real deployment, real evaluation, and the gap decomposition. W4, the writeups.

## Option 3 — Build a benchmark: a VLA-REPLICA-subset rig

This option replicates a subset of VLA-REPLICA (arXiv 2605.20774), which specifies an SO-101 with a controlled light box and off-the-shelf parts, assembleable in under an hour, together with in-distribution and out-of-distribution protocols. You then benchmark three policies from your own course history, ACT and Diffusion Policy from H3 and the fine-tuned SmolVLA from H4, under that protocol, and publish your rig specification and results so that another builder can reproduce them. The premise is that in a field with an abundance of models, a trustworthy yardstick is the scarcer artifact.

The headline result is a table of three policies against the ID and OOD conditions under a published, replicable protocol, plus a replication delta: your numbers against any the paper reports for comparable settings.

The main risk is physical reproducibility, since lighting drift, camera pose, and reset discipline can each move the numbers. Instrument all three, with light-box readings, an AprilTag-anchored camera pose, and reset jigs. Related work to cite and differentiate from is the SO-101 failure and recovery benchmarking study (arXiv 2606.08881).

Milestones: W1, rig build, protocol document, and reset jig. W2, all three policies under the ID condition. W3, the OOD conditions and re-runs for variance. W4, publication of the rig specification, protocol, table, and writeups.

## Option 4 — Research extension: knowledge insulation at small scale

Driess et al. (2025) showed that naive action fine-tuning erodes a VLA's language and VQA competence, and that insulating the backbone from action gradients prevents the erosion. This option tests whether both the effect and the cure replicate at SmolVLA scale. You fine-tune with the VLM backbone frozen and unfrozen, and with and without the stop-gradient if the codebase permits, probe VQA capability before and after on a fixed prompt set, and measure policy success at the same time. It is small-scale, simulation-only, and the cheapest of the four options, and it is the most research-shaped.

The headline result is a 2×2 (or 2×1) grid of the change in VQA score against the change in policy success per regime, with at least three seeds each, supporting the claim that the knowledge-insulation trade-off does or does not appear at 450M-parameter scale.

The main risk is that the effect size may be small at this scale, so the experiment must be powered accordingly: three seeds at minimum, and a VQA probe set large enough (at least 200 prompts) that a five-point swing lies outside the noise. Decide the probe metric, exact match or LLM-judged, in advance. A null result here is genuinely informative, and the report should say what it bounds.

Milestones: W1, the probe set, baseline probes, and `PROTOCOL.md`. W2, the fine-tune grid in the cloud (~$10–20). W3, probes, evaluations, and seeds. W4, the writeups.

You may instead propose your own project if something from Lessons 20 or 21 pulled harder. The same rules and rubric apply, and the proposal must name which of the options above it most resembles and explain why it is the better choice.

## Deliverables and rubrics

| Artifact | Acceptance criteria |
|---|---|
| `PROPOSAL.md` (1 page) | passes the proposal rubric below; merged before any headline work; states the expected headline number |
| `PROTOCOL.md` | an evaluation specification exact enough that a stranger runs the same experiment; committed before the headline run |
| Code + configs + seeds | one documented entry point per pipeline stage; reproduces the headline number from one command |
| `report.pdf` (2–3 pages) | the structure below; limitations written as a rebuttal; the expected and measured headline reconciled |
| Video (3–5 min) | the structure below; includes at least one failure clip |
| Closing blog post *(optional)* | the build-in-public finale, if you publish one: what the course claimed and what your numbers showed |

The proposal is graded against the rubric below, and each row must be unambiguously present.

| Section | Bar |
|---|---|
| Problem + headline claim | one sentence, one number-to-be, falsifiable; and your predicted value with a reason |
| Protocol sketch | trials, conditions, success definition, CI plan |
| Schedule | 4 weekly milestones, each with a checkable artifact |
| Risks + fallback | the top 2 risks, and the salvage plan if W2's milestone slips |
| Budget | $ and hours, itemized |

The report is two to three pages in five sections. Section 1 states the problem and the claim, including the pre-registered expectation (a quarter page). Section 2 describes the system you built, with one figure (half a page). Section 3 presents the experiments: the protocol, the headline table with confidence intervals, and one ablation or decomposition (one page). Section 4 is the limitations section written as a rebuttal, giving the three strongest objections and your honest response to each (half a page). Section 5 says what you would do with another month (a quarter page).

The video is structured in three parts. The first thirty seconds state the task and the claim over real footage. The middle shows the system both working and failing, narrated causally ("it misses here because…"). The final thirty seconds put the headline number on screen and give the one thing you would tell someone starting the same build.

## Done when

- [ ] The git history shows proposal, then protocol, then results, in that order.
- [ ] The headline number appears identically in the repository README, the report, and the video, with its confidence interval, and the report reconciles it against the proposal's prediction.
- [ ] A named stranger, or a clean-machine run, has reproduced the pipeline from the repository alone.
- [ ] The proposal, protocol, code, report, and video have all shipped.

## Self-check

1. Your headline improved when you re-ran with a "small fix" after seeing results. What is the legitimate way to report this, and what is the illegitimate one?
2. For your chosen option, which single measurement, if it came out badly, would falsify the project's premise, and does your W1 milestone surface it early?
3. Why does the rubric grade limitations as a rebuttal rather than a list?
4. What makes 20 pre-registered trials with confidence intervals more publishable than 100 exploratory trials without them?
5. Your AI tool drafted the training and evaluation code. Which parts of the pipeline must you personally be able to explain line by line to a reviewer, and why those?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| W1 slips into W2 | bring-up underestimated (especially options 1–2) | the W1 milestone is deliberately the riskiest item; if it slips, invoke the proposal's fallback now, not in W3 |
| Headline number drifts across drafts | re-runs cherry-picked | the number comes from one committed script's output artifact; drafts cite the artifact |
| "Reproducible" repo fails on a clean machine | environment rot | pin everything; a CI job or a clean cloud-instance dry run in W4 |
| Video runs long and shows only successes | montage instinct | script it to the structure; the failure clip is mandatory |
| Temptation to amend the protocol mid-run | an OOD condition turned out too hard | amend with date and rationale in `PROTOCOL.md`, and report both pre- and post-amendment results |
| Eval script runs but you cannot explain its success criterion | the AI-drafted check was never read against the protocol | diff the script's success test against `PROTOCOL.md`'s sentence before the headline run; they must match verbatim |

## References

- MIT Robotic Manipulation, final-project format (manipulation.csail.mit.edu).
- VLA-REPLICA, arXiv 2605.20774; SO-101 failure/recovery benchmark, arXiv 2606.08881 (option 3).
- Driess et al., *Knowledge Insulating VLA Models*, 2025, arXiv:2505.23705 (option 4).
- NVIDIA Sim-to-Real SO-101 learning path; verify the current docs (option 2).
- CoRL author kit (report format).
