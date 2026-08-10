# H3 — ACT & Diffusion Policy on Real Hardware

Train both workhorse imitation policies on your H2 dataset, deploy them on the arm, and evaluate them with a pre-registered 20-trial ID/OOD protocol — the difference between "it works (in the demo I chose to film)" and a number you'd defend.

| | |
|---|---|
| **Phase** | Hardware track |
| **Time** | ~1 h launch + audit per training run (cloud, parallel), ~1 h deployment debugging, 2 × ~2 h physical eval sessions, ~1 h analysis |
| **Cost** | ~$2–6 cloud GPU (ACT ~1 h + DP ~1–2 h on a 4090/A100; `lerobot-train` runs unmodified on RunPod/Vast) |
| **Prerequisites** | H2 (the dataset), 14–15 (you've trained and evaluated both policies in sim), 16 (async stack, reused in Part 4) |
| **Feeds into** | H4 (these numbers are the baseline VLAs must beat), H5 (deployment fluency) |

## Learning objectives

After this lesson you can:

1. **Train** sim-validated recipes on real teleop data and read the loss curves for real-data-specific trouble.
2. **Deploy** a policy on physical hardware with exact observation parity — and debug the silent failure when parity is violated.
3. **Pre-register** an evaluation protocol (conditions, start positions, N, success criterion, failure taxonomy) before running a single trial, and explain what pre-registration buys.
4. **Quantify** an ID/OOD gap with confidence intervals honest about N=20.
5. **Attribute** failures to grasp, perception, policy indecision, or hardware — from video evidence, not vibes.

## Background

**What changes from sim (Lessons 14–15).** Three things: observation parity is now *your* problem (camera keys, resolutions, and processing must match training exactly — in sim the env guaranteed it); the 30 Hz control loop shares a machine with two cameras and inference; and evaluation costs minutes of physical labor per trial, so N=20 per condition is the realistic budget — which makes protocol discipline load-bearing rather than decorative.

**Why pre-registration.** With N=20, success differences of 2–3 trials are noise (a 95% CI on 14/20 spans roughly 48–85%). The temptation is to nudge conditions mid-eval — retry a "bad start," swap lighting, re-aim a camera — until the number looks right. Pre-registering conditions, start positions, N, and the success criterion *before* the first trial removes those degrees of freedom. VLA-REPLICA (arXiv 2605.20774) is the course's reference standard for this discipline on SO-101-class rigs — mirror its structure: fixed start grid, fixed trial count, published protocol, results reported against it unmodified. (Post-cutoff source: pull the current protocol details from the paper itself.)

**The two policies, one dataset.** ACT (~1 h train) is fast at inference and precise but commits hard to its training distribution. DP (~1–2 h) is slower per step (iterative denoising — budget from Lesson 15's sampler study) but handles multimodality natively. On 50 clean single-strategy episodes, expect them close in ID success; the interesting result is *where* they differ (OOD behavior, failure styles: ACT confidently wrong vs DP indecisive-dithering).

| Source | Read for |
|---|---|
| [IL-on-real-robot docs](https://huggingface.co/docs/lerobot/il_robots) | current train/deploy commands; `lerobot-rollout` strategies |
| Lesson 14/15 `RESULTS.md` (yours) | your sim baselines and hyperparameter deviations — the priors for debugging real runs |
| VLA-REPLICA (arXiv 2605.20774) | the evaluation-protocol reference standard |

## Part 1 — Train both policies (cloud, parallelizable)

1. ACT:
   ```bash
   lerobot-train \
     --dataset.repo_id=<you>/so101_pickplace_50ep \
     --policy.type=act \
     --output_dir=outputs/train/act_h2 --job_name=act_h2 \
     --policy.device=cuda --wandb.enable=true \
     --policy.repo_id=<you>/act_so101_pickplace
   ```
2. DP: same with `--policy.type=diffusion --output_dir=outputs/train/dp_h2 --policy.repo_id=<you>/dp_so101_pickplace`. Carry over any deviations you settled in Lesson 15 (`lerobot-train --help` on the box is authoritative for current flag names).
3. Real-data curve reading, both runs: loss should fall smoothly as in sim; a *plateauing-high* L1 on ACT often means inconsistent demos (H2 protocol violations surface here first). Compare final losses against your sim runs — same order of magnitude expected.
4. Checkpoints push to the Hub via `--policy.repo_id`. Pin what you'll evaluate: note the exact checkpoint step / revision now (`--policy.pretrained_revision` at load time targets it) so "which checkpoint did we eval?" has one answer.

No rented GPU? `--job.target=a10g-small` runs the same command on HF Jobs — pay-per-second, priced with `hf jobs hardware`.

**✅ Checkpoint:** two Hub checkpoints with W&B curves; final losses within ~2× of your sim equivalents; evaluated revisions pinned in `RESULTS.md`.

## Part 2 — Pre-register the protocol (before any deployment)

Commit `PROTOCOL.md` to the repo — the commit timestamp is the pre-registration:

1. **Conditions:** ID = start positions drawn from H2's demonstrated grid cells; OOD-position = the held-out cells H2 already earmarked; OOD-distractor = ID cells + one novel object placed at a fixed marked position. (Two OOD conditions ≈ 120 total trials across two policies; drop OOD-distractor to 10 trials each if session time forces it — decide *now*, in writing.)
2. **Trials:** 20 per policy × condition; start positions follow a fixed published sequence over the cells (write the literal sequence); success = H2's criterion sentence, verbatim; 60 s timeout = failure; no retries, no excluded trials — a hardware fault mid-trial is recorded as `hardware`, not rerun.
3. **Failure taxonomy, frozen:** `grasp-miss` (contact but no acquire) / `perception` (approaches wrong location or ignores object) / `policy-indecision` (dithers, stalls, oscillates until timeout) / `hardware` (servo cutoff, USB stall, camera drop). Every failed trial gets exactly one primary label + a video.
4. **Trial sheet template:** trial #, condition, start cell, success, time-to-success, failure label, video filename.
5. **Order:** interleave policies within a session (A,B,B,A blocks) so rig drift and operator fatigue don't load onto one policy.

**✅ Checkpoint:** `PROTOCOL.md` committed before the first rollout; grid cells and trial sequence physically marked on the workspace tape.

## Part 3 — Deploy and evaluate (2 sessions, ~2 h each)

1. Observation-parity preflight (the #1 silent killer): camera *keys* (`front`, `wrist`), resolutions, and fps in the rollout command must match the H2 dataset features exactly — print `dataset.features` and diff against your `--robot.cameras` JSON, key by key. Run H2's `PREFLIGHT.md` too (same rig, same witness marks).
2. Smoke test, ACT first (inference on the Mac, `mps`):
   ```bash
   lerobot-rollout \
     --strategy.type=base \
     --policy.path=<you>/act_so101_pickplace \
     --robot.type=so101_follower --robot.port=<f-port> --robot.id=H1_follower \
     --robot.cameras="{front: {...}, wrist: {...}}" \
     --task="Pick up the cube and place it in the bin" \
     --duration=60
   ```
   E-stop in reach — a policy's first real rollout is the least predictable motion this arm will ever make. Two sane rollouts before formal trials; debugging happens now or never.
3. Formal trials per the protocol, `--strategy.type=episodic` (reset phases between episodes) or `base` per trial — whichever your rig makes reliable. Camera-record every trial (phone on tripod covering the workspace is fine); fill the sheet in real time.
4. DP: same flow. Check achieved control rate in the logs — if denoising can't hold 30 Hz on `mps`, drop to DDIM steps per your Lesson 15 sampler table and *record the setting in the trial sheet*.
5. Compute per-condition success with 95% Wilson intervals (Lesson 14's `eval.py` stats helper); label every failure video against the taxonomy.

**✅ Checkpoint:** 20 trials × 2 policies × ID condition minimum done to protocol with zero excluded trials; every failure has a video + label.

## Part 4 — Sync vs async inference (~1 h)

Lesson 16's stack, now with physical consequences: run PolicyServer on the Mac (or a cloud GPU) and RobotClient driving the arm; 5 ID trials per policy async. Compare against sync on control-loop regularity (measured loop period jitter), qualitative smoothness on video, and success. DP benefits more (inference latency hides behind the action queue) — verify or refute on your rig, and note whether queue-threshold `g` from your Lesson 16 sweep transfers.

**✅ Checkpoint:** async runs on real hardware; loop-jitter comparison plotted; one-paragraph verdict in `RESULTS.md`.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| Hub checkpoints ×2 | evaluated revision pinned; model cards link this lesson + the H2 dataset |
| `PROTOCOL.md` | committed before first rollout (git history proves it) |
| Trial sheets + videos | every trial, every failure labeled; videos named per sheet |
| `RESULTS.md` | success table (policy × condition, Wilson CIs), ID/OOD gap quantified, failure-taxonomy histogram per policy, sync-vs-async verdict |
| Failure showreel | 2–3 min cut of representative failures per class — the most instructive artifact you'll produce this course |

## Done when

- [ ] ≥ 20 trials per policy per condition, run to the pre-registered protocol, zero exclusions.
- [ ] ID/OOD gap quantified with CIs for both policies.
- [ ] Failure histogram supports at least one concrete claim about how ACT and DP fail *differently*.
- [ ] Async deployment ran on the physical arm.
- [ ] The trained-policy demo video exists — but the number you quote is the table, not the video.

## Self-check

1. What, precisely, does pre-registration protect against at N=20 — and what does it *not* protect against?
2. A trial fails with the arm hovering 2 cm off the cube, oscillating. Which taxonomy label, and what evidence would move it to another?
3. Why must camera *keys* (not just resolutions) match training? Trace the failure mechanism through the policy's input pipeline.
4. Your OOD-position success is near-ID for ACT but collapsed for DP. What would you check before believing it — in the data, and in N?
5. Why interleave policies within a session instead of evaluating them on different days?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Policy runs but arm drifts to one region and stalls | observation mismatch (keys/resolution/order) — garbage in-distribution-shaped | diff `dataset.features` vs rollout camera config; this is Part 3 step 1 for a reason |
| Great in ID trials, dead in OOD | that's the finding, not a bug | report it; resist "fixing" the eval |
| DP misses 30 Hz on `mps` | denoising latency | fewer DDIM steps (Lesson 15 table); or serve from GPU via Part 4's async stack |
| Success varies wildly between the two sessions | rig drift (lighting/camera/calibration) between days | witness marks + preflight; if drift confirmed, report sessions separately — don't pool |
| First rollout swings violently | policy extrapolating from a bad first observation | start the arm from H2's home pose; verify with a single `get_observation()` dump before enabling the loop |
| `lerobot-rollout` flag rejected | CLI drift vs docs | `lerobot-rollout --help`; strategies verified Aug 2026: `base`/`episodic`/`sentry`/`highlight`/`dagger` |

## Stretch

- **Data scaling:** retrain ACT on a 25-episode subset, run 10 ID trials. One point on the data-scaling curve for ~$1 — and a preview of why H4's fine-tuning from a pretrained VLA changes the economics.
- **Smoothness on hardware:** compute Lesson 14's mean-squared-jerk metric from the logged real-robot action streams, per policy. Sim said DP is smoother than un-ensembled ACT; physical servos add their own filtering — check whether the ordering survives contact with reality, and whether jerk correlates with your `grasp-miss` rate.

## References

- [LeRobot IL-on-real-robot docs](https://huggingface.co/docs/lerobot/il_robots) — train + rollout commands verified Aug 2026.
- VLA-REPLICA (arXiv 2605.20774) — protocol reference standard.
- Your Lesson 14/15/16 `RESULTS.md` — the sim baselines these numbers are read against.
