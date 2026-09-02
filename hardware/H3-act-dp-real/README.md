# H3 — ACT & Diffusion Policy on Real Hardware

In this lesson you train the two imitation policies from Lessons 14 and 15 on the dataset you recorded in H2, deploy them on the physical arm, and evaluate them under a protocol that you commit to before the first trial. The evaluation is the substance of the lesson. A demonstration video shows that a policy can work; a pre-registered set of twenty trials per condition, with confidence intervals and a failure taxonomy, produces a number you could defend to someone who did not watch you run it.

| | |
|---|---|
| **Phase** | Hardware track |
| **Time** | ~1 h launch + audit per training run (cloud, parallel), ~1 h deployment debugging, 2 × ~2 h physical eval sessions, ~1 h analysis |
| **Cost** | ~$2–6 cloud GPU (ACT ~1 h + DP ~1–2 h on a 4090/A100; `lerobot-train` runs unmodified on RunPod/Vast) |
| **Prerequisites** | H2 (the dataset), 14–15 (you've trained and evaluated both policies in sim; `eval.py`'s Wilson-CI helper), 16 (async stack, reused in Exercise 6) |
| **Feeds into** | H4 (these numbers are the baseline VLAs must beat), H5 (deployment fluency) |

## Learning objectives

After this lesson you can:

1. **Predict and read** real-data loss curves against your simulation baselines, and describe what a real-data-specific pathology looks like.
2. **Diagnose** the silent failure of broken observation parity from a rollout's behaviour alone.
3. **Pre-register** an evaluation protocol (conditions, start positions, trial count, success criterion, failure taxonomy) before running a single trial, and explain what pre-registration buys.
4. **Quantify** an ID/OOD gap with confidence intervals that are honest about N=20.
5. **Attribute** failures to grasp, perception, policy indecision, or hardware from video evidence.

## Principles

### What changes from simulation

Three things change when a policy leaves simulation. In simulation the environment guaranteed that the observations at evaluation time matched those at training time; on hardware, observation parity is your responsibility, and camera keys, resolutions and preprocessing must match the training data exactly. The 30 Hz control loop now shares one machine with two camera captures and policy inference, so timing is no longer free. And evaluation costs minutes of physical labour per trial, which makes twenty trials per condition the realistic budget. That small trial count is what makes protocol discipline load-bearing rather than decorative.

### Why pre-registration

With twenty trials, a difference of two or three successes is within noise: the 95% confidence interval on 14 out of 20 spans roughly 48% to 85%. Under those conditions, small adjustments made during the evaluation, such as retrying a start that looked unlucky, changing the lighting, or re-aiming a camera, can move the number by more than the effect you are trying to measure. Pre-registration removes those degrees of freedom by fixing the conditions, the start positions, the trial count, and the success criterion before the first trial, in a commit whose timestamp proves the order. VLA-REPLICA (arXiv 2605.20774) is the course's reference standard for this discipline on SO-101-class rigs: a fixed start grid, a fixed trial count, a published protocol, and results reported against it without modification. It is a post-cutoff source, so pull the current protocol details from the paper itself.

### Observation parity fails silently

A policy that is fed the wrong camera under the right key, or the right camera at the wrong resolution, does not crash. It receives inputs of the right shape with the wrong content, and it produces confident, wrong motion, most often drifting to one region of the workspace and stalling there. Nothing in the rollout reports an error, because nothing is malformed. The only reliable defense is mechanical: diff `dataset.features` against the rollout configuration, key by key, before the first trial. Exercise 3 has you plant this failure deliberately so that you recognize it later.

### The two policies on one dataset

ACT trains in about an hour, is fast at inference, and is precise, but it commits hard to its training distribution. Diffusion Policy trains in one to two hours, is slower per step because it denoises iteratively (budget the step count from Lesson 15's sampler study), and handles multimodality natively. On fifty clean single-strategy episodes you should expect the two to be close in in-distribution success. The informative result is where they differ: in out-of-distribution behaviour and in failure style, with ACT tending to be confidently wrong and Diffusion Policy tending to dither indecisively.

**Carry forward**

- Pre-register the conditions, start sequence, trial count, success sentence and failure taxonomy in a commit before the first trial, because at N=20 any adjustment made after seeing results can move the number by more than the effect being measured.
- Twenty trials gives a 95% confidence interval roughly ±20 points wide, so report the interval and never the point estimate alone.
- Observation parity is checked by diffing the dataset features against the rollout configuration, not by watching the arm, because a parity failure produces confident motion rather than an error.
- Every failed trial receives exactly one primary label from the frozen taxonomy, decided from video; a hardware fault is labeled `hardware` and counted, never excluded.
- Policies are interleaved within a session so that rig drift and operator fatigue affect both arms of the comparison equally.

| Source | Read for |
|---|---|
| [IL-on-real-robot docs](https://huggingface.co/docs/lerobot/il_robots) | current train/deploy commands; `lerobot-rollout` strategies |
| Lesson 14/15 `RESULTS.md` (yours) | your sim baselines and hyperparameter deviations, the priors for debugging real runs |
| VLA-REPLICA (arXiv 2605.20774) | the evaluation-protocol reference standard |

## Exercise 1 — Train both policies [Predict → Run]

You train ACT and Diffusion Policy on the H2 dataset in the cloud, using the same commands as in simulation, and read the loss curves against the simulation runs from Lessons 14 and 15. Real data introduces one pathology that simulation did not: inconsistent demonstrations produce a loss that plateaus high, so the simulation curves are the prior against which you read the real ones. The two runs can be launched in parallel.

1. Before launching, write down, from your Lesson 14 and 15 `RESULTS.md`, the final loss you expect for each policy on fifty real episodes relative to simulation (the same order of magnitude, within about 2×), and what a loss that plateaus high on ACT would indicate.
2. Train ACT:
   ```bash
   lerobot-train \
     --dataset.repo_id=<you>/so101_pickplace_50ep \
     --policy.type=act \
     --output_dir=outputs/train/act_h2 --job_name=act_h2 \
     --policy.device=cuda --wandb.enable=true \
     --policy.repo_id=<you>/act_so101_pickplace
   ```
3. Train Diffusion Policy with the same command, substituting `--policy.type=diffusion --output_dir=outputs/train/dp_h2 --policy.repo_id=<you>/dp_so101_pickplace`. Carry over any deviations you settled on in Lesson 15; `lerobot-train --help` on the box is authoritative for current flag names.
4. Read the curves. The loss should fall smoothly as it did in simulation. An L1 loss on ACT that plateaus high usually means inconsistent demonstrations, which is where violations of H2's protocol surface first. Compare the final losses against your simulation runs and reconcile with step 1.
5. The checkpoints push to the Hub through `--policy.repo_id`. Pin what you will evaluate by noting the exact checkpoint step or revision now (`--policy.pretrained_revision` at load time targets it), so that the question of which checkpoint was evaluated has one answer.

If you have no rented GPU, `--job.target=a10g-small` runs the same command on HF Jobs, billed per second and priced with `hf jobs hardware`.

**✅ Checkpoint:** two Hub checkpoints with W&B curves; final losses within about 2× of your simulation equivalents, or the deviation explained; the evaluated revisions pinned in `RESULTS.md`.

## Exercise 2 — Pre-register the protocol [Write]

Here you write `PROTOCOL.md` and commit it before any deployment. The commit timestamp is the pre-registration. The document fixes every degree of freedom that could otherwise be adjusted after seeing results, and it seals your predictions alongside them so that Exercise 4 can reconcile the trials against what you expected.

1. Conditions: ID uses start positions drawn from H2's demonstrated grid cells; OOD-position uses the held-out cells H2 already earmarked; OOD-distractor uses ID cells with one novel object placed at a fixed, marked position. Two OOD conditions make roughly 120 trials across two policies. If session time forces it, drop OOD-distractor to 10 trials each, but decide that now, in writing.
2. Trials: 20 per policy per condition. Start positions follow a fixed, published sequence over the cells; write the literal sequence. Success is H2's criterion sentence, verbatim. A 60 s timeout counts as failure. There are no retries and no excluded trials; a hardware fault mid-trial is recorded as `hardware`, not rerun.
3. Failure taxonomy, frozen: `grasp-miss` (contact but no acquisition), `perception` (approaches the wrong location or ignores the object), `policy-indecision` (dithers, stalls, or oscillates until timeout), `hardware` (servo cutoff, USB stall, camera drop). Every failed trial gets exactly one primary label and a video.
4. Trial sheet template: trial number, condition, start cell, success, time to success, failure label, video filename.
5. Order: interleave policies within a session in A,B,B,A blocks, so that rig drift and operator fatigue do not load onto one policy.
6. Predictions, sealed in the same commit: for each policy, the ID success you expect, the direction and rough size of the gap from ID to OOD-position, and the failure class you expect to dominate. Exercise 4 reconciles these.

**✅ Checkpoint:** `PROTOCOL.md` is committed before the first rollout with the predictions included, and the grid cells and trial sequence are physically marked on the workspace tape.

## Exercise 3 — Verify observation parity, then break it [Diagnose]

In this exercise you check parity mechanically, run a smoke test, and then deliberately swap the two cameras to see what a parity failure looks like on the arm. Experiencing the failure once, with the power switch in hand, is the most reliable way to recognize it later when it happens by accident.

1. Preflight: the camera keys (`front`, `wrist`), resolutions, and fps in the rollout command must match the H2 dataset features exactly. Print `dataset.features` and diff it against your `--robot.cameras` JSON, key by key. Run H2's `PREFLIGHT.md` as well, since this is the same rig with the same witness marks.
2. Smoke test with ACT first, with inference on the Mac on `mps`:
   ```bash
   lerobot-rollout \
     --strategy.type=base \
     --policy.path=<you>/act_so101_pickplace \
     --robot.type=so101_follower --robot.port=<f-port> --robot.id=H1_follower \
     --robot.cameras="{front: {...}, wrist: {...}}" \
     --task="Pick up the cube and place it in the bin" \
     --duration=60
   ```
   Keep the e-stop in reach: a policy's first real rollout is the least predictable motion this arm will make. Run two sane rollouts before the formal trials, because debugging done during trials contaminates them.
3. Now plant the bug. With the arm at H2's home pose, the workspace clear, and your hand on the power switch, swap the two cameras' `index_or_path` values in `--robot.cameras`, leaving the keys unchanged so that nothing rejects the configuration. Before running, write down what you expect the arm to do. Run the same smoke command with `--duration=15` and cut power the moment the motion becomes unsafe. Record what happened. If your LeRobot version validates keys or shapes at load time, note that too: it means only content mismatches, such as swapped cameras or changed exposure, are silent, and those are the ones H2's rig discipline exists to prevent.
4. Restore the correct configuration, re-run the parity diff, and run one more sane rollout.
5. Repeat the preflight and smoke test for Diffusion Policy. Check the achieved control rate in the logs. If denoising cannot hold 30 Hz on `mps`, reduce the DDIM step count according to your Lesson 15 sampler table, and record the setting in the trial sheet.

**✅ Checkpoint:** the parity diff is clean for both policies; the planted swap produced the confident-wrong behaviour you predicted, or your version's load-time guard caught it and you documented that; two sane rollouts per policy.

## Exercise 4 — Run the formal trials [Predict → Run]

You now run the trials exactly as the protocol specifies and compute per-condition success with confidence intervals. The exercise tests the predictions sealed in Exercise 2, and the reconciliation is where you learn which of your intuitions about the two policies survive contact with the hardware.

1. Run the formal trials per the protocol, using `--strategy.type=episodic` (reset phases between episodes) or `base` per trial, whichever your rig makes reliable. Record every trial on camera; a phone on a tripod covering the workspace is fine. Fill in the sheet in real time.
2. Compute per-condition success with 95% Wilson intervals, using the statistics helper from Lesson 14's `eval.py`.
3. Reconcile with the predictions from Exercise 2: which policy's ID number, which gap, and which dominant failure class you got right, and what the ones you got wrong teach.

**✅ Checkpoint:** at least 20 trials per policy in the ID condition are done to protocol with zero excluded trials; the success table has confidence intervals; the predictions are reconciled.

## Exercise 5 — Attribute the failures [Diagnose]

The success table says how often each policy failed; the failure histogram says how, and it is the histogram that supports any claim about the two policies being different. You label every failure from its video against the frozen taxonomy and cut a short reel of representative failures per class.

1. Label every failure video against the frozen taxonomy, with exactly one primary label per trial. Where two labels compete, for instance an arm that hovers and oscillates 2 cm off the cube, which could be `grasp-miss` or `policy-indecision`, write down the evidence that decided it.
2. Produce the failure-taxonomy histogram per policy, and state at least one concrete claim about how ACT and Diffusion Policy fail differently, with the trial numbers that support it.
3. Cut the failure showreel: two to three minutes of representative failures per class. This is the artifact you will return to most often when debugging H4 and H5.

**✅ Checkpoint:** every failure has a video and a label, and the histogram supports a stated difference between the two policies.

## Exercise 6 — Sync versus async inference [Predict → Run]

Lesson 16 derived the condition under which asynchronous inference removes idle frames, using simulated latencies. Here you run the same policy server and robot client against the physical arm and compare loop regularity and success against synchronous execution.

1. Before running, write down which policy you expect to benefit more from asynchronous inference, and why, in terms of the server latency $l_S$ and the action queue; and whether the queue threshold `g` from your Lesson 16 sweep should transfer to a 30 Hz real loop.
2. Run the PolicyServer on the Mac or a cloud GPU and the RobotClient driving the arm, for 5 ID trials per policy.
3. Compare against synchronous execution on control-loop regularity (measured loop-period jitter), qualitative smoothness on video, and success. Reconcile with your prediction from step 1.

**✅ Checkpoint:** asynchronous inference runs on real hardware; the loop-jitter comparison is plotted; a one-paragraph verdict is in `RESULTS.md`.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| Hub checkpoints ×2 | evaluated revision pinned; model cards link this lesson + the H2 dataset |
| `PROTOCOL.md` | committed before first rollout with predictions sealed (git history proves it) |
| Trial sheets + videos | every trial, every failure labeled; videos named per sheet |
| `RESULTS.md` | success table (policy × condition, Wilson CIs), ID/OOD gap quantified, predictions reconciled, parity-bug account, failure-taxonomy histogram per policy, sync-vs-async verdict |
| Failure showreel | 2–3 min cut of representative failures per class |

## Done when

- [ ] At least 20 trials per policy per condition, run to the pre-registered protocol, with zero exclusions.
- [ ] The ID/OOD gap is quantified with confidence intervals for both policies and reconciled against the sealed predictions.
- [ ] The planted parity bug was predicted and observed, or caught by a documented load-time guard.
- [ ] The failure histogram supports at least one concrete claim about how ACT and Diffusion Policy fail differently.
- [ ] Asynchronous deployment ran on the physical arm.
- [ ] The trained-policy demo video exists, and the number you quote is the one in the table rather than the one in the video.

## Self-check

1. What, precisely, does pre-registration protect against at N=20, and what does it not protect against?
2. A trial fails with the arm hovering 2 cm off the cube, oscillating. Which taxonomy label applies, and what evidence would move it to another?
3. Why must camera keys, and not just resolutions, match training? Trace the failure mechanism through the policy's input pipeline.
4. Your OOD-position success is near ID for ACT but collapsed for Diffusion Policy. What would you check before believing it, in the data and in the trial count?
5. Why interleave policies within a session instead of evaluating them on different days?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Policy runs but arm drifts to one region and stalls | observation mismatch (keys/resolution/order): garbage with an in-distribution shape | diff `dataset.features` vs rollout camera config (Exercise 3 step 1) |
| Great in ID trials, dead in OOD | that is the finding, not a bug | report it; resist "fixing" the eval |
| DP misses 30 Hz on `mps` | denoising latency | fewer DDIM steps (Lesson 15 table); or serve from GPU via Exercise 6's async stack |
| Success varies wildly between the two sessions | rig drift (lighting/camera/calibration) between days | witness marks + preflight; if drift confirmed, report sessions separately, don't pool |
| First rollout swings violently | policy extrapolating from a bad first observation | start the arm from H2's home pose; verify with a single `get_observation()` dump before enabling the loop |
| `lerobot-rollout` flag rejected | CLI drift vs docs | `lerobot-rollout --help`; strategies verified Aug 2026: `base`/`episodic`/`sentry`/`highlight`/`dagger` |

## Going deeper

- **Data scaling.** Retrain ACT on a 25-episode subset and run 10 ID trials. This gives one point on the data-scaling curve for about a dollar, and it previews why H4's fine-tuning from a pretrained VLA changes the economics of data collection.
- **Smoothness on hardware.** Compute Lesson 14's mean-squared-jerk metric from the logged real-robot action streams for each policy. In simulation, Diffusion Policy was smoother than un-ensembled ACT, but physical servos add their own filtering. Check whether the ordering survives on hardware and whether jerk correlates with your `grasp-miss` rate.

## References

- [LeRobot IL-on-real-robot docs](https://huggingface.co/docs/lerobot/il_robots): train and rollout commands verified Aug 2026.
- VLA-REPLICA (arXiv 2605.20774): protocol reference standard.
- Your Lesson 14/15/16 `RESULTS.md`: the sim baselines these numbers are read against.
