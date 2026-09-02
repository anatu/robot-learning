# H2 — Real Teleop Data Collection

In this lesson you record and publish a fifty-episode pick-and-place dataset on your own hardware. That dataset is the raw material for every policy in H3 through H5, and because imitation learning reproduces whatever it is shown, the quality of these demonstrations sets an upper bound on the quality of every policy trained from them. The lesson turns that observation into a written protocol: a task specification, a preflight checklist, binding rules for the recording sessions, and an audit that checks the result against what you planned.

| | |
|---|---|
| **Phase** | Hardware track |
| **Time** | 1 session task design + rig lock-down (~2 h), 1–2 recording sessions (~2–3 h for 50+ episodes incl. resets), ~1 h audit + card + writeup |
| **Cost** | $0 |
| **Prerequisites** | H1 (calibrated pair, cameras, teleop fluency), 01–02 (you know exactly what a valid v3 dataset looks like) |
| **Feeds into** | H3 (ACT/DP train on this), H4 (SmolVLA fine-tune + eval task), H5 (demo buffer) |

## Learning objectives

After this lesson you can:

1. **Design** a manipulation task whose reset, start distribution, and success criterion are defined precisely enough to be evaluated twenty trials at a time later.
2. **Run** `lerobot-record` fluently: episode and reset phases, keyboard flow control, resuming, and recovery from a crashed session.
3. **Explain** the mechanism by which each data-quality rule changes the trained policy, and enforce those rules during recording rather than rationalizing deviations afterwards.
4. **Predict and audit** your own dataset: episode statistics, synchronization, coverage of the start distribution, and a failure log others can learn from.
5. **Publish** a Hub dataset with a card from which a stranger could reproduce the setup.

## Principles

### Why the recording session decides everything downstream

ACT and Diffusion Policy imitate the demonstrations they are given; they do not filter or repair them. This has three consequences, and each becomes a rule in Exercise 3. First, every hesitation in a demonstration becomes a hesitation in the policy. Lesson 14 showed that a state where the demonstrator paused becomes a state where the policy pauses, and it may never leave. Second, if you use two different grasp strategies across episodes, a policy that cannot represent multiple modes will average them into a motion nobody demonstrated, and even a generative head that can represent both modes pays for the split in sample efficiency. Third, start positions you never demonstrated are out of distribution at evaluation time. H3's protocol evaluates in-distribution and out-of-distribution positions separately, so the coverage you record now determines which of H3's numbers you can claim at all.

The LeRobot documentation's rule of thumb is the right bar for a demonstration: you should be able to perform the task yourself while looking only at the camera images. If the object leaves both camera frames, the policy has no way to find it either.

### How recording flows

`lerobot-record` alternates a record phase, whose length is set by `--dataset.episode_time_s` (default 60 s), with a reset phase, set by `--dataset.reset_time_s` (default 60 s), during which you physically restage the scene. The reset phase is idle time by design and is not part of the episode. Three keyboard controls manage the flow. **→** or **n** ends the current episode early and moves on; use it the moment the place completes, so that episodes are not padded with idle frames. **←** or **r** discards the current episode and re-records it. **ESC** or **q** ends the session, encodes the videos, and pushes. Data lands in `~/.cache/huggingface/lerobot/<repo-id>` and is pushed to the Hub by default.

### What to do with failed demonstrations

You will fail some demonstrations, and the dataset needs a written policy for them before the first one happens. There are two defensible options. A curated dataset re-records every failure with **←**, so that the dataset contains successes only; this is the right choice for H3's behaviour-cloning training, which has no way to learn from a failure except to imitate it. A labeled dataset keeps the failures and notes their episode indices in the card, which gives H5's reward classifier extra positive and negative examples. Choose the curated option unless you are already committed to H5. In either case the card states the policy. What is not defensible is a dataset with unlabeled failures mixed in, because a downstream trainer cannot distinguish them from successes.

### The rig is part of the dataset

Lighting, camera pose, and camera auto-adjustment are features of the images, and the policy learns them along with everything else. Anything that drifts between sessions is therefore a distribution shift, and unlike a bad demonstration it cannot be re-recorded away after the fact. The protocol handles this physically rather than by vigilance: witness marks fix the positions of the robot base, container and cameras; lamp-only lighting removes daylight variation; and locked UVC controls stop auto-exposure and white balance from changing image statistics mid-session. A preflight checklist verifies all of these before every session.

**Carry forward**

- A demonstration is usable only if you could perform the task from the camera images alone, because the policy has no other input.
- Demonstrations should use one grasp strategy, contain no pauses, and cover the same start distribution that will be evaluated, because a policy imitates hesitation, averages over strategies, and cannot generalize to positions it never saw.
- The policy for failed demonstrations is written before recording begins; mixing unlabeled failures into the dataset is the one choice that cannot be defended, because the trainer cannot tell them apart from successes.
- Rig drift between sessions is a distribution shift that cannot be undone, so it is prevented physically with witness marks and a preflight checklist rather than by attention.
- `--resume=true` counts additional episodes rather than the total, and it requires `--dataset.root`.

| Source | Read for |
|---|---|
| Tutorial §1.3 | the recording recipe this lesson instantiates (its Code 2 is v0.4-era — the commands below are the current form) |
| [Recording docs](https://huggingface.co/docs/lerobot/il_robots) | flag reference, resume semantics, keyboard-backend caveats |
| [HF "what makes a good dataset" post](https://huggingface.co/blog/lerobot-datasets#what-makes-a-good-dataset) | the community's accumulated data-quality advice, cross-checked against your protocol |

## Exercise 1 — Design the task and lock the rig [Write]

The decisions you make in this exercise are frozen for H3 through H5, because changing the task after training means recollecting the dataset. You write `TASK.md`, which specifies the object, the start distribution, the success criterion and the failed-demo policy, and `PREFLIGHT.md`, the checklist that is run before every recording session from now on.

1. Choose the task: a single pick-and-place of one graspable object (rigid, at least 3 cm, matte; a wooden cube is the usual choice) from a start zone into a fixed container. One behaviour, one object, no distractors. Variation is introduced later, when H4 probes for it; it does not belong in the base dataset.
2. Define the start distribution physically. Tape a grid into the start zone, for example 3×3 cells over roughly 15×15 cm. Plan about 50 episodes at roughly 10 per position over about 5 grid cells, matching the documentation's guidance of 10 episodes per location, and log the cell for every episode as you record. H3's in-distribution evaluation draws from these cells, and its out-of-distribution condition uses the cells you deliberately hold out, so decide now which those are.
3. Write the success criterion as a sentence, for example "object fully inside container, arm returned to home, within 60 s." This sentence is reused verbatim in H3's trial sheets.
4. Lock the rig. Mount the overhead camera rigidly, framing the start zone and the container; aim the wrist camera so that it shows the fingers; light the scene with a desk lamp and close the blinds, because daylight drift between sessions is a distribution shift you cannot undo; and tape witness marks around the robot base, the container, and the camera positions so that the rig can be restaged exactly.
5. Disable camera auto-adjustment. Auto-exposure and auto-white-balance change image statistics mid-session. Fix them through the UVC controls where the camera model allows it, and verify by comparing frame brightness at the start and end of a session.
6. Write `PREFLIGHT.md`, to be run before every session in this lesson and later: power on; run `lerobot-find-cameras opencv` and re-verify the index-to-camera map; check that the witness marks are aligned; lamp on, blinds closed; 30 seconds of teleoperation warm-up; one throwaway episode with both views inspected.
7. For each of the seven rules in Exercise 3, write one sentence in `TASK.md` naming the mechanism by which breaking it would change the trained policy. This is objective 3; "because the documentation says so" does not count.

**✅ Checkpoint:** the task specification (object, grid, per-cell episode counts, held-out cells, success sentence, failed-demo policy, per-rule mechanisms) is written in `TASK.md` before any recording, and there is a phone photo of the rig with the witness marks visible.

## Exercise 2 — Dress rehearsal [Predict → Run]

Before the real sessions you record a three-episode throwaway dataset with the exact command you will use later. The rehearsal exercises the keyboard flow and the reset, and it checks that the dataset carries the frame rate, camera keys and shapes you expect. Writing those expectations down first turns the rehearsal into a test of your understanding of the format from Lessons 01 and 02.

1. Before recording, write down the episode length in frames you expect for a clean demonstration at 30 fps, the camera keys and tensor shapes the dataset should contain, and what the action trace should look like around the moment you press **→**.
2. Record the throwaway dataset with the real command:
   ```bash
   lerobot-record \
     --robot.type=so101_follower --robot.port=<f-port> --robot.id=H1_follower \
     --robot.cameras="{front: {type: opencv, index_or_path: <i>, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: <j>, width: 640, height: 480, fps: 30}}" \
     --teleop.type=so101_leader --teleop.port=<l-port> --teleop.id=H1_leader \
     --display_data=true \
     --dataset.repo_id=<you>/h2_rehearsal \
     --dataset.num_episodes=3 \
     --dataset.single_task="Pick up the cube and place it in the bin" \
     --dataset.episode_time_s=60 --dataset.reset_time_s=15 \
     --dataset.push_to_hub=false
   ```
   The keyboard flow keys need the terminal to be focused, and on macOS you must grant Accessibility permission as described in the documentation.
3. Practice the full loop: perform the task, press **→** at completion, reset within 15 seconds, and start the next episode. Then practice one **←** re-record on purpose.
4. Load the rehearsal with `LeRobotDataset` and check that the fps is 30, both camera keys are present, the episode lengths are plausible, and the action and state traces are smooth with no dropouts. Watch one episode's video: the object should be visible in the overhead view throughout, and the grasp should be visible in the wrist view. Reconcile with your predictions from step 1.
5. Delete the rehearsal repository.

**✅ Checkpoint:** the full cycle of record, early stop, reset, and re-record has been exercised; the rehearsal data passes inspection and matches your predicted shapes; you can restage a reset in under 15 seconds.

## Exercise 3 — The recording sessions [Predict → Run]

This is the recording itself, run under seven binding rules. Each rule prevents a specific failure described in the Principles section, and the rules are binding because their violations do not show up until training, when they are expensive to trace back. Before the first session, write down what you expect the audit in Exercise 4 to find, so that the audit tests your plan and not only the data.

Before session 1, write in `RESULTS.md` your planned per-cell counts after 50 episodes, the median and range of episode durations you expect, and how many episodes you expect to discard with **←** per session. Exercise 4's audit reconciles these.

The data-quality protocol is numbered and binding; deviations are logged, not rationalized.

1. Use one grasp strategy for all episodes: the same approach direction, the same grip point, the same place motion. Uniformity is the goal, because variation between episodes is exactly what a policy averages over.
2. Follow the per-cell plan, and log each episode's start cell in a session sheet as you go.
3. Keep episodes under 60 seconds by stopping the moment the place completes; 15–30 seconds is typical.
4. Allow no mid-episode pauses. If you hesitate or stall, press **←** and re-record. A state where you hesitated becomes a state where the policy hesitates.
5. Handle failed demonstrations according to your written policy, mechanically.
6. Keep the rig frozen: no camera nudges, no lamp moves, no chair through the frame. If anything moves, stop, restage from the witness marks, and note it in the failure log.
7. Cap each session at about 30 episodes, because teleoperation quality degrades with fatigue and the degradation shows in the data.

Run Exercise 2's command with `--dataset.repo_id=<you>/so101_pickplace_50ep --dataset.num_episodes=30 --dataset.push_to_hub=true`. For the second session, use the same command with `--resume=true --dataset.num_episodes=20 --dataset.root=<local-path>`; the documentation is explicit that resume counts additional episodes rather than the total. Run `PREFLIGHT.md` first both times.

**✅ Checkpoint (per session, before teardown):** the episode count matches the session sheet; the last episode's video has been spot-checked; the session sheet's per-cell totals match the plan so far; every deviation from rules 1–7 has a line in the failure log.

## Exercise 4 — Audit the dataset [Build]

The audit is the dataset's evidence, and it is where the predictions from Exercise 3 meet the data. You specify a script that checks the episode count, the duration distribution, the per-cell coverage, and the frame timing, and you then confirm by eye and by replay that what was recorded is what you meant to record. Have an AI tool draft `audit.py` from the specification below, using `LeRobotDataset` and numpy.

- Input: a repository id, local or on the Hub. Output: a printed report and `audit.json`.
- Four checks: the episode count is at least 50; the per-episode duration histogram, flagging a median outside 15–35 s and any episode at the 60 s cap; per-cell episode counts against the plan in `TASK.md`, with cells taken from your session sheet (for example a CSV of `episode_index, cell`); and frame-timestamp gaps, flagging any gap greater than twice the frame period as dropped frames.
- One diagnostic: the mean per-step joint delta per episode, since outliers indicate jerky teleoperation worth reviewing.
- The check: the script runs without error against data of the `<you>/h2_rehearsal` form, and its four verdicts match a by-hand count on three episodes.

Then:

1. Run `audit.py` on the final dataset, and reconcile the duration histogram, per-cell counts, and **←** count with the predictions you wrote in Exercise 3 in `RESULTS.md`.
2. Perform a visual audit in the [dataset visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset) on five episodes, checking synchronization between the cameras and the motion and that the grasp is visible in the wrist view every time.
3. Replay episode 0 on the physical arm as an end-to-end integrity test; the arm should reproduce the motion in the real scene:
   ```bash
   lerobot-replay --robot.type=so101_follower --robot.port=<f-port> --robot.id=H1_follower \
     --dataset.repo_id=<you>/so101_pickplace_50ep --dataset.episode=0
   ```

**✅ Checkpoint:** the audit script passes all four checks; the predictions are reconciled; the visualizer renders; the replay reproduces the motion.

## Exercise 5 — Write the card and the failure log [Write]

The last exercise applies the stranger test twice: once to the dataset card, which must let someone else rebuild your rig, and once to the failure log, which must let someone else avoid your mistakes.

1. Write the dataset card: the task sentence, the success criterion, the episode count and duration statistics, the fps, a photo of the camera layout, a diagram of the start-position grid with per-cell counts, the failed-demo policy, the lighting setup, the robot ids, and the LeRobot version. The bar is that a stranger could rebuild your setup from the card alone.
2. Write `FAILURES.md`: everything that went wrong, including dropped frames, USB stalls, camera drift, calibration wobble, and the effects of teleoperation fatigue, with what you did about each. A failure log with nothing in it almost always means that problems went unrecorded rather than that none occurred.

**✅ Checkpoint:** the card is live on the Hub, and `FAILURES.md` has at least five concrete entries.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| Hub dataset `<you>/so101_pickplace_50ep` | ≥ 50 episodes, loads in `LeRobotDataset`, renders in the visualizer |
| `TASK.md` + `PREFLIGHT.md` | task spec frozen before recording, incl. per-rule mechanisms; checklist actually usable in 2 min |
| Session sheets | per-episode start cell + notes, totals matching the dataset |
| `audit.py` + `audit.json` | the four automated checks, run against the published repo |
| Dataset card | passes the stranger test; includes rig photo + grid diagram |
| `FAILURES.md` | ≥ 5 concrete entries |
| `RESULTS.md` | Exercise 2 and 3 predictions reconciled against the audit; rule deviations listed |

## Done when

- [ ] 50 or more episodes are on the Hub, the audit is green, and the visualizer is clean.
- [ ] Start-position coverage matches the written plan cell by cell, and the predicted-versus-actual histogram is in `RESULTS.md`.
- [ ] Episode 0 replays correctly on the arm.
- [ ] The card and the failure log are published.
- [ ] H3's ID/OOD split is already implied by your grid, as demonstrated cells versus held-out cells, in writing.

## Self-check

1. By what mechanism does an inconsistent grasp strategy hurt ACT? Diffusion Policy? (The answers differ.)
2. Why must the start-position distribution you record match the one you will evaluate on, and what specifically breaks when it does not?
3. Why are pauses in demonstrations worse for behaviour cloning than slightly jerky motion?
4. Your two sessions were on different days. List three rig properties that could have silently drifted and say how your protocol catches each.
5. When is keeping failed demonstrations the right call, and what must accompany them?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Arrow keys do nothing during recording | terminal not focused / macOS Accessibility not granted | focus the launching terminal; use `n`/`r`/`q` letter keys; grant permission per docs |
| Frame-gap flags in audit | USB bandwidth or an overloaded encoder | cameras on separate USB controllers; H1's measured-fps test at recording settings |
| Sessions differ visibly in brightness | auto-exposure or daylight | fix UVC exposure/WB; lamp-only lighting; brightness check in preflight |
| Resume re-records from zero / errors | `--resume=true` without `--dataset.root`, or num_episodes set to the total | per docs: root required, num_episodes = *additional* |
| Replay diverges from the recorded scene | calibration drifted (transport, knock) since recording | recalibrate; witness-mark check; recalibration invalidates cross-session mixing — note it in the card |
| Wrist view shows fingers but never the object | mount angle | re-aim before recording 50 episodes, not after |

## Going deeper

Record a ten-episode variation set, using a second object or the held-out grid cells, as a separate repository. H4's generalization probes want exactly this data, and collecting it now, while the rig is warm and staged, costs one short session.

## References

- [LeRobot recording & IL docs](https://huggingface.co/docs/lerobot/il_robots): flags and keyboard controls verified Aug 2026.
- Tutorial §1.3; [what makes a good dataset](https://huggingface.co/blog/lerobot-datasets#what-makes-a-good-dataset).
- [Dataset visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset).
