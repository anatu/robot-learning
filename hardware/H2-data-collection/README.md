# H2 — Real Teleop Data Collection

Record and publish a 50-episode pick-place dataset on your own hardware — the raw material for every policy in H3–H5. Demo quality caps policy quality; this lesson is where that stops being a slogan and becomes a protocol.

| | |
|---|---|
| **Phase** | Hardware track |
| **Time** | 1 session task design + rig lock-down (~2 h), 1–2 recording sessions (~2–3 h for 50+ episodes incl. resets), ~1 h card + writeup |
| **Cost** | $0 |
| **Prerequisites** | H1 (calibrated pair, cameras, teleop fluency), 01–02 (you know exactly what a valid v3 dataset looks like) |
| **Feeds into** | H3 (ACT/DP train on this), H4 (SmolVLA fine-tune + eval task), H5 (demo buffer) |

## Learning objectives

After this lesson you can:

1. **Design** a manipulation task whose reset, start distribution, and success criterion are defined precisely enough to be evaluated 20 trials at a time later.
2. **Run** `lerobot-record` fluently: episode/reset phases, keyboard flow control, resume, and recovery from a crashed session.
3. **Enforce** a written data-quality protocol — and articulate the *mechanism* by which each rule affects the trained policy.
4. **Audit** your own dataset: episode stats, sync, coverage of the start distribution, and a failure log others can learn from.
5. **Publish** a Hub dataset with a card a stranger could reproduce the setup from.

## Background

**Why the recording session decides everything downstream.** ACT and Diffusion Policy imitate; they don't filter or repair. Every hesitation becomes a policy hesitation (Lesson 14: pause states are attractors). Two grasp strategies across episodes hand a multimodal average to anything that can't model modes — and even generative heads pay for it in sample efficiency. Start positions you never demonstrated are OOD at eval time, and H3's ID/OOD protocol will charge you for them. The docs' rule of thumb is the right bar: **you should be able to do the task yourself looking only at the camera images.** If the object leaves both frames, so does the policy's ability to find it.

**How recording actually flows.** `lerobot-record` alternates a *record phase* (`--dataset.episode_time_s`, default 60) with a *reset phase* (`--dataset.reset_time_s`, default 60) during which you physically restage the scene — the reset phase is idle time by design, not part of the episode. Keyboard flow control: **→ / n** end episode early and move on (use it the second the place completes — don't pad episodes), **← / r** discard and re-record the current episode, **ESC / q** end session, encode videos, push. Data lands in `~/.cache/huggingface/lerobot/<repo-id>` and pushes to the Hub by default.

**Failed demos: pick a policy and write it down.** Two defensible options: (a) *curated* — re-record failures via `←`, dataset is successes only (right choice for H3's BC training); (b) *labeled* — keep failures, note episode indices in the card (extra positives/negatives for H5's reward classifier). Choose (a) unless you're already committed to H5; either way the card states the policy. What's not defensible is unlabeled failures mixed in silently.

| Source | Read for |
|---|---|
| Tutorial §1.3 | the recording recipe this lesson instantiates (its Code 2 is v0.4-era — the commands below are the current form) |
| [Recording docs](https://huggingface.co/docs/lerobot/il_robots) | flag reference, resume semantics, keyboard-backend caveats |
| [HF "what makes a good dataset" post](https://huggingface.co/blog/lerobot-datasets#what-makes-a-good-dataset) | the community's accumulated data-quality folklore, cross-checked against your protocol |

## Part 1 — Task and rig design (~2 h, no recording)

Decisions made here are frozen for H3–H5; changing the task after training means re-collecting.

1. **Task:** single pick-place — one graspable object (rigid, ≥ 3 cm, matte; a wooden cube is the classic) from a start zone into a fixed container. One behavior, one object, no distractors. Variation comes later (H4 probes it); it doesn't belong in the base dataset.
2. **Start distribution — define it physically.** Tape a grid into the start zone (e.g. 3×3 cells over ~15×15 cm). Plan ~50 episodes ≈ 10 per position over ~5 grid cells (matching the docs' 10-per-location guidance). Log the cell for every episode as you record. H3's ID evaluation draws from these cells; OOD is the cells you deliberately held out — decide now which those are.
3. **Success criterion, written:** e.g. "object fully inside container, arm returned to home, within 60 s." This sentence gets reused verbatim in H3's trial sheets.
4. **Lock the rig:** overhead camera on rigid mount framing start zone + container; wrist camera showing the fingers; desk lamp for constant lighting (kill the window — daylight drift between sessions is a distribution shift you can't undo); tape witness marks around the robot base, container, and camera positions so the rig can be re-staged exactly.
5. **Kill camera auto-adjustment.** Auto-exposure and auto-white-balance change image statistics mid-session. Fix them via UVC controls where the model allows, and verify by comparing frame brightness at session start vs end.
6. Write `PREFLIGHT.md` — run before *every* session, this lesson and later: power on → `lerobot-find-cameras opencv` and re-verify index↔camera map → witness marks aligned → lamp on, blinds closed → 30 s teleop warm-up → one throwaway episode, inspect both views.

**✅ Checkpoint:** task spec (object, grid, per-cell episode counts, success sentence, failed-demo policy) written in `TASK.md` before any recording; a phone photo of the rig with witness marks visible.

## Part 2 — Dress rehearsal (~30 min)

1. Record a 3-episode throwaway dataset with the real command:
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
   (Keyboard flow keys need the terminal focused; on macOS grant Accessibility permission per the docs.)
2. Practice the full loop: task → `→` at completion → reset within 15 s → next episode. Then practice one `←` re-record on purpose.
3. Load the rehearsal with `LeRobotDataset`, assert: fps 30, both camera keys present, episode lengths plausible, action/state traces smooth (no dropouts). Watch one episode's video: object visible in the overhead view *throughout*, grasp visible in the wrist view.
4. Delete the rehearsal repo.

**✅ Checkpoint:** full record → early-stop → reset → re-record cycle exercised; rehearsal data passes inspection; you can restage a reset in < 15 s.

## Part 3 — The recording sessions (2–3 h total)

The data-quality protocol, numbered and binding — deviations get logged, not rationalized:

1. One grasp strategy, all episodes: same approach direction, same grip point, same place motion. Boring is the goal.
2. Follow the per-cell plan; log each episode's start cell in a session sheet as you go.
3. Episodes < 60 s (early-stop the moment the place completes; target 15–30 s typical).
4. No mid-episode pauses. If you hesitate or stall: `←`, re-record. Hesitation states become policy attractors.
5. Failed demos handled per your written policy, mechanically.
6. Rig frozen: no camera nudges, no lamp moves, no chair-through-frame. If anything moves, stop, re-stage from witness marks, note it in the failure log.
7. Session cap ~30 episodes — teleop quality degrades with fatigue, and it shows in the data.

Run Part 2's command with `--dataset.repo_id=<you>/so101_pickplace_50ep --dataset.num_episodes=30 --dataset.push_to_hub=true`. Second session: same command with `--resume=true --dataset.num_episodes=20 --dataset.root=<local-path>` (resume counts *additional* episodes — docs are explicit). Run `PREFLIGHT.md` first both times.

**✅ Checkpoint (per session, before teardown):** episode count matches the sheet; last episode's video spot-checked; session sheet totals per start cell match the plan so far.

## Part 4 — Audit, card, publish (~1 h)

1. `audit.py` against the final dataset: episode count ≥ 50; per-episode duration histogram (median 15–35 s, none at the 60 s cap); per-cell episode counts vs plan; frame-timestamp gaps (flag > 2× frame period — dropped frames); mean per-step joint delta per episode (outliers = jerky teleop worth watching).
2. Visual audit: [dataset visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset) on 5 episodes — sync between cameras and motion, grasp visible in wrist view every time.
3. Replay episode 0 on the physical arm as an end-to-end integrity test (arm reproduces the motion in the real scene):
   ```bash
   lerobot-replay --robot.type=so101_follower --robot.port=<f-port> --robot.id=H1_follower \
     --dataset.repo_id=<you>/so101_pickplace_50ep --dataset.episode=0
   ```
4. Dataset card: task sentence, success criterion, episode count + duration stats, fps, camera layout **photo**, start-position grid diagram with per-cell counts, failed-demo policy, lighting setup, robot ids, LeRobot version. Bar: a stranger rebuilds your setup from the card alone.
5. `FAILURES.md`: everything that went wrong — dropped frames, USB stalls, camera drift, calibration wobble, teleop fatigue effects — with what you did about each.

**✅ Checkpoint:** audit script passes; visualizer renders; replay reproduces the motion; card live on the Hub.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| Hub dataset `<you>/so101_pickplace_50ep` | ≥ 50 episodes, loads in `LeRobotDataset`, renders in the visualizer |
| `TASK.md` + `PREFLIGHT.md` | task spec frozen before recording; checklist actually usable in 2 min |
| Session sheets | per-episode start cell + notes, totals matching the dataset |
| `audit.py` + output | the four automated checks, run against the published repo |
| Dataset card | passes the stranger test; includes rig photo + grid diagram |
| `FAILURES.md` | ≥ 5 concrete entries (a clean session is evidence of an unobservant log) |

## Done when

- [ ] 50+ episodes on the Hub, audit green, visualizer clean.
- [ ] Start-position coverage matches the written plan cell-by-cell.
- [ ] Episode 0 replays correctly on the arm.
- [ ] Card + failure log published.
- [ ] H3's ID/OOD split is already implied by your grid: demonstrated cells vs held-out cells, in writing.

## Self-check

1. By what *mechanism* does inconsistent grasp strategy hurt ACT? Diffusion Policy? (Different answers.)
2. Why must the start-position distribution you record match the one you'll evaluate on — and what specifically breaks when it doesn't?
3. Why are pauses in demos worse for BC than slightly jerky motion?
4. Your two sessions were on different days. List three rig properties that could have silently drifted and how your protocol catches each.
5. When is keeping failed demos the right call, and what must accompany them?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Arrow keys do nothing during recording | terminal not focused / macOS Accessibility not granted | focus the launching terminal; use `n`/`r`/`q` letter keys; grant permission per docs |
| Frame-gap flags in audit | USB bandwidth or an overloaded encoder | cameras on separate USB controllers; H1's measured-fps test at recording settings |
| Sessions differ visibly in brightness | auto-exposure or daylight | fix UVC exposure/WB; lamp-only lighting; brightness check in preflight |
| Resume re-records from zero / errors | `--resume=true` without `--dataset.root`, or num_episodes set to the total | per docs: root required, num_episodes = *additional* |
| Replay diverges from the recorded scene | calibration drifted (transport, knock) since recording | recalibrate; witness-mark check; recalibration invalidates cross-session mixing — note it in the card |
| Wrist view shows fingers but never the object | mount angle | re-aim before recording 50 episodes, not after |

## Stretch

Record a 10-episode *variation* set (second object, or the held-out grid cells) as a separate repo — H4's generalization probes want it, and collecting it now costs one warm session.

## References

- [LeRobot recording & IL docs](https://huggingface.co/docs/lerobot/il_robots) — flags and keyboard controls verified Aug 2026.
- Tutorial §1.3; [what makes a good dataset](https://huggingface.co/blog/lerobot-datasets#what-makes-a-good-dataset).
- [Dataset visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset).
