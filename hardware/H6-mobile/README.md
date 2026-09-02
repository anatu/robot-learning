# H6 — Mobile Manipulation: LeKiwi → XLeRobot (Stretch)

Put your SO-101 on wheels: a holonomic base, a distributed robot (Pi on the robot, brain on your Mac), a mobile fetch-and-carry dataset, and a trained policy whose action space now includes the base. The build-in-public series ends with a robot that crosses the room on command.

| | |
|---|---|
| **Phase** | Hardware track (stretch) |
| **Time** | ~1 session base assembly + Pi setup, ~1 session bring-up + teleop practice, 1–2 recording sessions, ~1 h cloud training + 1 eval session |
| **Cost** | LeKiwi kit $179 + Raspberry Pi 5 ~$80 (+ SD card, battery). XLeRobot (Going deeper): ~$250 incremental |
| **Prerequisites** | H1–H3 (the entire arm workflow, fluent), 04 (kinematics intuition for the holonomic base) |
| **Feeds into** | 22 capstone; the closing demo of the series |

## Learning objectives

After this lesson you can:

1. **Bring up** a distributed robot: motors + cameras on a Pi host, teleop and recording clients on the Mac, ZeroMQ in between.
2. **Derive** a 3-omniwheel holonomic base's wheel-velocity kinematics, explain why it translates and rotates independently, and predict where its dead reckoning drifts.
3. **Collect** mobile-manipulation data with a 9-D action space (6 arm + 3 base) and state what changes vs fixed-base data doctrine.
4. **Train and evaluate** a policy that coordinates base and arm on a fetch-and-carry task.
5. **Diagnose** network-induced data-quality problems (latency, jitter) that fixed-base work never surfaces.

## Principles

**The architecture is the lesson.** LeKiwi runs a *host* process on a Raspberry Pi 5 bolted to the base (motor bus + both cameras plug into the Pi) and a *client* on your Mac (leader arm + keyboard). Observations stream Pi→Mac, actions Mac→Pi over ZeroMQ on WiFi. Teleop latency is therefore a *network* property — every glitch lands in your dataset as a teleop artifact, which is why this lesson's data discipline adds a latency budget to H2's rules. (A wired variant exists — everything runs on the laptop — killing the latency issue and the mobility; the docs cover both.)

**Holonomic base.** Three omniwheels at 120°: wheel speeds map linearly to body-frame $(v_x, v_y, \omega)$ — the base translates in any direction while rotating freely. For wheel $i$ at angle $\theta_i$ from the body x-axis, radius $R$ from center, wheel radius $r$, the wheel's tangential speed is $r\,\dot\phi_i = -\sin\theta_i\, v_x + \cos\theta_i\, v_y + R\,\omega$; stacking three rows gives a $3\times3$ matrix that is invertible whenever the three $\theta_i$ are distinct — that invertibility *is* holonomy. A differential drive has only two independent wheel speeds for three body DoF, so it cannot. The wheel motors are STS3215s on the *same* bus as the arm, IDs 7/8/9, one control board for all nine motors; wheels need no calibration (continuous rotation — no range to find).

**One stack, new embodiment.** `--robot.type=lekiwi` gives a 9-D action space; record/train/eval are the same `LeRobotDataset` → `lerobot-train` → rollout pipeline as H2/H3. That continuity — new robot, zero new learning infrastructure — is the point of the LeRobot abstraction, and this lesson is its proof.

**Task framing.** Fetch-and-carry = navigate to table B → grasp → carry → place at table A. It decomposes into phases with different demands (driving precision vs manipulation precision), which makes it the right first mobile task: failures localize cleanly.

**Carry forward**

- On a distributed robot, network latency is a data-quality variable; budget it and gate episodes on it.
- Holonomy is a rank condition on the wheel-to-body map; dead reckoning drifts because that map integrates slip it cannot see.
- Phase consistency in mobile demos plays the role grasp consistency played in H2.
- A new embodiment costs a config, not a pipeline.

| Source | Read for |
|---|---|
| [LeKiwi docs](https://huggingface.co/docs/lerobot/lekiwi) | assembly links, motor IDs, host/client commands, keyboard map, speed modes — the walkthrough for Exercises 2–4 |
| [SIGRobotics LeKiwi repo](https://github.com/SIGRobotics-UIUC/LeKiwi) | BOM, printed parts, base Assembly.md |
| xlerobot.readthedocs.io | the upgrade path docs (community-maintained — expect to read issues; verify current state before buying parts) |

## Exercise 1 — Omniwheel kinematics [Derive]

Tests objective 2, before any hardware.

1. On paper: write the $3\times3$ wheel-to-body matrix for wheels at $\theta_i \in \{90°, 210°, 330°\}$ (check the docs' mounting diagram and use its convention), invert it, and read off which wheels turn for pure $+v_x$, pure $+v_y$, and pure $+\omega$.
2. Predict the drift mechanism: name two physical effects the matrix cannot represent (wheel slip; roller compliance) and which of $(v_x, v_y, \omega)$ each corrupts most.
3. Put both in `RESULTS.md`; Exercise 4's replay-drift measurement checks the prediction.

**✅ Checkpoint:** the matrix, its inverse, and the three unit-motion wheel patterns are written down; a drift prediction is on record.

## Exercise 2 — Build and Pi setup [Build]

Tests objective 1's host side. ~1 session.

1. Order early (lead time): LeKiwi kit, Pi 5 + SD card, battery per BOM.
2. Pi: flash OS, enable SSH, verify `ssh pi@<ip>` from the Mac; install LeRobot per the docs + `pip install -e ".[lekiwi]"` (Feetech SDK + ZeroMQ). Same install with the extra on the Mac.
3. Assemble the base (SIGRobotics Assembly.md); mount your existing SO-101 follower on it; wire all nine motors to the one control board; mount wrist + front cameras to the Pi.
4. Motor setup — arm IDs 6→1 then wheels 9/8/7, one script run on the Pi:
   ```bash
   lerobot-setup-motors --robot.type=lekiwi --robot.port=<port-on-pi>
   ```
   Wheel mounting positions must match the docs' ID diagram — swapped wheel IDs make the base drive sideways-wrong later (your Exercise 1 inverse tells you exactly how).
5. Calibrate the arm (on the Pi via SSH; wheels skip calibration): `lerobot-calibrate --robot.type=lekiwi --robot.id=H6_kiwi`. Leader stays Mac-side with its H1 calibration.

**✅ Checkpoint:** SSH works headless; all 9 motors enumerate; arm calibration values sane (H1's audit habit); base rolls freely by hand, power off.

## Exercise 3 — Distributed bring-up [Predict → Run]

Tests objectives 1 and 5: the network is now inside your control loop.

1. **Write first:** your predicted median Pi↔Mac ping on your WiFi, and whether you expect spikes > 200 ms in 5 minutes. Predict which teleop channel (arm vs base) will show the first visible stutter and why.
2. Host on the Pi:
   ```bash
   python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=H6_kiwi
   ```
   Client on the Mac: the docs' `examples/lekiwi/teleoperate.py` with `remote_ip` set (expect the `Connected to remote robot at tcp://<ip>:5555 ... video at :5556` line).
3. Controls: leader arm drives the arm; keyboard drives the base — W/A/S/D translate, Z/X rotate, R/F cycle speed modes (slow 0.1 m/s / medium 0.25 / fast 0.4). Note the docs' caveat: base keyboard teleop needs a real key backend (macOS: grant Terminal Input Monitoring; not headless).
4. **Measure the network before trusting it:** ping Pi↔Mac over your WiFi (target: median < 20 ms, no multi-hundred-ms spikes over 5 min); then a 60 s teleop log timing the client loop — spikes here are future dataset artifacts. If ugly: dedicated hotspot/AP, or accept the wired variant. Reconcile against step 1.
5. Practice 20 minutes at slow speed: figure-eights, doorway alignment, drive-then-grasp transitions. Two-input teleop (hand on leader, hand on keys) is a genuine skill — budget for it.
6. Battery discipline: log runtime from full charge; brownout on the Pi corrupts sessions — stop sessions at a voltage/time margin, and never let the Pi die mid-recording.

**✅ Checkpoint:** smooth teleop at medium speed with live video; network log saved and within budget (or the wired decision recorded); you can dock the base at a table and grasp within ~2 min.

## Exercise 4 — Mobile dataset [Write] + [Predict → Run]

Tests objectives 3 and 2 (the drift prediction).

1. Task spec, H2 style, frozen in `TASK.md`: object on table B start zone (taped grid, 3 positions), carry to bin on table A; base start pose taped; success sentence; < 90 s episodes. Floor markers make base starts repeatable — dead reckoning won't.
2. H2's protocol carries over with three mobile amendments: (a) *phase consistency* — same phase order every episode (drive → grasp → drive → place), no mid-drive grasps; (b) *latency gate* — abort/re-record any episode with a visible teleop stutter; (c) *camera framing* — front camera must see the destination table during driving phases (the "do the task from camera images alone" rule now covers navigation).
3. Record 40–50 episodes via the docs' `examples/lekiwi/record.py` flow (set `remote_ip`, `repo_id`, `task`), ~25/session. H2 preflight + battery check each session.
4. Audit with H2's `audit.py` plus one mobile check: base-action channels active in driving phases, near-zero during manipulation (phase discipline, verified from data).
5. **Write first:** predicted replay drift in cm at table B, from your Exercise 1 mechanism. Then visualize 3 episodes and replay one on the robot (`examples/lekiwi/replay.py`) — dead-reckoning drift vs the taped marks is your first quantitative reality-gap datum; measure and log it in cm and reconcile.

**✅ Checkpoint:** 40+ episodes on the Hub, audit + phase check green, replay drift measured and compared to the prediction.

## Exercise 5 — Train, deploy, demo [Predict → Run]

Tests objective 4. ~1 session + cloud hours.

1. Train ACT on the mobile dataset (H3's recipe verbatim, ~$1–3 cloud). SmolVLA fine-tune instead if H4 left you with a working recipe and budget — one policy is enough to close the loop.
2. **Write first:** predicted success out of 10, and which taxonomy class (H3's four + `navigation`) will dominate failures.
3. Deploy via the docs' `examples/lekiwi/evaluate.py` pattern (policy on the Mac, actions streamed to the Pi host). Smoke-test 3 rollouts at slow speed with a clear floor.
4. Eval, pre-registered (H3 discipline, scaled to a stretch lesson): 10 trials, fixed base-start + 3 object positions from the grid; success = object in bin, ≤ 90 s; failure taxonomy = H3's four + `navigation` (base fails to reach either table). Videos on. Reconcile against step 2.
5. **The demo:** one clean uncut take — robot at table A, command issued, fetches from table B, returns, places. That's the closing shot of the build-in-public series.

**✅ Checkpoint:** ≥ 10 pre-registered trials logged with the dominant failure class named; the uncut demo video exists.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| Build + bring-up log | Pi setup, motor map, network measurements, battery runtime, every gotcha |
| Hub dataset `<you>/lekiwi_fetch_carry` | 40+ episodes, audit + phase-consistency check green |
| Trained checkpoint | on Hub, card links dataset + this lesson |
| Eval sheet + videos | 10 pre-registered trials, taxonomy incl. `navigation`, Wilson CI |
| The demo video | one uncut fetch-and-carry |
| `RESULTS.md` | Exercise 1 derivation; predictions vs outcomes for Exercises 3–5; replay-drift number; latency budget vs reality; where mobile data discipline differed from H2's in practice |

## Done when

- [ ] The robot fetches an object from another table on command, on video, uncut.
- [ ] 10-trial pre-registered eval quantifies how reliably.
- [ ] Network and battery budgets are documented numbers, not vibes.
- [ ] Replay drift was predicted from the kinematics before it was measured.
- [ ] A reader of your log could decide in 10 minutes whether LeKiwi is worth it for them.

## Self-check

1. Why can a 3-omniwheel base translate and rotate independently when a differential-drive base can't? Where does its dead reckoning drift come from?
2. Trace one 200 ms WiFi spike during teleop into the dataset: what exactly does the policy later learn from that episode?
3. Why enforce phase consistency (drive → grasp → drive → place) rather than letting natural variation in?
4. Wheels skip calibration but the arm can't. What property of each makes it so?
5. Your trained policy grasps well but parks 10 cm off at table B. Which data-collection choice most likely caused it, and what's the cheapest fix?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Client can't connect | wrong `remote_ip` (Pi got a new DHCP lease) / SSH-only firewall | `hostname -I` on the Pi each session; reserve a static lease |
| Base drives mirrored/rotated vs keys | wheel IDs 7/8/9 mounted in wrong positions | re-check against the docs' diagram; re-ID, don't rewire |
| Teleop stutters only with video on | WiFi bandwidth (two streams + actions) | 5 GHz/dedicated AP, lower camera resolution, or wired variant |
| Pi crashes mid-session | battery sag under motor load browning out the Pi | separate 5 V supply/BEC for the Pi; stop sessions at margin |
| Base keyboard keys dead on Mac | Input Monitoring permission (pynput) | grant Terminal the permission per docs tip |
| Replay ends far from taped marks | normal open-loop dead-reckoning drift | it's data, not a bug — measure it; keep episodes short; policies get vision to compensate |
| Base creeps when keys released | key-release events lost over a laggy session | stop teleop, re-focus the terminal; if chronic, remap keys via `LeKiwiClientConfig` (see `robots/lekiwi/config_lekiwi.py`) and lower speed mode |
| XLeRobot docs disagree with parts in hand | community project moving fast | build from the docs' pinned versions; file issues; budget tinker time |

## Going deeper

**XLeRobot upgrade.** Second SO-101 arm set + IKEA RASKOG cart + battery (~$250 incremental) turns LeKiwi into XLeRobot, a dual-arm mobile household robot with a community-maintained ManiSkill sim and docs at xlerobot.readthedocs.io. This is community-project territory: read the current docs and open issues *before* ordering, expect breakage, and treat your build log as a contribution (file the issues you hit). Scope it as its own mini-project with H-track discipline: bring-up log, one bimanual-mobile task, pre-registered 10-trial eval.

## Version note

The LeKiwi client examples (`teleoperate.py`, `record.py`, `replay.py`, `evaluate.py`) live under `examples/lekiwi/` in the LeRobot repo and are configured by editing the script constants (`remote_ip`, `port`, `repo_id`), not CLI flags — a different convention from the arm-only stack. Key bindings and speed modes are defined in `LeKiwiClientConfig`; verified Aug 2026, recheck against the current lekiwi docs page after any LeRobot upgrade.

## References

- [LeKiwi docs](https://huggingface.co/docs/lerobot/lekiwi) — host/client commands, keyboard map, speed modes verified Aug 2026.
- [SIGRobotics-UIUC/LeKiwi](https://github.com/SIGRobotics-UIUC/LeKiwi) — BOM + assembly.
- xlerobot.readthedocs.io — community docs; verify current state before purchase.
- H2/H3 protocol docs (yours) — the discipline this lesson inherits.
