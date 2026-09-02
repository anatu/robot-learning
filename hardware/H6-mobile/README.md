# H6 — Mobile Manipulation: LeKiwi → XLeRobot (Stretch)

This lesson puts the SO-101 on a wheeled base and turns it into a distributed robot: the motors and cameras hang off a Raspberry Pi on the base, the teleoperation and training clients run on your Mac, and the two communicate over a wireless link. You will derive the kinematics of the holonomic base, bring the distributed system up, record a fetch-and-carry dataset whose action space now includes the base, and train and evaluate a policy that coordinates driving and manipulation. The lesson ends with a robot that crosses the room and retrieves an object on command, and along the way it surfaces a class of data-quality problem, network latency, that fixed-base work never encounters.

| | |
|---|---|
| **Phase** | Hardware track (stretch) |
| **Time** | ~1 session base assembly + Pi setup, ~1 session bring-up + teleop practice, 1–2 recording sessions, ~1 h cloud training + 1 eval session |
| **Cost** | LeKiwi kit $179 + Raspberry Pi 5 ~$80 (+ SD card, battery). XLeRobot (Going deeper): ~$250 incremental |
| **Prerequisites** | H1–H3 (the entire arm workflow, fluent), 04 (kinematics intuition for the holonomic base) |
| **Feeds into** | 22 capstone; the closing demo of the series |

## Learning objectives

After this lesson you can:

1. **Bring up** a distributed robot, with motors and cameras on a Pi host, teleoperation and recording clients on the Mac, and ZeroMQ carrying observations and actions between them.
2. **Derive** the wheel-velocity kinematics of a three-omniwheel holonomic base, explain why it can translate and rotate independently, and predict where its dead reckoning will drift.
3. **Collect** mobile-manipulation data with a nine-dimensional action space (six arm, three base), and state what changes relative to the fixed-base data doctrine of H2.
4. **Train and evaluate** a policy that coordinates the base and the arm on a fetch-and-carry task.
5. **Diagnose** network-induced data-quality problems, latency and jitter, that fixed-base work never surfaces.

## Principles

### A distributed robot

LeKiwi runs as two processes on two machines. A host process on a Raspberry Pi 5 bolted to the base owns the motor bus and both cameras. A client process on your Mac owns the leader arm and the keyboard. Observations stream from the Pi to the Mac and actions stream back, over ZeroMQ on WiFi. Teleoperation latency is therefore a property of the network rather than of the robot, and every glitch in that network lands in your dataset as a teleoperation artifact: a frame where the operator's command arrived late looks, to a learning algorithm, exactly like a frame where the operator hesitated. This is why the data discipline in this lesson adds a latency budget to H2's rules. A wired variant exists, in which everything runs on the laptop; it removes the latency problem and also the mobility, and the documentation covers both.

### The holonomic base

The base has three omniwheels mounted at 120° intervals. Each omniwheel drives along its own axis while rolling freely sideways on its rollers, so the three wheel speeds together determine the body-frame velocity $(v_x, v_y, \omega)$ linearly. For wheel $i$ mounted at angle $\theta_i$ from the body $x$-axis, at distance $R$ from the centre, with wheel radius $r$, the wheel's tangential speed is

$$r\,\dot\phi_i = -\sin\theta_i\, v_x + \cos\theta_i\, v_y + R\,\omega .$$

Stacking the three rows gives a $3\times3$ matrix from body velocity to wheel speeds. That matrix is invertible whenever the three mounting angles are distinct, and its invertibility is what holonomy means: any body velocity, including pure sideways translation and pure rotation, can be produced by some combination of wheel speeds. A differential-drive base has only two independent wheel speeds for three body degrees of freedom, so its matrix cannot be inverted and it cannot translate sideways without first turning.

Dead reckoning integrates the wheel speeds through this matrix to estimate position. The estimate drifts because the matrix describes ideal rolling; wheel slip and roller compliance both move the base by amounts the matrix never sees, and the errors accumulate. The wheel motors are STS3215 servos on the same bus as the arm, with IDs 7, 8 and 9, and one control board serves all nine motors. The wheels need no calibration, because they rotate continuously and have no range to find.

### One software stack, a new embodiment

Setting `--robot.type=lekiwi` gives a nine-dimensional action space, and recording, training and evaluation go through the same `LeRobotDataset`, `lerobot-train`, and rollout pipeline as H2 and H3. Nothing in the learning infrastructure changes when the embodiment does. That continuity is what the LeRobot abstraction promises, and this lesson tests whether the promise holds.

### The task

Fetch-and-carry means navigating to table B, grasping an object, carrying it, and placing it at table A. The task decomposes into phases with different demands, driving precision in some and manipulation precision in others, which makes it the right first mobile task: when a trial fails, the failure can be attributed to a phase.

**Carry forward**

- On a distributed robot, network latency is a data-quality variable, because a late command is indistinguishable from an operator's hesitation in the recorded data; budget the latency and gate episodes on it.
- Holonomy is a rank condition on the wheel-to-body velocity map, and dead reckoning drifts because that map integrates ideal rolling while the real base slips.
- Phase consistency in mobile demonstrations plays the role that grasp consistency played in H2: a policy trained on demonstrations with varying phase order has to learn a decision the demonstrator never made deliberately.
- A new embodiment costs a configuration, not a new pipeline, and that is the property of the software stack worth verifying.

| Source | Read for |
|---|---|
| [LeKiwi docs](https://huggingface.co/docs/lerobot/lekiwi) | assembly links, motor IDs, host/client commands, keyboard map, speed modes — the walkthrough for Exercises 2–4 |
| [SIGRobotics LeKiwi repo](https://github.com/SIGRobotics-UIUC/LeKiwi) | BOM, printed parts, base Assembly.md |
| xlerobot.readthedocs.io | the upgrade path docs (community-maintained — expect to read issues; verify current state before buying parts) |

## Exercise 1 — Derive the omniwheel kinematics [Derive]

Before any hardware arrives, you work out the base's kinematics on paper and use them to predict how the base will misbehave. This exercise tests objective 2, and it produces two things that later exercises check: the inverse matrix that tells you which wheels turn for each unit body motion, and a prediction about dead-reckoning drift.

1. Write the $3\times3$ wheel-to-body matrix for wheels at $\theta_i \in \{90°, 210°, 330°\}$ (check the documentation's mounting diagram and use its convention), invert it, and read off which wheels turn for pure $+v_x$, pure $+v_y$, and pure $+\omega$.
2. Predict the drift mechanism. Name two physical effects the matrix cannot represent (wheel slip and roller compliance are the two to consider) and say which of $(v_x, v_y, \omega)$ each corrupts most.
3. Record both in `RESULTS.md`. The replay-drift measurement in Exercise 4 checks the prediction.

**✅ Checkpoint:** the matrix, its inverse, and the three unit-motion wheel patterns are written down, and a drift prediction is on record.

## Exercise 2 — Assemble the base and set up the Pi [Build]

In this exercise you build the base, mount your existing follower arm on it, and bring up the Pi as the robot's host. This is the host side of objective 1. Budget one session.

1. Order early, because of lead time: the LeKiwi kit, a Pi 5 with an SD card, and a battery per the bill of materials.
2. Set up the Pi: flash the OS, enable SSH, and verify `ssh pi@<ip>` from the Mac. Install LeRobot per the documentation with `pip install -e ".[lekiwi]"`, which brings in the Feetech SDK and ZeroMQ. Do the same install with the same extra on the Mac.
3. Assemble the base following the SIGRobotics Assembly.md, mount your existing SO-101 follower on it, wire all nine motors to the single control board, and mount the wrist and front cameras to the Pi.
4. Set the motor IDs, arm 6→1 and then wheels 9/8/7, in one run of the setup script on the Pi:
   ```bash
   lerobot-setup-motors --robot.type=lekiwi --robot.port=<port-on-pi>
   ```
   The wheel mounting positions must match the documentation's ID diagram. If two wheel IDs are swapped, the base will drive in the wrong direction for a given key, and your Exercise 1 inverse tells you exactly which direction.
5. Calibrate the arm on the Pi over SSH; the wheels skip calibration: `lerobot-calibrate --robot.type=lekiwi --robot.id=H6_kiwi`. The leader arm stays on the Mac with its H1 calibration.

**✅ Checkpoint:** SSH works headless; all nine motors enumerate; the arm calibration values are sane by H1's audit standard; the base rolls freely by hand with the power off.

## Exercise 3 — Bring up the distributed system [Predict → Run]

Here you run the host and client processes, teleoperate the base and arm together, and measure the network that now sits inside your control loop. The exercise tests objectives 1 and 5. The prediction concerns the network, because it is the component you have not worked with before and the one most likely to surprise you.

1. Before starting, write down your predicted median round-trip ping between the Pi and the Mac on your WiFi, whether you expect any spikes above 200 ms within five minutes, and which teleoperation channel, arm or base, you expect to show the first visible stutter and why.
2. Start the host on the Pi:
   ```bash
   python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=H6_kiwi
   ```
   Start the client on the Mac using the documentation's `examples/lekiwi/teleoperate.py` with `remote_ip` set. Expect the line `Connected to remote robot at tcp://<ip>:5555 ... video at :5556`.
3. The controls are as follows. The leader arm drives the arm. The keyboard drives the base: W/A/S/D translate, Z/X rotate, and R/F cycle the speed modes (slow 0.1 m/s, medium 0.25 m/s, fast 0.4 m/s). Note the documentation's caveat that base keyboard teleoperation needs a real key backend; on macOS, grant Terminal the Input Monitoring permission, and do not expect it to work headless.
4. Measure the network before trusting it. Ping between the Pi and the Mac over your WiFi, with a target of a median below 20 ms and no spikes of several hundred milliseconds over five minutes. Then record a 60-second teleoperation log that times the client loop; spikes in that log are future dataset artifacts. If the numbers are bad, use a dedicated hotspot or access point, or accept the wired variant. Reconcile the measurements against your prediction from step 1.
5. Practise for twenty minutes at slow speed: figure-eights, doorway alignment, and drive-then-grasp transitions. Two-input teleoperation, with one hand on the leader and one on the keys, is a genuine skill and needs time budgeted for it.
6. Establish battery discipline. Log the runtime from a full charge. A brownout on the Pi corrupts a session, so stop sessions at a voltage or time margin and never let the Pi die during a recording.

**✅ Checkpoint:** teleoperation is smooth at medium speed with live video; the network log is saved and within budget, or the decision to go wired is recorded; you can dock the base at a table and grasp within about two minutes.

## Exercise 4 — Record the mobile dataset [Write] + [Predict → Run]

In this exercise you write the task specification, record the dataset under H2's protocol with three mobile amendments, and measure dead-reckoning drift by replaying an episode. The exercise tests objective 3 and checks the drift prediction from Exercise 1.

1. Write the task specification in `TASK.md` in the H2 style: the object starts in a taped grid on table B with three positions, is carried to a bin on table A, the base start pose is taped, the success sentence is written, and episodes are under 90 seconds. Floor markers make base starts repeatable, because dead reckoning will not.
2. Carry over H2's protocol with three mobile amendments. First, phase consistency: the same phase order in every episode (drive, grasp, drive, place), with no grasps during a drive. Second, a latency gate: abort and re-record any episode with a visible teleoperation stutter. Third, camera framing: the front camera must see the destination table during driving phases, because H2's rule that the task must be doable from the camera images alone now covers navigation.
3. Record 40–50 episodes through the documentation's `examples/lekiwi/record.py` flow, setting `remote_ip`, `repo_id`, and `task`, at about 25 per session. Run the H2 preflight and a battery check before each session.
4. Audit with H2's `audit.py` plus one mobile check: the base-action channels should be active during driving phases and near zero during manipulation, which verifies phase discipline from the data.
5. Before replaying, write down your predicted replay drift in centimetres at table B, derived from the mechanism you named in Exercise 1. Then visualise three episodes and replay one on the robot with `examples/lekiwi/replay.py`. The distance between where the base ends up and the taped marks is your first quantitative measurement of the reality gap; measure it in centimetres, log it, and reconcile it against the prediction.

**✅ Checkpoint:** 40 or more episodes are on the Hub; the audit and phase check pass; the replay drift is measured and compared to the prediction.

## Exercise 5 — Train, deploy, and record the demonstration [Predict → Run]

Finally you train a policy on the mobile dataset, deploy it through the host, evaluate it under a pre-registered protocol, and record an uncut demonstration. The exercise tests objective 4. Budget one session plus the cloud training time.

1. Train ACT on the mobile dataset using H3's recipe unchanged, at about $1–3 of cloud time. If H4 left you with a working SmolVLA fine-tuning recipe and budget, fine-tune that instead; one policy is enough to close the loop.
2. Before evaluating, write down your predicted success out of ten and which taxonomy class (H3's four plus `navigation`) you expect to dominate the failures.
3. Deploy using the documentation's `examples/lekiwi/evaluate.py` pattern, with the policy on the Mac and actions streamed to the Pi host. Smoke-test three rollouts at slow speed with a clear floor.
4. Evaluate under a pre-registered protocol, scaled from H3 to suit a stretch lesson: ten trials, a fixed base start and three object positions from the grid, success defined as the object in the bin within 90 seconds, and a failure taxonomy of H3's four classes plus `navigation` (the base fails to reach either table). Record every trial on video, and reconcile against the prediction from step 2.
5. Record the demonstration as one clean, uncut take: the robot starts at table A, receives the command, fetches from table B, returns, and places.

**✅ Checkpoint:** ten or more pre-registered trials are logged with the dominant failure class named, and the uncut demonstration video exists.

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
- [ ] A ten-trial pre-registered evaluation quantifies how reliably it does so.
- [ ] The network and battery budgets are documented as measured numbers.
- [ ] Replay drift was predicted from the kinematics before it was measured.
- [ ] A reader of your log could decide in ten minutes whether LeKiwi is worth building.

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
| Replay ends far from taped marks | normal open-loop dead-reckoning drift | expected; measure it, keep episodes short, and rely on the policy's vision to compensate |
| Base creeps when keys released | key-release events lost over a laggy session | stop teleop, re-focus the terminal; if chronic, remap keys via `LeKiwiClientConfig` (see `robots/lekiwi/config_lekiwi.py`) and lower speed mode |
| XLeRobot docs disagree with parts in hand | community project moving fast | build from the docs' pinned versions; file issues; budget tinker time |

## Going deeper

**The XLeRobot upgrade.** A second SO-101 arm set, an IKEA RASKOG cart, and a battery (about $250 incremental) turn LeKiwi into XLeRobot, a dual-arm mobile household robot with a community-maintained ManiSkill simulation and documentation at xlerobot.readthedocs.io. This is community-project territory: read the current documentation and the open issues before ordering, expect breakage, and treat your build log as a contribution by filing the issues you hit. Scope it as its own mini-project with the hardware track's discipline: a bring-up log, one bimanual mobile task, and a pre-registered ten-trial evaluation.

## Version note

The LeKiwi client examples (`teleoperate.py`, `record.py`, `replay.py`, `evaluate.py`) live under `examples/lekiwi/` in the LeRobot repo and are configured by editing the script constants (`remote_ip`, `port`, `repo_id`), not CLI flags — a different convention from the arm-only stack. Key bindings and speed modes are defined in `LeKiwiClientConfig`; verified Aug 2026, recheck against the current lekiwi docs page after any LeRobot upgrade.

## References

- [LeKiwi docs](https://huggingface.co/docs/lerobot/lekiwi) — host/client commands, keyboard map, speed modes verified Aug 2026.
- [SIGRobotics-UIUC/LeKiwi](https://github.com/SIGRobotics-UIUC/LeKiwi) — BOM + assembly.
- xlerobot.readthedocs.io — community docs; verify current state before purchase.
- H2/H3 protocol docs (yours): the discipline this lesson inherits.
