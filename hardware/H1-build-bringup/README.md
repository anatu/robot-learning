# H1 — Bring-Up the SO-101

This lesson takes the assembled SO-101 leader/follower pair from the box to a working instrument. You will verify the build against the gear-ratio table, configure and calibrate the servo bus, get two cameras streaming at a measured frame rate, confirm that teleoperation works, and then drive the follower with the controller you wrote in Lesson 04 while measuring how far the real arm departs from the commanded path. Everything in the hardware track after this point assumes the rig you produce here, so the log you keep is as important as the arm.

| | |
|---|---|
| **Phase** | Hardware track (start any time after Lesson 02) |
| **Time** | ~1 session: 1 h unbox + inspect, 2–3 h motor setup / calibration / cameras / teleop, 1–2 h controller trace |
| **Cost** | $0 beyond the kit already ordered in Lesson 00 (assembled Partabot kit — see `hardware/ORDER.md`) |
| **Prerequisites** | 00 (kit ordered, LeRobot env), 03 (your `fk(q)`, which scores the trace), 04 (your controller behind `qdot = controller.step(q_meas, t)`) |
| **Feeds into** | H2 (recording rig), H3–H5 (the physical platform), every teleop session after this |

## Learning objectives

After this lesson you can:

1. **Verify** an assembled pair against the gear-ratio table and explain why the leader is geared differently from the follower.
2. **Explain** how a Feetech servo bus is configured: ports, motor IDs, baudrate, and which settings live in EEPROM as opposed to RAM.
3. **Diagnose** a bad calibration from the numbers in the calibration file alone, before the arm moves.
4. **Operate** the rig with the safety doctrine below as routine: torque discipline, workspace clearance, and a power-cut e-stop within reach.
5. **Quantify** the reality gap at its lowest level, the difference between commanded and measured joint angles and the resulting end-effector error in millimetres computed through your own FK, and name its dominant source from the plots.

## Principles

### Why the leader and follower are geared differently

The follower's six STS3215 motors are all geared 1/345. That ratio gives high torque, so the arm holds its pose against gravity and tracks commanded positions stiffly, which is what a follower needs. The leader must be moved by a human hand, so it uses three ratios chosen for back-drivability: 1/191 on the base and elbow, 1/345 on the shoulder lift, and 1/147 on wrist flex, wrist roll and gripper. The shoulder lift keeps the high ratio because it has to support the weight of the arm above it; without that torque the leader would slump when released. The wrist joints use the lowest ratio because they carry almost no load and should move freely under the hand. The consequences of a mis-sorted motor follow directly: a 1/147 motor in a shoulder lets the arm slump, and a 1/345 motor in a wrist makes teleoperation feel like stirring cement. Your kit arrived assembled, but the ratios are still yours to verify, because a mis-sorted motor shows up as slump or stiffness rather than as an error message.

### The servo bus

All six motors are daisy-chained on a single three-pin serial bus to the control board. For the board to address them individually, each motor needs a unique ID (1 for the shoulder pan through 6 for the gripper), and all motors must share a baudrate. Both values are written once into each motor's EEPROM by `lerobot-setup-motors`, where they persist across power cycles. New motors all ship with ID 1, which is why the setup script asks you to connect motors one at a time: if two motors with the same ID were on the bus together, both would answer the same command and the write would be ambiguous. The `--robot.id` and `--teleop.id` values you choose name the calibration file for each arm. Use the same ids for the rest of the course ("H1_follower", "H1_leader"), because every later command looks the calibration up by that name.

### What calibration is

A calibration is a mapping from each motor's raw magnetic-encoder ticks to joint degrees with a shared zero. It is recorded by holding the arm at a mid-range pose to define the zero and then sweeping each joint through its full range so that the software learns the tick values at each limit. Without a calibration, a policy trained on one arm means nothing on another, and the leader cannot drive the follower, because neither arm's encoder counts correspond to the other's. The calibration file is the whole calibration, which means a lazy sweep is visible in the file before it is visible in motion: a joint whose recorded minimum and maximum are nearly equal, or whose range spans only a few ticks, was not swept to its limits, and every downstream lesson inherits the clipped workspace.

### Where the reality gap begins

Commanded and measured joint angles differ, and they differ for three separable reasons, each of which leaves a distinct signature in a trace plot. Tracking lag is the delay between a command and the motor reaching it, so its error is proportional to velocity and is largest where the reference moves fastest. A calibration offset is a fixed displacement of the whole trace, independent of speed and configuration. Gravity sag depends on how far the arm is extended, so its error follows configuration rather than velocity. Learning to read these signatures in Exercise 7 is the skill that H3 through H5 rely on when a trained policy nearly works and you need to know why it does not.

### Safety doctrine

The following rules apply to every hardware lesson. (1) The power supply's switch or plug is the e-stop; it stays within arm's reach whenever torque is on. (2) Never leave an arm powered and unattended. (3) Clear a radius of about 40 cm around the follower before enabling torque; the first bug you write will command a full-speed swing. (4) STS3215 servos cut out on over-temperature or over-load. If a servo goes limp and hot after stalling against a limit, power off and let it cool; repeated stalls kill servos.

**Carry forward**

- The leader uses three gear ratios (1/191 on base and elbow, 1/345 on shoulder lift, 1/147 on wrist flex, wrist roll and gripper) and the follower uses 1/345 throughout, because the leader must be back-drivable by hand while the follower must hold its pose.
- Motor IDs and baudrate are written once into each motor's EEPROM, while the calibration lives in a file named by `--robot.id`; that id must never change, because every later command looks the calibration up by it.
- A calibration is only as good as the sweep that produced it, and a bad sweep is visible in the file as a narrow or inverted joint range before it is visible in motion.
- The three sources of tracking error have distinct signatures: lag scales with velocity, a calibration offset is constant, and gravity sag depends on configuration.
- The e-stop stays within reach whenever torque is on, without exception.

| Source | Read for |
|---|---|
| [SO-101 docs](https://huggingface.co/docs/lerobot/so101) | the gear-ratio table, per-joint assembly videos (for inspection and any repair), setup/calibration commands — keep it open on a second screen throughout |
| TheRobotStudio [SO-ARM100 repo](https://github.com/TheRobotStudio/SO-ARM100) | printed-parts fit notes, camera-mount STLs used in Exercise 4 |
| [LeRobot camera docs](https://huggingface.co/docs/lerobot/cameras) | `lerobot-find-cameras`, camera config JSON, why indices drift |

## Exercise 1 — Unbox and inspect the arms [Read]

In this exercise you check the assembled arms against the gear-ratio table and confirm that every joint moves freely. It applies the first principle above: the ratios determine how each arm behaves, and a mis-sorted motor shows up as a mechanical symptom rather than a software error, so this is the only point at which catching one is cheap.

1. Unbox both arms. Check every motor's ratio by its label against the table: leader 2× 1/191 (base, elbow), 1× 1/345 (shoulder lift), 3× 1/147 (wrist flex, wrist roll, gripper); follower 6× 1/345. A label you cannot read gets the back-drive test in step 2.
2. With power off, back-drive every joint by hand through its full range. There should be no grinding and no binding. The leader's shoulder lift should hold position when released, which confirms the 1/345 motor is in the right place; the leader's wrist and gripper should move freely, which confirms the 1/147 motors.
3. Trace the daisy-chain cable through the elbow and confirm that every three-pin connector is reachable without disassembly, because Exercise 2 may need motors isolated one at a time.
4. Note the fastener facts for any future repair: there are two screw sizes only, M2×6 into motor bodies and M3×6 into horns and structure; top horns are secured with the M3 horn screw, and bottom horns are unscrewed by design.
5. Start `BRINGUP.md` now, recording the ratio check per motor, any deviation from the documented assembly, and anything loose.

**✅ Checkpoint:** every motor's ratio is verified and logged; every joint is back-drivable through its full range with power off; the leader's shoulder lift holds when released.

## Exercise 2 — Ports, motor IDs and baudrate [Predict → Run]

Here you find each control board's serial port and, if the vendor has not already done so, program the motor IDs and baudrate. The exercise tests whether you can say what the setup script writes and where, which is the second principle above. The prediction you write first is checked against what the script prints.

1. Before running anything, write in `BRINGUP.md` what `lerobot-setup-motors` writes, into which memory on the motor, and what you expect to happen if two factory-fresh motors share the bus when you press Enter.
2. Find each control board's port with `lerobot-find-port`, which asks you to unplug the board and identifies it by the difference in the port list. Record both port paths. On Linux, run `sudo chmod 666 /dev/ttyACM*`.
3. An assembled kit may ship with IDs already programmed, so check the vendor's notes. If it did, go straight to Exercise 3 and come back here only if calibration cannot see all six motors. Otherwise, set up the follower one motor at a time, starting with the gripper, exactly as the script instructs:
   ```bash
   lerobot-setup-motors --robot.type=so101_follower --robot.port=<follower-port>
   ```
   The script walks from motor 6 down to motor 1, and each motor must be the only one on the bus when you press Enter. Expect output such as `'gripper' motor id set to 6`.
4. Set up the leader the same way, noting the different namespace:
   ```bash
   lerobot-setup-motors --teleop.type=so101_leader --teleop.port=<leader-port>
   ```
5. Reconnect the full daisy-chain on both arms, and reconcile the prediction from step 1 with what the script printed.

**✅ Checkpoint:** both arms enumerate all six motors, and `BRINGUP.md` records the port-to-arm mapping, whether the IDs were vendor-set, and any problem encountered.

## Exercise 3 — Calibrate, then plant a lazy sweep [Diagnose]

You calibrate both arms, then deliberately recalibrate the leader with one joint swept through only part of its range. The purpose is to see a bad calibration in the file before you see it in motion, so that in later lessons you check the numbers rather than trusting how the arm feels.

1. Calibrate the follower:
   ```bash
   lerobot-calibrate --robot.type=so101_follower --robot.port=<port> --robot.id=H1_follower
   ```
   Hold the mid-range pose, press Enter, then sweep every joint through its full range. A lazy sweep clips the usable workspace.
2. Calibrate the leader the same way, with `--teleop.type=so101_leader --teleop.id=H1_leader`.
3. Open the written calibration file (under `~/.cache/huggingface/lerobot/calibration/`) and copy the per-joint minimum, center and maximum into `BRINGUP.md`. Check that ranges span hundreds of ticks, that no joint has a minimum close to its maximum, and that no range is inverted.
4. Now plant the bug. Recalibrate the leader only, sweeping `wrist_flex` through about a quarter of its range and every other joint fully. Before opening the file, write down which numbers will differ, by roughly how much, and what teleoperation will do on that joint. Then open the file, run 30 seconds of bare teleoperation using the command from Exercise 5 step 1, and watch the follower's wrist.
5. Recalibrate the leader properly and confirm that the file shows full ranges again.

**✅ Checkpoint:** both calibrations are logged and sane; the planted lazy sweep appeared in the file as a narrow `wrist_flex` range and in teleoperation as a clipped follower wrist, and you predicted both before looking.

## Exercise 4 — Mount and measure the cameras [Predict → Run]

In this exercise you mount both cameras, map each physical camera to its operating-system index, and measure the frame rate each one actually delivers. The recording rate that H2 assumes is only real if the cameras deliver it, and many webcams advertise 30 fps but deliver half that at high resolution or on a shared USB hub, so the number has to be measured rather than read from the box.

Lesson 00 had you order two different webcam models, and this was deliberate. Two identical models are indistinguishable to the operating system, so a reboot or replug could silently swap their indices, and the wrist camera would become the overhead camera in your dataset without any error.

1. Attach the wrist mount and fix the overhead camera on a rigid mount. Do not use a gooseneck, because it drifts between sessions.
2. Enumerate the cameras with `lerobot-find-cameras opencv` and record which physical camera maps to which index. Indices can change after a reboot or replug, so recheck at every session start; this becomes the first line of H2's preflight checklist.
3. Before measuring, write down the frame rate you expect each camera to deliver at 640×480 and 30 fps on its own, and then with both cameras on one USB hub. Then measure:
   ```python
   import time
   from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
   with OpenCVCamera(OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480)) as cam:
       t0 = time.perf_counter(); [cam.read() for _ in range(90)]
       print(90 / (time.perf_counter() - t0), "fps")
   ```
   Run it for each camera alone, then for both simultaneously in two processes.
4. If either camera measures below 28 fps, move the cameras to separate USB controllers (avoid one hub for both) or drop the resolution. Record the configuration that passed.

**✅ Checkpoint:** both cameras deliver at least 28 measured fps simultaneously; the wrist camera sees the gripper fingers; the overhead camera sees the whole workspace; the measured numbers and the USB topology are in `BRINGUP.md`.

## Exercise 5 — Teleoperation [Predict → Run]

Now you drive the follower from the leader, first without cameras and then with both camera streams and the visualizer running. The question the exercise answers is whether the control loop keeps its rate once the same machine is also capturing video, because teleoperation quality is a bandwidth property as much as a mechanical one.

1. Before starting, write down whether you expect adding two camera streams to change the smoothness of teleoperation, and why. Then run bare teleoperation with the e-stop reachable and the workspace clear:
   ```bash
   lerobot-teleoperate \
     --robot.type=so101_follower --robot.port=<f-port> --robot.id=H1_follower \
     --teleop.type=so101_leader --teleop.port=<l-port> --teleop.id=H1_leader
   ```
2. Then add the cameras and the rerun visualizer by appending:
   ```bash
   --robot.cameras="{front: {type: opencv, index_or_path: <i>, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: <j>, width: 640, height: 480, fps: 30}}" \
   --display_data=true
   ```
3. Drive the follower through its workspace for five minutes. Feel for lag or stutter, and watch the joint plots in rerun for staircase patterns, which indicate that the loop is missing its rate. Practice a smooth pick motion, because H2 will ask for fifty of them.
4. Record a 30–60 second teleoperation video (a phone is fine), and reconcile the prediction from step 1 with what you observed.

**✅ Checkpoint:** the follower tracks the leader with no visible lag; both camera streams are live in rerun; the teleoperation video is saved.

## Exercise 6 — Write `trace.py` [Build]

Everything so far has run LeRobot's own code. In this exercise you write the specification for a script that drives the follower with the controller from Lesson 04, through the low-level robot API, so that the interface contract you established in simulation is tested against real motors. Have an AI tool draft the script from the specification below.

- Connect via `SO101Follower(SO101FollowerConfig(port=..., id="H1_follower"))`.
- Loop at 30 Hz: read `q = robot.get_observation()`, call `qdot = controller.step(q_meas, t)` with Lesson 04's controller imported unmodified, integrate one step, and send `robot.send_action(q_next)`. Cap per-step joint deltas at about 2° as a software speed limit. This cap is a software safeguard only; the power switch remains the real e-stop.
- Take targets from Lesson 04's trajectory generators: a 10 cm horizontal line, then a circle of 6 cm radius, at an end-effector speed of about 3 cm/s, well inside the workspace.
- Log each step to `logs/<target>.csv` with the timestep, the commanded q, and the measured q.
- The check: `python trace.py --target line` runs from one command, logs at 30 Hz or better, and the arm stops at the end of the trajectory.

Run the line first with your hand on the power switch, then the circle.

**✅ Checkpoint:** both logs exist at 30 Hz or better, and the arm's motion looked like a line and a circle to the eye.

## Exercise 7 — Score the trace [Predict → Run]

Finally, you compute where the end-effector actually went, using your own forward kinematics on the measured joint angles rather than the servo's report of itself, and compare it against the reference path and against the simulated run of the same controller. This is the first measurement of the reality gap in the course, and the exercise asks you to name its dominant source from the shape of the error rather than from a guess.

1. Before plotting, write down which of tracking lag, calibration offset, or gravity sag you expect to dominate the end-effector error, and the RMS error in millimetres you expect.
2. Compute end-effector positions from the measured angles through your Lesson 03 `fk(q)`, and compare them against the reference path and against the same controller's trajectory in the Lesson 04 MuJoCo simulation.
3. Plot the per-joint commanded-versus-measured overlay and the end-effector path (reference, simulation, real), and report the RMS and maximum end-effector error in millimetres.
4. Diagnose from the plots using the three signatures from the Principles section: lag makes the error grow where the reference moves fastest and lead or lag it in time; an offset displaces the whole trace by a fixed amount; sag makes the error follow the arm's extension rather than its speed. Name the dominant source and reconcile it with your prediction from step 1.

**✅ Checkpoint:** the circle is visibly circular; the RMS end-effector error is in single-digit millimetres; the dominant error source is named from a specific plot feature.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `BRINGUP.md` | ratio check per motor, port map, whether IDs were vendor-set, calibration table per arm (incl. the planted lazy-sweep numbers), camera↔index map with measured fps and USB topology, every gotcha hit |
| Teleop video | 30–60 s, smooth tracking, both cameras visible in the frame or in rerun |
| `trace.py` + `logs/*.csv` | reruns from one command; logs commanded + measured q at ≥ 30 Hz |
| `plots/` + `RESULTS.md` | line + circle EE paths (reference/sim/real), RMS + max error in mm, Exercise 2/3/4/5/7 predictions with reconciliations, error-source diagnosis in ≤ 8 sentences |

## Done when

- [ ] Every motor's ratio is verified; both arms are calibrated; the calibration values are recorded and sane.
- [ ] The planted lazy sweep was predicted, observed in the file and in teleoperation, and corrected.
- [ ] Both cameras deliver at least 28 measured fps simultaneously.
- [ ] Teleoperation is smooth end to end with `--display_data=true` running.
- [ ] Your controller traces the circle with single-digit-millimetre RMS error, scored through your own FK, with the dominant error source named.
- [ ] You have practiced reaching the e-stop once, deliberately, with the arm moving.

## Self-check

1. Why does the leader need three gear ratios while the follower uses one?
2. What exactly does `lerobot-setup-motors` write, where, and why does each motor need to be alone on the bus for it?
3. Your circle trace shows a constant offset of about 4 mm in one direction. Calibration, lag, or gravity? How do you tell from the data alone?
4. Why must `--robot.id` stay identical across teleoperation, recording, and evaluation?
5. Why two different webcam models?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Setup script can't find a motor | not alone on bus, dead 3-pin cable, or Waveshare board jumpers not on channel B (USB) | isolate the motor; swap cable; set jumpers per docs |
| Two motors respond to one ID | both left at factory ID 1 on the bus together | disconnect all, redo setup one at a time |
| Port path changed since yesterday | OS re-enumeration after replug/reboot | rerun `lerobot-find-port`; identify by unplug-diff, never by memory |
| Servo limp + hot after stall | STS3215 over-load/over-temp protection | power off, cool 10 min; find and remove the mechanical stall cause |
| Teleop stutters with cameras on, smooth without | USB bandwidth saturation | different USB controllers/ports; lower resolution; never one unpowered hub for both cams |
| Follower mirrors leader inverted on one joint | sloppy range sweep or horn mounted off-center | recalibrate that arm; verify the joint's min/max in the calibration file |
| Trace wildly off but teleop fine | your FK's zero/sign convention ≠ calibration convention | check per-joint sign and offset against `so101_new_calib.xml` (Lesson 03) |

## Going deeper

If you build a second pair or need to repair one, the assembly takes about two hours per arm once you have done one, working joint by joint against the per-joint videos in the documentation. Sort the motors by gear ratio first, reading the ratio printed on each label, because a mislabeled motor discovered later costs a full disassembly. Clean support material off the printed parts and dry-fit each motor before screwing it down. Do the motor ID setup before final assembly, or at minimum leave the three-pin connectors accessible. Assemble joints 1 through 6 in order using the two screw sizes noted in Exercise 1, and route the daisy-chain cable as you go, since retrofitting it through the elbow is difficult.

## References

- [SO-101 assembly & calibration docs](https://huggingface.co/docs/lerobot/so101): the authoritative walkthrough. CLI surface verified Aug 2026; reverify with `--help` if a flag is rejected.
- [LeRobot camera docs](https://huggingface.co/docs/lerobot/cameras); [teleop + recording tutorial](https://huggingface.co/docs/lerobot/il_robots).
- TheRobotStudio [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100): BOM, STLs, camera mounts.
- `hardware/ORDER.md`: the assembled-kit decision and its cost.
