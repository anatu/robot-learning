# H1 — Build & Bring-Up the SO-101

Assemble and calibrate the leader/follower pair, get two cameras streaming, prove teleop works, then close the loop with your own controller — turning a box of servos into an instrument you trust.

| | |
|---|---|
| **Phase** | Hardware track (start any time after Lesson 02) |
| **Time** | ~2 sessions: 3–5 h assembly + 2–3 h motor setup/calibration/teleop + 1–2 h controller trace |
| **Cost** | $0 beyond the kit already ordered in Lesson 00 |
| **Prerequisites** | 00 (kit ordered, LeRobot env), 03 (your `fk(q)` — used to score the trace), 04 (your diff-IK controller `q̇ = f(q, target)`) |
| **Feeds into** | H2 (recording rig), H3–H5 (the physical platform), every teleop session after this |

## Learning objectives

After this lesson you can:

1. **Assemble** both arms without mixing the leader's three gear ratios, and explain why the leader is geared differently at all.
2. **Configure** a Feetech servo bus from scratch: ports, motor IDs, baudrate, and what lives in EEPROM vs RAM.
3. **Calibrate** both arms and interpret the calibration file well enough to spot a bad calibration from the numbers.
4. **Operate** the rig safely: torque discipline, workspace clearance, and a power-cut e-stop within reach — as habits, not aspirations.
5. **Quantify** the reality gap at the lowest level: commanded vs measured joint angles, and end-effector error in mm through your own FK.

## Background

**Why the leader moves easily and the follower doesn't.** The follower uses 6× STS3215 motors all geared 1/345 — high torque, holds pose against gravity. The leader mixes three ratios so a human hand can back-drive it: 1/191 on base and elbow, 1/345 on shoulder lift (it must hold its own weight), 1/147 on the wrist and gripper joints. Put a 1/147 motor in a shoulder and the arm slumps; put a 1/345 in a wrist and teleop feels like stirring cement. Sort the motors by ratio *before* assembly.

**The bus.** All six motors daisy-chain on one 3-pin serial bus to the control board. Each motor needs a unique ID (1 = shoulder pan … 6 = gripper) and a common baudrate, written once into motor EEPROM by `lerobot-setup-motors`. New motors all ship as ID 1 — that's why setup connects them one at a time. The `--robot.id`/`--teleop.id` you choose names the calibration file; use the same id forever after ("H1_follower", "H1_leader").

**What calibration is.** A mapping from raw magnetic-encoder ticks to degrees with a shared zero, recorded by holding mid-range and then sweeping each joint's full range. Without it, a policy trained on one arm means nothing on another — and your leader can't drive your follower.

**Safety doctrine (applies to every hardware lesson).** (1) The power supply's switch or plug is the e-stop; it stays within arm's reach whenever torque is on. (2) Never leave an arm powered and unattended. (3) Clear a ~40 cm radius around the follower before enabling torque; the first bug you write will command a full-speed swing. (4) STS3215s cut out on over-temperature/over-load — if a servo goes limp and hot after stalling against a limit, power off and let it cool; repeated stalls kill servos.

| Source | Read for |
|---|---|
| [SO-101 docs](https://huggingface.co/docs/lerobot/so101) | the gear-ratio table, per-joint assembly videos, setup/calibration commands — keep it open on a second screen throughout |
| TheRobotStudio [SO-ARM100 repo](https://github.com/TheRobotStudio/SO-ARM100) | printed-parts fit notes, camera-mount STLs used in Part 4 |
| [LeRobot camera docs](https://huggingface.co/docs/lerobot/cameras) | `lerobot-find-cameras`, camera config JSON, why indices drift |

## Part 1 — Assembly (3–5 h)

Two arms, ~2 h each once you've done one. Work joint by joint against the docs' per-joint videos.

1. Unbox and **sort motors by gear ratio first** (ratio is printed on the motor label). Leader kit: 2× 1/191 (base, elbow), 1× 1/345 (shoulder lift), 3× 1/147 (wrist flex, wrist roll, gripper). Follower: 6× 1/345. Mislabeling here costs a full disassembly later.
2. Clean support material off printed parts; dry-fit each motor before screwing.
3. **Do motor ID setup before final assembly** (Part 2) — or at minimum leave the 3-pin connectors accessible: setup requires connecting motors to the board one at a time.
4. Assemble joints 1→6 per the docs. Two screw sizes only: M2×6 into motor bodies, M3×6 into horns/structure. Secure top horns with the M3 horn screw; bottom horns are unscrewed by design.
5. Route the daisy-chain cable as you go — retrofitting it through the elbow is miserable.

**✅ Checkpoint:** both arms assembled; every joint back-drivable by hand through its full range with power off, no grinding; the leader's shoulder lift holds position when released (1/345 confirmed in the right place).

## Part 2 — Ports, motor IDs, baudrate (~45 min)

1. Find each control board's port (`lerobot-find-port` — unplug when prompted; it identifies by diff). Record both port paths. On Linux: `sudo chmod 666 /dev/ttyACM*`.
2. Follower, one motor at a time starting with the gripper, exactly as the script instructs:
   ```bash
   lerobot-setup-motors --robot.type=so101_follower --robot.port=<follower-port>
   ```
   The script walks motor 6 → 1; each motor must be the *only* one on the bus when you press Enter. Expect `'gripper' motor id set to 6` etc.
3. Leader (note the different namespace):
   ```bash
   lerobot-setup-motors --teleop.type=so101_leader --teleop.port=<leader-port>
   ```
4. Reconnect the full daisy-chain on both arms.

**✅ Checkpoint:** setup script completes for both arms without a retry; bring-up log records port ↔ arm mapping and any hiccup (this log is a deliverable — start it now).

## Part 3 — Calibration (~30 min)

1. Follower:
   ```bash
   lerobot-calibrate --robot.type=so101_follower --robot.port=<port> --robot.id=H1_follower
   ```
   Hold mid-range pose → Enter → sweep every joint through its **full** range. Lazy sweeps clip the usable workspace.
2. Leader: same with `--teleop.type=so101_leader --teleop.id=H1_leader`.
3. Open the written calibration file (under `~/.cache/huggingface/lerobot/calibration/`) and copy the per-joint min/center/max into your bring-up log. Sanity-check: ranges spanning hundreds of ticks, no joint with min ≈ max, no inverted range.

**✅ Checkpoint:** calibration values logged and sane; you can say which physical pose is "all joints centered."

## Part 4 — Cameras (~45 min)

Two *different* webcam models (Lesson 00's order) — deliberate, so OS enumeration can't silently swap them.

1. Print/attach the wrist mount and fix the overhead camera on a rigid mount (no gooseneck — it drifts between sessions).
2. Enumerate: `lerobot-find-cameras opencv`. Record which physical camera maps to which index. Indices can change after reboot/replug — recheck at every session start (this becomes line 1 of H2's pre-flight checklist).
3. Measure actual throughput at recording settings (640×480@30) — many webcams advertise 30 fps but deliver 15 at 1080p or on a shared USB hub:
   ```python
   import time
   from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
   with OpenCVCamera(OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480)) as cam:
       t0 = time.perf_counter(); [cam.read() for _ in range(90)]
       print(90 / (time.perf_counter() - t0), "fps")
   ```
4. If either camera measures < 28 fps: separate USB controllers (avoid one hub for both), or drop resolution.

**✅ Checkpoint:** both cameras ≥ 28 measured fps simultaneously; wrist camera sees the gripper fingers; overhead sees the whole workspace.

## Part 5 — Teleoperation (~30 min)

1. Bare teleop first — e-stop reachable, workspace clear:
   ```bash
   lerobot-teleoperate \
     --robot.type=so101_follower --robot.port=<f-port> --robot.id=H1_follower \
     --teleop.type=so101_leader --teleop.port=<l-port> --teleop.id=H1_leader
   ```
2. Then with cameras and rerun visualization — add:
   ```bash
   --robot.cameras="{front: {type: opencv, index_or_path: <i>, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: <j>, width: 640, height: 480, fps: 30}}" \
   --display_data=true
   ```
3. Drive the follower through its workspace for 5 minutes. Feel for lag or stutter; watch the joint plots in rerun for staircase patterns (loop-rate trouble). Practice a smooth pick motion — H2 will demand fifty of them.
4. Record a 30–60 s teleop video (phone is fine).

**✅ Checkpoint:** follower tracks the leader with no visible lag; both camera streams live in rerun; teleop video saved.

## Part 6 — Close the loop with your own controller (1–2 h)

Everything so far is LeRobot's code. Now drive the arm with *yours*, reusing Lesson 04's diff-IK controller (interface: `q̇ = f(q, target)`) against the low-level API:

1. Write `trace.py`: connect via `SO101Follower(SO101FollowerConfig(port=..., id="H1_follower"))`; loop at 30 Hz: `q = robot.get_observation()` → controller → integrate → `robot.send_action(q_next)`. Cap per-step joint deltas (~2°) as a software speed limit.
2. Targets from Lesson 04: a 10 cm horizontal line, then a 6 cm-radius circle, EE speed ~3 cm/s, well inside the workspace.
3. Log per step: timestep, commanded q, measured q. Compute EE positions from *measured* angles through your Lesson 03 `fk(q)` and compare against the reference path and against the same controller's trajectory in the Lesson 04 MuJoCo sim.
4. Plot: per-joint commanded-vs-measured overlay; EE path (reference / sim / real); report RMS and max EE error in mm.

**✅ Checkpoint:** circle visibly circular; RMS EE error single-digit mm; you can name the dominant error source (tracking lag vs calibration offset vs gravity sag — the plots distinguish them: lag follows velocity, offset is constant, sag is configuration-dependent).

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `BRINGUP.md` | port map, motor-ID sequence, calibration table per arm, camera↔index map with measured fps, every gotcha hit |
| Teleop video | 30–60 s, smooth tracking, both cameras visible in the frame or in rerun |
| `trace.py` + `logs/*.csv` | reruns from one command; logs commanded + measured q at ≥ 30 Hz |
| Trace plots + `RESULTS.md` | line + circle EE paths (reference/sim/real), RMS + max error in mm, error-source diagnosis in ≤ 8 sentences |

## Done when

- [ ] Both arms calibrated; calibration values recorded and sane.
- [ ] Both cameras ≥ 28 measured fps simultaneously.
- [ ] Teleop is smooth end-to-end with `--display_data=true` running.
- [ ] Your controller traces the circle with single-digit-mm RMS error, scored through your own FK.
- [ ] You've practiced the e-stop reach once, deliberately, with the arm moving.

## Self-check

1. Why does the leader need three gear ratios while the follower uses one?
2. What exactly does `lerobot-setup-motors` write, where, and why does each motor need to be alone on the bus for it?
3. Your circle trace shows a constant ~4 mm offset in one direction. Calibration, lag, or gravity? How do you tell from the data alone?
4. Why must `--robot.id` stay identical across teleop, recording, and evaluation?
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

## References

- [SO-101 assembly & calibration docs](https://huggingface.co/docs/lerobot/so101) — the authoritative walkthrough; CLI surface verified Aug 2026, reverify with `--help` if flags reject.
- [LeRobot camera docs](https://huggingface.co/docs/lerobot/cameras); [teleop + recording tutorial](https://huggingface.co/docs/lerobot/il_robots).
- TheRobotStudio [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) — BOM, STLs, camera mounts.
