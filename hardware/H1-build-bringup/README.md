# H1 — Bring-Up the SO-101

Turn an assembled leader/follower pair into an instrument you trust: verify the build, configure and calibrate the bus, get two cameras streaming, prove teleop, then close the loop with your own controller and measure the reality gap in millimetres.

| | |
|---|---|
| **Phase** | Hardware track (start any time after Lesson 02) |
| **Time** | ~1 session: 1 h unbox + inspect, 2–3 h motor setup / calibration / cameras / teleop, 1–2 h controller trace |
| **Cost** | $0 beyond the kit already ordered in Lesson 00 (assembled Partabot kit — see `hardware/ORDER.md`) |
| **Prerequisites** | 00 (kit ordered, LeRobot env), 03 (your `fk(q)` — scores the trace), 04 (your controller behind `qdot = controller.step(q_meas, t)`) |
| **Feeds into** | H2 (recording rig), H3–H5 (the physical platform), every teleop session after this |

## Learning objectives

After this lesson you can:

1. **Verify** an assembled pair against the gear-ratio table and explain why the leader is geared differently at all.
2. **Explain** a Feetech servo bus: ports, motor IDs, baudrate, and what lives in EEPROM vs RAM.
3. **Diagnose** a bad calibration from the numbers in the calibration file alone.
4. **Operate** the rig safely: torque discipline, workspace clearance, and a power-cut e-stop within reach — as habits, not aspirations.
5. **Quantify** the reality gap at the lowest level — commanded vs measured joint angles, end-effector error in mm through your own FK — and name its dominant source from the plots.

## Principles

**Why the leader moves easily and the follower doesn't.** The follower uses 6× STS3215 motors all geared 1/345 — high torque, holds pose against gravity. The leader mixes three ratios so a human hand can back-drive it: 1/191 on base and elbow, 1/345 on shoulder lift (it must hold its own weight), 1/147 on the wrist and gripper joints. Put a 1/147 motor in a shoulder and the arm slumps; put a 1/345 in a wrist and teleop feels like stirring cement. Your kit arrived assembled; the ratios are still yours to verify, because a mis-sorted motor shows up as slump or stiffness, not as an error message.

**The bus.** All six motors daisy-chain on one 3-pin serial bus to the control board. Each motor needs a unique ID (1 = shoulder pan … 6 = gripper) and a common baudrate, written once into motor EEPROM by `lerobot-setup-motors`. New motors all ship as ID 1 — that's why setup connects them one at a time. The `--robot.id`/`--teleop.id` you choose names the calibration file; use the same id forever after ("H1_follower", "H1_leader").

**What calibration is.** A mapping from raw magnetic-encoder ticks to degrees with a shared zero, recorded by holding mid-range and then sweeping each joint's full range. Without it, a policy trained on one arm means nothing on another — and your leader can't drive your follower. The file is the whole calibration: a joint whose min ≈ max, or whose range is a few ticks wide, was swept lazily, and every downstream lesson inherits the clipped workspace.

**The reality gap starts here.** Commanded and measured joint angles differ for three separable reasons, each with a signature in a trace plot: *tracking lag* follows velocity, *calibration offset* is constant, *gravity sag* is configuration-dependent. Reading those signatures is the skill H3–H5 lean on when a policy "almost" works.

**Safety doctrine (applies to every hardware lesson).** (1) The power supply's switch or plug is the e-stop; it stays within arm's reach whenever torque is on. (2) Never leave an arm powered and unattended. (3) Clear a ~40 cm radius around the follower before enabling torque; the first bug you write will command a full-speed swing. (4) STS3215s cut out on over-temperature/over-load — if a servo goes limp and hot after stalling against a limit, power off and let it cool; repeated stalls kill servos.

**Carry forward**

- Leader 1/191 (base, elbow) · 1/345 (shoulder lift) · 1/147 (wrist flex, wrist roll, gripper); follower 6× 1/345.
- Motor IDs and baudrate live in EEPROM, written once; calibration lives in a file named by `--robot.id`, and that id never changes.
- A calibration is only as good as the sweep: check the file, not the feel.
- Lag follows velocity, offset is constant, sag is configuration-dependent.
- E-stop within reach whenever torque is on. No exceptions.

| Source | Read for |
|---|---|
| [SO-101 docs](https://huggingface.co/docs/lerobot/so101) | the gear-ratio table, per-joint assembly videos (for inspection and any repair), setup/calibration commands — keep it open on a second screen throughout |
| TheRobotStudio [SO-ARM100 repo](https://github.com/TheRobotStudio/SO-ARM100) | printed-parts fit notes, camera-mount STLs used in Exercise 4 |
| [LeRobot camera docs](https://huggingface.co/docs/lerobot/cameras) | `lerobot-find-cameras`, camera config JSON, why indices drift |

## Exercise 1 — Unbox and inspect [Read]

Tests objective 1: the gear-ratio principle, applied to a build you didn't do.

1. Unbox both arms. **Check every motor's ratio by its label** against the table: leader 2× 1/191 (base, elbow), 1× 1/345 (shoulder lift), 3× 1/147 (wrist flex, wrist roll, gripper); follower 6× 1/345. A label you can't read gets the back-drive test in step 2.
2. Power off, back-drive every joint by hand through its full range: no grinding, no binding. The leader's shoulder lift should hold position when released (1/345 confirmed in the right place); leader wrist and gripper should move freely (1/147).
3. Trace the daisy-chain cable through the elbow and confirm every 3-pin connector is reachable without disassembly — Exercise 2 may need motors isolated one at a time.
4. Note the fastener facts for any repair: two screw sizes only, M2×6 into motor bodies, M3×6 into horns/structure; top horns secured with the M3 horn screw; bottom horns are unscrewed by design.
5. Start `BRINGUP.md` now: ratio check per motor, any deviation from the docs' assembly, anything loose.

**✅ Checkpoint:** every motor's ratio verified and logged; every joint back-drivable through its full range with power off; leader shoulder lift holds when released.

## Exercise 2 — Ports, motor IDs, baudrate [Predict → Run]

Tests objective 2: what setup writes, where, and why a motor must be alone on the bus.

1. **Write first** in `BRINGUP.md`: what `lerobot-setup-motors` writes, into which memory, and what you expect to happen if two factory-fresh motors share the bus when you press Enter.
2. Find each control board's port (`lerobot-find-port` — unplug when prompted; it identifies by diff). Record both port paths. On Linux: `sudo chmod 666 /dev/ttyACM*`.
3. An assembled kit may ship with IDs already programmed — check the vendor's notes. If it did, go straight to Exercise 3 and come back here only if calibration cannot see all six motors. Otherwise, follower, one motor at a time starting with the gripper, exactly as the script instructs:
   ```bash
   lerobot-setup-motors --robot.type=so101_follower --robot.port=<follower-port>
   ```
   The script walks motor 6 → 1; each motor must be the *only* one on the bus when you press Enter. Expect `'gripper' motor id set to 6` etc.
4. Leader (note the different namespace):
   ```bash
   lerobot-setup-motors --teleop.type=so101_leader --teleop.port=<leader-port>
   ```
5. Reconnect the full daisy-chain on both arms. Reconcile step 1's prediction with what the script printed.

**✅ Checkpoint:** both arms enumerate all six motors; `BRINGUP.md` records port ↔ arm mapping, whether IDs were vendor-set, and any hiccup.

## Exercise 3 — Calibrate, then break it on purpose [Diagnose]

Tests objective 3: a bad calibration is visible in the file before it is visible in motion.

1. Follower:
   ```bash
   lerobot-calibrate --robot.type=so101_follower --robot.port=<port> --robot.id=H1_follower
   ```
   Hold mid-range pose → Enter → sweep every joint through its **full** range. Lazy sweeps clip the usable workspace.
2. Leader: same with `--teleop.type=so101_leader --teleop.id=H1_leader`.
3. Open the written calibration file (under `~/.cache/huggingface/lerobot/calibration/`) and copy the per-joint min/center/max into `BRINGUP.md`. Sanity-check: ranges spanning hundreds of ticks, no joint with min ≈ max, no inverted range.
4. **Plant the bug.** Recalibrate the *leader* only, sweeping `wrist_flex` through about a quarter of its range and everything else fully. Before opening the file, write down: which numbers will differ, by roughly how much, and what teleop will do on that joint. Open the file; then run 30 s of bare teleop (Exercise 5 step 1's command) and watch the follower's wrist.
5. Recalibrate the leader properly. Confirm the file is back to full ranges.

**✅ Checkpoint:** both calibrations logged and sane; the planted lazy sweep is visible in the file as a narrow `wrist_flex` range and in teleop as a clipped follower wrist — and you predicted both before looking.

## Exercise 4 — Cameras [Predict → Run]

Tests the recording-rate assumption H2 depends on: advertised fps is not delivered fps.

Two *different* webcam models (Lesson 00's order) — deliberate, so OS enumeration can't silently swap them.

1. Attach the wrist mount and fix the overhead camera on a rigid mount (no gooseneck — it drifts between sessions).
2. Enumerate: `lerobot-find-cameras opencv`. Record which physical camera maps to which index. Indices can change after reboot/replug — recheck at every session start (this becomes line 1 of H2's pre-flight checklist).
3. **Write first:** the fps you expect each camera to deliver at 640×480@30 alone, and both together on one USB hub. Then measure — many webcams advertise 30 fps but deliver 15 at 1080p or on a shared USB hub:
   ```python
   import time
   from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
   with OpenCVCamera(OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480)) as cam:
       t0 = time.perf_counter(); [cam.read() for _ in range(90)]
       print(90 / (time.perf_counter() - t0), "fps")
   ```
   Run it for each camera alone, then both simultaneously (two processes).
4. If either camera measures < 28 fps: separate USB controllers (avoid one hub for both), or drop resolution. Record the configuration that passed.

**✅ Checkpoint:** both cameras ≥ 28 measured fps simultaneously; wrist camera sees the gripper fingers; overhead sees the whole workspace; measured numbers and the USB topology in `BRINGUP.md`.

## Exercise 5 — Teleoperation [Predict → Run]

Tests whether the loop rate survives cameras: teleop quality is a bandwidth property.

1. **Write first:** whether you expect adding two camera streams to change teleop smoothness, and why. Bare teleop first — e-stop reachable, workspace clear:
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
4. Record a 30–60 s teleop video (phone is fine). Reconcile step 1.

**✅ Checkpoint:** follower tracks the leader with no visible lag; both camera streams live in rerun; teleop video saved.

## Exercise 6 — `trace.py` [Build]

Tests the interface contract from Lesson 04 against real motors. Everything so far is LeRobot's code; now the arm runs *yours*. Spec for an AI tool:

- Connect via `SO101Follower(SO101FollowerConfig(port=..., id="H1_follower"))`.
- Loop at 30 Hz: `q = robot.get_observation()` → `qdot = controller.step(q_meas, t)` (Lesson 04's controller, imported unmodified) → integrate one step → `robot.send_action(q_next)`. Cap per-step joint deltas (~2°) as a software speed limit — this cap is the software e-stop; the power switch is the real one.
- Targets from Lesson 04's trajectory generators: a 10 cm horizontal line, then a 6 cm-radius circle, EE speed ~3 cm/s, well inside the workspace.
- Log per step to `logs/<target>.csv`: timestep, commanded q, measured q.
- The check: `python trace.py --target line` runs from one command, logs at ≥ 30 Hz, and the arm stops at the end of the trajectory.

Run the line first with your hand on the power switch, then the circle.

**✅ Checkpoint:** both logs exist at ≥ 30 Hz; the arm's motion looked like a line and a circle to the eye.

## Exercise 7 — Score the trace [Predict → Run]

Tests objective 5: the reality gap, measured through your own FK rather than the servo's opinion of itself.

1. **Write first:** which of tracking lag, calibration offset, or gravity sag you expect to dominate the EE error, and the RMS error in mm you expect.
2. Compute EE positions from *measured* angles through your Lesson 03 `fk(q)`; compare against the reference path and against the same controller's trajectory in the Lesson 04 MuJoCo sim.
3. Plot: per-joint commanded-vs-measured overlay; EE path (reference / sim / real); report RMS and max EE error in mm.
4. **Diagnose from the plots**, using the three signatures: lag follows velocity (error grows where the reference moves fastest and leads/lags in time), offset is constant (a fixed displacement of the whole trace), sag is configuration-dependent (error tracks the arm's extension, not its speed). Name the dominant source and reconcile with step 1.

**✅ Checkpoint:** circle visibly circular; RMS EE error single-digit mm; the dominant error source is named from a specific plot feature, not a guess.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `BRINGUP.md` | ratio check per motor, port map, whether IDs were vendor-set, calibration table per arm (incl. the planted lazy-sweep numbers), camera↔index map with measured fps and USB topology, every gotcha hit |
| Teleop video | 30–60 s, smooth tracking, both cameras visible in the frame or in rerun |
| `trace.py` + `logs/*.csv` | reruns from one command; logs commanded + measured q at ≥ 30 Hz |
| `plots/` + `RESULTS.md` | line + circle EE paths (reference/sim/real), RMS + max error in mm, Exercise 2/3/4/5/7 predictions with reconciliations, error-source diagnosis in ≤ 8 sentences |

## Done when

- [ ] Every motor's ratio verified; both arms calibrated; calibration values recorded and sane.
- [ ] The planted lazy sweep was predicted, observed in the file and in teleop, and corrected.
- [ ] Both cameras ≥ 28 measured fps simultaneously.
- [ ] Teleop is smooth end-to-end with `--display_data=true` running.
- [ ] Your controller traces the circle with single-digit-mm RMS error, scored through your own FK, with the dominant error source named.
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

## Going deeper

**DIY assembly**, for a second pair or a repair. Two arms, ~2 h each once you've done one; work joint by joint against the docs' per-joint videos. Sort motors by gear ratio first (ratio is printed on the label) — mislabeling costs a full disassembly later. Clean support material off printed parts; dry-fit each motor before screwing. Do motor ID setup *before* final assembly, or at minimum leave the 3-pin connectors accessible. Assemble joints 1→6 per the docs with the two screw sizes from Exercise 1. Route the daisy-chain cable as you go — retrofitting it through the elbow is miserable.

## References

- [SO-101 assembly & calibration docs](https://huggingface.co/docs/lerobot/so101) — the authoritative walkthrough; CLI surface verified Aug 2026, reverify with `--help` if flags reject.
- [LeRobot camera docs](https://huggingface.co/docs/lerobot/cameras); [teleop + recording tutorial](https://huggingface.co/docs/lerobot/il_robots).
- TheRobotStudio [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) — BOM, STLs, camera mounts.
- `hardware/ORDER.md` — the assembled-kit decision and its cost.
