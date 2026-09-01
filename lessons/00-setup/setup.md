# Lesson 00 — setup notes

## Version pins (verified 2026-09-01)

| | |
|---|---|
| OS | macOS (Darwin 25.6.0), Apple Silicon |
| Python | 3.12.12 (uv-managed) |
| lerobot | 0.6.1 |
| torch | 2.11.0 (MPS available) |
| mujoco | 3.12.0 |
| huggingface_hub / `hf` CLI | 1.29.0 |
| Full freeze | `requirements.lock` at repo root |

Accounts: HF `natuanand93` (token `robot-learning-token`), W&B `natu-anand` (set `WANDB_ENTITY` — session defaults to an org entity), GitHub `anatu`.

## Part 3 — model choice for Lesson 02

Use `SO-ARM100/Simulation/SO101/so101_new_calib.xml`: its six actuated joints (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`) match the LeRobot dataset state/action feature names verbatim, so no name-mapping layer is needed when writing episodes; the menagerie `trs_so_arm100` model has the same 6 DoF but its own names (`Rotation`, `Pitch`, `Elbow`, `Wrist_Pitch`, `Wrist_Roll`, `Jaw`).

Both models: nq = 6, 6 actuators, 5 revolute joints + gripper. Verified loading and opening in the viewer (plain `python -m mujoco.viewer` — see the mjpython pitfalls in README).
