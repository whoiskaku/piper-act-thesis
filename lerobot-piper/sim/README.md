# Piper MuJoCo Simulation (Stage 1 of RESEARCH_PLAN_v4)

This directory contains the MuJoCo simulation setup used to generate noisy
training data for RefineNet (per RESEARCH_PLAN_v4 §5 Stage 1-2).

## Setup (first-time after fresh clone)

The `meshes/` directory is gitignored due to size (~60 MB STL files). It must
be copied locally from the official Piper ROS repository:

```bash
# From the lerobot-piper repo root:
cp -r /path/to/piper_ros/src/piper_description/meshes lerobot-piper/sim/
```

On the development machine (enolab-guo), the source is:
```
/home/enolab-guo/piper_ros/src/piper_description/meshes/
```

The model XML (`piper_model.xml`) was originally
`piper_ros/src/piper_description/mujoco_model/piper_description.xml` with
mesh paths rewritten from `../meshes/` to `meshes/` to match this directory
layout. It is committed to git (small, ~5 KB).

## Files

| File | Description |
|---|---|
| `piper_model.xml` | MuJoCo model: 6 arm joints + 2 gripper fingers + wrist IMU |
| `meshes/` | STL/DAE meshes (gitignored, copy from `piper_ros`) |
| `test_load_model.py` | Stage 1.1 smoke test: load model, verify physics, log IMU |

## Sim conventions

- Control rate: 30 Hz (matches v3 dataset fps and real robot)
- Physics timestep: 0.002 s (500 Hz, MuJoCo default)
- Action format: position commands per actuator (7 arm + gripper, indices 0..7)
- IMU site: on link6 at `pos="0 0 0.13"` (matches TSND151 wrist mount)
- IMU sensors: `ee_accel` (3-dim, m/s²), `ee_gyro` (3-dim, rad/s), `ee_quat` (4-dim)

## Status (Stage 1 sub-tasks)

- [x] **1.1 Smoke test** — `test_load_model.py` validates loading + physics + IMU output
- [ ] **1.2 Home pose calibration** — set qpos initial values to match real-robot home pose
- [ ] **1.3 Noise injection** — implement 3 noise layers (action, IMU sensor, vibration) per `outputs/sim/noise_params.json`
- [ ] **1.4 Demo replay validation** — replay v3 demo actions in sim, compare resulting IMU PSD to real
- [ ] **1.5 Noise param iteration** — refine heuristic_defaults_for_stage1 based on PSD mismatch

## Outputs

Stage 1 smoke test outputs are in `lerobot-piper/outputs/sim/stage1_smoke/`:
- `model_info.txt` — joints, actuators, sensors listing
- `home_pose_stability.png` — qpos drift + IMU at home (ctrl=0)
- `sinusoid_drive_*.png` — joint and IMU responses to sin-wave action drive
