#!/usr/bin/env python3
"""Stage 1.1 — Smoke test: load Piper MuJoCo model, verify kinematics, log IMU.

Goals:
1. Load piper_model.xml without errors
2. Print joint info (count, ranges, types)
3. Run 1s of physics at home pose — assert stable (no drift)
4. Drive joints with sinusoidal action for 10s — log joint trajectories and IMU
5. Plot trajectories + IMU to verify physics looks reasonable

Outputs (to outputs/sim/stage1_smoke/):
- model_info.txt
- home_pose_stability.png
- sinusoid_drive_trajectories.png
- sinusoid_drive_imu.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np


MODEL_PATH = Path(__file__).resolve().parent / "piper_model.xml"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs/sim/stage1_smoke"

# Real-robot control rate (matches v3 dataset fps)
CONTROL_HZ = 30.0
CONTROL_DT = 1.0 / CONTROL_HZ


def load_and_print_model_info():
    print(f"[load] {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    lines = []
    lines.append(f"Model loaded from {MODEL_PATH}")
    lines.append(f"  njoints (free + hinge + slide): {model.njnt}")
    lines.append(f"  nq (qpos size): {model.nq}")
    lines.append(f"  nv (qvel size): {model.nv}")
    lines.append(f"  nactuator: {model.nu}")
    lines.append(f"  nsensordata: {model.nsensordata}")
    lines.append(f"  opt.timestep: {model.opt.timestep} (physics dt)")
    lines.append("")
    lines.append("Joints:")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_type = ["free", "ball", "slide", "hinge"][model.jnt_type[i]]
        if model.jnt_limited[i]:
            r = model.jnt_range[i]
            lines.append(f"  [{i}] {name}: {joint_type}, range [{r[0]:.3f}, {r[1]:.3f}]")
        else:
            lines.append(f"  [{i}] {name}: {joint_type}, unlimited")
    lines.append("")
    lines.append("Actuators:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        ctrlrange = model.actuator_ctrlrange[i]
        lines.append(f"  [{i}] {name}: ctrlrange [{ctrlrange[0]:.3f}, {ctrlrange[1]:.3f}]")
    lines.append("")
    lines.append("Sensors:")
    for i in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        dim = model.sensor_dim[i]
        adr = model.sensor_adr[i]
        lines.append(f"  [{i}] {name}: dim={dim}, adr={adr}")

    info_str = "\n".join(lines)
    print(info_str)
    (OUT_DIR / "model_info.txt").write_text(info_str)
    return model, data


def home_pose_stability(model, data, duration_s=1.0):
    """Reset to zero qpos and run physics — should stay near zero (gravity will sag joints)."""
    print("\n[test] home pose stability")
    mujoco.mj_resetData(model, data)
    # Zero ctrl (= position control commands at neutral)
    data.ctrl[:] = 0.0

    n_steps = int(duration_s / model.opt.timestep)
    log_qpos = []
    log_imu = []
    for step in range(n_steps):
        mujoco.mj_step(model, data)
        log_qpos.append(data.qpos.copy())
        log_imu.append(data.sensordata.copy())

    log_qpos = np.array(log_qpos)
    log_imu = np.array(log_imu)
    t = np.arange(n_steps) * model.opt.timestep

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    for i in range(min(log_qpos.shape[1], 7)):
        axes[0].plot(t, log_qpos[:, i], label=f"q[{i}]")
    axes[0].set_title("Joint positions (qpos) at home with ctrl=0")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("rad / m")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    # IMU channels (3 accel + 3 gyro + 4 quat = 10 dims if all sensors)
    n_imu_dim = min(log_imu.shape[1], 6)
    names = ["ax", "ay", "az", "gx", "gy", "gz"]
    for i in range(n_imu_dim):
        axes[1].plot(t, log_imu[:, i], label=names[i])
    axes[1].set_title("IMU readings at home")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "home_pose_stability.png", dpi=130)
    plt.close(fig)

    # Stability check: max qpos drift should be small
    qpos_range = log_qpos.max(axis=0) - log_qpos.min(axis=0)
    print(f"  per-joint qpos range over {duration_s}s: {qpos_range}")
    print(f"  IMU az mean: {log_imu[:, 2].mean():.3f} (expect ~9.81 if free-fall direction)")


def sinusoid_drive(model, data, duration_s=10.0):
    """Drive joints 1-3 with small-amplitude sin waves; log everything."""
    print(f"\n[test] sinusoid drive for {duration_s}s")
    mujoco.mj_resetData(model, data)

    n_phys_per_ctrl = int(round(CONTROL_DT / model.opt.timestep))
    n_ctrl_steps = int(duration_s * CONTROL_HZ)
    print(f"  physics dt = {model.opt.timestep}, control dt = {CONTROL_DT}, "
          f"n_phys_per_ctrl = {n_phys_per_ctrl}")
    print(f"  n_ctrl_steps = {n_ctrl_steps}")

    # Joint indices for joints 1, 2, 3 in the actuator order
    # Per model_info, joints 1-6 are the arm, joint7/8 are gripper fingers
    log_t = []
    log_action = []
    log_qpos = []
    log_imu = []

    for s in range(n_ctrl_steps):
        t = s * CONTROL_DT
        # Small amplitudes per joint (well within range)
        action = np.zeros(model.nu)
        action[0] = 0.3 * np.sin(2 * np.pi * 0.2 * t)         # j1: 0.2 Hz, ±0.3 rad
        action[1] = 1.0 + 0.5 * np.sin(2 * np.pi * 0.15 * t)  # j2: 0.15 Hz, 0.5-1.5 rad
        action[2] = -0.5 + 0.3 * np.sin(2 * np.pi * 0.25 * t) # j3: 0.25 Hz, -0.8 to -0.2 rad
        # j4-6 stay at zero, gripper open
        data.ctrl[:] = action

        # Run physics for n_phys_per_ctrl steps
        for _ in range(n_phys_per_ctrl):
            mujoco.mj_step(model, data)

        log_t.append(t)
        log_action.append(action.copy())
        log_qpos.append(data.qpos.copy())
        log_imu.append(data.sensordata.copy())

    log_t = np.array(log_t)
    log_action = np.array(log_action)
    log_qpos = np.array(log_qpos)
    log_imu = np.array(log_imu)

    # Plot trajectories
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    titles = ["Commanded action (ctrl)", "Joint position (qpos)", "Tracking error (qpos - ctrl) for j1-j3"]
    # Note: qpos may include extra free-base DOFs; assume first 8 = joints 1..8
    for i in range(min(8, log_qpos.shape[1])):
        axes[0].plot(log_t, log_action[:, i] if i < log_action.shape[1] else np.zeros_like(log_t), label=f"a[{i}]")
        axes[1].plot(log_t, log_qpos[:, i], label=f"q[{i}]")
    axes[0].set_title(titles[0])
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[0].grid(alpha=0.3)
    axes[1].set_title(titles[1])
    axes[1].legend(loc="upper right", fontsize=8, ncol=2)
    axes[1].grid(alpha=0.3)

    # Tracking error for j1-j3
    for i in range(3):
        axes[2].plot(log_t, log_qpos[:, i] - log_action[:, i], label=f"j{i+1}")
    axes[2].set_title(titles[2])
    axes[2].set_xlabel("time (s)")
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "sinusoid_drive_trajectories.png", dpi=130)
    plt.close(fig)

    # Plot IMU
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for i, n in enumerate(["ax", "ay", "az"]):
        axes[0].plot(log_t, log_imu[:, i], label=n)
    axes[0].set_title("Accelerometer (m/s²) — wrist IMU site on link6")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(alpha=0.3)
    for i, n in enumerate(["gx", "gy", "gz"]):
        axes[1].plot(log_t, log_imu[:, 3 + i], label=n)
    axes[1].set_title("Gyroscope (rad/s)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "sinusoid_drive_imu.png", dpi=130)
    plt.close(fig)

    # Summary stats
    print(f"  j1-3 max tracking error: "
          f"{np.abs(log_qpos[:, :3] - log_action[:, :3]).max(axis=0)}")
    print(f"  IMU accel range:")
    for i, n in enumerate(["ax", "ay", "az"]):
        print(f"    {n}: [{log_imu[:, i].min():.3f}, {log_imu[:, i].max():.3f}]  "
              f"mean={log_imu[:, i].mean():.3f}")
    print(f"  IMU gyro range:")
    for i, n in enumerate(["gx", "gy", "gz"]):
        print(f"    {n}: [{log_imu[:, 3+i].min():.3f}, {log_imu[:, 3+i].max():.3f}]  "
              f"std={log_imu[:, 3+i].std():.3f}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, data = load_and_print_model_info()
    home_pose_stability(model, data, duration_s=1.0)
    sinusoid_drive(model, data, duration_s=10.0)
    print(f"\n[done] outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
