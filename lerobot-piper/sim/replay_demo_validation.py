#!/usr/bin/env python3
"""Stage 1.4 — Demo replay validation.

Goal: Validate sim physics fidelity before adding any noise. Run a v3 demo
action sequence in MuJoCo (no noise), record sim IMU/state, compare against
the real-robot IMU/state recorded in v3 dataset.

This is a CLEAN sim — no action noise, no IMU sensor noise, no vibration.
If sim baseline doesn't qualitatively match real, adding noise won't fix it;
we'd need to fix the physical model first (URDF mass/inertia, damping, etc).

Outputs to outputs/sim/stage1_replay/:
  ep<i>_state_vs_action.png   — sim state trajectory vs commanded actions
  ep<i>_imu_compare.png       — sim IMU vs real IMU time-domain
  ep<i>_psd_compare.png       — sim IMU vs real IMU PSD overlay
  ep<i>_metrics.json          — per-channel correlation, RMS error
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
import json


SIM_DIR = Path(__file__).resolve().parent
MODEL_PATH = SIM_DIR / "piper_model.xml"
DEMO_DIR = Path.home() / ".cache/huggingface/lerobot/local/piper_pick_place_imu_v3"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs/sim/stage1_replay"

# State encoding: state[mdeg] / 57295.78 = action[rad]
MDEG_TO_RAD = np.pi / 180 / 1000
# Gripper conversion (placeholder, less important for IMU)
GRIPPER_SCALE = 1.0 / 800000

CONTROL_HZ = 30.0
CONTROL_DT = 1.0 / CONTROL_HZ


def load_real_demo(ep_idx: int):
    pq = DEMO_DIR / f"data/chunk-000/episode_{ep_idx:06d}.parquet"
    df = pd.read_parquet(pq)
    return {
        "action":      np.stack(df["action"].to_numpy()).astype(np.float32),    # (T, 7), rad
        "state_mdeg":  np.stack(df["observation.state"].to_numpy()).astype(np.float32),  # (T, 7), mdeg
        "imu":         np.stack(df["observation.imu_real"].to_numpy()).astype(np.float32),  # (T, 6)
    }


def state_mdeg_to_rad(state_mdeg):
    """Convert v3 state (encoder mdeg) to radians for sim qpos."""
    out = np.zeros(8)  # mujoco model has 8 qpos: 6 arm + 2 gripper fingers
    out[:6] = state_mdeg[:6] * MDEG_TO_RAD
    # Gripper: state[6] is some scaled value; map to slide range [0, 0.035]
    # For now use action-space gripper directly (clipped to range)
    grip_val = float(state_mdeg[6]) * GRIPPER_SCALE
    grip_val = max(0.0, min(0.035, grip_val))
    out[6] = grip_val
    out[7] = -grip_val  # joint8 is mirror of joint7
    return out


def replay_in_sim(model, data, real_demo, set_initial_pose=True):
    """Replay real demo action sequence in sim, return sim state + IMU traces."""
    action_seq = real_demo["action"]
    n_steps = len(action_seq)
    n_phys_per_ctrl = int(round(CONTROL_DT / model.opt.timestep))

    # Initialize sim
    mujoco.mj_resetData(model, data)
    if set_initial_pose:
        # Set qpos from real ep's frame 0 state
        qpos_init = state_mdeg_to_rad(real_demo["state_mdeg"][0])
        data.qpos[:8] = qpos_init
        mujoco.mj_forward(model, data)  # propagate kinematics

    sim_qpos = np.zeros((n_steps, 8))
    sim_imu = np.zeros((n_steps, 6))

    for t in range(n_steps):
        a = action_seq[t]
        # Map 7-D action to 8-D actuator ctrl:
        # actuators are [j1..j6, joint7, joint8]; v3 action is [j1..j6, gripper]
        # Use action[6] for joint7, -action[6] for joint8 (mirror)
        ctrl = np.zeros(model.nu)
        ctrl[:6] = a[:6]
        # Gripper: clip to valid range
        grip_cmd = float(a[6])
        ctrl[6] = max(0.0, min(0.035, grip_cmd))
        ctrl[7] = -max(0.0, min(0.035, grip_cmd))
        data.ctrl[:] = ctrl

        # Step physics
        for _ in range(n_phys_per_ctrl):
            mujoco.mj_step(model, data)

        sim_qpos[t] = data.qpos[:8]
        sim_imu[t] = data.sensordata[:6]  # 3 accel + 3 gyro

    return {"qpos": sim_qpos, "imu": sim_imu}


def compare_imu(sim_imu, real_imu, out_path: Path):
    """Time-domain side-by-side comparison of sim vs real IMU."""
    n = min(len(sim_imu), len(real_imu))
    sim = sim_imu[:n]
    real = real_imu[:n]
    t = np.arange(n) * CONTROL_DT

    fig, axes = plt.subplots(6, 1, figsize=(13, 12), sharex=True)
    names = ["ax", "ay", "az", "gx", "gy", "gz"]
    for i in range(6):
        ax = axes[i]
        ax.plot(t, real[:, i], color="#1f77b4", lw=0.8, label="real", alpha=0.85)
        ax.plot(t, sim[:, i], color="#d62728", lw=0.8, label="sim",  alpha=0.85)
        ax.set_ylabel(names[i])
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=9)
            ax.set_title("IMU: sim (red) vs real (blue) — clean replay, no noise")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def compare_psd(sim_imu, real_imu, out_path: Path):
    n = min(len(sim_imu), len(real_imu))
    sim = sim_imu[:n]
    real = real_imu[:n]

    # Remove DC per channel for each signal
    sim_ac = sim - sim.mean(axis=0, keepdims=True)
    real_ac = real - real.mean(axis=0, keepdims=True)

    # Hann window
    w = np.hanning(n)[:, None]
    sim_w = sim_ac * w
    real_w = real_ac * w
    sim_X = np.fft.rfft(sim_w, axis=0)
    real_X = np.fft.rfft(real_w, axis=0)
    sim_P = np.abs(sim_X) ** 2
    real_P = np.abs(real_X) ** 2
    freqs = np.fft.rfftfreq(n, d=CONTROL_DT)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    # Accel PSD
    for i, name in enumerate(["ax", "ay", "az"]):
        axes[0].semilogy(freqs, real_P[:, i] + 1e-12,
                          color="#1f77b4", alpha=0.5, lw=1, label=f"real {name}")
        axes[0].semilogy(freqs, sim_P[:, i] + 1e-12,
                          color="#d62728", alpha=0.5, lw=1, label=f"sim {name}")
    axes[0].set_title("Accel PSD: sim (red) vs real (blue)")
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("power")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim(0, CONTROL_HZ / 2)

    # Gyro PSD
    for i, name in enumerate(["gx", "gy", "gz"]):
        axes[1].semilogy(freqs, real_P[:, 3 + i] + 1e-12,
                          color="#1f77b4", alpha=0.5, lw=1, label=f"real {name}")
        axes[1].semilogy(freqs, sim_P[:, 3 + i] + 1e-12,
                          color="#d62728", alpha=0.5, lw=1, label=f"sim {name}")
    axes[1].set_title("Gyro PSD: sim (red) vs real (blue)")
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("power")
    axes[1].legend(loc="upper right", fontsize=8, ncol=2)
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(0, CONTROL_HZ / 2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def compare_state(sim_qpos, real_state_mdeg, real_action, out_path: Path):
    """Joint trajectory: sim qpos vs real state, with action overlay."""
    n = min(len(sim_qpos), len(real_state_mdeg))
    real_rad = real_state_mdeg[:n, :6] * MDEG_TO_RAD

    fig, axes = plt.subplots(6, 1, figsize=(13, 12), sharex=True)
    t = np.arange(n) * CONTROL_DT
    for j in range(6):
        ax = axes[j]
        ax.plot(t, real_action[:n, j], color="black", lw=0.8, ls=":", label="action (cmd)")
        ax.plot(t, real_rad[:, j], color="#1f77b4", lw=0.8, label="real state", alpha=0.85)
        ax.plot(t, sim_qpos[:n, j], color="#d62728", lw=0.8, label="sim qpos", alpha=0.85)
        ax.set_ylabel(f"j{j+1} (rad)")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(loc="upper right", fontsize=9)
            ax.set_title("Joint trajectory: sim qpos (red) vs real state (blue) vs command (black dotted)")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def compute_metrics(sim_imu, real_imu):
    n = min(len(sim_imu), len(real_imu))
    sim = sim_imu[:n]
    real = real_imu[:n]

    metrics = {"n_frames": n}
    names = ["ax", "ay", "az", "gx", "gy", "gz"]
    for i, name in enumerate(names):
        s = sim[:, i]
        r = real[:, i]
        corr = float(np.corrcoef(s, r)[0, 1]) if s.std() > 1e-9 and r.std() > 1e-9 else 0.0
        rms_err = float(np.sqrt(((s - r) ** 2).mean()))
        metrics[name] = {
            "correlation": corr,
            "rms_error": rms_err,
            "sim_mean":  float(s.mean()), "sim_std":  float(s.std()),
            "real_mean": float(r.mean()), "real_std": float(r.std()),
        }
    return metrics


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[load] {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # Replay a handful of representative episodes
    ep_indices = [0, 7, 15, 23, 29]
    all_metrics = {}

    for ep_idx in ep_indices:
        print(f"\n[ep {ep_idx}]")
        real = load_real_demo(ep_idx)
        print(f"  real demo length: {len(real['action'])} frames")

        sim = replay_in_sim(model, data, real, set_initial_pose=True)

        compare_state(sim["qpos"], real["state_mdeg"], real["action"],
                       OUT_DIR / f"ep{ep_idx:02d}_state_vs_action.png")
        compare_imu(sim["imu"], real["imu"],
                     OUT_DIR / f"ep{ep_idx:02d}_imu_compare.png")
        compare_psd(sim["imu"], real["imu"],
                     OUT_DIR / f"ep{ep_idx:02d}_psd_compare.png")
        m = compute_metrics(sim["imu"], real["imu"])
        (OUT_DIR / f"ep{ep_idx:02d}_metrics.json").write_text(json.dumps(m, indent=2))
        all_metrics[f"ep{ep_idx:02d}"] = m

        # Print summary
        print(f"  IMU correlation per channel:")
        for ch in ["ax", "ay", "az", "gx", "gy", "gz"]:
            print(f"    {ch}: corr={m[ch]['correlation']:+.3f}  "
                  f"rms_err={m[ch]['rms_error']:.3f}  "
                  f"sim_std={m[ch]['sim_std']:.3f}  real_std={m[ch]['real_std']:.3f}")

    (OUT_DIR / "summary_metrics.json").write_text(json.dumps(all_metrics, indent=2))

    # Cross-ep average correlation per channel
    print("\n========== Cross-episode summary ==========")
    print("Average IMU correlation across {} eps:".format(len(ep_indices)))
    for ch in ["ax", "ay", "az", "gx", "gy", "gz"]:
        corrs = [all_metrics[k][ch]["correlation"] for k in all_metrics]
        avg = np.mean(corrs)
        print(f"  {ch}: mean corr = {avg:+.3f}  (range [{min(corrs):+.3f}, {max(corrs):+.3f}])")
    print(f"\n[done] outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
