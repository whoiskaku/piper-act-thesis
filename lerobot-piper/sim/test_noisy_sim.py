#!/usr/bin/env python3
"""Stage 1.3 verification — does NoisyPiperSim produce IMU PSD matching real?

Replays a v3 demo through NoisyPiperSim (noise ON) and compares the resulting
noisy IMU PSD against the real v3 IMU PSD. Goal: weak-axis variance should now
be non-zero (so RefineNet later sees realistic noise distribution at train and
deploy time).

Output: outputs/sim/stage1_noisy/
  ep<i>_noisy_imu_compare.png    — clean sim vs noisy sim vs real time-domain
  ep<i>_noisy_psd_compare.png    — PSD overlay
  noisy_metrics.json             — per-channel std and correlation
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json

from replay_demo_validation import (
    MODEL_PATH, CONTROL_HZ, CONTROL_DT,
    load_real_demo, state_mdeg_to_rad,
)
from noisy_piper_sim import NoisyPiperSim, action_7_to_ctrl_8


OUT_DIR = Path(__file__).resolve().parents[1] / "outputs/sim/stage1_noisy"
NOISE_PARAMS = Path(__file__).resolve().parents[1] / "outputs/sim/noise_params.json"


# Backwards-compatible alias (kept for any earlier callers).
def real_action_to_ctrl(real_action_chunk):
    """Map v3 7-dim action to 8-dim actuator ctrl. Wrapper around action_7_to_ctrl_8."""
    return action_7_to_ctrl_8(real_action_chunk)


def compute_psd(x, fs=CONTROL_HZ):
    """Return (freqs, power_per_channel) for a signal x shape (T, C)."""
    n = x.shape[0]
    x_ac = x - x.mean(axis=0, keepdims=True)
    w = np.hanning(n)[:, None]
    X = np.fft.rfft(x_ac * w, axis=0)
    P = np.abs(X) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, P


def plot_imu_time(clean, noisy, real, out_path):
    n = min(len(clean), len(noisy), len(real))
    t = np.arange(n) * CONTROL_DT
    fig, axes = plt.subplots(6, 1, figsize=(13, 12), sharex=True)
    names = ["ax", "ay", "az", "gx", "gy", "gz"]
    for i in range(6):
        ax = axes[i]
        ax.plot(t, real[:n, i], color="#1f77b4", lw=0.8, alpha=0.85, label="real")
        ax.plot(t, clean[:n, i], color="#2ca02c", lw=0.6, alpha=0.6, label="sim clean")
        ax.plot(t, noisy[:n, i], color="#d62728", lw=0.6, alpha=0.7, label="sim noisy")
        ax.set_ylabel(names[i])
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=9)
            ax.set_title("IMU time-domain: real (blue) vs sim clean (green) vs sim noisy (red)")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_imu_psd(clean, noisy, real, out_path):
    n = min(len(clean), len(noisy), len(real))
    f, P_real  = compute_psd(real[:n])
    _, P_clean = compute_psd(clean[:n])
    _, P_noisy = compute_psd(noisy[:n])

    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    for ax_idx, group, slc in [(0, "Accel", slice(0, 3)), (1, "Gyro", slice(3, 6))]:
        ax = axes[ax_idx]
        for i in range(slc.start, slc.stop):
            name = ["ax", "ay", "az", "gx", "gy", "gz"][i]
            ax.semilogy(f, P_real[:, i] + 1e-12,
                        color="#1f77b4", lw=1.0, alpha=0.6, label=f"real {name}")
            ax.semilogy(f, P_clean[:, i] + 1e-12,
                        color="#2ca02c", lw=0.8, alpha=0.5, ls="--")
            ax.semilogy(f, P_noisy[:, i] + 1e-12,
                        color="#d62728", lw=0.8, alpha=0.6)
        ax.set_title(f"{group} PSD: real (blue) vs sim clean (green dashed) vs sim noisy (red)")
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("power")
        ax.set_xlim(0, CONTROL_HZ / 2)
        ax.grid(alpha=0.3)
        if ax_idx == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[load] sim model = {MODEL_PATH}")
    print(f"[load] noise params = {NOISE_PARAMS}")

    ep_indices = [0, 7, 15, 23, 29]
    all_metrics = {}

    sim = NoisyPiperSim(MODEL_PATH, NOISE_PARAMS, seed=42)
    print(f"  active_actuators={sim.active_actuators}, sigma_motor={sim.sigma_motor}")
    print(f"  sigma_accel={sim.sigma_accel}, sigma_gyro={sim.sigma_gyro}")
    print(f"  vib_freq={sim.vib_freq}, amp_accel={sim.vib_amp_accel}, amp_gyro={sim.vib_amp_gyro}")

    for ep_idx in ep_indices:
        print(f"\n[ep {ep_idx}]")
        real = load_real_demo(ep_idx)
        ctrl = real_action_to_ctrl(real["action"])
        T = ctrl.shape[0]
        start_qpos = state_mdeg_to_rad(real["state_mdeg"][0])

        result = sim.execute_action_chunk(start_qpos, ctrl, common_seed=ep_idx)

        plot_imu_time(result["IMU_clean"], result["IMU_noisy"], real["imu"][:T],
                       OUT_DIR / f"ep{ep_idx:02d}_noisy_imu_compare.png")
        plot_imu_psd(result["IMU_clean"], result["IMU_noisy"], real["imu"][:T],
                      OUT_DIR / f"ep{ep_idx:02d}_noisy_psd_compare.png")

        # Per-channel metrics
        n = min(T, len(result["IMU_noisy"]), len(real["imu"]))
        clean = result["IMU_clean"][:n]
        noisy = result["IMU_noisy"][:n]
        rl = real["imu"][:n]
        m = {"n_frames": int(n)}
        names = ["ax", "ay", "az", "gx", "gy", "gz"]
        for i, name in enumerate(names):
            m[name] = {
                "real_std":      float(rl[:, i].std()),
                "sim_clean_std": float(clean[:, i].std()),
                "sim_noisy_std": float(noisy[:, i].std()),
                "corr_clean_real": float(np.corrcoef(clean[:, i], rl[:, i])[0, 1])
                                    if clean[:, i].std() > 1e-9 and rl[:, i].std() > 1e-9 else 0.0,
                "corr_noisy_real": float(np.corrcoef(noisy[:, i], rl[:, i])[0, 1])
                                    if noisy[:, i].std() > 1e-9 and rl[:, i].std() > 1e-9 else 0.0,
            }
        all_metrics[f"ep{ep_idx:02d}"] = m

        # Print compact summary
        print(f"  channel | real_std | clean_std | noisy_std | corr(clean,real) | corr(noisy,real)")
        for name in names:
            v = m[name]
            print(f"  {name:>6}  | {v['real_std']:7.3f}  | {v['sim_clean_std']:8.3f}  "
                  f"| {v['sim_noisy_std']:8.3f}  | {v['corr_clean_real']:+10.3f}     "
                  f"| {v['corr_noisy_real']:+10.3f}")

    (OUT_DIR / "noisy_metrics.json").write_text(json.dumps(all_metrics, indent=2))

    # Cross-ep summary
    print("\n========== Cross-episode noisy-sim summary ==========")
    print(f"{'channel':>8} | mean_clean_std | mean_noisy_std | mean_real_std | "
          f"mean_corr_noisy_vs_real")
    for name in names:
        clean_stds = [all_metrics[k][name]["sim_clean_std"] for k in all_metrics]
        noisy_stds = [all_metrics[k][name]["sim_noisy_std"] for k in all_metrics]
        real_stds  = [all_metrics[k][name]["real_std"]      for k in all_metrics]
        corrs      = [all_metrics[k][name]["corr_noisy_real"] for k in all_metrics]
        print(f"{name:>8} | {np.mean(clean_stds):14.3f} | {np.mean(noisy_stds):14.3f} "
              f"| {np.mean(real_stds):13.3f} | {np.mean(corrs):+22.3f}")

    print(f"\n[done] outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
