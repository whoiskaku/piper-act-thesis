#!/usr/bin/env python3
"""Stage 0 — Noise calibration from v3 demo data.

Per RESEARCH_PLAN_v4 §5 Stage 0, with revisions after first-pass diagnostics
(see Stage 0 review). Key principle:

   v3 demo dataset CAN support:
     - gyro static noise (per channel)
     - active-joint velocity distribution
     - demo jerk RMS baseline (for Phase 3 target)
     - low-frequency tracking / task motion scale
   v3 demo dataset CANNOT support:
     - true accel sensor noise (contaminated by pose-dependent gravity projection)
     - 30-100 Hz mechanical vibration (Nyquist = 15 Hz at 30 fps)
     - j4/j5/j6 dynamics (these joints are inactive in v3 pick-place task)

Outputs:
  outputs/sim/noise_params.json   — separated calibrated + heuristic + uncalibrated
  outputs/sim/noise_analysis.png  — diagnostic plots
  outputs/sim/action_stats.json   — action statistics for sim trajectory design
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEMO_DIR_DEFAULT = Path.home() / ".cache/huggingface/lerobot/local/piper_pick_place_imu_v3"
OUT_DIR_DEFAULT = Path(__file__).resolve().parents[1] / "outputs/sim"

FPS = 30
DT = 1.0 / FPS

# Per-joint action gripper convention:
N_JOINTS = 6
N_IMU = 6
N_ACCEL = 3
N_GYRO = 3
ACCEL_NAMES = ["ax", "ay", "az"]
GYRO_NAMES = ["gx", "gy", "gz"]

# Thresholds for segment classification
STATIONARY_VEL_THRESHOLD = 0.05   # rad/s in action space — stationary if commanded velocity below this
MOTION_VEL_THRESHOLD = 0.3        # rad/s — actively moving if above this


def load_episode(parquet_path: Path):
    df = pd.read_parquet(parquet_path)
    return {
        "action": np.stack(df["action"].to_numpy()).astype(np.float32),       # (T, 7)
        "state":  np.stack(df["observation.state"].to_numpy()).astype(np.float32),
        "imu":    np.stack(df["observation.imu_real"].to_numpy()).astype(np.float32),
        "ts":     df["timestamp"].to_numpy().astype(np.float64),
        "name":   parquet_path.stem,
    }


def load_all_demos(demo_dir: Path):
    parquets = sorted(demo_dir.glob("data/chunk-*/episode_*.parquet"))
    eps = [load_episode(p) for p in parquets]
    print(f"  loaded {len(eps)} episodes, total {sum(len(e['action']) for e in eps)} frames")
    return eps


def compute_action_velocity(action: np.ndarray) -> np.ndarray:
    """Per-step joint velocity from action sequence. shape (T, 7)."""
    # diff along time, prepend zeros for first frame
    v = np.diff(action, axis=0, prepend=action[:1]) / DT
    return v


def classify_segments(velocity_norm: np.ndarray) -> dict:
    """Classify each frame as stationary / motion based on velocity magnitude."""
    stationary_mask = velocity_norm < STATIONARY_VEL_THRESHOLD
    motion_mask = velocity_norm > MOTION_VEL_THRESHOLD
    return {
        "stationary_mask": stationary_mask,
        "motion_mask": motion_mask,
        "stationary_frac": float(stationary_mask.mean()),
        "motion_frac": float(motion_mask.mean()),
    }


# ─── (1) IMU stationary statistics ──────────────────────────────────
def calibrate_imu_stationary(eps):
    """Compute per-channel IMU std from stationary segments.

    IMPORTANT: For accel, this is NOT true sensor noise — it includes
    pose-dependent gravity projection, settling motion, and residual low-freq
    motion. For gyro, this is closer to true sensor noise (gravity-insensitive).

    To get true accel sensor noise, do a separate calibration:
    - Lift IMU off arm, place on rigid stationary surface
    - Record 1-3 minutes at fixed pose
    - Compute std → true sensor noise
    """
    # Per-episode normalization: subtract per-ep mean to remove gravity bias differences
    # between episodes (since each ep starts at slightly different home pose)
    stationary_imu_per_ep_centered = []
    stationary_imu_raw = []

    for ep in eps:
        action_vel = compute_action_velocity(ep["action"])
        vel_norm = np.linalg.norm(action_vel[:, :N_JOINTS], axis=1)
        seg = classify_segments(vel_norm)
        if seg["stationary_mask"].sum() > 0:
            ep_stationary_imu = ep["imu"][seg["stationary_mask"]]
            # Center each ep's stationary IMU separately (removes pose-specific gravity)
            stationary_imu_per_ep_centered.append(
                ep_stationary_imu - ep_stationary_imu.mean(axis=0, keepdims=True)
            )
            stationary_imu_raw.append(ep_stationary_imu)

    if not stationary_imu_raw:
        print("  WARN: no stationary segments found")
        return None

    raw_all = np.concatenate(stationary_imu_raw, axis=0)
    centered_all = np.concatenate(stationary_imu_per_ep_centered, axis=0)

    raw_std = raw_all.std(axis=0)
    raw_mean = raw_all.mean(axis=0)
    centered_std = centered_all.std(axis=0)

    print(f"  total stationary frames: {raw_all.shape[0]}")
    print(f"  raw std (gravity-contaminated):")
    for i, n in enumerate(ACCEL_NAMES + GYRO_NAMES):
        print(f"    {n}: raw_std={raw_std[i]:.4f}  ep-centered_std={centered_std[i]:.4f}")

    return {
        "n_stationary_frames": int(raw_all.shape[0]),
        "raw_mean": {n: float(raw_mean[i]) for i, n in enumerate(ACCEL_NAMES + GYRO_NAMES)},
        "raw_std": {n: float(raw_std[i]) for i, n in enumerate(ACCEL_NAMES + GYRO_NAMES)},
        "ep_centered_std": {n: float(centered_std[i]) for i, n in enumerate(ACCEL_NAMES + GYRO_NAMES)},
    }


# ─── (2) Vibration peaks via FFT on motion segments ──────────────────
def calibrate_vibration_peaks(eps, n_top=5):
    """Find dominant vibration frequencies in IMU during motion segments."""
    motion_imu_chunks = []
    for ep in eps:
        action_vel = compute_action_velocity(ep["action"])
        vel_norm = np.linalg.norm(action_vel[:, :N_JOINTS], axis=1)
        seg = classify_segments(vel_norm)
        if seg["motion_mask"].sum() > 30:  # at least 1 second
            # Take continuous motion segment of at least 30 frames
            motion_imu_chunks.append(ep["imu"][seg["motion_mask"]])

    if not motion_imu_chunks:
        print("  WARN: no sustained motion segments found")
        return None

    all_motion = np.concatenate(motion_imu_chunks, axis=0)
    print(f"  total motion frames: {all_motion.shape[0]}")

    # Remove DC (mean per channel)
    motion_ac = all_motion - all_motion.mean(axis=0, keepdims=True)

    # FFT (use a smaller window so frequency resolution is reasonable)
    # Take last full-power-of-2 chunk for FFT
    n_fft = 2 ** int(np.log2(min(motion_ac.shape[0], 4096)))
    motion_truncated = motion_ac[:n_fft]

    # Apply Hann window per channel
    w = np.hanning(n_fft)[:, None]
    X = np.fft.rfft(motion_truncated * w, axis=0)
    P = np.abs(X) ** 2  # (n_freq, 6)
    freqs = np.fft.rfftfreq(n_fft, d=DT)

    # Find top-N peaks in summed power across all channels
    total_power = P.sum(axis=1)
    # Skip DC bin
    total_power[0] = 0

    # Smooth the power spectrum a little before peak picking
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(total_power, distance=3, prominence=total_power.max() * 0.01)

    if len(peaks) == 0:
        print("  WARN: no clear peaks found")
        peak_info = []
    else:
        # Sort by power descending
        peak_powers = total_power[peaks]
        sorted_idx = np.argsort(peak_powers)[::-1][:n_top]
        peak_info = [
            {
                "freq_hz": float(freqs[peaks[i]]),
                "amplitude_total": float(np.sqrt(total_power[peaks[i]])),
                "per_channel_amplitude": {
                    name: float(np.sqrt(P[peaks[i], j]))
                    for j, name in enumerate(ACCEL_NAMES + GYRO_NAMES)
                },
            }
            for i in sorted_idx
        ]

    print(f"  found {len(peak_info)} peaks (top by power):")
    for p in peak_info:
        print(f"    {p['freq_hz']:.1f} Hz  amplitude {p['amplitude_total']:.3f}")

    return {
        "n_motion_frames": int(all_motion.shape[0]),
        "n_fft": int(n_fft),
        "frequency_resolution_hz": float(freqs[1] - freqs[0]),
        "peaks": peak_info,
        "_psd_for_plot": {
            "freqs": freqs.tolist(),
            "total_power_log": np.log10(total_power + 1e-12).tolist(),
        },
    }


# ─── (3) Action smoothness baseline (jerk in demos) ──────────────────
def baseline_action_smoothness(eps):
    """Compute jerk statistics across all demo episodes."""
    jerk_rms_list = []
    for ep in eps:
        a = ep["action"]
        if len(a) < 4:
            continue
        v = np.diff(a, axis=0) / DT
        acc = np.diff(v, axis=0) / DT
        jerk = np.diff(acc, axis=0) / DT
        jerk_rms_per_joint = np.sqrt((jerk ** 2).mean(axis=0))
        jerk_rms_list.append(jerk_rms_per_joint)

    arr = np.stack(jerk_rms_list)  # (n_ep, 7)
    return {
        "n_episodes": int(arr.shape[0]),
        "jerk_rms_mean_per_joint": arr.mean(axis=0).tolist(),
        "jerk_rms_std_per_joint": arr.std(axis=0).tolist(),
        "jerk_rms_overall_mean": float(arr.mean()),
    }


# ─── (4) Action statistics ──────────────────────────────────────────
def action_statistics(eps):
    """Per-joint range and velocity distribution."""
    all_actions = np.concatenate([ep["action"] for ep in eps])
    all_velocities = np.concatenate([compute_action_velocity(ep["action"]) for ep in eps])

    return {
        "n_frames_total": int(all_actions.shape[0]),
        "action_min_per_joint": all_actions.min(axis=0).tolist(),
        "action_max_per_joint": all_actions.max(axis=0).tolist(),
        "action_mean_per_joint": all_actions.mean(axis=0).tolist(),
        "action_std_per_joint": all_actions.std(axis=0).tolist(),
        "velocity_max_per_joint": np.abs(all_velocities).max(axis=0).tolist(),
        "velocity_p99_per_joint": np.percentile(np.abs(all_velocities), 99, axis=0).tolist(),
    }


# Motor noise / overshoot placeholders are now part of the
# heuristic_defaults_for_stage1 block in noise_params (see main()).
# They will be refined by sim2real PSD matching in Stage 1.


# ─── Plots ──────────────────────────────────────────────────────────
def make_diagnostic_plots(imu_noise_info, vibration_info, action_stats, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Stage 0 — Noise Calibration from v3 Demo", fontsize=13, fontweight="bold")

    # (a) IMU sensor noise σ per channel
    ax = axes[0, 0]
    if imu_noise_info:
        names = list(imu_noise_info["channel_std"].keys())
        stds = list(imu_noise_info["channel_std"].values())
        colors = ["#1f77b4"] * 3 + ["#d62728"] * 3
        ax.bar(names, stds, color=colors)
        for i, s in enumerate(stds):
            ax.text(i, s, f"{s:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"IMU sensor noise σ (stationary, n={imu_noise_info['n_stationary_frames']} frames)")
        ax.set_ylabel("std (m/s² for accel, rad/s for gyro)")
        ax.grid(axis="y", alpha=0.3)

    # (b) Vibration PSD
    ax = axes[0, 1]
    if vibration_info and vibration_info.get("_psd_for_plot"):
        psd = vibration_info["_psd_for_plot"]
        freqs = np.array(psd["freqs"])
        log_power = np.array(psd["total_power_log"])
        ax.plot(freqs, log_power, color="#2ca02c", lw=1.2)
        # Mark peaks
        for p in vibration_info["peaks"]:
            ax.axvline(p["freq_hz"], color="red", linestyle=":", alpha=0.7, lw=1)
            ax.text(p["freq_hz"], log_power.max(), f"{p['freq_hz']:.1f}Hz",
                    fontsize=8, ha="left", rotation=90, color="red")
        ax.set_title(f"Vibration PSD (motion segments, n={vibration_info['n_motion_frames']} frames)")
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("log10 power")
        ax.set_xlim(0, FPS / 2)
        ax.grid(alpha=0.3)

    # (c) Action range per joint
    ax = axes[1, 0]
    joint_names = [f"j{i+1}" for i in range(6)] + ["grip"]
    a_min = np.array(action_stats["action_min_per_joint"])
    a_max = np.array(action_stats["action_max_per_joint"])
    a_mean = np.array(action_stats["action_mean_per_joint"])
    x = np.arange(7)
    ax.bar(x, a_max - a_min, bottom=a_min, color="#1f77b4", alpha=0.5, label="range")
    ax.scatter(x, a_mean, color="red", zorder=5, label="mean")
    ax.set_xticks(x); ax.set_xticklabels(joint_names)
    ax.set_title("Action range per joint")
    ax.set_ylabel("value")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # (d) Velocity p99 per joint
    ax = axes[1, 1]
    v_p99 = action_stats["velocity_p99_per_joint"]
    ax.bar(joint_names, v_p99, color="#ff7f0e")
    for i, v in enumerate(v_p99):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Action velocity p99 per joint (rad/s or grip/s)")
    ax.set_ylabel("|velocity|")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path.name}")


# ─── Main ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-dir", type=Path, default=DEMO_DIR_DEFAULT)
    ap.add_argument("--out-dir",  type=Path, default=OUT_DIR_DEFAULT)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.demo_dir}")
    eps = load_all_demos(args.demo_dir)

    print("\n[calib 1] IMU stationary statistics (NOT pure sensor noise for accel)")
    imu_stationary = calibrate_imu_stationary(eps)

    print("\n[calib 2] Vibration peaks (motion segments, FFT)")
    vibration = calibrate_vibration_peaks(eps)

    print("\n[calib 3] Action smoothness baseline (jerk RMS)")
    smoothness_baseline = baseline_action_smoothness(eps)

    print("\n[calib 4] Action statistics")
    act_stats = action_statistics(eps)

    # Motor noise & overshoot: per Stage 0 revision, NOT estimated from v3
    # (state is in encoder counts not radians; no clean way to compare commanded
    # vs achieved). Heuristic defaults provided in noise_params["heuristic_defaults_for_stage1"].

    # Identify active joints (joints with non-trivial velocity in v3)
    v_p99 = np.array(act_stats["velocity_p99_per_joint"])
    active_joints = [i for i in range(7) if v_p99[i] > 0.01]
    uncalibrated_joints = [i for i in range(7) if v_p99[i] <= 0.01]

    # ─── Assemble noise_params.json with strict categorization ────────
    gyro_sigma = (
        {n: imu_stationary["raw_std"][n] for n in GYRO_NAMES}  # gyro raw_std ≈ sensor noise
        if imu_stationary else None
    )
    accel_demo_residual = (
        {n: imu_stationary["raw_std"][n] for n in ACCEL_NAMES}
        if imu_stationary else None
    )
    accel_ep_centered_residual = (
        {n: imu_stationary["ep_centered_std"][n] for n in ACCEL_NAMES}
        if imu_stationary else None
    )

    # Low-freq vibration peaks that ARE observable (<15 Hz) — these are mostly
    # trajectory shape, not motor vibration. Stored but flagged.
    observable_low_freq_peaks = [
        {"freq_hz": p["freq_hz"], "amplitude_per_channel": p["per_channel_amplitude"]}
        for p in (vibration["peaks"] if vibration else [])
        if p["freq_hz"] <= 15.0
    ]

    noise_params = {
        "version": "v4.3-stage0-revised",
        "source_dataset": str(args.demo_dir),
        "fps": FPS,
        "nyquist_hz": FPS / 2,
        "n_episodes_used": len(eps),

        # ════════ CALIBRATED from v3 demo (use directly in sim) ════════
        "calibrated_from_v3": {
            "gyro_sensor_noise_sigma": gyro_sigma,
            "gyro_calibration_note": (
                "From IMU stationary (action vel < 0.05 rad/s) raw std. "
                "Gyro is gravity-insensitive so this is close to true sensor noise."
            ),

            "demo_jerk_rms_baseline": smoothness_baseline,
            "active_joints_indices": active_joints,
            "active_joint_velocity_p99": {
                f"joint_{i+1}": float(v_p99[i]) for i in active_joints
            },
        },

        # ════════ MEASURED but CONTAMINATED — for reference only ════════
        "measured_but_contaminated": {
            "accel_demo_residual_sigma": accel_demo_residual,
            "accel_demo_residual_note": (
                "Raw std of accel during stationary segments (action vel small). "
                "INCLUDES pose-dependent gravity projection, settling, and low-freq "
                "motion residual. DO NOT use as sensor white noise injection."
            ),
            "accel_ep_centered_residual_sigma": accel_ep_centered_residual,
            "accel_ep_centered_note": (
                "After per-episode mean subtraction. Closer to within-ep "
                "fluctuation but still includes settling/residual motion. Better "
                "than raw_std for noise reference, but not pure sensor noise."
            ),

            "observable_low_freq_peaks_hz": observable_low_freq_peaks,
            "low_freq_note": (
                "These are <=15 Hz PSD peaks observed in motion segments. "
                "Mostly reflect trajectory shape (low-freq task motion), "
                "NOT mechanical vibration. Do not interpret as motor vibration model."
            ),
        },

        # ════════ NOT CALIBRATED from v3 — use heuristic in Stage 1 ════════
        "not_calibrated_use_heuristic": {
            "true_accel_sensor_noise_sigma": "needs_static_calibration",
            "true_accel_calibration_recipe": (
                "Lift IMU off arm OR fully restrain arm; record 1-3 min at fixed pose; "
                "compute std → true accel sensor noise. Expected ~0.05-0.1 m/s²."
            ),
            "high_freq_vibration": "not_observable_at_30Hz",
            "high_freq_note": (
                "Real motor mechanical vibration (30-100 Hz) cannot be sampled at "
                "fps=30 (Nyquist 15 Hz). Phase 3 sim should either (A) inject only "
                "<=15 Hz vibration to avoid aliasing, or (B) upgrade IMU logging "
                "rate and re-record. Current paper takes path A."
            ),
            "uncalibrated_joints_indices": uncalibrated_joints,
            "uncalibrated_joint_strategy": (
                "j4/j5/j6 are inactive in v3 pick-place task. Use conservative "
                "defaults (e.g., copy j2/j3 parameters) for sim noise model. "
                "Or run dedicated per-joint calibration script with small sinusoids."
            ),
        },

        # ════════ HEURISTIC defaults for Stage 1 sim (revise after sim2real validation) ════════
        "heuristic_defaults_for_stage1": {
            "accel_sensor_sigma_per_channel": 0.1,
            "accel_sensor_sigma_rationale": "TSND151 datasheet typical, conservative",
            "vibration_sinusoid_freq_hz": 12.0,
            "vibration_sinusoid_amplitude_accel": 0.05,
            "vibration_sinusoid_amplitude_gyro": 0.01,
            "vibration_rationale": (
                "Low-freq vibration heuristic (12 Hz < Nyquist). NOT real "
                "30-100 Hz motor vibration. Adjust after Stage 1 sim2real PSD match."
            ),
            "motor_noise_sigma_per_active_joint_rad": 0.005,
            "motor_noise_rationale": (
                "~1% of typical chunk velocity. Refine in Stage 1 by matching "
                "sim state trajectory to real."
            ),
            "overshoot_coefficient": 0.02,
            "overshoot_velocity_threshold_rad_per_s": 0.5,
            "overshoot_rationale": (
                "Heuristic; refine by matching sim vs real trajectory on fast moves."
            ),
        },
    }

    # action_stats.json for sim trajectory design reference
    action_stats_doc = {
        "smoothness_baseline_demo_jerk_rms": smoothness_baseline,
        "action_statistics": act_stats,
    }

    # Save
    noise_path = args.out_dir / "noise_params.json"
    stats_path = args.out_dir / "action_stats.json"
    plot_path = args.out_dir / "noise_analysis.png"

    noise_path.write_text(json.dumps(noise_params, indent=2))
    stats_path.write_text(json.dumps(action_stats_doc, indent=2))
    print(f"\n[save] {noise_path}")
    print(f"[save] {stats_path}")

    # Plot uses imu_stationary; adapt the plot helper input expectation.
    plot_imu_input = None
    if imu_stationary:
        plot_imu_input = {
            "n_stationary_frames": imu_stationary["n_stationary_frames"],
            "channel_std": imu_stationary["raw_std"],
        }
    make_diagnostic_plots(plot_imu_input, vibration, act_stats, plot_path)

    print("\n========== Summary ==========")
    print("CALIBRATED from v3 (use directly):")
    if imu_stationary:
        print(f"  gyro sensor noise σ (rad/s):")
        for n in GYRO_NAMES:
            print(f"    {n}: {imu_stationary['raw_std'][n]:.4f}")
    print(f"  demo jerk RMS baseline: {smoothness_baseline['jerk_rms_overall_mean']:.3f}")
    print(f"  active joints (v_p99 > 0.01): {active_joints}")
    print(f"  uncalibrated joints: {uncalibrated_joints}")
    print()
    print("CONTAMINATED (do NOT use as sensor noise):")
    if imu_stationary:
        print(f"  accel demo residual σ (m/s², gravity-contaminated):")
        for n in ACCEL_NAMES:
            print(f"    {n}: raw={imu_stationary['raw_std'][n]:.3f}  "
                  f"ep-centered={imu_stationary['ep_centered_std'][n]:.3f}")
    print()
    print("NOT CALIBRATED (use heuristic in Stage 1):")
    print("  - true accel sensor noise → needs static calibration recording")
    print("  - 30-100 Hz mechanical vibration → unobservable at fps=30 (Nyquist 15)")
    print("  - j4/j5/j6 dynamics → use conservative copy from j2/j3")
    print("=============================")


if __name__ == "__main__":
    main()
