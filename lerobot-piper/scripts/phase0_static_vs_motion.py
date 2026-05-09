#!/usr/bin/env python3
"""Phase 0 预实验：静态 IMU vs 真机 rollout IMU 对比。

核心问题：真机振动 RMS 里，**固有噪声**（电机/结构/传感器）占多少？
若占 >70% → 动作 HF 再怎么降也没用（SAH 收益封顶 <30%）
若占 <30% → 动作 HF 是振动主来源，SAH 值得做

数据源：
  1. tsnd_static.npz       机械臂静止 (10 s @ 100 Hz)
  2. eval_vanilla_v1        vanilla ACT 真机 rollout (5 ep × 20s @ 30 Hz)
  3. eval_imu_only_v1       α-imu-only 真机 rollout (5 ep × 20s @ 30 Hz)
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_static_npz(path: Path):
    """返回 (accel (N,3), gyro (N,3), fs)"""
    d = np.load(path)
    n = len(d["device_ms"])
    fs = 1000.0 / d["interval_ms"].item()
    return d["accel"].astype(np.float32), d["gyro"].astype(np.float32), fs


def load_rollout_imu(ds_root: Path):
    """从 LeRobot v2 dataset 读所有 episode 的 observation.imu_real。"""
    parquets = sorted(ds_root.glob("data/chunk-*/episode_*.parquet"))
    if not parquets:
        raise RuntimeError(f"no parquet under {ds_root}")
    accel_all, gyro_all = [], []
    for pq in parquets:
        df = pd.read_parquet(pq)
        imu = np.stack(df["observation.imu_real"].to_numpy()).astype(np.float32)
        accel_all.append(imu[:, :3])
        gyro_all.append(imu[:, 3:])
    return np.concatenate(accel_all), np.concatenate(gyro_all), 30.0


def rms_per_channel(x):
    """x: (N, D). Returns RMS of AC part (mean removed)."""
    x_ac = x - x.mean(axis=0, keepdims=True)
    return np.sqrt((x_ac ** 2).mean(axis=0))


def rms_magnitude(x):
    """|x|^2 = sum of ch^2. Returns total sqrt(mean(|x|^2))."""
    x_ac = x - x.mean(axis=0, keepdims=True)
    return float(np.sqrt((x_ac ** 2).sum(axis=1).mean()))


def psd(x, fs):
    """per-channel PSD (one-sided). Returns freqs, PSD (F, D)."""
    N, D = x.shape
    x_ac = x - x.mean(axis=0, keepdims=True)
    win = np.hanning(N)[:, None]
    X = np.fft.rfft(x_ac * win, axis=0)
    power = np.abs(X) ** 2
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    return freqs, power


def hf_ratio(freqs, power, cutoff=5.0):
    mask = freqs > cutoff
    total = power.sum(axis=0)
    return power[mask].sum(axis=0) / (total + 1e-30)


def analyze(label, accel, gyro, fs):
    accel_rms = rms_magnitude(accel)
    gyro_rms = rms_magnitude(gyro)
    accel_per_ch = rms_per_channel(accel)
    gyro_per_ch = rms_per_channel(gyro)

    a_f, a_p = psd(accel, fs)
    g_f, g_p = psd(gyro, fs)
    a_hf = hf_ratio(a_f, a_p, 5.0)
    g_hf = hf_ratio(g_f, g_p, 5.0)

    return {
        "label": label,
        "n_samples": int(accel.shape[0]),
        "duration_s": float(accel.shape[0] / fs),
        "fs": fs,
        "accel_rms_total": accel_rms,
        "gyro_rms_total": gyro_rms,
        "accel_rms_per_ch": accel_per_ch.tolist(),
        "gyro_rms_per_ch": gyro_per_ch.tolist(),
        "accel_mean": accel.mean(axis=0).tolist(),  # shows gravity vector
        "gyro_mean": gyro.mean(axis=0).tolist(),
        "accel_hf5_ratio_per_ch": a_hf.tolist(),
        "gyro_hf5_ratio_per_ch": g_hf.tolist(),
        "accel_hf5_mean": float(a_hf.mean()),
        "gyro_hf5_mean": float(g_hf.mean()),
        "_psd_accel_freqs": a_f,
        "_psd_accel": a_p,
        "_psd_gyro_freqs": g_f,
        "_psd_gyro": g_p,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--static-npz",
        default="/home/enolab-guo/claude-test-2/lerobot-piper/scripts/tsnd_real/tsnd_static.npz",
    )
    ap.add_argument(
        "--vanilla-ds",
        default=str(Path.home() / ".cache/huggingface/lerobot/local/eval_vanilla_v1"),
    )
    ap.add_argument(
        "--alpha-ds",
        default=str(Path.home() / ".cache/huggingface/lerobot/local/eval_imu_only_v1"),
    )
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] static")
    a_s, g_s, fs_s = load_static_npz(Path(args.static_npz))
    print(f"   {a_s.shape[0]} samples @ {fs_s} Hz = {a_s.shape[0]/fs_s:.1f} s")

    print("[load] vanilla rollout")
    a_v, g_v, fs_v = load_rollout_imu(Path(args.vanilla_ds))
    print(f"   {a_v.shape[0]} samples @ {fs_v} Hz = {a_v.shape[0]/fs_v:.1f} s")

    print("[load] alpha-imu-only rollout")
    a_a, g_a, fs_a = load_rollout_imu(Path(args.alpha_ds))
    print(f"   {a_a.shape[0]} samples @ {fs_a} Hz = {a_a.shape[0]/fs_a:.1f} s")

    res_s = analyze("Static (no motion)", a_s, g_s, fs_s)
    res_v = analyze("Vanilla ACT rollout", a_v, g_v, fs_v)
    res_a = analyze("α-imu-only rollout", a_a, g_a, fs_a)

    # ===== the critical ratios =====
    accel_static_over_motion = res_s["accel_rms_total"] / res_v["accel_rms_total"]
    gyro_static_over_motion = res_s["gyro_rms_total"] / res_v["gyro_rms_total"]

    print("\n" + "=" * 86)
    print(f"{'Source':<28s}  {'accel RMS':>11s}  {'gyro RMS':>11s}  {'>5Hz acc':>10s}  {'>5Hz gyr':>10s}")
    print("-" * 86)
    for r in (res_s, res_v, res_a):
        print(f"{r['label']:<28s}  {r['accel_rms_total']:>10.4f}  "
              f"{r['gyro_rms_total']:>10.4f}  "
              f"{r['accel_hf5_mean']*100:>9.3f}%  "
              f"{r['gyro_hf5_mean']*100:>9.3f}%")
    print("=" * 86)

    print(f"\n>>> static / vanilla_motion  accel RMS ratio = {accel_static_over_motion*100:.1f}%")
    print(f">>> static / vanilla_motion  gyro  RMS ratio = {gyro_static_over_motion*100:.1f}%")

    # interpretation
    head = max(accel_static_over_motion, gyro_static_over_motion) * 100
    print("\n" + "=" * 86)
    if head > 70:
        verdict = f"⚠️  FLOOR-DOMINATED ({head:.0f}%): 固有噪声占主导，SAH 对真机振动收益上限 < {100-head:.0f}%"
    elif head > 30:
        verdict = f"🟡 MIXED ({head:.0f}%): 固有噪声和动作 HF 共享来源，SAH 可贡献 ~{100-head:.0f}%"
    else:
        verdict = f"✅ MOTION-DOMINATED ({head:.0f}%): 动作 HF 是振动主来源，SAH 可贡献 ~{100-head:.0f}% 改善"
    print(verdict)
    print("=" * 86)

    # ===== plot: overlaid PSD (accel + gyro, summed power across channels) =====
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for r, col in zip((res_s, res_v, res_a), ("k", "C3", "C0")):
        a_f, a_p = r["_psd_accel_freqs"], r["_psd_accel"]
        axes[0].semilogy(a_f, a_p.sum(axis=1) + 1e-18, color=col,
                         label=f"{r['label']} (RMS={r['accel_rms_total']:.3f})")
        g_f, g_p = r["_psd_gyro_freqs"], r["_psd_gyro"]
        axes[1].semilogy(g_f, g_p.sum(axis=1) + 1e-18, color=col,
                         label=f"{r['label']} (RMS={r['gyro_rms_total']:.3f})")

    for ax, title, xlabel in zip(
        axes,
        ["Accelerometer PSD (summed across x/y/z)", "Gyroscope PSD (summed across x/y/z)"],
        ["frequency [Hz]"] * 2,
    ):
        ax.axvline(5.0, color="red", linestyle=":", alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("power (log)")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Static vs real-robot rollout IMU spectra\n"
                 "(overlap of static & motion at high freq → inherent noise floor)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "static_vs_motion_psd.png", dpi=130)
    plt.close(fig)

    # ===== RMS bar chart =====
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    labels = [r["label"] for r in (res_s, res_v, res_a)]
    accel_vals = [r["accel_rms_total"] for r in (res_s, res_v, res_a)]
    gyro_vals = [r["gyro_rms_total"] for r in (res_s, res_v, res_a)]
    colors = ["gray", "C3", "C0"]
    axes[0].bar(labels, accel_vals, color=colors, alpha=0.85)
    axes[0].set_ylabel("Accel RMS (m/s²)")
    axes[0].set_title(f"Accel RMS: static/motion = {accel_static_over_motion*100:.1f}%")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(labels, gyro_vals, color=colors, alpha=0.85)
    axes[1].set_ylabel("Gyro RMS (rad/s)")
    axes[1].set_title(f"Gyro RMS: static/motion = {gyro_static_over_motion*100:.1f}%")
    axes[1].grid(axis="y", alpha=0.3)
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out_dir / "static_vs_motion_rms.png", dpi=130)
    plt.close(fig)

    # save JSON (drop internal arrays)
    def dump_ready(r):
        return {k: v for k, v in r.items() if not k.startswith("_")}
    out = {
        "static": dump_ready(res_s),
        "vanilla_motion": dump_ready(res_v),
        "alpha_motion": dump_ready(res_a),
        "ratios": {
            "accel_static_over_motion": accel_static_over_motion,
            "gyro_static_over_motion": gyro_static_over_motion,
            "verdict": verdict,
        },
    }
    with open(out_dir / "static_vs_motion_report.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nsaved -> {out_dir}")


if __name__ == "__main__":
    main()
