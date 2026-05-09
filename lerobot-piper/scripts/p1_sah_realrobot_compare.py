#!/usr/bin/env python3
"""P1 Stage 6/7：四方真机 IMU 对比。
  Vanilla / α-imu-only / SAH-only / SAH+TE
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_imu_action(ds_root: Path):
    parquets = sorted(ds_root.glob("data/chunk-*/episode_*.parquet"))
    accel, gyro, action = [], [], []
    for pq in parquets:
        df = pd.read_parquet(pq)
        imu = np.stack(df["observation.imu_real"].to_numpy()).astype(np.float32)
        a = np.stack(df["action"].to_numpy()).astype(np.float32)
        accel.append(imu[:, :3])
        gyro.append(imu[:, 3:])
        action.append(a)
    return (np.concatenate(accel), np.concatenate(gyro),
            np.concatenate(action), len(parquets))


def rms_total(x):
    x_ac = x - x.mean(axis=0, keepdims=True)
    return float(np.sqrt((x_ac ** 2).sum(axis=1).mean()))


def hf_ratio(x, fs, cutoff=5.0):
    N = x.shape[0]
    x_ac = x - x.mean(axis=0, keepdims=True)
    w = np.hanning(N)[:, None]
    X = np.fft.rfft(x_ac * w, axis=0)
    p = np.abs(X) ** 2
    f = np.fft.rfftfreq(N, d=1.0 / fs)
    return float(p[f > cutoff].sum() / (p.sum() + 1e-30))


def jerk_rms(action, dt=1.0 / 30.0):
    v = np.diff(action, axis=0) / dt
    a = np.diff(v, axis=0) / dt
    j = np.diff(a, axis=0) / dt
    return float(np.sqrt((j ** 2).mean()))


def analyze(label, ds_root):
    accel, gyro, action, n_eps = load_imu_action(ds_root)
    return {
        "label": label,
        "n_episodes": n_eps,
        "n_frames": int(accel.shape[0]),
        "accel_rms": rms_total(accel),
        "gyro_rms": rms_total(gyro),
        "accel_hf5_pct": hf_ratio(accel, fs=30.0) * 100,
        "gyro_hf5_pct": hf_ratio(gyro, fs=30.0) * 100,
        "action_jerk_rms": jerk_rms(action),
        "action_hf5_pct": hf_ratio(action, fs=30.0) * 100,
    }


def main():
    ap = argparse.ArgumentParser()
    base = Path.home() / ".cache/huggingface/lerobot/local"
    ap.add_argument("--vanilla",   default=str(base / "eval_vanilla_v1"))
    ap.add_argument("--imu-only",  default=str(base / "eval_imu_only_v1"))
    ap.add_argument("--sah-only",  default=str(base / "eval_sah_k5bsp_only_v1"))
    ap.add_argument("--sah-te",    default=str(base / "eval_sah_k5bsp_te_v1"))
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        ("Vanilla",      args.vanilla,   "C7"),
        ("α-imu-only",   args.imu_only,  "C0"),
        ("SAH-only",     args.sah_only,  "C2"),
        ("SAH+TE",       args.sah_te,    "C3"),
    ]
    results = []
    for label, root, _ in sources:
        r = analyze(label, Path(root))
        results.append(r)
        print(f"[{label:15s}]  ep={r['n_episodes']}  "
              f"a_rms={r['accel_rms']:.3f}  g_rms={r['gyro_rms']:.3f}  "
              f"a_hf={r['accel_hf5_pct']:.2f}%  g_hf={r['gyro_hf5_pct']:.2f}%  "
              f"jerk={r['action_jerk_rms']:.1f}  act_hf={r['action_hf5_pct']:.4f}%")

    # Comparison table
    print("\n" + "=" * 96)
    print(f"{'metric':28s} | {'Vanilla':>11s} | {'α-imu-only':>11s} | {'SAH-only':>11s} | {'SAH+TE':>11s}")
    print("-" * 96)
    metrics = [
        ("Accel RMS (m/s²)", "accel_rms", "{:.3f}"),
        ("Gyro RMS (rad/s)", "gyro_rms", "{:.4f}"),
        ("Accel >5Hz %", "accel_hf5_pct", "{:.2f}%"),
        ("Gyro >5Hz %", "gyro_hf5_pct", "{:.2f}%"),
        ("Action jerk RMS", "action_jerk_rms", "{:.1f}"),
        ("Action >5Hz %", "action_hf5_pct", "{:.4f}%"),
    ]
    for name, key, fmt in metrics:
        row = f"{name:28s}"
        for r in results:
            row += f" | {fmt.format(r[key]):>11s}"
        print(row)
    print("=" * 96)

    # Improvement vs vanilla and vs imu-only
    print("\n=== Improvement over Vanilla (negative = better) ===")
    for r in results[1:]:
        for name, key in [("Accel RMS", "accel_rms"), ("Accel HF", "accel_hf5_pct"),
                          ("Gyro RMS", "gyro_rms"), ("Action jerk", "action_jerk_rms")]:
            v = results[0][key]
            change = (r[key] - v) / v * 100
            sign = "↓" if change < 0 else "↑"
            print(f"  {r['label']:15s} {name:15s}: {sign}{abs(change):5.1f}%")
        print()

    # ===== Plots =====
    # 1. Bar charts of all metrics
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    plot_specs = [
        ("Accel RMS (m/s²)", "accel_rms"),
        ("Gyro RMS (rad/s)", "gyro_rms"),
        ("Accel >5Hz %",     "accel_hf5_pct"),
        ("Gyro >5Hz %",      "gyro_hf5_pct"),
        ("Action jerk RMS",  "action_jerk_rms"),
        ("Action >5Hz %",    "action_hf5_pct"),
    ]
    labels = [s[0] for s in sources]
    colors = [s[2] for s in sources]
    for ax, (title, key) in zip(axes.flat, plot_specs):
        vals = [r[key] for r in results]
        bars = ax.bar(labels, vals, color=colors, alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=15, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{v:.3g}', ha='center', va='bottom', fontsize=8)
    fig.suptitle("P1 Stage 7: Real-robot IMU & action smoothness, four-way comparison",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "p1_realrobot_4way.png", dpi=130)
    plt.close(fig)

    # 2. PSD overlay (accel total power)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label, root, color in sources:
        accel, gyro, _, _ = load_imu_action(Path(root))
        for ax, x, name in [(axes[0], accel, "Accel"), (axes[1], gyro, "Gyro")]:
            x_ac = x - x.mean(axis=0, keepdims=True)
            N = x_ac.shape[0]
            w = np.hanning(N)[:, None]
            X = np.fft.rfft(x_ac * w, axis=0)
            p = (np.abs(X) ** 2).sum(axis=1)
            f = np.fft.rfftfreq(N, d=1.0/30.0)
            ax.semilogy(f, p + 1e-18, color=color, label=label, alpha=0.85)
            ax.set_title(f"{name} PSD (summed across xyz)")
            ax.set_xlabel("frequency [Hz]")
            ax.set_ylabel("power (log)")
            ax.axvline(5.0, color="red", linestyle=":", alpha=0.5)
            ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "p1_realrobot_psd.png", dpi=130)
    plt.close(fig)

    # save JSON
    json.dump(results, open(out_dir / "p1_realrobot_compare.json", "w"), indent=2)
    print(f"\nsaved -> {out_dir}")


if __name__ == "__main__":
    main()
