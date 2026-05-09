#!/usr/bin/env python3
"""P1-v2 真机 4 方对比 (multi-seed aggregated)。

对比：Vanilla / α-imu-only / SAH-only (3 seeds) / SAH+TE (3 seeds)
指标：Accel/Gyro RMS, >5Hz 能量比, action jerk, action HF
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_imu_action(ds_root):
    pqs = sorted(ds_root.glob("data/chunk-*/episode_*.parquet"))
    accel, gyro, action = [], [], []
    for pq in pqs:
        df = pd.read_parquet(pq)
        imu = np.stack(df["observation.imu_real"].to_numpy()).astype(np.float32)
        a = np.stack(df["action"].to_numpy()).astype(np.float32)
        accel.append(imu[:, :3]); gyro.append(imu[:, 3:]); action.append(a)
    return (np.concatenate(accel), np.concatenate(gyro),
            np.concatenate(action), len(pqs))


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


def jerk_rms(action, dt=1.0/30.0):
    v = np.diff(action, axis=0) / dt
    a = np.diff(v, axis=0) / dt
    j = np.diff(a, axis=0) / dt
    return float(np.sqrt((j ** 2).mean()))


def analyze_one(ds_root, label):
    accel, gyro, action, n_eps = load_imu_action(ds_root)
    return {
        "label": label,
        "n_episodes": n_eps,
        "accel_rms":         rms_total(accel),
        "gyro_rms":          rms_total(gyro),
        "accel_hf5_pct":     hf_ratio(accel, 30.0) * 100,
        "gyro_hf5_pct":      hf_ratio(gyro, 30.0) * 100,
        "action_jerk_rms":   jerk_rms(action),
        "action_hf5_pct":    hf_ratio(action, 30.0) * 100,
    }


def aggregate_seeds(records):
    keys = ["accel_rms", "gyro_rms", "accel_hf5_pct",
            "gyro_hf5_pct", "action_jerk_rms", "action_hf5_pct"]
    out = {"n_seeds": len(records)}
    for k in keys:
        vals = [r[k] for r in records]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                  "values": vals}
    return out


def main():
    ap = argparse.ArgumentParser()
    base = Path.home() / ".cache/huggingface/lerobot/local"
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "Vanilla":     [("vanilla", "eval_v2_vanilla")],
        "α-imu-only":  [("imu_only", "eval_v2_imu_only")],
        "SAH-only":    [(f"sah_only_seed{s}", f"eval_v2_sah_only_seed{s}") for s in (1000, 2000, 3000)],
        "SAH+TE":      [(f"sah_te_seed{s}",   f"eval_v2_sah_te_seed{s}")   for s in (1000, 2000, 3000)],
    }
    colors = {"Vanilla": "C7", "α-imu-only": "C0", "SAH-only": "C2", "SAH+TE": "C3"}

    all_records = {}
    aggregated = {}
    for label, runs in sources.items():
        records = [analyze_one(base / repo, run) for run, repo in runs]
        all_records[label] = records
        aggregated[label] = aggregate_seeds(records)
        for r in records:
            print(f"[{label:14s} | {r['label']:25s}]  "
                  f"a_rms={r['accel_rms']:.3f}  g_rms={r['gyro_rms']:.3f}  "
                  f"a_hf={r['accel_hf5_pct']:.2f}%  g_hf={r['gyro_hf5_pct']:.2f}%  "
                  f"jerk={r['action_jerk_rms']:.1f}  "
                  f"act_hf={r['action_hf5_pct']:.4f}%")

    # ===== Comparison table =====
    print("\n" + "=" * 100)
    print(f"{'metric':28s} | "
          f"{'Vanilla':>15s} | "
          f"{'α-imu-only':>15s} | "
          f"{'SAH-only (μ±σ)':>20s} | "
          f"{'SAH+TE (μ±σ)':>20s}")
    print("-" * 100)

    metrics = [("Accel RMS (m/s²)",  "accel_rms",       "{:.3f}"),
               ("Gyro RMS (rad/s)",  "gyro_rms",        "{:.4f}"),
               ("Accel >5Hz %",      "accel_hf5_pct",   "{:.2f}%"),
               ("Gyro >5Hz %",       "gyro_hf5_pct",    "{:.2f}%"),
               ("Action jerk RMS",   "action_jerk_rms", "{:.1f}"),
               ("Action >5Hz %",     "action_hf5_pct",  "{:.4f}%")]
    for name, key, fmt in metrics:
        v = aggregated["Vanilla"][key]["mean"]
        i = aggregated["α-imu-only"][key]["mean"]
        s_only = aggregated["SAH-only"][key]
        s_te = aggregated["SAH+TE"][key]
        print(f"{name:28s} | {fmt.format(v):>15s} | {fmt.format(i):>15s} | "
              f"{fmt.format(s_only['mean']) + ' ± ' + fmt.format(s_only['std']):>20s} | "
              f"{fmt.format(s_te['mean']) + ' ± ' + fmt.format(s_te['std']):>20s}")
    print("=" * 100)

    # ===== Improvement vs Vanilla =====
    print("\n=== Improvement over Vanilla (negative = better) ===")
    for label in ["α-imu-only", "SAH-only", "SAH+TE"]:
        for name, key, _ in metrics:
            v = aggregated["Vanilla"][key]["mean"]
            target = aggregated[label][key]["mean"]
            change = (target - v) / max(abs(v), 1e-9) * 100
            sign = "↓" if change < 0 else "↑"
            print(f"  {label:14s} {name:18s}: {sign}{abs(change):6.1f}%")
        print()

    # ===== Plots =====
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    plot_specs = [
        ("Accel RMS (m/s²)",   "accel_rms"),
        ("Gyro RMS (rad/s)",   "gyro_rms"),
        ("Accel >5Hz %",       "accel_hf5_pct"),
        ("Gyro >5Hz %",        "gyro_hf5_pct"),
        ("Action jerk RMS",    "action_jerk_rms"),
        ("Action >5Hz %",      "action_hf5_pct"),
    ]
    labels_order = ["Vanilla", "α-imu-only", "SAH-only", "SAH+TE"]
    for ax, (title, key) in zip(axes.flat, plot_specs):
        means = [aggregated[l][key]["mean"] for l in labels_order]
        stds  = [aggregated[l][key]["std"]  for l in labels_order]
        bars = ax.bar(labels_order, means, yerr=stds, capsize=4,
                      color=[colors[l] for l in labels_order], alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=15, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{v:.3g}', ha='center', va='bottom', fontsize=8)
    fig.suptitle("P1-v2 four-way real-robot comparison "
                 "(multi-seed: SAH-only & SAH+TE error bars across 3 seeds)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "p1v2_realrobot_4way.png", dpi=130)
    plt.close(fig)

    # PSD plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label, runs in sources.items():
        # average PSD across seeds for SAH variants
        psd_accel_runs = []
        psd_gyro_runs = []
        for _, repo in runs:
            accel, gyro, _, _ = load_imu_action(base / repo)
            for x_arr, store in [(accel, psd_accel_runs), (gyro, psd_gyro_runs)]:
                x_ac = x_arr - x_arr.mean(axis=0, keepdims=True)
                N = x_ac.shape[0]
                w = np.hanning(N)[:, None]
                X = np.fft.rfft(x_ac * w, axis=0)
                p = (np.abs(X) ** 2).sum(axis=1)
                f = np.fft.rfftfreq(N, d=1.0/30.0)
                store.append((f, p))
        # use the longest (or first) frequency grid
        for ax, runs_list, name in [(axes[0], psd_accel_runs, "Accel"),
                                     (axes[1], psd_gyro_runs, "Gyro")]:
            # average spectra (use first f grid as reference if matching length)
            if len(runs_list) > 1:
                # find common min length
                min_len = min(len(p) for _, p in runs_list)
                ps = np.stack([p[:min_len] for _, p in runs_list])
                avg_p = ps.mean(axis=0)
                f = runs_list[0][0][:min_len]
            else:
                f, avg_p = runs_list[0]
            ax.semilogy(f, avg_p + 1e-18, color=colors[label], label=label, alpha=0.85)
            ax.set_title(f"{name} PSD (summed xyz, multi-seed avg)")
            ax.set_xlabel("frequency [Hz]")
            ax.set_ylabel("power (log)")
            ax.axvline(5.0, color="red", linestyle=":", alpha=0.5)
            ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "p1v2_realrobot_psd.png", dpi=130)
    plt.close(fig)

    out = {
        "aggregated": aggregated,
        "all_records": {k: v for k, v in all_records.items()},
    }
    json.dump(out, open(out_dir / "p1v2_realrobot_compare.json", "w"), indent=2)
    print(f"\nsaved -> {out_dir}")


if __name__ == "__main__":
    main()
