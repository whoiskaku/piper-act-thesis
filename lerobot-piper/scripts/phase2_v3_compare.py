#!/usr/bin/env python3
"""2-way comparison plots: Phase2-v3 (RL-finetune) vs C1-v3 (BC starting point).

Reads:
  ~/.cache/huggingface/lerobot/local/eval_C1_v3/      (5 ep deterministic deploy)
  ~/.cache/huggingface/lerobot/local/eval_phase2_v3/  (5 ep deterministic deploy)

Outputs:
  outputs/eval/phase2_v3_vs_C1_v3/bar_chart.png       — 6-metric bar chart, per-episode dots
  outputs/eval/phase2_v3_vs_C1_v3/psd.png             — accel + gyro PSD overlay
  outputs/eval/phase2_v3_vs_C1_v3/per_episode.png     — per-episode jerk + IMU traces
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"C1-v3": "#1f77b4", "Phase2-v3": "#d62728"}


def load_episodes(ds_root: Path):
    """Return list of dicts (one per episode) with action / accel / gyro arrays."""
    eps = []
    for pq in sorted(ds_root.glob("data/chunk-*/episode_*.parquet")):
        df = pd.read_parquet(pq)
        action = np.stack(df["action"].to_numpy()).astype(np.float32)
        imu = np.stack(df["observation.imu_real"].to_numpy()).astype(np.float32)
        eps.append({
            "name": pq.stem,
            "action": action,
            "accel": imu[:, :3],
            "gyro":  imu[:, 3:],
            "frames": len(df),
        })
    return eps


def jerk_rms(action, dt=1.0/30.0):
    v = np.diff(action, axis=0) / dt
    a = np.diff(v, axis=0) / dt
    j = np.diff(a, axis=0) / dt
    return float(np.sqrt((j ** 2).mean()))


def rms_ac(x):
    x_ac = x - x.mean(axis=0, keepdims=True)
    return float(np.sqrt((x_ac ** 2).sum(axis=1).mean()))


def hf_ratio(x, fs=30.0, cutoff=5.0):
    N = x.shape[0]
    x_ac = x - x.mean(axis=0, keepdims=True)
    w = np.hanning(N)[:, None]
    X = np.fft.rfft(x_ac * w, axis=0)
    p = np.abs(X) ** 2
    f = np.fft.rfftfreq(N, d=1.0 / fs)
    return float(p[f > cutoff].sum() / (p.sum() + 1e-30))


def per_episode_metrics(eps):
    rows = []
    for e in eps:
        rows.append({
            "name": e["name"],
            "frames": e["frames"],
            "action_jerk_rms": jerk_rms(e["action"]),
            "action_hf_pct":   hf_ratio(e["action"]) * 100,
            "accel_rms":       rms_ac(e["accel"]),
            "gyro_rms":        rms_ac(e["gyro"]),
            "accel_hf_pct":    hf_ratio(e["accel"]) * 100,
            "gyro_hf_pct":     hf_ratio(e["gyro"]) * 100,
        })
    return rows


def avg_psd(eps, kind):
    """kind in {'accel','gyro'}. Return (freq, mean_power) summed over xyz."""
    all_p, all_f = [], None
    for e in eps:
        x = e[kind]
        x_ac = x - x.mean(axis=0, keepdims=True)
        N = x_ac.shape[0]
        w = np.hanning(N)[:, None]
        X = np.fft.rfft(x_ac * w, axis=0)
        p = (np.abs(X) ** 2).sum(axis=1)
        f = np.fft.rfftfreq(N, d=1.0 / 30.0)
        all_p.append(p)
        all_f = f if all_f is None or len(f) < len(all_f) else all_f
    min_len = min(len(p) for p in all_p)
    P = np.stack([p[:min_len] for p in all_p])
    return all_f[:min_len], P.mean(axis=0)


def bar_chart(rows_c1, rows_p2, out):
    metrics = [
        ("Action jerk RMS",      "action_jerk_rms",  "{:.0f}"),
        ("Action >5Hz %",        "action_hf_pct",    "{:.3f}"),
        ("Accel RMS (m/s²)",     "accel_rms",        "{:.2f}"),
        ("Gyro RMS (rad/s)",     "gyro_rms",         "{:.2f}"),
        ("Accel >5Hz %",         "accel_hf_pct",     "{:.1f}"),
        ("Gyro >5Hz %",          "gyro_hf_pct",      "{:.1f}"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    labels = ["C1-v3", "Phase2-v3"]
    for ax, (title, key, fmt) in zip(axes.flat, metrics):
        v_c1 = [r[key] for r in rows_c1]
        v_p2 = [r[key] for r in rows_p2]
        means = [np.mean(v_c1), np.mean(v_p2)]
        stds  = [np.std(v_c1),  np.std(v_p2)]

        x = np.arange(2)
        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color=[COLORS[l] for l in labels], alpha=0.78)
        # per-episode dots
        ax.scatter([0]*len(v_c1), v_c1, color="black", s=22, zorder=3, alpha=0.7)
        ax.scatter([1]*len(v_p2), v_p2, color="black", s=22, zorder=3, alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    fmt.format(m), ha="center", va="bottom", fontsize=9, fontweight="bold")

        # delta arrow
        delta_pct = (means[1] - means[0]) / max(abs(means[0]), 1e-9) * 100
        sign = "↑" if delta_pct > 0 else "↓"
        color = "red" if delta_pct > 0 else "green"
        ax.text(0.5, 1.04, f"{sign}{abs(delta_pct):.1f}%",
                transform=ax.transAxes, ha="center", fontsize=10,
                color=color, fontweight="bold")

    fig.suptitle("Phase2-v3 (RL fine-tune) vs C1-v3 (BC baseline) — 5 ep deterministic deploy each",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def psd_plot(eps_c1, eps_p2, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for kind, ax in zip(["accel", "gyro"], axes):
        f1, p1 = avg_psd(eps_c1, kind)
        f2, p2 = avg_psd(eps_p2, kind)
        ax.semilogy(f1, p1 + 1e-18, color=COLORS["C1-v3"],   label="C1-v3", lw=1.6)
        ax.semilogy(f2, p2 + 1e-18, color=COLORS["Phase2-v3"], label="Phase2-v3", lw=1.6)
        ax.axvline(5.0, color="red", linestyle=":", alpha=0.5, label="5 Hz")
        ax.set_title(f"{kind.capitalize()} PSD (avg over 5 ep, summed xyz)")
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("power (log)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def per_episode_jerk_plot(eps_c1, eps_p2, out):
    """Plot per-episode commanded action signal (joint 1 as representative) +
    accel norm, top row C1-v3, bottom row Phase2-v3, 5 columns."""
    fig, axes = plt.subplots(2, 5, figsize=(18, 5.5), sharey="row")
    for col in range(5):
        for row, (eps, label) in enumerate([(eps_c1, "C1-v3"), (eps_p2, "Phase2-v3")]):
            ax = axes[row, col]
            e = eps[col]
            t = np.arange(e["frames"]) / 30.0
            accel_norm = np.linalg.norm(e["accel"] - e["accel"].mean(axis=0), axis=1)
            ax.plot(t, accel_norm, color=COLORS[label], lw=0.8, alpha=0.9)
            if col == 0:
                ax.set_ylabel(f"{label}\n|accel - mean|", fontsize=10)
            ax.set_title(f"ep {col}", fontsize=9)
            ax.grid(alpha=0.3)
            if row == 1:
                ax.set_xlabel("time [s]", fontsize=9)
    fig.suptitle("Per-episode IMU accel magnitude (DC-removed) — C1-v3 (top) vs Phase2-v3 (bottom)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1-dataset",
                    default=str(Path.home() / ".cache/huggingface/lerobot/local/eval_C1_v3"))
    ap.add_argument("--phase2-dataset",
                    default=str(Path.home() / ".cache/huggingface/lerobot/local/eval_phase2_v3"))
    ap.add_argument("--output-dir", default="outputs/eval/phase2_v3_vs_C1_v3")
    args = ap.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"[load] C1-v3:    {args.c1_dataset}")
    eps_c1 = load_episodes(Path(args.c1_dataset))
    print(f"[load] Phase2-v3:{args.phase2_dataset}")
    eps_p2 = load_episodes(Path(args.phase2_dataset))
    assert len(eps_c1) == 5 and len(eps_p2) == 5, "expected 5 ep each"

    rows_c1 = per_episode_metrics(eps_c1)
    rows_p2 = per_episode_metrics(eps_p2)

    print("\n--- per-episode (C1-v3) ---")
    for r in rows_c1:
        print(f"  {r['name']}  jerk={r['action_jerk_rms']:7.2f}  "
              f"act_hf={r['action_hf_pct']:.4f}%  accel_rms={r['accel_rms']:.3f}  "
              f"gyro_rms={r['gyro_rms']:.3f}")
    print("\n--- per-episode (Phase2-v3) ---")
    for r in rows_p2:
        print(f"  {r['name']}  jerk={r['action_jerk_rms']:7.2f}  "
              f"act_hf={r['action_hf_pct']:.4f}%  accel_rms={r['accel_rms']:.3f}  "
              f"gyro_rms={r['gyro_rms']:.3f}")

    bar_chart(rows_c1, rows_p2, out / "bar_chart.png")
    psd_plot(eps_c1, eps_p2, out / "psd.png")
    per_episode_jerk_plot(eps_c1, eps_p2, out / "per_episode_accel.png")

    summary = {"C1-v3": rows_c1, "Phase2-v3": rows_p2}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))

    print(f"\nsaved to {out}/")
    print(f"  bar_chart.png")
    print(f"  psd.png")
    print(f"  per_episode_accel.png")
    print(f"  metrics.json")


if __name__ == "__main__":
    main()
