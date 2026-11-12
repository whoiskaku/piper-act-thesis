#!/usr/bin/env python3
"""3-way comparison plots (similar to phase2_v3_compare.py) but with gt_IMU
as the third "trace" alongside C1-v3 and Phase2-v3 real IMU.

gt is derived by querying the reward computer at each chunk start of each
C1-v3 episode (k=1 NN), un-normalizing back to raw IMU scale, and
concatenating. This is "the IMU that the Round A reward would have used as
target along C1-v3's trajectory".

Outputs (under outputs/eval/phase2_v3_vs_C1_v3_with_gt/):
  bar_chart_3way.png       — 6-metric bar chart, gt bar shown only for IMU
                              metrics (action metrics don't apply to gt)
  psd_3way.png             — accel + gyro PSD overlay, 3 curves
  per_episode_3way.png     — per-ep IMU accel magnitude, 3 colors per panel
"""
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RC_PKL = "/home/enolab-guo/claude-test-2/lerobot-piper/outputs/eval/d2_imu_reward_computer.pkl"
EVAL_C1 = Path.home() / ".cache/huggingface/lerobot/local/eval_C1_v3"
EVAL_P2 = Path.home() / ".cache/huggingface/lerobot/local/eval_phase2_v3"
OUT     = Path("/home/enolab-guo/claude-test-2/lerobot-piper/outputs/eval/phase2_v3_vs_C1_v3_with_gt")

CHUNK_SIZE = 100
COLORS = {
    "C1-v3":     "#1f77b4",
    "Phase2-v3": "#d62728",
    "gt (Round A target)": "#2ca02c",
}


# ─── data loading ─────────────────────────────────────────────────
def load_episodes(ds_root: Path):
    eps = []
    for pq in sorted(ds_root.glob("data/chunk-*/episode_*.parquet")):
        df = pd.read_parquet(pq)
        eps.append({
            "name":  pq.stem,
            "state": np.stack(df["observation.state"].to_numpy()).astype(np.float32),
            "imu":   np.stack(df["observation.imu_real"].to_numpy()).astype(np.float32),
            "action":np.stack(df["action"].to_numpy()).astype(np.float32),
            "frames": len(df),
        })
    return eps


def query_gt_k1(rc, state_raw: np.ndarray):
    """Return the 1-NN demo's next chunk_size IMU window (still normalized)."""
    q = (state_raw - rc.state_mean) / (rc.state_std + 1e-8)
    _, idxs = rc.kdtree.query(q.reshape(1, -1), k=1)
    return rc.imu_windows[int(idxs[0, 0])]  # (chunk, 6) normalized


def build_gt_trace(rc, ep):
    """Build episode-length gt IMU trace (raw scale) from C1-v3 ep states.

    Walks through chunks of CHUNK_SIZE, queries gt at each chunk start, and
    concatenates. Un-normalizes back to raw IMU scale for fair comparison
    with raw real IMU.
    """
    n_chunks = ep["frames"] // CHUNK_SIZE
    pieces = []
    for ck in range(n_chunks):
        s0 = ep["state"][ck * CHUNK_SIZE]
        gt_norm = query_gt_k1(rc, s0)
        gt_raw = gt_norm * (rc.imu_std + 1e-8) + rc.imu_mean
        pieces.append(gt_raw)
    return np.vstack(pieces)  # (n_chunks * CHUNK_SIZE, 6)


# ─── metrics ──────────────────────────────────────────────────────
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


def per_episode_metrics(eps, has_action=True):
    rows = []
    for e in eps:
        row = {
            "name":          e["name"],
            "frames":        len(e["imu"]) if "imu" in e else 0,
            "accel_rms":     rms_ac(e["imu"][:, :3]),
            "gyro_rms":      rms_ac(e["imu"][:, 3:]),
            "accel_hf_pct":  hf_ratio(e["imu"][:, :3]) * 100,
            "gyro_hf_pct":   hf_ratio(e["imu"][:, 3:]) * 100,
        }
        if has_action and "action" in e:
            row["action_jerk_rms"] = jerk_rms(e["action"])
            row["action_hf_pct"]   = hf_ratio(e["action"]) * 100
        rows.append(row)
    return rows


def avg_psd(eps, which="accel"):
    """which in {'accel', 'gyro'}."""
    all_p, all_f = [], None
    for e in eps:
        x = e["imu"][:, :3] if which == "accel" else e["imu"][:, 3:]
        x_ac = x - x.mean(axis=0, keepdims=True)
        N = x_ac.shape[0]
        w = np.hanning(N)[:, None]
        X = np.fft.rfft(x_ac * w, axis=0)
        p = (np.abs(X) ** 2).sum(axis=1)
        f = np.fft.rfftfreq(N, d=1.0 / 30.0)
        all_p.append(p)
        if all_f is None or len(f) < len(all_f):
            all_f = f
    min_len = min(len(p) for p in all_p)
    P = np.stack([p[:min_len] for p in all_p])
    return all_f[:min_len], P.mean(axis=0)


# ─── plots ────────────────────────────────────────────────────────
def bar_chart(rows_c1, rows_p2, rows_gt, out_path: Path):
    # Only IMU metrics — action metrics removed since gt has no action data.
    metrics = [
        ("Accel RMS (m/s²)",     "accel_rms",        "{:.2f}"),
        ("Gyro RMS (rad/s)",     "gyro_rms",         "{:.2f}"),
        ("Accel >5Hz %",         "accel_hf_pct",     "{:.1f}"),
        ("Gyro >5Hz %",          "gyro_hf_pct",      "{:.1f}"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (title, key, fmt) in zip(axes.flat, metrics):
        labels = ["gt", "C1-v3", "Phase2-v3"]
        colors_used = [COLORS["gt (Round A target)"], COLORS["C1-v3"], COLORS["Phase2-v3"]]
        v_gt = [r[key] for r in rows_gt]
        v_c1 = [r[key] for r in rows_c1]
        v_p2 = [r[key] for r in rows_p2]
        means = [np.mean(v_gt), np.mean(v_c1), np.mean(v_p2)]
        stds  = [np.std(v_gt),  np.std(v_c1),  np.std(v_p2)]
        data_lists = [v_gt, v_c1, v_p2]

        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors_used, alpha=0.82)
        for i, vals in enumerate(data_lists):
            ax.scatter([i] * len(vals), vals, color="black", s=22, zorder=3, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    fmt.format(m), ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")

    fig.suptitle("3-way comparison — gt (Round A reward target) vs C1-v3 (BC) vs Phase2-v3 (RL Round A)\n"
                 "5 ep deterministic each. gt traces derived from C1-v3 ep states (k=1 NN).",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path.name}")


def psd_plot(eps_c1, eps_p2, eps_gt, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for which, ax in zip(["accel", "gyro"], axes):
        for label, eps in [("gt (Round A target)", eps_gt),
                            ("C1-v3", eps_c1),
                            ("Phase2-v3", eps_p2)]:
            f, p = avg_psd(eps, which)
            ax.semilogy(f, p + 1e-18, color=COLORS[label], label=label, lw=1.6)
        ax.axvline(5.0, color="red", linestyle=":", alpha=0.5, label="5 Hz")
        ax.set_title(f"{which.capitalize()} PSD (avg over 5 ep, summed xyz)")
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("power (log)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path.name}")


def per_episode_plot(eps_c1, eps_p2, eps_gt, out_path: Path):
    """5 columns (episodes) × 3 rows (gt / C1 / Phase2). Accel magnitude."""
    fig, axes = plt.subplots(3, 5, figsize=(18, 7.5), sharey="row")
    rows = [
        (eps_gt, "gt (Round A target)"),
        (eps_c1, "C1-v3"),
        (eps_p2, "Phase2-v3"),
    ]
    for row_idx, (eps, label) in enumerate(rows):
        for col in range(5):
            ax = axes[row_idx, col]
            e = eps[col]
            t = np.arange(len(e["imu"])) / 30.0
            accel = e["imu"][:, :3]
            mag = np.linalg.norm(accel - accel.mean(axis=0), axis=1)
            ax.plot(t, mag, color=COLORS[label], lw=0.8, alpha=0.9)
            if col == 0:
                ax.set_ylabel(f"{label}\n|accel - mean|", fontsize=9.5)
            if row_idx == 0:
                ax.set_title(f"ep {col}", fontsize=10)
            if row_idx == 2:
                ax.set_xlabel("time [s]", fontsize=9)
            ax.grid(alpha=0.3)
    fig.suptitle("Per-episode IMU accel magnitude (DC-removed)\n"
                 "Top: gt (Round A reward target)  |  Middle: C1-v3 real  |  Bottom: Phase2-v3 real",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved {out_path.name}")


# ─── main ─────────────────────────────────────────────────────────
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("[load] reward computer pickle")
    with open(RC_PKL, "rb") as f:
        rc = pickle.load(f)
    print("[load] eval_C1_v3 (5 ep)")
    eps_c1 = load_episodes(EVAL_C1)
    print("[load] eval_phase2_v3 (5 ep)")
    eps_p2 = load_episodes(EVAL_P2)

    print("[build] gt traces (k=1 NN, per ep, raw scale)")
    eps_gt = []
    for ep in eps_c1:
        gt_imu = build_gt_trace(rc, ep)
        eps_gt.append({"name": f"gt_{ep['name']}", "imu": gt_imu})
        print(f"  {ep['name']}: gt trace shape {gt_imu.shape}")

    print("\n[metrics] computing per-episode rows")
    rows_c1 = per_episode_metrics(eps_c1, has_action=True)
    rows_p2 = per_episode_metrics(eps_p2, has_action=True)
    rows_gt = per_episode_metrics(eps_gt, has_action=False)

    print("\n--- per-episode IMU metrics (5 ep) ---")
    for label, rows in [("gt", rows_gt), ("C1-v3", rows_c1), ("Phase2-v3", rows_p2)]:
        print(f"\n[{label}]")
        for r in rows:
            print(f"  {r['name']}  accel_rms={r['accel_rms']:.3f}  "
                  f"gyro_rms={r['gyro_rms']:.3f}  "
                  f"accel_hf={r['accel_hf_pct']:.2f}%  "
                  f"gyro_hf={r['gyro_hf_pct']:.2f}%")

    print("\n[plot] bar chart")
    bar_chart(rows_c1, rows_p2, rows_gt, OUT / "bar_chart_3way.png")
    print("[plot] PSD")
    psd_plot(eps_c1, eps_p2, eps_gt, OUT / "psd_3way.png")
    print("[plot] per-episode")
    per_episode_plot(eps_c1, eps_p2, eps_gt, OUT / "per_episode_3way.png")

    print(f"\nall plots saved to {OUT}/")


if __name__ == "__main__":
    main()
