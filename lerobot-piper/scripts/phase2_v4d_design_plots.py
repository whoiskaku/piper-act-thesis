#!/usr/bin/env python3
"""v4d design-aid plots — visualize gt_IMU candidates and how they relate to
real-policy IMU traces, to inform the asymmetric reward design choice.

Outputs (under outputs/eval/phase2_v4d_design/):
  Fig 1 gt_candidates.png      — gt_k1 vs gt_k5_mean vs gt_k5_smoothest vs
                                  real C1-v3 IMU, for 3 representative chunks
  Fig 2 demo_distribution.png  — all 30 v3 demos' IMU norm overlaid
  Fig 3 gt_vs_policies.png     — gt_k1 vs gt_k5_mean vs C1-v3 vs Phase2-v3,
                                  same 3 chunks
"""
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ─── paths ───────────────────────────────────────────────────────
RC_PKL = "/home/enolab-guo/claude-test-2/lerobot-piper/outputs/eval/d2_imu_reward_computer.pkl"
EVAL_C1 = Path.home() / ".cache/huggingface/lerobot/local/eval_C1_v3"
EVAL_P2 = Path.home() / ".cache/huggingface/lerobot/local/eval_phase2_v3"
DEMOS   = Path.home() / ".cache/huggingface/lerobot/local/piper_pick_place_imu_v3"
OUT     = Path("/home/enolab-guo/claude-test-2/lerobot-piper/outputs/eval/phase2_v4d_design")

CHUNK_SIZE   = 100
CHUNKS_TO_SHOW = [0, 2, 4]  # frame 0, 200, 400 within an episode
EP_TO_SHOW   = 0            # episode 0 from each eval set
K_DEMOS      = 5            # number of cross-demo neighbors


# ─── load ────────────────────────────────────────────────────────
def load_parquet_episode(ds_root: Path, ep_idx: int):
    """Return dict with action / state / imu / frames for episode ep_idx."""
    pq = sorted(ds_root.glob(f"data/chunk-*/episode_{ep_idx:06d}.parquet"))[0]
    df = pd.read_parquet(pq)
    return {
        "state":  np.stack(df["observation.state"].to_numpy()).astype(np.float32),
        "imu":    np.stack(df["observation.imu_real"].to_numpy()).astype(np.float32),
        "frames": len(df),
    }


def all_demo_imus(ds_root: Path):
    """Return list of (name, imu_array) for every demo episode."""
    out = []
    for pq in sorted(ds_root.glob("data/chunk-*/episode_*.parquet")):
        df = pd.read_parquet(pq)
        imu = np.stack(df["observation.imu_real"].to_numpy()).astype(np.float32)
        out.append((pq.stem, imu))
    return out


def query_gt_candidates(rc, state_raw: np.ndarray, k_demos: int = 5, k_query: int = 50):
    """Return dict with the 3 gt candidates (normalized IMU windows).

    keys:
        gt_k1            (chunk, 6)
        gt_k5_mean       (chunk, 6)
        gt_k5_smoothest  (chunk, 6)
        nn_dist          mean over the k_demos selected
    """
    q = (state_raw - rc.state_mean) / (rc.state_std + 1e-8)
    k = min(k_query, rc.kdtree.data.shape[0])
    dists, idxs = rc.kdtree.query(q.reshape(1, -1), k=k)
    dists, idxs = dists[0], idxs[0]

    # Filter to k_demos distinct demos
    seen, sel_idx, sel_dist = set(), [], []
    for d, i in zip(dists, idxs):
        did = int(rc.demo_ids[int(i)])
        if did in seen:
            continue
        seen.add(did)
        sel_idx.append(int(i)); sel_dist.append(float(d))
        if len(sel_idx) >= k_demos:
            break

    windows = rc.imu_windows[sel_idx]   # (k, chunk, 6)  already normalized

    # k=1: nearest only
    gt_k1 = windows[0]
    # k=5 mean
    gt_k5_mean = windows.mean(axis=0)
    # k=5 smoothest (lowest total ||IMU||²)
    energies = (windows ** 2).sum(axis=(1, 2))
    gt_k5_smoothest = windows[int(np.argmin(energies))]

    return {
        "gt_k1": gt_k1,
        "gt_k5_mean": gt_k5_mean,
        "gt_k5_smoothest": gt_k5_smoothest,
        "nn_dist_mean": float(np.mean(sel_dist)),
    }


# ─── helpers for IMU norms ───────────────────────────────────────
def accel_norm(imu):     # (T, 6) → (T,)
    return np.linalg.norm(imu[:, :3], axis=1)

def gyro_norm(imu):      # (T, 6) → (T,)
    return np.linalg.norm(imu[:, 3:], axis=1)


# ─── Fig 1: gt candidates vs real C1-v3 ──────────────────────────
def make_fig_gt_candidates(rc, c1_ep, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharex="col")
    fig.suptitle(f"Fig 1 — gt_IMU candidates vs C1-v3 ep {EP_TO_SHOW} real IMU\n"
                 "(red shaded region: real exceeds gt, would be penalized under asymmetric reward)",
                 fontsize=12)

    for col, ck_idx in enumerate(CHUNKS_TO_SHOW):
        start = ck_idx * CHUNK_SIZE
        if start + CHUNK_SIZE > c1_ep["frames"]:
            continue
        state_t = c1_ep["state"][start]
        real_chunk = c1_ep["imu"][start:start + CHUNK_SIZE]
        # Normalize real for fair comparison (windows are normalized)
        real_chunk_norm = (real_chunk - rc.imu_mean) / (rc.imu_std + 1e-8)

        cand = query_gt_candidates(rc, state_t, k_demos=K_DEMOS)

        # Top row: accel norm
        ax = axes[0, col]
        t = np.arange(CHUNK_SIZE)
        ax.plot(t, accel_norm(real_chunk_norm), color="black",  lw=1.4, label="real C1-v3")
        ax.plot(t, accel_norm(cand["gt_k1"]),         color="#888", lw=1.0, ls="-",  label="gt_k1 (1-NN)")
        ax.plot(t, accel_norm(cand["gt_k5_mean"]),    color="#1f77b4", lw=1.2, ls="--", label="gt_k5_mean")
        ax.plot(t, accel_norm(cand["gt_k5_smoothest"]),color="#2ca02c", lw=1.2, ls=":",  label="gt_k5_smoothest")
        # shade where real exceeds gt_k5_mean (most likely candidate)
        excess = np.maximum(0, accel_norm(real_chunk_norm) - accel_norm(cand["gt_k5_mean"]))
        ax.fill_between(t, accel_norm(cand["gt_k5_mean"]),
                          accel_norm(cand["gt_k5_mean"]) + excess,
                          color="red", alpha=0.18, label="excess vs gt_k5_mean")
        ax.set_title(f"chunk {ck_idx} (frame {start}-{start+CHUNK_SIZE-1})\n"
                     f"nn_dist={cand['nn_dist_mean']:.3f}", fontsize=10)
        if col == 0:
            ax.set_ylabel("|accel| (normalized)", fontsize=10)
        ax.grid(alpha=0.3)
        if col == 2:
            ax.legend(loc="upper right", fontsize=8)

        # Bottom row: gyro norm
        ax = axes[1, col]
        ax.plot(t, gyro_norm(real_chunk_norm), color="black",  lw=1.4)
        ax.plot(t, gyro_norm(cand["gt_k1"]),         color="#888", lw=1.0, ls="-")
        ax.plot(t, gyro_norm(cand["gt_k5_mean"]),    color="#1f77b4", lw=1.2, ls="--")
        ax.plot(t, gyro_norm(cand["gt_k5_smoothest"]),color="#2ca02c", lw=1.2, ls=":")
        excess_g = np.maximum(0, gyro_norm(real_chunk_norm) - gyro_norm(cand["gt_k5_mean"]))
        ax.fill_between(t, gyro_norm(cand["gt_k5_mean"]),
                          gyro_norm(cand["gt_k5_mean"]) + excess_g,
                          color="red", alpha=0.18)
        if col == 0:
            ax.set_ylabel("|gyro| (normalized)", fontsize=10)
        ax.set_xlabel("step within chunk", fontsize=10)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


# ─── Fig 2: all 30 demos overlay ────────────────────────────────
def make_fig_demo_distribution(demos, rc, out_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    fig.suptitle("Fig 2 — v3 demo IMU distribution (30 demos overlaid, normalized)",
                 fontsize=12)

    max_len = max(d[1].shape[0] for d in demos)
    accel_stack, gyro_stack = [], []
    for name, imu in demos:
        imu_n = (imu - rc.imu_mean) / (rc.imu_std + 1e-8)
        axes[0].plot(accel_norm(imu_n), color="gray", alpha=0.25, lw=0.6)
        axes[1].plot(gyro_norm(imu_n),  color="gray", alpha=0.25, lw=0.6)
        # pad to max_len with NaN for averaging
        pad = max_len - imu_n.shape[0]
        if pad > 0:
            imu_n = np.vstack([imu_n, np.full((pad, 6), np.nan)])
        accel_stack.append(accel_norm(imu_n))
        gyro_stack.append(gyro_norm(imu_n))

    accel_arr = np.stack(accel_stack)
    gyro_arr  = np.stack(gyro_stack)
    accel_mean = np.nanmean(accel_arr, axis=0)
    accel_std  = np.nanstd(accel_arr, axis=0)
    gyro_mean  = np.nanmean(gyro_arr, axis=0)
    gyro_std   = np.nanstd(gyro_arr, axis=0)

    t = np.arange(max_len)
    axes[0].plot(t, accel_mean, color="red", lw=2.0, label="mean across demos")
    axes[0].fill_between(t, accel_mean - accel_std, accel_mean + accel_std,
                          color="red", alpha=0.18, label="±1σ band")
    axes[0].set_ylabel("|accel| (normalized)", fontsize=10)
    axes[0].set_title("Accel norm")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, gyro_mean,  color="red", lw=2.0)
    axes[1].fill_between(t, gyro_mean - gyro_std, gyro_mean + gyro_std,
                          color="red", alpha=0.18)
    axes[1].set_ylabel("|gyro| (normalized)", fontsize=10)
    axes[1].set_xlabel("frame within episode", fontsize=10)
    axes[1].set_title("Gyro norm")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


# ─── Fig 3: gt + C1-v3 + Phase2-v3 ───────────────────────────────
def make_fig_gt_vs_policies(rc, c1_ep, p2_ep, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharex="col")
    fig.suptitle(f"Fig 3 — gt_IMU vs two policies (C1-v3 BC vs Phase2-v3 RL Round A), ep {EP_TO_SHOW}",
                 fontsize=12)

    for col, ck_idx in enumerate(CHUNKS_TO_SHOW):
        start = ck_idx * CHUNK_SIZE
        if start + CHUNK_SIZE > min(c1_ep["frames"], p2_ep["frames"]):
            continue
        # Use C1-v3's state to query gt (same as training would do)
        state_t = c1_ep["state"][start]
        cand = query_gt_candidates(rc, state_t, k_demos=K_DEMOS)

        c1_chunk = c1_ep["imu"][start:start + CHUNK_SIZE]
        p2_chunk = p2_ep["imu"][start:start + CHUNK_SIZE]
        c1_n = (c1_chunk - rc.imu_mean) / (rc.imu_std + 1e-8)
        p2_n = (p2_chunk - rc.imu_mean) / (rc.imu_std + 1e-8)

        t = np.arange(CHUNK_SIZE)
        ax = axes[0, col]
        ax.plot(t, accel_norm(cand["gt_k1"]),      color="#888", lw=1.0, ls=":",  label="gt_k1")
        ax.plot(t, accel_norm(cand["gt_k5_mean"]), color="green", lw=1.4, ls="--", label="gt_k5_mean")
        ax.plot(t, accel_norm(c1_n),               color="#1f77b4", lw=1.4, label="C1-v3 real")
        ax.plot(t, accel_norm(p2_n),               color="#d62728", lw=1.4, label="Phase2-v3 real")
        ax.set_title(f"chunk {ck_idx} (frame {start}-{start+CHUNK_SIZE-1})", fontsize=10)
        if col == 0:
            ax.set_ylabel("|accel| (normalized)", fontsize=10)
        ax.grid(alpha=0.3)
        if col == 2:
            ax.legend(loc="upper right", fontsize=8)

        ax = axes[1, col]
        ax.plot(t, gyro_norm(cand["gt_k1"]),      color="#888", lw=1.0, ls=":")
        ax.plot(t, gyro_norm(cand["gt_k5_mean"]), color="green", lw=1.4, ls="--")
        ax.plot(t, gyro_norm(c1_n),               color="#1f77b4", lw=1.4)
        ax.plot(t, gyro_norm(p2_n),               color="#d62728", lw=1.4)
        if col == 0:
            ax.set_ylabel("|gyro| (normalized)", fontsize=10)
        ax.set_xlabel("step within chunk", fontsize=10)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


# ─── Fig 4: gt_k1 only + C1-v3 + Phase2-v3 ──────────────────────
def make_fig_gt_k1_vs_policies(rc, c1_ep, p2_ep, out_path: Path):
    """Same as Fig 3 but ONLY gt_k1 (the gt that was actually used in Round A)."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharex="col")
    fig.suptitle(f"Fig 4 — gt_k1 (Round A actually used) vs C1-v3 vs Phase2-v3 real IMU, ep {EP_TO_SHOW}",
                 fontsize=12)

    for col, ck_idx in enumerate(CHUNKS_TO_SHOW):
        start = ck_idx * CHUNK_SIZE
        if start + CHUNK_SIZE > min(c1_ep["frames"], p2_ep["frames"]):
            continue
        state_t = c1_ep["state"][start]
        cand = query_gt_candidates(rc, state_t, k_demos=K_DEMOS)
        gt_k1 = cand["gt_k1"]

        c1_chunk = c1_ep["imu"][start:start + CHUNK_SIZE]
        p2_chunk = p2_ep["imu"][start:start + CHUNK_SIZE]
        c1_n = (c1_chunk - rc.imu_mean) / (rc.imu_std + 1e-8)
        p2_n = (p2_chunk - rc.imu_mean) / (rc.imu_std + 1e-8)

        t = np.arange(CHUNK_SIZE)
        # Accel norm
        ax = axes[0, col]
        ax.plot(t, accel_norm(gt_k1), color="#888", lw=1.4, ls="--", label="gt_k1 (target)")
        ax.plot(t, accel_norm(c1_n),  color="#1f77b4", lw=1.5, label="C1-v3 real")
        ax.plot(t, accel_norm(p2_n),  color="#d62728", lw=1.5, label="Phase2-v3 real")
        ax.set_title(f"chunk {ck_idx} (frame {start}-{start+CHUNK_SIZE-1})\n"
                     f"nn_dist={cand['nn_dist_mean']:.3f}", fontsize=10)
        if col == 0:
            ax.set_ylabel("|accel| (normalized)", fontsize=10)
        ax.grid(alpha=0.3)
        if col == 2:
            ax.legend(loc="upper right", fontsize=9)

        # Gyro norm
        ax = axes[1, col]
        ax.plot(t, gyro_norm(gt_k1), color="#888", lw=1.4, ls="--")
        ax.plot(t, gyro_norm(c1_n),  color="#1f77b4", lw=1.5)
        ax.plot(t, gyro_norm(p2_n),  color="#d62728", lw=1.5)
        if col == 0:
            ax.set_ylabel("|gyro| (normalized)", fontsize=10)
        ax.set_xlabel("step within chunk", fontsize=10)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


# ─── main ────────────────────────────────────────────────────────
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("[load] reward computer pickle")
    with open(RC_PKL, "rb") as f:
        rc = pickle.load(f)
    print("[load] C1-v3 ep 0")
    c1_ep = load_parquet_episode(EVAL_C1, EP_TO_SHOW)
    print("[load] Phase2-v3 ep 0")
    p2_ep = load_parquet_episode(EVAL_P2, EP_TO_SHOW)
    print("[load] all v3 demos")
    demos = all_demo_imus(DEMOS)
    print(f"  {len(demos)} demos loaded")

    print("\n[plot] Fig 1: gt candidates")
    make_fig_gt_candidates(rc, c1_ep, OUT / "fig1_gt_candidates.png")
    print("[plot] Fig 2: demo distribution")
    make_fig_demo_distribution(demos, rc, OUT / "fig2_demo_distribution.png")
    print("[plot] Fig 3: gt vs policies")
    make_fig_gt_vs_policies(rc, c1_ep, p2_ep, OUT / "fig3_gt_vs_policies.png")
    print("[plot] Fig 4: gt_k1 vs policies (no k=5)")
    make_fig_gt_k1_vs_policies(rc, c1_ep, p2_ep, OUT / "fig4_gt_k1_vs_policies.png")
    print(f"\nall figures saved to {OUT}/")


if __name__ == "__main__":
    main()
