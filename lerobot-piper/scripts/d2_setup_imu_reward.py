#!/usr/bin/env python3
"""D2 setup script: build IMURewardComputer from v2 demos + C1 deploy data.

Run once before Phase 2 training. Output:
  outputs/eval/d2_imu_reward_computer.pkl   # serialized IMURewardComputer
  outputs/eval/d2_nn_distance_hist.png      # diagnostic histogram

Pipeline:
  1. Load v2 dataset → extract per-frame (state, IMU window) for valid frames
                       (i.e., frames with chunk_size remaining in same demo)
  2. Normalize state with v2 state stats; normalize IMU with v2 IMU stats
  3. Build KDTree over normalized states
  4. Compute v2 inter-demo NN distance distribution (exclude same-demo)
     → p95
  5. Compute C1 deploy NN distance distribution (each frame's nn into v2 tree)
     → mean + 3σ, p95
  6. τ = max(v2_inter_p95, c1_mean+3σ, c1_p95)   if cfg.nn_threshold_method=="auto"
  7. Save IMURewardComputer pickle + plot histogram
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.neighbors import KDTree
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.configuration_phase2 import Phase2Config, validate_config
from lerobot.common.policies.act.imu_reward import IMURewardComputer


def build_nodes(base: LeRobotDataset, chunk_size: int):
    """For each demo, extract frames [0, T-chunk_size] as KDTree nodes.
    Each node stores (state, imu_window of next chunk_size frames)."""
    nodes_state, nodes_imu, demo_ids, frame_indices = [], [], [], []
    for ep in tqdm(range(base.num_episodes), desc="Building KDTree nodes"):
        ep_from = int(base.episode_data_index["from"][ep])
        ep_to = int(base.episode_data_index["to"][ep])
        ep_len = ep_to - ep_from
        if ep_len < chunk_size:
            print(f"  WARN: ep {ep} too short ({ep_len} < {chunk_size}), skipped")
            continue

        # Slice the whole episode in one go (fast)
        ep_states = base.hf_dataset[ep_from:ep_to]["observation.state"]
        ep_imus = base.hf_dataset[ep_from:ep_to]["observation.imu_real"]
        ep_states = (
            ep_states if torch.is_tensor(ep_states) else torch.stack(list(ep_states))
        ).numpy().astype(np.float32)
        ep_imus = (
            ep_imus if torch.is_tensor(ep_imus) else torch.stack(list(ep_imus))
        ).numpy().astype(np.float32)

        # Valid frame range: [0, ep_len - chunk_size]
        for f in range(ep_len - chunk_size + 1):
            nodes_state.append(ep_states[f])
            nodes_imu.append(ep_imus[f : f + chunk_size])
            demo_ids.append(ep)
            frame_indices.append(f)

    return (
        np.array(nodes_state, dtype=np.float32),
        np.array(nodes_imu, dtype=np.float32),
        np.array(demo_ids, dtype=np.int32),
        np.array(frame_indices, dtype=np.int32),
    )


def compute_inter_demo_nn(tree: KDTree, norm_states: np.ndarray,
                           demo_ids: np.ndarray, k_search: int = 20) -> np.ndarray:
    """For each node, find nearest neighbor in a DIFFERENT demo.
    Returns array of distances."""
    dists, idxs = tree.query(norm_states, k=k_search)
    out = np.zeros(len(norm_states), dtype=np.float32)
    for i in tqdm(range(len(norm_states)), desc="Inter-demo NN"):
        own = demo_ids[i]
        for d, j in zip(dists[i], idxs[i]):
            if demo_ids[j] != own:
                out[i] = d
                break
        else:
            out[i] = dists[i, -1]  # fallback (rare)
    return out


def compute_c1_nn(tree: KDTree, c1_states_norm: np.ndarray) -> np.ndarray:
    """For each C1 deploy frame, find nearest v2 neighbor distance."""
    dists, _ = tree.query(c1_states_norm, k=1)
    return dists[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1-deploy-repo", default="local/eval_C1_history",
                    help="LeRobot repo of C1 real-robot deployment data")
    ap.add_argument("--output-pickle", default=None,
                    help="Where to save IMURewardComputer (default: from cfg)")
    ap.add_argument("--output-plot", default=None,
                    help="Where to save NN distance histogram")
    args = ap.parse_args()

    cfg = Phase2Config()
    validate_config(cfg)

    out_pkl = Path(args.output_pickle or cfg.reward_computer_pickle)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    out_plot = Path(args.output_plot or out_pkl.parent / "d2_nn_distance_hist.png")

    print("=" * 60)
    print("D2 IMU REWARD SETUP")
    print("=" * 60)
    print(f"v2 dataset    : {cfg.v2_dataset_repo}")
    print(f"C1 deploy data: {args.c1_deploy_repo}")
    print(f"chunk_size    : {cfg.chunk_size_for_gt}")
    print(f"output pickle : {out_pkl}")

    # ── Step 1: Load v2 + build nodes ───────────────────────────
    base = LeRobotDataset(cfg.v2_dataset_repo, root=None)
    print(f"\nv2 stats: {base.num_episodes} episodes, {base.num_frames} frames")

    nodes_state, nodes_imu, demo_ids, frame_indices = build_nodes(
        base, cfg.chunk_size_for_gt
    )
    n_nodes = len(nodes_state)
    print(f"KDTree nodes: {n_nodes} (state shape {nodes_state.shape[1:]}, "
          f"IMU window shape {nodes_imu.shape[1:]})")

    # ── Step 2: Normalize state + IMU ───────────────────────────
    state_mean = np.array(base.meta.stats["observation.state"]["mean"], dtype=np.float32)
    state_std = np.array(base.meta.stats["observation.state"]["std"], dtype=np.float32)
    imu_mean = np.array(base.meta.stats["observation.imu_real"]["mean"], dtype=np.float32)
    imu_std = np.array(base.meta.stats["observation.imu_real"]["std"], dtype=np.float32)

    norm_states = (nodes_state - state_mean) / (state_std + 1e-8)
    # Normalize IMU windows: shape (n_nodes, chunk_size, 6)
    norm_imu_windows = (nodes_imu - imu_mean) / (imu_std + 1e-8)

    print(f"\nState stats: mean={state_mean}, std={state_std}")
    print(f"IMU   stats: mean={imu_mean}, std={imu_std}")

    # ── Step 3: KDTree ──────────────────────────────────────────
    print("\nBuilding KDTree...")
    tree = KDTree(norm_states, leaf_size=40)

    # ── Step 4: v2 inter-demo NN distances ──────────────────────
    print("\nComputing v2 inter-demo NN distances...")
    v2_inter_dists = compute_inter_demo_nn(tree, norm_states, demo_ids, k_search=30)
    v2_inter_p95 = float(np.percentile(v2_inter_dists, 95))
    v2_inter_mean = float(v2_inter_dists.mean())
    v2_inter_std = float(v2_inter_dists.std())

    print(f"  v2 inter-demo NN: mean={v2_inter_mean:.3f}, "
          f"std={v2_inter_std:.3f}, p95={v2_inter_p95:.3f}")

    # ── Step 5: C1 deploy NN distances ──────────────────────────
    print(f"\nLoading C1 deploy data: {args.c1_deploy_repo}")
    c1_data = LeRobotDataset(args.c1_deploy_repo, root=None)
    c1_states_raw = []
    for i in range(c1_data.num_frames):
        c1_states_raw.append(c1_data.hf_dataset[i]["observation.state"].numpy())
    c1_states_raw = np.array(c1_states_raw, dtype=np.float32)
    c1_states_norm = (c1_states_raw - state_mean) / (state_std + 1e-8)
    print(f"C1 frames: {len(c1_states_raw)}")

    c1_dists = compute_c1_nn(tree, c1_states_norm)
    c1_mean = float(c1_dists.mean())
    c1_std = float(c1_dists.std())
    c1_p95 = float(np.percentile(c1_dists, 95))
    c1_p99 = float(np.percentile(c1_dists, 99))

    print(f"  C1 NN dist: mean={c1_mean:.3f}, std={c1_std:.3f}, "
          f"p95={c1_p95:.3f}, p99={c1_p99:.3f}")

    # ── Step 6: Decide τ ────────────────────────────────────────
    tau_v2_p95 = v2_inter_p95
    tau_c1_3sigma = c1_mean + 3 * c1_std
    tau_c1_p95 = c1_p95
    if cfg.nn_threshold_value is not None:
        tau = float(cfg.nn_threshold_value)
        print(f"\nτ (overridden from config): {tau:.3f}")
    else:
        tau = max(tau_v2_p95, tau_c1_3sigma, tau_c1_p95)
        print(f"\nτ candidates:")
        print(f"  v2 inter p95       : {tau_v2_p95:.3f}")
        print(f"  C1 mean + 3σ       : {tau_c1_3sigma:.3f}")
        print(f"  C1 p95             : {tau_c1_p95:.3f}")
        print(f"  τ (= max of above) : {tau:.3f}")

    # ── Step 7: Build & save IMURewardComputer ──────────────────
    nn_stats = {
        "v2_inter_mean": v2_inter_mean,
        "v2_inter_std": v2_inter_std,
        "v2_inter_p95": v2_inter_p95,
        "c1_mean": c1_mean,
        "c1_std": c1_std,
        "c1_p95": c1_p95,
        "c1_p99": c1_p99,
        "tau": tau,
    }
    computer = IMURewardComputer(
        kdtree=tree,
        imu_windows=norm_imu_windows,
        demo_ids=demo_ids,
        frame_indices=frame_indices,
        state_mean=state_mean,
        state_std=state_std,
        imu_mean=imu_mean,
        imu_std=imu_std,
        tau=tau,
        chunk_size=cfg.chunk_size_for_gt,
        n_demos=int(base.num_episodes),
        state_dim=int(nodes_state.shape[1]),
        imu_dim=int(nodes_imu.shape[2]),
        nn_dist_stats=nn_stats,
    )
    computer.save(str(out_pkl))
    print(f"\nIMURewardComputer saved to {out_pkl}")
    print(f"  size: {out_pkl.stat().st_size / 1e6:.1f} MB")

    # ── Step 8: Plot histogram ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(0, max(v2_inter_dists.max(), c1_dists.max()), 50)

    axes[0].hist(v2_inter_dists, bins=bins, alpha=0.6, color="C0",
                 label=f"v2 inter-demo (n={len(v2_inter_dists)})")
    axes[0].hist(c1_dists, bins=bins, alpha=0.6, color="C3",
                 label=f"C1 deploy (n={len(c1_dists)})")
    axes[0].axvline(tau, color="black", linestyle="--",
                    label=f"τ = {tau:.3f}")
    axes[0].set_xlabel("NN distance (normalized state space)")
    axes[0].set_ylabel("count")
    axes[0].set_title("NN distance distributions")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # CDF
    for label, data, color in [
        ("v2 inter-demo", v2_inter_dists, "C0"),
        ("C1 deploy", c1_dists, "C3"),
    ]:
        sd = np.sort(data)
        cdf = np.linspace(0, 1, len(sd))
        axes[1].plot(sd, cdf, label=label, color=color)
    axes[1].axvline(tau, color="black", linestyle="--", label=f"τ = {tau:.3f}")
    axes[1].axhline(0.95, color="grey", linestyle=":", alpha=0.6)
    axes[1].set_xlabel("NN distance")
    axes[1].set_ylabel("CDF")
    axes[1].set_title("CDF (95% line marked)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"D2 setup: τ = {tau:.3f} (v2 inter p95={v2_inter_p95:.2f}, "
                 f"C1 mean+3σ={tau_c1_3sigma:.2f}, C1 p95={c1_p95:.2f})")
    fig.tight_layout()
    fig.savefig(out_plot, dpi=130)
    print(f"Histogram saved to {out_plot}")

    print("\n" + "=" * 60)
    print(f"D2 setup complete. τ = {tau:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
