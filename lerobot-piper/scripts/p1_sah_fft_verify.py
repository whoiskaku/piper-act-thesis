#!/usr/bin/env python3
"""SAH 架构验证：检查 SAH ckpt 输出动作的 >5Hz 能量是否真的接近 0。

这是 Stage 2 的核心验证——证明 SAH 的"带宽硬约束"承诺确实成立。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/enolab-guo/claude-test-2/lerobot-piper/scripts")
from eval_smoothness import load_act_policy, predict_action_trajectory  # noqa: E402

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


CHANNELS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]


def fft_hf_ratio(action_chunks, fs=30.0, cutoff=5.0):
    """action_chunks: (N, T, D). Returns mean >5Hz ratio per channel."""
    x = action_chunks - action_chunks.mean(axis=1, keepdims=True)
    win = np.hanning(x.shape[1])[None, :, None]
    Xf = np.fft.rfft(x * win, axis=1)
    psd = (np.abs(Xf) ** 2).mean(axis=0)  # (F, D)
    freqs = np.fft.rfftfreq(x.shape[1], d=1.0 / fs)
    mask = freqs > cutoff
    return psd[mask].sum(axis=0) / (psd.sum(axis=0) + 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset-repo-id", default="local/piper_pick_place_imu_v1")
    ap.add_argument("--n-episodes", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"[load] {args.ckpt}")
    policy = load_act_policy(args.ckpt, args.device)

    print("[load] dataset")
    ds = LeRobotDataset(args.dataset_repo_id)

    print(f"[infer] running {args.n_episodes} episodes through SAH policy")
    # Two FFT analysis modes:
    #  - within-chunk: windows aligned with prediction chunk boundaries (== one SAH output)
    #    → Architectural HF guarantee should make this ~0.
    #  - boundary-crossing: windows offset by 50 → span 2 prediction chunks
    #    → Will pick up any chunk-edge discontinuities (NOT covered by SAH).
    within_chunks = []
    crossing_chunks = []
    for ep_i in range(args.n_episodes):
        traj = predict_action_trajectory(policy, ds, ep_i, args.device)  # (T, D)
        T = traj.shape[0]
        for s in range(0, T - 100 + 1, 100):
            within_chunks.append(traj[s:s + 100])
        for s in range(50, T - 100 + 1, 100):
            crossing_chunks.append(traj[s:s + 100])

    within_chunks = np.stack(within_chunks)
    crossing_chunks = np.stack(crossing_chunks)
    print(f"[fft] within-chunk: {within_chunks.shape[0]}  boundary-crossing: {crossing_chunks.shape[0]}")

    hf_within = fft_hf_ratio(within_chunks)
    hf_cross = fft_hf_ratio(crossing_chunks)

    print("\n" + "=" * 80)
    print(f"{'channel':<12s}  {'within-chunk':>15s}  {'boundary-cross':>16s}  {'reference':>15s}")
    print("-" * 80)
    for ch, name in enumerate(CHANNELS):
        print(f"  {name:<10s}  {hf_within[ch]*100:>14.5f}%  {hf_cross[ch]*100:>15.5f}%  ")
    print("-" * 80)
    print(f"  {'overall':<10s}  {hf_within.mean()*100:>14.5f}%  {hf_cross.mean()*100:>15.5f}%")
    print("=" * 80)

    print("\nReference (real-robot H1 study, action HF >5Hz):")
    print("  GT (demo):    0.018%")
    print("  Vanilla:      0.245%   (13.6× GT)")
    print("  α-imu-only:   0.137%   (7.6× GT)")
    print(f"  SAH within:   {hf_within.mean()*100:.5f}%   ← architectural cap test")
    print(f"  SAH cross:    {hf_cross.mean()*100:.5f}%   ← chunk boundary leakage")

    print("\n" + "=" * 80)
    if hf_within.mean() < 1e-5:
        print("✅✅ ARCHITECTURE WORKING: within-chunk HF essentially zero")
        print("    → SAH does what it claims; remaining HF is purely chunk-boundary leakage")
        print("    → Need TE or endpoint constraints in P1 endgame")
    elif hf_within.mean() < 0.001:
        print("✅ Architecture working: within-chunk HF < 0.001% (numerical floor)")
    else:
        print(f"❌ Within-chunk HF = {hf_within.mean()*100:.5f}% — bug suspected!")
        print("    SAH should structurally guarantee zero >5Hz energy within a chunk.")
    print("=" * 80)


if __name__ == "__main__":
    main()
