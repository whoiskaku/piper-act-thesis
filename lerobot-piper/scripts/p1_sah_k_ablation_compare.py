#!/usr/bin/env python3
"""K 值消融对比：用 4 个 SAH ckpt 跑离线 FFT 测 within-chunk vs boundary-cross HF。"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/enolab-guo/claude-test-2/lerobot-piper/scripts")
from eval_smoothness import load_act_policy, predict_action_trajectory  # noqa: E402

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def fft_hf_ratio(action_chunks, fs=30.0, cutoff=5.0):
    x = action_chunks - action_chunks.mean(axis=1, keepdims=True)
    win = np.hanning(x.shape[1])[None, :, None]
    Xf = np.fft.rfft(x * win, axis=1)
    psd = (np.abs(Xf) ** 2).mean(axis=0)
    freqs = np.fft.rfftfreq(x.shape[1], d=1.0 / fs)
    mask = freqs > cutoff
    return psd[mask].sum(axis=0) / (psd.sum(axis=0) + 1e-30)


def evaluate(ckpt, label, dataset, n_eps=3, device="cuda"):
    print(f"[{label}] loading {ckpt}")
    policy = load_act_policy(ckpt, device)
    within, cross = [], []
    for ep_i in range(n_eps):
        traj = predict_action_trajectory(policy, dataset, ep_i, device)
        T = traj.shape[0]
        for s in range(0, T - 100 + 1, 100):
            within.append(traj[s:s + 100])
        for s in range(50, T - 100 + 1, 100):
            cross.append(traj[s:s + 100])
    within = np.stack(within)
    cross = np.stack(cross)
    hf_w = fft_hf_ratio(within)
    hf_c = fft_hf_ratio(cross)
    return {
        "label": label,
        "ckpt": str(ckpt),
        "n_within": len(within),
        "n_cross": len(cross),
        "within_hf_per_ch": hf_w.tolist(),
        "cross_hf_per_ch": hf_c.tolist(),
        "within_hf_mean": float(hf_w.mean()),
        "cross_hf_mean": float(hf_c.mean()),
    }


def main():
    configs = [
        ("K=3 cheb",  "outputs/train/2026-04-26/22-47-51_p1_sah_k3_cheb_sanity/checkpoints/005000/pretrained_model"),
        ("K=5 cheb",  "outputs/train/2026-04-26/20-50-52_p1_sah_k5_sanity/checkpoints/005000/pretrained_model"),
        ("K=8 cheb",  "outputs/train/2026-04-26/23-04-31_p1_sah_k8_cheb_sanity/checkpoints/005000/pretrained_model"),
        ("K=5 bsp",   "outputs/train/2026-04-26/23-21-10_p1_sah_k5_bsp_sanity/checkpoints/005000/pretrained_model"),
    ]
    ds = LeRobotDataset("local/piper_pick_place_imu_v1")
    results = []
    for label, ckpt in configs:
        results.append(evaluate(ckpt, label, ds, n_eps=3))

    print("\n" + "=" * 78)
    print(f"{'config':<14s}  {'within HF':>14s}  {'cross HF':>14s}  {'cross/within':>14s}")
    print("-" * 78)
    for r in results:
        ratio = r["cross_hf_mean"] / max(r["within_hf_mean"], 1e-12)
        print(f"  {r['label']:<12s}  {r['within_hf_mean']*100:>13.5f}%  "
              f"{r['cross_hf_mean']*100:>13.5f}%  {ratio:>13.0f}×")
    print("=" * 78)
    print("\nReference:")
    print("  GT (demo):    0.018%")
    print("  Vanilla:      0.245%")
    print("  α-imu-only:   0.137%")

    json.dump(results, open("outputs/phase0/p1_k_ablation_fft.json", "w"), indent=2)
    print("\nsaved -> outputs/phase0/p1_k_ablation_fft.json")


if __name__ == "__main__":
    main()
