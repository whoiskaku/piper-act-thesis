#!/usr/bin/env python3
"""大数据重训(2026-06-10) — 稠密 (ACT_output → A_clean) 配对数据生成。

背景: 原 phase1_paired_data.h5 的 960 train 实为 240 个唯一起点 × 4 扰动变体
(ACT_output/act_img 由真机观测决定, 4 变体共享同一输入 → 唯一输入仅 240)。
abs-loss 实验显示模型在 train 上能压绝对高频到 0.04(架构表达得了可泛化的
"低通自身输入"), 但 240 个唯一输入太少 → 记忆胜过泛化(val 1.36)。
本脚本把 stride 50→5, 唯一输入 ×10, 测试可泛化抵消是否随数据涌现。

与原管线的差异(论文需写明):
  - 无扰动变体(K=1 identity) → 同时消除"同输入多目标"的标签噪声
  - A_clean = 真 demo 动作切片(无 sim 执行; S0/Hattn 不用 IMU/noisy 路)
  - history_action/history_imu 填零(S0/Hattn use_history_*=False, 不读)
  - 一次 ACT teacher-forced 前向同时取 ACT_output + act_img_tokens(encoder hook)

输出: outputs/sim/phase1_dense.h5 (train/val 组, 键兼容 RefinePhase1Dataset)
用法: python sim/generate_dense_act_pairs.py

#65 扩展 — K 噪声实现增广(--k-train K --out phase1_dense_k4.h5):
ACT_output 的抖动部分由 dataset history 的随机采样驱动(同帧两次 history 差≈4.0
→ ACT_output 差≈0.026 rad, 模型本身确定)。对同一 (ep,start) 重复 K 次前向
= 同一 A_clean 目标、K 个独立噪声实现 —— 教模型分离"信号(共享)与噪声(各异)",
比幅度扰动更对路的增广。val 保持 K=1(与既有结果可比)。
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lerobot.common.datasets.factory import make_dataset
from sim.precompute_act_output import (
    load_frozen_act, gather_batch, build_ep_frame_to_global,
    C1_CKPT, DATASET_REPO, BATCH_SIZE, DEVICE)
from sim.replay_demo_validation import load_real_demo, state_mdeg_to_rad
from sim.noisy_piper_sim import qpos_8_to_state_7

OUT_H5 = REPO_ROOT / "outputs/sim/phase1_dense.h5"
CHUNK = 100
STRIDE = 5                    # 原 50 → 5: 唯一起点 ×10
HISTORY_LEN = 600
TRAIN_EPS = list(range(0, 24))
VAL_EPS = list(range(24, 30))


def collect_starts(eps):
    """[(ep, start, A_clean(100,7), start_state_7)] for all dense windows."""
    out = []
    for ep in eps:
        real = load_real_demo(ep)
        action = real["action"].astype(np.float32)          # (T,7)
        state_mdeg = real["state_mdeg"]
        T = len(action)
        for s in range(0, T - CHUNK + 1, STRIDE):
            st7 = qpos_8_to_state_7(state_mdeg_to_rad(state_mdeg[s])).astype(np.float32)
            out.append((ep, s, action[s:s + CHUNK], st7))
        print(f"  [ep{ep}] T={T} → {len(range(0, T - CHUNK + 1, STRIDE))} starts")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-train", type=int, default=1,
                    help="train 每起点的噪声实现数(同目标多输入增广); val 恒为 1")
    ap.add_argument("--out", type=Path, default=OUT_H5)
    cli = ap.parse_args()

    policy, cfg = load_frozen_act(C1_CKPT, DATASET_REPO, DEVICE)
    dataset = make_dataset(cfg)
    ep_from, _ = build_ep_frame_to_global(dataset)
    p = cfg.policy

    # image-token offset(同 precompute_act_img)
    n_prefix = 1
    n_prefix += int(getattr(p, "robot_state_feature", None) is not None)
    n_prefix += int(getattr(p, "env_state_feature", None) is not None)
    n_prefix += int(getattr(p, "enable_imu_input", False) and getattr(p, "imu_feature", None) is not None)
    hL = p.history_max_length
    n_hist = (hL if p.history_use_action else 0) + (hL if p.history_use_imu else 0)
    offset = n_prefix + n_hist
    print(f"[offset] n_prefix={n_prefix} n_hist={n_hist} → image tokens 从 {offset} 起")

    cap = {}
    hook = policy.model.encoder.register_forward_hook(lambda m, i, o: cap.__setitem__("e", o))
    n_action = policy.config.n_action_steps

    with h5py.File(cli.out, "w") as f:
        for split, eps, k_real in (("train", TRAIN_EPS, cli.k_train), ("val", VAL_EPS, 1)):
            print(f"[{split}] collecting dense starts (stride={STRIDE}, k_real={k_real}) ...")
            items = collect_starts(eps)
            n0 = len(items)
            n = n0 * k_real
            src_ep0 = np.array([it[0] for it in items], np.int32)
            src_start0 = np.array([it[1] for it in items], np.int32)
            gidx = ep_from[src_ep0] + src_start0
            A_clean0 = np.stack([it[2] for it in items])
            state0 = np.stack([it[3] for it in items])

            grp = f.create_group(split)
            # 静态键: K 个实现共享同一目标/状态 → 块状复制(块 k = [k*n0,(k+1)*n0))
            grp.create_dataset("A_clean", data=np.tile(A_clean0, (k_real, 1, 1)))
            grp.create_dataset("start_state_7", data=np.tile(state0, (k_real, 1)))
            grp.create_dataset("src_ep", data=np.tile(src_ep0, k_real))
            grp.create_dataset("src_start", data=np.tile(src_start0, k_real))
            grp.create_dataset("real_id", data=np.repeat(np.arange(k_real, dtype=np.int32), n0))
            # S0/Hattn 不读的占位键(RefinePhase1Dataset 必需)
            grp.create_dataset("A_noisy", shape=(n, CHUNK, 7), dtype="float32")
            grp.create_dataset("history_action", shape=(n, HISTORY_LEN, 7), dtype="float32")
            grp.create_dataset("history_imu", shape=(n, HISTORY_LEN, 6), dtype="float32")
            grp.create_dataset("history_pad_mask", shape=(n, HISTORY_LEN), dtype="bool")
            d_act = grp.create_dataset("ACT_output", shape=(n, CHUNK, 7), dtype="float32")
            d_img = None

            t0 = time.time()
            for k in range(k_real):
                base = k * n0
                for b in range(0, n0, BATCH_SIZE):
                    e = min(b + BATCH_SIZE, n0)
                    # dataset 每次 __getitem__ 重新随机采样 history → 每个 k 是独立噪声实现
                    samples = [dataset[int(gidx[i])] for i in range(b, e)]
                    batch = gather_batch(samples)
                    with torch.no_grad():
                        bi = policy.normalize_inputs(batch)
                        if policy.config.image_features:
                            bi = dict(bi)
                            bi["observation.images"] = torch.stack(
                                [bi[k2] for k2 in policy.config.image_features], dim=-4)
                        bi.pop("action", None); bi.pop("action_is_pad", None)
                        out = policy.model(bi)
                        actions = out[0] if isinstance(out, (list, tuple)) else out
                        actions = policy.unnormalize_outputs({"action": actions})["action"]
                        img = cap["e"][offset:].permute(1, 0, 2)        # (B, n_img, 512)
                    if d_img is None:
                        n_img = img.shape[1]
                        print(f"[{split}] N={n} (n0={n0}×k{k_real})  n_img={n_img}  (S={cap['e'].shape[0]})")
                        d_img = grp.create_dataset("act_img_tokens", shape=(n, n_img, 512),
                                                   dtype="float16", compression="gzip",
                                                   compression_opts=4)
                    d_act[base + b:base + e] = actions[:, :n_action].cpu().numpy().astype(np.float32)
                    d_img[base + b:base + e] = img.cpu().numpy().astype(np.float16)
                    if (b // BATCH_SIZE) % 20 == 0:
                        done = base + e
                        print(f"  [{split}] k{k} {done}/{n} ({done / max(time.time() - t0, 1e-9):.1f}/s)")
            print(f"  [{split}] done n={n} {time.time() - t0:.1f}s "
                  f"finite={np.isfinite(d_act[:5]).all()}")
    hook.remove()
    print(f"[done] {cli.out}")


if __name__ == "__main__":
    main()
