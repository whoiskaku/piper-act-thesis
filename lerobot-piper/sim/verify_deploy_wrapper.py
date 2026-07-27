#!/usr/bin/env python3
"""部署 wrapper 回归测试（2026-07-22）— 无需机时。

三项判据:
  ① **流式装配 == 离线构造**: 逐帧喂真墙钟 (IMU, 编码器), 按 chunk 分段喂指令,
     产生的 51 维特征与 δa 必须与 build_offline 逐点一致。
  ② **逐 chunk 渲染 == 整集渲染**: 部署时 ideal 只能按 chunk 渲染(拿不到未来 chunk),
     须与训练时用的整集 kin_ideal_20 足够接近。
  ③ 实时性与安全: 逐帧延迟 < 50ms 预算, ε 硬界不越, gate 触发率与离线一致。

对齐口径备忘(标签 npz):
    z["real"][i] = real_wall[i+L]   z["ideal"][i] = ideal_wall[i]
    z["a_arm"][i] = cmd_wall[i]     z["qw"][i]    = enc_wall[i]
所以回放时墙钟第 i 帧的 IMU 要取自**原始日志** d["imu_real"][i], 不能用 z["real"]。
用法: python sim/verify_deploy_wrapper.py [pp2-deploy|ins-deploy]
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.common.policies.act.b_runtime import (
    CHUNK, EPS_B, FEAT_DIM, WIN, FeatureAssembler, IdealChunkRenderer, build_offline, load_tcn,
)
from compare_ideal_methods import load_episode

ROOT = Path(__file__).resolve().parents[1]
TAG = sys.argv[1] if len(sys.argv) > 1 else "pp2-deploy"
REPO = {"pp2-deploy": "enolab-guo/eval_pickplace2_100k",
        "ins-deploy": "enolab-guo/eval_insert_100k"}[TAG]
EP, L = 8, 3
CKPT = ROOT / "outputs/p2_pi/pi_v1_s0.pt"


def main() -> int:
    d = ROOT / "outputs/p2_labels" / TAG
    z = np.load(d / f"ep{EP:03d}.npz")
    man = json.loads((d / "manifest.json").read_text())
    R, b = np.array(man["R"], np.float32), np.array(man["b"], np.float32)
    pi = load_tcn(CKPT)
    ep_raw = load_episode(REPO, EP)
    imu_wall = ep_raw["imu_real"].astype(np.float32)          # 真墙钟 IMU

    # ---------- 训练路径(离线) ----------
    X_off, _, _ = build_offline(z, L=L)
    n = len(X_off)
    with torch.no_grad():
        da_off = pi(torch.from_numpy(X_off).unsqueeze(0))[0].numpy()

    # ---------- 部署路径(流式, 按 chunk 分段) ----------
    # 装配器用单位标定 + 直接喂已标定 ideal, 以隔离"流式装配逻辑"(避免 R 往返浮点噪声)
    asm = FeatureAssembler(np.eye(3, dtype=np.float32), np.zeros(3, np.float32), L=L)
    ideal_sim = z["ideal"].astype(np.float32)
    cmd7 = np.concatenate([z["a_arm"], z["grip"][:, None]], 1).astype(np.float32)
    qw = z["qw"].astype(np.float32)

    rows_on = np.zeros((n, FEAT_DIM), np.float32)
    da_on = np.zeros((n, 6), np.float32)
    gates, lat = [], []
    fed = 0
    for i in range(n):
        if i >= fed:                                          # 到达新 chunk 边界
            hi = min(fed + CHUNK, n)
            asm.on_new_chunk(cmd7[fed:hi], ideal_sim[fed:hi])
            fed = hi
        t0 = time.perf_counter()
        win, gate = asm.step(imu_wall[i], qw[i])
        with torch.no_grad():
            out = pi(torch.from_numpy(win).unsqueeze(0))
        lat.append((time.perf_counter() - t0) * 1e3)
        rows_on[i], da_on[i], _ = win[-1], out[0, -1].numpy(), gates.append(gate)

    # ---------- ① 一致性 ----------
    dif = np.abs(rows_on - X_off)
    feat_err = float(dif.max())
    da_err = float(np.abs(da_on[WIN:] - da_off[WIN:]).max() * 1e3)   # 暖机段豁免(部署侧已 gate)
    print(f"=== {TAG} ep{EP}  n={n} 帧, {int(np.ceil(n/CHUNK))} 个 chunk ===")
    print(f"① 特征逐点最大差 : {feat_err:.3e}", end="")
    if feat_err > 1e-5:
        bad = np.argwhere(dif > 1e-5)
        blk = ["e", "ideal", "a", "qd", "phase"][
            min(np.searchsorted([6, 24, 42, 48, 51], bad[0][1], "right"), 4)]
        print(f"   ⚠ 首个不符 帧{bad[0][0]} 维{bad[0][1]}({blk}块) 共{len(bad)}处")
    else:
        print("   ✓")
    print(f"   δa 逐点最大差 : {da_err:.6f} mrad (窗满后; 暖机 {WIN} 帧部署侧已 gate)   "
          f"相关={np.corrcoef(da_on.ravel(), da_off.ravel())[0,1]:.6f}")

    # ---------- ② 逐 chunk 渲染 vs 整集渲染 ----------
    rend = IdealChunkRenderer()
    reps, prev = [], None
    for s in range(0, n, CHUNK):
        c = cmd7[s:min(s + CHUNK, n)]
        reps.append(rend.render(c, prev))
        prev = c
    ideal_chunked = np.concatenate(reps)[:n] @ np.block(
        [[R.T, np.zeros((3, 3), np.float32)], [np.zeros((3, 3), np.float32), R.T]])
    g_err = np.abs(ideal_chunked[:, 3:] - ideal_sim[:, 3:])
    rel = float(np.sqrt((g_err ** 2).mean()) / np.sqrt((ideal_sim[:, 3:] ** 2).mean()))
    print(f"② 逐chunk渲染 vs 整集: 陀螺相对误差 {rel*100:.2f}%  最大 {g_err.max():.4f} rad/s"
          f"   {'✓' if rel < 0.05 else '⚠ 边界效应偏大'}")

    # ---------- ③ 实时性与安全 ----------
    print(f"③ 逐帧延迟 p50={np.percentile(lat,50):.2f}ms p95={np.percentile(lat,95):.2f}ms "
          f"(预算 50ms)  {'✓' if np.percentile(lat,95) < 50 else '✗'}")
    print(f"   gate 触发率 {100*np.mean(np.array(gates)<0.04):.1f}% "
          f"(离线 still {100*z['mask_still'].mean():.1f}%)  "
          f"饱和 {100*np.mean(np.abs(da_on).max(1)>0.95*EPS_B):.2f}%  "
          f"‖δa‖={np.sqrt((da_on**2).mean())*1e3:.2f}mrad")
    print(f"   ε 硬界 {'✓ 未越界' if np.abs(da_on).max() <= EPS_B + 1e-9 else '✗ 越界!'}")

    ok = (feat_err < 1e-5 and da_err < 0.01 and rel < 0.05
          and np.abs(da_on).max() <= EPS_B + 1e-9 and np.percentile(lat, 95) < 50.0)
    print(f"\n判决: {'✅ PASS — 部署路径与训练路径一致' if ok else '❌ FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
