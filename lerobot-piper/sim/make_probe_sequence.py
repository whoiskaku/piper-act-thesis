#!/usr/bin/env python3
"""③a 探针序列生成（2026-07-22）— 生成 δa 脉冲表 + 安全检查 + Ĝ 的预测。

设计（协议 PROTOCOL_robot_validation.md ③a）:
  · 幅度 5 mrad（ε 的 50%）— 大到压过复放噪声, 小到仍在已认证安全区
  · 单帧脉冲, 每 2 秒(40帧)一发, 关节轮流 — 间隔 > 800ms 尾巴, 互不干扰
  · 关节 j2/j3/j5（墙的主载体）, 正负各半 — 顺带验奇对称性
  · 静止段不发（与部署的 still-gate 同判据）

输出 outputs/probe/probe_seq.npz:
  da        (T,6)  探针序列, 直接加到指令上
  hits      (N,3)  [帧, 关节, 符号] 便于分析时定位
  pred      (K,T,3) Ĝ 各成员对该探针的预测增量（墙钟, 真机系）
用法: python sim/make_probe_sequence.py [ins-deploy|pp2-deploy] [ep]
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch

import train_pi_e2 as TP
from g_model import Rigid, load_ensemble
from wall_portrait import smooth

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/probe"
TAG = sys.argv[1] if len(sys.argv) > 1 else "ins-deploy"
EP = int(sys.argv[2]) if len(sys.argv) > 2 else 0        # 复放用训练集的集即可
AMP = 0.006                       # v2: 6 mrad (仍 ≪ ε=10)
PERIOD = 40                       # 2 秒
JOINTS = (1, 2)                   # v2: 只 j2/j3 (j5 模型峰 12 mrad/s, 设计上测不到)
GATE_TH = 0.04                    # 与部署 still-gate 同
WAVELET = (1, 1, -1, -1, 1, 1, -1, -1)   # v2: 8帧 5Hz 方波小波 — 打在共振带(3.7-5Hz)
NW = len(WAVELET)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(TP.LROOT / TAG / f"ep{EP:03d}.npz")
    man = json.loads((TP.LROOT / TAG / "manifest.json").read_text())
    R = np.array(man["R"], np.float32)
    a_arm = z["a_arm"].astype(np.float32)
    grip = z["grip"].astype(np.float32)
    cmd7 = np.concatenate([a_arm, grip[:, None]], 1)
    T = len(a_arm)

    # 运动判据: 用该集的 real gyro 模(与部署 gate 同口径)
    gate = smooth(np.linalg.norm(z["real"][:, 3:6], axis=1))
    moving = gate >= GATE_TH

    # 关节限位(避免探针加深越界 —— 基准轨迹本身可能已贴/越限, 如 j3 上界)
    import mujoco
    from ideal_imu_kinematic import MODEL_PATH
    mj = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    lo, hi = mj.jnt_range[:6, 0], mj.jnt_range[:6, 1]
    exc = lambda v, j: max(lo[j] - v, v - hi[j], 0.0)      # 越界深度

    da = np.zeros((T, 6), np.float32)
    hits = []
    k = 0
    skipped = 0
    for t in range(PERIOD, T - PERIOD, PERIOD):
        if not moving[t]:                       # 静止段不发
            continue
        j = JOINTS[k % len(JOINTS)]
        s = 1.0 if (k // len(JOINTS)) % 2 == 0 else -1.0     # 首极性交替(防系统性偏置)
        if t + NW >= T:
            continue
        # 整个小波(±都出现)做限位检查: 任一帧加深越界 → 整发跳过(小波正负对称, 翻转无益)
        bad = any(exc(a_arm[t+i, j] + w*s*AMP, j) > exc(a_arm[t+i, j], j) + 1e-9
                  for i, w in enumerate(WAVELET))
        if bad:
            skipped += 1
            continue
        for i, w in enumerate(WAVELET):
            da[t+i, j] = w * s * AMP
        hits.append((t, j, int(s)))
        k += 1
    hits = np.array(hits, np.int32)
    if skipped:
        print(f"  (跳过 {skipped} 发: 两种符号都会加深越界)")

    # 安全检查
    assert np.abs(da).max() <= TP.EPS_B + 1e-12, "探针超过 ε 硬界!"
    assert (np.abs(da) > 0).sum(1).max() <= 1, "同一帧不得多关节同时发"
    if len(hits) > 1:
        gap = np.diff(hits[:, 0]) - NW                     # 发尾到下发头
        assert gap.min() >= 16, f"发间隔 {gap.min()} 帧 < 16(800ms 尾巴), 会互相污染"

    # Ĝ 的预测(各成员) — 分析时的对照
    rig = Rigid()
    models = load_ensemble(rigid=rig)
    pred = []
    for g in models:
        g.R = R
        pred.append(g.increment_fast(cmd7, da))
    pred = np.stack(pred)                                    # (K,T,3) 墙钟

    np.savez(OUT / "probe_seq.npz", da=da, hits=hits, pred=pred, R=R,
             cmd7=cmd7, tag=TAG, ep=EP, amp=AMP)
    pk = np.abs(pred).max(axis=(1, 2)) * 1e3
    print(f"探针序列 v2: {len(hits)} 发小波 × {AMP*1e3:.0f} mrad × {NW}帧@5Hz  "
          f"(T={T} 帧 = {T/20:.0f}s)")
    print(f"  关节分布: " + ", ".join(f"j{j+1}×{(hits[:,1]==j).sum()}" for j in JOINTS))
    print(f"  符号: +{(hits[:,2]>0).sum()} / −{(hits[:,2]<0).sum()}")
    if len(hits) > 1:
        gap = np.diff(hits[:, 0]) - NW
        print(f"  发间净间隔 ≥{gap.min()} 帧 ({gap.min()*50}ms)  ✓ > 800ms 尾巴")
    print(f"  Ĝ 预测的增量峰值(各成员): {np.round(pk,1)} mrad/s")
    print(f"→ {OUT/'probe_seq.npz'}")
    print(f"\n下一步: 用 sim/replay_probe.py 在真机上复放 A1 / B / A2")


if __name__ == "__main__":
    main()
