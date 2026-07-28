#!/usr/bin/env python3
"""同位姿门：用该位姿**自己的 A** 决定是否放 B±（c-pilot-v2，2026-07-29）。

用户 2026-07-29 决定：不能用 p99 点估计，也不能用 P1 的值外推到别的位姿。

    rm_gate = m_command^{B worst} − UCB95( p99_tracking^{A} ) − 0.3°
    只有 rm_gate > 0 才放 B+。

UCB95 由**慢动周期分块 bootstrap** 给出（块长 = 一个 dither 周期，
保留周期内的相关结构）。

用法: python sim/gate_same_pose_C.py --pose 56 --joint 2
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

import make_probe_b2 as MB
import replay_probe as RP

ROOT = Path(__file__).resolve().parents[1]
PC = ROOT / "outputs/probe_c"
FPS = 20.0
SAFE = np.radians(0.3)


def block_boot_p99(err, blk, nboot=4000, seed=0):
    """慢动周期分块 bootstrap ⇒ p99 的 95% 上界。

    返回 (p99_point, ucb95, n_blocks)。块数很少时 UCB 会很宽 —— 那是**保守方向**，
    但必须如实报告块数，不能假装样本量充足。
    """
    n = len(err)
    nb = max(n // blk, 1)
    blocks = [err[i * blk:(i + 1) * blk] for i in range(nb)]
    rng = np.random.default_rng(seed)
    pt = np.percentile(err, 99)
    if nb < 2:
        return pt, float("nan"), nb
    bs = np.array([np.percentile(np.concatenate(
        [blocks[i] for i in rng.integers(nb, size=nb)]), 99) for _ in range(nboot)])
    return pt, float(np.percentile(bs, 95)), nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", type=int, required=True)
    ap.add_argument("--joint", type=int, default=2)
    ap.add_argument("--rep", type=int, default=1)
    a = ap.parse_args()

    p = PC / f"replay_c_p{a.pose}_j{a.joint+1}_A{a.rep}_c{a.pose}_{a.joint+1}.json"
    if not p.exists():
        raise SystemExit(f"✗ 缺同位姿 A 记录: {p.name}")
    R = json.loads(p.read_text())
    meta = R["seq_meta"]
    hz, blk = meta["hz"], int(round(FPS / meta["hz"]))
    st = np.array(R["state"], float)[:, :6] * np.pi / 180 / 1000
    cm = np.array(R["cmd"], float)[:, :6]
    seg = slice(meta["n_appr"], len(st))               # 进场之后 = 保持 + 注入段
    err = np.abs(st[seg] - cm[seg])
    lo, hi = MB.reach_limits()

    print(f"=== 同位姿门: 位姿 {a.pose}, 注入 j{a.joint+1} (用 A{a.rep}) ===")
    print(f"  dither {np.degrees(meta['dither_rad']):.1f}° @ {hz} Hz  "
          f"⇒ 块长 {blk} 帧 ({blk/FPS:.1f}s)；评估段 {err.shape[0]} 帧\n")

    print(f"  【A 臂跟踪误差：分块 bootstrap（块长扫描）+ 逐块最大】")
    print(f"    评估段只有 {err.shape[0]/FPS:.1f}s 而 dither 周期 {1/hz:.1f}s ⇒ 整周期块只有"
          f" {err.shape[0]//blk} 块。")
    print(f"    2 块的 bootstrap 只有 3 种重采样组合, **给不出有意义的 95% 上界**")
    print(f"    (UCB/点 会退化到 ~1.06)。故同时扫短块并取**逐块最大 p99**, 取三者最大。")
    print(f"  {'关节':>5}{'p99点估计':>11}{'UCB(1T)':>10}{'UCB(1/2T)':>11}"
          f"{'UCB(1/4T)':>11}{'逐块max':>10}{'**采用**':>11}")
    ucb = {}
    for j in (1, 2):
        pt = np.percentile(err[:, j], 99)
        us = []
        for div in (1, 2, 4):
            bl = max(blk // div, 5)
            _, u, nbi = block_boot_p99(err[:, j], bl)
            us.append(u if np.isfinite(u) else pt)
        nbm = max(err.shape[0] // blk, 1)
        bmax = max(np.percentile(err[i*blk:(i+1)*blk, j], 99) for i in range(nbm))
        take = max(max(us), bmax)
        ucb[j] = take
        print(f"  {'j'+str(j+1):>5}{np.degrees(pt):>11.4f}"
              + "".join(f"{np.degrees(u):>{w}.4f}" for u, w in zip(us, (10, 11, 11)))
              + f"{np.degrees(bmax):>10.4f}{np.degrees(take):>11.4f}")
    print(f"    ⇒ 采用值 = max(三种块长的 UCB95, 逐块最大 p99) —— 保守方向")

    # B± 的最坏命令余量（不需要真跑 B 就能算）
    print(f"\n  【B± 最坏命令余量 m_command】(由已生成的 npz 直接算)")
    worst = {}
    for arm in ("Bp", "Bm"):
        f = RP.PROBE / f"c_p{a.pose}_j{a.joint+1}_{arm}.npz"
        if not f.exists():
            raise SystemExit(f"✗ 缺 {f.name} —— 先跑一次 --arm {arm} --dry-run 生成序列")
        z = np.load(f)
        q = z["cmd7"][:, :6] + z["da"]
        for j in (1, 2):
            m = min((q[:, j] - lo[j]).min(), (hi[j] - q[:, j]).min())
            worst[j] = min(worst.get(j, np.inf), m)
    for j in (1, 2):
        print(f"    j{j+1}: {np.degrees(worst[j]):+.4f}°")

    print(f"\n  【放行判据】rm_gate = m_command − UCB95(p99_A) − 0.3°")
    ok = True
    for j in (1, 2):
        g = np.degrees(worst[j] - ucb[j] - SAFE)
        pt_g = np.degrees(worst[j] - np.percentile(err[:, j], 99) - SAFE)
        ok &= g > 0
        print(f"    j{j+1}: rm_gate = {g:+.4f}°   (点估计口径 {pt_g:+.4f}°)"
              f"   {'✓' if g > 0 else '**✗**'}")
    print(f"\n  ⇒ B± {'**GO**' if ok else '**HOLD** —— 不放 B±'}")
    if not ok:
        print(f"    按既定判决路径: 若为 P3, 降 dither 至 1° 属**设计变更**,")
        print(f"    须重算 φ 秩/条件数、速度配对、⓪、能量与 hash 并重新冻结 tag;")
        print(f"    或跑一个明确标记 pilot-only 的 1° 版本(不能证明原 2° 设计安全)。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
