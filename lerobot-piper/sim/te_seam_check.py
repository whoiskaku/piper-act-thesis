#!/usr/bin/env python3
"""TE 是否消掉 chunk 接缝？—— 用 phase1 已录的 te / no_te 对照（2026-07-29，零机时）

背景：FF 的 −33% 是相对 **TE 关闭**的基线测的（当前部署 `temporal_ensemble_coeff=None`,
`n_action_steps=100`）。TE 与 blend 互斥（TE 要求 `n_action_steps=1`，没有接缝可 blend），
与 lag 正交。这是 `PREREG_FF_insert.md` §4 记下的已知未闭混淆项。

phase1 留有 `eval_phase1_noimu_te` / `eval_phase1_noimu_no_te` 两个数据集（30 Hz），
可以离线回答：**接缝在指令上有多大 → 有没有传到 IMU → TE 消掉多少。**

三个量，故意用**互相独立的口径**，避免任何单一伪影主导结论：
  ⓪ 指令侧    |Δa| 按 chunk 内位置 —— 不含渲染、不含 IMU，最干净
  ① 真机侧    带通后 real 陀螺能量按位置 —— 不含渲染，但含"本该有的运动"
  ② gap 侧    real − ideal 按位置 —— 隔离了偏离，但 ideal 在跳变处有样条伪影
     · no_te 用**逐 chunk 渲染**（部署口径；自然边界伪影已在 ff_seam_confound 里量过）
     · te   用**整集渲染**（te 本来就没有 chunk，这是它的正确口径，且无跳变可求导 ⇒ 干净）

注意：两个数据集**集数不同**(25 vs 5)，可能不同批次。所有跨数据集的绝对值比较都不可靠，
只有**各自数据集内部的"位置0 / 中段"倍率**是免疫的 —— 结论只建立在倍率上。

用法: python sim/te_seam_check.py
"""
from __future__ import annotations
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

import sys
sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parents[1]
CACHE = os.path.expanduser("~/.cache/huggingface/lerobot/enolab-guo")
FPS1, CHUNK, SMAX = 30.0, 100, 16         # phase1 是 30 Hz（SMAX 需宽到不触边）
SETS = [("eval_phase1_noimu_no_te", "TE 关（当前部署）", True),
        ("eval_phase1_noimu_te", "TE 开", False)]


def bp30(x):
    ba = butter(3, [0.5 / (FPS1 / 2), 9.0 / (FPS1 / 2)], "bandpass")
    return filtfilt(*ba, x, axis=0)


def episodes(name):
    for f in sorted(glob.glob(os.path.join(CACHE, name, "data/**/*.parquet"),
                              recursive=True)):
        d = pd.read_parquet(f)
        yield (np.stack(d["action"].values).astype(np.float64),
               np.stack(d["observation.imu_real"].values).astype(np.float64))


def prof(vals, start=0):
    """按 chunk 内位置聚合。"""
    pos = (np.arange(len(vals)) + start) % CHUNK
    return np.array([vals[pos == k].mean() if (pos == k).any() else np.nan
                     for k in range(CHUNK)])


def ratio(p):
    return float(p[0] / np.nanmean(p[30:90]))


def render(cmd, chunked):
    """cmd(T,7) → ideal IMU(T,6)。30 Hz ⇒ 必须改渲染器的 FPS（它是模块级常量）。"""
    import lerobot.common.policies.act.b_runtime as BR
    old = BR.FPS
    BR.FPS = FPS1                                   # 显式改，用完还原
    try:
        # hi_fps 必须整除 FPS，否则降采样时基漂移（默认 200 对 30 Hz 是 6.67 ⇒ 不行）
        rend = BR.IdealChunkRenderer(hi_fps=240.0)
        if not chunked:
            return rend.render(cmd, None)[:len(cmd)]
        out, prev = [], None
        for s in range(0, len(cmd), CHUNK):
            out.append(rend.render(cmd[s:s + CHUNK], prev))
            prev = cmd[:s + CHUNK]
        return np.concatenate(out)[:len(cmd)]
    finally:
        BR.FPS = old


def kabsch(A, B):
    U, _, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def main():
    print("=" * 84)
    print("TE 是否消掉 chunk 接缝 —— phase1 te / no_te 对照（30 Hz，零机时）")
    print("  结论只建立在**各数据集内部**的「位置0 / 中段(30-90)」倍率上（跨集数不可比）\n")

    rows = []
    for name, label, chunked in SETS:
        eps = list(episodes(name))
        # ⓪ 指令跳变
        p_cmd = np.nanmean([prof(np.linalg.norm(np.diff(a[:, :6], axis=0), axis=1), 1)
                            for a, _ in eps], 0)
        # 平滑度（速度尺度不变）: RMS(Δ²a)/RMS(Δa)
        smooth = np.mean([np.sqrt((np.diff(a[:, :6], 2, axis=0) ** 2).mean())
                          / np.sqrt((np.diff(a[:, :6], axis=0) ** 2).mean())
                          for a, _ in eps])
        # ① 真机侧 + ② gap 侧
        pr, pg, lags, gap_abs = [], [], [], []
        for a, imu in eps:
            T = min(len(a), len(imu))
            ideal6 = render(a[:T].astype(np.float32), chunked)
            T = min(T, len(ideal6))                     # 渲染可能短几帧（末段重采样）
            gi, gr = ideal6[:T, 3:], imu[:T, 3:]
            # 逐集滞后：**与 pp2/insert 同口径** —— 逐 shift 重拟 R 后取 gap 能量最小
            # （此前用未标定系的模长互相关，最优值顶到 SMAX 边界 ⇒ 不可信，已弃）
            n = T - SMAX
            sel = np.linalg.norm(gi[:n], axis=1) > 0.15
            if sel.sum() < 50:
                continue
            Rs, es = [], []
            for s in range(SMAX + 1):
                Rk = kabsch(gi[:n][sel], gr[s:s + n][sel])
                Rs.append(Rk)
                e = bp30(gr[s:s + n] - gi[:n] @ Rk.T)
                es.append(float((e ** 2).sum() / n))
            L = int(np.argmin(es))
            if L in (0, SMAX):
                print(f"    ! 滞后触边 L={L} (SMAX={SMAX}) — 该集的对齐不可信")
            lags.append(L); R = Rs[L]
            g = bp30(gr[L:L + n] - gi[:n] @ R.T)
            gap_abs.append(float(np.nanmean(prof((g ** 2).sum(1))[30:90])))
            pg.append(prof((g ** 2).sum(1)))
            pr.append(prof((bp30(gr[L:L + n]) ** 2).sum(1)))
        p_real, p_gap = np.nanmean(pr, 0), np.nanmean(pg, 0)
        rows.append((label, len(eps), np.median(lags), smooth,
                     ratio(p_cmd), ratio(p_real), ratio(p_gap), np.mean(gap_abs)))

    print(f"  {'':>16}{'集数':>5}{'滞后帧':>7}{'=ms':>6}{'Δ²/Δ':>8}"
          f"{'⓪指令跳变':>12}{'①真机HF':>11}{'②gap':>9}   ← 位置0/中段 倍率")
    for lab, n, L, sm, c, r, g, _ab in rows:
        print(f"  {lab:>16}{n:>5}{L:>7.0f}{L/FPS1*1e3:>6.0f}{sm:>8.3f}"
              f"{c:>11.2f}×{r:>10.2f}×{g:>8.2f}×")

    (_, _, L0, sm0, c0, r0, g0, ab0), (_, _, L1, sm1, c1, r1, g1, ab1) = rows
    print(f"\n  接缝倍率 TE关 → TE开:  指令 {c0:.2f}× → {c1:.2f}× | "
          f"真机 {r0:.2f}× → {r1:.2f}× | gap {g0:.2f}× → {g1:.2f}×")
    print(f"  全程平滑度 Δ²/Δ:       {sm0:.3f} → {sm1:.3f}"
          f"  （{'TE 更平滑' if sm1 < sm0 else 'TE 未更平滑'}，速度尺度不变量）")
    print(f"  滞后:                  {L0:.0f} 帧 = {L0/FPS1*1e3:.0f} ms  →  "
          f"{L1:.0f} 帧 = {L1/FPS1*1e3:.0f} ms   （TE 不做时移，理应不变）")
    print(f"\n  [仅供参考，**未控制**] 中段绝对 gap/帧: {ab0:.5f} → {ab1:.5f} "
          f"({ab1/ab0*100-100:+.1f}%)")
    print("    两数据集集数 25 vs 5、可能不同批次 ⇒ 跨数据集绝对值不可比，此行不作结论。")
    print("\n  判读：三个独立口径若同向 ⇒ 结论不依赖任何单一伪影。")
    print("        TE 若把接缝倍率压到 ~1 ⇒ 它在这条路径上与 blend 同靶，二者互斥不可叠加。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
