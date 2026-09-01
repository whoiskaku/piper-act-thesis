#!/usr/bin/env python3
"""**在 pp2 上实测可重复性上界** —— 会话2 的 6 条重复 vs 会话1 的同一条。

`PREREG_G_VNEXT.md` §5 的门槛 0.789 / 0.707 锚在 0.986 / 0.884，
但那两个数是在 `probe_b2` 的 6 次**同会话**裸复放上测的。本脚本在 **pp2** 上、
**跨会话**（绑带重装过）测同一量。

⚠ 跨会话必然低于同会话（绑带重装、热态、位姿微差）⇒ 这是**保守**上界。
⚠ **数值门槛事前冻结，不随本结果浮动**（否则是移动球门）。本结果只作背景。
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load, to20, bp, best_fit, score, q_at, FPS

ROOT = Path(__file__).resolve().parents[2]
REPEATS = ["pp2_ep000","pp2_ep004","pp2_ep009","pp2a_ep000","pp2a_ep004","pp2a_ep009"]
BANDS = ((0.5,3.0),(3.0,9.0),(0.5,9.0))

print("pp2 跨会话可重复性（会话2 重复 vs 会话1 同序列）\n")
print(f"  {'序列':<14}" + "".join(f"{f'{lo}-{hi}Hz':>12}" for lo,hi in BANDS) + "   IMU")
acc = {b: [] for b in BANDS}; accq = {b: [] for b in BANDS}
for nm in REPEATS:
    d1 = load(ROOT/f"outputs/step4_gvnext/s1/replay_{nm}.json")
    d2 = load(ROOT/f"outputs/step4_gvnext/s2/replay_{nm}.json")
    # IMU: 抗混叠到 20 Hz, 会话间拟合装着旋转 + 位移
    A1, A2 = to20(d1["gyro100"]), to20(d2["gyro100"])
    row = ""
    for b in BANDS:
        L, r2, S = best_fit(A1, A2, *b)
        n = min(len(A1)-L, len(S))
        a_, s_ = bp(A1[L:L+n], *b), bp(S[:n], *b)
        resid = 1.0 - float(((a_-s_)**2).sum())/max(float((a_*a_).sum()),1e-30)
        acc[b].append((r2, resid)); row += f"{r2:>7.3f}/{resid:>5.3f}"
    print(f"  {nm:<14}{row}")
    # 关节角: 同一时长网格上比（不需旋转）
    n = min(len(d1["tg_q"]), len(d2["tg_q"]))
    t = d1["tg_q"][:n] - d1["tg_q"][0]
    q1 = q_at(d1, d1["tg_q"][0]+t)[::10]; q2 = q_at(d2, d2["tg_q"][0]+t)[::10]
    for b in BANDS:
        rr = [best_fit(q1[:,j:j+1], q2[:,j:j+1], *b, rotate=False)[1]
              for j in range(6) if np.degrees(q1[:,j].max()-q1[:,j].min()) > 5]
        accq[b].append(float(np.mean(rr)))
import math
print(f"\n  {'IMU 中位 ρ²/resid':<20}" + "".join(
    f"{np.median([x[0] for x in acc[b]]):>7.3f}/{np.median([x[1] for x in acc[b]]):>5.3f}" for b in BANDS))
print(f"  {'关节 中位 ρ²':<20}" + "".join(f"{np.median(accq[b]):>13.3f}" for b in BANDS))
print("\n  **两个指标不是一回事**（预注册的 0.986/0.884 是 resid 口径，判据用的是 ρ²）")
print("  噪声模型 trial = 信号 + 独立噪声，令 r = S/(S+N)：")
print("      试次间 ρ² = r²        试次间 resid = 2r−1        **模型可达 ρ² = r**")
print(f"\n  {'带':<12}{'r(由ρ²)':>10}{'r(由resid)':>12}{'模型可达 ρ²':>13}{'其 80%':>9}{'冻结门槛':>10}")
FROZEN = {(0.5,3.0): 0.789, (3.0,9.0): 0.707}
for b in BANDS:
    if b not in FROZEN: continue
    r_a = math.sqrt(max(np.median([x[0] for x in acc[b]]), 0.0))
    r_b = (np.median([x[1] for x in acc[b]]) + 1) / 2
    r = r_a
    print(f"  {f'{b[0]}-{b[1]} Hz':<12}{r_a:>10.3f}{r_b:>12.3f}{r:>13.3f}{0.8*r:>9.3f}{FROZEN[b]:>10.3f}"
          + ("   **门槛高于 80%, 更严**" if FROZEN[b] > 0.8*r else "   门槛低于 80%")
          + ("" if FROZEN[b] <= r else "   ⚠**门槛超过上界, 不可达**"))
print("\n  对照 probe_b2 **同会话** resid 口径: 0.5-3 Hz 0.986 / 3-9 Hz 0.884")
print("  本表是 pp2 **跨会话**(绑带重装) ⇒ 比同会话保守。门槛事前冻结, 不随本结果浮动。")
