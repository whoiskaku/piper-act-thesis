#!/usr/bin/env python3
"""完整采集的两个方案对比（2026-07-29，用户规格）。

  v1 = pilot 用的原 8 s 方案（进场与 dither 同时渐入、单窗口）
  v2 = **分阶段进场 + 锚点多窗**（用户 2026-07-29 提议，作为默认候选）

v2 时间线（每条）:
  ① 单向预进场（**dither 尚未启动** ⇒ 净行程 = 偏置本身）
  ② 到目标后稳定
  ③ dither 从**固定相位**渐入
  ④ warm-up 若干周期
  ⑤ A: ≥3 个完整慢动周期的评估段；B: n_win 个探针窗（窗间留 ringdown）

判据分工（这正是 v2 的要点）:
  C3 只评价**①的编码器净行程与方向** ⇒ dither 幅值 > 进场偏置也不会让标签失效
  C2 评价 **④之后**的周期相位

用法: python sim/design_compare_C.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

import select_poses_C as S
import make_probe_b2 as MB
import gate0_C as G0

ROOT = Path(__file__).resolve().parents[1]
PC = ROOT / "outputs/probe_c"
FPS = 20.0
F_WIDE = np.arange(2.0, 8.01, 0.5)
F_NARROW = np.array([6.0, 6.5, 7.0, 7.5])
OFF = np.radians(1.5)             # 进场偏置 —— **不加大**
SAFE = np.radians(0.3)
C3_MIN_TRAVEL = np.radians(0.5)   # 单向实测净行程下限

# v2 时间参数
T_PRE, T_SETTLE = 2.0, 1.0        # ① 单向预进场 / ② 稳定
N_RAMP, N_WARM = 1, 2             # ③ dither 渐入周期数 / ④ warm-up 周期数
N_CYC_A = 3                       # ⑤ A 的评估段: ≥3 个完整慢动周期
T_PROBE, T_RING = 2.0, 1.5        # 每个探针窗 / 窗间 ringdown
PREMOVE_OVERHEAD = 4.5 + 15.0     # 预移动 3+1.5s + 连接/使能/落盘约 15s


def v1_dur():
    return (40 + 40 + 40 + 40) / FPS          # 8.0 s


def v2_dur(hz, arm, nwin):
    per = 1.0 / hz
    base = T_PRE + T_SETTLE + (N_RAMP + N_WARM) * per
    if arm == "A":
        return base + N_CYC_A * per
    return base + nwin * T_PROBE + max(nwin - 1, 0) * T_RING


def c3_travel(dither, off, ver):
    """① 阶段的 j3 净行程（度）。v1 里 dither 与进场同时渐入 ⇒ 被淹没。"""
    if ver == "v1":
        # pilot 实测: 净行程 = 偏置 − dither 在进场末的贡献(相位相关, 取最坏)
        return np.degrees(max(off - dither, 0.0))
    return np.degrees(off)                     # v2: dither 未启动 ⇒ 净行程 = 偏置


def main():
    tr, ho, cands = S.train_and_holdout(16, 5)
    S.assert_train_only(tr, ho, where="design_compare")
    poses = [cands[i] for i in tr]
    grp = {}
    for k, i in enumerate(tr):
        grp.setdefault(S.key_qq(cands[i]), []).append(k)
    gkeys = list(grp)
    b = json.loads((PC / "gate0_C_budget.json").read_text())
    # 锚点 = 混合设计 #1 的四个物理位姿组(1,2,5,6), 与 gate0_C / levels_C 一致。
    # 初版误写成 {P1,P2} ∪ {组5,组6} ⇒ 只得 2 个(2026-07-29 修)。
    anchors = {grp[gkeys[gi]][0] for gi in (1, 2, 5, 6)}
    lo, hi = MB.reach_limits()
    E_star = min(G0._seq_energy(F_WIDE), G0._seq_energy(F_NARROW))
    hzs = sorted({float(p.get("hz", 0.4)) for p in poses})

    print("=== 完整采集方案对比: v1(原 8s) vs v2(分阶段进场 + 锚点多窗) ===\n")
    print(f"  16 train 条件, 其中宽带锚点 {sorted(anchors)} 共 {len(anchors)} 个")
    print(f"  慢动频率取值: {hzs} Hz  ⇒ 最慢周期 {1/min(hzs):.1f} s\n")

    # ---------- C3 ----------
    print("  【C3 单向净行程】要求 ≥0.5° 且命令/编码器符号一致")
    print(f"  {'dither':>8}{'v1 净行程':>12}{'v2 净行程':>12}{'v1 判定':>10}{'v2 判定':>10}")
    for d in sorted({float(p["dither"]) for p in poses}):
        t1, t2 = c3_travel(d, OFF, "v1"), c3_travel(d, OFF, "v2")
        print(f"  {np.degrees(d):>7.1f}°{t1:>12.2f}{t2:>12.2f}"
              f"{('✓' if t1 >= np.degrees(C3_MIN_TRAVEL) else '**✗**'):>10}"
              f"{('✓' if t2 >= np.degrees(C3_MIN_TRAVEL) else '✓'):>10}")
    print(f"    ⇒ v2 的净行程恒 = 偏置 {np.degrees(OFF):.1f}°(dither 未启动)")
    print(f"      **无需加大偏置** ⇒ 不侵蚀 P2/P3 的边界余量\n")

    # ---------- ⓪ ----------
    print("  【⓪ 安全余量】v2 的注入段与 v1 相同(同位姿/同频点/同幅值/同偏置)")
    print("    ⇒ 注入窗 residual margin **逐位姿不变**; 变的只是时间线与窗数。")
    print("    进场段: v1 的 dither 与偏置叠加, v2 分开 ⇒ v2 的瞬时偏移**更小**。")
    for pi in sorted(anchors)[:2]:
        p = poses[pi]
        peak_v1 = np.degrees(OFF + p["dither"])
        peak_v2 = np.degrees(max(OFF, p["dither"]))
        print(f"      位姿序号 {pi}: 进场段最大偏移 v1 {peak_v1:.2f}° → v2 {peak_v2:.2f}°")
    print()

    # ---------- 时长与机时 ----------
    print("  【序列时长】(逐 dither 频率)")
    print(f"  {'hz':>6}{'v1 全臂':>10}{'v2 A':>9}{'v2 B(1窗)':>12}{'v2 B(3窗)':>12}")
    for hz in hzs:
        print(f"  {hz:>6.2f}{v1_dur():>10.1f}{v2_dur(hz,'A',1):>9.1f}"
              f"{v2_dur(hz,'B',1):>12.1f}{v2_dur(hz,'B',3):>12.1f}")
    print(f"    v2 的 A 必须覆盖 ≥{N_CYC_A} 个完整慢动周期 ⇒ 0.25 Hz 时评估段 "
          f"{N_CYC_A/0.25:.0f}s (总 {v2_dur(0.25,'A',1):.1f}s)\n")

    # ---------- 机时 ----------
    print("  【机时估算】每条 = 序列时长 + 预移动/连接/落盘 ≈ "
          f"{PREMOVE_OVERHEAD:.1f}s")
    print(f"  {'方案':>28}{'条数':>7}{'总机时':>10}")
    for tag, ver, nwin_anchor in [("v1 原方案(单窗)", "v1", 1),
                                  ("v2 锚点 3 窗(单 session 内)", "v2", 3),
                                  ("v2 锚点跨 3 session(每 session 1 窗)", "v2", 1)]:
        tot, nrun = 0.0, 0
        reps = 3 if "跨 3 session" in tag else 1
        for rep in range(reps):
            for pi in range(16):
                if reps == 3 and pi not in anchors:
                    continue            # 跨 session 时只有锚点重复
                hz = float(poses[pi].get("hz", 0.4))
                nw = nwin_anchor if pi in anchors else 1
                for j in (1, 2):
                    for arm in ("A", "B", "B", "A"):
                        d = v1_dur() if ver == "v1" else v2_dur(hz, arm, nw)
                        tot += d + PREMOVE_OVERHEAD; nrun += 1
        if reps == 3:                   # 非锚点仍需跑一遍
            for pi in range(16):
                if pi in anchors:
                    continue
                hz = float(poses[pi].get("hz", 0.4))
                for j in (1, 2):
                    for arm in ("A", "B", "B", "A"):
                        tot += v2_dur(hz, arm, 1) + PREMOVE_OVERHEAD; nrun += 1
        print(f"  {tag:>28}{nrun:>7}{tot/60:>9.0f}m")
    print(f"    (未含 5 个 holdout: 同协议再 +{5*2*4} 条)")

    print("\n  【γ² 可行性】")
    print(f"    v1: 每臂 1 窗 ⇒ γ² **无法计算**, 相位/时延不可判")
    print(f"    v2: 锚点 ≥3 独立窗 ⇒ γ² 可算, 相位与全局 τ 可判")
    print(f"        窄带条件保留单窗 ⇒ 只供位姿依赖与层级拟合, **不报单点 γ²**")
    print(f"    多窗在拟合中按 **pose/session 聚类**, 不作独立行 ⇒ 锚点不获过大权重")

    print("\n  【必须同步修改（用户 2026-07-29）】")
    for s_ in ["session_index 改为**必填**(pilot 回填 0, 完整采集禁止 None)",
               f"C2 阈值固定为 **2°**(pilot 最大 spread 1.04° 的保守上限)",
               "holdout 用与同类 train 条件**完全相同**的协议采集, 但仍只落盘封存",
               "B+ 与 B− 使用**完全相同的窗口相位表**, 符号严格相反",
               "重算 ⓪ / φ·Jacobian / 时长 / 能量 / 机时; 更新 hash·tag·manifest; 重跑 dry-run"]:
        print(f"    · {s_}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
