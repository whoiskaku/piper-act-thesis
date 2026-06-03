#!/usr/bin/env python3
"""Stage 5 — 低层控制器跟踪诊断(commanded action → measured joint state)。

Stage5 链路诊断发现最大瓶颈在 action→state(指令 hp 降40%但执行态 hp 只降18%)。
本脚本在真机 A/B 数据上做**频率分辨**分析, 定位收益在哪丢的:

  对每个 active 臂关节(j1/j2/j3, action 与 state 均可换算到 rad):
    · PSD 分带能量(0-5 / 5-10 / 10-15 Hz)of commanded action 与 measured state
    · 控制器带增益 gain(f) = sqrt(P_state / P_action)  —— <1 表示该频段被衰减(低通)
    · coherence(action, state) 分带 —— state 中有多少被 action 线性解释
    · 指令无关地板: floor = (1 - coh²)·P_state  —— RefineNet 根本碰不到的部分
  A(纯ACT) vs B(ACT+RefineNet): 指令变平滑后, state 的 command-driven 高频降没降。

判读:
  gain 高频<<1 → 控制器低通, 指令高频本就到不了关节 → RefineNet 高频收益被控制器吞
  floor 占 state 高频大头 → state 高频主要是指令无关(量化/控制器抖/传感器) → 不可由动作修

用法: python scripts/stage5_controller_diagnosis.py
"""
from __future__ import annotations
import glob, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import welch, coherence

LR = Path.home() / ".cache/huggingface/lerobot/enolab-guo"
FS = 30.0
ARM = [0, 1, 2]                     # j1/j2/j3: action 与 state 同为关节角(rad)
MDEG_TO_RAD = math.pi / 180.0 / 1000.0
BANDS = [(0, 5), (5, 10), (10, 15)]


def load(ds):
    out = []
    for pq in sorted(glob.glob(str(LR / ds / "data/chunk-*/episode_*.parquet"))):
        df = pd.read_parquet(pq)
        act = np.stack(df["action"].to_numpy()).astype(np.float64)
        st = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
        st[:, :6] *= MDEG_TO_RAD                    # 臂关节 mdeg→rad
        out.append((act, st))
    return out


def bandpow(x, lo, hi):
    f, p = welch(x, fs=FS, nperseg=min(256, len(x)))
    m = (f >= lo) & (f < hi)
    return p[m].sum()


def coh_band(x, y, lo, hi):
    f, c = coherence(x, y, fs=FS, nperseg=min(256, len(x)))
    m = (f >= lo) & (f < hi)
    return c[m].mean() if m.any() else float("nan")


def analyze(eps, label):
    # 聚合所有 ep、所有臂关节
    rows = {b: {"Pa": [], "Ps": [], "coh": []} for b in BANDS}
    for act, st in eps:
        for j in ARM:
            for b in BANDS:
                rows[b]["Pa"].append(bandpow(act[:, j], *b))
                rows[b]["Ps"].append(bandpow(st[:, j], *b))
                rows[b]["coh"].append(coh_band(act[:, j], st[:, j], *b))
    print(f"\n[{label}]  (臂关节 j1/j2/j3 聚合; Pa=指令带能量, Ps=执行态带能量)")
    print(f"  {'band(Hz)':>9} {'gain√(Ps/Pa)':>13} {'coh²(解释比)':>13} {'floor占state%':>13}")
    res = {}
    for b in BANDS:
        Pa, Ps, coh = np.mean(rows[b]["Pa"]), np.mean(rows[b]["Ps"]), np.nanmean(rows[b]["coh"])
        gain = math.sqrt(Ps / Pa) if Pa > 0 else float("nan")
        floor_frac = (1 - coh ** 2)                 # state 中指令无关比例
        res[b] = {"Pa": Pa, "Ps": Ps, "gain": gain, "coh2": coh ** 2, "floor": floor_frac}
        print(f"  {b[0]:>3}-{b[1]:<5} {gain:>13.3f} {coh**2:>13.3f} {100*floor_frac:>12.0f}%")
    return res


def main():
    A = load("eval_refine_A"); B = load("eval_refine_B")
    print("=" * 70)
    print("Stage 5 低层控制器跟踪诊断  (commanded action → measured joint state)")
    print("=" * 70)
    rA = analyze(A, "A: 纯 ACT")
    rB = analyze(B, "B: ACT+RefineNet")

    print("\n[A→B 高频(5-15Hz)指令 vs 执行态的变化]")
    for lab, r in [("A", rA), ("B", rB)]:
        hi_a = r[(5, 10)]["Pa"] + r[(10, 15)]["Pa"]
        hi_s = r[(5, 10)]["Ps"] + r[(10, 15)]["Ps"]
        print(f"  {lab}: 指令高频能量={hi_a:.2e}  执行态高频能量={hi_s:.2e}")
    haA = rA[(5, 10)]["Pa"] + rA[(10, 15)]["Pa"]; haB = rB[(5, 10)]["Pa"] + rB[(10, 15)]["Pa"]
    hsA = rA[(5, 10)]["Ps"] + rA[(10, 15)]["Ps"]; hsB = rB[(5, 10)]["Ps"] + rB[(10, 15)]["Ps"]
    print(f"  指令高频 B/A = {haB/haA:.3f}   执行态高频 B/A = {hsB/hsA:.3f}")
    print("\n判读:")
    print("  · 若 gain 在 5-15Hz << 1 → 控制器低通, 指令高频到不了关节(RefineNet 高频收益被吞)")
    print("  · 若 floor占state% 高 → 执行态高频主要指令无关(量化/控制器/传感器), 动作平滑改不动")
    print("  · 指令高频 B/A 明显<1 但 执行态高频 B/A≈1 → 收益确实在 action→state 这层丢失")


if __name__ == "__main__":
    main()
