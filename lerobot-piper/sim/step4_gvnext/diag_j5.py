#!/usr/bin/env python3
"""**j5 线索的诊断**（不是搜索，无门禁）—— 它现在还是最弱的那个吗，弱在哪。

已知（旧口径、G-q-v1、诊断集）：
  j5 占 3–9 Hz **绝对速度误差**的 39.6%，误差 ≈ 信号本身（误差/信号 0.97），
  且**不是时序问题**（最优位移 0.5 帧，对齐后 0.972→0.966）。
  各关节误差/信号：j2 0.54 · j1 0.63 · j6 0.72 · j3 0.81 · **j5 0.97**。

本脚本在**新数据 + G-2** 上重测，并问四个问题：
  ① 加了 r_q 之后 j5 还是最弱的吗（速度层，绝对量）
  ② 误差随**夹爪状态**变吗（负载相关？j5 是腕部俯仰，托着夹爪+物体）
  ③ 误差随**位形**变吗（用 j2+j3 代表臂的伸展）
  ④ 误差随**速度大小**变吗（非线性摩擦？）

口径与全项目一致：真机 100 Hz→9 Hz 低通→20 Hz；关节真值来自 can_q200；
逐系列最优位移；留出 = 会话2 的 pp2f + pp2f60。

用法:
    python -u sim/step4_gvnext/diag_j5.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g", ROOT / "sim/step3_canobs", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data import load, bp, q_at, valid_files, FPS                 # noqa: E402
from train_rq import features as rq_features, fir_design, MOVING  # noqa: E402
from eval_gate import base_q                                      # noqa: E402

S2 = ROOT / "outputs/step4_gvnext/s2"
RQ = ROOT / "outputs/step4_gvnext/rq"
BAND = (3.0, 9.0)
GRIP_OPEN = 0.030                      # m，接近全开
GRIP_SHUT = 0.005                      # m，接近闭合（多半在夹持）


def main() -> int:
    import gc
    gc.disable()
    zq = np.load(RQ / "rq_fir.npz")
    sq = json.loads((RQ / "select.json").read_text())
    EP = []
    for f in valid_files(S2, "replay_pp2f*.json"):
        dd = load(f)
        t20 = dd["tg_i"][::5]
        Xq, _, _ = rq_features(dd)
        qb = base_q(dd, t20)
        qh = qb + fir_design(Xq, sq["nlag"]) @ zq["W"]
        qt = q_at(dd, t20)
        g = np.interp(t20, dd["t_read"], dd["cmd"][:, 6])
        EP.append({"name": dd["name"], "qt": qt, "qb": qb, "qh": qh, "g": g})

    # ── ① 速度层的绝对误差（与旧诊断同口径）────────────────────────────
    print("① 3–9 Hz **关节速度**的绝对量（rad/s RMS，留出 8 条）\n")
    print(f"  {'关节':<5}{'真值 RMS':>10}{'G-q-v1 误差':>13}{'G-2 误差':>11}"
          f"{'误差/信号':>11}{'占总误差':>10}")
    sig = {j: 0.0 for j in MOVING}
    eb = {j: 0.0 for j in MOVING}
    eh = {j: 0.0 for j in MOVING}
    n = 0
    for e in EP:
        vt = np.gradient(e["qt"], 1 / FPS, axis=0)
        vb = np.gradient(e["qb"], 1 / FPS, axis=0)
        vh = np.gradient(e["qh"], 1 / FPS, axis=0)
        n += len(vt)
        for j in MOVING:
            sig[j] += float(bp(vt[:, j:j + 1], *BAND)[:, 0] @ bp(vt[:, j:j + 1], *BAND)[:, 0])
            eb[j] += float(((bp(vt[:, j:j+1], *BAND) - bp(vb[:, j:j+1], *BAND))[:, 0] ** 2).sum())
            eh[j] += float(((bp(vt[:, j:j+1], *BAND) - bp(vh[:, j:j+1], *BAND))[:, 0] ** 2).sum())
    tot = sum(np.sqrt(eh[j] / n) for j in MOVING)
    for j in MOVING:
        s_, b_, h_ = np.sqrt(sig[j] / n), np.sqrt(eb[j] / n), np.sqrt(eh[j] / n)
        print(f"  j{j+1:<4}{s_:>10.4f}{b_:>13.4f}{h_:>11.4f}{h_/max(s_,1e-12):>11.3f}"
              f"{h_/tot*100:>9.1f}%")

    # ── ②③④ j5 的误差和什么相关 ──────────────────────────────────────
    print(f"\n② 夹爪状态分层（j5，3–9 Hz 速度误差/信号）")
    bins = {"開 (>30mm)": [], "中": [], "閉 (<5mm)": []}
    for e in EP:
        vt = np.gradient(e["qt"], 1 / FPS, axis=0)
        vh = np.gradient(e["qh"], 1 / FPS, axis=0)
        a = bp(vt[:, 4:5], *BAND)[:, 0]; b = bp(vh[:, 4:5], *BAND)[:, 0]
        for lb, m in (("開 (>30mm)", e["g"] > GRIP_OPEN),
                      ("閉 (<5mm)", e["g"] < GRIP_SHUT),
                      ("中", (e["g"] >= GRIP_SHUT) & (e["g"] <= GRIP_OPEN))):
            if m.sum() > 40:
                bins[lb].append((float(((a - b)[m] ** 2).sum()), float((a[m] ** 2).sum()),
                                 int(m.sum())))
    for lb, v in bins.items():
        if not v:
            print(f"  {lb:<12} 样本不足"); continue
        e_ = sum(x[0] for x in v); s_ = sum(x[1] for x in v); k = sum(x[2] for x in v)
        print(f"  {lb:<12}{np.sqrt(e_/max(s_,1e-30)):>8.3f}   ({k} 帧)")

    for tag, keyf, edges in (
        ("③ 位形分层（j2+j3，代表臂的伸展）", lambda e: np.degrees(e["qt"][:, 1] + e["qt"][:, 2]), None),
        ("④ |j5 速度| 分层", lambda e: np.abs(np.gradient(e["qt"], 1/FPS, axis=0)[:, 4]), None)):
        vals = np.concatenate([keyf(e) for e in EP])
        qs = np.percentile(vals, [0, 33, 67, 100])
        print(f"\n{tag}  三分位 {np.round(qs,2)}")
        for lo, hi, lb in ((qs[0], qs[1], "低"), (qs[1], qs[2], "中"), (qs[2], qs[3], "高")):
            e_ = s_ = 0.0; k = 0
            for e in EP:
                vt = np.gradient(e["qt"], 1/FPS, axis=0); vh = np.gradient(e["qh"], 1/FPS, axis=0)
                a = bp(vt[:, 4:5], *BAND)[:, 0]; b = bp(vh[:, 4:5], *BAND)[:, 0]
                m = (keyf(e) >= lo) & (keyf(e) <= hi)
                e_ += float(((a - b)[m] ** 2).sum()); s_ += float((a[m] ** 2).sum()); k += int(m.sum())
            print(f"  {lb:<4}{np.sqrt(e_/max(s_,1e-30)):>8.3f}   ({k} 帧)")
    print("\n  误差/信号 1.0 = 与「预测全零」等效。")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
