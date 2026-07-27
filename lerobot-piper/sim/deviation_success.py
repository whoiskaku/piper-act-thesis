#!/usr/bin/env python3
"""成功集的"执行 vs 理想(命令)"偏离审计 — 找老师关心的问题(偏离理想/过冲)。

只用 no-TE 成功集(stage3); 执行=observation.state(→rad), 理想=action(rad), 臂 j1/j2/j3。
① #77 口径整体: 滞后lag / 跟踪RMS / 滞后补偿后 低频(轨迹失真+过冲) vs 高频(振动) / corr。
② chunk 边界处: 命令在边界大跳(4.3×), 看执行如何响应 —— 边界对齐的 执行-命令 偏离 +
   阶跃响应归一化(峰值>1 = 过冲, 老师⑤的假设)。
③ 时域示例: 一条成功集一关节, 命令(理想) vs 执行(真机), 标 chunk 边界 —— 给老师"看得见"。

用法: /home/enolab-guo/miniconda3/envs/lerobot/bin/python sim/deviation_success.py
"""
from __future__ import annotations
import csv
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LR = Path.home() / ".cache/huggingface/lerobot/enolab-guo"
OUT = Path(__file__).resolve().parents[1] / "outputs/eval/deviation_success.png"
REPO = "eval_phase1_noimu_no_te"
FS, CHUNK, ARM = 30.0, 100, [0, 1, 2]
MDEG, R2D, MAXLAG = np.pi / 180 / 1000, 180 / np.pi, 15
_LO = butter(4, 2.0 / 15.0, "low")
_HI = butter(4, 2.0 / 15.0, "high")
W0, W1 = -8, 32                                    # 边界对齐窗


def best_lag(e, c):
    best_d, best_r = 0, -2.0
    for d in range(MAXLAG + 1):
        a, b = e[d:], c[:len(c) - d]
        if len(a) < 30 or a.std() < 1e-9 or b.std() < 1e-9:
            continue
        r = np.corrcoef(a, b)[0, 1]
        if r > best_r:
            best_r, best_d = r, d
    return best_d


def main():
    stages = {}
    for r in csv.reader((Path(__file__).resolve().parents[1] / "outputs/review" /
                         f"enolab-guo__{REPO}_eval.csv").open()):
        if len(r) >= 2:
            stages[int(r[0])] = int(r[1])
    succ = sorted(ep for ep, s in stages.items() if s == 3)

    raw, lo, hi, mot, corrs, lags = [], [], [], [], [], []
    bnd_dev = []                                    # 边界对齐 |执行-命令| (lag补偿后)
    step_resp = []                                  # 边界阶跃响应(归一化)
    example = None
    for ep in succ:
        pq = next(glob.iglob(str(LR / REPO / f"data/chunk-*/episode_{ep:06d}.parquet")))
        df = pd.read_parquet(pq)
        act = np.stack(df["action"].to_numpy()).astype(float)
        st = np.stack(df["observation.state"].to_numpy()).astype(float)
        st[:, :6] *= MDEG
        for j in ARM:
            c, e = act[:, j], st[:, j]
            if len(c) < 60 or c.std() < 1e-9:
                continue
            d = best_lag(e, c)
            ec, cc = e[d:], c[:len(c) - d]
            dv = ec - cc
            raw.append(rms_arr(e - c)); lo.append(filtfilt(*_LO, dv)); hi.append(filtfilt(*_HI, dv))
            mot.append(cc - cc.mean()); corrs.append(np.corrcoef(ec, cc)[0, 1]); lags.append(d)
            # 边界对齐(用原始未补偿, 对齐到命令边界)
            for b in range(CHUNK, len(c), CHUNK):
                if b + W1 < len(c) and b + W0 >= 0:
                    seg_e, seg_c = e[b + W0:b + W1], c[b + W0:b + W1]
                    bnd_dev.append(np.abs(seg_e - seg_c))
                    step = c[b + W1 - 1] - c[b - 1]      # 该 chunk 命令净步
                    if abs(step) > 0.02:                  # 显著步(rad)才算阶跃
                        step_resp.append((e[b + W0:b + W1] - e[b - 1]) / step)
            if example is None and j == 1:
                example = (ep, j, c.copy(), e.copy(), d)

    cat = np.concatenate
    rms = lambda x: float(np.sqrt((cat(x) ** 2).mean()) * R2D)
    print(f"\n{'='*64}\n成功集(n={len(succ)}) 执行 vs 理想(命令) 偏离审计  [臂 j1/j2/j3, 单位°]\n{'='*64}")
    print(f"  corr(执行,命令) = {np.mean(corrs):.3f}   滞后 lag = {np.mean(lags)/FS*1e3:.0f} ms")
    print(f"  跟踪RMS(含滞后) = {rms(raw):.2f}°")
    print(f"  滞后补偿后: 低频(轨迹失真/过冲) = {rms(lo):.2f}°   高频(振动) = {rms(hi):.2f}°")
    print(f"  运动幅度(参考) = {rms(mot):.2f}°   → 低频偏离占运动 {rms(lo)/rms(mot)*100:.1f}%")

    sr = np.array(step_resp); k = np.arange(W0, W1) / FS
    bd = np.array(bnd_dev).mean(0) * R2D
    sr_mean = sr.mean(0) if len(sr) else np.zeros(W1 - W0)
    overshoot = (sr_mean.max() - 1) * 100 if len(sr) else float("nan")
    print(f"\n  chunk 边界阶跃响应(n={len(sr)} 显著步): 峰值 {sr_mean.max():.2f} (1.0=刚好到位)"
          f" → 过冲 {overshoot:+.0f}%")
    print(f"  边界处 |执行-命令| 峰值 = {bd.max():.2f}° (窗内基线 {bd[:5].mean():.2f}°)")

    # ---- 图 ----
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    ep, j, c, e, d = example
    t = np.arange(len(c)) / FS
    ax[0].plot(t, c * R2D, "C0", lw=1.3, label="ACT command (ideal)")
    ax[0].plot(t, e * R2D, "C3", lw=1.1, label="executed (real state)")
    for b in range(CHUNK, len(c), CHUNK):
        ax[0].axvline(b / FS, color="gray", ls=":", lw=.8)
    ax[0].set(title=f"ep{ep} j{j+1}: 命令(理想) vs 执行 (灰线=chunk边界)", xlabel="s", ylabel="角度°")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(k, bd, "C3", lw=1.6); ax[1].axvline(0, color="gray", ls="--", lw=1)
    ax[1].set(title="边界对齐 平均 |执行-命令|", xlabel="到边界 s", ylabel="偏离°"); ax[1].grid(alpha=.3)
    if len(sr):
        ax[2].plot(k, sr_mean, "C3", lw=1.6, label="executed (归一化)")
        ax[2].axhline(1.0, color="C0", ls="--", lw=1, label="command 目标=1")
        ax[2].axvline(0, color="gray", ls=":", lw=1)
        ax[2].set(title=f"chunk边界 阶跃响应(峰{sr_mean.max():.2f}, >1=过冲)", xlabel="到边界 s", ylabel="归一化位置")
        ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    fig.suptitle("成功集: 执行 vs 理想(命令) — 偏离/过冲 主要在 chunk 边界?", fontsize=12)
    fig.tight_layout(); OUT.parent.mkdir(parents=True, exist_ok=True); fig.savefig(OUT, dpi=130)
    print(f"\n[saved] {OUT}")


def rms_arr(x):
    return x


if __name__ == "__main__":
    main()
