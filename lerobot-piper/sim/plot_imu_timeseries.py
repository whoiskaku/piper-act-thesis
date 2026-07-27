#!/usr/bin/env python3
"""demo vs no-TE 部署: 原始 IMU 6 通道随时间变化对比。

代表性选集: 各数据集取 accel 2-4.5Hz 带功率**最接近该集合中位数**的那条 episode
(= 典型抖动水平的代表集, 非离群、非挑选)。可用 argv 指定固定编号覆盖。

注: demo 与 no-TE 是不同 episode(轨迹/物体位置/时长不同), 同一物理机器人+IMU;
低频差异来自不同轨迹(重力投影随姿态), 高频毛刺=振动(no-TE 的 chunk 边界更明显)。

用法: /home/enolab-guo/miniconda3/envs/lerobot/bin/python sim/plot_imu_timeseries.py [demo_ep no_te_ep]
"""
from __future__ import annotations
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LR = Path.home() / ".cache/huggingface/lerobot/enolab-guo"
OUT = Path(__file__).resolve().parents[1] / "outputs/eval/imu_timeseries_demo_vs_noTE.png"
FS, NPERSEG = 30.0, 256
CH = ["ax", "ay", "az", "gx", "gy", "gz"]
DS = {"demo (teleop)": ("pickplace_clean", "C0"), "deploy no-TE": ("eval_phase1_noimu_no_te", "C3")}


def load_all(repo, skip=1):
    out = []
    for pq in sorted(glob.glob(str(LR / repo / "data/chunk-*/episode_*.parquet")))[skip:]:
        ep = int(Path(pq).stem.split("_")[1])
        im = np.stack(pd.read_parquet(pq)["observation.imu_real"].to_numpy()).astype(float)
        out.append((ep, im))
    return out


def amag(im):
    return np.linalg.norm(im[:, :3] - im[:, :3].mean(0), axis=1)


def bp(sig):
    f, P = welch(sig, fs=FS, nperseg=min(NPERSEG, len(sig)))
    m = (f >= 2.0) & (f < 4.5)
    return float(np.trapezoid(P[m], f[m]))


def pick_median(eps):
    bps = np.array([bp(amag(im)) for _, im in eps])
    idx = int(np.argmin(np.abs(bps - np.median(bps))))     # 最接近中位数的集
    return eps[idx][0], eps[idx][1], bps[idx], float(np.median(bps))


def main():
    args = sys.argv[1:1 + len(DS)]               # 每数据集一个: 整数=指定集, 'm'/省略=中位代表集
    sel = {}
    for i, (lab, (repo, _)) in enumerate(DS.items()):
        eps = load_all(repo)
        spec = args[i] if i < len(args) else "m"
        if spec in ("m", ""):
            sel[lab] = pick_median(eps)
        else:
            d = dict(eps); ep = int(spec)
            med = float(np.median([bp(amag(im)) for _, im in eps]))
            sel[lab] = (ep, d[ep], bp(amag(d[ep])), med)

    fig, ax = plt.subplots(3, 2, figsize=(15, 9), sharex=True)
    for r, c, ci in [(0, 0, 0), (1, 0, 1), (2, 0, 2), (0, 1, 3), (1, 1, 4), (2, 1, 5)]:
        a = ax[r, c]
        for lab, (repo, col) in DS.items():
            ep, im, _, _ = sel[lab]
            a.plot(np.arange(len(im)) / FS, im[:, ci], col, lw=0.7, alpha=.85,
                   label=f"{lab} ep{ep}")
        a.set_ylabel(CH[ci]); a.grid(alpha=.3)
        if r == 0:
            a.set_title("accel  (raw, m/s²)" if c == 0 else "gyro  (raw, rad/s)")
        if r == 0 and c == 1:
            a.legend(fontsize=9, loc="upper right")
    ax[2, 0].set_xlabel("time (s)"); ax[2, 1].set_xlabel("time (s)")
    fig.suptitle("Raw IMU over time: demo vs no-TE  (representative = median-band-power episode; same Piper)",
                 fontsize=12)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"[saved] {OUT}")
    for lab, (ep, im, b, med) in sel.items():
        print(f"  {lab}: 代表集 ep{ep} ({len(im)/FS:.1f}s) accel带功率={b:.4f} (该集合中位={med:.4f})")


if __name__ == "__main__":
    main()
