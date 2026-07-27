#!/usr/bin/env python3
"""逐集 2-4.5Hz IMU 带功率分布: demo(全集) vs no-TE(全集)。

回答"对比是不是挑了某一条": 把每一集的带功率画成散点, 看 no-TE 是否一致地
高于 demo 的分布, 而非个别 episode 凑巧。长度无关(Welch PSD 积分)。

用法: /home/enolab-guo/miniconda3/envs/lerobot/bin/python sim/plot_imu_band_distribution.py
"""
from __future__ import annotations
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LR = Path.home() / ".cache/huggingface/lerobot/enolab-guo"
OUT = Path(__file__).resolve().parents[1] / "outputs/eval/imu_band_distribution.png"
FS, NPERSEG = 30.0, 256
DS = {"demo (teleop)": ("pickplace_clean", "C0"), "deploy no-TE": ("eval_phase1_noimu_no_te", "C3")}
RNG = np.random.default_rng(0)                      # 仅用于散点抖动(可复现)


def bp(sig):
    f, P = welch(sig, fs=FS, nperseg=min(NPERSEG, len(sig)))
    m = (f >= 2.0) & (f < 4.5)
    return float(np.trapezoid(P[m], f[m]))


def amag(im):
    return np.linalg.norm(im[:, :3] - im[:, :3].mean(0), axis=1)


def gmag(im):
    return np.linalg.norm(im[:, 3:] - im[:, 3:].mean(0), axis=1)


def per_ep(repo, skip=1):
    out = []
    for pq in sorted(glob.glob(str(LR / repo / "data/chunk-*/episode_*.parquet")))[skip:]:
        im = np.stack(pd.read_parquet(pq)["observation.imu_real"].to_numpy()).astype(float)
        out.append((bp(amag(im)), bp(gmag(im))))
    return np.array(out)


def main():
    data = {lab: per_ep(repo) for lab, (repo, _) in DS.items()}
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for j, title in enumerate(["accel  2-4.5Hz band power", "gyro  2-4.5Hz band power"]):
        a = ax[j]
        for i, (lab, (repo, c)) in enumerate(DS.items()):
            v = data[lab][:, j]
            x = i + RNG.uniform(-0.09, 0.09, len(v))
            a.scatter(x, v, s=28, c=c, alpha=.6, edgecolors="none", label=f"{lab} (n={len(v)})")
            a.hlines(np.median(v), i - 0.22, i + 0.22, color=c, lw=2.5)     # 中位数
        a.set_xticks([0, 1]); a.set_xticklabels([k for k in DS])
        a.set_yscale("log"); a.set_ylabel("band power (log)"); a.set_title(title)
        a.grid(alpha=.3, axis="y"); a.legend(fontsize=8)
    fig.suptitle("Per-episode IMU 2-4.5Hz band power: demo vs no-TE  (横线=中位数; 看分布是否分离)",
                 fontsize=12)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)

    print(f"\n{'='*64}\n逐集 2-4.5Hz 带功率: 中位数 [最小, 最大] (n)\n{'='*64}")
    for sig, j in [("accel", 0), ("gyro", 1)]:
        print(f"\n[{sig}]")
        for lab in DS:
            v = data[lab][:, j]
            print(f"  {lab:<16} 中位 {np.median(v):.4f}  [{v.min():.4f}, {v.max():.4f}]  n={len(v)}")
        d, n = data["demo (teleop)"][:, j], data["deploy no-TE"][:, j]
        n_above = int((n > np.median(d)).sum())
        print(f"  → no-TE 有 {n_above}/{len(n)} 集高于 demo 中位数; "
              f"no-TE 最小({n.min():.4f}) vs demo 中位({np.median(d):.4f})")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
