#!/usr/bin/env python3
"""25 条 no-TE 部署逐集 IMU 随时间网格 + 汇总表。

每格: >2Hz |accel| 振动包络(红) + 有效运动终点(灰虚线, 关节速度判)。
看 25 条异质性: 活跃时长 / 空转尾巴 / 振动幅度 / 是否撞 30s 上限。
标题 bp = 该集 2-4.5Hz accel 带功率; 红色标题=高于 demo 中位。

用法: /home/enolab-guo/miniconda3/envs/lerobot/bin/python sim/plot_imu_grid_noTE.py
"""
from __future__ import annotations
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LR = Path.home() / ".cache/huggingface/lerobot/enolab-guo"
OUT = Path(__file__).resolve().parents[1] / "outputs/eval/imu_grid_noTE.png"
REPO = "eval_phase1_noimu_no_te"
FS, NPERSEG = 30.0, 256
_HP = butter(4, 2.0 / (FS / 2), "high")


def amag_hp(im):
    a = np.linalg.norm(im[:, :3] - im[:, :3].mean(0), axis=1)
    return filtfilt(*_HP, a)


def bp(im):
    a = np.linalg.norm(im[:, :3] - im[:, :3].mean(0), axis=1)
    f, P = welch(a, fs=FS, nperseg=min(NPERSEG, len(a)))
    m = (f >= 2.0) & (f < 4.5)
    return float(np.trapezoid(P[m], f[m]))


def active_end(st):
    v = np.linalg.norm(np.diff(st[:, :6], axis=0), axis=1)
    v = np.convolve(v, np.ones(9) / 9, mode="same")
    idx = np.where(v > 0.1 * np.percentile(v, 95))[0]
    return ((idx[-1] + 1) if len(idx) else len(st)) / FS


def load(repo, skip=0):
    out = []
    for pq in sorted(glob.glob(str(LR / repo / "data/chunk-*/episode_*.parquet")),
                     key=lambda p: int(Path(p).stem.split("_")[1]))[skip:]:
        df = pd.read_parquet(pq)
        ep = int(Path(pq).stem.split("_")[1])
        im = np.stack(df["observation.imu_real"].to_numpy()).astype(float)
        st = np.stack(df["observation.state"].to_numpy()).astype(float)
        out.append((ep, im, st))
    return out


def main():
    eps = load(REPO, skip=0)                                    # 全 25 条
    demo_bps = [bp(im) for _, im, _ in load("pickplace_clean", skip=1)]
    demo_med = float(np.median(demo_bps))

    n = len(eps); cols = 5; rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(16, 2.4 * rows), sharex=True, sharey=True)
    for k, (ep, im, st) in enumerate(eps):
        a = ax.flat[k]
        a.plot(np.arange(len(im)) / FS, amag_hp(im), "C3", lw=0.5)
        a.axvline(active_end(st), color="gray", ls=":", lw=1.2)
        b = bp(im)
        a.set_title(f"ep{ep}  {len(im)/FS:.0f}s  bp={b:.3f}", fontsize=8,
                    color=("red" if b > demo_med else "black"))
        a.grid(alpha=.2)
    for k in range(n, rows * cols):
        ax.flat[k].axis("off")
    fig.suptitle(f"no-TE 25 episodes: >2Hz |accel| vibration (red) + motion-end (gray dotted).  "
                 f"demo median band power = {demo_med:.3f} (title red = above it)", fontsize=11)
    fig.supxlabel("time (s)"); fig.supylabel("accel HF |a| (raw)")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"[saved] {OUT}")

    print(f"\n{'ep':>4}{'总时长':>8}{'有效运动':>9}{'空转尾':>8}{'accel_bp':>10}{'vs_demo中位':>11}")
    print("-" * 52)
    tots, acts, bps = [], [], []
    for ep, im, st in eps:
        tot, ae, b = len(im) / FS, active_end(st), bp(im)
        tots.append(tot); acts.append(ae); bps.append(b)
        print(f"{ep:>4}{tot:>7.1f}s{ae:>8.1f}s{tot-ae:>7.1f}s{b:>10.3f}{'高' if b > demo_med else '低':>10}")
    print("-" * 52)
    tots, acts, bps = np.array(tots), np.array(acts), np.array(bps)
    print(f"{'中位':>4}{np.median(tots):>7.1f}s{np.median(acts):>8.1f}s"
          f"{np.median(tots-acts):>7.1f}s{np.median(bps):>10.3f}")
    print(f"demo accel band power 中位={demo_med:.3f}; no-TE 高于它的: {int((bps>demo_med).sum())}/{n}")
    print(f"撞 ~30s 上限(总>29s): {int((tots>29).sum())}/{n}; 有效运动>25s: {int((acts>25).sum())}/{n}")


if __name__ == "__main__":
    main()
