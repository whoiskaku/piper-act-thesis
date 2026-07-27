#!/usr/bin/env python3
"""新数据集(20Hz)复算两个执行侧偏离: ① 执行-指令滞后 ② chunk 边界跳变。

① 滞后: 对每关节扫描平移 d∈0..8 帧, 取 corr 最大的 d* → ms。均匀性=关节间/集间一致。
② 边界: 无TE每100步(5s)重规划; 边界步|Δcmd|/段内|Δcmd|; IMU(2Hz高通)边界后0.5s/全段。
   demo 打假边界作对照(应≈1)。
数据: eval_pickplace2_100k / eval_insert_100k / eval_insert_200k + 各自 demo。
用法: python sim/lag_boundary_new.py
"""
from __future__ import annotations
import glob, os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LR = os.path.expanduser("~/.cache/huggingface/lerobot/enolab-guo")
OUT = os.path.expanduser("~/claude-test-2/lerobot-piper/outputs/eval/lag_boundary_new.png")
FS, CHUNK, MAXLAG = 20.0, 100, 8
ARM = [0, 1, 2]
W0, W1 = -20, 40
_HP = butter(4, 2.0 / (FS / 2), "high")
R2D = 180 / np.pi
BLUE, GRAY = "#2a78d6", "#898781"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"

SETS = [
    ("pickplace2", "eval_pickplace2_100k", "pickplace2_train"),
    ("insert@100K", "eval_insert_100k", "insert_train"),
    ("insert@200K", "eval_insert_200k", "insert_train"),
]


def load(repo):
    out = []
    for p in sorted(glob.glob(f"{LR}/{repo}/data/chunk-*/episode_*.parquet"),
                    key=lambda q: int(q.split("_")[-1][:-8])):
        df = pd.read_parquet(p)
        out.append((np.stack(df["action"].to_numpy()).astype(float),          # rad
                    np.stack(df["observation.state"].to_numpy()).astype(float) * 1e-3 / R2D,  # 毫度→rad
                    np.stack(df["observation.imu_real"].to_numpy()).astype(float)))
    return out


def best_lag(e, c):
    bd, br = 0, -2.0
    for d in range(MAXLAG + 1):
        a, b = e[d:], c[:len(c) - d]
        if len(a) < 30 or a.std() < 1e-9 or b.std() < 1e-9:
            continue
        r = np.corrcoef(a, b)[0, 1]
        if r > br:
            br, bd = r, d
    return bd


def dcmd(a):
    return np.r_[0.0, np.linalg.norm(np.diff(a[:, ARM], axis=0), axis=1)]


def ahf(im):
    return filtfilt(*_HP, np.linalg.norm(im[:, :3] - im[:, :3].mean(0), axis=1))


def boundary_excess(a, im):
    n = len(a)
    B = np.arange(CHUNK, n, CHUNK)
    if not len(B):
        return None
    dc, ah = dcmd(a), ahf(im)
    bm = np.zeros(n, bool); bm[B] = True
    wm = ~bm; wm[0] = False
    cmd_ex = dc[B].mean() / (dc[wm].mean() + 1e-9)
    post = np.concatenate([np.abs(ah[b:b + 10]) for b in B if b + 10 < n])
    imu_ex = post.mean() / (np.abs(ah).mean() + 1e-9)
    segs = [dc[b + W0:b + W1] for b in B if b + W0 >= 0 and b + W1 < n]
    return cmd_ex, imu_ex, segs


def main():
    print(f"\n{'='*78}\n新数据集(20Hz) 滞后 + chunk边界跳变 复算\n{'='*78}")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.patch.set_facecolor(SURF)

    for col, (task, ev_repo, dm_repo) in enumerate(SETS):
        ev, dm = load(ev_repo), load(dm_repo)
        # ① 滞后: 逐集(臂关节均值)
        lags = []
        for a, st, _ in ev:
            ds = [best_lag(st[:, j], a[:, j]) for j in ARM]
            lags.append(np.mean(ds) / FS * 1e3)
        # 关节级一致性(所有集所有关节的 d* 分布)
        alljoint = [best_lag(st[:, j], a[:, j]) for a, st, _ in ev for j in ARM]
        # ② 边界
        ev_ex = [x for x in (boundary_excess(a, im) for a, _, im in ev) if x]
        dm_ex = [x for x in (boundary_excess(a, im) for a, _, im in dm) if x]
        cmd_e = [x[0] for x in ev_ex]; imu_e = [x[1] for x in ev_ex]
        cmd_d = [x[0] for x in dm_ex]; imu_d = [x[1] for x in dm_ex]
        n_over = sum(1 for c in cmd_e if c > 1.5)

        print(f"\n[{task}]  (eval n={len(ev)}, demo n={len(dm)})")
        print(f"  ① 滞后: 中位 {np.median(lags):.0f}ms  范围 {min(lags):.0f}-{max(lags):.0f}ms"
              f"   关节级 d* 分布: {dict(zip(*np.unique(alljoint, return_counts=True)))} (帧, 1帧=50ms)")
        print(f"  ② 边界跳变(边界/段内): 指令 eval中位 {np.median(cmd_e):.2f}x (demo假边界 {np.median(cmd_d):.2f}x)"
              f"  |  IMU eval {np.median(imu_e):.2f}x (demo {np.median(imu_d):.2f}x)")
        print(f"     指令>1.5x 的集数: {n_over}/{len(cmd_e)}")

        # 图: 边界对齐 |Δcmd| eval vs demo假边界
        ax = axes[col]; ax.set_facecolor(SURF)
        k = np.arange(W0, W1) / FS
        segs_e = np.array(sum((x[2] for x in ev_ex), []))
        segs_d = np.array(sum((x[2] for x in dm_ex), []))
        ax.plot(k, segs_d.mean(0) * R2D, color=GRAY, lw=1.5)
        ax.plot(k, segs_e.mean(0) * R2D, color=BLUE, lw=1.5)
        ax.axvline(0, color=BASE, ls="--", lw=1)
        ax.text(k[-1], segs_d.mean(0)[-1] * R2D, "  demo", color=INK2, fontsize=8, va="center")
        ax.text(k[-1], segs_e.mean(0)[-1] * R2D, "  deploy", color=BLUE, fontsize=8, va="center", fontweight="bold")
        ax.set_title(f"{task} — |Δcommand| aligned at chunk boundary", fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("time to boundary (s)", fontsize=8.5, color=INK2)
        if col == 0:
            ax.set_ylabel("|Δcmd| (deg/step)", fontsize=8.5, color=INK2)
        ax.grid(color=GRID, lw=0.7); ax.tick_params(colors=MUTED, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(BASE)

    fig.suptitle("Chunk-boundary command jump — new 20 Hz datasets (deploy vs demo pseudo-boundary)",
                 fontsize=11.5, color=INK, x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT, dpi=150, facecolor=SURF)
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
