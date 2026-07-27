#!/usr/bin/env python3
"""IMU 加速度: 100K 部署 vs demo 对比图 (pickplace2 + insert 两任务).

左列: 加速度幅值时间序列 (各组取"带功率中位"的那条 episode, 统计上有代表性);
右列: 每条 episode 的 2-4.5Hz 带功率 (Welch, 长度不变) 分布点图 + 中位线。
Emphasis 形式: 部署=蓝(主体), demo=灰(参照)。FS=20。

用法: python sim/plot_imu_100k_vs_demo.py
"""
from __future__ import annotations
import glob, os
import numpy as np
import pandas as pd
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LR = os.path.expanduser("~/.cache/huggingface/lerobot/enolab-guo")
OUT = os.path.expanduser("~/claude-test-2/lerobot-piper/outputs/eval/imu_100k_vs_demo.png")
FS = 20.0

# 调色板 (dataviz reference palette, light)
BLUE, GRAY = "#2a78d6", "#898781"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"

TASKS = [
    ("Pick-and-place", "eval_pickplace2_100k", "pickplace2_train"),
    ("Slot-battery insertion", "eval_insert_100k", "insert_train"),
]


def accel_mag(im):
    return np.linalg.norm(im[:, :3] - im[:, :3].mean(0), axis=1)


def band_power(im):
    a = accel_mag(im)
    f, P = welch(a, fs=FS, nperseg=min(256, len(a)))
    m = (f >= 2.0) & (f < 4.5)
    return float(np.trapezoid(P[m], f[m]))


def load_imu(repo):
    out = []
    for p in sorted(glob.glob(f"{LR}/{repo}/data/chunk-*/episode_*.parquet"),
                    key=lambda q: int(q.split("_")[-1][:-8])):
        out.append(np.stack(pd.read_parquet(p)["observation.imu_real"].to_numpy()).astype(float))
    return out


def median_ep(imus, bps):
    return imus[int(np.argsort(bps)[len(bps) // 2])]


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.2), width_ratios=[2.1, 1])
    fig.patch.set_facecolor(SURF)

    for r, (task, ev_repo, dm_repo) in enumerate(TASKS):
        ev, dm = load_imu(ev_repo), load_imu(dm_repo)
        ev_bp, dm_bp = [band_power(x) for x in ev], [band_power(x) for x in dm]

        # ---- 左: 时间序列 (中位带功率 episode) ----
        ax = axes[r, 0]; style(ax)
        d, e = accel_mag(median_ep(dm, dm_bp)), accel_mag(median_ep(ev, ev_bp))
        ax.plot(np.arange(len(d)) / FS, d, color=GRAY, lw=0.9, alpha=0.85)
        ax.plot(np.arange(len(e)) / FS, e, color=BLUE, lw=0.9)
        ax.text(len(d) / FS, d[-1], "  demo", color=INK2, fontsize=8.5, va="center")
        ax.text(len(e) / FS, e[-1], "  deploy", color=BLUE, fontsize=8.5, va="center", fontweight="bold")
        ax.set_title(f"{task} — IMU accel magnitude (median-BP episode)",
                     fontsize=10, color=INK, loc="left")
        ax.set_ylabel("|accel − mean|  (m/s²)", fontsize=8.5, color=INK2)
        if r == 1:
            ax.set_xlabel("time (s)", fontsize=8.5, color=INK2)

        # ---- 右: 带功率分布 ----
        ax = axes[r, 1]; style(ax)
        rng = np.random.default_rng(0)
        for x0, vals, c, lab in [(0, dm_bp, GRAY, "demo"), (1, ev_bp, BLUE, "deploy")]:
            xs = x0 + rng.uniform(-0.10, 0.10, len(vals))
            ax.scatter(xs, vals, s=26, color=c, alpha=0.75, edgecolors=SURF, linewidths=1.2, zorder=3)
            med = float(np.median(vals))
            ax.hlines(med, x0 - 0.22, x0 + 0.22, color=c, lw=2.2, zorder=4)
            ax.text(x0 + 0.26, med, f"{med:.2f}", color=INK2, fontsize=8, va="center")
        ax.set_xticks([0, 1], [f"demo (n={len(dm_bp)})", f"deploy (n={len(ev_bp)})"], fontsize=8.5)
        ax.set_title("Vibration band power 2–4.5 Hz", fontsize=10, color=INK, loc="left")
        ax.set_xlim(-0.55, 1.75)
        ax.set_ylim(bottom=0)

    fig.suptitle("Deployed ACT (100K, no-TE) vs human demos — IMU acceleration",
                 fontsize=12, color=INK, x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=SURF)
    print(f"[saved] {OUT}")
    for task, ev_repo, dm_repo in TASKS:
        ev_bp = [band_power(x) for x in load_imu(ev_repo)]
        dm_bp = [band_power(x) for x in load_imu(dm_repo)]
        print(f"  {task}: demo med={np.median(dm_bp):.3f}  deploy med={np.median(ev_bp):.3f} "
              f"({np.median(ev_bp)/np.median(dm_bp):.2f}x)")


if __name__ == "__main__":
    main()
