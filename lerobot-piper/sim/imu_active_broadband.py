#!/usr/bin/env python3
"""IMU 全频段·活动段 严格对比: 100K 部署 vs demo (pickplace2 + insert)。

修正两个口径问题:
① 只取"活动段"(关节还在动, 由 observation.state 判定) —— 排除 deploy 静止悬停尾巴稀释平均;
② 不只 2-4.5Hz: 全频段 RMS + 分频段 (0.5-2 / 2-4.5 / 4.5-10 Hz) + 整条谱对比。

逐条计算 → 中位数; 谱 = 各 episode 活动段 Welch 后逐频中位。FS=20 (Nyquist 10Hz)。
用法: python sim/imu_active_broadband.py
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
OUT = os.path.expanduser("~/claude-test-2/lerobot-piper/outputs/eval/imu_active_broadband.png")
FS = 20.0
BLUE, GRAY = "#2a78d6", "#898781"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
TASKS = [("Pick-and-place", "eval_pickplace2_100k", "pickplace2_train"),
         ("Slot-battery insertion", "eval_insert_100k", "insert_train")]
BANDS = [(0.5, 2.0), (2.0, 4.5), (4.5, 10.0)]


def load(repo):
    out = []
    for p in sorted(glob.glob(f"{LR}/{repo}/data/chunk-*/episode_*.parquet"),
                    key=lambda q: int(q.split("_")[-1][:-8])):
        df = pd.read_parquet(p)
        st = np.stack(df["observation.state"].to_numpy()).astype(float)[:, :6] * 1e-3  # 度
        im = np.stack(df["observation.imu_real"].to_numpy()).astype(float)
        out.append((st, im))
    return out


def active_mask(st):
    """关节速度门限: 活动段 = 速度 > 5% 峰值速度 的帧(带 0.45s 形态学闭合)。"""
    v = np.linalg.norm(np.diff(st, axis=0), axis=1)
    v = np.convolve(v, np.ones(9) / 9, "same")
    m = v > 0.05 * np.percentile(v, 99)
    m = np.convolve(m.astype(float), np.ones(9), "same") > 0        # 闭合小空洞
    return np.append(m, m[-1])


def accel_dev(im):
    return np.linalg.norm(im[:, :3] - im[:, :3].mean(0), axis=1)


def ep_metrics(st, im):
    m = active_mask(st)
    a = accel_dev(im)[m]
    if len(a) < 64:
        return None
    f, P = welch(a - a.mean(), fs=FS, nperseg=min(256, len(a)))
    rms = float(np.sqrt(np.mean((a - a.mean()) ** 2)))               # 全频段 RMS (去均值)
    bps = [float(np.trapezoid(P[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)])) for lo, hi in BANDS]
    return rms, bps, (f, P), float(m.mean() * len(m) / FS)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.4), width_ratios=[1.5, 1])
    fig.patch.set_facecolor(SURF)
    print(f"\n{'='*76}\n活动段·全频段 IMU 对比 (100K deploy vs demo)  [逐条→中位]\n{'='*76}")

    for r, (task, ev_repo, dm_repo) in enumerate(TASKS):
        groups = {}
        for name, repo in [("demo", dm_repo), ("deploy", ev_repo)]:
            ms = [x for x in (ep_metrics(st, im) for st, im in load(repo)) if x]
            groups[name] = ms
        print(f"\n[{task}]")
        print(f"{'':10}{'活动时长s':>9}{'全频RMS':>9}" + "".join(f"{lo}-{hi}Hz".rjust(11) for lo, hi in BANDS))
        med = {}
        for name in ("demo", "deploy"):
            ms = groups[name]
            rms_m = np.median([x[0] for x in ms])
            bp_m = np.median([x[1] for x in ms], 0)
            dur_m = np.median([x[3] for x in ms])
            med[name] = (rms_m, bp_m)
            print(f"  {name:<8}{dur_m:>9.1f}{rms_m:>9.3f}" + "".join(f"{b:>11.3f}" for b in bp_m))
        rr = med["deploy"][0] / med["demo"][0]
        br = med["deploy"][1] / med["demo"][1]
        print(f"  → deploy/demo: 全频RMS={rr:.2f}x " +
              " ".join(f"{lo}-{hi}Hz={x:.2f}x" for (lo, hi), x in zip(BANDS, br)))

        # ---- 左: 活动段功率谱 (逐频中位, 全 0-10Hz) ----
        ax = axes[r, 0]; ax.set_facecolor(SURF)
        for name, c in [("demo", GRAY), ("deploy", BLUE)]:
            fs0 = groups[name][0][2][0]
            Ps = [x[2][1] for x in groups[name] if len(x[2][0]) == len(fs0)]
            Pm = np.median(Ps, 0)
            ax.semilogy(fs0, Pm, color=c, lw=1.6)
            ax.text(fs0[-1], Pm[-1], f"  {name}", color=(INK2 if name == "demo" else BLUE),
                    fontsize=8.5, va="center", fontweight=("bold" if name == "deploy" else "normal"))
        ax.grid(color=GRID, lw=0.7)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(BASE)
        ax.set_title(f"{task} — accel PSD, active segment (median of episodes)",
                     fontsize=10, color=INK, loc="left")
        ax.set_ylabel("PSD ((m/s²)²/Hz)", fontsize=8.5, color=INK2)
        if r == 1:
            ax.set_xlabel("frequency (Hz)", fontsize=8.5, color=INK2)
        ax.set_xlim(0, 10)

        # ---- 右: 活动段 全频段 RMS 分布 ----
        ax = axes[r, 1]; ax.set_facecolor(SURF)
        rng = np.random.default_rng(0)
        for x0, name, c in [(0, "demo", GRAY), (1, "deploy", BLUE)]:
            vals = [x[0] for x in groups[name]]
            xs = x0 + rng.uniform(-0.10, 0.10, len(vals))
            ax.scatter(xs, vals, s=26, color=c, alpha=0.75, edgecolors=SURF, linewidths=1.2, zorder=3)
            mv = float(np.median(vals))
            ax.hlines(mv, x0 - 0.22, x0 + 0.22, color=c, lw=2.2, zorder=4)
            ax.text(x0 + 0.26, mv, f"{mv:.2f}", color=INK2, fontsize=8, va="center")
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(BASE)
        ax.set_xticks([0, 1], [f"demo (n={len(groups['demo'])})", f"deploy (n={len(groups['deploy'])})"],
                      fontsize=8.5)
        ax.set_title("Broadband accel RMS, active segment", fontsize=10, color=INK, loc="left")
        ax.set_xlim(-0.55, 1.75)
        ax.set_ylim(bottom=0)

    fig.suptitle("Active-segment, full-band IMU comparison — deployed ACT (100K) vs human demos",
                 fontsize=12, color=INK, x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT, dpi=150, facecolor=SURF)
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
