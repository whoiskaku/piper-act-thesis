#!/usr/bin/env python3
"""Policy command 平滑度审计 (VIP 主张的直接检验)。

只测指令流(action, rad→度), 不看 IMU/执行。chunk 内(开环, 不跨边界), 真运动窗(指令净位移≥10°/5s)。
关节 j1,j2,j3,j5。指标:
  · 时域: 二阶差分 RMS (deg/step², 指令曲线的"毛糙度")
  · 频域: 2-4.5Hz / 4.5-10Hz 带功率 (welch, nperseg=100)
  · 边界: 边界步|Δcmd| / 段内|Δcmd| (整条信号上算)
沿训练量轴 insert 30K/100K/200K + pickplace2 100K, demo(遥操作指令)作参照。
检验: "优化充分→指令自发平滑" ⇒ 预测比值应随步数下降; 平线=否。
用法: python sim/cmd_smoothness_audit.py
"""
from __future__ import annotations
import glob, os
import numpy as np, pandas as pd
from scipy.signal import welch, detrend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LR = os.path.expanduser("~/.cache/huggingface/lerobot/enolab-guo")
OUT = os.path.expanduser("~/claude-test-2/lerobot-piper/outputs/eval/cmd_smoothness_audit.png")
FS, CH, R2D = 20.0, 100, 180 / np.pi
J = [0, 1, 2, 4]
BLUE, GRAY, AQUA = "#2a78d6", "#898781", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"

SETS = [
    ("insert", "insert_train", [("30K", "eval_insert_30k"), ("100K", "eval_insert_100k"), ("200K", "eval_insert_200k")]),
    ("pickplace2", "pickplace2_train", [("100K", "eval_pickplace2_100k")]),
]


def load_actions(repo):
    out = []
    for p in sorted(glob.glob(f"{LR}/{repo}/data/chunk-*/episode_*.parquet"),
                    key=lambda q: int(q.split("_")[-1][:-8])):
        out.append(np.stack(pd.read_parquet(p)["action"].to_numpy()).astype(float) * R2D)
    return out


def window_metrics(eps):
    """真运动窗逐窗: (d2rms, bp_mid, bp_hf)"""
    rows = []
    for a in eps:
        for c0 in range(0, len(a) - CH + 1, CH):
            q = a[c0:c0 + CH, :]
            if np.abs(q[-1, J] - q[0, J]).sum() < 10.0:
                continue
            d2 = np.diff(q[:, J], n=2, axis=0)
            d2rms = float(np.sqrt((d2 ** 2).mean()))
            bm = bh = 0.0
            for j in J:
                f, P = welch(detrend(q[:, j]), fs=FS, nperseg=CH)
                bm += float(np.trapezoid(P[(f >= 2.0) & (f < 4.5)], f[(f >= 2.0) & (f < 4.5)]))
                bh += float(np.trapezoid(P[(f >= 4.5) & (f < 10.0)], f[(f >= 4.5) & (f < 10.0)]))
            rows.append((d2rms, bm, bh))
    return rows


def boundary_jump(eps):
    r = []
    for a in eps:
        dc = np.r_[0.0, np.linalg.norm(np.diff(a[:, J], axis=0), axis=1)]
        B = np.arange(CH, len(a), CH)
        if not len(B):
            continue
        bm = np.zeros(len(a), bool); bm[B] = True
        wm = ~bm; wm[0] = False
        r.append(dc[B].mean() / (dc[wm].mean() + 1e-9))
    return float(np.median(r)) if r else float("nan")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    fig.patch.set_facecolor(SURF)

    def style(ax):
        ax.set_facecolor(SURF); ax.grid(color=GRID, lw=0.7)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color(BASE)

    print(f"\n{'='*86}\nPolicy command 平滑度审计 (chunk内·真运动窗·j1/j2/j3/j5)\n{'='*86}")
    curves = {}
    d2_groups = []          # (label, values, color)
    for task, dm_repo, evs in SETS:
        dm = window_metrics(load_actions(dm_repo))
        dmed = np.median(dm, 0)
        print(f"\n[{task}] demo(遥操作指令) n={len(dm)}窗: d2RMS={dmed[0]:.4f}  2-4.5Hz={dmed[1]:.4f}  4.5-10Hz={dmed[2]:.4f}")
        print(f"  {'ckpt':<6}{'n窗':>5}{'d2RMS':>10}{'(比)':>7}{'2-4.5Hz':>10}{'(比)':>7}{'4.5-10Hz':>10}{'(比)':>7}{'边界跳变':>9}")
        xs, mids, hfs = [], [], []
        for tag, repo in evs:
            ev = window_metrics(load_actions(repo))
            emed = np.median(ev, 0)
            bj = boundary_jump(load_actions(repo))
            step = int(tag[:-1])
            xs.append(step); mids.append(emed[1] / dmed[1]); hfs.append(emed[2] / dmed[2])
            print(f"  {tag:<6}{len(ev):>5}{emed[0]:>10.4f}{emed[0]/dmed[0]:>6.2f}x"
                  f"{emed[1]:>10.4f}{emed[1]/dmed[1]:>6.2f}x{emed[2]:>10.4f}{emed[2]/dmed[2]:>6.2f}x{bj:>8.2f}x")
            if tag == "100K":
                d2_groups.append((f"{task}\ndemo", [r[0] for r in dm], GRAY))
                d2_groups.append((f"{task}\n100K", [r[0] for r in ev], BLUE))
        curves[task] = (xs, mids, hfs)

    # ---- 图A: 带功率比 vs 训练量 (检验"练久了会平滑") ----
    ax = axes[0]; style(ax)
    xs, mids, hfs = curves["insert"]
    ax.plot(xs, hfs, "o-", color=BLUE, lw=1.8, ms=6, label="insert 4.5-10Hz")
    ax.plot(xs, mids, "s-", color=AQUA, lw=1.8, ms=6, label="insert 2-4.5Hz")
    xp, mp, hp = curves["pickplace2"]
    ax.scatter(xp, hp, marker="o", s=55, facecolors=SURF, edgecolors=BLUE, linewidths=1.8, label="pickplace2 4.5-10Hz", zorder=5)
    ax.scatter(xp, mp, marker="s", s=55, facecolors=SURF, edgecolors=AQUA, linewidths=1.8, label="pickplace2 2-4.5Hz", zorder=5)
    ax.axhline(1.0, color=INK2, ls="--", lw=1.2)
    ax.text(202, 1.0, " demo level", color=INK2, fontsize=8, va="center")
    ax.set(xticks=[30, 100, 200])
    ax.set_title("Command band power vs training steps (ratio to demo)", fontsize=9.5, color=INK, loc="left")
    ax.set_xlabel("training steps (K)", fontsize=8.5, color=INK2)
    ax.set_ylabel("deploy / demo", fontsize=8.5, color=INK2)
    ax.legend(fontsize=7.5, framealpha=0)

    # ---- 图B: d2RMS 分布 (时域) ----
    ax = axes[1]; style(ax)
    rng = np.random.default_rng(0)
    for i, (lab, vals, c) in enumerate(d2_groups):
        xsc = i + rng.uniform(-0.10, 0.10, len(vals))
        ax.scatter(xsc, vals, s=14, color=c, alpha=0.55, edgecolors=SURF, linewidths=0.8, zorder=3)
        m = float(np.median(vals)); ax.hlines(m, i - 0.24, i + 0.24, color=c, lw=2.2, zorder=4)
    ax.set_xticks(range(len(d2_groups)), [g[0] for g in d2_groups], fontsize=8)
    ax.set_title("Command 2nd-difference RMS per window (deg/step²)", fontsize=9.5, color=INK, loc="left")

    # ---- 图C: 指令曲线放大对照 (j5, 运动量匹配窗, 中位d2选取) ----
    ax = axes[2]; style(ax)
    def pick(eps):
        W = []
        for a in eps:
            for c0 in range(0, len(a) - CH + 1, CH):
                q = a[c0:c0 + CH]
                net = np.abs(q[-1, J] - q[0, J]).sum()
                if 30 <= net <= 120:
                    d2 = float(np.sqrt(np.diff(q[:, J], n=2, axis=0).__pow__(2).mean()))
                    W.append((d2, q))
        W.sort(key=lambda w: w[0])
        return W[len(W) // 2][1]
    qd = pick(load_actions("insert_train")); qe = pick(load_actions("eval_insert_100k"))
    t = np.arange(CH) / FS
    ax.plot(t, qd[:, 4] - qd[0, 4], color=GRAY, lw=1.4, label="demo command j5")
    ax.plot(t, qe[:, 4] - qe[0, 4], color=BLUE, lw=1.4, label="deploy command j5 (100K)")
    ax.set_title("Command trace j5, motion-matched window (median-d2 pick)", fontsize=9.5, color=INK, loc="left")
    ax.set_xlabel("time in window (s)", fontsize=8.5, color=INK2)
    ax.set_ylabel("deg (offset)", fontsize=8.5, color=INK2)
    ax.legend(fontsize=7.5, framealpha=0)

    fig.suptitle("Policy command smoothness audit — does more optimization smooth the command? (VIP claim test)",
                 fontsize=11.5, color=INK, x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT, dpi=150, facecolor=SURF)
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
