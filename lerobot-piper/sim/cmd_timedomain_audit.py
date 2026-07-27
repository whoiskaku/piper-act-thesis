#!/usr/bin/env python3
"""Policy command 纯时域平滑度审计: Δ¹相邻变化量 / Δ²加速度 / Δ³jerk。

只测指令流(action→度)。chunk 内(不跨边界), 真运动窗(净位移≥10°/5s), 关节 j1,j2,j3,j5。
每窗: Δ¹RMS(deg/step, 反映运动速度+毛刺), Δ²RMS(deg/step²), Δ³RMS(deg/step³)。
另: 边界步 |Δ¹| 单列(与段内 Δ¹ 对比, 同单位可比)。
输出: 全窗表 + 运动量匹配(60-120°行程)表 + 分布图。不写入任何文档。
用法: python sim/cmd_timedomain_audit.py
"""
from __future__ import annotations
import glob, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LR = os.path.expanduser("~/.cache/huggingface/lerobot/enolab-guo")
OUT = os.path.expanduser("~/claude-test-2/lerobot-piper/outputs/eval/cmd_timedomain_audit.png")
FS, CH, R2D = 20.0, 100, 180 / np.pi
J = [0, 1, 2, 4]
BLUE, GRAY = "#2a78d6", "#898781"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"

GROUPS = [
    ("ins-demo", "insert_train", GRAY),
    ("ins-30K", "eval_insert_30k", BLUE),
    ("ins-100K", "eval_insert_100k", BLUE),
    ("ins-200K", "eval_insert_200k", BLUE),
    ("pp2-demo", "pickplace2_train", GRAY),
    ("pp2-100K", "eval_pickplace2_100k", BLUE),
]


def load_actions(repo):
    out = []
    for p in sorted(glob.glob(f"{LR}/{repo}/data/chunk-*/episode_*.parquet"),
                    key=lambda q: int(q.split("_")[-1][:-8])):
        out.append(np.stack(pd.read_parquet(p)["action"].to_numpy()).astype(float) * R2D)
    return out


def rms(x):
    return float(np.sqrt((x ** 2).mean()))


def windows(eps):
    """真运动窗逐窗: (travel, d1, d2, d3)  [RMS over steps×joints]"""
    rows = []
    for a in eps:
        for c0 in range(0, len(a) - CH + 1, CH):
            q = a[c0:c0 + CH][:, J]
            if np.abs(q[-1] - q[0]).sum() < 10.0:
                continue
            travel = np.abs(np.diff(q, axis=0)).sum()
            rows.append((travel,
                         rms(np.diff(q, n=1, axis=0)),
                         rms(np.diff(q, n=2, axis=0)),
                         rms(np.diff(q, n=3, axis=0))))
    return rows


def boundary_step(eps):
    """边界步 |Δ¹|(deg/step, 4关节向量范数) 与 段内 |Δ¹| 的中位对照"""
    b, w = [], []
    for a in eps:
        dc = np.linalg.norm(np.diff(a[:, J], axis=0), axis=1)   # index t = q[t+1]-q[t]
        for t in range(len(dc)):
            (b if (t + 1) % CH == 0 else w).append(dc[t])
    return (float(np.median(b)) if b else float("nan")), float(np.median(w))


def report(title, sel):
    print(f"\n{title}")
    print(f"{'group':<10}{'n窗':>5}{'行程°':>7}{'Δ¹RMS':>9}{'(比)':>7}{'Δ²RMS':>9}{'(比)':>7}{'Δ³RMS':>9}{'(比)':>7}")
    base = {}
    for name, repo, _ in GROUPS:
        task = name.split("-")[0]
        W = [w for w in ALL[name] if sel(w)]
        if len(W) < 5:
            print(f"{name:<10}{len(W):>5}   (样本不足)")
            continue
        m = np.median(W, 0)
        if name.endswith("demo"):
            base[task] = m
            print(f"{name:<10}{len(W):>5}{m[0]:>7.0f}{m[1]:>9.3f}{'--':>7}{m[2]:>9.3f}{'--':>7}{m[3]:>9.3f}{'--':>7}")
        else:
            d = base[task]
            print(f"{name:<10}{len(W):>5}{m[0]:>7.0f}{m[1]:>9.3f}{m[1]/d[1]:>6.2f}x{m[2]:>9.3f}{m[2]/d[2]:>6.2f}x{m[3]:>9.3f}{m[3]/d[3]:>6.2f}x")


ALL = {}
for name, repo, _ in GROUPS:
    ALL[name] = windows(load_actions(repo))

print(f"{'='*88}\nPolicy command 纯时域审计 (chunk内·j1/j2/j3/j5·RMS)   [×20 = deg/s 口径]\n{'='*88}")
report("① 全部真运动窗(净位移≥10°):", lambda w: True)
report("② 运动量匹配窗(行程 60-120°/5s):", lambda w: 60 <= w[0] <= 120)

print(f"\n③ 边界步 vs 段内 |Δ¹| (deg/step, 中位):")
print(f"{'group':<10}{'段内':>8}{'边界步':>9}{'边界/段内':>10}")
for name, repo, _ in GROUPS:
    if name.endswith("demo"):
        continue
    bm, wm = boundary_step(load_actions(repo))
    print(f"{name:<10}{wm:>8.3f}{bm:>9.3f}{bm/wm:>9.1f}x")

# ---- 图: 三个导数的逐窗分布 ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
fig.patch.set_facecolor(SURF)
rng = np.random.default_rng(0)
for k, (ax, ttl, idx) in enumerate(zip(axes,
        ["Δ¹  per-step change (deg/step)", "Δ²  acceleration (deg/step²)", "Δ³  jerk (deg/step³)"],
        [1, 2, 3])):
    ax.set_facecolor(SURF); ax.grid(axis="y", color=GRID, lw=0.7)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(BASE)
    for i, (name, repo, c) in enumerate(GROUPS):
        vals = [w[idx] for w in ALL[name]]
        xs = i + rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(xs, vals, s=10, color=c, alpha=0.5, edgecolors=SURF, linewidths=0.6, zorder=3)
        m = float(np.median(vals))
        ax.hlines(m, i - 0.26, i + 0.26, color=c, lw=2.2, zorder=4)
    ax.set_xticks(range(len(GROUPS)), [g[0] for g in GROUPS], rotation=30, ha="right")
    ax.set_title(ttl, fontsize=9.5, color=INK, loc="left")
fig.suptitle("Command time-domain smoothness — directed-motion windows, within-chunk (no boundary steps)",
             fontsize=11.5, color=INK, x=0.02, ha="left", fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT, dpi=150, facecolor=SURF)
print(f"\n[saved] {OUT}")
