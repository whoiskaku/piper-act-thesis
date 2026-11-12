#!/usr/bin/env python3
"""评测运行的 IMU 时序图：每条一行，左陀螺（rad/s），右加速度（m/s²，减去本条中位数）。

数据来自 `record` 跑策略时自动写的高频日志 `<数据集>/high_rate/episode_XXXXXX.json`：
IMU 100 Hz，时刻 = 设备 ms 经最小二乘映射到 monotonic（同 sim/step4_gvnext/data.load），
横轴从该条第一帧算起。不滤波。加速度减中位数只是去掉重力+零偏这个常量，三轴才能同刻度看。
对比开/不开 TE 的同一任务时，两次用同样的 --gyro-lim / --acc-lim。

用法：
    python scripts/plot_eval_imu.py enolab-guo/eval_pickcube_noTE_look --chunk 100
    python scripts/plot_eval_imu.py enolab-guo/eval_pickcube_TE --gyro-lim 1.2 --acc-lim 5
输出：outputs/eval_imu/<数据集名>.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LR = Path.home() / ".cache/huggingface/lerobot"
OUT = Path(__file__).resolve().parents[1] / "outputs/eval_imu"
AXES = [("x", "#2a78d6"), ("y", "#eb6834"), ("z", "#1baf7a")]   # 分类色前 3 位，已过色盲校验
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def load(path: Path):
    """→ (IMU 时刻 s, 加速度 (n,3), 陀螺 (n,3), 各帧时刻 s)，时刻都从第一帧起算。"""
    h = json.loads(path.read_text())
    imu = np.asarray(h["imu100"], float)
    ms, ht = imu[:, 1], imu[:, 0]
    a, c = np.linalg.lstsq(np.stack([ms, np.ones_like(ms)], 1), ht, rcond=None)[0]
    tf = np.asarray(h["t"])
    return a * ms + c - tf[0], imu[:, 2:5], imu[:, 5:8], tf - tf[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id")
    ap.add_argument("--chunk", type=int, default=0, help="每隔几帧推理一次（不开 TE 时 = n_action_steps）；0 = 不画")
    ap.add_argument("--gyro-lim", type=float, help="陀螺纵轴 ±范围；缺省按数据取")
    ap.add_argument("--acc-lim", type=float, help="加速度纵轴 ±范围；缺省按数据取")
    a = ap.parse_args()

    root = LR / a.repo_id
    n_ep = json.loads((root / "meta/info.json").read_text())["total_episodes"]
    data = [load(root / "high_rate" / f"episode_{i:06d}.json") for i in range(n_ep)]
    data = [(ti, acc - np.median(acc, 0), gyr, tf) for ti, acc, gyr, tf in data]
    glim = a.gyro_lim or 1.05 * max(np.abs(g).max() for _, _, g, _ in data)
    alim = a.acc_lim or 1.05 * max(np.abs(acc).max() for _, acc, _, _ in data)

    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 9,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(n_ep, 2, figsize=(14, 1.9 * n_ep + 1.0), sharex=True, squeeze=False)
    cols = (("陀螺（rad/s）", glim, 2), ("加速度 − 本条中位数（m/s²）", alim, 1))
    for r, (ti, acc, gyr, tf) in enumerate(data):
        for c, (title, lim, k) in enumerate(cols):
            A, y = ax[r, c], (gyr, acc)[c]
            A.grid(axis="y", color=GRID, lw=0.8)
            A.set_axisbelow(True)
            for side in ("top", "right"):
                A.spines[side].set_visible(False)
            if a.chunk:
                for f in range(0, len(tf), a.chunk):
                    A.axvline(tf[f], color=MUTED, lw=0.8, ls="--")
            for j, (name, col) in enumerate(AXES):
                A.plot(ti, y[:, j], color=col, lw=0.9, label=name)
            A.set_ylim(-lim, lim)
            A.set_xlim(0, max(ti[-1], tf[-1]))
            if r == 0:
                A.set_title(title, loc="left", color=INK)
            if c == 0:
                A.set_ylabel(f"第 {r} 条", rotation=0, ha="right", va="center")
    for A in ax[-1]:
        A.set_xlabel("时间（s，从第一帧起）")
    handles = [plt.Line2D([], [], color=col, lw=2, label=f"{name} 轴") for name, col in AXES]
    if a.chunk:
        handles.append(plt.Line2D([], [], color=MUTED, lw=0.8, ls="--", label=f"ACT 推理（每 {a.chunk} 帧）"))
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), frameon=False, labelcolor=INK)
    fig.suptitle(f"{a.repo_id}：IMU 100 Hz 原始数据，未滤波", x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{a.repo_id.split('/')[-1]}.png"
    fig.savefig(out, dpi=120, facecolor="#fcfcfb")
    print(f"陀螺 ±{glim:.2f} rad/s · 加速度 ±{alim:.2f} m/s² → {out}")


if __name__ == "__main__":
    main()
