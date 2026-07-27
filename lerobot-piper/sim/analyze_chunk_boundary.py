#!/usr/bin/env python3
"""no-TE 部署的 chunk 边界效应分析 + 与 demo 对照。

no-TE(receding-horizon, n_action_steps=100): 每 100 控制步重规划一次新 chunk,
新旧 chunk 衔接处命令可能跳变。本脚本:
  · 边界对齐叠加: 把每集帧 100/200/... 对齐, 平均 |Δcommand|(臂关节) 与 IMU accel-HF,
    看边界处(k=0)是否系统性凸起。
  · 逐集"边界超出量": 边界步 |Δcmd| / 段内步 |Δcmd|  (>1 = 边界跳得更大)。
  · demo 无 chunk 结构 → 同样按每100帧打"假边界"作对照, 应 ≈1、无凸起。

用法: /home/enolab-guo/miniconda3/envs/lerobot/bin/python sim/analyze_chunk_boundary.py
"""
from __future__ import annotations
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LR = Path.home() / ".cache/huggingface/lerobot/enolab-guo"
OUT = Path(__file__).resolve().parents[1] / "outputs/eval/chunk_boundary_effect.png"
FS, CHUNK, ARM = 30.0, 100, [0, 1, 2]
W0, W1 = -20, 40                                   # 边界对齐窗(帧)
_HP = butter(4, 2.0 / (FS / 2), "high")


def load(repo, skip):
    out = []
    for pq in sorted(glob.glob(str(LR / repo / "data/chunk-*/episode_*.parquet")),
                     key=lambda p: int(Path(p).stem.split("_")[1]))[skip:]:
        df = pd.read_parquet(pq)
        ep = int(Path(pq).stem.split("_")[1])
        a = np.stack(df["action"].to_numpy()).astype(float)
        im = np.stack(df["observation.imu_real"].to_numpy()).astype(float)
        out.append((ep, a, im))
    return out


def dcmd(a):                                       # |Δcommand|(臂), 索引 t = action[t]-action[t-1]
    d = np.linalg.norm(np.diff(a[:, ARM], axis=0), axis=1)
    return np.r_[0.0, d]


def ahf(im):                                       # accel HF 幅值(振动)
    m = np.linalg.norm(im[:, :3] - im[:, :3].mean(0), axis=1)
    return filtfilt(*_HP, m)


def epochs(sig, bounds):                            # 边界对齐窗堆叠
    segs = []
    for b in bounds:
        if b + W0 >= 0 and b + W1 < len(sig):
            segs.append(sig[b + W0: b + W1])
    return segs


def analyze(data):
    d_ep, a_ep = [], []           # 每集: cmd边界超出, imu边界超出
    d_segs, a_segs = [], []       # 全体边界对齐窗
    excess = []                   # (ep, cmd_excess, imu_excess)
    for ep, a, im in data:
        n = len(a)
        B = np.arange(CHUNK, n, CHUNK)            # 边界帧
        if len(B) == 0:
            continue
        dc, ah = dcmd(a), ahf(im)
        bmask = np.zeros(n, bool); bmask[B] = True
        within = ~bmask; within[0] = False
        cmd_ex = dc[B].mean() / (dc[within].mean() + 1e-9)
        # imu 边界超出: 边界后 [0,10] 帧 vs 整段
        post = np.concatenate([np.abs(ah[b:b + 10]) for b in B if b + 10 < n]) if len(B) else np.array([0.])
        imu_ex = post.mean() / (np.abs(ah).mean() + 1e-9)
        excess.append((ep, cmd_ex, imu_ex))
        d_segs += epochs(dc, B); a_segs += epochs(ah, B)
    return np.array(excess), np.array(d_segs), np.array(a_segs)


def main():
    no = load("eval_phase1_noimu_no_te", skip=0)
    dm = load("pickplace_clean", skip=1)
    ex_no, ds_no, as_no = analyze(no)
    ex_dm, ds_dm, as_dm = analyze(dm)
    k = np.arange(W0, W1) / FS

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    # A: 边界对齐 |Δcommand|
    ax[0].plot(k, np.abs(ds_no).mean(0), "C3", lw=1.6, label="no-TE")
    ax[0].plot(k, np.abs(ds_dm).mean(0), "C0", lw=1.6, label="demo (假边界)")
    ax[0].axvline(0, color="gray", ls="--", lw=1)
    ax[0].set(title="边界对齐 平均 |Δcommand| (臂)", xlabel="到 chunk 边界的时间 (s)", ylabel="|Δcmd| (rad)")
    # B: 边界对齐 IMU accel-HF
    ax[1].plot(k, np.abs(as_no).mean(0), "C3", lw=1.6, label="no-TE")
    ax[1].plot(k, np.abs(as_dm).mean(0), "C0", lw=1.6, label="demo (假边界)")
    ax[1].axvline(0, color="gray", ls="--", lw=1)
    ax[1].set(title="边界对齐 平均 |accel HF|", xlabel="到 chunk 边界的时间 (s)", ylabel="|accel HF|")
    # C: 逐集 命令边界超出量
    ax[2].scatter(np.zeros(len(ex_dm)) + 0, ex_dm[:, 1], s=24, c="C0", alpha=.5, label=f"demo (n={len(ex_dm)})")
    ax[2].scatter(np.zeros(len(ex_no)) + 1, ex_no[:, 1], s=30, c="C3", alpha=.7, label=f"no-TE (n={len(ex_no)})")
    ax[2].axhline(1.0, color="gray", ls=":", lw=1)
    ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["demo", "no-TE"])
    ax[2].set(title="逐集 命令边界超出量 = 边界步/段内步", ylabel="ratio (>1=边界跳更大)")
    for a in ax:
        a.legend(fontsize=8); a.grid(alpha=.3)
    fig.suptitle("Chunk 边界效应: no-TE(每100步重规划) vs demo(无chunk). 命令跳变→IMU瞬态", fontsize=12)
    fig.tight_layout(); OUT.parent.mkdir(parents=True, exist_ok=True); fig.savefig(OUT, dpi=130)
    print(f"[saved] {OUT}")

    print(f"\n{'='*58}\nchunk 边界超出量(>1=边界处比段内跳得大)\n{'='*58}")
    print(f"{'':>12}{'命令 cmd':>12}{'IMU accel':>12}")
    print(f"{'demo 中位':>12}{np.median(ex_dm[:,1]):>11.2f}{np.median(ex_dm[:,2]):>11.2f}")
    print(f"{'no-TE 中位':>12}{np.median(ex_no[:,1]):>11.2f}{np.median(ex_no[:,2]):>11.2f}")
    print(f"\n逐集 no-TE (命令边界超出 / IMU边界超出):")
    for ep, c, i in ex_no:
        flag = "  ←命令边界跳大" if c > 1.5 else ""
        print(f"  ep{int(ep):>2}  cmd {c:>5.2f}   imu {i:>5.2f}{flag}")


if __name__ == "__main__":
    main()
