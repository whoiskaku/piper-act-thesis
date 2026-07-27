#!/usr/bin/env python3
"""#75 — 回放 ACT vs RefineNet 命令动作 做可视化评价(动作空间, 非IMU)。

"回放评价": 取 sim val chunk, 复现 ACT 命令 与 RefineNet 精修命令, 直接画出来看平滑:
  · 左列: val 平均频谱 |FFT| (ACT/refined/clean), 2-4.5Hz 任务相关带阴影 → 严谨, 不挑样本。
  · 右列: 抖动最大的单个 chunk, 2Hz 高通后的抖动时域 (ACT vs refined) → 肉眼看见抖动被削。
每行一个 active joint (j1/j2/j3/grip)。

注: 视觉回放(渲染机械臂)看不出 ~0.09° 的 HF 抖动差异; 要"看见"平滑须看信号(本脚本)。

用法: /home/enolab-guo/miniconda3/envs/lerobot/bin/python sim/replay_eval_action.py
      [--ckpt <ckpt_best.pt>]   # 默认 Hattn cut2.0(最佳操作点); 可指向自训 ckpt
"""
from __future__ import annotations
from pathlib import Path
import argparse

import h5py
import numpy as np
import torch
from scipy.signal import butter, filtfilt

from sim.eval_action_smoothness import reconstruct, run_model, RND, H5, DEV
from sim.train_refine_net import ACTIVE_JOINTS as AJ, FPS

REPO = Path(__file__).resolve().parents[1]
OUT_PNG = REPO / "outputs/eval/replay_action_ACT_vs_refined.png"
DEFAULT = "phase1_dense_Hattn+absboth_cut2.0_hf1.0_dl0.1_lr0.0003_bs64_ep150"
JNAMES = {0: "j1", 1: "j2", 2: "j3", 6: "grip"}
_HP2 = butter(4, 2.0 / (FPS / 2.0), "high")


def avg_spectrum(x, j):
    """val 平均 |FFT| (Hann, 去DC) for joint j。x (N,T,7) → (f, mag)。"""
    N, Tn, _ = x.shape
    w = np.hanning(Tn)[None, :]
    X = np.fft.rfft((x[:, :, j] - x[:, :, j].mean(1, keepdims=True)) * w, axis=1)
    f = np.fft.rfftfreq(Tn, d=1.0 / FPS)
    return f, np.abs(X).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=RND / DEFAULT / "ckpt_best.pt")
    cli = ap.parse_args()

    with h5py.File(H5, "r") as fh:
        g = fh["val"]
        ACT = g["ACT_output"][:].astype(np.float32)
        CLN = g["A_clean"][:].astype(np.float32)
        IMG = g["act_img_tokens"][:]
    ckpt = torch.load(cli.ckpt, map_location=DEV, weights_only=False)
    m, a = reconstruct(ckpt)
    REF = run_model(m, ACT, IMG)
    print(f"[replay] ckpt={cli.ckpt.parent.name} ep{ckpt.get('epoch')}  val n={len(ACT)}")

    # 选"抖动最大"的 chunk(ACT 在 active joints 的 2-4.5Hz 能量最大) → 清晰演示, 非挑refined表现
    w = np.hanning(ACT.shape[1])[None, :, None]
    F = np.fft.rfftfreq(ACT.shape[1], d=1.0 / FPS)
    band = (F >= 2.0) & (F < 4.5)
    Xa = np.abs(np.fft.rfft((ACT - ACT.mean(1, keepdims=True)) * w, axis=1))
    jit = (Xa[:, band][:, :, AJ] ** 2).sum(axis=(1, 2))
    ci = int(jit.argmax())
    print(f"[replay] 演示 chunk #{ci} (ACT 2-4.5Hz 抖动最大)")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nj = len(AJ)
    fig, ax = plt.subplots(nj, 2, figsize=(13, 2.5 * nj))
    t = np.arange(ACT.shape[1]) / FPS
    for r, j in enumerate(AJ):
        # 左: val 平均频谱
        f, ma = avg_spectrum(ACT, j); _, mr = avg_spectrum(REF, j); _, mc = avg_spectrum(CLN, j)
        axL = ax[r, 0]
        axL.semilogy(f, ma, label="ACT", color="C0")
        axL.semilogy(f, mr, label="refined", color="C3")
        axL.semilogy(f, mc, label="clean(demo)", color="C2", alpha=.7)
        axL.axvspan(2, 4.5, color="orange", alpha=.12)
        axL.axvline(5, color="gray", ls=":", lw=.8)
        axL.set_xlim(0, 15); axL.set_ylabel(f"{JNAMES[j]}  |FFT| (val avg)")
        if r == 0:
            axL.legend(fontsize=8, loc="upper right"); axL.set_title("val-avg spectrum (2-4.5Hz shaded)")
        if r == nj - 1:
            axL.set_xlabel("Hz")
        # 右: 演示 chunk 2Hz 高通抖动时域
        ha = filtfilt(*_HP2, ACT[ci, :, j]); hr = filtfilt(*_HP2, REF[ci, :, j])
        axR = ax[r, 1]
        axR.plot(t, ha, color="C0", lw=.9, label=f"ACT (rms {np.sqrt((ha**2).mean())*1e3:.2f}m°)")
        axR.plot(t, hr, color="C3", lw=.9, label=f"refined (rms {np.sqrt((hr**2).mean())*1e3:.2f}m°)")
        axR.axhline(0, color="k", lw=.4)
        axR.set_ylabel(f"{JNAMES[j]}  >2Hz jitter (rad)")
        axR.legend(fontsize=7, loc="upper right")
        if r == 0:
            axR.set_title(f"chunk #{ci} >2Hz jitter (ACT vs refined)")
        if r == nj - 1:
            axR.set_xlabel("s")
    fig.suptitle(f"Replay eval: ACT vs RefineNet ({cli.ckpt.parent.name[:40]})", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    print(f"[saved] {OUT_PNG}")


if __name__ == "__main__":
    main()
