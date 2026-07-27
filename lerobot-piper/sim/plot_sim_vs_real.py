#!/usr/bin/env python3
"""#76 — 核心叙事图: 同一 Hattn cut2.0 残差, sim 平滑命令(-40%) vs 真机变差(+12~29%)。

左: SIM 命令谱(per-chunk, ACT vs refined) — refined 在 2-4.5Hz 下降(平滑);
中: REAL 命令谱(first-chunk, A=裸ACT vs B=+残差) — B 在 2-4.5Hz 上升(变差);
右: 命令 2-4.5Hz RMS 比值(refined/ACT 同口径 bp_rms): SIM<1 vs REAL>1, 线=1。
一图说尽: sim 的平滑不迁移到真机(sim→real 输入漂移)。

用法: /home/enolab-guo/miniconda3/envs/lerobot/bin/python sim/plot_sim_vs_real.py
"""
from __future__ import annotations
import glob
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt

from sim.eval_action_smoothness import reconstruct, run_model, RND, H5, DEV

LR = Path.home() / ".cache/huggingface/lerobot/enolab-guo"
REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/eval/sim_vs_real_transfer.png"
HATTN_DIR = "phase1_dense_Hattn+absboth_cut2.0_hf1.0_dl0.1_lr0.0003_bs64_ep150"
DS_A, DS_B = "eval_rc_A_act", "eval_rc_B_hattn_cut2"
FS, CHUNK, AJ = 30.0, 100, [0, 1, 2]
MDEG = np.pi / 180.0 / 1000.0
OFFLINE_RATIO = 1.12   # eval_realchunk_offline.py: raw 2.367m → ref 2.662m (matched, same obs)
_BP = butter(4, [2.0 / 15.0, 4.5 / 15.0], "band")


def spec(arr):
    """arr (N,T,7) → 平均 |FFT| over N × arm joints。"""
    w = np.hanning(arr.shape[1])[None, :, None]
    X = np.abs(np.fft.rfft((arr - arr.mean(1, keepdims=True)) * w, axis=1))[:, :, AJ]
    f = np.fft.rfftfreq(arr.shape[1], d=1.0 / FS)
    return f, X.mean((0, 2))


def bp_ratio(ref, act):
    """2-4.5Hz **FFT 频段能量** ratio-of-sums(Hann, arm joints) = #75 口径(锐 bins, 能量加权)。
    ref,act (N,T,7)。返回能量比(0.60≈#75; RMS 比=sqrt)。"""
    def be(arr):
        w = np.hanning(arr.shape[1])[None, :, None]
        P = (np.abs(np.fft.rfft((arr - arr.mean(1, keepdims=True)) * w, axis=1)) ** 2)[:, :, AJ]
        f = np.fft.rfftfreq(arr.shape[1], d=1.0 / FS)
        return float(P[:, (f >= 2.0) & (f < 4.5)].sum())
    return be(ref) / (be(act) + 1e-18)


def load_first(ds):
    out = []
    for pq in sorted(glob.glob(str(LR / ds / "data/chunk-*/episode_*.parquet")))[1:]:  # skip ep0
        a = np.stack(pd.read_parquet(pq)["action"].to_numpy()).astype(np.float64)[:CHUNK]
        out.append(a)
    return np.stack(out)   # (n,100,7)


def main():
    # ---- SIM: Hattn cut2.0 on h5 val ----
    with h5py.File(H5, "r") as f:
        g = f["val"]; ACT = g["ACT_output"][:].astype(np.float32); IMG = g["act_img_tokens"][:]
    ckpt = torch.load(RND / HATTN_DIR / "ckpt_best.pt", map_location=DEV, weights_only=False)
    m, _ = reconstruct(ckpt)
    REF = run_model(m, ACT, IMG)
    sim_ratio = bp_ratio(REF, ACT)
    fS, mS_act = spec(ACT); _, mS_ref = spec(REF)
    print(f"[sim ] command 2-4.5Hz bp_rms refined/ACT = {sim_ratio:.2f}  (n={len(ACT)})")

    # ---- REAL: A vs B first-chunk ----
    A = load_first(DS_A); B = load_first(DS_B)
    real_ratio = bp_ratio(B, A)
    fR, mR_a = spec(A); _, mR_b = spec(B)
    print(f"[real] command 2-4.5Hz bp_rms B/A = {real_ratio:.2f} (A/B-run, n={len(A)}); matched offline = {OFFLINE_RATIO:.2f}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    # left: SIM spectrum
    ax[0].semilogy(fS, mS_act, color="C0", lw=1.2, label="ACT")
    ax[0].semilogy(fS, mS_ref, color="C3", lw=1.2, label="ACT + residual")
    ax[0].axvspan(2, 4.5, color="orange", alpha=.15); ax[0].axvline(5, color="gray", ls=":", lw=.8)
    ax[0].set(xlim=(0, 15), xlabel="Hz", ylabel="|FFT| (command, arm)",
              title=f"SIM (per-chunk): residual SMOOTHS\n2-4.5Hz ×{sim_ratio:.2f} (-{100*(1-sim_ratio):.0f}%)")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=.3, which="both")
    # middle: REAL spectrum
    ax[1].semilogy(fR, mR_a, color="C0", lw=1.2, label="A: bare ACT")
    ax[1].semilogy(fR, mR_b, color="C3", lw=1.2, label="B: ACT + residual")
    ax[1].axvspan(2, 4.5, color="orange", alpha=.15); ax[1].axvline(5, color="gray", ls=":", lw=.8)
    ax[1].set(xlim=(0, 15), xlabel="Hz",
              title=f"REAL (first chunk): residual WORSE\n2-4.5Hz ×{real_ratio:.2f} (+{100*(real_ratio-1):.0f}%)")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=.3, which="both")
    # right: bar summary
    labels = ["SIM\n(per-chunk)", "REAL\n(first chunk)"]
    vals = [sim_ratio, real_ratio]
    cols = ["green", "C3"]
    bars = ax[2].bar(labels, vals, color=cols, alpha=.85, width=0.6)
    ax[2].axhline(1.0, color="k", ls="--", lw=1, label="= no change (ACT)")
    for b, v in zip(bars, vals):
        ax[2].text(b.get_x() + b.get_width()/2, v + 0.03, f"×{v:.2f}", ha="center",
                   fontsize=12, fontweight="bold", color=("green" if v < 1 else "C3"))
    ax[2].set(ylabel="command 2-4.5Hz band ENERGY  refined/ACT",
              ylim=(0, max(1.4, real_ratio * 1.15)),
              title="Does it transfer?  SIM <1 (smooth) vs REAL >1 (worse)")
    ax[2].legend(fontsize=9); ax[2].grid(alpha=.3, axis="y")
    ax[2].text(0.5, -0.23, f"matched offline recompute (same obs) also worse: RMS ×{OFFLINE_RATIO:.2f}",
               transform=ax[2].transAxes, ha="center", fontsize=7.5, style="italic", color="dimgray")
    fig.suptitle("Same Hattn cut2.0 residual — SIM smooths the command, but it does NOT transfer to the real robot (worse):  sim→real input drift",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
