#!/usr/bin/env python3
"""#76 — (a) 公平 sim↔real 叙事图 + (b) per-chunk 运动依赖图。

统一口径: 2-4.5Hz FFT 频段能量(Hann, arm joints), ratio-of-sums。
sim: Hattn cut2.0 on h5 val (refined/ACT)。
real: 离线重算 npz(同观测 raw=裸ACT vs ref=+残差, 每集所有 chunk-start)。

(a) sim_vs_real_transfer.png(覆盖, 公平版): real 用 later/ALL chunk(非 first-chunk 特例)。
(b) realchunk_perchunk_motion.png: 每 chunk ref/raw vs raw 抖动量, 显示"高运动平滑/近静止略差"。

用法: /home/enolab-guo/miniconda3/envs/lerobot/bin/python sim/plot_realchunk_chunks.py
"""
from __future__ import annotations
from pathlib import Path

import h5py
import numpy as np
import torch

from sim.eval_action_smoothness import reconstruct, run_model, RND, H5, DEV

REPO = Path(__file__).resolve().parents[1]
NPZ = REPO / "outputs/eval/realchunk_offline_chunks.npz"
HATTN_DIR = "phase1_dense_Hattn+absboth_cut2.0_hf1.0_dl0.1_lr0.0003_bs64_ep150"
FS, AJ = 30.0, [0, 1, 2]


def band_E(arr):
    """(N,T,7) → (N,) 每 chunk 的 2-4.5Hz FFT 频段能量(arm joints, Hann)。"""
    w = np.hanning(arr.shape[1])[None, :, None]
    P = (np.abs(np.fft.rfft((arr - arr.mean(1, keepdims=True)) * w, axis=1)) ** 2)[:, :, AJ]
    f = np.fft.rfftfreq(arr.shape[1], d=1.0 / FS)
    return P[:, (f >= 2.0) & (f < 4.5)].sum(axis=(1, 2))


def spec(arr):
    w = np.hanning(arr.shape[1])[None, :, None]
    X = np.abs(np.fft.rfft((arr - arr.mean(1, keepdims=True)) * w, axis=1))[:, :, AJ]
    return np.fft.rfftfreq(arr.shape[1], d=1.0 / FS), X.mean((0, 2))


def main():
    # ---- SIM ----
    with h5py.File(H5, "r") as fh:
        g = fh["val"]; ACT = g["ACT_output"][:].astype(np.float32); IMG = g["act_img_tokens"][:]
    ckpt = torch.load(RND / HATTN_DIR / "ckpt_best.pt", map_location=DEV, weights_only=False)
    mdl, _ = reconstruct(ckpt); REFs = run_model(mdl, ACT, IMG)
    sE_a, sE_r = band_E(ACT), band_E(REFs)
    sim_ratio = float(sE_r.sum() / sE_a.sum())
    fS, spA = spec(ACT); _, spR = spec(REFs)

    # ---- REAL (offline npz: raw=裸ACT, ref=+残差, 同观测) ----
    d = np.load(NPZ); raw, ref, ci = d["raw"], d["ref"], d["ci"]
    rE_raw, rE_ref = band_E(raw), band_E(ref)
    def rt(mask): return float(rE_ref[mask].sum() / rE_raw[mask].sum())
    r_first = rt(ci == 0); r_later = rt(ci >= 1); r_all = rt(np.ones(len(ci), bool))
    fR, rpA = spec(raw[ci >= 1]); _, rpR = spec(ref[ci >= 1])
    print(f"[sim] {sim_ratio:.2f} | [real] first={r_first:.2f} later={r_later:.2f} ALL={r_all:.2f}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ===== (a) 公平 sim↔real 叙事图 =====
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    ax[0].semilogy(fS, spA, "C0", lw=1.2, label="ACT"); ax[0].semilogy(fS, spR, "C3", lw=1.2, label="ACT+residual")
    ax[0].axvspan(2, 4.5, color="orange", alpha=.15); ax[0].axvline(5, color="gray", ls=":", lw=.8)
    ax[0].set(xlim=(0, 15), xlabel="Hz", ylabel="|FFT| (command, arm)",
              title=f"SIM (per-chunk): residual SMOOTHS\n2-4.5Hz ×{sim_ratio:.2f}")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=.3, which="both")
    ax[1].semilogy(fR, rpA, "C0", lw=1.2, label="raw ACT (offline)"); ax[1].semilogy(fR, rpR, "C3", lw=1.2, label="+residual (offline)")
    ax[1].axvspan(2, 4.5, color="orange", alpha=.15); ax[1].axvline(5, color="gray", ls=":", lw=.8)
    ax[1].set(xlim=(0, 15), xlabel="Hz",
              title=f"REAL (later chunks, matched obs): SMOOTHS too\n2-4.5Hz ×{r_later:.2f}")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=.3, which="both")
    labels = ["SIM\n(per-chunk)", "REAL\n(later/ALL)", "REAL\n(first chunk)"]
    vals = [sim_ratio, r_all, r_first]; cols = ["green", "darkorange", "C3"]
    bars = ax[2].bar(labels, vals, color=cols, alpha=.85, width=0.62)
    ax[2].axhline(1.0, color="k", ls="--", lw=1, label="= no change")
    for b, v in zip(bars, vals):
        ax[2].text(b.get_x()+b.get_width()/2, v+0.03, f"×{v:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax[2].annotate("cold-start\noutlier", (2, r_first), (2, r_first+0.18), ha="center", fontsize=8,
                   color="C3", arrowprops=dict(arrowstyle="->", color="C3"))
    ax[2].set(ylabel="command 2-4.5Hz band ENERGY  refined/ACT",
              ylim=(0, max(1.35, r_first*1.2)), title="Command smoothing TRANSFERS (SIM→REAL); first-chunk = cold-start outlier")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3, axis="y")
    fig.suptitle(f"Hattn cut2.0 residual: command 2-4.5Hz smoothing TRANSFERS to real (SIM 0.75 → REAL {r_all:.2f}); "
                 f"BUT executed joint/IMU unchanged — that band is mostly command-independent floor (controller ceiling, #74).", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_a = REPO / "outputs/eval/sim_vs_real_transfer.png"; fig.savefig(out_a, dpi=130); print(f"[saved] {out_a}")

    # ===== (b) per-chunk 运动依赖 =====
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5.5))
    raw_rms = np.sqrt(rE_raw / raw.shape[1])      # band 振幅(抖动量) per chunk
    ratio_pc = rE_ref / (rE_raw + 1e-18)
    sc = ax2.scatter(raw_rms * 1e3, ratio_pc, c=ci, cmap="viridis", s=70, edgecolor="k", lw=.4)
    ax2.axhline(1.0, color="k", ls="--", lw=1)
    ax2.text(ax2.get_xlim()[1]*0.97, 1.02, "residual WORSE ↑", ha="right", fontsize=8, color="C3")
    ax2.text(ax2.get_xlim()[1]*0.97, 0.97, "residual SMOOTHS ↓", ha="right", va="top", fontsize=8, color="green")
    plt.colorbar(sc, label="chunk index within episode")
    ax2.set(xlabel="raw ACT 2-4.5Hz amplitude per chunk (m°, jitter level)",
            ylabel="residual/raw  (2-4.5Hz band energy)",
            title="Per-chunk: residual SMOOTHS high-jitter chunks (<1), slightly WORSENS near-static (>1)")
    ax2.grid(alpha=.3)
    fig2.tight_layout()
    out_b = REPO / "outputs/eval/realchunk_perchunk_motion.png"; fig2.savefig(out_b, dpi=130); print(f"[saved] {out_b}")


if __name__ == "__main__":
    main()
