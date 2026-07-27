#!/usr/bin/env python3
"""#76 — 统一口径汇总(唯一真相源)。

统一口径(canonical): **2-4.5Hz FFT 频段能量(Hann 窗, 锐 bins), ratio-of-sums(能量加权)**。
通道: 命令/关节用臂关节 j1/j2/j3; IMU 用 accel/gyro 幅值。
(注: 这是窄带能量比; 更宽的 filtfilt 带通会把残差的窄带平滑稀释到 ~中性, 见下方 broad 列。)

数字来源:
  · SIM 命令:  outputs/sim/phase1_dense.h5 val 的 ACT vs (ACT+Hattn cut2.0 残差)
  · REAL 命令: outputs/eval/realchunk_offline_chunks.npz (同观测 raw=裸ACT vs ref=+残差)
  · REAL 执行关节/IMU: 真机录制 eval_rc_A_act(裸ACT) vs eval_rc_B_hattn_cut2(+残差), within-chunk
输出: 控制台汇总表 + outputs/eval/canonical_summary.png

用法: /home/enolab-guo/miniconda3/envs/lerobot/bin/python sim/canonical_summary.py
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
NPZ = REPO / "outputs/eval/realchunk_offline_chunks.npz"
OUT = REPO / "outputs/eval/canonical_summary.png"
HATTN_DIR = "phase1_dense_Hattn+absboth_cut2.0_hf1.0_dl0.1_lr0.0003_bs64_ep150"
FS, CHUNK, ARM = 30.0, 100, [0, 1, 2]
MDEG = np.pi / 180.0 / 1000.0
_BPW = butter(4, [2.0 / 15.0, 4.5 / 15.0], "band")    # 宽带 filtfilt(仅作 broad 对照)


def band_E(x):
    """canonical: x (N,T,C) → (N,) 每段 2-4.5Hz FFT 频段能量(Hann, 锐 bins, 求和 over band+C)。"""
    w = np.hanning(x.shape[1])[None, :, None]
    P = np.abs(np.fft.rfft((x - x.mean(1, keepdims=True)) * w, axis=1)) ** 2
    f = np.fft.rfftfreq(x.shape[1], d=1.0 / FS)
    return P[:, (f >= 2.0) & (f < 4.5)].sum(axis=(1, 2))


def E_ratio(ref, raw):
    """canonical 能量比 = ratio-of-sums。ref,raw 已是 (N,T,C) 同通道。"""
    return float(band_E(ref).sum() / (band_E(raw).sum() + 1e-18))


def broad_ratio(ref, raw):
    """宽带 filtfilt RMS 比(对照, 看窄带效应被稀释)。"""
    yr = filtfilt(*_BPW, ref, axis=1); ya = filtfilt(*_BPW, raw, axis=1)
    return float(np.sqrt((yr ** 2).sum() / ((ya ** 2).sum() + 1e-18)))


def load_chunks(ds, skip=1):
    """每集(skip 前 skip 集)切非重叠 100 步 chunk → (M,100,7) state(rad), (M,100,6) imu。"""
    S, I = [], []
    for pq in sorted(glob.glob(str(LR / ds / "data/chunk-*/episode_*.parquet")))[skip:]:
        df = pd.read_parquet(pq)
        st = np.stack(df["observation.state"].to_numpy()).astype(np.float64); st[:, :6] *= MDEG
        im = np.stack(df["observation.imu_real"].to_numpy()).astype(np.float64)
        for s in range(0, len(st) - CHUNK + 1, CHUNK):
            S.append(st[s:s+CHUNK]); I.append(im[s:s+CHUNK])
    return np.array(S), np.array(I)


def main():
    rows = []   # (信号, 来源, refined/raw 能量比 canonical, broad 对照)

    # ---- SIM 命令 (h5, ACT vs +Hattn残差) ----
    with h5py.File(H5, "r") as f:
        g = f["val"]; ACT = g["ACT_output"][:].astype(np.float32); IMG = g["act_img_tokens"][:]
    ckpt = torch.load(RND / HATTN_DIR / "ckpt_best.pt", map_location=DEV, weights_only=False)
    mdl, _ = reconstruct(ckpt); REFs = run_model(mdl, ACT, IMG)
    rows.append(("命令 command", "SIM (per-chunk)",
                 E_ratio(REFs[:, :, ARM], ACT[:, :, ARM]), broad_ratio(REFs[:, :, ARM], ACT[:, :, ARM])))

    # ---- REAL 命令 (offline npz, 同观测 raw vs ref) ----
    d = np.load(NPZ); raw, ref, ci = d["raw"], d["ref"], d["ci"]
    rows.append(("命令 command", "REAL offline ALL",
                 E_ratio(ref[:, :, ARM], raw[:, :, ARM]), broad_ratio(ref[:, :, ARM], raw[:, :, ARM])))
    rows.append(("命令 command", "REAL offline later(≥1)",
                 E_ratio(ref[ci >= 1][:, :, ARM], raw[ci >= 1][:, :, ARM]), float("nan")))
    rows.append(("命令 command", "REAL offline first(冷启动)",
                 E_ratio(ref[ci == 0][:, :, ARM], raw[ci == 0][:, :, ARM]), float("nan")))

    # ---- REAL 执行关节 + IMU (A=裸ACT vs B=+残差 录制, within-chunk) ----
    nA = len(glob.glob(str(LR / "eval_rc_A_act" / "data/chunk-*/episode_*.parquet")))
    nB = len(glob.glob(str(LR / "eval_rc_B_hattn_cut2" / "data/chunk-*/episode_*.parquet")))
    joint_imu_ok = min(nA, nB) >= 3
    if joint_imu_ok:
        Sa, Ia = load_chunks("eval_rc_A_act"); Sb, Ib = load_chunks("eval_rc_B_hattn_cut2")
        rows.append(("执行关节 joint", "REAL A vs B",
                     E_ratio(Sb[:, :, ARM], Sa[:, :, ARM]), broad_ratio(Sb[:, :, ARM], Sa[:, :, ARM])))
        accA = np.linalg.norm(Ia[:, :, :3] - Ia[:, :, :3].mean(1, keepdims=True), axis=2)[:, :, None]
        accB = np.linalg.norm(Ib[:, :, :3] - Ib[:, :, :3].mean(1, keepdims=True), axis=2)[:, :, None]
        gyrA = np.linalg.norm(Ia[:, :, 3:], axis=2)[:, :, None]; gyrB = np.linalg.norm(Ib[:, :, 3:], axis=2)[:, :, None]
        rows.append(("IMU accel", "REAL A vs B", E_ratio(accB, accA), broad_ratio(accB, accA)))
        rows.append(("IMU gyro", "REAL A vs B", E_ratio(gyrB, gyrA), broad_ratio(gyrB, gyrA)))
    else:
        print(f"\n[warn] 录制集数不足 (A={nA}, B={nB}; 需≥3) → 跳过执行关节/IMU。"
              f"\n       录制目录似被覆盖/重录(原各6集)。命令结果来自 npz(原完整数据)仍有效;"
              f"\n       执行关节/IMU 之前(6集 bp_rms)≈1.0(无改善), 需完整重录才能出 canonical 数。")

    print(f"\n{'='*78}\n统一口径: 2-4.5Hz FFT 频段能量比(ratio-of-sums); refined/raw 或 B/A。 <1=平滑\n{'='*78}")
    print(f"{'信号':>14} {'来源':>26} {'能量比(canonical)':>17} {'broad对照':>10}")
    print("-" * 78)
    for sig, src, can, broad in rows:
        bs = f"{broad:.2f}" if broad == broad else "  -"
        print(f"{sig:>14} {src:>26} {can:>15.2f}   {bs:>8}")
    print("=" * 78)
    print("读法: 命令 SIM~0.75 / REAL~0.83 = 残差平滑命令且迁移到真机(窄带; broad 列显示宽带则被稀释到~中性)。")
    print("      执行关节/IMU ≈1.0 = 命令平滑到不了物理运动(控制器低通 + 命令无关地板 #71/#74)。")

    # ---- 汇总图 ----
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ["command\nSIM", "command\nREAL", "joint\nREAL", "IMU acc\nREAL", "IMU gyro\nREAL"]
    vals = [rows[0][2], rows[1][2], rows[4][2], rows[5][2], rows[6][2]]
    cols = ["green", "seagreen", "gray", "gray", "gray"]
    bars = ax.bar(labels, vals, color=cols, alpha=.85, width=0.62)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="= no change")
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.02, f"×{v:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.set(ylabel="2-4.5Hz band ENERGY  (residual / no-residual)", ylim=(0, 1.3),
           title="Canonical (2-4.5Hz FFT band energy): residual smooths COMMAND (sim+real), "
                 "but executed joint / IMU unchanged (controller ceiling)")
    ax.legend(fontsize=9); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT, dpi=130)
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
