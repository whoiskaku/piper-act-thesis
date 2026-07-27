#!/usr/bin/env python3
"""#76 — 真机数据对比图: A=裸ACT vs B=ACT+Hattn cut2.0 残差 (first-chunk, 单chunk)。

用真机录制(eval_rc_A_act / eval_rc_B_hattn_cut2)的第一个 chunk(frame 0-100, 无边界):
  上排 4 面板: 命令/执行关节/IMU accel/IMU gyro 的平均频谱(A蓝 vs B红, 2-4.5Hz 阴影);
  下排: 2-4.5Hz RMS 柱状(A vs B + B/A 比值)。
直观呈现 #76 决定性结果: 残差让【命令】2-4.5Hz 变差, 执行关节/IMU 无改善。

用法: /home/enolab-guo/miniconda3/envs/lerobot/bin/python sim/plot_realchunk_compare.py
"""
from __future__ import annotations
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

LR = Path.home() / ".cache/huggingface/lerobot/enolab-guo"
REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/eval/realchunk_compare_A_vs_B.png"
DS_A, DS_B = "eval_rc_A_act", "eval_rc_B_hattn_cut2"
FS, CHUNK = 30.0, 100
AJ = [0, 1, 2]
MDEG = np.pi / 180.0 / 1000.0
_BP = butter(4, [2.0 / 15.0, 4.5 / 15.0], "band")


def load_firstchunks(ds, skip=1):
    """返回每集第一个 chunk(0-100): list of (act(100,7), st(100,7) rad, imu(100,6))。"""
    out = []
    for pq in sorted(glob.glob(str(LR / ds / "data/chunk-*/episode_*.parquet")))[skip:]:
        df = pd.read_parquet(pq)
        act = np.stack(df["action"].to_numpy()).astype(np.float64)[:CHUNK]
        st = np.stack(df["observation.state"].to_numpy()).astype(np.float64)[:CHUNK]
        imu = np.stack(df["observation.imu_real"].to_numpy()).astype(np.float64)[:CHUNK]
        st = st.copy(); st[:, :6] *= MDEG
        out.append((act, st, imu))
    return out


def spec_arm(chunks, pick):
    """平均频谱 over 集 × 臂关节(命令/关节)。pick(act,st,imu)->(T,7 取AJ)。"""
    mags = []
    for c in chunks:
        x = pick(*c)[:, AJ]                       # (T,3)
        w = np.hanning(len(x))[:, None]
        X = np.abs(np.fft.rfft((x - x.mean(0)) * w, axis=0))  # (F,3)
        mags.append(X.mean(1))
    f = np.fft.rfftfreq(CHUNK, d=1.0 / FS)
    return f, np.mean(mags, 0)


def spec_mag(chunks, pick):
    """平均频谱 over 集, pick->1D 幅值信号(IMU accel/gyro mag)。"""
    mags = []
    for c in chunks:
        s = pick(*c)
        w = np.hanning(len(s))
        mags.append(np.abs(np.fft.rfft((s - s.mean()) * w)))
    f = np.fft.rfftfreq(CHUNK, d=1.0 / FS)
    return f, np.mean(mags, 0)


def bp_rms_arm(chunks, pick):
    vals = [np.sqrt((filtfilt(*_BP, pick(*c)[:, AJ], axis=0) ** 2).mean()) for c in chunks]
    return float(np.mean(vals))


def bp_rms_mag(chunks, pick):
    vals = [np.sqrt((filtfilt(*_BP, pick(*c)) ** 2).mean()) for c in chunks]
    return float(np.mean(vals))


CMD = lambda a, s, i: a
JNT = lambda a, s, i: s
IACC = lambda a, s, i: np.linalg.norm(i[:, :3] - i[:, :3].mean(0), axis=1)
IGYR = lambda a, s, i: np.linalg.norm(i[:, 3:], axis=1)


def main():
    A = load_firstchunks(DS_A); B = load_firstchunks(DS_B)
    print(f"[data] A({DS_A})={len(A)} eps, B({DS_B})={len(B)} eps (skip ep0)")

    chans = [
        ("command (action, arm)", CMD, spec_arm, bp_rms_arm),
        ("executed joint (state, arm)", JNT, spec_arm, bp_rms_arm),
        ("IMU accel |a-mean|", IACC, spec_mag, bp_rms_mag),
        ("IMU gyro |w|", IGYR, spec_mag, bp_rms_mag),
    ]
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(2, 4, height_ratios=[2, 1.3], hspace=0.38, wspace=0.28)

    ratios = []
    for k, (name, pick, specfn, rmsfn) in enumerate(chans):
        ax = fig.add_subplot(gs[0, k])
        fA, mA = specfn(A, pick); fB, mB = specfn(B, pick)
        ax.semilogy(fA, mA, color="C0", lw=1.1, label="A: bare ACT")
        ax.semilogy(fB, mB, color="C3", lw=1.1, label="B: +residual")
        ax.axvspan(2, 4.5, color="orange", alpha=0.13)
        ax.axvline(5, color="gray", ls=":", lw=0.8)
        ax.set_xlim(0, 15); ax.set_title(name, fontsize=9)
        ax.set_xlabel("Hz", fontsize=8)
        if k == 0:
            ax.set_ylabel("|FFT| (ep-avg, first chunk)", fontsize=8); ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        rA, rB = rmsfn(A, pick), rmsfn(B, pick)
        ratios.append((name.split(" (")[0], rA, rB))

    axb = fig.add_subplot(gs[1, :])
    x = np.arange(len(ratios)); w = 0.36
    rsA = np.array([r[1] for r in ratios]); rsB = np.array([r[2] for r in ratios])
    nA, nB = rsA / rsA, rsB / rsA          # normalize to A=1 for visual comparability
    b1 = axb.bar(x - w/2, nA, w, color="C0", label="A: bare ACT (=1)")
    b2 = axb.bar(x + w/2, nB, w, color="C3", label="B: +residual")
    axb.axhline(1.0, color="k", ls=":", lw=0.8)
    for i, (nm, ra, rb) in enumerate(ratios):
        axb.text(i + w/2, nB[i] + 0.02, f"×{rb/ra:.2f}", ha="center", fontsize=9,
                 color=("C3" if rb/ra > 1.02 else "green"), fontweight="bold")
    axb.set_xticks(x); axb.set_xticklabels([r[0] for r in ratios], fontsize=9)
    axb.set_ylabel("2-4.5Hz RMS (norm. to A)", fontsize=9)
    axb.set_title("2-4.5Hz jitter: B/A ratio  (>1 = residual WORSE; ≈1 = no change)", fontsize=10)
    axb.legend(fontsize=8); axb.grid(alpha=0.3, axis="y")

    fig.suptitle("Real robot (first chunk, single-chunk): A=bare ACT vs B=ACT+Hattn cut2.0 residual",
                 fontsize=12)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print("[2-4.5Hz RMS] " + " | ".join(f"{nm}: A={ra*1e3:.3f}m B={rb*1e3:.3f}m (×{rb/ra:.2f})"
                                        for nm, ra, rb in ratios))
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
