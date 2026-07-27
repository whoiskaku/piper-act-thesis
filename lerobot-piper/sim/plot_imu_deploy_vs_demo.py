#!/usr/bin/env python3
"""真机 IMU 对比: 策略部署(no-TE / TE) vs 人工示范(demo, teleop)。

问题: 策略执行出来的真机振动, 比人示范更抖吗? TE 是否压低振动?
- 时域: accel/gyro 幅值的 >2Hz 高通(振动包络), 代表性 episode。
- 频域: 各集平均 PSD, 标出 2-4.5Hz 平滑带。
- 量化: 2-4.5Hz FFT 频段能量(每集均值) + 相对 demo 的比。
全部同一物理机器人, 故振动地板可比(命令无关分量见 #74)。

用法: /home/enolab-guo/miniconda3/envs/lerobot/bin/python sim/plot_imu_deploy_vs_demo.py
"""
from __future__ import annotations
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LR = Path.home() / ".cache/huggingface/lerobot/enolab-guo"
OUT = Path(__file__).resolve().parents[1] / "outputs/eval/imu_deploy_vs_demo.png"
FS = 30.0
_HP = butter(4, 2.0 / (FS / 2), "high")          # >2Hz 高通 = 振动层
DS = {  # label: (repo, color)
    "demo (teleop)":  ("pickplace_clean",          "C0"),
    "deploy no-TE":   ("eval_phase1_noimu_no_te",  "C3"),
}


def load(repo, skip=1):
    eps = []
    for pq in sorted(glob.glob(str(LR / repo / "data/chunk-*/episode_*.parquet")))[skip:]:
        im = np.stack(pd.read_parquet(pq)["observation.imu_real"].to_numpy()).astype(float)
        eps.append(im)
    return eps


def amag(im):  # 去均值 accel 幅值
    return np.linalg.norm(im[:, :3] - im[:, :3].mean(0), axis=1)


def gmag(im):
    return np.linalg.norm(im[:, 3:] - im[:, 3:].mean(0), axis=1)


NPERSEG = 256                                    # 固定窗 → 长度无关的 PSD(Welch), 各集同一频率栅格

def band_power(sig):  # 2-4.5Hz 带功率(Welch PSD 积分, 长度无关)
    f, P = welch(sig, fs=FS, nperseg=min(NPERSEG, len(sig)))
    m = (f >= 2.0) & (f < 4.5)
    return float(np.trapz(P[m], f[m]))


def avg_psd(eps, magfn):  # 各集 Welch PSD 平均(同一栅格), 返回 (f, meanP)
    f0, Ps = None, []
    for im in eps:
        f, P = welch(magfn(im), fs=FS, nperseg=min(NPERSEG, len(im)))
        f0 = f; Ps.append(P)
    return f0, np.mean(Ps, axis=0)


def main():
    data = {lab: load(repo) for lab, (repo, _) in DS.items()}

    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    for lab, (repo, c) in DS.items():
        eps = data[lab]
        if not eps:
            print(f"[skip] {lab}: 无数据"); continue
        rep = eps[0]                                  # 代表集(skip 首集后的第一条)
        t = np.arange(len(rep)) / FS
        ax[0, 0].plot(t, filtfilt(*_HP, amag(rep)), c, lw=0.8, alpha=.85, label=lab)
        ax[0, 1].plot(t, filtfilt(*_HP, gmag(rep)), c, lw=0.8, alpha=.85, label=lab)
        fa, Pa = avg_psd(eps, amag); fg, Pg = avg_psd(eps, gmag)
        ax[1, 0].semilogy(fa, Pa + 1e-12, c, lw=1.5, label=lab)
        ax[1, 1].semilogy(fg, Pg + 1e-12, c, lw=1.5, label=lab)

    ax[0, 0].set(title="Accel vibration  |a-mean|  >2Hz highpass  (representative episode)",
                 xlabel="time (s)", ylabel="accel HF (raw)")
    ax[0, 1].set(title="Gyro vibration  |w-mean|  >2Hz highpass",
                 xlabel="time (s)", ylabel="gyro HF (raw)")
    for a in (ax[1, 0], ax[1, 1]):
        a.axvspan(2.0, 4.5, color="orange", alpha=.15, label="2-4.5Hz band")
        a.set(xlabel="frequency (Hz)", xlim=(0, 10))
    ax[1, 0].set(title="Accel avg PSD across episodes", ylabel="power")
    ax[1, 1].set(title="Gyro avg PSD across episodes", ylabel="power")
    for a in ax.flat:
        a.legend(fontsize=8); a.grid(alpha=.3)
    fig.suptitle("Real-robot IMU: ACT deploy (no-TE) vs human demo  — same physical Piper",
                 fontsize=12)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)

    # ---- 量化: 2-4.5Hz 频段能量(每集均值) + 相对 demo ----
    def mean_be(eps, magfn):
        return float(np.mean([band_power(magfn(im)) for im in eps])) if eps else float("nan")
    da, dg = mean_be(data["demo (teleop)"], amag), mean_be(data["demo (teleop)"], gmag)
    print(f"\n{'='*68}\n2-4.5Hz IMU 频段能量(每集均值; 同一机器人; 越小越平滑)\n{'='*68}")
    print(f"{'数据集':<16}{'accel能量':>11}{'/demo':>7}{'gyro能量':>11}{'/demo':>7}{'  n':>4}")
    print("-" * 68)
    for lab in DS:
        eps = data[lab]
        if not eps:
            continue
        a, g = mean_be(eps, amag), mean_be(eps, gmag)
        print(f"{lab:<16}{a:>11.3f}{a/da:>7.2f}{g:>11.3f}{g/dg:>7.2f}{len(eps):>4}")
    print("-" * 68)
    print("读法: deploy/demo >1 = 策略比人示范抖; TE/noTE 比较 = TE 是否压低振动。")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
