#!/usr/bin/env python3
"""读三个 IMU npz (A raw / B smooth / C jittered)，输出时域图、PSD 图、r1/r2 判据。

r1 = E_hf(C) / E_hf(B)   # 强注入 vs 强平滑
r2 = E_hf(A) / E_hf(B)   # demo 自身 vs 强平滑

判据：
  r1 > 3 且 8Hz 峰明显 → 方法可行
  r1 < 1.5            → MuJoCo 吞高频，方法不可行，走退路 A/B
  其它                 → 灰色地带，需追加实验
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch


def load(path):
    d = np.load(path)
    return d["t"], d["accel"]


def accel_mag(accel):
    # 去重力近似：减去时间均值；假设坐标系下重力近似恒定
    return np.linalg.norm(accel - accel.mean(axis=0, keepdims=True), axis=1)


def psd(sig, fs):
    # 短序列用较小的 nperseg
    nper = min(256, max(32, len(sig) // 4))
    f, p = welch(sig, fs=fs, nperseg=nper)
    return f, p


def hf_energy(f, p, lo=5.0, hi=15.0):
    mask = (f >= lo) & (f <= hi)
    if not mask.any():
        return 0.0
    return float(np.trapz(p[mask], f[mask]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imu-a", required=True, help="raw demo")
    ap.add_argument("--imu-b", required=True, help="smoothed")
    ap.add_argument("--imu-c", required=True, help="jittered (8Hz)")
    ap.add_argument("--output-dir", default="./out_verify")
    ap.add_argument("--fs", type=float, default=30.0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tA, aA = load(args.imu_a)
    tB, aB = load(args.imu_b)
    tC, aC = load(args.imu_c)

    mA, mB, mC = accel_mag(aA), accel_mag(aB), accel_mag(aC)

    fA, pA = psd(mA, args.fs)
    fB, pB = psd(mB, args.fs)
    fC, pC = psd(mC, args.fs)

    eA = hf_energy(fA, pA)
    eB = hf_energy(fB, pB)
    eC = hf_energy(fC, pC)
    r1 = eC / eB if eB > 0 else float("inf")
    r2 = eA / eB if eB > 0 else float("inf")

    # 检查 8Hz 峰：C 在 [7,9]Hz 的能量是否显著高于 B
    peak_mask = (fC >= 7.0) & (fC <= 9.0)
    peak_C = float(pC[peak_mask].mean()) if peak_mask.any() else 0.0
    peak_B_mask = (fB >= 7.0) & (fB <= 9.0)
    peak_B = float(pB[peak_B_mask].mean()) if peak_B_mask.any() else 0.0
    peak_ratio = peak_C / peak_B if peak_B > 0 else float("inf")

    # 判据
    if r1 > 3.0 and peak_ratio > 3.0:
        verdict = "PASS — MuJoCo 保留高频，方法可行（原方向 + ΦSim）"
    elif r1 < 1.5:
        verdict = "FAIL — MuJoCo 吞高频，方法不可行，走退路 A/B"
    else:
        verdict = "GRAY — 灰色地带，建议加做中等幅度扰动组"

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(tA, mA, label="A raw", alpha=0.7)
    axes[0].plot(tB, mB, label="B smoothed", alpha=0.7)
    axes[0].plot(tC, mC, label="C jittered 8Hz", alpha=0.7)
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel("||accel - mean|| [m/s^2]")
    axes[0].set_title("IMU accel magnitude (gravity-removed by mean subtraction)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(fA, pA, label=f"A raw (E_hf={eA:.3e})", alpha=0.7)
    axes[1].semilogy(fB, pB, label=f"B smoothed (E_hf={eB:.3e})", alpha=0.7)
    axes[1].semilogy(fC, pC, label=f"C jittered (E_hf={eC:.3e})", alpha=0.7)
    axes[1].axvspan(5, 15, alpha=0.1, color="gray", label="HF band [5,15]Hz")
    axes[1].axvline(8.0, color="red", linestyle="--", alpha=0.5, label="injected 8Hz")
    axes[1].set_xlabel("f [Hz]")
    axes[1].set_ylabel("PSD")
    axes[1].set_title("Welch PSD of IMU accel magnitude")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "verification_plot.png")
    plt.savefig(plot_path, dpi=120)

    report = (
        f"=== Verification Report ===\n"
        f"E_hf(A raw)      = {eA:.4e}\n"
        f"E_hf(B smoothed) = {eB:.4e}\n"
        f"E_hf(C jittered) = {eC:.4e}\n"
        f"r1 = E_hf(C)/E_hf(B) = {r1:.3f}\n"
        f"r2 = E_hf(A)/E_hf(B) = {r2:.3f}\n"
        f"peak ratio @8Hz (C/B) = {peak_ratio:.3f}\n"
        f"\nVerdict: {verdict}\n"
        f"\nPlot saved to: {plot_path}\n"
    )
    print(report)
    with open(os.path.join(args.output_dir, "verdict.txt"), "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
