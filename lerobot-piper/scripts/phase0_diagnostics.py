#!/usr/bin/env python3
"""Phase 0 诊断 A + B：为 H1/H2 取证。

诊断 A：演示 action 的功率谱与 95%/99% 累积能量带宽（per-channel + overall）。
诊断 B：Chebyshev / Fourier / B-spline 三种基函数重构 GT chunk 的 MSE vs K 曲线。

输入：LeRobot v2 数据集（包含 action + timestamp 列）。
输出：JSON + PNG 在指定 --output-dir。
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial import chebyshev as cheb
from scipy.interpolate import splrep, splev


CHANNEL_NAMES = [
    "joint_1", "joint_2", "joint_3", "joint_4",
    "joint_5", "joint_6", "gripper",
]


# ------------------------------- data loading ------------------------------- #
def load_all_actions(dataset_root: Path):
    parquets = sorted(dataset_root.glob("data/chunk-*/episode_*.parquet"))
    if not parquets:
        raise RuntimeError(f"no parquet under {dataset_root}")
    episodes = []
    for pq in parquets:
        df = pd.read_parquet(pq)
        a = np.stack(df["action"].to_numpy()).astype(np.float32)
        t = df["timestamp"].to_numpy().astype(np.float64)
        episodes.append({"name": pq.stem, "action": a, "t": t})
    return episodes


def make_chunks(episodes, T=100, stride=10):
    chunks = []
    for ep in episodes:
        n = len(ep["action"])
        if n < T:
            continue
        for s in range(0, n - T + 1, stride):
            chunks.append(ep["action"][s:s + T])
    return np.stack(chunks)  # (N, T, 7)


# ------------------------------- diagnostic A ------------------------------- #
def power_spectrum(chunks, fs=30.0):
    """chunks: (N, T, D). Returns freqs (F,), mean PSD (F, D) averaged over N."""
    N, T, D = chunks.shape
    # Subtract chunk mean to remove DC, so HF ratio isn't dominated by offset
    x = chunks - chunks.mean(axis=1, keepdims=True)
    win = np.hanning(T)[None, :, None]  # reduce spectral leakage
    Xf = np.fft.rfft(x * win, axis=1)                         # (N, F, D)
    psd = (np.abs(Xf) ** 2).mean(axis=0)                      # (F, D)
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    return freqs, psd


def bandwidth_at(freqs, psd, frac=0.95):
    """Return per-channel frequency at which cumulative energy reaches `frac`.
    psd: (F, D). Returns (D,) band-width."""
    cum = np.cumsum(psd, axis=0)
    cum_norm = cum / (cum[-1:] + 1e-30)
    idx = (cum_norm >= frac).argmax(axis=0)  # first index reaching frac
    return freqs[idx]


def hf_ratio(freqs, psd, cutoff=5.0):
    """Fraction of total energy above `cutoff` Hz, per channel."""
    mask = freqs > cutoff
    total = psd.sum(axis=0)
    high = psd[mask].sum(axis=0)
    return high / (total + 1e-30)


def diagnostic_A(chunks, out_dir: Path, fs=30.0):
    freqs, psd = power_spectrum(chunks, fs=fs)
    bw95 = bandwidth_at(freqs, psd, 0.95)
    bw99 = bandwidth_at(freqs, psd, 0.99)
    hf5 = hf_ratio(freqs, psd, 5.0)
    hf2 = hf_ratio(freqs, psd, 2.0)

    # Overall curve (average across channels, equal weight)
    psd_mean_ch = psd.mean(axis=1)
    cum = np.cumsum(psd_mean_ch) / (psd_mean_ch.sum() + 1e-30)
    overall_bw95 = float(freqs[(cum >= 0.95).argmax()])
    overall_bw99 = float(freqs[(cum >= 0.99).argmax()])
    overall_hf5 = float(
        psd_mean_ch[freqs > 5.0].sum() / (psd_mean_ch.sum() + 1e-30)
    )

    # --- plots ---
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    # (1) per-channel PSD
    ax = axes[0]
    for ch in range(psd.shape[1]):
        ax.semilogy(freqs, psd[:, ch] + 1e-20, label=CHANNEL_NAMES[ch], alpha=0.8)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("PSD (log)")
    ax.set_title("Per-channel power spectrum (avg over chunks)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.axvline(5.0, color="red", linestyle="--", alpha=0.5, label="5 Hz")

    # (2) cumulative energy vs frequency
    ax = axes[1]
    for ch in range(psd.shape[1]):
        cum_ch = np.cumsum(psd[:, ch]) / (psd[:, ch].sum() + 1e-30)
        ax.plot(freqs, cum_ch, label=CHANNEL_NAMES[ch], alpha=0.8)
    ax.plot(freqs, cum, "k--", linewidth=2, label="overall (ch-avg)")
    ax.axhline(0.95, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(0.99, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("cumulative energy fraction")
    ax.set_title("Cumulative spectral energy → bandwidth at 95% / 99%")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.02])

    fig.tight_layout()
    fig.savefig(out_dir / "A_power_spectrum.png", dpi=130)
    plt.close(fig)

    report = {
        "n_chunks": int(chunks.shape[0]),
        "fs_hz": fs,
        "chunk_len_T": int(chunks.shape[1]),
        "freq_resolution_hz": float(fs / chunks.shape[1]),
        "per_channel": {
            CHANNEL_NAMES[ch]: {
                "bw_95_hz": float(bw95[ch]),
                "bw_99_hz": float(bw99[ch]),
                "hf5_ratio": float(hf5[ch]),
                "hf2_ratio": float(hf2[ch]),
            }
            for ch in range(len(CHANNEL_NAMES))
        },
        "overall": {
            "bw_95_hz": overall_bw95,
            "bw_99_hz": overall_bw99,
            "hf5_ratio_mean_ch": overall_hf5,
        },
    }
    return report


# ------------------------------- diagnostic B ------------------------------- #
def fit_chebyshev(chunk, K):
    """Least-squares fit K Chebyshev basis on [-1,1] per channel.
    chunk: (T, D). Returns reconstruction (T, D)."""
    T, D = chunk.shape
    x = np.linspace(-1.0, 1.0, T)
    recon = np.zeros_like(chunk)
    for d in range(D):
        coef = cheb.chebfit(x, chunk[:, d], K - 1)
        recon[:, d] = cheb.chebval(x, coef)
    return recon


def fit_fourier(chunk, K):
    """Keep top K low-frequency Fourier bins (0..K-1), zero rest, inverse FFT.
    chunk: (T, D). Returns recon (T, D)."""
    T, D = chunk.shape
    x = chunk - chunk.mean(axis=0, keepdims=True)  # remove DC for cleaner behavior
    X = np.fft.rfft(x, axis=0)
    # zero out bins >= K
    X[K:, :] = 0
    recon = np.fft.irfft(X, n=T, axis=0)
    recon = recon + chunk.mean(axis=0, keepdims=True)  # add DC back
    return recon


def fit_bspline(chunk, K, s_deg=3):
    """Cubic B-spline with K knots (roughly K control points).
    chunk: (T, D). Returns recon (T, D)."""
    T, D = chunk.shape
    x = np.linspace(0.0, 1.0, T)
    recon = np.zeros_like(chunk)
    for d in range(D):
        # Use a fixed knot vector with K-s_deg-1 internal knots so free params ≈ K
        n_internal = max(1, K - s_deg - 1)
        if n_internal >= 1:
            internal_knots = np.linspace(0.05, 0.95, n_internal)
            try:
                tck = splrep(x, chunk[:, d], k=s_deg, t=internal_knots)
                recon[:, d] = splev(x, tck)
            except Exception:
                # Fallback: smoothing spline with free knots
                tck = splrep(x, chunk[:, d], k=s_deg, s=0)
                recon[:, d] = splev(x, tck)
        else:
            # too few params — linear
            tck = splrep(x, chunk[:, d], k=min(3, T - 1), s=0)
            recon[:, d] = splev(x, tck)
    return recon


def diagnostic_B(chunks, out_dir: Path, K_list=(3, 5, 8, 12, 20, 50)):
    """For each basis type and K, compute MSE (averaged over chunks, channels)."""
    results = {b: {} for b in ["chebyshev", "fourier", "bspline"]}
    action_var = chunks.var(axis=(0, 1))          # (D,)  per-channel variance
    action_std_mean = float(np.sqrt(action_var).mean())

    for K in K_list:
        # reconstruct all chunks
        recons = {
            "chebyshev": np.stack([fit_chebyshev(c, K) for c in chunks]),
            "fourier": np.stack([fit_fourier(c, K) for c in chunks]),
            "bspline": np.stack([fit_bspline(c, K) for c in chunks]),
        }
        for name, recon in recons.items():
            err = recon - chunks
            mse = float((err ** 2).mean())
            # per-channel MSE
            mse_per_ch = (err ** 2).mean(axis=(0, 1))  # (D,)
            # normalized: relative to chunk variance
            rel_mse = float(mse / action_var.mean())
            results[name][K] = {
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "rmse_per_ch": list(map(float, np.sqrt(mse_per_ch))),
                "rel_mse_var": rel_mse,
            }

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for name, color in zip(["chebyshev", "fourier", "bspline"], ["C0", "C1", "C2"]):
        ks = sorted(results[name].keys())
        rmse_vals = [results[name][k]["rmse"] for k in ks]
        ax.semilogy(ks, rmse_vals, "-o", color=color, label=name)
    ax.set_xlabel("K (number of basis functions)")
    ax.set_ylabel("overall RMSE [rad / gripper units]")
    ax.set_title("Reconstruction RMSE vs K")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    for name, color in zip(["chebyshev", "fourier", "bspline"], ["C0", "C1", "C2"]):
        ks = sorted(results[name].keys())
        rel_vals = [results[name][k]["rel_mse_var"] for k in ks]
        ax.semilogy(ks, rel_vals, "-o", color=color, label=name)
    ax.set_xlabel("K (number of basis functions)")
    ax.set_ylabel("MSE / var(action)")
    ax.set_title("Normalised reconstruction error vs K\n(1.0 = predicting mean)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(0.01, color="red", linestyle="--", alpha=0.6, label="1% of variance")

    fig.tight_layout()
    fig.savefig(out_dir / "B_reconstruction.png", dpi=130)
    plt.close(fig)

    # Find K* = smallest K achieving rel_mse < 0.01 for each basis
    K_star = {}
    for name in results:
        found = None
        for K in sorted(results[name].keys()):
            if results[name][K]["rel_mse_var"] < 0.01:
                found = K
                break
        K_star[name] = found

    return {
        "action_std_mean": action_std_mean,
        "K_list": list(K_list),
        "results": {k: {str(kk): vv for kk, vv in v.items()} for k, v in results.items()},
        "K_star_at_1pct_var": K_star,
    }


# ---------------------------------- main ----------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-root",
        default=str(Path.home() / ".cache/huggingface/lerobot/local/piper_pick_place_imu_v1"),
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--fs", type=float, default=30.0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.dataset_root}")
    episodes = load_all_actions(Path(args.dataset_root))
    print(f"[load] {len(episodes)} episodes, "
          f"frames total = {sum(len(e['action']) for e in episodes)}")

    chunks = make_chunks(episodes, T=args.T, stride=args.stride)
    print(f"[chunks] {chunks.shape[0]} chunks × {args.T} steps × {chunks.shape[2]} ch")

    print("\n=== Diagnostic A: power spectrum ===")
    rep_A = diagnostic_A(chunks, out_dir, fs=args.fs)
    print(f"  overall bw95 = {rep_A['overall']['bw_95_hz']:.2f} Hz   "
          f"bw99 = {rep_A['overall']['bw_99_hz']:.2f} Hz   "
          f">5Hz ratio = {rep_A['overall']['hf5_ratio_mean_ch']*100:.2f}%")
    for ch, data in rep_A["per_channel"].items():
        print(f"  {ch:10s}  bw95={data['bw_95_hz']:5.2f}Hz  "
              f"bw99={data['bw_99_hz']:5.2f}Hz  "
              f">5Hz={data['hf5_ratio']*100:5.2f}%")

    print("\n=== Diagnostic B: basis reconstruction ===")
    rep_B = diagnostic_B(chunks, out_dir)
    print(f"  K* achieving rel_mse<1% : {rep_B['K_star_at_1pct_var']}")
    print(f"  action std (mean over ch) = {rep_B['action_std_mean']:.4f}")
    for name in ["chebyshev", "fourier", "bspline"]:
        print(f"  {name}:")
        for K in sorted([int(k) for k in rep_B["results"][name].keys()]):
            r = rep_B["results"][name][str(K)]
            print(f"    K={K:3d}  RMSE={r['rmse']:.5f}  rel_MSE/var={r['rel_mse_var']:.5f}")

    full = {
        "diagnostic_A": rep_A,
        "diagnostic_B": rep_B,
        "meta": {
            "T": args.T, "stride": args.stride, "fs_hz": args.fs,
            "n_chunks": int(chunks.shape[0]), "n_episodes": len(episodes),
        },
    }
    with open(out_dir / "phase0_report.json", "w") as f:
        json.dump(full, f, indent=2)

    print(f"\nsaved -> {out_dir}")


if __name__ == "__main__":
    main()
