#!/usr/bin/env python3
"""Phase 0 诊断 B2 (H1 直接证据)：对比三个 action 来源的功率谱。

  1. GT 演示 (piper_pick_place_imu_v1)
  2. Vanilla ACT 真机执行 (eval_vanilla_v1)
  3. α-imu-only ACT 真机执行 (eval_imu_only_v1)

在同一张图里画：
  - per-channel PSD 对比（每个关节一张子图）
  - overall 累积能量曲线
  - 高频 (>5Hz) 能量占比 bar chart
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CHANNEL_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4",
                 "joint_5", "joint_6", "gripper"]


def load_actions(dataset_root: Path):
    parquets = sorted(dataset_root.glob("data/chunk-*/episode_*.parquet"))
    if not parquets:
        raise RuntimeError(f"no parquet under {dataset_root}")
    eps = []
    for pq in parquets:
        df = pd.read_parquet(pq)
        a = np.stack(df["action"].to_numpy()).astype(np.float32)
        eps.append(a)
    return eps


def make_chunks(eps, T=100, stride=10):
    chs = []
    for a in eps:
        n = len(a)
        if n < T:
            continue
        for s in range(0, n - T + 1, stride):
            chs.append(a[s:s + T])
    return np.stack(chs)


def psd(chunks, fs=30.0):
    N, T, D = chunks.shape
    x = chunks - chunks.mean(axis=1, keepdims=True)
    w = np.hanning(T)[None, :, None]
    Xf = np.fft.rfft(x * w, axis=1)
    power = (np.abs(Xf) ** 2).mean(axis=0)  # (F, D)
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    return freqs, power


def hf_ratio(freqs, power, cutoff=5.0):
    mask = freqs > cutoff
    total = power.sum(axis=0)
    return power[mask].sum(axis=0) / (total + 1e-30)


def bw_at(freqs, power, frac=0.95):
    cum = np.cumsum(power, axis=0)
    cum_n = cum / (cum[-1:] + 1e-30)
    idx = (cum_n >= frac).argmax(axis=0)
    return freqs[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dataset",
                    default=str(Path.home() / ".cache/huggingface/lerobot/local/piper_pick_place_imu_v1"))
    ap.add_argument("--vanilla-dataset",
                    default=str(Path.home() / ".cache/huggingface/lerobot/local/eval_vanilla_v1"))
    ap.add_argument("--alpha-dataset",
                    default=str(Path.home() / ".cache/huggingface/lerobot/local/eval_imu_only_v1"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--fs", type=float, default=30.0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "GT (demo)": Path(args.gt_dataset),
        "Vanilla ACT (real-robot exec)": Path(args.vanilla_dataset),
        "α-imu-only (real-robot exec)": Path(args.alpha_dataset),
    }
    colors = {"GT (demo)": "black", "Vanilla ACT (real-robot exec)": "C3",
              "α-imu-only (real-robot exec)": "C0"}
    styles = {"GT (demo)": "--", "Vanilla ACT (real-robot exec)": "-",
              "α-imu-only (real-robot exec)": "-"}

    spectra = {}
    for label, root in sources.items():
        eps = load_actions(root)
        chunks = make_chunks(eps, T=args.T, stride=args.stride)
        freqs, p = psd(chunks, fs=args.fs)
        spectra[label] = {
            "freqs": freqs, "psd": p,
            "n_chunks": len(chunks),
            "bw95": bw_at(freqs, p, 0.95),
            "bw99": bw_at(freqs, p, 0.99),
            "hf5_ratio": hf_ratio(freqs, p, 5.0),
            "hf2_ratio": hf_ratio(freqs, p, 2.0),
        }
        print(f"[{label}]  n_chunks={len(chunks)}  "
              f"overall_bw95={bw_at(freqs, p.mean(axis=1,keepdims=True), 0.95)[0]:.2f}Hz  "
              f"hf5={p[freqs>5].sum()/p.sum()*100:.3f}%")

    # ================== plot 1: per-channel PSD overlay ==================
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    for ch, ax in zip(range(7), axes.flat):
        for label in sources:
            s = spectra[label]
            ax.semilogy(s["freqs"], s["psd"][:, ch] + 1e-18,
                        color=colors[label], linestyle=styles[label],
                        label=label, alpha=0.9)
        ax.set_title(f"{CHANNEL_NAMES[ch]}")
        ax.grid(alpha=0.3)
        ax.axvline(5.0, color="red", linestyle=":", alpha=0.5)
        if ch == 0:
            ax.legend(loc="upper right", fontsize=8)
    # hide unused subplots (7th and 8th)
    for ch in range(7, 9):
        axes.flat[ch].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("frequency [Hz]")
    for ax in axes[:, 0]:
        ax.set_ylabel("PSD (log)")
    fig.suptitle("Action power spectrum: GT vs ACT variants\n"
                 "(red dotted = 5 Hz; ACT curves rising above GT in mid/high freq = HF pollution)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "H1_psd_compare.png", dpi=130)
    plt.close(fig)

    # ================== plot 2: overall cumulative energy ==================
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for label in sources:
        s = spectra[label]
        p_mean = s["psd"].mean(axis=1)
        cum = np.cumsum(p_mean) / (p_mean.sum() + 1e-30)
        ax.plot(s["freqs"], cum, color=colors[label], linestyle=styles[label],
                linewidth=2, label=label)
    ax.axhline(0.95, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(0.99, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(5.0, color="red", linestyle=":", alpha=0.5, label="5 Hz cutoff")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("cumulative energy fraction")
    ax.set_title("Cumulative spectral energy (overall, channel-averaged)\n"
                 "GT reaches 0.95 fastest → demonstrates strong low-pass prior ACT is violating")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    fig.savefig(out_dir / "H1_cumulative_energy.png", dpi=130)
    plt.close(fig)

    # ================== plot 3: >5Hz HF ratio bar chart per-channel ==================
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    labels = list(sources.keys())
    width = 0.25
    x = np.arange(len(CHANNEL_NAMES))
    for i, label in enumerate(labels):
        vals = spectra[label]["hf5_ratio"] * 100
        ax.bar(x + (i - 1) * width, vals, width=width, label=label,
               color=colors[label], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNEL_NAMES, rotation=30)
    ax.set_ylabel(">5Hz energy ratio [%]")
    ax.set_title("Action high-frequency energy per channel\n"
                 "(GT ≈ 0%, ACT variants significantly elevated → H1 direct evidence)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "H1_hf_ratio_bars.png", dpi=130)
    plt.close(fig)

    # ================== summary JSON ==================
    report = {}
    for label, s in spectra.items():
        report[label] = {
            "n_chunks": s["n_chunks"],
            "per_channel": {
                CHANNEL_NAMES[ch]: {
                    "bw_95_hz": float(s["bw95"][ch]),
                    "bw_99_hz": float(s["bw99"][ch]),
                    "hf5_ratio_pct": float(s["hf5_ratio"][ch] * 100),
                    "hf2_ratio_pct": float(s["hf2_ratio"][ch] * 100),
                } for ch in range(len(CHANNEL_NAMES))
            },
            "overall": {
                "hf5_ratio_pct": float((s["psd"][s["freqs"] > 5].sum() / s["psd"].sum()) * 100),
                "hf2_ratio_pct": float((s["psd"][s["freqs"] > 2].sum() / s["psd"].sum()) * 100),
            },
        }
    with open(out_dir / "H1_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # print comparison table
    print("\n" + "=" * 82)
    print(f"{'Source':<40s}  {'overall >5Hz':>13s}  {'j1 >5Hz':>10s}  {'grip >5Hz':>10s}")
    print("-" * 82)
    for label in sources:
        o = report[label]["overall"]["hf5_ratio_pct"]
        j1 = report[label]["per_channel"]["joint_1"]["hf5_ratio_pct"]
        gr = report[label]["per_channel"]["gripper"]["hf5_ratio_pct"]
        print(f"{label:<40s}  {o:>12.4f}%  {j1:>9.4f}%  {gr:>9.4f}%")
    print("=" * 82)

    print(f"\nsaved -> {out_dir}")


if __name__ == "__main__":
    main()
