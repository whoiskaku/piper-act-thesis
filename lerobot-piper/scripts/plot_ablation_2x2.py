#!/usr/bin/env python3
"""绘制 IMU 输入 / aux_head 消融实验的 2×2 组合柱状图。

四个配置：
  Vanilla         (IMU=off, aux=off)  - 基线
  α-imu-only      (IMU=on,  aux=off)  - 单 IMU
  α-aux-only      (IMU=off, aux=on)   - 单 aux
  α-full          (IMU=on,  aux=on)   - 双开
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    base = Path("/home/enolab-guo/claude-test-2/lerobot-piper/outputs/eval")

    # Load ablation data
    full = json.load(open(base / "step6_smoothness_v1/results.json"))["summary"]
    imu_only = json.load(open(base / "step6_ablation_imu_only/results.json"))["summary"]
    aux_only = json.load(open(base / "step6_ablation_aux_only/results.json"))["summary"]

    configs = [
        ("Vanilla\n(no IMU, no aux)",     full["rms_jerk_mean"]["vanilla"],   "C7"),
        ("α-imu-only\n(IMU only)",        imu_only["rms_jerk_mean"]["alpha"], "C2"),
        ("α-aux-only\n(aux only)",        aux_only["rms_jerk_mean"]["alpha"], "C1"),
        ("α-full\n(IMU + aux)",           full["rms_jerk_mean"]["alpha"],     "C3"),
    ]
    labels = [c[0] for c in configs]
    vals = [c[1] for c in configs]
    colors = [c[2] for c in configs]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

    # Annotate values + improvement vs vanilla
    vanilla_v = vals[0]
    for bar, v, label in zip(bars, vals, labels):
        change_pct = (v - vanilla_v) / vanilla_v * 100
        sign = "↓" if change_pct < 0 else "↑" if change_pct > 0 else ""
        annotation = f"{v:.0f}"
        if change_pct != 0:
            color = "green" if change_pct < 0 else "red"
            annotation += f"\n({sign}{abs(change_pct):.0f}%)"
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 5,
            annotation,
            ha="center", va="bottom",
            fontsize=11, fontweight="bold",
        )

    # Reference line for GT
    ax.axhline(147, color="black", linestyle="--", linewidth=1, alpha=0.6,
               label=f"GT demonstrations (jerk = 147)")

    ax.set_ylabel("Action RMS jerk (lower = smoother)", fontsize=12)
    ax.set_title("Ablation study: IMU input vs ΦSim aux loss\n"
                 "(Offline eval: 6 val episodes, identical 50K-step training)",
                 fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, max(vals) * 1.18])

    # Annotate boxes with key insights
    fig.text(0.02, 0.02,
             "Key findings:\n"
             "  • IMU input alone IMPROVES smoothness (jerk ↓ 11%)\n"
             "  • aux_head loss HARMS smoothness (jerk ↑ 67%)\n"
             "  • Combining both: aux dominates → worse than vanilla\n"
             "  → DECISION: keep IMU input, drop aux_head",
             fontsize=9, family="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.subplots_adjust(bottom=0.20)
    out_path = "/home/enolab-guo/claude-test-2/lerobot-piper/outputs/phase0/ablation_imu_vs_aux.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
