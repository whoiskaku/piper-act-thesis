#!/usr/bin/env python3
"""FF 参数 → gap 的变化（2026-07-29）。

采样路径（两个因子各自固定另一个）：
  ① `ff_lag` 扫描：2 → 3        （有效 blend 固定为 2）
  ② 有效 blend 扫描：2 → 5 → 10 （`ff_lag` 固定为 3）

有效 blend = `ff_blend` − `ff_lag`（先 blend 后移位，前 lag 帧被丢弃）。

gap/帧 = Σ(bandpass(real−ideal)²)/帧数，陀螺三轴，0.5–9 Hz FIR，未对齐口径。
误差棒 = 10 集的 SEM。

用法: python sim/plot_ff_sweep.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
SC = Path("/tmp/claude-1000/-home-enolab-guo-claude-test-2/"
          "675e2ec5-9ec0-48d8-a265-066e293ec7fe/scratchpad")

# 臂 → (ff_lag, 有效 blend, 残余滞后[帧], 标注)
CFG = {
    "A":       (0, 0, 3, "裸 ACT（无 FF）"),
    "B":       (2, 2, 1, "原设"),
    "B3":      (3, 2, 0, ""),
    "B3b8":    (3, 5, None, ""),
    "B3b13":   (3, 10, None, "定档"),
    "B3b13r2": (3, 10, None, "独立重复"),
}


def load(a):
    return np.array(json.load(open(SC / f"g3_{a}.json")))[:, 0]


def main():
    E = {a: load(a) for a in CFG}
    A0 = E["A"].mean()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle("FF 参数 → gap 的变化　（pp2，每点 10 集，误差棒 = SEM，"
                 "gap = real−ideal 陀螺 0.5–9 Hz 能量/帧）", fontsize=11)

    def draw(a_, arms, xs, xlabel, title, extra=None):
        for x, arm in zip(xs, arms):
            v = E[arm]
            a_.errorbar(x, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)),
                        fmt="o", ms=8, capsize=5, lw=2,
                        color="tab:brown" if arm.startswith("B3b13") else "tab:green")
            a_.scatter(np.full(len(v), x) + np.random.default_rng(0).normal(0, .04, len(v)),
                       v, s=12, alpha=.35, color="gray", zorder=0)
            a_.annotate(f"{v.mean():.4f}\n{v.mean()/A0*100-100:+.1f}%",
                        (x, v.mean()), textcoords="offset points", xytext=(11, -4),
                        fontsize=8.5)
        a_.axhline(A0, ls="--", c="tab:blue", lw=1.5, label=f"A 裸 ACT {A0:.4f}")
        a_.set_xlabel(xlabel); a_.set_ylabel("gap 能量 / 帧")
        a_.set_title(title, fontsize=10); a_.grid(alpha=.3); a_.legend(fontsize=8.5)

    # ① ff_lag（有效 blend 固定 = 2）
    draw(ax[0], ["B", "B3"], [2, 3], "ff_lag（帧）",
         "① ff_lag 扫描　有效 blend 固定 = 2")
    ax[0].set_xticks([2, 3])
    ax[0].set_xlim(1.5, 3.9)

    # ② 有效 blend（ff_lag 固定 = 3）
    draw(ax[1], ["B3", "B3b8", "B3b13"], [2, 5, 10], "有效 blend（帧）",
         "② 有效 blend 扫描　ff_lag 固定 = 3")
    v = E["B3b13r2"]
    ax[1].errorbar(10.6, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)),
                   fmt="s", ms=8, capsize=5, lw=2, color="tab:orange",
                   label=f"独立重复 {v.mean():.4f}")
    ax[1].legend(fontsize=8.5)
    ax[1].set_xticks([2, 5, 10]); ax[1].set_xlim(1, 12.5)

    # ③ 机制：残余滞后 vs ff_lag
    lags = [0, 2, 3]
    res = [3, 1, 0]
    ax[2].plot(lags, res, "o-", ms=9, lw=2, color="tab:red")
    for x, y in zip(lags, res):
        ax[2].annotate(f"{y} 帧", (x, y), textcoords="offset points",
                       xytext=(9, 4), fontsize=9)
    ax[2].plot([0, 3], [2.74, -0.26], ":", c="gray", lw=1.5,
               label="理论：2.74 − ff_lag\n（亚帧实测真值 2.74 帧 = 137 ms）")
    ax[2].axhline(0, c="k", lw=.8)
    ax[2].set_xlabel("ff_lag（帧）"); ax[2].set_ylabel("残余滞后（帧）")
    ax[2].set_title("③ 机制：残余滞后随 ff_lag 精确抵消\n（由对齐移位扫描独立测得）",
                    fontsize=10)
    ax[2].set_xticks(lags); ax[2].grid(alpha=.3); ax[2].legend(fontsize=8.5)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = ROOT / "outputs/ab2/FF_sweep.png"
    fig.savefig(out, dpi=140)
    print(f"→ {out.relative_to(ROOT.parent)}\n")
    print(f"  {'臂':>9}{'ff_lag':>8}{'有效blend':>11}{'残余滞后':>10}"
          f"{'gap/帧':>11}{'±SEM':>9}{'相对 A':>10}")
    for a, (lg, bl, rl, note) in CFG.items():
        v = E[a]
        print(f"  {a:>9}{lg:>8}{bl:>11}{str(rl) if rl is not None else '—':>10}"
              f"{v.mean():>11.5f}{v.std(ddof=1)/np.sqrt(len(v)):>9.5f}"
              f"{v.mean()/A0*100-100:>9.1f}%  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
