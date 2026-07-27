#!/usr/bin/env python3
"""决策图 — β 兑换率曲线 × 数据扩容 (A+B 实验主结论, 2026-07-22)。
左: 效果 vs β(new25, 3种子) + old16 对照; 中: 效果 vs 修正力度‖δa‖(兑换率);
右: 数据扩容因子图(种子散点, 显示方差塌缩)。输出 reports/figures/beta_data_tradeoff.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports/figures"
low = json.loads((ROOT / "outputs/p2_pi/beta_low.json").read_text())
ab = json.loads((ROOT / "outputs/p2_pi/ab_expand.json").read_text())
C_PP, C_INS, C_FORK = "#2563eb", "#dc2626", "#059669"

betas = [30, 100, 300, 1000]
new = {30: low["b30"], 100: low["b100"], 300: ab["new25_b300"], 1000: ab["new25_b1000"]}
old = {300: ab["old16_b300"], 1000: ab["old16_b1000"]}
DEF = 300

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))

# ---- (A) 效果 vs β ----
ax = axes[0]
for key, c, lbl in [("pp2", C_PP, "pick-place"), ("ins", C_INS, "insert"),
                    ("ppf", C_FORK, "fork variant")]:
    m = [new[b][key][0] for b in betas]
    e = [new[b][key][1] for b in betas]
    ax.errorbar(betas, m, yerr=e, marker="o", color=c, lw=2, capsize=3, label=f"{lbl} (new25)")
for key, c in [("pp2", C_PP), ("ins", C_INS)]:
    bs = sorted(old)
    ax.plot(bs, [old[b][key][0] for b in bs], marker="s", ms=5, color=c, lw=1.2,
            ls=":", alpha=0.65, label=None)
ax.axvline(DEF, color="#111827", lw=1.4, ls="--")
ax.annotate("adopted default", xy=(DEF, 30), xytext=(DEF*2.4, 33),
            fontsize=9, weight="bold",
            arrowprops=dict(arrowstyle="->", lw=1.2))
ax.set_xscale("log")
ax.set_xlabel("$\\beta$  (effort price)")
ax.set_ylabel("band-gap energy reduction (%)")
ax.set_title("(A) monotone — no interior optimum\n(dotted = old16 data, for contrast)",
             fontsize=11, weight="bold")
ax.grid(alpha=0.3)
ax.legend(fontsize=8.5, loc="upper right")
ax.invert_xaxis()

# ---- (B) 兑换率: 效果 vs 力度 ----
ax = axes[1]
for key, c, lbl in [("pp2", C_PP, "pick-place"), ("ins", C_INS, "insert")]:
    x = [new[b]["da"][0] for b in betas]
    y = [new[b][key][0] for b in betas]
    ax.plot(x, y, marker="o", color=c, lw=2, label=lbl)
    for b, xi, yi in zip(betas, x, y):
        ax.annotate(f"$\\beta$={b}", (xi, yi), textcoords="offset points",
                    xytext=(5, -11), fontsize=7.5, color=c)
ax.axvline(new[DEF]["da"][0], color="#111827", lw=1.4, ls="--")
ax.axvspan(0, 10, color="#f3f4f6", zorder=0)
ax.set_xlabel("$\\|\\delta a\\|_{RMS}$ (mrad)      —      $\\epsilon$ hard bound = 10 mrad")
ax.set_ylabel("reduction (%)")
ax.set_title("(B) exchange rate: ~4 pp per mrad\nmore effort $\\Rightarrow$ more exposure "
             "to $\\hat{G}$ gain error (1.3–1.5$\\times$)", fontsize=11, weight="bold")
ax.set_xlim(0, 2.8)
ax.grid(alpha=0.3)
ax.legend(fontsize=8.5, loc="lower right")

# ---- (C) 数据扩容 ----
ax = axes[2]
seeds_old = [6.3, 13.3, 8.9]
seeds_new = [12.1, 11.8, 11.3]
seeds_old3 = [8.7, 14.3, 10.7]
seeds_new3 = [16.6, 15.7, 15.9]
groups = [("old16\n$\\beta$=1000", seeds_old), ("new25\n$\\beta$=1000", seeds_new),
          ("old16\n$\\beta$=300", seeds_old3), ("new25\n$\\beta$=300", seeds_new3)]
for i, (lbl, vals) in enumerate(groups):
    c = C_PP if "new" in lbl else "#9ca3af"
    ax.scatter([i] * 3, vals, color=c, s=52, zorder=3, alpha=0.9)
    ax.errorbar(i, np.mean(vals), yerr=np.std(vals), color=c, capsize=6, lw=2.2, zorder=2)
    ax.annotate(f"±{np.std(vals):.1f}", (i, np.mean(vals)),
                textcoords="offset points", xytext=(26, -3),
                fontsize=9, weight="bold", color=c)
ax.set_xticks(range(4))
ax.set_xticklabels([g[0] for g in groups], fontsize=9)
ax.set_ylabel("pick-place reduction (%)")
ax.set_xlim(-0.45, 3.75)
ax.set_title("(C) data expansion: mean $\\uparrow$ and seed spread $\\downarrow$ 7$\\times$\n"
             "(high seed variance = the fingerprint of data starvation)",
             fontsize=11, weight="bold")
ax.grid(alpha=0.3, axis="y")

fig.suptitle("Choosing the operating point — $\\beta$ (effort price) $\\times$ training-data size"
             "   ·   TCN-deep, 3 seeds, model-internal caliber (contact frames excluded)",
             fontsize=12.5, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(FIG / "beta_data_tradeoff.png", dpi=155)
print(FIG / "beta_data_tradeoff.png")
