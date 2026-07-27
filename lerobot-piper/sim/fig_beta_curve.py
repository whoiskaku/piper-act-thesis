#!/usr/bin/env python3
"""β 完整曲线 (新口径: 逐chunk ideal, π v1=TCN, new25 数据)。
(A) 效果 vs β  (B) 兑换率: 效果 vs 修正力度  (C) 峰值/饱和 —— ε 何时开始约束
输出 reports/figures/beta_curve.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports/figures"
P = ROOT / "outputs/p2_pi"
fin = json.loads((P / "beta_curve_final.json").read_text())     # β=1000, 100
eps = json.loads((P / "eps_probe.json").read_text())            # β∈{300,30}×ε
v1 = json.loads((P / "pi_v1.json").read_text())                 # β=300 定档(5种子)

D = {1000: fin["b1000"], 300: eps["b300_e10"], 100: fin["b100"], 30: eps["b30_e10"]}
D[300] = {**D[300], "ppf": v1["ppf"]}                            # 定档的 fork 值
betas = [30, 100, 300, 1000]
DEF = 300
C_PP, C_INS, C_FORK = "#2563eb", "#dc2626", "#059669"

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))

# ---- (A) 效果 vs β ----
ax = axes[0]
for key, c, lbl in [("pp2", C_PP, "pick-place"), ("ins", C_INS, "insert")]:
    m = [D[b][key][0] for b in betas]
    e = [D[b][key][1] for b in betas]
    ax.errorbar(betas, m, yerr=e, marker="o", ms=6, color=c, lw=2, capsize=3, label=lbl)
    for b, y in zip(betas, m):
        ax.annotate(f"{y:.1f}", (b, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.5, color=c)
ax.axvline(DEF, color="#111827", lw=1.4, ls="--")
ax.annotate("adopted\ndefault", xy=(DEF, 25), xytext=(DEF * 2.6, 27),
            fontsize=9, weight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", lw=1.2))
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlabel("$\\beta$   (effort price)      ←  weaker penalty")
ax.set_ylabel("band-gap energy reduction (%)")
ax.set_title("(A)  monotone to $\\beta$=30 — no interior optimum\n"
             "$\\beta$ is not a tuning knob, it is an operating point",
             fontsize=11, weight="bold")
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc="lower left")

# ---- (B) 兑换率 ----
ax = axes[1]
for key, c, lbl in [("pp2", C_PP, "pick-place"), ("ins", C_INS, "insert")]:
    x = [D[b]["da"][0] for b in betas]
    y = [D[b][key][0] for b in betas]
    ax.plot(x, y, marker="o", ms=6, color=c, lw=2, label=lbl)
    k = np.polyfit(x, y, 1)[0]
    ax.text(0.03, 0.93 if key == "pp2" else 0.86,
            f"{lbl}: {k:.1f} pp per mrad", transform=ax.transAxes,
            fontsize=9, weight="bold", color=c)
for b in betas:
    ax.annotate(f"$\\beta$={b}", (D[b]["da"][0], D[b]["pp2"][0]),
                textcoords="offset points", xytext=(4, -12), fontsize=7.5, color=C_PP)
ax.axvline(D[DEF]["da"][0], color="#111827", lw=1.4, ls="--")
ax.set_xlabel("$\\|\\delta a\\|_{RMS}$  (mrad)        —        $\\epsilon$ hard bound = 10 mrad")
ax.set_ylabel("reduction (%)")
ax.set_title("(B)  exchange rate: effect $\\leftrightarrow$ effort\n"
             "more effort $\\Rightarrow$ more exposure to $\\hat{G}$ gain error (1.3–1.5$\\times$)",
             fontsize=11, weight="bold")
ax.set_xlim(0, 2.7)
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc="lower right")

# ---- (C) 峰值 / 饱和 ----
ax = axes[2]
pk = [D[b]["damax"][0] for b in betas]
rms = [D[b]["da"][0] for b in betas]
sat = [D[b]["sat"][0] for b in betas]
ax.plot(betas, pk, marker="s", ms=6, color="#7c3aed", lw=2, label="peak $|\\delta a|$")
ax.plot(betas, rms, marker="o", ms=6, color="#9ca3af", lw=2, label="RMS $\\|\\delta a\\|$")
ax.axhline(10, color="#111827", lw=1.6, ls="--")
ax.text(700, 10.4, "$\\epsilon$ = 10 mrad  (hard bound)", fontsize=9, weight="bold")
ax.axvline(DEF, color="#111827", lw=1.0, ls=":")
for b, p, s in zip(betas, pk, sat):
    if s > 0:
        ax.annotate(f"sat {s:.1f}%", (b, p), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=8, weight="bold", color="#b91c1c")
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlabel("$\\beta$")
ax.set_ylabel("$\\delta a$  (mrad)")
ax.set_ylim(0, 12.5)
ax.set_title("(C)  when does $\\epsilon$ start to bind?\n"
             "peaks — not RMS — decide; 0.7% saturation already costs 3.7 pp",
             fontsize=11, weight="bold")
ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc="center left")

fig.suptitle("$\\beta$ curve — new caliber (chunk-rendered ideal, $\\pi$ v1 = TCN, new25 data, "
             "3 seeds; model-internal, contact frames excluded)",
             fontsize=12.5, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(FIG / "beta_curve.png", dpi=155)

rows = []
for b in betas:
    d = D[b]
    rows.append((b, d["pp2"], d["ins"], d.get("ppf", [float("nan"), 0]),
                 d["da"], d["damax"], d["sat"]))
print(f"{'β':>6}{'pp2':>15}{'ins':>15}{'fork':>8}{'‖δa‖':>8}{'峰值':>8}{'饱和':>8}")
for b, pp, ins, pf, da, mx, st in rows:
    pfs = "—" if pf[0] != pf[0] else f"{pf[0]:.1f}"
    print(f"{b:>6}{pp[0]:>9.1f}±{pp[1]:<4.1f}{ins[0]:>9.1f}±{ins[1]:<4.1f}"
          f"{pfs:>8}{da[0]:>8.2f}{mx[0]:>8.1f}{st[0]:>7.1f}%")
print(FIG / "beta_curve.png")
