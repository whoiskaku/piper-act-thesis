#!/usr/bin/env python3
"""钱图 — 修正效果三联(留出集, 墙最厚的3个窗):
  行1 原始陀螺: 黄=ideal / 蓝=real / 红=修正后(模型内预测)
  行2 带通gap: 修正前 vs 修正后 —— 消减在这一行才看得见
  行3 δa: 修正器实际做了什么(含 ε 硬界参考线)
诚实标注: "修正后"= real_measured + Ĝ_δ[δa], **模型内预测**, 非真机实测。
输出 reports/figures/correction_effect.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import train_pi_e2 as TP
from rerace_dropout import TCNDeep_D

import sys

FIG = Path(__file__).resolve().parents[1] / "reports/figures"
FPS, CHUNK = 20.0, 100
C_IDEAL, C_REAL, C_CORR = "#d9a400", "#2563eb", "#dc2626"
TAG = sys.argv[1] if len(sys.argv) > 1 else "pp2-deploy"
EP, SEED = 8, 0

TP.BP = TP.make_bp()
G = TP.GEnsemble()
eps = TP.load_eps([EP], G)
ep = [e for e in eps if e["tag"] == TAG][0]

model = TCNDeep_D(d=TP.FEAT).to(TP.DEV)
model.load_state_dict(torch.load(TP.OUT / f"pi3_TCN-deep_s{SEED}.pt", map_location=TP.DEV))
model.eval()
with torch.no_grad():
    delta = model(ep["X"].unsqueeze(0))                      # (1,n,6)
    de = TP.apply_G(delta, ep["epJ"], ep["alpha"], ep["RB"], ep["RD"], G)  # (1,K,n,3)
da = delta[0].cpu().numpy()
de_mean = de[0].mean(0).cpu().numpy()                        # K成员均值 (n,3)

z = np.load(TP.LROOT / TAG / f"ep{EP:03d}.npz")
ideal = z["ideal"][:, 3:6].astype(np.float32)                # e-索引, 陀螺
real = z["real"][:, 3:6].astype(np.float32)
corr = real + de_mean                                        # 修正后(模型内预测)
n = len(ideal)

bp = lambda x: TP.BP(torch.from_numpy(x).view(1, 1, -1, 3).to(TP.DEV))[0, 0].cpu().numpy()
gap0, gap1 = bp(real - ideal), bp(corr - ideal)

# 选墙最厚的 3 个非重叠窗
w = int(4 * FPS)
en = np.convolve((gap0 ** 2).sum(1), np.ones(w), "same")
en[:w] = en[-w:] = 0
picks = []
tmp = en.copy()
while len(picks) < 3:
    c = int(np.argmax(tmp))
    picks.append(c)
    tmp[max(0, c - w):c + w] = 0
picks.sort()

fig, axes = plt.subplots(3, 3, figsize=(15.5, 9.6), sharex="col",
                         gridspec_kw={"height_ratios": [1.15, 1, 0.85]})
for col, c in enumerate(picks):
    s0, s1 = max(0, c - w // 2), min(n, c + w // 2)
    ch = int(np.argmax((gap0[s0:s1] ** 2).sum(0)))
    t = np.arange(s0, s1) / FPS

    ax = axes[0, col]
    ax.plot(t, ideal[s0:s1, ch], color=C_IDEAL, lw=1.9, label="ideal (intent)")
    ax.plot(t, real[s0:s1, ch], color=C_REAL, lw=1.5, label="real (measured)")
    ax.plot(t, corr[s0:s1, ch], color=C_CORR, lw=1.3, ls="--",
            label="corrected (model-predicted)")
    for b in range(s0 - s0 % CHUNK, s1 + CHUNK, CHUNK):
        if s0 <= b <= s1:
            ax.axvline(b / FPS, color="#9ca3af", lw=0.9, ls=":")
    ax.set_title(f"hard spot {col+1}  (gyro-{'xyz'[ch]})", fontsize=10.5, weight="bold")
    ax.grid(alpha=0.25)
    if col == 0:
        ax.set_ylabel("angular vel (rad/s)")
        ax.legend(fontsize=8.2, loc="lower right")

    ax = axes[1, col]
    ax.plot(t, gap0[s0:s1, ch], color=C_REAL, lw=1.6, label="gap before = real $-$ ideal")
    ax.plot(t, gap1[s0:s1, ch], color=C_CORR, lw=1.6, label="gap after correction")
    ax.axhline(0, color="#9ca3af", lw=0.8)
    r = 1 - (gap1[s0:s1] ** 2).sum() / (gap0[s0:s1] ** 2).sum()
    ax.text(0.02, 0.05, f"window energy {(-r)*100:+.0f}%", transform=ax.transAxes,
            fontsize=9.5, weight="bold", color=C_CORR)
    ax.grid(alpha=0.25)
    if col == 0:
        ax.set_ylabel("band-passed gap 0.5–9 Hz\n(rad/s)")
        ax.legend(fontsize=8.2, loc="upper right")

    ax = axes[2, col]
    for j in range(6):
        ax.plot(t, da[s0:s1, j] * 1000, lw=1.0, alpha=0.85, label=f"j{j+1}" if col == 0 else None)
    ax.axhline(TP.EPS_B * 1000, color="#111827", lw=0.9, ls="--")
    ax.axhline(-TP.EPS_B * 1000, color="#111827", lw=0.9, ls="--")
    ax.set_ylim(-TP.EPS_B * 1100, TP.EPS_B * 1100)
    ax.set_xlabel("time (s)")
    ax.grid(alpha=0.25)
    if col == 0:
        ax.set_ylabel("$\\delta a$ (mrad)\ndashed = $\\pm\\epsilon$ hard bound")
        ax.legend(fontsize=7.5, ncol=3, loc="upper right")

full0 = 1 - (gap1 ** 2).sum() / (gap0 ** 2).sum()
fig.suptitle(f"Does the correction move real toward ideal?  —  {TAG} ep{EP} (held-out), "
             f"3 worst-deviation windows;  episode band-gap energy $-${full0*100:.0f}%\n"
             "$\\pi$ = TCN-deep (race3);  \"corrected\" $=$ measured real $+\\ \\hat{G}_\\delta[\\delta a]$ "
             "— MODEL-PREDICTED, not yet measured on the robot",
             fontsize=12, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIG / f"correction_effect_{TAG.split('-')[0]}.png", dpi=155)

full = 1 - (gap1 ** 2).sum() / (gap0 ** 2).sum()
mot = ~z["mask_still"]
fm = 1 - (gap1[mot] ** 2).sum() / (gap0[mot] ** 2).sum()
print(f"整条 episode: gap 能量 −{full*100:.1f}%   运动段 −{fm*100:.1f}%   "
      f"‖δa‖={np.sqrt((da**2).mean())*1000:.2f}mrad  峰值={np.abs(da).max()*1000:.1f}mrad")
print(FIG / f"correction_effect_{TAG.split('-')[0]}.png")
