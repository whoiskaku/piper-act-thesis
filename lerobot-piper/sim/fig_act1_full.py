#!/usr/bin/env python3
"""第①幕 全程版 — ideal / real / Ĝ-simulated 整条 episode（与 hard-spot 版互补）。

hard-spot 版回答「难处对不对」; 本图回答「全程是否一致、有没有挑窗嫌疑」。
三轴各一行(不再自动挑主导轴), 底部为带通 gap 全程对照; 灰底=hard-spot 版选的 3 个窗,
让两张图能对上号。数值仍为零(第②幕才标效果量)。
用法: python sim/fig_act1_full.py [pp2-deploy|ins-deploy]
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import train_pi_e2 as TP
from compare_ideal_methods import load_episode
from g_model import Rigid, load_ensemble
from g_verdict import load as gv_load

FIG = Path(__file__).resolve().parents[1] / "reports/figures"
FPS, CHUNK = 20.0, 100
C_IDEAL, C_REAL, C_G = "#d9a400", "#2563eb", "#dc2626"
TAG = sys.argv[1] if len(sys.argv) > 1 else "ins-deploy"
REPO = {"pp2-deploy": "enolab-guo/eval_pickplace2_100k",
        "ins-deploy": "enolab-guo/eval_insert_100k"}[TAG]
EP = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SHORT = TAG.split("-")[0]

TP.BP = TP.make_bp()
bp = lambda x: TP.BP(torch.from_numpy(np.ascontiguousarray(x, np.float32))
                     .view(1, 1, -1, 3).to(TP.DEV))[0, 0].cpu().numpy()

R = np.array(json.loads((TP.LROOT / TAG / "manifest.json").read_text())["R"], np.float32)
v = [x for x in gv_load([EP], with_gap=True) if x["tag"] == TAG][0]
real = v["g"].astype(np.float32)                       # 真机系, 墙钟
ideal = (v["g"] - v["gap"]).astype(np.float32)
T = len(real)
cmd7 = load_episode(REPO, EP)["action"][:T].astype(np.float32)
g0 = load_ensemble(rigid=Rigid())
g0 = g0[0]
g0.R = R
pred = g0.forward(cmd7)
gap_m, gap_s = bp(real - ideal), bp(pred - ideal)
t = np.arange(T) / FPS

# hard-spot 版选的 3 个窗(用第②幕口径重算, 便于两图对照)
zl = np.load(TP.LROOT / TAG / f"ep{EP:03d}.npz")
g0e = bp(zl["real"][:, 3:6].astype(np.float32) - zl["ideal"][:, 3:6].astype(np.float32))
w = int(4 * FPS)
en = np.convolve((g0e ** 2).sum(1), np.ones(w), "same")
en[:w] = en[-w:] = 0
picks, tmp = [], en.copy()
while len(picks) < 3:
    c = int(np.argmax(tmp))
    picks.append(c)
    tmp[max(0, c - w):c + w] = 0
picks.sort()

fig, axes = plt.subplots(4, 1, figsize=(16, 11), sharex=True,
                         gridspec_kw={"height_ratios": [1, 1, 1, 1.15]})
for ax in axes:
    for c in picks:                                    # 标出 hard-spot 版的 3 个窗
        ax.axvspan((c + TP.LSHIFT - w // 2) / FPS, (c + TP.LSHIFT + w // 2) / FPS,
                   color="#fde68a", alpha=0.35, zorder=0)
    for b in range(0, T, CHUNK):
        ax.axvline(b / FPS, color="#d1d5db", lw=0.7, ls=":", zorder=1)
    ax.grid(alpha=0.22)

for i, axis in enumerate("xyz"):
    ax = axes[i]
    ax.plot(t, ideal[:, i], color=C_IDEAL, lw=1.4, label="ideal (intent)")
    ax.plot(t, real[:, i], color=C_REAL, lw=1.0, alpha=0.9, label="real (measured)")
    ax.plot(t, pred[:, i], color=C_G, lw=1.0, ls="--", alpha=0.9,
            label="$\\hat{G}$-simulated")
    ax.set_ylabel(f"gyro-{axis}\n(rad/s)")
    if i == 0:
        ax.legend(fontsize=9, ncol=3, loc="upper right")

ax = axes[3]
ax.plot(t, np.linalg.norm(gap_m, axis=1), color=C_REAL, lw=1.1,
        label="measured gap  |real $-$ ideal|  (band-passed, 3-axis norm)")
ax.plot(t, np.linalg.norm(gap_s, axis=1), color=C_G, lw=1.1, ls="--",
        label="simulated gap  |$\\hat{G}$ $-$ ideal|")
ax.set_ylabel("band-passed gap\nmagnitude (rad/s)")
ax.set_xlabel("time (s)      —      dotted verticals = chunk boundaries,  "
              "shaded = the 3 windows shown in the hard-spot figure")
ax.legend(fontsize=9, loc="upper right")

fig.suptitle(f"Act 1 (full episode) — ideal / real / $\\hat{{G}}$-simulated   ·   "
             f"{TAG} ep{EP}, held-out, entire {T/FPS:.0f} s, all 3 gyro axes\n"
             "no window selection, no averages — complements the hard-spot view",
             fontsize=13, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIG / f"evidence_act1_full_{SHORT}_ep{EP}.png", dpi=150)
print(FIG / f"evidence_act1_full_{SHORT}_ep{EP}.png")
