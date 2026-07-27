#!/usr/bin/env python3
"""老师要的图 — 难处波形叠图(留出集): 黄=ideal, 蓝=real, 红=Ĝ合成。
选"墙最厚"的3个窗口(非平均!), 上排=波形叠图, 下排=事后算的gap对账。
零指标数字(老师: それっぽい指標は意味がない)。输出 reports/figures/G_hardspots.png"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from compare_ideal_methods import kin_ideal_20, load_episode
from fit_g_level2 import FlexL2, excitation, raw_feats
from fit_g_production import BP, OUT, bp_t, rigid_parts
from g_verdict import DATASETS, VAL_EPS, load

FIG = Path(__file__).resolve().parents[1] / "reports/figures"
FPS, CHUNK = 20.0, 100
C_IDEAL, C_REAL, C_G = "#d9a400", "#2563eb", "#dc2626"   # 黄/蓝/红(老师白板配色)

# ---- member-0 Ĝ 重建(同 fig_g_waveform) ----
z = np.load(OUT / "g_ensemble.npz")
c_enc, mu, sd = z["c_enc"][0], z["mu"][0], z["sd"][0]
fx = FlexL2([3.1, 5.1, 5.7])
with torch.no_grad():
    for name in ["log_om", "raw_ze", "W", "B", "D", "theta"]:
        getattr(fx, name).copy_(torch.from_numpy(z[name][0]))

# ---- 留出集 pp2 ep8 ----
val = load(VAL_EPS, with_gap=True)
i = 0
for repo, tag in DATASETS:
    for ei in VAL_EPS:
        d = load_episode(repo, ei)
        val[i]["a7"] = d["action"][:len(val[i]["a"])].astype(np.float32)
        val[i]["_ep"], val[i]["_repo"] = ei, repo
        i += 1
e = val[0]
kin, _, _ = kin_ideal_20(e["_repo"], e["_ep"])
T = len(e["a"])
ideal_g = kin[:T, 3:].astype(np.float32)                # ideal gyro(ideal系, 无滞后=意图)
real_g = (e["g"] @ e["R"]).astype(np.float32)           # real gyro 旋到 ideal系
_, _, w, _ = rigid_parts(e, c_enc)
u = excitation(e["a"], c_enc)
phi = ((raw_feats(e["a7"]) - mu) / sd).astype(np.float32)
with torch.no_grad():
    flex = fx(torch.from_numpy(u), torch.from_numpy(phi)).numpy()
ghat = w + flex                                          # Ĝ 合成(含滞后/震铃)

# ---- 找"墙最厚"的3个窗(带通gap能量, 非重叠) ----
wall = np.linalg.norm(bp_t(real_g - ideal_g, BP), axis=1)
win = int(4 * FPS)
sm = np.convolve(wall ** 2, np.ones(win), "same")
picks = []
tmp = sm.copy()
tmp[:win], tmp[-win:] = 0, 0
while len(picks) < 3:
    c = int(np.argmax(tmp))
    picks.append(c)
    tmp[max(0, c - win):c + win] = 0
picks.sort()

fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), sharex="col")
for col, c in enumerate(picks):
    s0, s1 = c - win // 2, c + win // 2
    ch = int(np.argmax((bp_t(real_g[s0:s1] - ideal_g[s0:s1], BP) ** 2).sum(0)))
    t = np.arange(s0, s1) / FPS
    ax = axes[0, col]
    ax.plot(t, ideal_g[s0:s1, ch], color=C_IDEAL, lw=1.8, label="ideal IMU (intent)")
    ax.plot(t, real_g[s0:s1, ch], color=C_REAL, lw=1.6, label="real IMU (measured)")
    ax.plot(t, ghat[s0:s1, ch], color=C_G, lw=1.4, ls="--",
            label="$\\hat{G}$ simulated IMU")
    for b in range(s0 - s0 % CHUNK, s1 + CHUNK, CHUNK):
        if s0 <= b <= s1:
            ax.axvline(b / FPS, color="#6b7280", lw=1, ls=":")
            ax.text(b / FPS, ax.get_ylim()[1], " chunk\n boundary", fontsize=7.5,
                    va="top", color="#6b7280")
    ax.set_title(f"hard spot {col+1}  (worst deviation window, gyro-{'xyz'[ch]})",
                 fontsize=10.5, weight="bold")
    if col == 0:
        ax.set_ylabel("angular vel (rad/s)")
        ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.25)

    ax = axes[1, col]
    ax.plot(t, real_g[s0:s1, ch] - ideal_g[s0:s1, ch], color=C_REAL, lw=1.6,
            label="measured gap = real $-$ ideal")
    ax.plot(t, ghat[s0:s1, ch] - ideal_g[s0:s1, ch], color=C_G, lw=1.4, ls="--",
            label="simulated gap = $\\hat{G}$ $-$ ideal")
    ax.axhline(0, color="#9ca3af", lw=0.8)
    if col == 0:
        ax.set_ylabel("gap (rad/s)")
        ax.legend(fontsize=8.5, loc="lower right")
    ax.set_xlabel("time (s)")
    ax.grid(alpha=0.25)

fig.suptitle("Held-out episode, 3 WORST-deviation windows (no averages) — "
             "does $\\hat{G}$ reproduce the deviation where it matters?",
             fontsize=12.5, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(FIG / "G_hardspots.png", dpi=160)
print(FIG / "G_hardspots.png", "picks(frame) =", picks)
