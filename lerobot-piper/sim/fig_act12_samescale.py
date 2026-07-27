#!/usr/bin/env python3
"""Act1 / Act2 同尺度对照版（2026-07-22 用户要求）。

两张图**完全同轴同刻度**：同一墙钟时间轴、同一 3 个窗、每列 y 轴上下限一致。
唯一的差别是红线的身份：
    Act 1  红 = $\\hat{G}$ 模拟的 real        （问: 纸上机器人像不像真机）
    Act 2  红 = 修正后的 real（模型内预测）   （问: 修正把 real 拉向 ideal 了吗）
黄(ideal)/蓝(real) 在两张图里**逐点相同**——所以滞后关系不会再变。

口径统一到墙钟: 修正量 de 原在 e-索引, 按 corrected_wall[j] = real_wall[j] + de[j−L] 搬回。
用法: python sim/fig_act12_samescale.py [ins-deploy|pp2-deploy] [ep]
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
from rerace_dropout import TCN_D

FIG = Path(__file__).resolve().parents[1] / "reports/figures"
FPS, CHUNK, WIN_S = 20.0, 100, 4.0
C_IDEAL, C_REAL, C_NEW = "#d9a400", "#2563eb", "#dc2626"
TAG = sys.argv[1] if len(sys.argv) > 1 else "ins-deploy"
EP = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SHORT, L = TAG.split("-")[0], TP.LSHIFT
REPO = {"pp2": "enolab-guo/eval_pickplace2_100k",
        "ins": "enolab-guo/eval_insert_100k"}[SHORT]

TP.BP = TP.make_bp()
bp = lambda x: TP.BP(torch.from_numpy(np.ascontiguousarray(x, np.float32))
                     .view(1, 1, -1, 3).to(TP.DEV))[0, 0].cpu().numpy()

# ---------- 墙钟基准：ideal / real / Ĝ ----------
R = np.array(json.loads((TP.LROOT / TAG / "manifest.json").read_text())["R"], np.float32)
v = [x for x in gv_load([EP], with_gap=True) if x["tag"] == TAG][0]
real_w = v["g"].astype(np.float32)
ideal_w = (v["g"] - v["gap"]).astype(np.float32)
T = len(real_w)
cmd7 = load_episode(REPO, EP)["action"][:T].astype(np.float32)
g0 = load_ensemble(rigid=Rigid())[0]
g0.R = R
pred_w = g0.forward(cmd7)

# ---------- 修正后（e-索引 → 搬回墙钟）----------
G = TP.GEnsemble()
ep = [e for e in TP.load_eps([EP], G) if e["tag"] == TAG][0]
des = []
for sd in range(5):
    m = TCN_D(d=TP.FEAT).to(TP.DEV)
    m.load_state_dict(torch.load(TP.OUT / f"pi_v1_s{sd}.pt", map_location=TP.DEV))
    m.eval()
    with torch.no_grad():
        des.append(TP.apply_G(m(ep["X"].unsqueeze(0)), ep["epJ"], ep["alpha"],
                              ep["RB"], ep["RD"], G)[0].mean(0).cpu().numpy())
corr_w_all = []
for de in des:                                     # corrected_wall[j] = real_wall[j] + de[j−L]
    c = real_w.copy()
    n = min(len(de), T - L)
    c[L:L + n] += de[:n]
    corr_w_all.append(c)
corr_w = corr_w_all[0]

gap_m = bp(real_w - ideal_w)                       # 实测 gap（墙钟）
gap_g = bp(pred_w - ideal_w)                       # Ĝ 模拟 gap
gap_c = [bp(c - ideal_w) for c in corr_w_all]      # 修正后 gap（5 种子）

# ---------- 选窗（墙钟口径，两图共用）----------
w = int(WIN_S * FPS)
en = np.convolve((gap_m ** 2).sum(1), np.ones(w), "same")
en[:w] = en[-w:] = 0
picks, tmp = [], en.copy()
while len(picks) < 3:
    c = int(np.argmax(tmp))
    picks.append(c)
    tmp[max(0, c - w):c + w] = 0
picks.sort()

# ---------- 每列的公共 y 范围（两图共用）----------
cols = []
for c in picks:
    s0, s1 = max(0, c - w // 2), min(T, c + w // 2)
    ch = int(np.argmax((gap_m[s0:s1] ** 2).sum(0)))
    top = np.concatenate([ideal_w[s0:s1, ch], real_w[s0:s1, ch],
                          pred_w[s0:s1, ch], corr_w[s0:s1, ch]])
    bot = np.concatenate([gap_m[s0:s1, ch], gap_g[s0:s1, ch], gap_c[0][s0:s1, ch]])
    pad = lambda a: (a.min() - 0.08 * np.ptp(a), a.max() + 0.08 * np.ptp(a))
    cols.append((s0, s1, ch, pad(top), pad(bot)))


def draw(act):
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.2), sharex="col")
    for k, (s0, s1, ch, ylim_t, ylim_b) in enumerate(cols):
        t = np.arange(s0, s1) / FPS
        ax = axes[0, k]
        ax.plot(t, ideal_w[s0:s1, ch], color=C_IDEAL, lw=1.9, label="ideal (intent)")
        ax.plot(t, real_w[s0:s1, ch], color=C_REAL, lw=1.5, label="real (measured)")
        red = pred_w if act == 1 else corr_w
        ax.plot(t, red[s0:s1, ch], color=C_NEW, lw=1.4, ls="--",
                label="$\\hat{G}$-simulated real" if act == 1
                else "corrected real (model-predicted)")
        ax.set_ylim(*ylim_t)
        for b in range(s0 - s0 % CHUNK, s1 + CHUNK, CHUNK):
            if s0 <= b <= s1:
                ax.axvline(b / FPS, color="#9ca3af", lw=0.9, ls=":")
        ax.set_title(f"hard spot {k+1}  (gyro-{'xyz'[ch]})", fontsize=10.5, weight="bold")
        ax.grid(alpha=0.25)
        if k == 0:
            ax.set_ylabel("angular vel (rad/s)")
            ax.legend(fontsize=8.4, loc="lower right")

        ax = axes[1, k]
        ax.plot(t, gap_m[s0:s1, ch], color=C_REAL, lw=1.6,
                label="measured gap = real $-$ ideal")
        g2 = gap_g if act == 1 else gap_c[0]
        ax.plot(t, g2[s0:s1, ch], color=C_NEW, lw=1.5, ls="--",
                label="$\\hat{G}$ simulated gap" if act == 1 else "gap after correction")
        ax.axhline(0, color="#9ca3af", lw=0.8)
        ax.set_ylim(*ylim_b)
        if act == 2:
            rs = np.array([1 - (g[s0:s1] ** 2).sum() / (gap_m[s0:s1] ** 2).sum()
                           for g in gap_c])
            ax.text(0.02, 0.05, f"window energy {(-rs.mean())*100:+.0f}"
                    f"$\\pm${rs.std()*100:.0f}%  (5 seeds)",
                    transform=ax.transAxes, fontsize=9.5, weight="bold", color=C_NEW)
        for b in range(s0 - s0 % CHUNK, s1 + CHUNK, CHUNK):
            if s0 <= b <= s1:
                ax.axvline(b / FPS, color="#9ca3af", lw=0.9, ls=":")
        ax.grid(alpha=0.25)
        ax.set_xlabel("time (s)   —   WALL-CLOCK (both figures identical axes)")
        if k == 0:
            ax.set_ylabel("band-passed gap 0.5–9 Hz\n(rad/s)")
            ax.legend(fontsize=8.4, loc="upper right")

    q = ("Act 1 — is the paper robot credible?   red = $\\hat{G}$-simulated real"
         if act == 1 else
         "Act 2 — does the correction move real toward ideal?   red = corrected real")
    fig.suptitle(f"{q}\n{TAG} ep{EP} (held-out) · wall-clock · same windows & same y-limits "
                 f"as the companion figure · yellow & blue are pointwise identical in both",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    p = FIG / f"same_act{act}_{SHORT}_ep{EP}.png"
    fig.savefig(p, dpi=155)
    return p


for a in (1, 2):
    print(draw(a))
tot = np.mean([1 - (g ** 2).sum() / (gap_m ** 2).sum() for g in gap_c])
print(f"整条 episode gap 能量: {-tot*100:+.1f}% (墙钟口径, 全帧, 5种子)")
