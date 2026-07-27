#!/usr/bin/env python3
"""证据链三部曲 · 第①②幕（统一版式, 2026-07-22 用户定稿结构）。

  第①幕 Ĝ 可信吗:   ideal / real / Ĝ-pred        (+ gap行: 实测gap vs 模拟gap)
  第②幕 π 有效吗:   ideal / real / corrected-pred (+ gap行: 修正前 vs 修正后; + δa行)
  第③幕 真机验证:   待真机 (fig_evidence_act3, 未来)

统一语法: 黄=ideal, 蓝=real, **红=本幕受审的新事物**; 三列=同三个"墙最厚"时刻
(由第②幕口径选窗, 第①幕换算 +LSHIFT 到墙钟) — 两幕看的是同三个时刻。
口径注: 第①幕=墙钟(未补滞后, forward 的自然口径); 第②幕=e-索引(补滞后, π 的口径)。
第①幕不标任何数字(纯波形判断, 老师规矩); 第②幕标窗口能量变化(效果量, 非拟合指标)。
用法: python sim/fig_evidence_chain.py [pp2-deploy|ins-deploy]
"""
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
SEED = 0
SHORT = TAG.split("-")[0]

TP.BP = TP.make_bp()
bp = lambda x: TP.BP(torch.from_numpy(np.ascontiguousarray(x, np.float32))
                     .view(1, 1, -1, 3).to(TP.DEV))[0, 0].cpu().numpy()

# ---------- 数据: 第②幕 (e-索引) ----------
G = TP.GEnsemble()
ep = [e for e in TP.load_eps([EP], G) if e["tag"] == TAG][0]
z = np.load(TP.LROOT / TAG / f"ep{EP:03d}.npz")
ideal_e = z["ideal"][:, 3:6].astype(np.float32)
real_e = z["real"][:, 3:6].astype(np.float32)
gap0 = bp(real_e - ideal_e)
n_e = len(ideal_e)
# 曲线画 s0; 窗口能量统计用全部 5 种子(2026-07-22: 单种子窗口数≈±3%噪声, 见探针2)
gap1_by_seed, da, corr_e = {}, None, None
for seed in range(5):
    model = TCN_D(d=TP.FEAT).to(TP.DEV)
    model.load_state_dict(torch.load(TP.OUT / f"pi_v1_s{seed}.pt",
                                     map_location=TP.DEV))
    model.eval()
    with torch.no_grad():
        delta = model(ep["X"].unsqueeze(0))
        de = TP.apply_G(delta, ep["epJ"], ep["alpha"], ep["RB"], ep["RD"], G)
    gap1_by_seed[seed] = bp(real_e + de[0].mean(0).cpu().numpy() - ideal_e)
    if seed == SEED:
        da = delta[0].cpu().numpy()
        corr_e = real_e + de[0].mean(0).cpu().numpy()
gap1 = gap1_by_seed[SEED]

# ---------- 数据: 第①幕 (墙钟) ----------
R = np.array(json.loads((TP.LROOT / TAG / "manifest.json").read_text())["R"], np.float32)
val = [v for v in gv_load([EP], with_gap=True) if v["tag"] == TAG][0]
real_w = val["g"].astype(np.float32)                       # 真机系, 墙钟
ideal_w = (val["g"] - val["gap"]).astype(np.float32)       # = kin @ R.T
T_w = len(real_w)
repo = dict(pp2="enolab-guo/eval_pickplace2_100k",
            ins="enolab-guo/eval_insert_100k")[SHORT]
cmd7 = load_episode(repo, EP)["action"][:T_w].astype(np.float32)
g0 = load_ensemble(rigid=Rigid())[0]
g0.R = R
pred_w = g0.forward(cmd7)                                  # Ĝ 模拟 real
gapm_w, gaps_w = bp(real_w - ideal_w), bp(pred_w - ideal_w)

# ---------- 共用选窗 (第②幕口径) ----------
w = int(WIN_S * FPS)
en = np.convolve((gap0 ** 2).sum(1), np.ones(w), "same")
en[:w] = en[-w:] = 0
picks, tmp = [], en.copy()
while len(picks) < 3:
    c = int(np.argmax(tmp))
    picks.append(c)
    tmp[max(0, c - w):c + w] = 0
picks.sort()


def style_col(ax, s0, s1, col, title):
    for b in range(s0 - s0 % CHUNK, s1 + CHUNK, CHUNK):
        if s0 <= b <= s1:
            ax.axvline(b / FPS, color="#9ca3af", lw=0.9, ls=":")
    ax.set_title(title, fontsize=10.5, weight="bold")
    ax.grid(alpha=0.25)


# ══════════ 第①幕 ══════════
fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.0), sharex="col")
for col, c in enumerate(picks):
    cw = c + TP.LSHIFT                                     # e-索引 → 墙钟
    s0, s1 = max(0, cw - w // 2), min(T_w, cw + w // 2)
    ch = int(np.argmax((gapm_w[s0:s1] ** 2).sum(0)))
    t = np.arange(s0, s1) / FPS
    ax = axes[0, col]
    ax.plot(t, ideal_w[s0:s1, ch], color=C_IDEAL, lw=1.9, label="ideal (intent)")
    ax.plot(t, real_w[s0:s1, ch], color=C_REAL, lw=1.5, label="real (measured)")
    ax.plot(t, pred_w[s0:s1, ch], color=C_NEW, lw=1.4, ls="--",
            label="$\\hat{G}$-simulated real")
    style_col(ax, s0, s1, col, f"hard spot {col+1}  (gyro-{'xyz'[ch]})")
    if col == 0:
        ax.set_ylabel("angular vel (rad/s)")
        ax.legend(fontsize=8.4, loc="lower right")
    ax = axes[1, col]
    ax.plot(t, gapm_w[s0:s1, ch], color=C_REAL, lw=1.6,
            label="measured gap = real $-$ ideal")
    ax.plot(t, gaps_w[s0:s1, ch], color=C_NEW, lw=1.5, ls="--",
            label="simulated gap = $\\hat{G}$ $-$ ideal")
    ax.axhline(0, color="#9ca3af", lw=0.8)
    style_col(ax, s0, s1, col, "")
    ax.set_xlabel("time (s)")
    if col == 0:
        ax.set_ylabel("band-passed gap 0.5–9 Hz\n(rad/s)")
        ax.legend(fontsize=8.4, loc="upper right")
fig.suptitle(f"Act 1/3 — Is the paper robot credible?   ideal / real / $\\hat{{G}}$-simulated"
             f"  ({TAG} ep{EP}, held-out; 3 worst-deviation moments, no averages)\n"
             "WALL-CLOCK time: real lags ideal by ~150 ms — that lag is real, and $\\hat{G}$ "
             "reproduces it\nlook for: lag timing · boundary-transient gap spikes · ringing frequency",
             fontsize=11.5, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.text(0.5, 0.005, "curves show the dominant gyro axis per window (auto-selected); ALL energy figures sum the 3 axes", ha="center", fontsize=9, style="italic", color="#374151")
fig.savefig(FIG / f"evidence_act1_{SHORT}_ep{EP}.png", dpi=155)

# ══════════ 第②幕 ══════════
fig, axes = plt.subplots(3, 3, figsize=(15.5, 9.6), sharex="col",
                         gridspec_kw={"height_ratios": [1.15, 1, 0.7]})
for col, c in enumerate(picks):
    s0, s1 = max(0, c - w // 2), min(n_e, c + w // 2)
    ch = int(np.argmax((gap0[s0:s1] ** 2).sum(0)))
    t = np.arange(s0, s1) / FPS
    ax = axes[0, col]
    ax.plot(t, ideal_e[s0:s1, ch], color=C_IDEAL, lw=1.9, label="ideal (intent)")
    ax.plot(t, real_e[s0:s1, ch], color=C_REAL, lw=1.5, label="real (measured)")
    ax.plot(t, corr_e[s0:s1, ch], color=C_NEW, lw=1.4, ls="--",
            label="corrected (model-predicted)")
    style_col(ax, s0, s1, col, f"hard spot {col+1}  (gyro-{'xyz'[ch]})")
    if col == 0:
        ax.set_ylabel("angular vel (rad/s)")
        ax.legend(fontsize=8.4, loc="lower right")
    ax = axes[1, col]
    ax.plot(t, gap0[s0:s1, ch], color=C_REAL, lw=1.6, label="gap before")
    ax.plot(t, gap1[s0:s1, ch], color=C_NEW, lw=1.6, label="gap after correction")
    ax.axhline(0, color="#9ca3af", lw=0.8)
    rs = np.array([1 - (gap1_by_seed[s][s0:s1] ** 2).sum() / (gap0[s0:s1] ** 2).sum()
                   for s in range(5)])
    ax.text(0.02, 0.05,
            f"window energy {(-rs.mean())*100:+.0f}$\\pm${rs.std()*100:.0f}%  (5 seeds)",
            transform=ax.transAxes, fontsize=9.5, weight="bold", color=C_NEW)
    style_col(ax, s0, s1, col, "")
    if col == 0:
        ax.set_ylabel("band-passed gap 0.5–9 Hz\n(rad/s)")
        ax.legend(fontsize=8.4, loc="upper right")
    ax = axes[2, col]
    for j in range(6):
        ax.plot(t, da[s0:s1, j] * 1000, lw=1.0, alpha=0.85,
                label=f"j{j+1}" if col == 0 else None)
    ax.axhline(TP.EPS_B * 1000, color="#111827", lw=0.9, ls="--")
    ax.axhline(-TP.EPS_B * 1000, color="#111827", lw=0.9, ls="--")
    ax.set_ylim(-TP.EPS_B * 1100, TP.EPS_B * 1100)
    ax.set_xlabel("time (s)")
    ax.grid(alpha=0.25)
    if col == 0:
        ax.set_ylabel("$\\delta a$ (mrad)\ndashed = $\\pm\\epsilon$")
        ax.legend(fontsize=7.5, ncol=3, loc="upper right")
msk_eval = (ep["w"] > 0).cpu().numpy()          # 评估口径: 剔接触帧(π 训练时零权)
full = 1 - (gap1[msk_eval] ** 2).sum() / (gap0[msk_eval] ** 2).sum()
full_all = 1 - (gap1 ** 2).sum() / (gap0 ** 2).sum()
fig.suptitle(f"Act 2/3 — Does the correction move real toward ideal?   "
             f"ideal / real / corrected  ({TAG} ep{EP}, held-out; episode band-gap "
             f"energy {(-full)*100:+.0f}% excl. contact / {(-full_all)*100:+.0f}% all frames)\n"
             "e-INDEX time: real shifted by +150 ms so the servo lag is already removed — "
             "the lag is FF1's job, $\\pi$ targets what remains\n"
             r"$\pi$ v1 = TCN (chunk-rendered ideal), new25, $\beta$=300;  corrected $=$ "
             "measured real $+\\ \\hat{G}_\\delta[\\delta a]$ — MODEL-PREDICTED, "
             "not yet measured on the robot (e-index caliber)",
             fontsize=12, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.text(0.5, 0.005, "curves show the dominant gyro axis per window (auto-selected); ALL energy figures sum the 3 axes", ha="center", fontsize=9, style="italic", color="#374151")
fig.savefig(FIG / f"evidence_act2_{SHORT}_ep{EP}.png", dpi=155)
print(FIG / f"evidence_act1_{SHORT}_ep{EP}.png")
print(FIG / f"evidence_act2_{SHORT}_ep{EP}.png")
