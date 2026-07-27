#!/usr/bin/env python3
"""汇报图 — Ĝ 模型示意图(结构 + 用法两联)。输出 reports/figures/G_schematic.png"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "reports/figures"
OUT.mkdir(parents=True, exist_ok=True)

C_ANA = "#dbeafe"   # analytic (zero-fit)  blue
C_FIT = "#fed7aa"   # fitted from logs     orange
C_FRZ = "#dcfce7"   # frozen schedule      green
C_DAT = "#e5e7eb"   # data / signal        gray
C_PI = "#fce7f3"    # pi model             pink
EC = "#374151"


def box(ax, x, y, w, h, text, fc, fs=9.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=EC, lw=1.2, mutation_scale=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, weight=weight, linespacing=1.35)


def arrow(ax, p0, p1, text="", fs=8.5, color=EC, ls="-", dy=0.018, ha="center"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                                 lw=1.4, color=color, linestyle=ls,
                                 shrinkA=2, shrinkB=2))
    if text:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my + dy, text, ha=ha, va="bottom", fontsize=fs, color="#111827")


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.2, 10.8),
                               gridspec_kw={"height_ratios": [1.25, 1]})
for ax in (ax1, ax2):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# ============ Panel A: inside G ============
ax1.set_title("(A)  Inside  $\\hat{G}$ :  command $\\rightarrow$ IMU response   "
              "(gray-box, all offline, fit on 20 existing log episodes)",
              fontsize=12.5, weight="bold", pad=6)

# input
box(ax1, 0.015, 0.56, 0.115, 0.16,
    "cmd  $a(t)$\n(+ $\\delta a$)\n6 joints, 20 Hz", C_DAT, 9)

# servo FIR
box(ax1, 0.185, 0.56, 0.155, 0.16,
    "① servo FIR  $\\hat{c}$\n10 taps / joint\ncmd $\\rightarrow$ joint vel $\\dot{q}$",
    C_FIT, 9)
arrow(ax1, (0.13, 0.64), (0.185, 0.64))

# rigid path
box(ax1, 0.42, 0.70, 0.20, 0.17,
    "② rigid projection\nJacobian  $J_r(q)$\n\"joints move $\\Rightarrow$ IMU moves\"",
    C_ANA, 9)
arrow(ax1, (0.34, 0.68), (0.42, 0.775), "$\\dot{q}$")

# excitation + flex path
box(ax1, 0.185, 0.18, 0.155, 0.16,
    "excitation\n$u = \\hat{c} * \\ddot{q}$\n(200 Hz spline-exact)", C_ANA, 8.8)
arrow(ax1, (0.075, 0.56), (0.185, 0.30), "$\\ddot{q}$ of cmd", ha="left")

box(ax1, 0.42, 0.14, 0.20, 0.24,
    "③ flexibility: 3 modal\nresonators (fitted poles\n3.1 / 5.1 / 5.7 Hz;\nfull-data member 3.68 Hz\n$\\approx$ measured 3.7 Hz)",
    C_FIT, 8.8)
arrow(ax1, (0.34, 0.26), (0.42, 0.26))

box(ax1, 0.42, 0.435, 0.20, 0.115,
    "pose-scheduled gains\n$\\alpha=\\mathrm{softplus}(\\theta^{T}\\phi(q_2,q_3,spd,grip))$",
    C_FRZ, 8.2)
arrow(ax1, (0.52, 0.435), (0.52, 0.38))
ax1.text(0.635, 0.475, "frozen w.r.t. $\\delta a$\n(nominal traj. only)", fontsize=7.6,
         color="#166534", ha="left", va="center")

# sum
box(ax1, 0.70, 0.42, 0.075, 0.14, "$+$", C_DAT, 15, "bold")
arrow(ax1, (0.62, 0.775), (0.72, 0.56), "rigid part", dy=-0.055, ha="left")
arrow(ax1, (0.62, 0.26), (0.72, 0.42), "flex part", ha="left")

# output
box(ax1, 0.835, 0.42, 0.15, 0.14,
    "predicted IMU\nresponse  $\\hat{y}(t)$\n(gyro 3ch focus)", C_DAT, 9)
arrow(ax1, (0.775, 0.49), (0.835, 0.49))

# legend + scoreboard
ax1.add_patch(FancyBboxPatch((0.70, 0.70), 0.285, 0.24, boxstyle="round,pad=0.012",
                             fc="white", ec=EC, lw=1))
ax1.text(0.715, 0.905, "acceptance (held-out):", fontsize=8.8, weight="bold", va="center")
ax1.text(0.715, 0.845, "gap-$R^2$ = 0.437  (family ceiling $\\approx$ 0.44)",
         fontsize=8.4, va="center")
ax1.text(0.715, 0.795, "signed direction cos = 0.758,  anti-phase 0%", fontsize=8.4, va="center")
ax1.text(0.715, 0.745, "$\\delta a$-channel strictly linear (probe 0.0%)", fontsize=8.4, va="center")
for x0, c, t in [(0.02, C_ANA, "analytic (zero fitting)"),
                 (0.27, C_FIT, "fitted from logs (least-squares / SGD)"),
                 (0.63, C_FRZ, "schedule frozen w.r.t. perturbation")]:
    ax1.add_patch(FancyBboxPatch((x0, 0.015), 0.028, 0.05, boxstyle="round,pad=0.008",
                                 fc=c, ec=EC, lw=0.8))
    ax1.text(x0 + 0.038, 0.04, t, fontsize=8.4, va="center")

# ============ Panel B: how G is used ============
ax2.set_title("(B)  How $\\hat{G}$ is used :  increment trick + closed-loop training of $\\pi$   "
              "($K{=}5$ bootstrap ensemble; $\\hat{G}$ never deployed on robot)",
              fontsize=12.5, weight="bold", pad=6)

box(ax2, 0.015, 0.60, 0.16, 0.22,
    "logged rollout\n$e_{log}=real-ideal$\n(nominal, $\\delta a\\equiv 0$)", C_DAT, 9)
box(ax2, 0.015, 0.13, 0.16, 0.22,
    "causal features\n(err window / ideal\npreview / cmd / $\\dot{q}$)", C_DAT, 9)

box(ax2, 0.25, 0.13, 0.14, 0.22, "$\\pi$\n(corrector,\nbeing trained)", C_PI, 10)
arrow(ax2, (0.175, 0.24), (0.25, 0.24))

box(ax2, 0.46, 0.13, 0.17, 0.22,
    "$\\hat{G}_{\\delta}$ : counterfactual\nincrement\n$\\Delta e=\\hat{G}[\\delta a]$",
    C_FIT, 9)
arrow(ax2, (0.39, 0.24), (0.46, 0.24))
ax2.text(0.425, 0.375, "$\\delta a$  ($\\pm10$ mrad hard bound)", fontsize=8.5, ha="center")

box(ax2, 0.70, 0.37, 0.115, 0.20, "$+$", C_DAT, 15, "bold")
arrow(ax2, (0.175, 0.71), (0.72, 0.52), "nominal: free, from data", dy=0.035, ha="left")
arrow(ax2, (0.63, 0.22), (0.735, 0.37))
ax2.text(0.655, 0.155, "increment: only part $\\hat{G}$ owns", fontsize=8.5, ha="left")

box(ax2, 0.855, 0.37, 0.13, 0.20,
    "$e_{sim}$\n= corrected\nworld", C_DAT, 9)
arrow(ax2, (0.815, 0.47), (0.855, 0.47))

box(ax2, 0.63, 0.72, 0.355, 0.21,
    "loss $=\\frac{1}{K}\\sum_{k}\\sum_{t} w(t)\\,\\|BP_{0.5-9\\,Hz}(e^{k}_{sim})\\|^2"
    "+\\beta\\|\\delta a\\|^2+\\gamma\\|\\Delta\\delta a\\|^2$\n"
    "= advisor's metric “distance to ideal”, band-limited, motion-weighted",
    "white", 9)
arrow(ax2, (0.92, 0.57), (0.87, 0.72))
arrow(ax2, (0.63, 0.80), (0.32, 0.35), color="#b91c1c", ls="--")
ax2.text(0.235, 0.52, "backprop through $\\hat{G}$\n(linear $\\Rightarrow$ gradient =\n"
         "identified impulse response)", fontsize=8.5, color="#b91c1c",
         ha="center", va="top")

ax2.text(0.5, 0.015,
         "deploy: $\\pi$ runs alone in real time (<1 ms/frame, IMU in loop);   "
         "$\\hat{G}$ is retired after training;   new logs $\\Rightarrow$ refit $\\hat{G}$ "
         "$\\Rightarrow$ retrain $\\pi$  (DAgger-style loop)",
         fontsize=9.2, ha="center", style="italic", color="#374151")

fig.tight_layout()
fig.savefig(OUT / "G_schematic.png", dpi=170)
print(OUT / "G_schematic.png")
