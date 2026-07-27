#!/usr/bin/env python3
"""模型结构一页图 — Ĝ(135参数 灰盒三层并联) vs π(11,334参数 因果膨胀卷积)。
输出 reports/figures/arch_spec.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "reports/figures"
C_ANA, C_FIT, C_FRZ, C_DAT, C_PI = "#dbeafe", "#fed7aa", "#dcfce7", "#e5e7eb", "#fce7f3"
EC = "#374151"


def box(ax, x, y, w, h, txt, fc, fs=8.6, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010",
                                fc=fc, ec=EC, lw=1.1))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
            fontsize=fs, weight=weight, linespacing=1.35)


def arr(ax, p0, p1, txt="", fs=7.8, dx=0.0, dy=0.015):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12,
                                 lw=1.25, color=EC, shrinkA=2, shrinkB=2))
    if txt:
        ax.text((p0[0] + p1[0]) / 2 + dx, (p0[1] + p1[1]) / 2 + dy, txt,
                ha="center", va="bottom", fontsize=fs)


fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.5, 8.4))
for ax in (axL, axR):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

# ═════════ 左: Ĝ ═════════
axL.set_title("$\\hat{G}$  —  world model (knowledge) · gray-box · 135 params\n"
              "3 parallel paths, every parameter has a physical name",
              fontsize=12, weight="bold", pad=8)
box(axL, 0.02, 0.86, 0.30, 0.09, "cmd  $a(t)$  (T,7)\nrad, 20 Hz   (+ $\\delta a$)", C_DAT, 9)
box(axL, 0.02, 0.68, 0.30, 0.11,
    "① servo FIR  $\\hat{c}$   (6,10)\ncmd vel → joint vel\n60 params · closed-form LS", C_FIT)
box(axL, 0.36, 0.68, 0.26, 0.11,
    "excitation  $u=\\hat{c}*\\ddot{q}$\n200 Hz spline-exact\n0 params", C_ANA)
arr(axL, (0.17, 0.86), (0.17, 0.79))
arr(axL, (0.32, 0.90), (0.49, 0.79), "$\\ddot{q}$", dy=-0.05)

box(axL, 0.68, 0.50, 0.30, 0.14,
    "② rigid  $J_r(q)$  (T,3,6)\nMuJoCo forward kinematics\n0 params · analytic", C_ANA)
arr(axL, (0.32, 0.735), (0.68, 0.615), "$\\dot{\\hat{q}}$", dx=0.10, dy=0.012)

box(axL, 0.02, 0.30, 0.40, 0.28,
    "③ flexible modes  (FlexL2)\n\n"
    "$W$ (3,6)   joint → mode          18\n"
    "$\\times\\ \\alpha(t)$ = softplus($\\theta^{T}\\phi$)   pose sched.\n"
    "$\\theta$ (3,8)                                24\n"
    "$\\circledast\\ k_m$ (3,60)  damped resonators\n"
    "$\\log\\omega$ (3) · $\\zeta$ (3)                     6\n"
    "$B$ (3,3)   mode → gyro axis         9\n\n"
    "57 params", C_FIT, 8.2)
box(axL, 0.46, 0.30, 0.20, 0.13,
    "③' direct  $D$ (3,6)\nno resonator,\nno scheduling\n18 params", C_FIT)
arr(axL, (0.45, 0.68), (0.26, 0.585), "$u$", dx=-0.02, dy=0.010)
arr(axL, (0.56, 0.68), (0.56, 0.435), "$u$", dx=0.025, dy=-0.12)

box(axL, 0.71, 0.30, 0.10, 0.10, "$\\Sigma$", C_DAT, 15, "bold")
arr(axL, (0.42, 0.44), (0.71, 0.37))
arr(axL, (0.66, 0.36), (0.71, 0.35))
arr(axL, (0.83, 0.50), (0.78, 0.40))
box(axL, 0.71, 0.14, 0.27, 0.09, "$\\hat{y}(t)$  (T,3)\nsimulated gyro  rad/s", C_DAT, 9)
arr(axL, (0.76, 0.30), (0.79, 0.23))

axL.text(0.02, 0.035,
         "structural guarantees (architecture, not training):\n"
         "· causal kernels + damping locked to (0, 0.5)  ⇒  never diverges\n"
         "· $\\alpha$ reads the nominal trajectory only  ⇒  $\\delta a$ channel exactly linear\n"
         "· parallel paths  ⇒  separable audit  (① vs encoder, ③ vs residual)\n"
         "fitted poles (member 0): 1.58 / 3.68 / 4.76 Hz   —   measured resonance 3.7 Hz\n"
         "ensemble K = 5  →  675 params total",
         fontsize=8.1, va="bottom", linespacing=1.5)

# ═════════ 右: π ═════════
axR.set_title("$\\pi$  —  corrector (skill) · causal dilated CNN · 11,334 params\n"
              "one structural constraint only:  $\\tanh\\times\\epsilon$",
              fontsize=12, weight="bold", pad=8)
box(axR, 0.06, 0.845, 0.88, 0.105,
    "input  (B, T, 51)\n"
    "$e$(6, ends at $t{-}150$ms) | ideal $t,t{+}4,t{+}8$ (18) | "
    "cmd $t,t{+}4,t{+}8$ (18) | $\\dot{q}$(6) | phase(3)", C_DAT, 8.6)

rows = [("Conv1d(51 → 32,  k=3,  dilation=1)   left-pad 2", "4,928", 0.68),
        ("Conv1d(32 → 32,  k=3,  dilation=2)   left-pad 4", "3,104", 0.545),
        ("Conv1d(32 → 32,  k=3,  dilation=4)   left-pad 8", "3,104", 0.41)]
for txt, n, y in rows:
    box(axR, 0.06, y, 0.70, 0.105, txt + "\nGELU  →  Dropout(0.1)  *train only*", C_PI, 8.6)
    axR.text(0.80, y + 0.052, n, fontsize=10, weight="bold", va="center")
    arr(axR, (0.41, y + 0.135), (0.41, y + 0.105))
arr(axR, (0.41, 0.845), (0.41, 0.785))

box(axR, 0.06, 0.275, 0.70, 0.095,
    "Linear(32 → 6)  →  $\\tanh$  →  $\\times\\ \\epsilon$ (= 0.010 rad)", C_PI, 9, "bold")
axR.text(0.80, 0.322, "198", fontsize=10, weight="bold", va="center")
arr(axR, (0.41, 0.41), (0.41, 0.37))

box(axR, 0.06, 0.145, 0.70, 0.085,
    "output  (B, T, 6)   $\\delta a \\in (-10, +10)$ mrad\n"
    "arm joints only — gripper passes through untouched", C_DAT, 8.8)
arr(axR, (0.41, 0.275), (0.41, 0.23))

axR.text(0.06, 0.025,
         "receptive field  1 + 2(1+2+4) = 15 frames = 0.75 s   ·   deploy window WIN = 16\n"
         "causality: left-padding only — architectural, not a penalty\n"
         "output bound: $\\tanh\\times\\epsilon$ — mathematically unreachable, not a soft cost\n"
         "single-frame inference 0.5 ms (budget 50 ms) · operating point "
         "$\\|\\delta a\\|$ = 1.1 mrad = 11% of bound, saturation 0%",
         fontsize=8.1, va="bottom", linespacing=1.5)

fig.suptitle("Model structures  —  knowledge in $\\hat{G}$ (falsifiable), "
             "skill in $\\pi$ (black box with a hard bound)",
             fontsize=13.5, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.945])
fig.savefig(OUT / "arch_spec.png", dpi=155)
print(OUT / "arch_spec.png")
