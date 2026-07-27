#!/usr/bin/env python3
"""E-1 汇报图 — Ĝ 结构示意图(框图 + 三块拟合内脏)。

上: 灰盒信号流(解析=绿 / 日志拟合=蓝 / π接口=橙), 含K=5集成与验收数字。
下: (b)伺服FIR ĉ  (c)模态冲激核(5成员叠加显示不确定度)  (d)位形调度 α(q2,q3)。
输出: outputs/p2_bakeoff/g_structure.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC",
                                   "WenQuanYi Micro Hei", "AR PL UKai CN", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
Z = np.load(ROOT / "outputs/p2_bakeoff/g_ensemble.npz")
DT = 0.05

C_ANA, C_FIT, C_PI = ("#e8f5e9", "#2e7d32"), ("#e3f2fd", "#1565c0"), ("#fff3e0", "#e65100")


def box(ax, x, y, w, h, text, colors, fs=10, ls="-"):
    fc, ec = colors
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                                fc=fc, ec=ec, lw=1.6, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p1, p2, label=None, off=(0, 1), color="0.25", ls="-"):
    ax.annotate("", p2, p1, arrowprops=dict(arrowstyle="-|>", lw=1.7,
                                            color=color, linestyle=ls))
    if label:
        ax.text((p1[0] + p2[0]) / 2 + off[0], (p1[1] + p2[1]) / 2 + off[1],
                label, ha="center", fontsize=9, color="0.2")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def modal_ker(om, ze, K=60):
    n = np.arange(K)[None, :]
    om, ze = om[:, None], ze[:, None]
    omd = om * np.sqrt(np.clip(1 - ze ** 2, 1e-4, None))
    return np.exp(-ze * om * n * DT) * np.sin(omd * n * DT) / omd * DT


fig = plt.figure(figsize=(16, 10.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.16,
                      wspace=0.26, left=0.05, right=0.97, top=0.92, bottom=0.07)

# ============ 上: 框图 ============
ax = fig.add_subplot(gs[0, :])
ax.set_xlim(0, 100); ax.set_ylim(0, 36); ax.axis("off")

# 图例
for i, (lab, (fc, ec)) in enumerate([("解析(零拟合)", C_ANA), ("从日志拟合", C_FIT),
                                     ("π 接口", C_PI)]):
    ax.add_patch(Rectangle((63 + i * 12, 33.2), 2.2, 1.8, fc=fc, ec=ec, lw=1.4))
    ax.text(65.7 + i * 12, 34.1, lab, va="center", fontsize=9)

# 第一行: 输入 → ⊕ → ① → ②
box(ax, 1.5, 26, 16, 6.5, "指令流 a(t)\nACT chunk\n(任意上游策略)", ("white", "0.35"))
ax.add_patch(Circle((21.5, 29.2), 1.3, fc="white", ec=C_PI[1], lw=1.8))
ax.text(21.5, 29.2, "+", ha="center", va="center", fontsize=15, color=C_PI[1])
box(ax, 13, 16.5, 15, 5.5, "δa(t) ≤ 10mrad\nπ 的修正注入口\n(=Ĝ 的扰动通道)", C_PI, fs=9.5)
arrow(ax, (20.5, 22), (21.4, 27.9))
arrow(ax, (17.5, 29.2), (20.2, 29.2))
box(ax, 25, 26, 20, 6.5, "① 激励整形 (解析)\n200Hz 样条 → 精确 q̈(t)\n→ 抗混叠降采 20Hz", C_ANA)
arrow(ax, (22.8, 29.2), (25, 29.2))
box(ax, 49, 26, 21, 6.5, "② 伺服层 (拟合)\nFIR ĉ: 每关节 10 抽头\n(编码器最小二乘)\nq̇̂ = ĉ∗ȧ,   u = ĉ∗q̈", C_FIT, fs=9.5)
arrow(ax, (45, 29.2), (49, 29.2))

# 第二行: ③ 刚体 / ④ 柔性
box(ax, 25, 15, 20, 6.5, "③ 刚体投影 (解析)\nω_rigid = J(q)·q̇̂\nMuJoCo 雅可比", C_ANA)
box(ax, 49, 13.2, 26, 8.3,
    "④ 柔性通道 (拟合)\n3 个阻尼谐振模态\nr̂ = Σₘ Bₘ·[kₘ ⊛ (wₘᵀu)·αₘ(φ)] + D·u\nαₘ(φ)=softplus(θₘᵀφ): 位形调度",
    C_FIT, fs=9.5)
ax.text(62, 11.9, "φ=[q2,q3,二次项,速度,夹爪] 只吃名义轨迹, 对 δa 冻结 → 扰动通道严格线性",
        ha="center", fontsize=8.5, color=C_FIT[1])
arrow(ax, (55, 26), (38, 21.5), "q̇̂", off=(-2, 0.3))
arrow(ax, (60, 26), (62, 21.5), "u", off=(2, 0.3))

# 第三行: Σ → 输出; 拟合目标; 集成徽章
ax.add_patch(Circle((42, 7.5), 1.4, fc="white", ec="0.3", lw=1.6))
ax.text(42, 7.5, "Σ", ha="center", va="center", fontsize=13)
arrow(ax, (35, 15), (41.2, 8.8))
arrow(ax, (58, 13.2), (43.3, 8.2))
box(ax, 46, 4.5, 22, 6, "R 会话标定旋转\n→ ĝ(t): 预测手腕 gyro", ("white", "0.35"))
arrow(ax, (43.4, 7.5), (46, 7.5))
box(ax, 72, 4.5, 26, 6.5,
    "拟合目标: 真机 gyro 残差 (0.5–9Hz)\n接触窗掩零 | 数据 = 已录 20 集日志\n(自然指令变化 = 免费激励, 零专门实验)",
    ("#fafafa", "0.45"), fs=9, ls="--")
arrow(ax, (68, 7.5), (72, 7.5), "最小二乘 + Adam", off=(0, 1.2), ls="--", color="0.45")
box(ax, 1.5, 4.5, 19, 6.5, "K=5 bootstrap 集成\nπ 对练: loss 取 K 成员均值\n评估看最差成员(防钻空子)", C_FIT, fs=9, ls="--")

ax.text(50, 0.8, "验收(留出集): gap-R²=0.44(线性族天花板≈0.44, 已证) · 带号方向余弦 0.76 · 反相 0% · "
                 "扰动线性度 0.0% · 极点 3.68Hz ≈ 实测共振 3.7Hz",
        ha="center", fontsize=10, bbox=dict(fc="#fffde7", ec="#f9a825", lw=1.2))

# ============ 下: 三块拟合内脏 ============
# (b) 伺服 FIR
axb = fig.add_subplot(gs[1, 0])
c_enc = Z["c_enc"][0]                                    # (6,10) 成员0
lags_ms = np.arange(c_enc.shape[1]) * DT * 1000
for j in range(6):
    axb.plot(lags_ms, c_enc[j], "-o", ms=3.5, lw=1.4, label=f"j{j+1}")
axb.axvline(100, color="0.4", ls="--", lw=1)
axb.text(105, axb.get_ylim()[1] * 0.05 + c_enc.max() * 0.82, "≈100ms\n伺服滞后", fontsize=8.5, color="0.3")
axb.set_xlabel("指令滞后 (ms)"); axb.set_ylabel("FIR 抽头权重")
axb.set_title("(b) ② 伺服 FIR ĉ — 滞后+低通的形状", fontsize=10.5)
axb.legend(fontsize=7, ncol=2); axb.grid(alpha=0.3)

# (c) 模态冲激核 (5 成员)
axc = fig.add_subplot(gs[1, 1])
t = np.arange(60) * DT
cols = ["#d32f2f", "#1976d2", "#388e3c"]
for k in range(5):
    om, ze = np.exp(Z["log_om"][k]), sigmoid(Z["raw_ze"][k]) * 0.5
    order = np.argsort(om)
    ker = modal_ker(om[order], ze[order])
    ker = ker / np.abs(ker).max(1, keepdims=True)
    for m in range(3):
        lab = f"f={om[order][m]/2/np.pi:.2f}Hz ζ={ze[order][m]:.2f}" if k == 0 else None
        axc.plot(t, ker[m] + (2 - m) * 2.2, color=cols[m], lw=1.8 if k == 0 else 0.7,
                 alpha=1.0 if k == 0 else 0.3, label=lab)
axc.set_yticks([])
axc.set_xlabel("时间 (s)")
axc.set_title("(c) ④ 模态冲激核 kₘ — 打一拳后怎么震\n(细线=bootstrap 成员 → 集成不确定度)", fontsize=10.5)
axc.legend(fontsize=8, loc="upper right"); axc.grid(alpha=0.3, axis="x")
axc.set_xlim(0, 2.0)

# (d) 位形调度 α(q2,q3)
axd = fig.add_subplot(gs[1, 2])
th, mu, sd = Z["theta"][0], Z["mu"][0], Z["sd"][0]
m_star = int(np.argmax(np.linalg.norm(th[:, 1:], axis=1)))
q2 = np.linspace(mu[1] - 2 * sd[1], mu[1] + 2 * sd[1], 80)
q3 = np.linspace(mu[2] - 2 * sd[2], mu[2] + 2 * sd[2], 80)
Q2, Q3 = np.meshgrid(q2, q3)
raw = np.stack([np.ones_like(Q2), Q2, Q3, Q2 ** 2, Q3 ** 2, Q2 * Q3,
                np.full_like(Q2, mu[6]), np.full_like(Q2, mu[7])], -1)
phi = (raw - mu) / sd
A = np.log1p(np.exp(np.clip(phi @ th[m_star], -30, 30)))
im = axd.pcolormesh(Q2, Q3, A, cmap="viridis", shading="auto")
fig.colorbar(im, ax=axd, label="αₘ (增益)")
om0 = np.exp(Z["log_om"][0])
axd.set_xlabel("q2 肩关节 (rad)"); axd.set_ylabel("q3 肘关节 (rad)")
axd.set_title(f"(d) ④ 位形调度 α(q2,q3) @ {om0[m_star]/2/np.pi:.1f}Hz 模态\n"
              "同一拳, 不同姿态震感不同(残差画像 0.32↔0.74 的机制)", fontsize=10.5)

fig.suptitle("Ĝ 结构: 灰盒「指令 → 手腕 IMU」响应模型 — 三块积木, 只有蓝色部分从日志拟合",
             fontsize=13.5, y=0.975)
out = ROOT / "outputs/p2_bakeoff/g_structure.png"
fig.savefig(out, dpi=140)
print(f"→ {out}")
