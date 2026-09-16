#!/usr/bin/env python3
"""π 网络结构图（论文风格）→ `outputs/p2_pi/fig_pi_arch.{png,pdf}`

设计取向
--------
- **颜色编码功能类别**（数据 / 可学习层 / 固定算子 / 网络外的安全约束），不做装饰性配色；
  取色盲友好的 Okabe–Ito 系。
- 横向数据流；框内只写层类型，**层间标出张量维度**；说明进图注，不进框。
- 带图例，使颜色的含义自明。

⚠ matplotlib **不解析 markdown** —— 正文里写星号强调会被原样画出来。

用法:
    python -u sim/fig_pi_arch.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 9,
})

OUT = Path(__file__).resolve().parents[1] / "outputs/p2_pi"

# Okabe–Ito（色盲友好）；每类一个边色 + 对应浅底
C = {
    "data":  ("#333333", "#e9ecef"),   # 数据张量
    "learn": ("#0072B2", "#d6e6f2"),   # 可学习层
    "fixed": ("#009E73", "#d6efe6"),   # 固定算子（无参数）
    "out":   ("#E69F00", "#fbecd2"),   # 输出
    "safe":  ("#D55E00", "#fbe3d6"),   # 网络外的安全约束
}
GRY = "#6c757d"
YC, H, G = .50, .28, .019                  # 主行底边 / 高度 / 常规间隙


def stack(ax, x, y, w, h, top, bot, kind, n=3, off=.0075):
    """重复层收成「叠层」块 —— 4 个只差一个数字的方框是视觉冗余。"""
    ec, fc = C[kind]
    for i in range(n - 1, 0, -1):
        ax.add_patch(Rectangle((x + i * off, y + i * off * 2.2), w, h,
                               fc=fc, ec=ec, lw=0.9, alpha=.55))
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2 + .034, top, ha="center", va="center", fontsize=8.4)
    ax.text(x + w / 2, y + h / 2 - .034, bot, ha="center", va="center",
            fontsize=7.8, color=ec)
    return x + w + (n - 1) * off


def blk(ax, x, y, w, h, top, bot, kind, ls="-"):
    ec, fc = C[kind]
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.2, ls=ls))
    ax.text(x + w / 2, y + h / 2 + (.030 if bot else 0), top,
            ha="center", va="center", fontsize=8.4)
    if bot:
        ax.text(x + w / 2, y + h / 2 - .032, bot, ha="center", va="center",
                fontsize=7.8, color=ec)
    return x + w


def shape(ax, x, y, txt):
    ax.text(x, y - .038, txt, ha="center", va="top", fontsize=7.6, color=GRY)


def arrow(ax, x0, x1, y, txt=""):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=9, lw=0.9, color="#333333",
                                 shrinkA=0, shrinkB=0))
    if txt:
        # ⚠ 标注必须放到**框行之上**；放在箭头旁会落进相邻方框里
        ax.text((x0 + x1) / 2, YC + H + .030, txt, ha="center", va="bottom",
                fontsize=7.4, color=GRY)


fig, ax = plt.subplots(figsize=(9.4, 3.5))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
cur = .018

x = blk(ax, cur, YC, .082, H, "ACT 指令", "", "data")
shape(ax, cur + .041, YC, "6 × 100"); cur = x
arrow(ax, cur, cur + G, YC + H / 2); cur += G

xs0 = cur
x = stack(ax, cur, YC, .215, H, "空洞卷积 × 4     k = 5,  64 ch",
          "dilation = 1, 2, 4, 8", "learn")
xs1 = x
cur = x
arrow(ax, cur, cur + G, YC + H / 2); cur += G

ax.annotate("", xy=(xs0, YC - .098), xytext=(xs1, YC - .098),
            arrowprops=dict(arrowstyle="|-|", lw=0.8, color=GRY, shrinkA=0, shrinkB=0))
ax.text((xs0 + xs1) / 2, YC - .132,
        "感受野 61 帧（3.05 s）· 对称填充，chunk 内双向 · 无残差",
        ha="center", va="top", fontsize=7.8, color=GRY)
ax.text((xs0 + xs1) / 2, YC - .190, "（网络实际输入另含上一 chunk 与一、二阶差分）",
        ha="center", va="top", fontsize=7.2, color=GRY, style="italic")

x = blk(ax, cur, YC, .078, H, "Conv1d  1×1", "→ 6 通道", "learn")
cur = x
arrow(ax, cur, cur + G, YC + H / 2); cur += G

x = blk(ax, cur, YC, .074, H, "tanh × ε", "ε = 10 mrad", "fixed")
cur = x
arrow(ax, cur, cur + G, YC + H / 2); cur += G

x = blk(ax, cur, YC, .066, H, "δa", "", "out")
shape(ax, cur + .033, YC, "100 × 6"); cur = x

# ── 图例 ──────────────────────────────────────────────
lx, ly, lw, lh = .560, .150, .026, .030
for i, (k, t) in enumerate([("learn", "可学习层（共 68,806 参数）"),
                            ("fixed", "固定算子（无参数）")]):
    ec, fc = C[k]
    xx = lx + i * .172
    ax.add_patch(Rectangle((xx, ly), lw, lh, fc=fc, ec=ec, lw=1.0))
    ax.text(xx + lw + .008, ly + lh / 2, t, ha="left", va="center",
            fontsize=7.4, color="#333333")

fig.text(.010, .020,
         "图 1  补正网络 π 的结构。整段 chunk 的指令在其起始时刻即已全部已知，"
         "故 chunk 内的双向卷积不违反因果性（非 TCN: 对称填充、无残差）。\n"
         "         幅值上界 ε = 10 mrad 由 tanh 结构性保证。"
         "单次前向 0.83 ms，每 chunk 只调用一次，因而补正自 chunk 首帧即可施加。",
         ha="left", va="bottom", fontsize=7.9, color="#333333", linespacing=1.55)

fig.tight_layout(rect=[0, 0.085, 1, 1])
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig_pi_arch.png", dpi=220)
fig.savefig(OUT / "fig_pi_arch.pdf")
print(f"→ {OUT / 'fig_pi_arch.png'}  (+ .pdf)")
