#!/usr/bin/env python3
"""π 网络结构图（论文风格，单联）→ `outputs/p2_pi/fig_pi_arch.{png,pdf}`

设计取向
--------
横向数据流；框内**只标层类型与张量维度**，不写说明文字；
配色收敛到一个灰阶 + 一个强调色；说明放图注，不放框里。
（三种训练目标的对比属于正文表格，不进结构图。）

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
ACC, GRY, LGY, EC = "#2f6f9f", "#8c8c8c", "#e8eaed", "#333333"


def tensor(ax, x, y, w, h, label, shape, fc=LGY):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=EC, lw=0.9))
    ax.text(x + w / 2, y + h + .025, label, ha="center", va="bottom", fontsize=8.5)
    ax.text(x + w / 2, y - .030, shape, ha="center", va="top", fontsize=8, color=GRY)


def layer(ax, x, y, w, h, top, bot):
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=ACC, lw=1.1))
    ax.text(x + w / 2, y + h / 2 + .034, top, ha="center", va="center", fontsize=8.2)
    ax.text(x + w / 2, y + h / 2 - .034, bot, ha="center", va="center",
            fontsize=7.8, color=ACC)


def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=9, lw=0.9, color=EC,
                                 shrinkA=0, shrinkB=0))


fig, ax = plt.subplots(figsize=(11.2, 3.8))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

YC, H = .46, .30

tensor(ax, .020, YC, .072, H, "指令序列", "21 × 200")
arrow(ax, .092, .118, YC + H / 2)

x0, wc, gap = .118, .104, .020
for i, d in enumerate((1, 2, 4, 8)):
    x = x0 + i * (wc + gap)
    layer(ax, x, YC, wc, H, "Conv1d  k=5", f"dilation {d}  ·  64ch")
    arrow(ax, x + wc, x + wc + gap, YC + H / 2)
xe = x0 + 4 * (wc + gap)

ax.annotate("", xy=(x0, YC - .075), xytext=(xe - gap, YC - .075),
            arrowprops=dict(arrowstyle="|-|", lw=0.8, color=GRY,
                            shrinkA=0, shrinkB=0))
ax.text((x0 + xe - gap) / 2, YC - .112, "感受野 61 帧（3.05 s）",
        ha="center", va="top", fontsize=8, color=GRY)

layer(ax, xe, YC, .082, H, "Conv1d  1×1", "64 → 6")
arrow(ax, xe + .082, xe + .102, YC + H / 2)
layer(ax, xe + .102, YC, .082, H, "tanh × ε", "ε = 10 mrad")
arrow(ax, xe + .184, xe + .204, YC + H / 2)
tensor(ax, xe + .204, YC, .072, H, "补正量 δa", "100 × 6", fc="#dfe9f2")

# 输入构成（左下细条）
yb, hb = .135, .050
ax.add_patch(Rectangle((.020, yb), .036, hb, fc="white", ec=GRY, lw=0.8))
ax.add_patch(Rectangle((.056, yb), .036, hb, fc=LGY, ec=GRY, lw=0.8))
# 两个标签靠外缘对齐, 否则在这个宽度下会叠字
ax.text(.020, yb - .028, "上一段", ha="left", va="top", fontsize=7.4, color=GRY)
ax.text(.092, yb - .028, "本段", ha="right", va="top", fontsize=7.4, color=GRY)
ax.plot([.056, .056], [yb + hb + .012, YC - .012], ls=":", lw=0.7, color=GRY)
ax.text(.104, yb + hb / 2,
        "每帧 7 维（6 关节 + 夹爪）→ [ 去偏置指令, 一阶差分, 二阶差分 ] = 21 通道",
        ha="left", va="center", fontsize=8, color=GRY)

fig.text(.012, .018,
         "图 1  补正网络 π 的结构。整段 chunk 的指令在其起始时刻即已全部已知，"
         "故 chunk 内的双向卷积不违反因果性。ε 由 tanh 结构性保证，\n"
         "         接触门控与硬限幅置于网络之外。共 68,806 个参数，"
         "单次前向 0.83 ms，每 chunk 只调用一次。",
         ha="left", va="bottom", fontsize=8, color=EC, linespacing=1.5)

fig.tight_layout(rect=[0, 0.09, 1, 1])
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig_pi_arch.png", dpi=220)
fig.savefig(OUT / "fig_pi_arch.pdf")
print(f"→ {OUT / 'fig_pi_arch.png'}  (+ .pdf)")
