#!/usr/bin/env python3
"""π 网络结构图（两联）→ `outputs/p2_pi/fig_pi_arch.png`

左：部署时的结构（三个 checkpoint 共用这一套架构）
右：三种训练目标的差别 —— 架构相同、结果天差地别，这才是要点

⚠ matplotlib **不解析 markdown**：正文里不要写 `**…**`，会原样画出来。
   强调改用 `weight="bold"` 或整框着色。

配色与画法沿用 `sim/fig_arch_spec.py`。

用法:
    python -u sim/fig_pi_arch.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parents[1] / "outputs/p2_pi"
C_IN, C_CONV, C_HEAD, C_OUT = "#dbeafe", "#fce7f3", "#fed7aa", "#dcfce7"
C_BAD, C_WARN, C_FRZ = "#fee2e2", "#fef9c3", "#e5e7eb"
EC = "#374151"


def box(ax, x, y, w, h, txt, fc, fs=8.6, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008",
                                fc=fc, ec=EC, lw=1.1))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
            fontsize=fs, weight=weight, linespacing=1.45)


def arr(ax, y0, y1, txt="", x=.50, fs=7.4):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.25, color=EC,
                                 shrinkA=1, shrinkB=1))
    if txt:
        ax.text(x + .015, (y0 + y1) / 2, txt, ha="left", va="center", fontsize=fs)


fig, (axL, axR) = plt.subplots(1, 2, figsize=(16.4, 9.2))
for ax in (axL, axR):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# ══════════════════ 左：部署结构 ══════════════════
axL.set_title("π 网络结构（部署时）—— pi-g2env-v1 / pi-distill-v2 共用同一架构",
              fontsize=12, weight="bold", pad=12)

box(axL, .04, .900, .92, .068,
    "ACT 策略输出   chunk = 100 帧 × 7（6 关节 + 夹爪）\n"
    "整个 chunk 的指令在 t0 就已全部已知", C_FRZ, 9.0)

arr(axL, .900, .878)
box(axL, .04, .772, .92, .106,
    "输入张量   21 通道 × 200 帧\n"
    "[ 上一 chunk 100 帧 ; 本 chunk 100 帧 ]  ×  [ rel(7) , 一阶差分(7) , 二阶差分(7) ]\n"
    "rel = 指令 − 本 chunk 首帧（去掉绝对位姿偏置）", C_IN, 8.4)

arr(axL, .772, .750, x=.30)
for i, (d, cin, npar) in enumerate([(1, 21, "6,784"), (2, 64, "20,544"),
                                    (4, 64, "20,544"), (8, 64, "20,544")]):
    y = .690 - i * .060
    box(axL, .09, y, .82, .048,
        f"Conv1d({cin} → 64,  kernel = 5,  dilation = {d})  +  GELU"
        f"          {npar} 参数", C_CONV, 8.3)

# 箭头走左侧，感受野注记放右侧，避免叠字
arr(axL, .510, .472, x=.30)
axL.text(.35, .491, "感受野 = 1 + 4×(1+2+4+8) = 61 帧 = 3.05 s",
         ha="left", va="center", fontsize=8.8, style="italic")
box(axL, .09, .412, .82, .058,
    "1×1 卷积（64 → 6）   390 参数     →     取末 100 帧", C_HEAD, 8.8)

arr(axL, .412, .392, x=.30)
box(axL, .09, .332, .82, .058,
    "tanh × ε = 10 mrad   —— 安全界做成结构性的，不是事后限幅", C_HEAD, 8.8)

arr(axL, .332, .312, x=.30)
box(axL, .09, .248, .82, .062,
    "输出   δa[0..99] × 6 关节   —— 一次前向出整个 chunk", C_OUT, 9.6, "bold")

arr(axL, .248, .228, x=.30)
box(axL, .09, .166, .82, .060,
    "网络外面：接触门控（抓取瞬间归零）  +  ε 硬限幅兜底", C_FRZ, 8.6)

arr(axL, .166, .146, x=.30)
box(axL, .04, .022, .92, .122,
    "68,806 参数    ·    CPU 单次前向 0.83 ms（真机 p95 4 ms，每 100 帧才跑一次）\n"
    "运行时：不跑 MuJoCo · 无 G-2 · 无 H · 无求解器 · 不读 IMU\n"
    "⇒ 计算空档 0 帧（真机实测 0.00%；对照：在线跑 G-2 的方案每 chunk 空 3–4 帧）",
    C_OUT, 8.8, "bold")

# ══════════════════ 右：三种训练目标 ══════════════════
axR.set_title("架构相同，训练目标不同 —— 差别全在这里",
              fontsize=12, weight="bold", pad=12)

box(axR, .03, .912, .94, .056,
    "三个 checkpoint 用的是同一套左边的网络；下面只列训练时的损失与梯度来源",
    C_FRZ, 8.8)

arr(axR, .912, .890)
box(axR, .03, .700, .94, .190,
    "①  pi-g2env-v1  ——  在冻结的 G-2 环境里学\n\n"
    "L = ‖ ω_G2(u + δa) − ω_ideal ‖²\n"
    "G-2 含 MuJoCo 不可微 ⇒ 用它自己的雅可比 J 线性化\n"
    "仿真内 +15.8%（真前向、留出集）        真机 +25.0%   → 显著变差\n"
    "成因：G-2 对 δa 的响应与真机夹角 83°（cos 0.12）、幅度 3.6×", C_BAD, 8.5)

arr(axR, .700, .672, "换掉不可信的梯度来源")
box(axR, .03, .472, .94, .200,
    "②  中间尝试（未上机）  ——  把响应模型换成实测 H\n\n"
    "L = ‖ ê0 + H * δa ‖²    （G-2 只负责预测墙，响应来自真机测量）\n"
    "α = 1.97  ·  δa RMS 3.78 mrad = 教师的 5.4×   ⇒   过量驱动\n"
    "解析解带岭正则、用药保守，π 去钻这个空子\n"
    "⇒ 落进 ① 已用真机证伪的大幅值区间，判否", C_WARN, 8.5)

arr(axR, .472, .444, "改为复现已验证的控制器")
box(axR, .03, .244, .94, .200,
    "③  pi-distill-v2  ——  直接回归教师 δa*（现行）\n\n"
    "L = ‖ δa − δa* ‖²        δa* = 冻结「G-2 预测墙 + H 解析求解」，零空档\n"
    "α = 0.90  ·  δa RMS 0.66 mrad（教师 0.70）⇒ 留在 H 被验证的区间\n"
    "离线预测 +11.23%        真机 −3.8%（p=0.30，每臂 15 集欠功率）⚠",
    C_OUT, 8.5)

box(axR, .03, .022, .94, .196,
    "要点\n\n"
    "· 三者网络完全一样 ⇒ 差别不是架构、不是容量，是训练信号\n"
    "· ① 证明 G 的增量通道（对 δa 的反事实响应）不可用\n"
    "· ② 证明 优化模型目标会外推出界 —— 而那个界是 ① 用真机划出来的\n"
    "· ③ 是目前唯一还开着的用法：π 的价值不在提供新知识，在消除延迟\n"
    "  （在线跑 G-2 的方案被空档砍掉一半：15.50% → 真机 −8.0%）",
    C_IN, 8.7)

fig.suptitle("π 网络：结构与三种训练目标（2026-09-16）", fontsize=13.5, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.962])
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig_pi_arch.png", dpi=170)
print(f"→ {OUT / 'fig_pi_arch.png'}")
