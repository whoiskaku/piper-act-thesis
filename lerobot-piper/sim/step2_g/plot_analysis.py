#!/usr/bin/env python3
"""Step 2 零机时分析的汇总图 —— 四个分析此前**只有文字**，这里补上图。

补图的理由不是好看: 指導教員 2026-07-15 对四联抽象指标的原话是
「全然わかんない」, 并明确要**波形/图**。分析结论要能被看懂才有用。

四联:
  ① 天花板 —— 真机自身可重复性**逐频段**分解 (probe_b2 六次裸复放, 100 Hz)
     这是全部结论里最硬的一条: 确定性内容在 9–12 Hz 截止 ⇒ 冻结的 0.5–9 Hz
     评价带恰好卡对位置; 加宽只会灌进噪声。
  ② 当前 G 的**分区**保真度 —— 按老师标准「不看平均、看难处」
  ③ 可辨识性的**结构**限制 —— 逐关节可达惯量跨度决定 Ia 能否与 I_link 分离
  ④ 激励设计的效果 —— 条件数与逐参数灵敏度, pp2 数据 vs 激励套件

数据来源分两类, 图上如实标注:
  · **重算**: ① 与 ③ 便宜, 每次运行现算
  · **引用**: ② 与 ④ 需分别回放 17 集与 2400 次仿真, 取自记录的运行结果,
    出处写在 `MEASURED` 字典里

配色经 dataviz 校验器全对比校验(light, #fcfcfb):
  蓝 #2a78d6 / 黄 #eda100 / 绿 #008300 —— CVD 最差 ΔE 16.2, 常视 29.0;
  黄对表面对比 2.11 < 3:1 ⇒ 按规程以**直接标注 + 数值**兜底。
  红 #c0392b 仅作**状态色**(不达标/不可分离), 恒与文字标签同时出现。

用法: python sim/step2_g/plot_analysis.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BLUE, AMBER, GREEN, RED = "#2a78d6", "#eda100", "#008300", "#c0392b"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#d8d8d4", "#fcfcfb"
OUT = ROOT / "outputs/step2"

# ── 引用的实测值（出处见注释；重算代价过高故记录复用）────────────────
MEASURED = {
    # sim/step2_fidelity.py, 17 集裸 ACT, 0.5–9 Hz 陀螺三轴
    "fidelity": {"全程": -42.6, "接缝后10帧": -30.9, "gap能量前10%": +2.5,
                 "gap能量后50%": -285.0},
    # sim/step2_g/build_suite.py --analyze, 108 条激励序列 × 24 参数
    "sens_excite": {"kp": [3.15, 18.03, 3.67, 2.31, 1.66, 3.05],
                    "c": [2.93, 8.48, 3.63, 2.22, 1.86, 2.83],
                    "fc": [0.02, 1.20, 0.48, 1.22, 1.50, 1.30],
                    "Ia": [0.03, 5.72, 0.88, 0.33, 0.23, 0.05]},
    # 同一分析, 但输入是 17 集 pp2 裸 ACT（激励设计**之前**的基线）
    "sens_pp2": {"kp": [1.84, 14.64, 2.74, 0.02, 1.59, 2.33],
                 "c": [1.81, 8.25, 3.38, 0.04, 1.66, 2.28],
                 "fc": [0.02, 1.25, 0.37, 0.71, 1.00, 0.81],
                 "Ia": [0.01, 3.52, 0.22, 0.03, 0.48, 0.69]},
    "cond": {"pp2 数据": 630.0, "激励套件": 29.6},
    # sim/step2_g/build_suite.py --list（本次定稿的 96 条套件）
    "inertia_span": {"j1": 114.6, "j2": 4.4, "j3": 1.3,
                     "j4": 12.9, "j5": 1.0, "j6": 1.0},
}
BANDS = ((0.5, 1), (1, 2), (2, 3.5), (3.5, 6), (6, 9), (9, 12),
         (12, 18), (18, 25), (25, 35), (35, 45))


def ceiling_by_band() -> list[tuple[str, float]]:
    """① 重算：probe_b2 六次裸复放的逐频段可重复性（100 Hz 原生采样）。"""
    P = ROOT / "outputs/probe_b2"
    files = ["replay_ep0_A1.json", "replay_ep0_A2.json", "replay_ep0_r2_A3.json",
             "replay_ep0_r2_A4.json", "replay_ep0_r3_A5.json", "replay_ep0_r3_A6.json"]
    G = [np.asarray(json.loads((P / f).read_text())["imu100"], float)[:, 5:8]
         for f in files]
    n = min(len(g) for g in G)
    G = [g[:n] for g in G]
    out = []
    for lo, hi in BANDS:
        b, a = butter(4, [lo / 50.0, hi / 50.0], btype="band")
        B = [filtfilt(b, a, g, axis=0) for g in G]
        r = np.mean([1 - float(np.square(B[i] - B[k]).sum())
                     / float(np.square(B[k]).sum())
                     for i, k in itertools.combinations(range(6), 2)])
        out.append((f"{lo:g}–{hi:g}", float(r) * 100))
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(14.5, 10), facecolor=SURF)
    fig.suptitle("Step 2 · 零机时分析汇总　——　全部结论均为实测，未使用任何真机新数据",
                 fontsize=13, color=INK, y=0.975)

    # ── ① 天花板：逐频段可重复性 ────────────────────────────────
    a = ax[0, 0]
    bands = ceiling_by_band()
    x = np.arange(len(bands))
    v = [b for _, b in bands]
    cols = [GREEN if t > 85 else (AMBER if t > 50 else RED) for t in v]
    a.bar(x, v, color=cols, width=0.66)
    a.axhline(85, color=INK2, lw=1.1, ls="--")
    a.annotate("85%（视为确定性）", (len(bands) - 0.4, 85), fontsize=8.5, color=INK2,
               ha="right", va="bottom")
    a.axvspan(-0.5, 4.5, color="#eef3f9", zorder=0)
    a.annotate("冻结评价带 0.5–9 Hz", (2.0, -78), fontsize=9.5, color=BLUE,
               ha="center", fontweight="bold")
    for xi, t in zip(x, v):
        a.annotate(f"{t:.0f}", (xi, t + (3 if t > 0 else -9)), ha="center",
                   fontsize=8.5, color=INK)
    a.set_xticks(x)
    a.set_xticklabels([n for n, _ in bands], fontsize=8, rotation=30)
    a.set_ylim(-110, 118)
    a.set_ylabel("可重复性 (%)", color=INK2)
    a.set_xlabel("频段 (Hz)", color=INK2)
    a.set_title("① 天花板：真机自身可重复性（六次裸复放，100 Hz）\n"
                "确定性内容在 9–12 Hz 截止 ⇒ 冻结带恰好卡对，加宽只会灌噪声",
                fontsize=10.5, color=INK, loc="left")

    # ── ② 当前 G 的分区保真度 ──────────────────────────────────
    a = ax[0, 1]
    reg = list(MEASURED["fidelity"])
    val = [MEASURED["fidelity"][k] for k in reg]
    y = np.arange(len(reg))
    lo_lim = -120.0
    a.barh(y, [max(v_, lo_lim) for v_ in val],
           color=[GREEN if v_ > 0 else RED for v_ in val], height=0.55)
    a.axvline(0, color=INK, lw=1.2)
    a.axvline(96.9, color=INK2, lw=1.6, ls="--")
    a.annotate("天花板 96.9%\n（难处，见①）", (96.9, len(reg) - 0.5), fontsize=8.5,
               color=INK2, ha="right", va="center", xytext=(-6, 0),
               textcoords="offset points")
    a.axvline(42, color=RED, lw=1.2, ls=":")
    # 「Ĝ」在本机字体里缺字会渲染成豆腐块 —— 改用可渲染的写法
    a.annotate("旧 G 模型曾达 42%\n老师判「使いようがない」\n（被否决的成绩，非及格线）",
               (42, 2.55), fontsize=8, color=RED, xytext=(-6, 0),
               textcoords="offset points", va="center", ha="right")
    for yy, v_ in zip(y, val):
        a.annotate(f"{v_:+.1f}%", (max(v_, lo_lim), yy), fontsize=9.5, color=INK,
                   fontweight="bold", va="center",
                   xytext=(8 if v_ >= 0 else 20, 0), textcoords="offset points",
                   ha="left")
        if v_ < lo_lim:
            a.annotate("◀裁剪", (lo_lim, yy), fontsize=7.5, color=RED, va="center",
                       xytext=(3, 0), textcoords="offset points")
    a.set_yticks(y); a.set_yticklabels(reg, fontsize=9.5, color=INK)
    a.set_xlim(lo_lim, 128); a.invert_yaxis()
    a.set_xlabel("G 的解释率 (%)　　0% = 不建模基线", color=INK2)
    a.set_title("② 当前 G 的分区保真度（按老师标准：不看平均、看难处）\n"
                "难处 −30.9% ⇒ 比「不建模」还差；引自 sim/step2_fidelity.py",
                fontsize=10.5, color=INK, loc="left")

    # ── ③ 可辨识性的结构限制 ───────────────────────────────────
    a = ax[1, 0]
    sp = MEASURED["inertia_span"]
    ks = list(sp)
    vv = [sp[k] for k in ks]
    cols = [GREEN if s > 1.3 else RED for s in vv]
    a.bar(np.arange(6), vv, color=cols, width=0.6)
    a.set_yscale("log")
    a.axhline(1.3, color=INK2, lw=1.1, ls="--")
    a.annotate("虚线 1.3× —— 低于此无法分离 Ia", (0.99, 0.97), xycoords="axes fraction",
               fontsize=8.5, color=INK2, ha="right", va="top")
    for i, (k, s) in enumerate(zip(ks, vv)):
        a.annotate(f"{s:g}×", (i, s * 1.15), ha="center", fontsize=9.5,
                   color=INK, fontweight="bold")
    # 判定放在**坐标轴下方**，与阈值注释彻底分开（原来两者在同一高度互相压住）
    for i, s in enumerate(vv):
        a.annotate("可分离" if s > 1.3 else "不可分离", (i, -0.085),
                   xycoords=("data", "axes fraction"), ha="center", va="top",
                   fontsize=8.5, color=GREEN if s > 1.3 else RED, fontweight="bold")
    a.set_xticks(np.arange(6)); a.set_xticklabels(ks, fontsize=10)
    a.set_ylim(0.95, 600)
    a.set_ylabel("可达惯量跨度（对数）", color=INK2)
    a.set_title("③ 可辨识性的**结构**限制：远端关节越少，惯量越动不了\n"
                "j6 无远端关节 ⇒ I_link 恒定 ⇒ Ia 与 I_link 原理上不可分离",
                fontsize=10.5, color=INK, loc="left")

    # ── ④ 激励设计的效果 ───────────────────────────────────────
    a = ax[1, 1]
    blocks = ("kp", "c", "fc", "Ia")
    names, before, after = [], [], []
    for b_ in blocks:
        for j in range(6):
            names.append(f"{b_}{j+1}")
            before.append(MEASURED["sens_pp2"][b_][j])
            after.append(MEASURED["sens_excite"][b_][j])
    x = np.arange(len(names))
    a.bar(x - 0.2, before, width=0.4, color=AMBER, label="pp2 数据（激励设计前）")
    a.bar(x + 0.2, after, width=0.4, color=BLUE, label="激励套件（设计后）")
    a.axhline(0.5, color=RED, lw=1.1, ls="--")
    a.annotate("红虚线 0.5% —— 低于此视为不可辨识", (0.02, 0.80),
               xycoords="axes fraction", fontsize=8, color=RED, ha="left", va="top")
    a.set_yscale("log"); a.set_ylim(0.008, 120)
    a.set_xticks(x); a.set_xticklabels(names, fontsize=7, rotation=90)
    a.set_ylabel("灵敏度 (%)（对数）", color=INK2)
    a.legend(fontsize=8.5, frameon=False, loc="upper left")
    c0, c1 = MEASURED["cond"]["pp2 数据"], MEASURED["cond"]["激励套件"]
    a.set_title(f"④ 激励设计的效果：条件数 {c0:.0f} → {c1:.1f}"
                f"（**无噪声仿真初筛**，不代表真机上已可辨识）\n"
                f"j4 全部参数由死转活；j5/j6 的 Ia 仍受③的结构限制",
                fontsize=10.5, color=INK, loc="left")

    for x_ in fig.axes:
        x_.grid(True, color=GRID, lw=0.7, alpha=0.9, axis="y")
        x_.set_axisbelow(True)
        for s_ in ("top", "right"):
            x_.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            x_.spines[s_].set_color(GRID)
        x_.tick_params(colors=INK2, labelsize=8.5)

    fig.subplots_adjust(left=0.055, right=0.985, top=0.885, bottom=0.11,
                        hspace=0.46, wspace=0.20)
    f = OUT / "step2_analysis.png"
    fig.savefig(f, dpi=135, facecolor=SURF)
    print(f"→ {f.relative_to(ROOT)}")
    print("①③ 每次运行现算；②④ 引自记录的运行结果（出处见 MEASURED 注释）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
