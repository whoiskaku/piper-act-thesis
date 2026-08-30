#!/usr/bin/env python3
"""**面谈用三张核心素材**（日文，投影可读）。

上一次面谈 32 分钟里约 20 分钟耗在「G のインプットとアウトプットは何ですか」，
且投影上的波形图老师说「見えない」。所以这三张的设计目标只有两个：
**大字号**、**一眼看懂输入/输出/参数的分工**。

1. `fig1_G_definition.png` —— G 定义图（校正フェーズ / 使用フェーズ）
2. `fig2_param_table.png`  —— 参数定义表（字段/作用/现值/范围/gauge）
3. `fig3_result_credibility.png` —— **结果与可信度**：不只给一个「最优点」，
   同时给近优解的范围，避免再次造成「这些就是实际物理参数」的误解。

用法:
    python sim/step2_g/fig_advisor_G.py            # 全部
    python sim/step2_g/fig_advisor_G.py --only 1
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g"):
    sys.path.insert(0, str(_p))

OUT = ROOT / "outputs/step2_g/advisor"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Serif CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

INK, INK2 = "#1d1d2e", "#5a5a72"
BLUE, WARM, GREY, RED = "#2a6f97", "#e07a5f", "#8d99ae", "#c1121f"


def _box(ax, x, y, w, h, text, fc, ec, fs=17, bold=False, tc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=2.2, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc or INK, zorder=3,
            fontweight="bold" if bold else "normal", linespacing=1.55)


def _arrow(ax, x1, y1, x2, y2, col=INK2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=26, lw=2.6, color=col, zorder=4))


def fig1() -> Path:
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.955, "G モデルとは何か", ha="center", fontsize=30,
            fontweight="bold", color=INK)

    ax.text(0.035, 0.845, "① 校正フェーズ（1 回だけ）", fontsize=21,
            fontweight="bold", color=BLUE)
    _box(ax, 0.04, 0.655, 0.27, 0.135,
         "実機データ 90 本\n（励起スイート）", "#eaf2f8", BLUE)
    _box(ax, 0.375, 0.655, 0.235, 0.135, "Optuna 探索", "#ffffff", BLUE, bold=True)
    _box(ax, 0.675, 0.655, 0.285, 0.135,
         "θ*\n（G の内部パラメータ 24 個）", "#eaf2f8", BLUE)
    _arrow(ax, 0.315, 0.7225, 0.371, 0.7225, BLUE)
    _arrow(ax, 0.614, 0.7225, 0.671, 0.7225, BLUE)

    ax.text(0.035, 0.545, "② 使用フェーズ（毎回）", fontsize=21,
            fontweight="bold", color=WARM)
    _box(ax, 0.04, 0.345, 0.27, 0.145, "入力\nACT の関節指令列", "#fdf0ec", WARM, bold=True)
    _box(ax, 0.375, 0.325, 0.235, 0.185,
         "G$_{θ*}$\nPiper の\nMuJoCo モデル", "#ffffff", WARM, fs=18, bold=True)
    _box(ax, 0.675, 0.345, 0.285, 0.145,
         "出力\n関節軌道 ＋ 手首 IMU 応答", "#fdf0ec", WARM, bold=True)
    _arrow(ax, 0.315, 0.4175, 0.371, 0.4175, WARM)
    _arrow(ax, 0.614, 0.4175, 0.671, 0.4175, WARM)
    _arrow(ax, 0.8175, 0.651, 0.8175, 0.516, GREY)
    ax.text(0.828, 0.583, "θ* を代入", fontsize=15, color=GREY, va="center")

    for i, t in enumerate([
        "24 個の数値は G の**内部パラメータ**。入力でも出力でもない。",
        "Optuna は校正時に使う**探索手段**であり、G そのものではない。",
        "G は**ニューラルネットワークではない**。物理シミュレータのパラメータ同定。",
    ]):
        ax.text(0.055, 0.225 - i * 0.062, "・" + t.replace("**", ""),
                fontsize=17.5, color=INK)
    ax.add_patch(FancyBboxPatch((0.035, 0.035), 0.93, 0.225,
                                boxstyle="round,pad=0.008,rounding_size=0.015",
                                fc="none", ec=GREY, lw=1.6, ls="--", zorder=1))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "fig1_G_definition.png"
    fig.savefig(p, dpi=170, facecolor="white"); plt.close(fig)
    return p


def fig2() -> Path:
    rows = [
        ["kp", "actuator_gainprm", "位置サーボ剛性\n（有効追従ゲイン）",
         "500 / 200 / 200\n100 / 50 / 50", "XML × [0.2, 20]"],
        ["c\n(damping)", "dof_damping", "粘性\nkv と厳密に縮退 → c に統合",
         "500 / 5 / 20\n5 / 5 / 5", "XML × [0.05, 20]"],
        ["Ia\n(armature)", "dof_armature", "減速機の換算慣性",
         "すべて 0\n（未同定）", "[1e-4, 1] kg·m²"],
        ["fc\n(frictionloss)", "dof_frictionloss", "静摩擦",
         "すべて 0\n（未同定）", "[1e-3, 5] N·m"],
    ]
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.955, "G の内部パラメータ（6 関節 × 4 分類 = 24 個）",
            ha="center", fontsize=27, fontweight="bold", color=INK)
    cx = [0.035, 0.165, 0.365, 0.615, 0.762]
    cw = [0.128, 0.198, 0.248, 0.145, 0.203]
    hdr = ["分類", "MuJoCo フィールド", "物理的な役割", "現在の XML 値", "探索範囲"]
    y = 0.795
    for x, w, h in zip(cx, cw, hdr):
        ax.add_patch(FancyBboxPatch((x, y), w, 0.075, boxstyle="square,pad=0",
                                    fc="#eef1f5", ec=GREY, lw=1.3))
        ax.text(x + w / 2, y + 0.0375, h, ha="center", va="center",
                fontsize=16, fontweight="bold", color=INK)
    for i, r in enumerate(rows):
        yy = y - (i + 1) * 0.135
        新 = i >= 2
        for x, w, t in zip(cx, cw, r):
            ax.add_patch(FancyBboxPatch((x, yy), w, 0.135, boxstyle="square,pad=0",
                                        fc="#fff6f2" if 新 else "white", ec=GREY, lw=1.1))
            ax.text(x + w / 2, yy + 0.0675, t, ha="center", va="center",
                    fontsize=14.5, color=RED if (新 and "0" in t and "未同定" in t) else INK,
                    linespacing=1.5,
                    fontweight="bold" if (新 and "未同定" in t) else "normal")
    ax.text(0.035, 0.185,
            "Ia と fc は これまで一度も同定されておらず、恒久的に 0 だった。",
            fontsize=17, color=RED, fontweight="bold")
    ax.text(0.035, 0.125,
            "kp6 は事前に宣言した gauge（50 に固定）。j6 の kp と c は自由度が 1 つしかない。",
            fontsize=16, color=INK)
    ax.text(0.035, 0.075, "  ⇒ 24 個の数値のうち、**自由度は 23**。".replace("**", ""),
            fontsize=16, color=INK, fontweight="bold")
    ax.text(0.035, 0.025,
            "kv は dof_damping と厳密に縮退するため biasprm[:,2] ≡ 0 に固定。",
            fontsize=15, color=INK2)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "fig2_param_table.png"
    fig.savefig(p, dpi=170, facecolor="white"); plt.close(fig)
    return p


def _imu_round():
    """IMU 轮的最优值与近优 10% 倍差（按 6 关节聚合到 4 类）。"""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from optuna_ident import STUDY, storage as st
    s = optuna.load_study(study_name=STUDY, storage=st())
    tri = [t for t in s.trials if t.value is not None]
    best = min(tri, key=lambda t: t.value)
    near = [t for t in tri if t.value <= best.value * 1.10]
    M = {"kp": "kp", "damp": "c", "arm": "Ia", "fric": "fc"}
    out = {}
    for c, blk in M.items():
        # **kp6 是 params.py 预先声明的 gauge，本不该被搜索 ⇒ 不计入 kp 的统计**
        js = [j for j in range(6) if not (c == "kp" and j == 5)]
        sp = [float(np.array([t.params[f"{c}{j+1}"] for t in near]).max() /
              max(np.array([t.params[f"{c}{j+1}"] for t in near]).min(), 1e-15))
              for j in js]
        out[blk] = (np.array([best.params[f"{c}{j+1}"] for j in range(6)]),
                    float(np.median(sp)))
    return out, best.value, len(tri), len(near)


def _num(x: float) -> str:
    """投影可读的紧凑记法：小量用 e 记法，避免撑破列宽。"""
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 0.01:
        return f"{x:.3g}"
    return f"{x:.2e}".replace("e-0", "e-")


def fig3() -> Path:
    blk, bv, ntri, nnear = _imu_round()
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.958, "同定結果と、その信頼度", ha="center", fontsize=28,
            fontweight="bold", color=INK)
    ax.text(0.5, 0.9, f"IMU を目標にした探索：{ntri} trial（最良 {bv:.4f}）"
            f"　※ kp6 は事前宣言した gauge のため統計から除外",
            ha="center", fontsize=15.5, color=INK2)

    cx = [0.035, 0.20, 0.575, 0.785]
    cw = [0.16, 0.37, 0.205, 0.18]
    hdr = ["分類", "最適値（例：j1–j3）", "最良の 10% 以内での\nばらつき", "同定できたか"]
    y = 0.775
    for x, w, h in zip(cx, cw, hdr):
        ax.add_patch(FancyBboxPatch((x, y), w, 0.09, boxstyle="square,pad=0",
                                    fc="#eef1f5", ec=GREY, lw=1.3))
        ax.text(x + w / 2, y + 0.045, h, ha="center", va="center", fontsize=15.5,
                fontweight="bold", color=INK, linespacing=1.4)
    order = [("kp", "kp"), ("c", "damping"), ("Ia", "armature"), ("fc", "frictionloss")]
    for i, (k, lab) in enumerate(order):
        v, sp = blk[k]
        yy = y - (i + 1) * 0.125
        ok = sp <= 10.0
        cells = [lab,
                 " / ".join(_num(x) for x in v[:3]),
                 f"{sp:.0f} 倍",
                 "○ 同定可" if ok else "× 決まらない"]
        for x, w, t in zip(cx, cw, cells):
            ax.add_patch(FancyBboxPatch((x, yy), w, 0.125, boxstyle="square,pad=0",
                                        fc="white" if ok else "#fdf0ec", ec=GREY, lw=1.1))
            ax.text(x + w / 2, yy + 0.0625, t, ha="center", va="center", fontsize=15.5,
                    color=INK if ok else RED,
                    fontweight="bold" if t.startswith(("○", "×")) else "normal")
    ax.add_patch(FancyBboxPatch((0.035, 0.045), 0.93, 0.185,
                                boxstyle="round,pad=0.01,rounding_size=0.015",
                                fc="#fdf0ec", ec=RED, lw=2.0))
    ax.text(0.5, 0.185, "「最適値」だけを見てはいけない", ha="center",
            fontsize=19, fontweight="bold", color=RED)
    ax.text(0.5, 0.128,
            "損失が最良の 10% 以内でも、armature と frictionloss は 3〜4 桁ばらつく。",
            ha="center", fontsize=16.5, color=INK)
    ax.text(0.5, 0.077,
            "⇒ 手首 IMU だけでは、この 24 個は一意に決まらない（有効値であって実機械値ではない）。",
            ha="center", fontsize=16.5, color=INK, fontweight="bold")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "fig3_result_credibility.png"
    fig.savefig(p, dpi=170, facecolor="white"); plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=0)
    a = ap.parse_args()
    for i, fn in ((1, fig1), (2, fig2), (3, fig3)):
        if a.only in (0, i):
            print(f"  → {fn()}")
    print("\n  老师用一句话:")
    print("  G は、ACT の関節指令列を入力し、関節軌道と手首 IMU 応答を出力する")
    print("  Piper の MuJoCo モデルです。24 個の数値は G の内部パラメータで、")
    print("  Optuna は校正時に使う探索手段であり、G そのものではありません。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
