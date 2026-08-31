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


def _enc_round():
    """编码器轮：3 seed 的最优、跨 seed 极差、以及整块置零的消融。"""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from optuna_ident_enc import SEEDS, study_name, storage, theta_of
    best, ntri = {}, 0
    for sd in SEEDS:
        st = optuna.load_study(study_name=study_name(sd), storage=storage(sd))
        tri = [t for t in st.trials if t.value is not None]
        ntri += len(tri)
        best[sd] = min(tri, key=lambda t: t.value)
    out = {}
    for blk in ("kp", "c", "fc", "Ia"):
        js = [j for j in range(6) if not (blk == "kp" and j == 5)]
        rng = [max(best[sd].params[f"{blk}{j+1}"] for sd in SEEDS) /
               max(min(best[sd].params[f"{blk}{j+1}"] for sd in SEEDS), 1e-15) for j in js]
        v = np.array([best[SEEDS[1]].params[f"{blk}{j+1}"] for j in range(3)])
        out[blk] = (v, float(np.median(rng)))
    return out, min(b.value for b in best.values()), ntri


def _num(x: float) -> str:
    """投影可读的紧凑记法：小量用 e 记法，避免撑破列宽。"""
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 0.01:
        return f"{x:.3g}"
    return f"{x:.2e}".replace("e-0", "e-")


# 整块置零的消融（`RESULT_OPTUNA_IDENT_ENC.md` §5，训练集相对变化）
ABLATION = {"kp": None, "c": None, "fc": -0.2, "Ia": +2.1}
LABEL = {"kp": "kp", "c": "damping", "fc": "frictionloss", "Ia": "armature"}


def fig3() -> Path:
    blk, bv, ntri = _enc_round()
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.965, "同定結果と、その信頼度", ha="center", fontsize=27,
            fontweight="bold", color=INK)
    ax.text(0.5, 0.915, "関節エンコーダを目標にした探索：3 seed × 約 1000 trial"
            f"（計 {ntri} trial）", ha="center", fontsize=15, color=INK2)

    # ── 二つの結果を並べる：良くなった側と悪くなった側 ──
    for x0, w, tt, l1, l2, col, fc_ in (
        (0.045, 0.44, "関節軌道（未知姿勢）", "0.1241  →  0.0356",
         "−71%  改善", BLUE, "#eaf2f8"),
        (0.515, 0.44, "手首 IMU（未知姿勢）", "0.5930  →  0.6832",
         "+15%  悪化", RED, "#fdf0ec")):
        ax.add_patch(FancyBboxPatch((x0, 0.775), w, 0.115,
                                    boxstyle="round,pad=0.008,rounding_size=0.015",
                                    fc=fc_, ec=col, lw=2.0))
        ax.text(x0 + w / 2, 0.862, tt, ha="center", va="center", fontsize=14, color=INK2)
        ax.text(x0 + w * 0.36, 0.812, l1, ha="center", va="center", fontsize=16, color=INK)
        ax.text(x0 + w * 0.82, 0.812, l2, ha="center", va="center", fontsize=17,
                color=col, fontweight="bold")

    cx = [0.035, 0.185, 0.475, 0.655, 0.815]
    cw = [0.145, 0.28, 0.17, 0.15, 0.15]
    hdr = ["分類", "最適値（j1-j3）", "3 seed 間の\nばらつき",
           "0 にした時の\n悪化", "同定できたか"]
    y = 0.665
    for x, w, h in zip(cx, cw, hdr):
        ax.add_patch(FancyBboxPatch((x, y), w, 0.085, boxstyle="square,pad=0",
                                    fc="#eef1f5", ec=GREY, lw=1.3))
        ax.text(x + w / 2, y + 0.0425, h, ha="center", va="center", fontsize=14,
                fontweight="bold", color=INK, linespacing=1.35)
    for i, k in enumerate(("kp", "c", "Ia", "fc")):
        v, rng = blk[k]
        yy = y - (i + 1) * 0.098
        ok = rng <= 3.0
        ab = ABLATION[k]
        cells = [LABEL[k], " / ".join(_num(x) for x in v), f"{rng:.1f} 倍",
                 "主要因" if ab is None else f"{ab:+.1f}%",
                 "○ 同定可" if ok else "× 決まらない"]
        for x, w, t in zip(cx, cw, cells):
            ax.add_patch(FancyBboxPatch((x, yy), w, 0.098, boxstyle="square,pad=0",
                                        fc="white" if ok else "#fdf0ec", ec=GREY, lw=1.1))
            ax.text(x + w / 2, yy + 0.049, t, ha="center", va="center", fontsize=14.5,
                    color=INK if ok else RED,
                    fontweight="bold" if t.startswith(("○", "×", "主")) else "normal")

    ax.add_patch(FancyBboxPatch((0.035, 0.028), 0.93, 0.215,
                                boxstyle="round,pad=0.01,rounding_size=0.015",
                                fc="#fdf0ec", ec=RED, lw=2.0))
    ax.text(0.5, 0.200, "「決まらない」のは データ不足ではなく、効かないから",
            ha="center", fontsize=18, fontweight="bold", color=RED)
    ax.text(0.5, 0.148,
            "armature と frictionloss を 0 にしても、損失は 2% しか変わらない。",
            ha="center", fontsize=15.5, color=INK)
    ax.text(0.5, 0.100,
            "3 倍の改善はすべて kp と damping による。",
            ha="center", fontsize=15.5, color=INK)
    ax.text(0.5, 0.050,
            "⇒ これは「関節軌道」の予測モデル。IMU の G の置き換えではない。",
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
