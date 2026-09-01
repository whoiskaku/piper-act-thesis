#!/usr/bin/env python3
"""**G-vNext 进展记分图** —— 留出集（未見軌跡 + 跨セッション）上分频段的达成度。

形式：每个频段一组**有序**横条（G-q-v1 → +r_q → G-vNext），共用 ρ² 轴；
门槛 / 刚体天花板 / 可重复性上界作参考线。三条是同一件事的递进 ⇒ 单色 ordinal 梯
（step 250/450/650，已过 `validate_palette.js --ordinal`）；门槛用保留的 status 色
并配 ✓/✗ 图标 + 文字，不靠颜色单独表意。

三个阶段在**本脚本内重算**（同一约定：真机逐集 Procrustes 旋入机体系、
逐系列最优位移、rotate=False），不拼接不同脚本的数字。

用法:
    python -u sim/step4_gvnext/fig_scorecard.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g", ROOT / "sim/step3_canobs", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data import load, bp, best_fit, valid_files, FPS          # noqa: E402
from train_rq import features as rq_features, fir_design       # noqa: E402
from train_rs import rs_features, sensor_target                # noqa: E402
from eval_gate import base_q                                   # noqa: E402
from ceiling_200hz import kin_gyro_fs                          # noqa: E402

S2 = ROOT / "outputs/step4_gvnext/s2"
RQ, RS = ROOT / "outputs/step4_gvnext/rq", ROOT / "outputs/step4_gvnext/rs"
BANDS = ((0.5, 3.0), (3.0, 9.0))
FRZ = {(0.5, 3.0): 0.789, (3.0, 9.0): 0.707}
REPEAT = {(0.5, 3.0): 0.983, (3.0, 9.0): 0.840}      # pp2 跨会话实测, 模型可达上界
# ── 设计系统参数（`references/palette.md`）────────────────────────────
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#a3a29b"
ORD = ["#86b6ef", "#2a78d6", "#104281"]              # ordinal 蓝梯 250/450/650
CRIT, GOOD = "#d03b3b", "#0ca30c"                    # status（保留色）


def main() -> int:
    import gc
    gc.disable()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    for _f in ("Noto Sans CJK JP", "Noto Sans CJK SC", "IPAexGothic"):
        if any(_f in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = _f
            break
    plt.rcParams["axes.unicode_minus"] = False

    zq, zs = np.load(RQ / "rq_fir.npz"), np.load(RS / "rs_fir.npz")
    sq = json.loads((RQ / "select.json").read_text())
    ss = json.loads((RS / "select.json").read_text())
    arms = ("G-q-v1", "+ r_q", "+ r_s = G-vNext")
    acc = {a: {b: [] for b in BANDS} for a in (*arms, "天井")}
    for f in valid_files(S2, "replay_pp2f*.json"):
        dd = load(f)
        t20 = dd["tg_i"][::5]
        Xq, Yq, _ = rq_features(dd)
        qb = base_q(dd, t20)
        Pq = fir_design(Xq, sq["nlag"]) @ zq["W"]
        Ab, Kt, _ = sensor_target(dd, t20)
        n = len(Kt)
        qh = qb + Pq
        Ps = fir_design(rs_features(dd, t20, qh)[:n], ss["nlag"]) @ zs["W"]
        pred = {"G-q-v1": kin_gyro_fs(qb, FPS)[:n],
                "+ r_q": kin_gyro_fs(qh, FPS)[:n],
                "+ r_s = G-vNext": kin_gyro_fs(qh, FPS)[:n] + Ps,
                "天井": Kt[:n]}
        for a, S in pred.items():
            m = min(len(Ab), len(S))
            for b in BANDS:
                acc[a][b].append(best_fit(Ab[:m], S[:m], *b, rotate=False)[1])
    V = {a: {b: float(np.median(acc[a][b])) for b in BANDS} for a in acc}

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 4.9), facecolor=SURFACE)
    for k, b in enumerate(BANDS):
        ax = axes[k]
        ax.set_facecolor(SURFACE)
        y = np.arange(3)[::-1]
        vals = [V[a][b] for a in arms]
        # 2px 表面间隙：用略小的高度 + 表面色描边
        ax.barh(y, vals, height=0.52, color=ORD, edgecolor=SURFACE, linewidth=2,
                zorder=3)
        for yy, v in zip(y, vals):
            ax.text(v + 0.012, yy, f"{v:.3f}", va="center", ha="left",
                    fontsize=11, color=INK, zorder=4)
        ax.set_yticks(y); ax.set_yticklabels(arms, fontsize=10.5, color=INK)
        # 参考线：门槛（status + 图标 + 文字）/ 刚体天井 / 可重复性上界
        ok = V[arms[-1]][b] >= FRZ[b]
        ax.axvline(FRZ[b], color=CRIT, lw=2, zorder=5)
        # 用 × 而不是 ✗：CJK 字体缺该字形，会渲染成方框（项目里踩过）
        ax.text(FRZ[b], 2.62, f"{'✓' if ok else '×'} 門槛 {FRZ[b]:.3f}",
                ha="center", va="bottom", fontsize=10.5,
                color=GOOD if ok else CRIT, fontweight="bold")
        # 参考线标签**沿线竖排**：横排放底部会互相重叠、并压住 x 刻度（已实测）
        for x, lab, st in ((V["天井"][b], f"刚体天井 {V['天井'][b]:.3f}", (4, 2)),
                           (REPEAT[b], f"可重复性上界 {REPEAT[b]:.3f}", (1, 2))):
            ax.axvline(x, color=MUTED, lw=1.4, ls=(0, st), zorder=2)
            ax.text(x - 0.012, -0.48, lab, rotation=90, ha="right", va="bottom",
                    fontsize=8.5, color=INK2, zorder=4)
        ax.set_xlim(0, 1.0); ax.set_ylim(-0.55, 2.55)
        ax.set_xlabel("ρ²（腕部陀螺，留出集中位）", fontsize=10, color=INK2)
        ax.set_title(f"{b[0]}–{b[1]} Hz", fontsize=12.5, color=INK, loc="left",
                     pad=22)
        ax.grid(axis="x", color=MUTED, alpha=.28, lw=.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(axis="x", colors=INK2, labelsize=9)
        ax.tick_params(axis="y", length=0)
    fig.legend(handles=[
        Line2D([], [], color=CRIT, lw=2, label="冻结门槛（预注册，事前定死）"),
        Line2D([], [], color=MUTED, lw=1.4, ls=(0, (4, 2)),
               label="刚体天井 = 真关节 + 刚体渲染"),
        Line2D([], [], color=MUTED, lw=1.4, ls=(0, (1, 2)),
               label="可重复性上界 = 真机自身（pp2 跨会话实测）")],
        loc="lower center", ncol=3, frameon=False, fontsize=9.5,
        bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("G-vNext 在**未見軌跡 + 跨セッション**留出上的达成度"
                 "　　0.5–3 Hz 已过门槛；3–9 Hz 的门槛本身高于刚体天井",
                 fontsize=13.5, y=0.985, color=INK)
    fig.text(0.5, 0.085,
             "留出 = 会话2 的 pp2f(5) + pp2f60(3)，轨迹与会话均未参与训练；两层的 nlag/λ 由训练族折外选定。"
             "口径：真机 100 Hz→9 Hz 低通→20 Hz，时基 = 设备 ms + 内核时刻，逐集 Procrustes 旋入机体系，逐系列最优位移。"
             "与 17 集 pp2 开发基准集不可并列。",
             ha="center", fontsize=8.6, color=INK2)
    fig.tight_layout(rect=(0, 0.115, 1, 0.945))
    for p in (ROOT / "outputs/step4_gvnext", ROOT / "outputs/step2_g/advisor"):
        p.mkdir(parents=True, exist_ok=True)
        fig.savefig(p / "fig10_gvnext_scorecard.png", dpi=170, facecolor=SURFACE)
    print(f"  → {ROOT/'outputs/step4_gvnext/fig10_gvnext_scorecard.png'}")
    for b in BANDS:
        print(f"  {b[0]}–{b[1]} Hz: " + " → ".join(f"{V[a][b]:.3f}" for a in arms)
              + f"   门槛 {FRZ[b]:.3f}  天井 {V['天井'][b]:.3f}  上界 {REPEAT[b]:.3f}")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
