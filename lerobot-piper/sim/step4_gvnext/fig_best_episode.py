#!/usr/bin/env python3
"""**留出集里效果最好的一条** —— G-vNext vs 真机 IMU，单轴 × 三个频段。

⚠ 这是**挑出来的最好一条**，不是典型水平。所以底部一行画出**全部 8 条**的分布，
把本图这条标出来，读者一眼能看到它在哪个位置。
老师明确要求过「平均指标だけでは不十分、難しい波形を見たい」⇒
**本图必须与 `fig12_one_axis_bands.png`（中位集 + 最难集）成对出示，不可单独用。**

轴的选法与 `fig12` 一致（全频段能量最大的那一轴），只换集 —— **只做一次挑选**。

用法:
    python -u sim/step4_gvnext/fig_best_episode.py
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

from data import load, bp, valid_files, FPS                    # noqa: E402
from train_rq import features as rq_features, fir_design       # noqa: E402
from train_rs import rs_features, sensor_target                # noqa: E402
from eval_gate import base_q                                   # noqa: E402
from ceiling_200hz import kin_gyro_fs                          # noqa: E402
from fig_raw_imu import r2_full                                # noqa: E402
from fig_one_axis_bands import rho2                            # noqa: E402

S2 = ROOT / "outputs/step4_gvnext/s2"
RQ, RS = ROOT / "outputs/step4_gvnext/rq", ROOT / "outputs/step4_gvnext/rs"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#a3a29b"
C_REAL, C_NX, C_HL = "#2a78d6", "#008300", "#eb6834"
AX = ("gx", "gy", "gz")
ROWS = (None, (0.5, 3.0), (3.0, 9.0))
WIN = 80


def main() -> int:
    import gc
    gc.disable()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, gridspec
    for _f in ("Noto Sans CJK JP", "Noto Sans CJK SC", "IPAexGothic"):
        if any(_f in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = _f
            break
    plt.rcParams["axes.unicode_minus"] = False

    zq, zs = np.load(RQ / "rq_fir.npz"), np.load(RS / "rs_fir.npz")
    sq = json.loads((RQ / "select.json").read_text())
    ss = json.loads((RS / "select.json").read_text())
    D = {}
    for f in valid_files(S2, "replay_pp2f*.json"):
        dd = load(f)
        t20 = dd["tg_i"][::5]
        Xq, _, _ = rq_features(dd)
        qh = base_q(dd, t20) + fir_design(Xq, sq["nlag"]) @ zq["W"]
        Ab, Kt, _ = sensor_target(dd, t20)
        n = len(Kt)
        Ps = fir_design(rs_features(dd, t20, qh)[:n], ss["nlag"]) @ zs["W"]
        D[dd["name"]] = {"A": Ab[:n], "S": kin_gyro_fs(qh, FPS)[:n] + Ps}
    rk = {k: r2_full(v["A"], v["S"]) for k, v in D.items()}
    o = sorted(D, key=lambda k: rk[k], reverse=True)
    nm = o[0]                                    # **唯一的一次挑选：最好的那条集**
    A, S = D[nm]["A"], D[nm]["S"]
    j = int(np.argmax((A ** 2).sum(0)))          # 轴的选法与 fig12 一致
    t = np.arange(len(A)) / FPS
    e = np.convolve(bp(A, 3.0, 9.0)[:, j] ** 2, np.ones(WIN), "same")
    k0 = int(np.argmax(e))
    w = slice(max(0, k0 - WIN // 2), k0 + WIN // 2)

    fig = plt.figure(figsize=(15.5, 9.4), facecolor=SURFACE)
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 0.5], hspace=0.62,
                           figure=fig)
    for r, band in enumerate(ROWS):
        ax = fig.add_subplot(gs[r]); ax.set_facecolor(SURFACE)
        aa = A[:, j] if band is None else bp(A, *band)[:, j]
        sm = S[:, j] if band is None else bp(S, *band)[:, j]
        tt, a_, s_ = (t[w], aa[w], sm[w]) if r == 2 else (t, aa, sm)
        if r < 2:
            ax.axvspan(t[w][0], t[w][-1], color=MUTED, alpha=.16, lw=0, zorder=0)
        ax.plot(tt, a_, color=C_REAL, lw=2.0, label="真机 IMU", zorder=2)
        ax.plot(tt, s_, color=C_NX, lw=1.5, ls=(0, (4, 2.4)),
                label="G-vNext（预测）", zorder=3)
        lbl = "全频段（不带通）" if band is None else f"{band[0]}–{band[1]} Hz"
        ax.set_ylabel(f"{lbl}\n[rad/s]", fontsize=10, color=INK2)
        ax.text(0.996, 0.05, f"ρ² = {rho2(aa, sm):.3f}（全集）",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=11, color=INK)
        ax.grid(color=MUTED, alpha=.28, lw=.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
        ax.tick_params(colors=INK2, labelsize=9)
        if r == 0:
            ax.set_title(f"**最好集** {nm}　軸 {AX[j]}（全频段能量最大轴）"
                         f"　　三轴合计 全频段 ρ² = {rk[nm]:.3f}",
                         fontsize=12.5, color=INK, loc="left", pad=8)
            ax.legend(fontsize=10, loc="upper right", framealpha=.92,
                      edgecolor=MUTED)
        if r == 2:
            ax.set_xlabel("時間 [s]　※放大到带内最强 4 秒；ρ² 仍为全集值",
                          fontsize=9.5, color=INK2)
    # ── 分布条：8 条留出集的全频段 ρ²，标出本图这条 ─────────────────────
    ax = fig.add_subplot(gs[3]); ax.set_facecolor(SURFACE)
    xs = [rk[k] for k in o]
    # 直接标注代替图例 —— 图例放在这里会压住上一行的 x 轴标签（已实测）
    ax.scatter(xs, np.zeros(len(xs)), s=64, color=MUTED, zorder=2)
    ax.scatter([rk[nm]], [0], s=150, color=C_HL, zorder=3)
    ax.scatter([np.median(xs)], [0], s=110, marker="|", linewidths=2.4,
               color=INK, zorder=4)
    ax.annotate("本图这条（最好）", (rk[nm], 0), textcoords="offset points",
                xytext=(0, 16), ha="center", fontsize=9.5, color=C_HL,
                fontweight="bold")
    ax.annotate(f"中位 {np.median(xs):.3f}", (np.median(xs), 0),
                textcoords="offset points", xytext=(0, 16), ha="center",
                fontsize=9.5, color=INK)
    for k in (o[0], o[-1]):
        ax.annotate(f"{k}\n{rk[k]:.3f}", (rk[k], 0), textcoords="offset points",
                    xytext=(0, -30), ha="center", fontsize=9, color=INK2)
    ax.set_xlim(0.68, 0.94); ax.set_ylim(-0.6, 0.45)
    ax.set_yticks([])
    ax.set_xlabel("全部 8 条留出集的**全频段三轴合计 ρ²** —— 本图是挑出来的最好一条，"
                  "不是典型水平", fontsize=10, color=INK)
    ax.grid(axis="x", color=MUTED, alpha=.28, lw=.8); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    fig.suptitle("G-vNext vs 真机 IMU：留出集里**效果最好的一条**"
                 "　　（需与中位集/最难集图成对出示）",
                 fontsize=13.5, y=0.982, color=INK)
    fig.text(0.5, 0.006,
             "口径：真机 100 Hz→9 Hz 低通→20 Hz（与仿真侧同型抗混叠），时基 = 设备 ms + 内核时刻，"
             "逐集正交 Procrustes 旋入机体系。留出 = 会话2 的 pp2f + pp2f60，轨迹与会话均未参与训练；"
             "两层 nlag/λ 由训练族折外选定。三个频段的数字不可互相替代引用。与 17 集 pp2 开发基准集不可并列。",
             ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.028, 1, 0.955))
    for p in (ROOT / "outputs/step4_gvnext", ROOT / "outputs/step2_g/advisor"):
        p.mkdir(parents=True, exist_ok=True)
        fig.savefig(p / "fig13_best_episode.png", dpi=170, facecolor=SURFACE)
    print(f"  最好集 {nm}  軸 {AX[j]}")
    for band in ROWS:
        aa = A[:, j] if band is None else bp(A, *band)[:, j]
        sm = S[:, j] if band is None else bp(S, *band)[:, j]
        print(f"    {'全频段' if band is None else f'{band[0]}–{band[1]} Hz':<10}"
              f"ρ² = {rho2(aa, sm):.3f}")
    print(f"  8 条分布: {' '.join(f'{rk[k]:.3f}' for k in o)}  中位 {np.median(xs):.3f}")
    print(f"\n  → {ROOT/'outputs/step4_gvnext/fig13_best_episode.png'}")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
