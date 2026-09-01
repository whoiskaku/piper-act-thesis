#!/usr/bin/env python3
"""**单轴 × 三个频段**：G-2 的 IMU 输出 vs 真机 IMU。

行 = 全频段（不带通）/ 0.5–3 Hz / 3–9 Hz　　列 = 留出集的中位集与最难集。
轴 = 该集**全频段能量最大**的那一轴（图中标出）。

3–9 Hz 行放大到**带内能量最强的 4 秒**（30 秒全画会糊成一团，已实测），
上两行用浅色带标出该窗口的位置，便于对照同一时刻在三个频段里的样子。

留出 = 会话2 的 pp2f + pp2f60，轨迹与会话均未参与训练。

用法:
    python -u sim/step4_gvnext/fig_one_axis_bands.py
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

S2 = ROOT / "outputs/step4_gvnext/s2"
RQ, RS = ROOT / "outputs/step4_gvnext/rq", ROOT / "outputs/step4_gvnext/rs"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#a3a29b"
C_REAL, C_NX = "#2a78d6", "#008300"
AX = ("gx", "gy", "gz")
ROWS = (None, (0.5, 3.0), (3.0, 9.0))          # None = 不带通
WIN = 80                                        # 4 s @ 20 Hz


def rho2(a: np.ndarray, s: np.ndarray) -> float:
    return float((a * s).sum() ** 2 / max((a * a).sum() * (s * s).sum(), 1e-30))


def main() -> int:
    import gc
    gc.disable()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
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
        qb = base_q(dd, t20)
        qh = qb + fir_design(Xq, sq["nlag"]) @ zq["W"]
        Ab, Kt, _ = sensor_target(dd, t20)
        n = len(Kt)
        Ps = fir_design(rs_features(dd, t20, qh)[:n], ss["nlag"]) @ zs["W"]
        D[dd["name"]] = {"A": Ab[:n], "S": kin_gyro_fs(qh, FPS)[:n] + Ps}
    rk = {k: r2_full(v["A"], v["S"]) for k, v in D.items()}
    o = sorted(D, key=lambda k: rk[k], reverse=True)
    show = [(o[len(o) // 2], "中位集"), (o[-1], "最难集")]

    fig, axes = plt.subplots(3, 2, figsize=(16, 9.0), facecolor=SURFACE)
    for c, (nm, kind) in enumerate(show):
        A, S = D[nm]["A"], D[nm]["S"]
        j = int(np.argmax((A ** 2).sum(0)))          # 全频段能量最大的轴
        t = np.arange(len(A)) / FPS
        a39 = bp(A, 3.0, 9.0)[:, j]
        e = np.convolve(a39 ** 2, np.ones(WIN), "same")
        k0 = int(np.argmax(e))
        w = slice(max(0, k0 - WIN // 2), k0 + WIN // 2)
        for r, band in enumerate(ROWS):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            aa = A[:, j] if band is None else bp(A, *band)[:, j]
            sMs = S[:, j] if band is None else bp(S, *band)[:, j]
            tt, a_, s_ = (t[w], aa[w], sMs[w]) if r == 2 else (t, aa, sMs)
            if r < 2:                                 # 标出 3–9 Hz 行的放大窗口
                ax.axvspan(t[w][0], t[w][-1], color=MUTED, alpha=.16, lw=0,
                           zorder=0)
            ax.plot(tt, a_, color=C_REAL, lw=2.0, label="真机 IMU", zorder=2)
            ax.plot(tt, s_, color=C_NX, lw=1.5, ls=(0, (4, 2.4)),
                    label="G-2", zorder=3)
            lbl = "全频段（不带通）" if band is None else f"{band[0]}–{band[1]} Hz"
            ax.set_ylabel(f"{lbl}\n[rad/s]", fontsize=9.5, color=INK2)
            # ρ² 一律是**全集**的值（三行同口径）；第三行只是画面放大，
            # 不要写成"此窗口"——那会凭空多出一个口径不同的数字。
            ax.text(0.995, 0.05, f"ρ² = {rho2(aa, sMs):.3f}（全集）",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=10.5, color=INK)
            ax.grid(color=MUTED, alpha=.28, lw=.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(MUTED)
            ax.tick_params(colors=INK2, labelsize=8.5)
            if r == 0:
                ax.set_title(f"{kind}  {nm}　軸 {AX[j]}（全频段能量最大轴）"
                             f"　　全频段 ρ² = {rk[nm]:.3f}",
                             fontsize=12, color=INK, loc="left", pad=8)
                if c == 0:
                    ax.legend(fontsize=9.5, loc="upper right", framealpha=.92,
                              edgecolor=MUTED)
            if r == 2:
                ax.set_xlabel("時間 [s]　※画面放大到带内能量最强的 4 秒（上两行灰带处）；"
                              "ρ² 仍为全集值",
                              fontsize=9.5, color=INK2)
        print(f"  {nm} 軸{AX[j]}: 全频段 {rho2(A[:,j],S[:,j]):.3f}  "
              f"0.5–3 Hz {rho2(bp(A,0.5,3)[:,j],bp(S,0.5,3)[:,j]):.3f}  "
              f"3–9 Hz {rho2(bp(A,3,9)[:,j],bp(S,3,9)[:,j]):.3f}")
    fig.suptitle("G-2 vs 真机 IMU：**单轴 × 三个频段**"
                 "　　留出集（未见轨迹 + 跨会话）",
                 fontsize=13.5, y=0.985, color=INK)
    fig.text(0.5, 0.008,
             "全频段被低频主导 ⇒ 数字最好看；3–9 Hz 才是缺口所在（留出集中位 ρ²=0.520，门槛 0.707，刚体天井 0.649）。"
             "**三个频段的数字不可互相替代引用。** 口径：真机 100 Hz→9 Hz 低通→20 Hz（与仿真侧同型抗混叠），"
             "时基 = 设备 ms + 内核时刻，逐集正交 Procrustes 旋入机体系。与 17 集 pp2 开发基准集不可并列。",
             ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.032, 1, 0.955))
    for p in (ROOT / "outputs/step4_gvnext", ROOT / "outputs/step2_g/advisor"):
        p.mkdir(parents=True, exist_ok=True)
        fig.savefig(p / "fig12_one_axis_bands.png", dpi=170, facecolor=SURFACE)
    print(f"\n  → {ROOT/'outputs/step4_gvnext/fig12_one_axis_bands.png'}")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
