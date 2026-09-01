#!/usr/bin/env python3
"""**G-2 的 IMU 输出 vs 真机 IMU —— 不带通、全频段原始波形。**

与 `fig9_gvnext_holdout.png` 的区别：那张按 0.5–3 / 3–9 Hz 分带看形状；
**本图不做任何带通**，直接叠三轴陀螺的原始波形（含直流）。

可以直接叠的依据：真机的直流均值 0.005–0.025 rad/s，只有 RMS(0.18–0.31) 的 5–10%，
且 `kin` 侧量级相当 ⇒ 不需要去均值，也不需要带通。

行 = 陀螺三轴（机体系）· 列 = 留出集的中位集与最难集（按**不带通**的 ρ² 排序）。
留出 = 会话2 的 pp2f + pp2f60，轨迹与会话均未参与训练。

用法:
    python -u sim/step4_gvnext/fig_raw_imu.py
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

from data import load, valid_files, FPS                       # noqa: E402
from train_rq import features as rq_features, fir_design      # noqa: E402
from train_rs import rs_features, sensor_target               # noqa: E402
from eval_gate import base_q                                  # noqa: E402
from ceiling_200hz import kin_gyro_fs                         # noqa: E402

S2 = ROOT / "outputs/step4_gvnext/s2"
RQ, RS = ROOT / "outputs/step4_gvnext/rq", ROOT / "outputs/step4_gvnext/rs"
# 设计系统参数（`references/palette.md`）；配对已过 validate_palette.js
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#a3a29b"
C_REAL, C_NX = "#2a78d6", "#008300"
AX = ("gx", "gy", "gz")


def r2_full(A: np.ndarray, S: np.ndarray) -> float:
    """**不带通**的 ρ²（三轴合计，允许一个最优标量增益）。"""
    aa = float((A * A).sum()); ss = float((S * S).sum()); as_ = float((A * S).sum())
    return as_ ** 2 / max(aa * ss, 1e-30)


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
    print("  不带通 ρ²（三轴合计，逐集）")
    for k in o:
        print(f"    {k:<16}{rk[k]:.3f}")

    fig, axes = plt.subplots(3, 2, figsize=(16, 8.6), sharex="col",
                             facecolor=SURFACE)
    for c, (nm, kind) in enumerate(show):
        A, S = D[nm]["A"], D[nm]["S"]
        t = np.arange(len(A)) / FPS
        for r in range(3):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            # **预测画在上层**：两条重合时下层会被完全遮住，读者看不出绿线在不在。
            # 虚线压实线，重合处的间隙里透出蓝色 —— 同时虚线本身也是二次编码
            # （配对的 tritan ΔE 落在 6–8 下限带，规范要求配二次编码）。
            ax.plot(t, A[:, r], color=C_REAL, lw=2.0, label="真机 IMU", zorder=2)
            ax.plot(t, S[:, r], color=C_NX, lw=1.5, ls=(0, (4, 2.4)),
                    label="G-2", zorder=3)
            ax.set_ylabel(f"{AX[r]}  [rad/s]", fontsize=9.5, color=INK2)
            ax.grid(color=MUTED, alpha=.28, lw=.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(MUTED)
            ax.tick_params(colors=INK2, labelsize=8.5)
            a_, s_ = A[:, r], S[:, r]
            ax.text(0.995, 0.04,
                    f"ρ² = {float((a_*s_).sum()**2/max((a_*a_).sum()*(s_*s_).sum(),1e-30)):.3f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=10, color=INK)
            if r == 0:
                ax.set_title(f"{kind}  {nm}    三轴合计 ρ² = {rk[nm]:.3f}",
                             fontsize=12, color=INK, loc="left", pad=8)
            if r == 0 and c == 0:
                ax.legend(fontsize=9.5, loc="upper right", framealpha=.92,
                          edgecolor=MUTED)
            if r == 2:
                ax.set_xlabel("時間 [s]", fontsize=9.5, color=INK2)
    fig.suptitle("G-2 的 IMU 输出 vs 真机 IMU —— **不带通、全频段原始波形**"
                 "　　留出集（未见轨迹 + 跨会话）",
                 fontsize=13.5, y=0.985, color=INK)
    fig.text(0.5, 0.008,
             "不去均值、不做任何带通：真机直流均值 0.005–0.025 rad/s，仅为 RMS(0.18–0.31) 的 5–10%。"
             "口径：真机 100 Hz→9 Hz 低通→20 Hz（抗混叠，与仿真侧同型），时基 = 设备 ms + 内核时刻，"
             "逐集正交 Procrustes 旋入机体系。留出 = 会话2 的 pp2f + pp2f60，两层 nlag/λ 由训练族折外选定。"
             "与 17 集 pp2 开发基准集不可并列。",
             ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.032, 1, 0.955))
    for p in (ROOT / "outputs/step4_gvnext", ROOT / "outputs/step2_g/advisor"):
        p.mkdir(parents=True, exist_ok=True)
        fig.savefig(p / "fig11_gvnext_raw_imu.png", dpi=170, facecolor=SURFACE)
    print(f"\n  → {ROOT/'outputs/step4_gvnext/fig11_gvnext_raw_imu.png'}")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
