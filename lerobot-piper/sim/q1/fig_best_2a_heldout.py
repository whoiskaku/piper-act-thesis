"""2a 在留出会话（ab5/ab6/ab7）上最好 / 中位 / 最难的一集：IMU 预测 vs 实测（+ G-2 参照线）。

每个会话用「没见过该会话」的那一折终训 2a（`jobs/2a/seed0/<会话>.pt`）预测；
G-2 = 同口径窗口重预测。按 2a 的「残差/实测」排序，取最小、中位、最大的三集各出一张图，
图上同时写出三会话逐集中位数。版式与计分沿用 `fig_pp2_q1.py`。

    OPENBLAS_NUM_THREADS=1 python -u sim/q1/fig_best_2a_heldout.py --out outputs/q1_stage1
"""
from __future__ import annotations
import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import q1_data as D                                           # noqa: E402
import q1_windows as W                                        # noqa: E402
from fig_pp2_q1 import imu_pair, ratio, best_shift, C_REAL, C_NEW, C_G2   # noqa: E402

SESSIONS = (("ab5", "A"), ("ab6", "A"), ("ab7", "Apost"))


def plot(e, kind, Ln, Lg, med_new, med_g2, n_eps, path: Path, span=200) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    ax_i = int(np.argmax(np.abs(e["real"]).max(0)))          # 能量最大的轴
    c = int(np.argmax(np.abs(e["real"][:, ax_i])))
    s = max(0, min(c - span // 2, len(e["real"]) - span))
    t = np.arange(span) / D.FS
    fig, ax = plt.subplots(figsize=(13.33, 5.2))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.78, bottom=0.16)
    ax.plot(t, e["real"][s:s + span, ax_i], color=C_REAL, lw=2.4, label="真机 IMU")
    ax.plot(t, e["new"][s:s + span, ax_i], color=C_NEW, lw=2.0, ls="--", label="2a 预测")
    ax.plot(t, e["g2"][s:s + span, ax_i], color=C_G2, lw=1.6, ls=":", label="G-2 预测（参照）")
    ax.set_xlabel("时间 (s)", fontsize=11)
    ax.set_ylabel("角速度 (rad/s)\n0.5–9 Hz 带通", fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=11, frameon=False, ncol=3, loc="upper right")
    fig.suptitle(f"2a 留出会话{kind}的一集：{e['name']}（轴 ω{'xyz'[ax_i]}）",
                 fontsize=17, fontweight="bold", color="#22223b", y=0.97)
    fig.text(0.5, 0.86, f"残差/实测：2a {e['r_new']:.3f} · G-2 {e['r_g2']:.3f}　"
             f"（三会话 {n_eps} 集逐集中位：2a {med_new:.3f} · G-2 {med_g2:.3f}）　"
             f"位移 2a L={Ln} / G-2 L={Lg}", ha="center", fontsize=11.5, color="#5a5a72")
    fig.text(0.5, 0.025, f"※ 按 2a 残差/实测排序取「{kind}」的一集；残差/实测 1.0 = 输出为零。"
             "模型训练时没见过该会话。数据来自旧桌面。", ha="center", fontsize=10, color="#c1121f")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def main(argv=None) -> int:
    gc.disable()
    import torch
    import q1_g2ref as G
    import q1_train as TR
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    R, b = D.calibration()
    g2 = G.load_g2()
    raw = []
    for s, arm in SESSIONS:
        ck = torch.load(Path(a.out) / "jobs/2a/seed0" / f"{s}_{arm}.pt", weights_only=False)
        model = TR.with_weights("2a", ck["stats"], ck["state_dict"])
        model.eval()
        for e in D.load_bare(((s, arm),)):
            ws = W.chunk_windows(e)
            real, pn = imu_pair(e, TR.predict_episode("2a", model, ck["stats"], e, ws, R))
            _, pg = imu_pair(e, G.episode_ehat(g2, e, ws, R, b))
            raw.append((e.name, real, pn, pg))
    Ln = best_shift([(r, p) for _, r, p, _ in raw])
    Lg = best_shift([(r, g) for _, r, _, g in raw])
    rows = [{"name": n, "real": r, "new": p, "g2": g, "r_new": ratio(r, p, Ln), "r_g2": ratio(r, g, Lg)}
            for n, r, p, g in raw]
    med_new = float(np.median([r["r_new"] for r in rows]))
    med_g2 = float(np.median([r["r_g2"] for r in rows]))
    print(f"  {len(rows)} 集 · 中位 2a {med_new:.3f} / G-2 {med_g2:.3f} · L={Ln}/{Lg}")
    order = sorted(rows, key=lambda r: r["r_new"])
    for kind, tag, e in (("最好", "best", order[0]), ("中位", "median", order[len(order) // 2]),
                         ("最难", "worst", order[-1])):
        p = Path(a.out) / "fig" / f"heldout_2a_{tag}.png"
        plot(e, kind, Ln, Lg, med_new, med_g2, len(rows), p)
        print(f"  {kind} {e['name']}: 2a {e['r_new']:.3f} / G-2 {e['r_g2']:.3f}  → {p}")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
