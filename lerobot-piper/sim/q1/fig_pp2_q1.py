"""pp2（17 集开发基准集）：候选 G 的 IMU 输出 vs 実機 IMU（+ G-2 参照线）。

规则「改完 G 必出 pp2 波形图」：沿用 `sim/step2_g/fig_pp2_gq_vs_real.py` 的版式 ——
中位集 + 最难集、0.5–9 Hz、各模型取自己的最优大域位移并写在图上、两条口径注记。
候选 = `q1_run.py fullfit` 的全数据拟合（seed 0，**只作图，不进任何判据**）；
G-2 = 同口径窗口重预测。窗口规则与阶段一相同（chunk 开头起点、末端钳位）。

    python -u sim/q1/fig_pp2_q1.py --out outputs/q1_stage1 --model 2b
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
import q1_data as D            # noqa: E402
import q1_windows as W         # noqa: E402

SHIFTS = (-2, -1, 0, 1, 2)
C_REAL, C_NEW, C_G2 = "#1f6fb2", "#8b2fb0", "#9aa0a6"


def bp(x) -> np.ndarray:
    import fit_step1_candidate as S1
    x = np.asarray(x, np.float64)
    return S1.bandpass41(x - x.mean(0, keepdims=True))


def imu_pair(ep, ehat) -> tuple[np.ndarray, np.ndarray]:
    """→ (帧 3…T−1 的实测陀螺, 预测陀螺 `ω̂_{t+3} = ideal_t + ê_t`)，均已带通。"""
    real = ep.w[D.ALIGN_L:D.ALIGN_L + ep.n]
    return bp(real), bp(ep.ideal[:ep.n] + ehat)


def ratio(real, pred, L: int = 0) -> float:
    """残差/实测（1.0 = 输出全零）；`L` = 大域位移（比较 real[k+L] 与 pred[k]）。"""
    if L >= 0:
        a, b = real[L + D.EDGE:len(real) - D.EDGE], pred[D.EDGE:len(pred) - D.EDGE - L]
    else:
        a, b = real[D.EDGE:len(real) - D.EDGE + L], pred[D.EDGE - L:len(pred) - D.EDGE]
    return float(((a - b) ** 2).sum() / max((a ** 2).sum(), 1e-12))


def best_shift(pairs) -> int:
    """全部集池化后残差最小的位移。"""
    def pooled(L):
        return sum(ratio(r, p, L) * float((r ** 2).sum()) for r, p in pairs)
    return min(SHIFTS, key=pooled)


def plot(rows, label: str, Ln: int, Lg: int, path: Path) -> None:
    """rows: [{"name","real","new","g2","r_new","r_g2"}]；按 r_new 取中位集与最难集。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.argsort([r["r_new"] for r in rows])
    med, hard = rows[order[len(order) // 2]], rows[order[-1]]
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(13.33, 7.5))
    gs = fig.add_gridspec(2, 1, hspace=0.42, left=0.065, right=0.985, top=0.845, bottom=0.105)

    def wave(ax, e, title, span=200):
        ax_i = int(np.argmax(np.abs(e["real"]).max(0)))
        c = int(np.argmax(np.abs(e["real"][:, ax_i])))
        s = max(0, min(c - span // 2, len(e["real"]) - span))
        t = np.arange(span) / D.FS
        ax.plot(t, e["real"][s:s + span, ax_i], color=C_REAL, lw=2.4, label="実機 IMU")
        ax.plot(t, e["new"][s:s + span, ax_i], color=C_NEW, lw=2.0, ls="--", label=label)
        ax.plot(t, e["g2"][s:s + span, ax_i], color=C_G2, lw=1.6, ls=":", label="G-2（参照）")
        ax.set_title(f"{title}  {e['name']}   残差/実測 {label} {e['r_new']:.3f} · G-2 {e['r_g2']:.3f}"
                     f"   軸 ω{'xyz'[ax_i]}", fontsize=12, color="#22223b", loc="left")
        ax.set_xlabel("時間 (s)", fontsize=11)
        ax.set_ylabel("角速度 (rad/s)\n0.5–9 Hz 帯域通過", fontsize=10.5)
        ax.grid(alpha=0.25)

    a1 = fig.add_subplot(gs[0])
    wave(a1, med, "① 中央値のエピソード")
    a1.legend(fontsize=11, frameon=False, ncol=3, loc="upper right")
    wave(fig.add_subplot(gs[1]), hard, "② 最も難しいエピソード")
    fig.suptitle(f"pp2（pick-and-place）：{label} の IMU 出力 と 実機 IMU",
                 fontsize=18, fontweight="bold", color="#22223b", y=0.982)
    fig.text(0.5, 0.912, f"帯域 0.5–9 Hz・大域シフト {label} L={Ln} / G-2 L={Lg}・"
             "chunk 先頭起点・末端クランプ・候補は 100 集全数学習（この 17 集は未使用）",
             ha="center", fontsize=11.5, color="#5a5a72")
    fig.text(0.5, 0.018, "※ 17 集は「pp2 開発基準集」であり最終密封 holdout ではない。"
             "残差/実測 は 1.0 が「出力ゼロ」相当。", ha="center", fontsize=10, color="#c1121f")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def main(argv=None) -> int:
    gc.disable()
    import q1_g2ref as G
    import q1_run as RUN
    import q1_train as TR
    from sim.step2_gap_baseline import bare_logs
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, choices=list(TR.MODELS))
    a = ap.parse_args(argv)
    R, b = D.calibration()
    model, st, Wrs, C = RUN.load_fullfit(a.out, a.model)      # 核对代码指纹与冻结步数
    g2 = G.load_g2()
    eps = [D.load_episode(p, "pp2/bare") for p in bare_logs()]
    raw = []
    for e in eps:
        ws = W.chunk_windows(e)
        new = TR.predict_episode(a.model, model, st, e, ws, R, Wrs, C)
        real, pn = imu_pair(e, new)
        _, pg = imu_pair(e, G.episode_ehat(g2, e, ws, R, b))
        raw.append((e.name, real, pn, pg))
    Ln = best_shift([(r, p) for _, r, p, _ in raw])
    Lg = best_shift([(r, g) for _, r, _, g in raw])
    rows = [{"name": n, "real": r, "new": p, "g2": g, "r_new": ratio(r, p, Ln), "r_g2": ratio(r, g, Lg)}
            for n, r, p, g in raw]
    for r in rows:
        print(f"  {r['name']:30s} {a.model} {r['r_new']:.3f}   G-2 {r['r_g2']:.3f}", flush=True)
    print(f"  逐集中位: {a.model} {np.median([r['r_new'] for r in rows]):.3f} · "
          f"G-2 {np.median([r['r_g2'] for r in rows]):.3f}   L={Ln}/{Lg}")
    p = Path(a.out) / "fig" / f"pp2_{a.model}_vs_real.png"
    plot(rows, a.model, Ln, Lg, p)
    print(f"  → {p}")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
