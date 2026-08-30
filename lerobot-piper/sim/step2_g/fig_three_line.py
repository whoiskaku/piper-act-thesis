#!/usr/bin/env python3
"""**受控激励域**的三线对比 —— 给指导教师的主图。

    蓝 = 现实（真机 IMU）
    绿 = 旧 G                        残差/实测 1.78（**比"输出全零"还差**），ρ = 0.22
    紫 = 新 G（加入固件加速度限幅器）    残差/实测 0.46，ρ = 0.77

两条模型线只差**一个因子**：MuJoCo 之前是否插入加速度限幅器。
`kp1 = 5000` 与其余全部参数两边逐字相同 ⇒ 差异可归因于该因子。

**域**：激励套件（真机 90 条中的 79 条）。同一改动在 pick-and-place 任务域
**不通过门槛**（6/17，中位 −13.7 pp），故**未**进入冻结的 G-π-phys-v1，
保留为**探针 / 长保持模式**。

代表序列取各类**中位数附近**那条（非择优），分布面板一并给出。
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "sim/step2_g"))
from sim.step1_v2_loss import BAND, bandpass41, load_calib   # noqa: E402
from sim.step2_fidelity import CALIB, apply_limiter          # noqa: E402
from fair_matrix import run_mj                               # noqa: E402
from calibrate import load_suite                             # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
C_REAL, C_OLD, C_NEW = "#2a78d6", "#008300", "#8b2fb0"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
OUT = ROOT / "outputs/step2_g/three_line"
PICK = [("S_j1_p2_r5_f1.00", "阶跃 j1", None),
        ("C_j1_p1", "扫频 j1 · 高频段（难处）", (0.60, 1.0)),
        ("M_all_s1", "多正弦 · 六关节同时", (0.0, 0.45))]
J = 1


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R, _ = load_calib(CALIB)
    data = {nm: (c, r) for nm, c, r in load_suite(True)}
    fig = plt.figure(figsize=(13.5, 10))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.25, 1.25, 1.25, 1.05], hspace=.72)
    fig.suptitle("**受控激励域**：真机 IMU vs 新旧 G　（真机 79 条）\n"
                 f"{BAND[0]}–{BAND[1]} Hz 陀螺 ω_y　"
                 "**蓝=现实（真机 IMU）　绿=旧 G　紫=新 G（加入固件加速度限幅器）**\n"
                 "两条模型线只差一个因子：是否插入限幅器；kp1=5000 与其余参数两边逐字相同",
                 fontsize=12, color=INK)

    for i, (nm, title, frac) in enumerate(PICK):
        cmd, real = data[nm]
        _, g0 = run_mj(cmd, {0: 5000.}, len(real))
        _, g1 = run_mj(apply_limiter(cmd, "a"), {0: 5000.}, len(real))
        k = min(len(g0), len(g1), len(real))
        a = bandpass41(real[:k]); b0 = bandpass41(g0[:k] @ R.T)
        b1 = bandpass41(g1[:k] @ R.T)
        lo, hi = (0, k) if frac is None else (int(frac[0] * k), int(frac[1] * k))
        x = np.arange(lo, hi) / 20.0
        ax = fig.add_subplot(gs[i])
        ax.plot(x, a[lo:hi, J], color=C_REAL, lw=2.2, label="现实（真机 IMU）", zorder=4)
        ax.plot(x, b0[lo:hi, J], color=C_OLD, lw=1.8, label="旧 G", zorder=3)
        ax.plot(x, b1[lo:hi, J], color=C_NEW, lw=2.0, label="新 G（+限幅器）", zorder=5)
        e0 = np.square(a[lo:hi] - b0[lo:hi]).sum()
        e1 = np.square(a[lo:hi] - b1[lo:hi]).sum()
        tt = np.square(a[lo:hi]).sum()
        c0 = float(np.corrcoef(a[lo:hi].ravel(), b0[lo:hi].ravel())[0, 1])
        c1 = float(np.corrcoef(a[lo:hi].ravel(), b1[lo:hi].ravel())[0, 1])
        ax.set_title(f"{title}　{nm}　（该类中位，非择优）\n"
                     f"残差/实测  旧 G **{e0/tt:.2f}** → 新 G **{e1/tt:.2f}**"
                     f"　（{(1-e1/e0)*100:+.0f}%）　"
                     f"相关 ρ  旧 G **{c0:+.2f}** → 新 G **{c1:+.2f}**",
                     fontsize=10, color=INK, loc="left")
        ax.set_ylabel("ω_y (rad/s)", fontsize=8.5, color=INK2)
        ax.set_xlabel("时间 (s)", fontsize=8.5, color=INK2)
        ax.grid(color=GRID, lw=.5); ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=9.5, frameon=False, ncol=3, loc="upper right")

    rows = []
    for nm, (cmd, real) in data.items():
        _, g0 = run_mj(cmd, {0: 5000.}, len(real))
        _, g1 = run_mj(apply_limiter(cmd, "a"), {0: 5000.}, len(real))
        k = min(len(g0), len(g1), len(real))
        a = bandpass41(real[:k]); tt = float(np.square(a).sum())
        rows.append((float(np.square(a - bandpass41(g0[:k] @ R.T)).sum()) / tt,
                     float(np.square(a - bandpass41(g1[:k] @ R.T)).sum()) / tt))
    r = np.array(rows)
    axb = fig.add_subplot(gs[3])
    for k_, (lab, col) in enumerate((("旧 G", C_OLD), ("新 G（+限幅器）", C_NEW))):
        v = r[:, k_]
        axb.scatter(np.full(len(v), k_) + np.linspace(-.2, .2, len(v)), v,
                    s=22, color=col, alpha=.6, zorder=3)
        axb.plot([k_ - .32, k_ + .32], [np.median(v)] * 2, color=INK, lw=2.6, zorder=4)
        axb.text(k_, np.median(v) + .12, f"中位 {np.median(v):.2f}", ha="center",
                 fontsize=11, color=INK, fontweight="bold")
    axb.axhline(1.0, color="#d1495b", ls="--", lw=1.4)
    axb.text(1.42, 1.05, "1.0 = 「预测全零」的水平", fontsize=9, color="#d1495b")
    axb.set_xticks([0, 1]); axb.set_xticklabels(["旧 G", "新 G（+限幅器）"], fontsize=11)
    axb.set_xlim(-.55, 1.75); axb.set_ylim(0, min(3.2, r.max() * 1.05))
    axb.set_ylabel("残差 / 实测", fontsize=9, color=INK2)
    axb.set_title(f"全 {len(r)} 条　每点一条序列，横线为中位　"
                  f"**改善 {int((r[:,1]<r[:,0]).sum())}/{len(r)}**　"
                  "旧 G 多数在 1.0 以上（**比不建模还差**），新 G 整体降到 0.5 附近",
                  fontsize=10, color=INK, loc="left")
    axb.grid(axis="y", color=GRID, lw=.5); axb.tick_params(labelsize=8.5)
    fig.text(.5, .006, "**域**：受控激励套件。同一改动在 pick-and-place 任务域"
             "**不通过门槛**（6/17，中位 −13.7 pp），故**未**进入冻结的 "
             "G-π-phys-v1，保留为探针 / 长保持模式。",
             ha="center", fontsize=9, color=INK2)
    fig.tight_layout(rect=[0, .024, 1, .915])
    fig.savefig(OUT / "three_line.png", dpi=150, facecolor="#fcfcfb")
    print(f"改善 {int((r[:,1]<r[:,0]).sum())}/{len(r)}　"
          f"旧 G 中位 {np.median(r[:,0]):.3f} → 新 G {np.median(r[:,1]):.3f}")
    print(f"→ {OUT/'three_line.png'}")


if __name__ == "__main__":
    main()
