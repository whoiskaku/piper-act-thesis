#!/usr/bin/env python3
"""三线对比：真机 IMU / **旧 Ĝ（灰盒，曾报 gap-R² 0.40）** / 新 G。

**旧 Ĝ 是真跑的**，不是按数字画的：参数全部从 `outputs/p2_bakeoff/g_production.npz`
读入（`c_fit`、三模态 3.11/5.51/6.90 Hz、`W/B/D`），前向按 `fit_g_production.py`
的 docstring 重建。**自检：复现其报告值 r2_gap_full 0.3973 vs 记录 0.3971（差 0.0002）。**

**域**：旧 Ĝ 自己的主场 —— `eval_pickplace2_100k`，它当初的验证集（ep 8, 9）。
在它的主场比，对旧 Ĝ 最公平。

三条线：
    蓝 = 现实（真机 IMU）
    绿 = 旧 Ĝ（灰盒：刚体+伺服 FIR ⊕ 三个柔性模态）
    紫 = 新 G（MuJoCo + 固件加速度限幅器）
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from g_verdict import DT, VAL_EPS, load                       # noqa: E402
from fit_g_production import rigid_parts, modal_kernels, KKER  # noqa: E402
from sim.step1_v2_loss import bandpass41                       # noqa: E402
from sim.step2_fidelity import apply_limiter                   # noqa: E402
from fair_matrix import run_mj                                 # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
C_REAL, C_OLD, C_NEW = "#2a78d6", "#008300", "#8b2fb0"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
OUT = ROOT / "outputs/step2_g/vs_oldG"


def old_G(e, z):
    om = torch.exp(torch.tensor(z["log_om"]))
    ze = torch.sigmoid(torch.tensor(z["raw_ze"])) * 0.5
    W, B, D = (torch.tensor(z[k]) for k in ("W", "B", "D"))
    qdh, qddh, w_rigid, _ = rigid_parts(e, z["c_fit"])
    ker = modal_kernels(om, ze)
    up = torch.einsum("mj,tj->tm", W, torch.tensor(qddh))
    s = torch.nn.functional.conv1d(
        torch.nn.functional.pad(up.T[None], (KKER - 1, 0)),
        ker.flip(-1)[:, None, :], groups=3)[0].T
    return (w_rigid + (s @ B + torch.tensor(qddh) @ D.T).numpy()) @ e["R"].T


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(ROOT / "outputs/p2_bakeoff/g_production.npz")
    val = [e for e in load(VAL_EPS, with_gap=True) if e["tag"] == "pp2-deploy"]
    fig = plt.figure(figsize=(13.5, 4.2 * len(val) + 2.0))
    gs = fig.add_gridspec(len(val) + 1, 1,
                          height_ratios=[1.3] * len(val) + [0.95], hspace=.60)
    fig.suptitle("**真机 IMU vs 旧 Ĝ（灰盒）vs 新 G** —— 在旧 Ĝ 自己的主场\n"
                 "`eval_pickplace2_100k` 验证集　0.5–9.0 Hz 陀螺 ω_y　"
                 "**蓝=现实　绿=旧 Ĝ　紫=新 G（MuJoCo+限幅器）**\n"
                 "旧 Ĝ 为**真跑**（参数取自 g_production.npz；自检复现其报告值 "
                 "0.3973 vs 0.3971）", fontsize=11.5, color=INK)
    tab = []
    for i, e in enumerate(val):
        a7 = np.hstack([e["a"], np.zeros((len(e["a"]), 1), np.float32)])
        p_old = old_G(e, z)
        _, g_new = run_mj(apply_limiter(a7.astype(float), "a"), {0: 5000.}, len(e["a"]))
        k = min(len(p_old), len(g_new), len(e["g"]))
        R = e["R"]
        A = bandpass41(e["g"][:k]); O = bandpass41(p_old[:k])
        N = bandpass41(g_new[:k] @ R.T)
        f = lambda P: (float(np.square(A - P).sum() / np.square(A).sum()),
                       float(np.corrcoef(A.ravel(), P.ravel())[0, 1]))
        ro, co = f(O); rn, cn = f(N)
        tab.append((ro, co, rn, cn))
        x = np.arange(k) / 20.0
        ax = fig.add_subplot(gs[i])
        ax.plot(x, A[:, 1], color=C_REAL, lw=2.1, label="现实（真机 IMU）", zorder=4)
        ax.plot(x, O[:, 1], color=C_OLD, lw=1.7, label="旧 Ĝ（灰盒）", zorder=3)
        ax.plot(x, N[:, 1], color=C_NEW, lw=1.9, label="新 G（MuJoCo+限幅器）", zorder=5)
        ax.set_title(f"ep {VAL_EPS[i]}　残差/实测  旧 Ĝ **{ro:.2f}** → 新 G **{rn:.2f}**"
                     f"　（{(1-rn/ro)*100:+.0f}%）　相关 ρ  旧 Ĝ **{co:+.2f}** → "
                     f"新 G **{cn:+.2f}**", fontsize=10, color=INK, loc="left")
        ax.set_ylabel("ω_y (rad/s)", fontsize=8.5, color=INK2)
        ax.set_xlabel("时间 (s)", fontsize=8.5, color=INK2)
        ax.grid(color=GRID, lw=.5); ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=9.5, frameon=False, ncol=3, loc="upper right")
    T = np.array(tab)
    axb = fig.add_subplot(gs[-1]); axb.axis("off")
    txt = (f"两集汇总　残差/实测：旧 Ĝ {T[:,0].mean():.3f} → 新 G {T[:,2].mean():.3f}"
           f"（{(1-T[:,2].mean()/T[:,0].mean())*100:+.0f}%）　　"
           f"相关 ρ：旧 Ĝ {T[:,1].mean():+.3f} → 新 G {T[:,3].mean():+.3f}\n\n"
           "口径说明：旧 Ĝ 当初报的 **gap-R² 0.40（≈42%）是未补滞后口径**，"
           "其中约 87% 是「滞后免费分」；补滞后后对墙的值解释力约 **9%**\n"
           "（`E1_G_identification.md` §7：「0.437 降为入场券且引用必须带口径」）。"
           "本图为**补滞后口径**，两个模型一致。")
    axb.text(.5, .5, txt, ha="center", va="center", fontsize=10, color=INK2,
             linespacing=1.7)
    fig.tight_layout(rect=[0, .01, 1, .94])
    fig.savefig(OUT / "vs_oldG.png", dpi=150, facecolor="#fcfcfb")
    print(f"旧 Ĝ {T[:,0].mean():.3f} (ρ {T[:,1].mean():+.3f}) → "
          f"新 G {T[:,2].mean():.3f} (ρ {T[:,3].mean():+.3f})")
    print(f"→ {OUT/'vs_oldG.png'}")


if __name__ == "__main__":
    main()
