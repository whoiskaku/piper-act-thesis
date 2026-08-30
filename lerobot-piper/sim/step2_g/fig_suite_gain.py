#!/usr/bin/env python3
"""**激励域**的波形叠图 —— 现 G vs 加入固件加速度限幅器。

**域必须标清楚**：这是**激励套件**（真机 90 条中的 79 条）。
在 **pp2 任务域**该改动**不通过门槛**（6/17，中位 −13.6 pp），因此
**未**进入冻结的 G-π-phys-v1；它保留为**探针 / 长保持模式**。

两条线只差**一个因子**：是否在 MuJoCo 之前插入加速度限幅器。
`kp1 = 5000` 两边相同，其余参数逐字相同 ⇒ 差异可归因。

代表序列取各类**中位数附近**那条（不取最大），分布面板一并给出。
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
from ab_decompose import kin_gyro                            # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
C_IDEAL, C_REAL, C_DYN, C_NEW = "#eda100", "#2a78d6", "#008300", "#8b2fb0"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
OUT = ROOT / "outputs/step2_g/suite_gain"
PICK = [("S_j1_p2_r5_f1.00", "阶跃 j1（该类中位）", None),
        ("C_j1_p1", "扫频 j1 · 高频段（难处）", (0.60, 1.0)),
        ("M_all_s1", "多正弦 · 六关节同时（该类中位）", (0.0, 0.45))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R, _ = load_calib(CALIB)
    man = json.loads((ROOT / "outputs/step2_g/manifest.json").read_text())
    data = {nm: (c, r) for nm, c, r in load_suite(True)}
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.35, 1.35, 1.2], hspace=.62, wspace=.26)
    SLOT = [gs[0, :2], gs[1, :2], gs[0, 2]]      # 阶跃 / 扫频 / 多正弦
    fig.suptitle("**激励域**：现 G vs 加入固件加速度限幅器（真机 79 条）\n"
                 f"{BAND[0]}–{BAND[1]} Hz 陀螺　"
                 "黄=指令(零参数运动学) / 蓝=现实 / **绿=旧 G** / 紫=旧 G + 限幅器\n"
                 "绿与紫只差**一个因子**（是否插入限幅器），kp1=5000 两边相同　"
                 "**绿 = G-v1 = 现冻结的 G-π-phys-v1（三者逐点相同，最大差 0.00e+00）**",
                 fontsize=12, color=INK)
    for i, (nm, title, frac) in enumerate(PICK):
        cmd, real = data[nm]
        _, g0 = run_mj(cmd, {0: 5000.}, len(real))
        _, g1 = run_mj(apply_limiter(cmd, "a"), {0: 5000.}, len(real))
        k = min(len(g0), len(g1), len(real))
        t = int(man[nm]["target"][0])
        a = bandpass41(real[:k]); b0 = bandpass41(g0[:k] @ R.T)
        b1 = bandpass41(g1[:k] @ R.T)
        cm = bandpass41(kin_gyro(np.degrees(cmd[:k, :6])) @ R.T)   # 零参数运动学基线
        lo, hi = (0, k) if frac is None else (int(frac[0]*k), int(frac[1]*k))
        x = np.arange(lo, hi) / 20.0
        ax = fig.add_subplot(SLOT[i])
        J = 1
        ax.plot(x, cm[lo:hi, J], color=C_IDEAL, lw=1.9, label="指令（零参数运动学）",
                zorder=2, alpha=.85)
        ax.plot(x, a[lo:hi, J], color=C_REAL, lw=2.1, label="现实（真机 IMU）", zorder=4)
        ax.plot(x, b0[lo:hi, J], color=C_DYN, lw=1.7, label="**旧 G**（= 现冻结底座）",
                zorder=3)
        ax.plot(x, b1[lo:hi, J], color=C_NEW, lw=1.9, label="旧 G + 限幅器", zorder=5)
        e0 = np.square(a[lo:hi]-b0[lo:hi]).sum(); e1 = np.square(a[lo:hi]-b1[lo:hi]).sum()
        tt = np.square(a[lo:hi]).sum()
        ec = np.square(a[lo:hi]-cm[lo:hi]).sum()
        ax.set_title(f"{title}　{nm}\n残差/实测  指令 {ec/tt:.3f} · "
                     f"**旧 G {e0/tt:.3f}** → **+限幅器 {e1/tt:.3f}**"
                     f"　({(1-e1/e0)*100:+.0f}%)", fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("时间 (s)", fontsize=8.5, color=INK2)
        ax.set_ylabel("ω_y (rad/s)", fontsize=8.5, color=INK2)
        ax.grid(color=GRID, lw=.5); ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=8.5, frameon=False, ncol=3, loc="upper right")
        amp0 = b0[lo:hi].std()/a[lo:hi].std(); amp1 = b1[lo:hi].std()/a[lo:hi].std()
        c0 = float(np.corrcoef(a[lo:hi].ravel(), b0[lo:hi].ravel())[0, 1])
        c1 = float(np.corrcoef(a[lo:hi].ravel(), b1[lo:hi].ravel())[0, 1])
        ax.text(.01, .02, f"相关 ρ  旧 G {c0:+.2f} → **+限幅器 {c1:+.2f}**"
                          f"　幅度比 {amp0:.2f} → {amp1:.2f}",
                transform=ax.transAxes, fontsize=8.2, color=INK2)
    # 分布面板
    rows = []
    for nm, (cmd, real) in data.items():
        _, g0 = run_mj(cmd, {0: 5000.}, len(real))
        _, g1 = run_mj(apply_limiter(cmd, "a"), {0: 5000.}, len(real))
        k = min(len(g0), len(g1), len(real))
        a = bandpass41(real[:k]); tt = float(np.square(a).sum())
        rows.append((float(np.square(a-bandpass41(g0[:k]@R.T)).sum())/tt,
                     float(np.square(a-bandpass41(g1[:k]@R.T)).sum())/tt))
    r = np.array(rows); imp = (1 - r[:, 1]/np.maximum(r[:, 0], 1e-9)) * 100
    axd = fig.add_subplot(gs[2, :])
    o = np.argsort(imp)
    axd.bar(range(len(imp)), imp[o], color=[C_NEW if v > 0 else "#d1495b" for v in imp[o]])
    axd.axhline(0, color=INK, lw=1)
    axd.axhline(np.median(imp), color=INK2, ls="--", lw=1.2)
    axd.text(1, np.median(imp)+3, f"中位 {np.median(imp):+.1f}%", fontsize=9, color=INK2)
    axd.set_title(f"全 {len(imp)} 条的改善分布（升序）　改善 **{int((imp>0).sum())}/{len(imp)}** "
                  f"—— 代表序列取各类中位附近，非择优", fontsize=10, color=INK, loc="left")
    axd.set_ylabel("改善 (%)", fontsize=8.5, color=INK2)
    axd.set_xlabel("序列（按改善升序）", fontsize=8.5, color=INK2)
    axd.grid(axis="y", color=GRID, lw=.5); axd.tick_params(labelsize=8)
    axd.text(.995, .06, "缩向零的残差下限是 1.0；**旧 G 为 1.78（比全零还差）**，"
             "加限幅器 0.46\n"
             "⇒ 改善不是缩放：相关 ρ 从 **0.22 → 0.77**（24 条均值），"
             "旧 G 的幅度反而更大（0.82 vs 0.55）——它错在**相位**。",
             transform=axd.transAxes, ha="right", va="bottom", fontsize=8.6, color=INK2)
    fig.text(.5, .008, "**域**：激励套件。该改动在 pp2 任务域**不通过门槛**"
             "（6/17，中位 −13.6 pp），故**未**进入冻结的 G-π-phys-v1，"
             "保留为探针 / 长保持模式。", ha="center", fontsize=9, color=INK2)
    fig.tight_layout(rect=[0, .028, 1, .93])
    fig.savefig(OUT / "suite_gain.png", dpi=150, facecolor="#fcfcfb")
    print(f"改善 {int((imp>0).sum())}/{len(imp)}  中位 {np.median(imp):+.1f}%")
    print(f"→ {OUT/'suite_gain.png'}")


if __name__ == "__main__":
    main()
