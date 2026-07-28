#!/usr/bin/env python3
"""Ĝ 增量通道的真机检验 + A/D 代表集曲线（2026-07-29）。

**检验思路**（第一次有配对真机数据可以直接验 Ĝ）：
  D 臂实际施加了 δa，得到 e_D。若 Ĝ 正确，则「扣掉 δa 的后果」应回到 A 臂水平：

      e_counterfactual = e_D − Ĝ[δa_D]      ← Ĝ 预测的「没施加 δa 会怎样」
      若 Ĝ 正确:  energy(e_counterfactual) ≈ energy(e_A)

  反向同理：对 A 臂离线跑 π 得 δa，Ĝ 预测 e_A + Ĝ[δa] 应 ≈ e_D 水平。
  两个方向都按**频段**拆开 —— 图③显示 4–8 Hz 抬升最明显。

口径：`e(i) = real[i+3] − ideal[i]`，陀螺三轴，0.5–9 Hz。

用法: python sim/gcheck_AD.py
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import sys
sys.path.insert(0, str(Path(__file__).parent))
import train_pi_e2 as TP
from analyze_ab import _bp
from lerobot.common.policies.act.b_runtime import IdealChunkRenderer

ROOT = Path(__file__).resolve().parents[1]
AB2 = ROOT / "outputs/ab2"
L, FPS = 3, 20.0
BANDS = [(0.5, 2), (2, 4), (4, 6), (6, 8), (8, 9)]


def load(path, R):
    d = json.loads(Path(path).read_text())
    fr, ch = d["frames"], d["chunks"]
    imu = np.array([f["imu"] for f in fr if f["imu"] is not None], np.float32)
    da = np.array([f["da"] for f in fr], np.float32)
    rend = IdealChunkRenderer(); outs, prev = [], None
    for c in ch:
        raw = np.array(c["raw"], np.float32); outs.append(rend.render(raw, prev)); prev = raw
    ideal_sim = np.concatenate(outs)[:len(imu)]
    raw_all = np.concatenate([np.array(c["raw"], np.float32) for c in ch])[:len(imu)]
    ideal = ideal_sim[:, 3:6] @ R.T
    real = imu[:len(ideal), 3:6]
    m = min(len(ideal), len(real), len(da), len(raw_all))
    return (real[L:m], ideal[:m - L], da[:m - L], raw_all[:m - L], ideal_sim[:m - L])


def band_energy(g):
    """逐频段能量（Welch，rad/s 的模）。"""
    from scipy.signal import welch
    f, p = welch(np.linalg.norm(g, axis=1), fs=FPS, nperseg=min(256, len(g)))
    return {f"{lo}-{hi}": float(p[(f >= lo) & (f < hi)].sum()) for lo, hi in BANDS}


def eps(arm):
    fs = sorted(AB2.glob(f"{arm}_ep*.json"),
                key=lambda p: int(re.search(r"ep(\d+)", p.name).group(1)))
    return [f for f in fs if json.loads(f.read_text())["summary"]["frames"] > 500]


def main():
    R = np.array(json.loads((ROOT / "outputs/p2_labels/pp2-deploy/manifest.json"
                             ).read_text())["R"], np.float32)
    G = TP.GEnsemble()
    bp = _bp()

    print("=== Ĝ 增量通道的真机检验 ===")
    print("  e_counterfactual = e_D − Ĝ[δa_D]   （Ĝ 预测「没施加 δa 会怎样」）")
    print("  若 Ĝ 正确 ⇒ 它应回到 A 臂水平\n")

    EA, ED, ECF = [], [], []
    BA, BD, BCF = [], [], []
    gA, gD = [], []
    for f in eps("A"):
        real, ideal, da, raw, _ = load(f, R)
        g = bp(real - ideal); gA.append(g)
        EA.append((g ** 2).sum() / len(g)); BA.append(band_energy(g))
    for f in eps("D"):
        real, ideal, da, raw, _ = load(f, R)
        g = bp(real - ideal); gD.append(g)
        ED.append((g ** 2).sum() / len(g)); BD.append(band_energy(g))
        # Ĝ 预测 δa 造成的 IMU 增量
        epj, al, RB, RD = G.ep_context(raw[:, :6], raw[:, 6], R)
        dt = torch.from_numpy(da[:, :6]).float().to(TP.DEV).unsqueeze(0)
        de = TP.apply_G(dt, epj.unsqueeze(0), al.unsqueeze(0), RB.unsqueeze(0),
                        RD.unsqueeze(0), G)                    # (1,K,n,3)
        de_m = de[0].mean(0).detach().cpu().numpy()            # K 成员均值
        gcf = bp((real - ideal) - de_m)
        ECF.append((gcf ** 2).sum() / len(gcf)); BCF.append(band_energy(gcf))
    EA, ED, ECF = map(np.array, (EA, ED, ECF))

    print(f"  {'量':>34}{'能量/帧':>12}{'相对 A':>10}")
    print(f"  {'A 实测（裸）':>34}{EA.mean():>12.5f}{0.0:>9.1f}%")
    print(f"  {'D 实测（ACT+π）':>34}{ED.mean():>12.5f}"
          f"{ED.mean()/EA.mean()*100-100:>9.1f}%")
    print(f"  {'D − Ĝ[δa]  ← Ĝ 说的反事实':>34}{ECF.mean():>12.5f}"
          f"{ECF.mean()/EA.mean()*100-100:>9.1f}%")
    print(f"\n  Ĝ 预测扣掉 δa 后应回到 A 的 {EA.mean():.5f}；实际算出 {ECF.mean():.5f}")
    err = (ECF.mean() - EA.mean()) / EA.mean() * 100
    print(f"  **偏差 {err:+.1f}%** —— Ĝ {'高估' if err > 0 else '低估'}了 δa 的消减作用")

    print(f"\n  【按频段拆解】(Welch 能量)")
    print(f"  {'频段(Hz)':>10}{'A 实测':>11}{'D 实测':>11}{'D−Ĝ[δa]':>12}"
          f"{'Ĝ 反事实 vs A':>15}")
    for k in BA[0]:
        a_, d_, c_ = (np.mean([b[k] for b in X]) for X in (BA, BD, BCF))
        print(f"  {k:>10}{a_:>11.5f}{d_:>11.5f}{c_:>12.5f}"
              f"{(c_-a_)/max(a_,1e-12)*100:>14.1f}%")

    # ---------- 代表集曲线 ----------
    kA = int(np.argsort(np.abs(EA - np.median(EA)))[0])
    kD = int(np.argsort(np.abs(ED - np.median(ED)))[0])
    rA, iA, _, _, _ = load(eps("A")[kA], R)
    rD, iD, daD, _, _ = load(eps("D")[kD], R)
    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(f"A(裸 ACT, 第{kA}集) vs D(ACT+π, 第{kD}集) —— 均值水平的代表集\n"
                 f"陀螺 y 轴（主载体）, LSHIFT=3 对齐", fontsize=12)
    t = np.arange(len(rA)) / FPS
    ax[0].plot(t, iA[:, 1], lw=.9, color="k", alpha=.7, label="ideal（ACT 指令的运动学渲染）")
    ax[0].plot(t, rA[:, 1], lw=.9, color="tab:blue", alpha=.85, label="real A（裸 ACT）")
    ax[0].set_ylabel("ω_y (rad/s)"); ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)
    ax[0].set_title("① A 臂：real vs ideal 角速度", fontsize=10)
    t2 = np.arange(len(rD)) / FPS
    ax[1].plot(t2, iD[:, 1], lw=.9, color="k", alpha=.7, label="ideal")
    ax[1].plot(t2, rD[:, 1], lw=.9, color="tab:red", alpha=.85, label="real D（ACT+π）")
    ax[1].set_ylabel("ω_y (rad/s)"); ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
    ax[1].set_title("② D 臂：real vs ideal 角速度", fontsize=10)
    ax[2].plot(t, bp(rA - iA)[:, 1], lw=.9, color="tab:blue", alpha=.85,
               label=f"A gap（能量/帧 {EA[kA]:.4f}）")
    ax[2].plot(t2, bp(rD - iD)[:, 1], lw=.9, color="tab:red", alpha=.85,
               label=f"D gap（能量/帧 {ED[kD]:.4f}）")
    ax[2].set_xlabel("时间 (s)"); ax[2].set_ylabel("gap ω_y (rad/s)")
    ax[2].legend(fontsize=9); ax[2].grid(alpha=.3)
    ax[2].set_title("③ gap = real − ideal（0.5–9 Hz 带通）", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = AB2 / "AD_curves.png"
    fig.savefig(out, dpi=140)
    print(f"\n→ {out.relative_to(ROOT.parent)}   (A 第{kA}集 / D 第{kD}集)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
