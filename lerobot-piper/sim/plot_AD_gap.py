#!/usr/bin/env python3
"""A vs D 的 gap 对比图（2026-07-29）。

口径 = **训练口径**：`e(i) = real[i+3] − ideal[i]`，带通 0.5–9 Hz，陀螺三轴。
ideal 从 `chunks[].raw`（ACT 原始指令）逐 chunk 渲染。

四联图：
  ① 逐集配对（A→D 连线）—— 看方向一致性，不被均值掩盖
  ② 时间序列（配对最典型的一集）—— 看 π 在哪一段把 gap 做大
  ③ 频谱 —— 看 π 的影响集中在哪个频段
  ④ π 的 δa 幅值分布 —— 看饱和

用法: python sim/plot_AD_gap.py
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import sys
sys.path.insert(0, str(Path(__file__).parent))
from analyze_ab import _bp
from lerobot.common.policies.act.b_runtime import IdealChunkRenderer

ROOT = Path(__file__).resolve().parents[1]
AB2 = ROOT / "outputs/ab2"
L, FPS = 3, 20.0
EPS_B = 10.0          # mrad 硬界


def load(path):
    R = np.array(json.loads((ROOT / "outputs/p2_labels/pp2-deploy/manifest.json"
                             ).read_text())["R"], np.float32)
    d = json.loads(Path(path).read_text())
    fr, ch = d["frames"], d["chunks"]
    imu = np.array([f["imu"] for f in fr if f["imu"] is not None], np.float32)
    da = np.array([f["da"] for f in fr], np.float32) * 1e3       # mrad
    rend = IdealChunkRenderer(); outs, prev = [], None
    for c in ch:
        raw = np.array(c["raw"], np.float32)
        outs.append(rend.render(raw, prev)); prev = raw
    ideal = np.concatenate(outs)[:len(imu)][:, 3:6] @ R.T
    real = imu[:len(ideal), 3:6]
    ideal, real = ideal[:-L], real[L:]                            # 训练口径对齐
    g = _bp()(real - ideal)
    return g, float((g ** 2).sum() / len(g)), da


def eps(arm):
    fs = sorted(AB2.glob(f"{arm}_ep*.json"),
                key=lambda p: int(re.search(r"ep(\d+)", p.name).group(1)))
    return [f for f in fs if json.loads(f.read_text())["summary"]["frames"] > 500]


def main():
    G, E, DA = {}, {}, {}
    for arm in ("A", "D"):
        gs, es, ds = [], [], []
        for f in eps(arm):
            g, e, da = load(f); gs.append(g); es.append(e); ds.append(da)
        G[arm], E[arm], DA[arm] = gs, np.array(es), ds
    n = len(E["A"])

    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.suptitle("A(裸 ACT) vs D(ACT+π, FF 关闭) —— gap = real−ideal 陀螺, "
                 "0.5–9 Hz, LSHIFT=3 对齐（训练口径）", fontsize=12, y=0.98)

    # ① 逐集配对
    a0 = ax[0, 0]
    for i in range(n):
        c = "tab:red" if E["D"][i] > E["A"][i] else "tab:green"
        a0.plot([0, 1], [E["A"][i], E["D"][i]], "-o", color=c, alpha=.65, ms=4.5, lw=1.3)
    a0.plot([0, 1], [E["A"].mean(), E["D"].mean()], "-s", color="k", lw=2.6, ms=8,
            label=f"均值 {E['A'].mean():.4f} → {E['D'].mean():.4f}")
    a0.set_xticks([0, 1]); a0.set_xticklabels(["A 裸 ACT", "D ACT+π"])
    a0.set_ylabel("gap 能量 / 帧"); a0.legend(fontsize=9)
    worse = int((E["D"] > E["A"]).sum())
    a0.set_title(f"① 逐集配对 (n={n}) —— **{worse}/{n} 集 π 更差**（红）", fontsize=10)
    a0.grid(alpha=.3)

    # ② 时间序列: 取 D 相对 A 恶化最接近中位的一对
    dd = E["D"] - E["A"]
    k = int(np.argsort(dd)[len(dd) // 2])
    t = np.arange(len(G["A"][k])) / FPS
    m = min(len(G["A"][k]), len(G["D"][k]))
    a1 = ax[0, 1]
    a1.plot(t[:m], np.linalg.norm(G["A"][k][:m], axis=1), lw=.8, color="tab:blue",
            alpha=.8, label="A 裸")
    a1.plot(t[:m], np.linalg.norm(G["D"][k][:m], axis=1), lw=.8, color="tab:red",
            alpha=.8, label="D ACT+π")
    a1.set_xlabel("时间 (s)"); a1.set_ylabel("‖gap‖ (rad/s)")
    a1.set_title(f"② 时间序列（中位恶化的一对，第 {k} 对）", fontsize=10)
    a1.legend(fontsize=9); a1.grid(alpha=.3)

    # ③ 频谱（全集平均）
    a2 = ax[1, 0]
    for arm, c in (("A", "tab:blue"), ("D", "tab:red")):
        P = []
        for g in G[arm]:
            f_, p_ = _welch(g)
            P.append(p_)
        Lmin = min(len(p) for p in P)
        a2.semilogy(f_[:Lmin], np.mean([p[:Lmin] for p in P], 0), color=c,
                    label=f"{arm}", lw=1.4)
    a2.axvspan(0.5, 9.0, color="gray", alpha=.12)
    a2.set_xlabel("频率 (Hz)"); a2.set_ylabel("PSD")
    a2.set_xlim(0, 10)
    a2.set_title("③ gap 频谱（10 集平均，灰=评分带 0.5–9 Hz）", fontsize=10)
    a2.legend(fontsize=9); a2.grid(alpha=.3)

    # ④ δa 幅值分布
    a3 = ax[1, 1]
    allda = np.abs(np.concatenate(DA["D"])).ravel()
    a3.hist(allda, bins=80, color="tab:red", alpha=.75)
    a3.axvline(EPS_B, color="k", ls="--", lw=1.6, label=f"ε 硬界 {EPS_B:.0f} mrad")
    sat = (allda > 0.95 * EPS_B).mean() * 100
    a3.set_yscale("log"); a3.set_xlabel("|δa| (mrad)"); a3.set_ylabel("计数")
    a3.set_title(f"④ π 的 δa 幅值（RMS {np.sqrt((allda**2).mean()):.2f}, "
                 f"峰值 {allda.max():.2f}, 触界 {sat:.2f}%）", fontsize=10)
    a3.legend(fontsize=9); a3.grid(alpha=.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = ROOT / "outputs/ab2/AD_gap.png"
    fig.savefig(out, dpi=140)
    print(f"→ {out.relative_to(ROOT.parent)}")
    print(f"  A {E['A'].mean():.5f}  D {E['D'].mean():.5f}  "
          f"D−A {(E['A']-E['D']).mean()/E['A'].mean()*100:+.1f}%  {worse}/{n} 集更差")
    return 0


def _welch(g):
    from scipy.signal import welch
    f, p = welch(np.linalg.norm(g, axis=1), fs=FPS, nperseg=min(256, len(g)))
    return f, p


if __name__ == "__main__":
    raise SystemExit(main())
