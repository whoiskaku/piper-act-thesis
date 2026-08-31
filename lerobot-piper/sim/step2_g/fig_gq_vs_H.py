#!/usr/bin/env python3
"""**离线证据图**：G-q-v1 vs 已辨识的六关节 0.5–9 Hz 开环响应 `H`。

**只复用现有 `outputs/ab4/` 的 23 集**（P1=8 用于辨识），**不采新数据**，
**不作为继续训练 π 的入口**（见 `BPRIME_GQ_INCREMENTAL_GATE.md` §6、
`G_Q_V1_FREEZE.md` §10）。

## 做法（全部复用现成件）

- `sim.fit_step1_candidate.load_episode` + `IdealChunkRenderer` 取官方 `gap`
  （`real[t+3] − ideal[t]`，已去 IMU 标定），与 `STEP1_RESULT_v1` 同一口径。
- `fit_step1_candidate.fit_H(p1, ridge)` 拟合 **`H_real`**：`(16 lag, 3 轴, 6 关节)`。
- **`H_gq`**：把同一 `p1` 的 `gap` 换成 G-q-v1 的**增量**响应
  `G(a) − G(a − da)`（旋到 IMU 系），用**同一个** `fit_H` 再拟一次。
- 频域比较：`FRF_{c,j}(f) = Σ_l H[l,c,j] e^{−i2πf l/FPS}`，三轴取范数，
  只画 **0.5–9 Hz**（π 损失的带）。

## 口径约束

- 这是**增量通道**的对比。G-q-v1 已验证的优势在**绝对通道**
  （pp2 关节 −44%、IMU −42%），**两者不可混谈**。
- `H_real` 本身已通过跨 seed（β 中位 0.9546）与幅值线性（比值 1.0134），
  但**按它构造的候选补正只交付 2.65% < 8% 闸门** ⇒ `H` 可信，可部署补正量不足。

用法:
    python sim/step2_g/fig_gq_vs_H.py
"""
from __future__ import annotations
import gc, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g"):
    sys.path.insert(0, str(_p))

FPS = 20.0
AB4 = ROOT / "outputs/ab4"
CALIB = ROOT / "outputs/p2_labels/pp2a-deploy/manifest.json"
OUT = ROOT / "outputs/step2_g/advisor"
RIDGE = 1.0                      # RIDGE_GRID 的中点；两次拟合用同一值
BAND = (0.5, 9.0)


def frf(H, jf, nf=257):
    """H:(L,3,6) → 每关节的三轴范数 |FRF|(f)。"""
    L = H.shape[0]
    f = np.linspace(0.0, FPS / 2, nf)
    e = np.exp(-2j * np.pi * np.outer(f, np.arange(L)) / FPS)   # (nf, L)
    G = np.einsum("fl,lcj->fcj", e, H)                          # (nf,3,6)
    return f, np.linalg.norm(G, axis=1)                         # (nf,6)


def main() -> int:
    gc.disable()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fit_step1_candidate import load_episode, fit_H
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    import g_q_v1 as GQ

    cal = json.loads(CALIB.read_text())
    R = np.asarray(cal["R"], float)
    b = np.asarray(cal.get("b", [0.0, 0.0, 0.0]), float)
    rend = IdealChunkRenderer()
    paths = sorted(AB4.glob("P1_ep*.json"))
    print(f"P1 {len(paths)} 集（只复用现有 ab4，不采新数据）", flush=True)
    p1 = [load_episode(p, "P1", R, b, rend, 400) for p in paths]
    H_real, lam = fit_H(p1, RIDGE)
    print(f"  H_real 拟合完成 {H_real.shape}  λ={lam:.4g}", flush=True)

    # G-q-v1 的增量响应，替换 gap 后用同一个 fit_H
    import copy
    p1g = []
    for p, e in zip(paths, p1):
        fr = json.loads(p.read_text())["frames"]
        a = np.asarray([x["a"] for x in fr], float)
        da = np.asarray([x["da"] for x in fr], float)
        n = min(len(a), len(e.da))
        a0 = a[:n].copy(); a0[:, :6] -= da[:n]
        g1 = GQ.predict_gyro(a[:n]); g0 = GQ.predict_gyro(a0)
        k = min(len(g1), len(g0), len(e.gap))
        gq = np.zeros_like(e.gap)
        gq[:k] = (g1[:k] - g0[:k]) @ R.T
        eg = copy.copy(e); eg.gap = gq
        p1g.append(eg)
        print(f"  {p.stem}: G-q-v1 增量 RMS {np.sqrt((gq**2).mean())*1000:.2f} "
              f"vs 真机 gap RMS {np.sqrt((e.gap**2).mean())*1000:.2f} mrad/s", flush=True)
    H_gq, lam_g = fit_H(p1g, RIDGE)
    print(f"  H_gq 拟合完成  λ={lam_g:.4g}", flush=True)

    f, Ar = frf(H_real, None)
    _, Ag = frf(H_gq, None)
    m = (f >= BAND[0]) & (f <= BAND[1])
    ratio = [float(np.median(Ag[m, j] / np.maximum(Ar[m, j], 1e-12))) for j in range(6)]
    print(f"\n  {'关节':6s}{'|H_real| 中位':>14s}{'|H_gq| 中位':>13s}{'比值':>8s}")
    for j in range(6):
        print(f"  j{j+1:<5d}{np.median(Ar[m,j]):>14.4g}{np.median(Ag[m,j]):>13.4g}"
              f"{ratio[j]:>8.2f}")
    print(f"\n  比值跨关节: 中位 **{np.median(ratio):.2f}**  范围 "
          f"**{min(ratio):.2f}–{max(ratio):.2f}**（差 {max(ratio)/max(min(ratio),1e-9):.1f}×）")

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 3, figsize=(13.33, 7.0), sharex=True)
    for j, ax in enumerate(axes.ravel()):
        ax.semilogy(f[m], Ar[m, j], color="#2a78d6", lw=2.2, label="実機同定 H")
        ax.semilogy(f[m], Ag[m, j], color="#8b2fb0", lw=2.2, ls="--", label="G-q-v1")
        ax.set_title(f"j{j+1}   比 {ratio[j]:.2f}", fontsize=13, color="#22223b", loc="left")
        ax.grid(alpha=0.3, which="both")
        if j >= 3:
            ax.set_xlabel("周波数 (Hz)", fontsize=11)
        if j % 3 == 0:
            ax.set_ylabel("|FRF|  (3軸ノルム)", fontsize=11)
        if j == 0:
            ax.legend(fontsize=11, frameon=False)
    fig.suptitle("増分チャネル：G-q-v1 と 実機同定 H（6 関節・0.5–9 Hz、ab4 の既存 23 集のみ）",
                 fontsize=15, fontweight="bold", color="#22223b")
    fig.text(0.5, 0.005,
             "※ G-q-v1 の検証済みの優位は「絶対チャネル」（pp2 関節 −44% / IMU −42%）。"
             "本図は増分チャネルであり、π 訓練の入口ではない。",
             ha="center", fontsize=10.5, color="#c1121f")
    fig.tight_layout(rect=[0, 0.022, 1, 0.955])
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "fig4_gq_vs_H.png"
    fig.savefig(p, dpi=170, facecolor="white")
    print(f"\n  → {p}")
    (OUT / "gq_vs_H.json").write_text(json.dumps(
        {"ratio_per_joint": ratio, "band": BAND, "ridge": RIDGE,
         "lam_real": lam, "lam_gq": lam_g, "n_p1": len(p1),
         "note": "增量通道; 只复用 ab4; 非 π 训练入口"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    _c = main()
    sys.stdout.flush()
    import os
    os._exit(_c)
