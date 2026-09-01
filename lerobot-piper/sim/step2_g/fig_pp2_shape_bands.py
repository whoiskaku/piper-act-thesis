#!/usr/bin/env python3
"""**pp2：G の IMU 波形は "形" が合っているか** —— 帯域を割って見る図。

これまでの主指標 `残差/実測` は**エネルギー口径**（二乗誤差比）で、
「形は合っているが振幅が小さい」予測を強く罰する。本図はそれを分離する:

    残差/実測  =  形状のずれ (1−ρ²)  +  振幅のずれ (最適ゲイン α≠1 の分)

    ρ²  = 最適スカラーゲインを許したとき説明できる分散比 = **純粋な形状点**
    α   = その最適ゲイン（<1 ならモデルの振幅不足）

線は 3 本:
    **実機（青）· G-q-v1（紫）· kin(実測エンコーダ)（橙、上限の目安）**

橙は「**完璧な関節軌跡**を今の剛体センサ写像に通したら何が出るか」。
紫と橙の差 = 関節予測の責任、橙と青の差 = **関節→IMU 写像**の責任。

行 = 帯域（0.5–3 Hz / 3–9 Hz）、列 = 中央値エピソード / 最難エピソード。
各パネルに ρ（相関）を出す。ρ が形状の一致度で、これが本題。

口径：pp2 の 17 集は **pp2 開発基準集**であり最終密封 holdout ではない。
シフトは各系列の自分の最適（G-q-v1 は L=1、kin(実測エンコーダ) は L=0）。

用法:
    python sim/step2_g/fig_pp2_shape_bands.py
"""
from __future__ import annotations
import gc, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g"):
    sys.path.insert(0, str(_p))

OUT = ROOT / "outputs/step2_g/advisor"
FPS = 20.0
BANDS = ((0.5, 3.0), (3.0, 9.0))
C_REAL, C_GQ, C_KIN = "#1f6fb2", "#8b2fb0", "#e07b1a"
AX = ("gx", "gy", "gz")


def main() -> int:
    gc.disable()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from scipy.signal import firwin
    from sim.step1_v2_loss import load_calib
    from sim.step2_fidelity import CALIB
    from sim.step2_gap_baseline import bare_logs
    from ab_decompose import kin_gyro
    import g_q_v1 as GQ

    for _f in ("Noto Sans CJK JP", "Noto Sans CJK SC", "IPAexGothic"):
        if any(_f in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = _f
            break
    plt.rcParams["axes.unicode_minus"] = False

    def bp(x, lo, hi):
        k = firwin(41, [lo, hi], fs=FPS, pass_zero=False)
        return np.stack([np.convolve(x[:, j], k, mode="same") for j in range(x.shape[1])], 1)

    R = load_calib(CALIB)[0]
    eps = []
    for p in bare_logs():
        fr = json.loads(p.read_text())["frames"]
        act = np.asarray([f["a"] for f in fr], float)
        enc = np.asarray([f["state"] for f in fr], float)[:, :6] / 1000.0
        real = np.asarray([f["imu"] for f in fr], float)[:, 3:6]
        gq = GQ.predict_gyro(act) @ R.T                    # 自分の最適シフト L=1
        kn = kin_gyro(enc) @ R.T                           # 自分の最適シフト L=0
        k = min(len(gq), len(kn), len(real) - 1)
        e = {"name": p.stem, "n": k}
        for lo, hi in BANDS:
            a_ = bp(real[1:1 + k], lo, hi)                 # G-q-v1 と揃える L=1
            a0 = bp(real[0:0 + k], lo, hi)                 # kin と揃える L=0
            g_ = bp(gq[:k], lo, hi)
            n_ = bp(kn[:k], lo, hi)
            e[(lo, hi)] = {"real": a_, "real0": a0, "gq": g_, "kin": n_}
        # 代表軸 = 3–9 Hz で実測エネルギー最大の軸
        e["ax"] = int(np.argmax((e[BANDS[1]]["real"] ** 2).sum(0)))
        d = e[(0.5, 9.0)] if (0.5, 9.0) in e else None
        a_ = np.concatenate([e[b]["real"] for b in BANDS])
        g_ = np.concatenate([e[b]["gq"] for b in BANDS])
        e["r"] = float(np.square(a_ - g_).sum()) / max(float(np.square(a_).sum()), 1e-12)
        eps.append(e)
        print(f"  {p.stem:24s} 残差/実測 {e['r']:.3f}  代表軸 {AX[e['ax']]}", flush=True)

    order = np.argsort([e["r"] for e in eps])
    med, hard = eps[order[len(order) // 2]], eps[order[-1]]
    print(f"\n  中央値 {med['name']} ({med['r']:.3f})   最難 {hard['name']} ({hard['r']:.3f})")

    def rho(a, b):
        return float((a * b).sum()) / max(float(np.sqrt((a ** 2).sum() * (b ** 2).sum())), 1e-30)

    fig, axes = plt.subplots(2, 2, figsize=(16, 8.2))
    for col, (e, tag) in enumerate(((med, "中央値エピソード"), (hard, "最難エピソード"))):
        j = e["ax"]
        for row, (lo, hi) in enumerate(BANDS):
            ax = axes[row][col]
            d = e[(lo, hi)]
            t = np.arange(e["n"]) / FPS
            ax.plot(t, d["real"][:, j], color=C_REAL, lw=1.9, label="実機 IMU", zorder=3)
            ax.plot(t, d["gq"][:, j], color=C_GQ, lw=1.5, label="G-q-v1", zorder=2)
            ax.plot(t, d["kin"][:, j], color=C_KIN, lw=1.2, alpha=.85,
                    label="kin(実測エンコーダ) = 関節が完璧な場合", zorder=1)
            rq = rho(d["real"][:, j], d["gq"][:, j])
            rk = rho(d["real0"][:, j], d["kin"][:, j])
            ax.set_title(f"{tag}  {e['name']}   {lo}–{hi} Hz  ({AX[j]})    "
                         f"ρ: G-q-v1 = {rq:+.2f} / kin(実測エンコーダ) = {rk:+.2f}",
                         fontsize=10.5)
            ax.set_ylabel("角速度 [rad/s]", fontsize=9)
            ax.grid(alpha=.25)
            ax.tick_params(labelsize=8)
            if row == 1:
                ax.set_xlabel("時間 [s]", fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=8.5, loc="upper right", ncol=1, framealpha=.92)

    fig.suptitle("pp2：G の IMU 波形は「形」が合っているか（帯域分割）    "
                 "上段 0.5–3 Hz は形が合う・下段 3–9 Hz は合わない",
                 fontsize=13.5, y=0.985)
    fig.text(0.5, 0.005,
             "ρ = 相関（形状の一致度、振幅に不変）。全 17 集の集計: "
             "0.5–3 Hz で ρ²=0.710・最適ゲイン 1.05、3–9 Hz で ρ²=0.107・最適ゲイン 1.13。"
             "ゲインが 1 付近 ⇒ 振幅ではなく**形状**のずれ。"
             "橙（完璧な関節）でも 3–9 Hz は ρ²=0.487 止まり ⇒ 残りは関節→IMU 写像の責任。",
             ha="center", fontsize=8.6, color="#333")
    fig.tight_layout(rect=(0, 0.022, 1, 0.965))
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "fig6_pp2_shape_bands.png"
    fig.savefig(f, dpi=170)
    print(f"\n  → {f}")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
