#!/usr/bin/env python3
"""**pp2 任务域：G-q-v1 の IMU 出力 vs 実機 IMU** —— 面談用の主図。

线：**実機（青）· G-q-v1（紫）** —— 参照線と逐集バーは 2026-09-02 に削除

- 帯域 0.5–9 Hz（π 損失と同一）、全体同一の大域シフト **L=1**
  （両モデルとも自分の最適シフトが L=1、`RESULT_OPTUNA_IDENT_ENC.md` §14）。
- 代表エピソードは**中央値**と**最難**の両方を出す。
  （平均指標だけでは不十分、難しい波形を見たい —— 2026-07-15 面談の指摘。）
- 参照線（G-π-phys-v1）と右下の逐集バーは**削除済み**（2026-09-02）。
  比較値はコンソール出力に残る。

口径：pp2 の 17 集は **pp2 開発基準集**（選択に 3 回使用済み）であり、
**最終密封 holdout ではない**。G-q-v1 はこの 17 集を一度も学習に使っていない。

用法:
    python sim/step2_g/fig_pp2_gq_vs_real.py
"""
from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g"):
    sys.path.insert(0, str(_p))

L = 1                       # 大域シフト（両モデル共通、各自の最適と一致）
OUT = ROOT / "outputs/step2_g/advisor"
C_REAL, C_GQ, C_PV = "#1f6fb2", "#8b2fb0", "#9aa0a6"


def main() -> int:
    gc.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--contact", action="store_true",
                    help="紫线换成 G-q-v1 + 桌面接触补丁，灰线为无补丁的 G-q-v1")
    A = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sim.step1_v2_loss import bandpass41, load_calib
    from sim.step2_fidelity import CALIB
    from sim.step2_gap_baseline import bare_logs
    import g_q_v1 as GQ
    if A.contact:
        import contact_patch as CP
        from g_q_v1 import THETA_STAR as _TH
        _zt = json.loads((CP.OUT / "calib.json").read_text())["z_best_mm"] / 1000.0
        _xml = CP.build_xml(_zt, CP.OUT / "table_best.xml")
        LBL_MAIN, LBL_REF = f"G-q-v1 + テーブル接触 (z_t={_zt*1000:.0f}mm)", "G-q-v1（参照）"
        TAG, SUB = "contact", "テーブル接触パッチ あり vs なし"
    else:
        import g_pi_phys_v1 as PV
        LBL_MAIN, LBL_REF = "G-q-v1", "G-π-phys-v1（参照）"
        TAG, SUB = "gq", "G-q-v1 vs G-π-phys-v1"

    R = load_calib(CALIB)[0]
    eps = []
    for p in bare_logs():
        fr = json.loads(p.read_text())["frames"]
        act = np.asarray([f["a"] for f in fr], float)
        real = np.asarray([f["imu"] for f in fr], float)[:, 3:6]
        if A.contact:
            gq = CP.roll(_xml, act, _TH)[1]
            pv = GQ.predict_gyro(act)
        else:
            gq = GQ.predict_gyro(act)
            pv = PV.predict_gyro(act, nframes=len(fr))
        k = min(len(gq), len(pv), len(real) - L)
        a_ = bandpass41(real[L:L + k])
        bq = bandpass41(gq[:k] @ R.T)
        bp = bandpass41(pv[:k] @ R.T)
        t_ = float(np.square(a_).sum())
        eps.append({"name": p.stem, "real": a_, "gq": bq, "pv": bp, "n": k,
                    "rq": float(np.square(a_ - bq).sum()) / max(t_, 1e-12),
                    "rp": float(np.square(a_ - bp).sum()) / max(t_, 1e-12)})
        print(f"  {p.stem:22s} G-q-v1 {eps[-1]['rq']:.3f}   phys-v1 {eps[-1]['rp']:.3f}",
              flush=True)

    order = np.argsort([e["rq"] for e in eps])
    med, hard = eps[order[len(order) // 2]], eps[order[-1]]
    print(f"\n  中位集 {med['name']} ({med['rq']:.3f}) · 最難集 {hard['name']} ({hard['rq']:.3f})")
    rq = np.array([e["rq"] for e in eps]); rp = np.array([e["rp"] for e in eps])
    print(f"  逐集中位: G-q-v1 {np.median(rq):.3f}  phys-v1 {np.median(rp):.3f}"
          f"   改善 {len([1 for a, b in zip(rq, rp) if a < b])}/{len(rq)} 集")

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(13.33, 7.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.42,
                          left=0.065, right=0.985, top=0.845, bottom=0.105)

    def wave(ax, e, title, span=200):
        ax_i = int(np.argmax(np.abs(e["real"]).max(0)))       # エネルギー最大の軸
        c = int(np.argmax(np.abs(e["real"][:, ax_i])))        # 最大振幅の周辺を切る
        s = max(0, min(c - span // 2, e["n"] - span)); sl = slice(s, s + span)
        t = np.arange(sl.stop - sl.start) / 20.0
        # 参照線（G-π-phys-v1）は描かない —— 本図は実機と本モデルの対比に絞る。
        # 比較値はコンソール出力に残してある。
        ax.plot(t, e["real"][sl, ax_i], color=C_REAL, lw=2.4, label="実機 IMU")
        ax.plot(t, e["gq"][sl, ax_i], color=C_GQ, lw=2.0, ls="--", label=LBL_MAIN)
        ax.set_title(f"{title}   残差/実測 {e['rq']:.3f}   軸 ω{'xyz'[ax_i]}",
                     fontsize=13, color="#22223b", loc="left")
        ax.set_xlabel("時間 (s)", fontsize=11)
        ax.set_ylabel("角速度 (rad/s)\n0.5–9 Hz 帯域通過", fontsize=10.5)
        ax.grid(alpha=0.25)

    a1 = fig.add_subplot(gs[0])
    wave(a1, med, "① 中央値のエピソード")
    a1.legend(fontsize=11.5, frameon=False, ncol=2, loc="upper right")
    a2 = fig.add_subplot(gs[1])
    wave(a2, hard, "② 最も難しいエピソード")

    fig.suptitle(f"pp2（pick-and-place）：{LBL_MAIN.split(chr(40))[0].strip()} の IMU 出力 と 実機 IMU",
                 fontsize=18, fontweight="bold", color="#22223b", y=0.982)
    fig.text(0.5, 0.912,
             "帯域 0.5–9 Hz・大域シフト L=1・G-q-v1 はこの 17 集を学習に使っていない（域外）",
             ha="center", fontsize=12, color="#5a5a72")
    fig.text(0.5, 0.018,
             "※ 17 集は「pp2 開発基準集」であり最終密封 holdout ではない。"
             "残差/実測 は 1.0 が「出力ゼロ」相当 ⇒ 帯域エネルギーの約 44% しか説明できていない。",
             ha="center", fontsize=10, color="#c1121f")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / (f"fig5_pp2_{TAG}_vs_real.png")
    fig.savefig(p, dpi=170, facecolor="white")
    print(f"\n  → {p}")
    return 0


if __name__ == "__main__":
    _c = main()
    sys.stdout.flush()
    import os
    os._exit(_c)
