#!/usr/bin/env python3
"""Step 1 v2 の IMU gap 波形対比図（`AD_curves.png` と同じ構成）。

口径は H1 主指標と同一：`gap(t) = real[t+3] − ideal_cal[t]`、0.5–9 Hz 帯域、
ideal は `chunks[].raw`（ACT 原指令）から逐 chunk 運動学レンダリング。**G も H も未使用。**

五連図:
  ① 素の ACT（基線）  real vs ideal
  ② 補正あり          real vs ideal
  ③ gap = real − ideal（帯域通過）両者重ね
  ④ 実際に加えた δa —— 継ぎ目パルスの実物。**gap との桁差**が要点
  ⑤ gap スペクトル（全エピソード平均、灰＝評点帯 0.5–9 Hz）

対比する 2 本は**時間的に隣接**したものを選ぶ（セッション内ドリフトを最小化）。

用法: python sim/plot_spsa_v2_curves.py [--axis 0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.step1_v2_loss import render_ideal, bandpass41, load_calib, ALIGN_L, BAND

OUT = ROOT / "outputs/spsa_v2"
FPS = 20.0
AXNAME = ["ω_x", "ω_y", "ω_z"]


def load_ep(tag: str, Rc: np.ndarray):
    """→ (ideal_gyro, real_gyro, gap_bp, da) すべて整列済み。"""
    p = json.loads((OUT / f"{tag}_ep0.json").read_text())
    fr = p["frames"]
    real = np.asarray([f["imu"] for f in fr], float)
    da = np.asarray([f["da"] for f in fr], float)
    ideal_sim = render_ideal(p["chunks"], len(fr))
    n = min(len(fr) - ALIGN_L, len(ideal_sim))
    ig = ideal_sim[:n, 3:6] @ Rc.T
    rg = real[ALIGN_L:ALIGN_L + n, 3:6]
    return ig, rg, bandpass41(rg - ig), da[:n], p["summary"]


def psd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    f = np.fft.rfftfreq(n, d=1 / FPS)
    P = np.abs(np.fft.rfft(x, axis=0)) ** 2 / n
    return f, P.sum(axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", type=int, default=0, help="表示するジャイロ軸 (0/1/2)")
    ap.add_argument("--tmax", type=float, default=None, help="表示秒数")
    a_ = ap.parse_args()
    J = a_.axis

    s = json.loads((OUT / "state.json").read_text())
    Rc, _ = load_calib(ROOT / "outputs/p2_labels/pp2a-deploy/manifest.json")

    # 時系列順に並べ、基線とその**直後**の補正集を選ぶ（ドリフト最小）
    seq = [(b["round"], -1, b["tag"], True) for b in s["baselines"]]
    for h in s["history"]:
        for i, side in enumerate(h["order"]):
            seq.append((h["round"], i, f"R{h['round']:02d}{side}", False))
    seq.sort(key=lambda r: (r[0], r[1]))
    pair = None
    for i in range(len(seq) - 1):
        if seq[i][3] and not seq[i + 1][3]:
            if (OUT / f"{seq[i][2]}_ep0.json").exists() and (OUT / f"{seq[i+1][2]}_ep0.json").exists():
                pair = (seq[i][2], seq[i + 1][2])
    assert pair, "隣接する 基線→補正 の組が見つからない"
    tb, tc = pair

    igb, rgb, gb, _, sb = load_ep(tb, Rc)
    igc, rgc, gc, dac, sc = load_ep(tc, Rc)
    th = sc.get("theta", {})
    n = min(len(gb), len(gc))
    t = np.arange(n) / FPS
    tmax = a_.tmax or t[-1]

    Lb = float(np.square(gb).sum(1).mean()); Lc = float(np.square(gc).sum(1).mean())

    fig, ax = plt.subplots(5, 1, figsize=(13.5, 14))
    fig.suptitle(
        f"Step 1 v2 — IMU gap 波形対比　{tb}（素の ACT）vs {tc}（補正あり）\n"
        f"θ = (A_prox {th.get('a_prox',0)*1e3:+.3f}, A_wrist {th.get('a_wrist',0)*1e3:+.3f} mrad, "
        f"τ {th.get('tau',0):.2f} frame)　|　時間的に隣接した 2 本（ドリフト最小）"
        f"　|　G も H も未使用", fontsize=11)

    ax[0].plot(t, igb[:n, J], color="0.25", lw=1.0, label="ideal（ACT 指令レンダリング）")
    ax[0].plot(t, rgb[:n, J], color="#3b7dd8", lw=1.0, label=f"real {tb}（素の ACT）")
    ax[0].set_title(f"① 素の ACT　real vs ideal　　L = {Lb:.5f}", fontsize=10)

    ax[1].plot(t, igc[:n, J], color="0.25", lw=1.0, label="ideal")
    ax[1].plot(t, rgc[:n, J], color="#d1495b", lw=1.0, label=f"real {tc}（補正あり）")
    ax[1].set_title(f"② 補正あり　real vs ideal　　L = {Lc:.5f}"
                    f"（この 1 組では {Lc/Lb*100-100:+.1f}% —— ⚠ 単一組の揺らぎ, "
                    f"効果ではない。⑤ の全平均を見ること）", fontsize=10)

    ax[2].plot(t, gb[:n, J], color="#3b7dd8", lw=1.0, label="素の ACT の gap")
    ax[2].plot(t, gc[:n, J], color="#d1495b", lw=1.0, label="補正ありの gap")
    for k in range(100, n, 100):
        ax[2].axvline(k / FPS, color="gray", ls=":", lw=.8)
    ax[2].set_title(f"③ gap = real − ideal（{BAND[0]}–{BAND[1]} Hz 帯域通過, {AXNAME[J]}）"
                    f"　—— 尖りは全て **chunk 継ぎ目**（点線, 5 s 毎）", fontsize=10)

    # ④ 実際に加えた δa（継ぎ目パルス）
    for j in range(6):
        pk = np.abs(dac[:n, j]).max() * 1e3
        ax[3].plot(t, dac[:n, j] * 1e3, lw=1.1 if pk > .05 else .7,
                   label=f"j{j+1} (max {pk:.2f})" if pk > .05 else None,
                   color=plt.cm.tab10(j), alpha=.9)
    for k in range(100, n, 100):                      # chunk 継ぎ目
        ax[3].axvline(k / FPS, color="gray", ls=":", lw=.8)
    ax[3].axhline(0, color="k", lw=.6)
    ax[3].set_title(f"④ 実際に加えた δa（mrad, 点線 = chunk 継ぎ目）　"
                    f"最大 {np.abs(dac[:n]).max()*1e3:.3f} mrad　"
                    f"—— ③ の gap（rad/s）とは**桁が違う**", fontsize=10)
    ax[3].set_ylabel("δa (mrad)")

    for i in range(4):
        ax[i].set_xlim(0, tmax); ax[i].grid(alpha=.3); ax[i].legend(fontsize=8, ncol=2)
        if i < 3:
            ax[i].set_ylabel(f"{AXNAME[J]} (rad/s)" if i < 2 else f"gap {AXNAME[J]} (rad/s)")
    ax[3].set_xlabel("時間 (s)")

    # ⑤ スペクトル（全エピソード平均）
    accb, accc, fq = [], [], None
    for rd, o, tag, isb in seq:
        f = OUT / f"{tag}_ep0.json"
        if not f.exists():
            continue
        try:
            _, _, g, _, _ = load_ep(tag, Rc)
        except Exception:                                       # noqa: BLE001
            continue
        fq, P = psd(g)
        (accb if isb else accc).append(P)
    m = min(min(len(x) for x in accb), min(len(x) for x in accc))
    Pb = np.mean([x[:m] for x in accb], axis=0)
    Pc = np.mean([x[:m] for x in accc], axis=0)
    ax[4].semilogy(fq[:m], Pb, color="#3b7dd8", lw=1.2, label=f"素の ACT（{len(accb)} 本平均）")
    ax[4].semilogy(fq[:m], Pc, color="#d1495b", lw=1.2, label=f"補正あり（{len(accc)} 本平均）")
    ax[4].axvspan(BAND[0], BAND[1], color="0.85", zorder=0)
    ax[4].set_xlim(0, 10); ax[4].set_xlabel("周波数 (Hz)"); ax[4].set_ylabel("PSD")
    kb = (fq[:m] >= BAND[0]) & (fq[:m] <= BAND[1])
    ax[4].set_title(f"⑤ gap スペクトル（全エピソード平均, 灰 = 評点帯 "
                    f"{BAND[0]}–{BAND[1]} Hz）　—— 帯内積分 素 {Pb[kb].sum():.2f} / "
                    f"補正 {Pc[kb].sum():.2f}（**{Pc[kb].sum()/Pb[kb].sum()*100-100:+.1f}%**）",
                    fontsize=10)
    ax[4].legend(fontsize=8); ax[4].grid(alpha=.3)

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = OUT / "spsa_v2_curves.png"
    fig.savefig(out, dpi=130)
    print(f"→ {out.relative_to(ROOT)}")
    print(f"  対比: {tb}（L={Lb:.5f}） vs {tc}（L={Lc:.5f}, {Lc/Lb*100-100:+.1f}%）")
    print(f"  δa 最大 {np.abs(dac[:n]).max()*1e3:.3f} mrad;  "
          f"gap RMS {np.sqrt(np.square(gc).mean()):.4f} rad/s")
    k = (fq[:m] >= BAND[0]) & (fq[:m] <= BAND[1])
    print(f"  評点帯内 PSD 積分: 素 {Pb[k].sum():.4f} / 補正 {Pc[k].sum():.4f}"
          f"  ({Pc[k].sum()/Pb[k].sum()*100-100:+.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
