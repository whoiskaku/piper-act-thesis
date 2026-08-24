#!/usr/bin/env python3
"""Step 1 v2 SPSA の gap 対比図（進行中スナップショット）。

口径は H1 主指標と同一：`gap(t) = real[t+3] − ideal_cal[t]`、帯域 0.5–9 Hz、
ジャイロ 3 軸、`L = mean_t‖·‖²`。ideal は `chunks[].raw` から逐 chunk 運動学レンダリング。
**G も H も使っていません。**

四連図:
  ① L の時系列 —— 素の ACT（基線）と補正ありを分けて、ドリフトと一緒に見る
  ② ドリフト除去後の分布 —— 「補正で下がったか」に直接答える
  ③ θ の軌跡 —— SPSA が中心をどこへ動かしたか
  ④ 効果量の区間 —— 現在 / 40 ラウンド投影 / 界内最適 を H1 門槛と並べる

用法: python sim/plot_spsa_v2_gap.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/spsa_v2"
N_ROUNDS = 40
H1_BAR = 0.15


def load():
    s = json.loads((OUT / "state.json").read_text())
    rows = []
    for b in s["baselines"]:
        rows.append({"rd": b["round"], "o": -1, "tag": b["tag"], "L": b["L"],
                     "x": None, "base": True})
    for h in s["history"]:
        for i, side in enumerate(h["order"]):
            rows.append({"rd": h["round"], "o": i, "tag": f"R{h['round']:02d}{side}",
                         "L": h["L_plus"] if side == "p" else h["L_minus"],
                         "x": h["x_plus"] if side == "p" else h["x_minus"],
                         "base": False})
    # **実際の時刻**で並べ、最大時間間隔をセッション境界とする。
    # 5 日超の中断は L に**階段状**の変化を生む ⇒ 線形 t では吸収できず、
    # 放置すると θ 係数がその段差を吸ってしまう（偽信号）。
    for r in rows:
        f = OUT / f"{r['tag']}_ep0.json"
        r["mt"] = f.stat().st_mtime if f.exists() else None
    rows = [r for r in rows if r["mt"] is not None]
    rows.sort(key=lambda r: r["mt"])
    mt = np.array([r["mt"] for r in rows])
    k = int(np.argmax(np.diff(mt)))
    for i, r in enumerate(rows):
        r["sess"] = 0 if i <= k else 1
    return s, rows, float(np.diff(mt)[k])


def fit(rows):
    """log L ~ 1 + t + **session** + x1 + x2 + x3（補正エピソードのみ）。

    `session` ダミーは必須。5 日超の中断による**階段状**の水準変化を線形 t は
    吸収できず、抜くと θ 係数がその段差を吸って偽の有意性が出る（実測: これを
    抜くと x1 が t=+2.10、入れると +1.52 に落ちる）。
    """
    t = np.arange(len(rows), dtype=float)
    y = np.log(np.array([r["L"] for r in rows]))
    isc = np.array([not r["base"] for r in rows])
    sess = np.array([float(r["sess"]) for r in rows])
    X = np.array([r["x"] if r["x"] is not None else [0., 0., 0.] for r in rows])
    A = np.column_stack([np.ones(int(isc.sum())), t[isc], sess[isc], X[isc]])
    beta, *_ = np.linalg.lstsq(A, y[isc], rcond=None)
    res = y[isc] - A @ beta
    cov = np.sum(res ** 2) / (len(y[isc]) - A.shape[1]) * np.linalg.inv(A.T @ A)
    return t, y, isc, X, beta, cov, res, sess


def effect(beta, cov, xa, xb):
    """x=xa → x=xb に動かしたときの Δlog L と SE（θ 項のみ）。"""
    c = np.concatenate([[0., 0., 0.], np.asarray(xb) - np.asarray(xa)])   # 1, t, session
    return float(c @ beta), float(np.sqrt(c @ cov @ c))


def main() -> int:
    s, rows, brk = load()
    t, y, isc, X, beta, cov, res, sess = fit(rows)
    L = np.array([r["L"] for r in rows])
    nb, nc = int((~isc).sum()), int(isc.sum())
    done = s["round"]

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Step 1 v2 SPSA — IMU gap 対比（進行中 {done}/{N_ROUNDS} ラウンド, "
        f"実機 {s['episodes_used']}+{s['episodes_wasted']} = "
        f"{s['episodes_used']+s['episodes_wasted']}/100 エピソード）\n"
        f"gap = real[t+3] − ideal_cal[t], 0.5–9 Hz, ジャイロ3軸;  L = mean‖·‖²  "
        f"— G も H も未使用;  回帰は**会話ダミー込み**（{brk/86400:.1f} 日の中断）", fontsize=11)

    # ── ① 時系列 ─────────────────────────────────────────────
    a = ax[0, 0]
    a.plot(t[isc], L[isc], "o-", ms=4, lw=.8, color="#3b7dd8", alpha=.75,
           label=f"補正あり (n={nc})")
    a.plot(t[~isc], L[~isc], "s", ms=9, color="#d1495b", zorder=5,
           label=f"素の ACT 基線 (n={nb})")
    sl, ic = np.polyfit(t[~isc], np.log(L[~isc]), 1)
    a.plot(t, np.exp(sl * t + ic), "--", color="#d1495b", lw=1.6,
           label=f"基線ドリフト {sl*100:+.2f}%/ep")
    a.set_xlabel("エピソード（時系列順）"); a.set_ylabel("L  (rad/s)²")
    a.set_title(f"① L の時系列 — 主な変動は θ ではなく**時間**", fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=.3)

    # ── ② ドリフト除去後の分布 ────────────────────────────────
    a = ax[0, 1]
    # 会話ごとに基線水準を引く（段差を残すと分布が二峰になる）
    off = np.zeros(len(rows))
    for g in (0, 1):
        mb = (~isc) & (sess == g)
        off[sess == g] = np.log(L[mb]).mean() if mb.sum() else np.log(L[~isc]).mean()
    rc = y[isc] - off[isc]
    rb = y[~isc] - off[~isc]
    allr = np.concatenate([rc, rb])
    lim = float(np.percentile(np.abs(allr), 96)) * 1.15
    bins = np.linspace(-lim, lim, 24)
    nout = int((np.abs(allr) > lim).sum())
    a.hist(np.clip(rc, -lim, lim) * 100, bins=bins * 100, alpha=.6, color="#3b7dd8",
           label=f"補正あり (n={nc})")
    a.hist(np.clip(rb, -lim, lim) * 100, bins=bins * 100, alpha=.6, color="#d1495b",
           label=f"基線 (n={nb})")
    if nout:
        a.text(.02, .78, f"（外れ {nout} 本は端に丸め）", transform=a.transAxes,
               fontsize=7, color="gray")
    tt = stats.ttest_1samp(rc, 0.0)
    sem = rc.std(ddof=1) / np.sqrt(len(rc))
    a.axvline(0, color="k", lw=1)
    a.axvline(rc.mean() * 100, color="#3b7dd8", lw=2,
              label=f"補正の平均 {rc.mean()*100:+.2f}%")
    a.axvspan((rc.mean() - 1.96 * sem) * 100, (rc.mean() + 1.96 * sem) * 100,
              color="#3b7dd8", alpha=.15, label="95% CI")
    a.set_xlabel("会話内基線からの残差  Δlog L (%)"); a.set_ylabel("エピソード数")
    a.set_title(f"② 会話内基線を除去後 — 補正 {rc.mean()*100:+.1f}%, p={tt.pvalue:.3f}", fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=.3)

    # ── ③ θ の軌跡 ───────────────────────────────────────────
    a = ax[1, 0]
    C = np.array(s["centers"])
    for j, (nm, col) in enumerate((("A_prox", "#2a9d8f"), ("A_wrist", "#e76f51"),
                                   ("τ", "#8d5bd6"))):
        v = C[:, j] * 2 if j < 2 else 6 + C[:, j] * 4
        a2 = a if j < 2 else a.twinx()
        a2.plot(np.arange(1, len(C) + 1), v, "-", color=col, lw=1.8,
                label=f"{nm} (mrad)" if j < 2 else "τ (frame, 右軸)")
        if j == 2:
            a2.set_ylabel("τ (frame)", color=col); a2.tick_params(axis="y", labelcolor=col)
            a2.set_ylim(2, 10)
    a.axhline(0, color="k", lw=.8, ls=":")
    a.axvline(10.5, color="gray", ls="--", lw=1)
    lo_, hi_ = a.get_ylim()
    a.text(10.8, lo_ + (hi_ - lo_) * .04, "← 第1段階 (τ=6 固定)  |  第2段階 →",
           fontsize=7.5, color="gray")
    a.set_xlabel("ラウンド"); a.set_ylabel("A (mrad)")
    a.set_title("③ SPSA が動かした θ の軌跡", fontsize=10)
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right"); a.grid(alpha=.3)

    # ── ④ 効果量の区間 ───────────────────────────────────────
    a = ax[1, 1]
    x0 = np.zeros(3)
    xc = np.array(s["x"])
    ak = lambda k: 0.35 / (k + 6) ** 0.602
    rem = sum(ak(k) for k in range(done, N_ROUNDS))
    g = np.array([np.mean([h["g"][j] for h in s["history"]
                           if not (j == 2 and h["round"] < 10)]) for j in range(3)])
    x40 = np.clip(xc - rem * g, -1, 1)
    xopt = np.clip(-np.sign(beta[3:6]), -1, 1)
    items = [(f"現在 ({done} ラウンド)", xc), (f"{N_ROUNDS} ラウンド投影", x40),
             ("界内最適（線形外挿）", xopt)]
    for i, (nm, xx) in enumerate(items):
        d, sd = effect(beta, cov, x0, xx)
        # Δlog L → L の相対変化
        lo, hi = (np.exp(d - 1.96 * sd) - 1) * 100, (np.exp(d + 1.96 * sd) - 1) * 100
        pt = (np.exp(d) - 1) * 100
        a.plot([lo, hi], [i, i], "-", lw=3, color="#3b7dd8", alpha=.5)
        a.plot(pt, i, "o", ms=10, color="#3b7dd8")
        a.text(pt, i - .22, f"{pt:+.1f}%", ha="center", va="bottom",
               fontsize=9.5, color="#1f4e96", fontweight="bold")
        a.text(hi + 1.5, i + .18, f"[{lo:+.0f}, {hi:+.0f}]", va="center",
               fontsize=7.5, color="gray")
    a.axvline(0, color="k", lw=1)
    a.axvline(-H1_BAR * 100, color="#d1495b", lw=2, ls="--",
              label=f"H1 門槛  L −{H1_BAR*100:.0f}%")
    a.set_yticks(range(len(items))); a.set_yticklabels([n for n, _ in items], fontsize=9)
    a.set_xlabel("L の相対変化 (%)  ← 改善      悪化 →")
    a.set_title("④ 効果量と 95% CI — いずれも 0 を跨ぐ", fontsize=10)
    a.legend(fontsize=8, loc="upper left"); a.grid(alpha=.3, axis="x")
    a.set_ylim(len(items) - .5, -.6)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT / "spsa_v2_gap.png"
    fig.savefig(out, dpi=140)
    print(f"→ {out.relative_to(ROOT)}")

    print(f"\n数値（図と同一）:")
    print(f"  基線 n={nb} 中央値 {np.median(L[~isc]):.5f} | "
          f"補正 n={nc} 中央値 {np.median(L[isc]):.5f}")
    print(f"  基線ドリフト {sl*100:+.2f}%/ep;  会話境界 {brk/86400:.2f} 日")
    for nm, i in (("t", 1), ("session", 2), ("x1", 3), ("x2", 4), ("x3", 5)):
        se = np.sqrt(cov[i, i])
        print(f"    {nm:<8} {beta[i]:+.4f} ± {se:.4f}  t={beta[i]/se:+.2f}")
    print(f"  ドリフト除去後の補正残差 {rc.mean()*100:+.2f}% ± {sem*100:.2f}% (SEM), p={tt.pvalue:.3f}")
    for nm, xx in items:
        d, sd = effect(beta, cov, x0, xx)
        print(f"  {nm:<22} L {(np.exp(d)-1)*100:+6.1f}%  "
              f"95%CI [{(np.exp(d-1.96*sd)-1)*100:+.1f}%, {(np.exp(d+1.96*sd)-1)*100:+.1f}%]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
