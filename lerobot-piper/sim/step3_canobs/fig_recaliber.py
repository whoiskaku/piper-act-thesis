#!/usr/bin/env python3
"""**pp2 诊断集：抗混叠口径修正前后的波形对比**（每次动 G 必出的图，本次是动口径）。

三条线：**実機 IMU（青）· G-q-v1（紫）· 真の関節から描画 = 天井（橙）**

上段 = 0.5–3 Hz、下段 = **3–9 Hz**。左右 = 2 条診断集。
橙は「200 Hz の真の関節軌跡を剛体センサ写像に通したら何が出るか」＝到達可能な上限。

この図の要点：**3–9 Hz で橙は実機を追えているのに紫は追えていない。**
⇒ 3–9 Hz の不足は**センサ層ではなく G の関節軌跡予測**にある。

口径（**必ず併記**）:
  · 実機側は **100 Hz から 9 Hz 抗混叠低通 → 20 Hz**（従来は無フィルタ間引き）
  · 時基は TSND **デバイス ms**（ステップ恒 10、欠損ゼロ）と CAN の**カーネル時刻**。
    この 2 集は 20 Hz 制御ループ自体が劣化していた（中央値 51.2 ms, p99 84.5 ms）ため、
    落盤された 20 Hz フィールドは**使っていない**。
  · IMU 装着回転は**集ごとに正交 Procrustes で推定**（新セッションの標定が無いため）。
  · pp2 診断集 2 条（同一セッション 2026-09-01）。**17 集の開発基準集とは並べられない。**

用法:
    python -u sim/step3_canobs/fig_recaliber.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/step3_canobs"
ADV = ROOT / "outputs/step2_g/advisor"
FPS = 20.0
C_REAL, C_GQ, C_KIN = "#1f6fb2", "#8b2fb0", "#e07b1a"
BANDS = ((0.5, 3.0), (3.0, 9.0))


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from scipy.signal import firwin
    for _f in ("Noto Sans CJK JP", "Noto Sans CJK SC", "IPAexGothic"):
        if any(_f in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = _f
            break
    plt.rcParams["axes.unicode_minus"] = False

    def bp(x, lo, hi):
        k = firwin(41, [lo, hi], fs=FPS, pass_zero=False)
        return np.stack([np.convolve(x[:, j], k, mode="same")
                         for j in range(x.shape[1])], 1)

    def rho(a, b):
        return float((a * b).sum()) / max(float(np.sqrt((a ** 2).sum() * (b ** 2).sum())), 1e-30)

    from scipy.linalg import orthogonal_procrustes
    z = np.load(OUT / "recaliber_traces.npz")
    eps = sorted({k.rsplit("_", 2)[0] for k in z.files if k.endswith("_A_aa")})

    def _r2(nm):                       # G-q-v1 の 3–9 Hz ρ²(自分の最適シフト)
        A0, S0 = z[f"{nm}_A_aa"], z[f"{nm}_gq20"]
        best = 0.0
        for L in range(4):
            k = min(len(A0) - L, len(S0)); A_, S_ = A0[L:L + k], S0[:k]
            Sr = S_ @ orthogonal_procrustes(S_, A_)[0]
            a_, s_ = bp(A_, *BANDS[1]), bp(Sr, *BANDS[1])
            best = max(best, float((a_ * s_).sum() ** 2
                                   / max((a_ * a_).sum() * (s_ * s_).sum(), 1e-30)))
        return best
    order = sorted(eps, key=_r2, reverse=True)         # 良い順
    # 慣例どおり **中央値集と最難集**（G-q-v1 の 3–9 Hz が中央 / 最悪）
    show = [(order[len(order) // 2], "中央値"), (order[-1], "最難")]
    fig, axes = plt.subplots(2, 2, figsize=(16, 8.2))
    for col, (nm, kind) in enumerate(show):
        A0 = z[f"{nm}_A_aa"]
        # ── **シフトは系列ごとに最適化する** ────────────────────────────
        # 20 Hz 出力では 1 サンプル = 50 ms、3–9 Hz(周期 111–333 ms)では 54–162°
        # の位相ずれ。実測: 天井は L=0 で ρ²=0.66-0.68 だが L=1 だと 0.008 に落ちる。
        # 共通 L を使うと天井を不当に低く見せる —— 一度踏んだ誤りの再発を防ぐ。
        # (天井 L=0 / G-q-v1 L=1 ⇒ G は真の関節より約 50 ms 遅れている。)
        # **装着回転も保存トレースには未適用**なので、ここで系列ごとに推定して掛ける。
        def _fit(S0):
            best = None
            for L in range(4):
                k = min(len(A0) - L, len(S0))
                A_, S_ = A0[L:L + k], S0[:k]
                Sr = S_ @ orthogonal_procrustes(S_, A_)[0]
                a_, s_ = bp(A_, *BANDS[1]), bp(Sr, *BANDS[1])
                v = float((a_ * s_).sum() ** 2 / max((a_ * a_).sum() * (s_ * s_).sum(), 1e-30))
                if best is None or v > best[0]:
                    best = (v, L, Sr)
            return best[1], best[2]
        Lg, G = _fit(z[f"{nm}_gq20"])
        Lk, K = _fit(z[f"{nm}_kin20"])
        n = min(len(G), len(K), len(A0) - max(Lg, Lk))
        A = A0[:n]                              # 各系列は自分の L で既に整列済み
        G, K = G[:n], K[:n]
        Ag, Ak = A0[Lg:Lg + n], A0[Lk:Lk + n]
        j = int(np.argmax((bp(Ag, *BANDS[1]) ** 2).sum(0)))
        for row, (lo, hi) in enumerate(BANDS):
            ax = axes[row][col]
            ag, ak = bp(Ag, lo, hi), bp(Ak, lo, hi)
            g_, k_ = bp(G, lo, hi), bp(K, lo, hi)
            sg = float((ag[:, j] * g_[:, j]).sum() / max((g_[:, j] ** 2).sum(), 1e-30))
            sk = float((ak[:, j] * k_[:, j]).sum() / max((k_[:, j] ** 2).sum(), 1e-30))
            t = np.arange(len(ag)) / FPS
            rg, rk = rho(ag[:, j], g_[:, j]), rho(ak[:, j], k_[:, j])
            if row == 1:                     # 3–9 Hz は 30 s 全部だと潰れて読めない
                e = np.convolve(ak[:, j] ** 2, np.ones(40), "same")
                c = int(np.argmax(e)); w = slice(max(0, c - 40), c + 40)
                ag, ak, g_, k_, t = ag[w], ak[w], g_[w], k_[w], t[w]
            ax.plot(t, ak[:, j], color=C_REAL, lw=1.9, label="実機 IMU（抗混叠）", zorder=3)
            ax.plot(t, k_[:, j] * sk, color=C_KIN, lw=1.5, alpha=.9, zorder=1,
                    label=f"真の関節 200 Hz から描画 = 天井 (L={Lk})")
            ax.plot(t, g_[:, j] * sg, color=C_GQ, lw=1.4, zorder=2,
                    label=f"G-q-v1 (L={Lg})")
            ax.set_title(f"{kind} {nm}  {lo}–{hi} Hz  (軸 {'xyz'[j]})"+ ("  ※最も強い 4 s を拡大" if row else "") + "    "
                         f"ρ: G-q-v1 = {rg:+.2f} / 天井 = {rk:+.2f}", fontsize=10.5)
            ax.set_ylabel("角速度 [rad/s]", fontsize=9)
            ax.grid(alpha=.25); ax.tick_params(labelsize=8)
            if row == 1:
                ax.set_xlabel("時間 [s]", fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=8.5, loc="upper right", framealpha=.92)
    fig.suptitle("pp2 診断集：抗混叠口径を揃えた後    "
                 "下段 3–9 Hz：**橙(真の関節)は追える・紫(G-q-v1)は追えない**（各系列は自分の最適シフト）",
                 fontsize=13.5, y=0.985)
    fig.text(0.5, 0.005,
             "集計 ρ²(3–9 Hz, 4 集/2 セッション): G-q-v1 = 0.235 / 天井 = 0.692  ⇒ 不足の 0.457 は"
             "**関節軌跡予測**、0.308 がセンサ・可観測層。"
             "口径: 実機側 100 Hz→9 Hz 低通→20 Hz（従来は無フィルタ間引き）。"
             "時基はデバイス ms とカーネル時刻（20 Hz ループは記録器で劣化, "
             "落盤 20 Hz フィールドは不使用）。装着回転は集ごとに Procrustes 推定。"
             "pp2 診断集 4 条・2 セッション、17 集の開発基準集とは並置不可。",
             ha="center", fontsize=8.4, color="#333")
    fig.tight_layout(rect=(0, 0.028, 1, 0.965))
    for p in (OUT, ADV):
        p.mkdir(parents=True, exist_ok=True)
        fig.savefig(p / "fig7_recaliber_pp2.png", dpi=170)
    print(f"  → {OUT/'fig7_recaliber_pp2.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
