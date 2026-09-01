#!/usr/bin/env python3
"""**带权目标重辨识的否定结果** —— 每次动 G 必出的 pp2 波形图（本次是判否）。

上段 = 腕部陀螺 3–9 Hz，下段 = **最难关节 j3 的关节角 3–9 Hz**（失败发生的层）。
线：**実機（青）· G-q-v1（紫）· 带权目标の新 θ（緑・破線）· 天井 = 真の関節（橙）**

要点：目标函数**明确要求了** 3–9 Hz 之后，紫と緑はほぼ重なる ——
**23 個の伺服パラメータではこの帯域は動かせない**（前六轮は「要求していなかった」
可能性が残っていたが、今回それは潰れた）。

口径:
  · 実機側 100 Hz → 9 Hz 抗混叠低通 → 20 Hz（従来は無フィルタ間引き）
  · 時基は TSND デバイス ms と CAN カーネル時刻。落盤 20 Hz フィールドは不使用。
  · 各系列は**自分の最適シフト**（共通 L を使うと天井が不当に低く出る）。
  · pp2 診断集 4 条 / 2 セッション。**17 集の開発基準集とは並置不可。**

用法:
    python -u sim/step2_g/fig_band_negative.py
"""
from __future__ import annotations
import gc, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g", ROOT / "sim/step3_canobs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DIAG = [(a, e) for a in ("A1", "A3") for e in ("ep0", "ep1")]
FS_I, FPS = 100.0, 20.0
OUT = ROOT / "outputs/step2_g/optuna_band"
ADV = ROOT / "outputs/step2_g/advisor"
C_REAL, C_V1, C_NEW, C_KIN = "#1f6fb2", "#8b2fb0", "#2e9e4f", "#e07b1a"
JB = (3.0, 9.0)


def main() -> int:
    gc.disable()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from scipy.signal import butter, filtfilt, firwin
    from scipy.linalg import orthogonal_procrustes
    import g_q_v1 as GQ
    from ceiling_200hz import load_streams, kin_gyro_fs
    for _f in ("Noto Sans CJK JP", "Noto Sans CJK SC", "IPAexGothic"):
        if any(_f in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = _f
            break
    plt.rcParams["axes.unicode_minus"] = False

    lp = lambda x, fc, fs: filtfilt(*butter(4, fc / (fs / 2)), x, axis=0)

    def bp(x, lo, hi):
        k = firwin(41, [lo, hi], fs=FPS, pass_zero=False)
        x = np.atleast_2d(x.T).T
        return np.stack([np.convolve(x[:, j], k, mode="same")
                         for j in range(x.shape[1])], 1)

    def fit(A0, S0):
        """→ (最適シフト, 回転済み S, その ρ²)。3–9 Hz で最適化。"""
        best = None
        for L in range(4):
            n = min(len(A0) - L, len(S0))
            A, S = A0[L:L + n], S0[:n]
            Sr = S @ orthogonal_procrustes(S, A)[0] if S.shape[1] == 3 else S
            a, s = bp(A, *JB), bp(Sr, *JB)
            v = float((a * s).sum() ** 2 / max((a * a).sum() * (s * s).sum(), 1e-30))
            if best is None or v > best[2]:
                best = (L, Sr, v)
        return best

    TH = np.array(json.loads((OUT / "gate.json").read_text())["theta"])
    D = {}
    for arm, ep in DIAG:
        d = json.loads((ROOT / f"outputs/probe/replay_{ep}_{arm}.json").read_text())
        g100, q200, tg_i, tg_q = load_streams(ep, arm)
        cmd = np.asarray(d["cmd"], float)
        ts, tv = np.asarray(d["t_send"], float), np.asarray(d["t"], float)
        A20, tg20 = lp(g100, 9.0, FS_I)[::5], tg_i[::5]
        qt = np.stack([np.interp(tg20, tg_q, lp(q200, 9.0, 200.0)[:, j])
                       for j in range(6)], 1)
        S = {}
        for tag, th in (("v1", None), ("new", TH)):
            qf = np.radians(GQ.predict_qpos(cmd, t_send=ts, t_read=tv) if th is None
                            else GQ.predict_with_theta(cmd, th, t_send=ts, t_read=tv)[0])
            qi = np.stack([np.interp(tg20, tv, qf[:, j]) for j in range(6)], 1)
            S[tag] = (qi, kin_gyro_fs(qi, FPS))
        D[f"{arm}_{ep}"] = {"A": A20, "qt": qt, **S,
                            "kin": kin_gyro_fs(qt, FPS)}

    order = sorted(D, key=lambda k: fit(D[k]["A"], D[k]["v1"][1])[2], reverse=True)
    show = [(order[len(order) // 2], "中央値"), (order[-1], "最難")]
    fig, axes = plt.subplots(2, 2, figsize=(16, 8.4))
    for col, (nm, kind) in enumerate(show):
        E = D[nm]
        # ── 上段: 腕部陀螺 3–9 Hz ────────────────────────────────────
        ax = axes[0][col]
        Lk, K, rk = fit(E["A"], E["kin"])
        L1, G1, r1 = fit(E["A"], E["v1"][1])
        L2, G2, r2 = fit(E["A"], E["new"][1])
        j = int(np.argmax((bp(E["A"], *JB) ** 2).sum(0)))
        n = min(len(K), len(G1), len(G2))
        aK, aG = bp(E["A"][Lk:Lk + n], *JB), bp(E["A"][L1:L1 + n], *JB)
        aN = bp(E["A"][L2:L2 + n], *JB)
        k_, g1, g2 = bp(K[:n], *JB), bp(G1[:n], *JB), bp(G2[:n], *JB)
        sc = lambda a, s: float((a[:, j] * s[:, j]).sum()
                                / max((s[:, j] ** 2).sum(), 1e-30))
        e = np.convolve(aK[:, j] ** 2, np.ones(40), "same")
        c = int(np.argmax(e)); w = slice(max(0, c - 40), c + 40)
        t = np.arange(n)[w] / FPS
        ax.plot(t, aK[w, j], color=C_REAL, lw=1.9, label="実機 IMU（抗混叠）", zorder=4)
        ax.plot(t, k_[w, j] * sc(aK, k_), color=C_KIN, lw=1.5, alpha=.9, zorder=1,
                label=f"天井 = 真の関節 200 Hz (L={Lk})")
        ax.plot(t, g1[w, j] * sc(aG, g1), color=C_V1, lw=1.4, zorder=3,
                label=f"G-q-v1 (L={L1})")
        ax.plot(t, g2[w, j] * sc(aN, g2), color=C_NEW, lw=1.6, ls="--", zorder=2,
                label=f"带权目标の新 θ (L={L2})")
        ax.set_title(f"{kind} {nm}  腕部陀螺 3–9 Hz (軸 {'xyz'[j]}, 最も強い 4 s)    "
                     f"ρ²: 天井 {rk:.3f} / v1 {r1:.3f} / 新 {r2:.3f}", fontsize=10.5)
        ax.set_ylabel("角速度 [rad/s]", fontsize=9)
        ax.grid(alpha=.25); ax.tick_params(labelsize=8)
        if col == 0:
            ax.legend(fontsize=8.3, loc="upper right", ncol=2, framealpha=.92)
        # ── 下段: j3 関節角 3–9 Hz（失敗が起きている層）────────────────
        ax = axes[1][col]
        jj = 2                                       # j3 = 最も悪い関節
        qt_, q1, q2 = (bp(E["qt"][:, jj:jj + 1], *JB), bp(E["v1"][0][:, jj:jj + 1], *JB),
                       bp(E["new"][0][:, jj:jj + 1], *JB))
        rr = lambda a, b: float((a * b).sum() ** 2
                                / max((a * a).sum() * (b * b).sum(), 1e-30))
        ax.plot(t, np.degrees(qt_[w, 0]), color=C_REAL, lw=1.9,
                label="真の関節角 200 Hz", zorder=3)
        ax.plot(t, np.degrees(q1[w, 0]), color=C_V1, lw=1.4, label="G-q-v1", zorder=2)
        ax.plot(t, np.degrees(q2[w, 0]), color=C_NEW, lw=1.6, ls="--",
                label="带权目标の新 θ", zorder=1)
        ax.set_title(f"{kind} {nm}  **j3 関節角** 3–9 Hz    "
                     f"ρ²: v1 {rr(qt_,q1):.3f} / 新 {rr(qt_,q2):.3f}", fontsize=10.5)
        ax.set_ylabel("関節角 [deg]", fontsize=9)
        ax.set_xlabel("時間 [s]", fontsize=9)
        ax.grid(alpha=.25); ax.tick_params(labelsize=8)
        if col == 0:
            ax.legend(fontsize=8.3, loc="upper right", framealpha=.92)
    fig.suptitle("带权目标重辨识：**判否** —— 目标函数明确要求 3–9 Hz 之后，"
                 "紫と緑はほぼ重なる（23 個の伺服パラメータでは動かせない）",
                 fontsize=13.5, y=0.985)
    fig.text(0.5, 0.005,
             "訓練 L_39 は 0.5416 → 0.5090（−6%）にしか動かず、その代償に L_broad が "
             "0.0307 → 0.0451（+47%）悪化。診断集の 3–9 Hz ρ² は 0.204 → 0.202。"
             "予算固定 3 seed × 1008 trial、選参は訓練損失のみ、診断集は選参に不使用。"
             "口径: 実機側 100 Hz→9 Hz 低通→20 Hz、時基はデバイス ms/カーネル時刻、"
             "各系列は自分の最適シフト、pp2 診断集 4 条・2 セッション（17 集とは並置不可）。",
             ha="center", fontsize=8.4, color="#333")
    fig.tight_layout(rect=(0, 0.028, 1, 0.965))
    for p in (OUT, ADV):
        p.mkdir(parents=True, exist_ok=True)
        fig.savefig(p / "fig8_band_negative.png", dpi=170)
    print(f"  → {OUT/'fig8_band_negative.png'}")
    return 0


if __name__ == "__main__":
    _c = main()
    sys.stdout.flush()
    import os
    os._exit(_c)
