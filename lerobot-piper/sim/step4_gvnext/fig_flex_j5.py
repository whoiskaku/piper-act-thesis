#!/usr/bin/env python3
"""**j5 下游有效柔性模态的 pp2 波形对比**（每次动 G 必出）—— 开发验证集。

五条线：実機（青）· G-q-v1（紫）· **G-2（緑）** · **G-2F = +j5 柔性铰（红虚）** ·
天井=真の関節（橙）。上段 0.5–3 Hz / 下段 3–9 Hz（最も強い 4 s を拡大）。

结论图：G-2F 与 G-2 **几乎重合** ⇒ 柔性铰对完整模型无增量。
"""
import sys, json, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
for p in (ROOT, ROOT/"sim", ROOT/"sim/step2_g", ROOT/"sim/step3_canobs", HERE):
    sys.path.insert(0, str(p))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from data import bp, valid_files, best_fit, FPS
from train_rq import fir_design
from ceiling_200hz import kin_gyro_fs
from stage2_flex import build, S2, OUT
import gc; gc.disable()
for _f in ("Noto Sans CJK JP", "Noto Sans CJK SC", "IPAexGothic"):
    if any(_f in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False
C_R, C_V1, C_NX, C_FX, C_K = "#1f6fb2", "#8b2fb0", "#2e9e4f", "#d62728", "#e07b1a"

from g_q_v1 import THETA_STAR
from sim.step2_gap_baseline import MODEL
from structure_compare import flex_xml
g = json.loads((OUT/"grid.json").read_text())
sel = max([r for r in g["rows"] if r["enc_ok"]], key=lambda r: r["r39"])
zq = np.load(ROOT/"outputs/step4_gvnext/rq/rq_fir.npz")
sq = json.load(open(ROOT/"outputs/step4_gvnext/rq/select.json"))
zs = np.load(ROOT/"outputs/step4_gvnext/rs/rs_fir.npz")
ss = json.load(open(ROOT/"outputs/step4_gvnext/rs/select.json"))
zf = np.load(OUT/"rs_flex.npz")
xml = flex_xml(4, sel["w"], sel["z"], THETA_STAR)

D = {}
for e in build(valid_files(S2, "replay_pp2f*.json"), MODEL, xml, THETA_STAR, zq, sq):
    kh = kin_gyro_fs(e["qh"], FPS)[:len(e["Y_new"])]
    D[e["name"]] = {
        "A": e["A"], "v1": kin_gyro_fs(e["qb"], FPS)[:len(e["A"])],
        "nx": kh + fir_design(e["Xs"], ss["nlag"]) @ zs["W"],
        "fx": kh + e["D"] + fir_design(e["Xs"], int(zf["nlag"])) @ zf["W"],
        "ceil": e["A"] - e["Y_old"]}
r2 = {k: best_fit(v["A"], v["nx"], 3.0, 9.0, rotate=False)[1] for k, v in D.items()}
o = sorted(D, key=lambda k: r2[k], reverse=True)
show = [(o[len(o)//2], "中央値"), (o[-1], "最難")]

fig, ax = plt.subplots(2, 2, figsize=(16, 8.2))
for c, (nm, kind) in enumerate(show):
    E = D[nm]; j = int(np.argmax((bp(E["A"], 3.0, 9.0)**2).sum(0)))
    for r, (lo, hi) in enumerate(((0.5, 3.0), (3.0, 9.0))):
        a = ax[r][c]; A = bp(E["A"], lo, hi)
        ser = [("実機 IMU", A, C_R, 1.9, "-"), ("天井=真の関節", bp(E["ceil"], lo, hi), C_K, 1.3, "-"),
               ("G-q-v1", bp(E["v1"], lo, hi), C_V1, 1.2, "-"),
               ("G-2", bp(E["nx"], lo, hi), C_NX, 1.9, "-"),
               ("G-2F (+j5 柔性铰)", bp(E["fx"], lo, hi), C_FX, 1.5, "--")]
        t = np.arange(len(A))/FPS
        if r == 1:
            en = np.convolve(A[:, j]**2, np.ones(40), "same"); k = int(np.argmax(en))
            w = slice(max(0, k-40), k+40); t = t[w]
            ser = [(n_, s[w], cc, lw, st) for n_, s, cc, lw, st in ser]
        for n_, s, cc, lw, st in ser:
            a.plot(t, s[:, j], color=cc, lw=lw, ls=st, label=n_)
        a.set_title(f"{kind} {nm}  {lo}–{hi} Hz (軸 {'xyz'[j]})" + ("  ※最も強い 4 s" if r else "")
                    + f"    ρ²（全集）: G-q-v1 {best_fit(E['A'],E['v1'],lo,hi,rotate=False)[1]:.3f}"
                    f" / G-2 {best_fit(E['A'],E['nx'],lo,hi,rotate=False)[1]:.3f}"
                    f" / G-2F {best_fit(E['A'],E['fx'],lo,hi,rotate=False)[1]:.3f}"
                    f" / 天井 {best_fit(E['A'],E['ceil'],lo,hi,rotate=False)[1]:.3f}", fontsize=9.5)
        a.set_ylabel("角速度 [rad/s]", fontsize=9); a.grid(alpha=.25); a.tick_params(labelsize=8)
        if r == 1: a.set_xlabel("時間 [s]", fontsize=9)
        if r == 0 and c == 0: a.legend(fontsize=8.3, loc="upper right", ncol=2, framealpha=.92)
fig.suptitle(f"j5 下游有効柔性モード ω={sel['w']:.2f} Hz ζ={sel['z']}："
             f"G-2F は G-2 とほぼ重なる ⇒ 増分なし    "
             f"3–9 Hz 中央値 0.520 → 0.524（門槛 0.560・不通過）    "
             f"開発検証集 8 条（密封 holdout ではない）", fontsize=12, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.955])
p = ROOT/"outputs/step4_gvnext/fig14_flex_j5.png"
fig.savefig(p, dpi=150); print(f"→ {p}")
sys.stdout.flush()
import os; os._exit(0)
