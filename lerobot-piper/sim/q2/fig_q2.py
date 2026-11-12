"""阶段二只报告的波形图：留出补正运行上「预测 IMU vs 实测 IMU」（每折一张，种子 0；中位集 + 最难集）。

沿用「改完 G 必出波形图」的版式（`sim/q1/fig_pp2_q1.py`）：0.5–9 Hz 带通、中位集与最难集、口径注记。
- 左列：实测 IMU 与四种预测的 IMU（`ideal_t + ê_t`，与实测 `ω_{t+3}` 对齐；无大域位移，同评分口径的 L = 3）。
- 右列：实测 gap 减名义预测（= 名义没解释的部分）与三种预测增量（固定 H、两种候选）。这只是直观对照，
  **不是**「已分离的真实增量」（闭环运行里残差还含名义模型自身的误差）。
- 中位 / 最难按零响应的逐集 ρ_e 排序（与候选无关）。预测取自终训产物的 `.npz`，作图前先通过 `q2_report.verify`。
**只作图，不进任何判据**；不选步、不选模。不在代码指纹里。

    python -u sim/q2/fig_q2.py --out outputs/q2_stage2
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent / "q1", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import fig_pp2_q1 as F1        # noqa: E402
import q1_data as D1           # noqa: E402
import q2_run as RUN           # noqa: E402

SEED = 0
FOLDS, FORMS = RUN.FOLDS, RUN.FORMS
PREDS = ("zero", "H") + FORMS
STYLE = {"zero": ("零応答（名義 2a のみ）", "#9aa0a6", ":", 1.8), "H": ("固定 H", "#e07a1f", "-.", 1.8),
         "nonlinear": ("非線形差分", "#8b2fb0", "--", 2.0), "condlinear": ("条件付き線形", "#2a9d4b", "--", 2.0)}
C_REAL = "#1f6fb2"


def fold_rows(out, fold: str, seed: int = SEED) -> tuple:
    """→ ([{"name", "ep", "pred": {预测: ê (n, 3)}, "rho": {预测: ρ_e}}], {候选: step*})。"""
    nb = RUN.nom_final_base(out, fold, seed)
    nj, nz = json.loads(nb.with_suffix(".json").read_text()), np.load(nb.with_suffix(".npz"))
    cj = {f: json.loads(RUN.inc_final_base(out, f, fold, seed).with_suffix(".json").read_text()) for f in FORMS}
    cz = {f: np.load(RUN.inc_final_base(out, f, fold, seed).with_suffix(".npz")) for f in FORMS}
    rho = lambda rows: {r["name"]: r["num"] / r["den"] for r in rows}
    R = {"zero": rho(nj["zero"]), "H": rho(nj["H"]), **{f: rho(cj[f]["cand"]) for f in FORMS}}
    rows = []
    for e in RUN.load_eps((fold,), ("corr",)):
        k = e.name.replace("/", "__")
        pred = {"zero": nz[f"nom__{k}"], "H": nz[f"H__{k}"], **{f: cz[f][f"cand__{k}"] for f in FORMS}}
        rows.append({"name": e.name, "ep": e, "pred": pred, "rho": {p: R[p][e.name] for p in PREDS}})
    return rows, {f: cj[f]["step_star"] for f in FORMS}


def pick(rows) -> tuple:
    """按零响应的逐集 ρ_e 取中位集与最难集。"""
    order = np.argsort([r["rho"]["zero"] for r in rows], kind="stable")
    return rows[order[len(order) // 2]], rows[order[-1]]


def plot(rows, fold: str, steps: dict, path: Path, span: int = 200) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(13.33, 7.5))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.16, left=0.06, right=0.99, top=0.83, bottom=0.13)
    for i, (r, title) in enumerate(zip(pick(rows), ("① 中央値のエピソード", "② 最も難しいエピソード"))):
        e = r["ep"]
        real = F1.bp(e.w[D1.ALIGN_L:D1.ALIGN_L + e.n])
        ax_i = int(np.argmax(np.abs(real).max(0)))
        c = int(np.argmax(np.abs(real[:, ax_i])))
        s = max(0, min(c - span // 2, len(real) - span))
        t = np.arange(span) / D1.FS
        a = fig.add_subplot(gs[i, 0])
        a.plot(t, real[s:s + span, ax_i], color=C_REAL, lw=2.4, label="実機 IMU")
        for p in PREDS:
            lab, col, ls, lw = STYLE[p]
            a.plot(t, F1.bp(e.ideal[:e.n] + r["pred"][p])[s:s + span, ax_i], color=col, ls=ls, lw=lw, label=lab)
        a.set_title(f"{title}  {r['name']}  軸 ω{'xyz'[ax_i]}\nρ_e：" + " · ".join(
            f"{STYLE[p][0].split('（')[0]} {r['rho'][p]:.3f}" for p in PREDS), fontsize=10.5, loc="left", color="#22223b")
        a.set_ylabel("角速度 (rad/s)\n0.5–9 Hz", fontsize=10)
        b = fig.add_subplot(gs[i, 1], sharex=a)
        b.plot(t, F1.bp(e.gap - r["pred"]["zero"])[s:s + span, ax_i], color=C_REAL, lw=2.2,
               label="実測 gap − 名義予測")
        for p in PREDS[1:]:
            lab, col, ls, lw = STYLE[p]
            b.plot(t, F1.bp(r["pred"][p] - r["pred"]["zero"])[s:s + span, ax_i], color=col, ls=ls, lw=lw,
                   label=f"{lab} の増分")
        b.set_title("名義が説明しない部分 と 予測増分（直観的対照のみ）", fontsize=10.5, loc="left", color="#22223b")
        for x in (a, b):
            x.set_xlabel("時間 (s)", fontsize=10)
            x.grid(alpha=0.25)
        if i == 0:
            a.legend(fontsize=8.5, frameon=False, ncol=3, loc="upper right")
            b.legend(fontsize=8.5, frameon=False, ncol=2, loc="upper right")
    note = "・".join(f"{STYLE[f][0]} step*={steps[f]}" + ("（零応答と同一）" if steps[f] == 0 else "") for f in FORMS)
    fig.suptitle(f"阶段二 開発検証：{fold} 留出・補正運転 — 予測 IMU と 実機 IMU（seed {SEED}）",
                 fontsize=16, fontweight="bold", color="#22223b", y=0.985)
    fig.text(0.5, 0.915, f"帯域 0.5–9 Hz・大域シフトなし（評価と同じ L=3）・chunk 先頭起点・末端クランプ・{note}",
             ha="center", fontsize=10.5, color="#5a5a72")
    fig.text(0.5, 0.012, "※ 開発検証（密封確認ではない）・図は報告のみで判定に使わない。ρ_e = ‖P(gap−ê)‖²/‖P(gap)‖²"
             "（1.0 = ê ≡ 0 相当）。中央値／最難は零応答の ρ_e で選択。\n"
             "δa はログに記録された要求補正量（実行変位ではない）；当該 chunk の補正は離線で与えた条件。"
             "右列の残差は名義モデル自身の誤差も含む（分離した真の増分ではない）。",
             ha="center", fontsize=9, color="#c1121f", linespacing=1.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def main(argv=None) -> int:
    gc.disable()
    import q2_report as REP
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    problems = REP.verify(a.out)
    if problems:
        print("产物核验未通过，不作图：\n  " + "\n  ".join(problems), flush=True)
        return 4
    for fold in FOLDS:
        rows, steps = fold_rows(a.out, fold)
        p = Path(a.out) / "fig" / f"q2_{fold}_seed{SEED}.png"
        plot(rows, fold, steps, p)
        print(f"  → {p}", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
