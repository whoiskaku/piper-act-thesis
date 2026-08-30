#!/usr/bin/env python3
"""**过拟合诊断** —— 训练集上更优的参数，在没见过的位形上是否也更优？

Optuna 的 640 trial 已给出：训练 0.5697 → **0.4004**，但留出 A（位形 p2）
只从 0.6627 → **0.5663**，离预注册门槛 0.35 很远。

两种解释，必须分开：

| 解释 | 预测 |
|---|---|
| **搜索不够** | 训练越优 ⇒ 留出也越优（散点沿对角线下行） |
| **参数在过拟合位形** | 训练越优 ⇒ 留出**持平或变差**（散点变平甚至上翘） |

本模块取训练集上最好的 K 个 trial，逐个在**留出 A/B** 上评估，画出
「训练 vs 留出」散点与「随 trial 数推进的最优-至今」曲线。

> ⚠ **本模块只作诊断，不得用于选参数。** 按留出集挑 trial 就等于把留出集
> 变成选择集 —— 本项目已在 pp2 的 17 集上犯过三次这个错。

用法:
    python sim/step2_g/optuna_holdout_curve.py --topk 60 --workers 14
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g"):
    sys.path.insert(0, str(p))

from optuna_ident import (BASE_PHYS_V1, BASE_XML, CLASSES, OUT, STUDY,   # noqa: E402
                          evaluate, prep, split, storage)

_G: dict = {}


def _init():
    import mujoco
    from sim.step1_v2_loss import load_calib
    from sim.step2_fidelity import CALIB
    from sim.step2_gap_baseline import MODEL
    _G["m"] = mujoco.MjModel.from_xml_path(str(MODEL))
    _G["R"] = load_calib(CALIB)[0]
    tr, ha, hb = split()
    _G["A"] = prep(ha)
    _G["B"] = prep(hb)


def _job(item):
    num, params = item
    a = evaluate(_G["m"], _G["A"], _G["R"], params)[0]
    b = evaluate(_G["m"], _G["B"], _G["R"], params)[0]
    return num, a, b


def figure(rows, base, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    INK, INK2 = "#22223b", "#5c5c7a"
    tr = np.array([r["train"] for r in rows])
    ha = np.array([r["hA"] for r in rows])
    n = np.array([r["num"] for r in rows])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    sc = ax.scatter(tr, ha, c=n, cmap="viridis", s=46, zorder=3,
                    edgecolor="white", linewidth=0.6)
    fig.colorbar(sc, ax=ax, label="trial 序号（越亮＝越晚）")
    for nm, (x, y), col in base:
        ax.scatter([x], [y], s=150, marker="*", color=col, zorder=5,
                   edgecolor="white", linewidth=1.0)
        ax.annotate(nm, (x, y), textcoords="offset points", xytext=(-11, -5),
                    ha="right", fontsize=11, color=col, fontweight="bold")
    ax.set_xlim(min(tr.min(), *[b[1][0] for b in base]) - 0.045,
                max(tr.max(), *[b[1][0] for b in base]) + 0.02)
    ax.axhline(0.35, color="#d1495b", ls="--", lw=1.6, zorder=2)
    ax.text(ax.get_xlim()[0], 0.352, "  预注册门槛 0.35", color="#d1495b", fontsize=10.5)
    if len(tr) > 2:
        k, b = np.polyfit(tr, ha, 1)
        xs = np.linspace(tr.min(), tr.max(), 10)
        ax.plot(xs, k * xs + b, color=INK2, lw=1.8, ls=":", zorder=4)
        ax.set_title(f"训练更优 ⇒ 留出更优？  拟合斜率 = {k:+.2f}"
                     f"（1.0 = 完全迁移，0 = 完全不迁移）",
                     fontsize=12, color=INK, loc="left")
    ax.set_xlabel("训练集 p0+p1  等权相对残差", fontsize=11, color=INK2)
    ax.set_ylabel("留出 A（位形 p2）等权相对残差", fontsize=11, color=INK2)
    ax.grid(alpha=0.25)

    o = np.argsort(n)
    bt, bh, seen = [], [], np.inf
    for i in o:
        if tr[i] < seen:
            seen = tr[i]
            bt.append(n[i]); bh.append(ha[i])
    ax2.step(bt, bh, where="post", color="#2a9d8f", lw=2.2,
             label="「训练最优-至今」的留出 A")
    ax2.axhline(base[1][1][1], color="#8d99ae", ls="-.", lw=1.5,
                label=f"G-π-phys-v1 = {base[1][1][1]:.4f}")
    ax2.axhline(0.35, color="#d1495b", ls="--", lw=1.6, label="预注册门槛 0.35")
    ax2.set_xlabel("trial 序号", fontsize=11, color=INK2)
    ax2.set_ylabel("留出 A 等权相对残差", fontsize=11, color=INK2)
    ax2.set_title("再多搜索，留出集会下去吗？", fontsize=12, color=INK, loc="left")
    ax2.legend(fontsize=10, frameon=False)
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    print(f"  → {path}")


def main() -> int:
    import multiprocessing as mp
    import optuna
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=60)
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.load_study(study_name=STUDY, storage=storage())
    done = sorted([t for t in st.trials if t.value is not None],
                  key=lambda t: t.value)[: a.topk]
    print(f"**过拟合诊断** —— 训练集上最好的 {len(done)} / "
          f"{sum(1 for t in st.trials if t.value is not None)} 个 trial")
    print(f"  ⚠ 只作诊断，**不得用于选参数**\n")
    jobs = [(t.number,
             {c: np.array([t.params[f"{c}{j+1}"] for j in range(6)]) for c in CLASSES})
            for t in done]
    jobs += [(-1, BASE_XML), (-2, BASE_PHYS_V1)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init) as pool:
        res = pool.map(_job, jobs)
    val = {n: (x, y) for n, x, y in res}
    tmap = {t.number: t.value for t in done}
    rows = [{"num": n, "train": tmap[n], "hA": val[n][0], "hB": val[n][1]}
            for n, _ in jobs if n >= 0]
    rows.sort(key=lambda r: r["train"])

    print(f"  {'trial':>6s}{'训练':>9s}{'留出A':>9s}{'留出B':>9s}")
    for r in rows[:12]:
        print(f"  {r['num']:>6d}{r['train']:>9.4f}{r['hA']:>9.4f}{r['hB']:>9.4f}")
    print(f"  {'...':>6s}")
    for r in rows[-3:]:
        print(f"  {r['num']:>6d}{r['train']:>9.4f}{r['hA']:>9.4f}{r['hB']:>9.4f}")

    tr = np.array([r["train"] for r in rows]); ha = np.array([r["hA"] for r in rows])
    k, _b = np.polyfit(tr, ha, 1)
    rho = float(np.corrcoef(tr, ha)[0, 1])
    print(f"\n  训练→留出A 迁移斜率 **{k:+.2f}**（1.0=完全迁移，0=完全不迁移）"
          f"  相关 ρ = {rho:+.3f}")
    print(f"  这 {len(rows)} 个 trial 里，留出A 最好 {ha.min():.4f} / "
          f"最差 {ha.max():.4f}；**没有一个** ≤ 0.35"
          if ha.min() > 0.35 else f"  留出A 最好 {ha.min():.4f}")
    print(f"  对照 G-π-phys-v1 留出A = {val[-2][0]:.4f} · 现 XML = {val[-1][0]:.4f}")

    base = [("现 XML", val[-1], "#8d99ae"), ("G-π-phys-v1", val[-2], "#e76f51")]
    OUT.mkdir(parents=True, exist_ok=True)
    figure(rows, base, OUT / "holdout_curve.png")
    (OUT / "holdout_curve.json").write_text(json.dumps(
        {"rows": rows, "slope": float(k), "rho": rho,
         "base_xml": list(val[-1]), "base_phys_v1": list(val[-2])},
        ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
