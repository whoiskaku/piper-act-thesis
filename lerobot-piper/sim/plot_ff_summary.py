#!/usr/bin/env python3
"""FF 两任务对照总图（2026-08-04）—— pp2 与 insert 并排。

四联：
  ① 逐集配对（A→B3→B3b13）—— 看方向一致性，不被均值掩盖
  ② 效应量 + bootstrap 95% CI —— 看结论稳不稳
  ③ **运动段 vs 静止段分解** —— insert 有 54.5% 的帧指令近乎静止（悬停失败），
     必须分开看，否则"降 43.6%"会被误读成任务性能
  ④ chunk 内位置剖面 —— blend 的作用面与它的越界效果

口径与 analyze_ab 一致：gap = bandpass(real−ideal) 陀螺三轴，0.5–9 Hz 41 抽头 FIR，
ideal 从 chunks[].raw 逐 chunk 渲染，未对齐。静止判据：指令角速度模长 < 0.05 rad/s。

用法: python sim/plot_ff_summary.py
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent))

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
CHUNK, STILL_TH = 100, 0.05
ARMS = ["A", "B3", "B3b13"]
LBL = {"A": "裸 ACT", "B3": "FF lag3 有效blend2", "B3b13": "FF lag3 有效blend10"}
COL = {"A": "tab:blue", "B3": "tab:orange", "B3b13": "tab:brown"}
TASKS = [("pp2", "outputs/ab2", 500, "pp2（pick-place）"),
         ("ins", "outputs/ab_ins", 900, "insert（成功率 0/10，全程悬停）")]


def collect():
    from analyze_ab import _bp
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    bp, rend = _bp(), IdealChunkRenderer()
    out = {}
    for tag, root, mf, _t in TASKS:
        R = np.array(json.loads((ROOT / f"outputs/p2_labels/{tag}-deploy/manifest.json"
                                 ).read_text())["R"], np.float32)
        out[tag] = {}
        for arm in ARMS:
            fs = sorted((ROOT / root).glob(f"{arm}_ep*.json"),
                        key=lambda p: int(re.search(r"ep(\d+)", p.name).group(1)))
            fs = [f for f in fs
                  if json.loads(f.read_text())["summary"]["frames"] > mf][:10]
            E, Em, Es, St, Pr = [], [], [], [], []
            for f in fs:
                d = json.loads(f.read_text())
                imu = np.array([x["imu"] for x in d["frames"] if x["imu"] is not None],
                               np.float32)
                outs, prev = [], None
                for c in d["chunks"]:
                    raw = np.array(c["raw"], np.float32)
                    outs.append(rend.render(raw, prev)); prev = raw
                ide = np.concatenate(outs)[:len(imu)]
                n = len(ide)
                g = (bp(imu[:n, 3:6] - ide[:, 3:6] @ R.T) ** 2).sum(1)
                still = np.linalg.norm(ide[:, 3:6], axis=1) < STILL_TH
                E.append(g.mean()); St.append(still.mean())
                Em.append(g[~still].mean()); Es.append(g[still].mean())
                pos = np.arange(n) % CHUNK
                Pr.append([g[pos == p].mean() for p in range(CHUNK)])
            out[tag][arm] = {"E": np.array(E), "Em": np.mean(Em), "Es": np.mean(Es),
                             "still": np.mean(St), "prof": np.mean(Pr, 0)}
    return out


def boot(x, y, rng, n=4000):
    v = [(y[i].mean() / x[i].mean() - 1) * 100 for i in rng.integers(0, len(x), (n, len(x)))]
    return np.percentile(v, [2.5, 97.5])


def main():
    D = collect()
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(2, 2, figsize=(14.5, 10))
    fig.suptitle("FF 两任务对照　gap = real−ideal 陀螺 0.5–9 Hz 能量/帧，每臂 10 集配对\n"
                 "insert 成功率 0/10（全部悬停未放置）—— 故第③格必须分开读",
                 fontsize=12.5)

    # ① 逐集配对
    a0 = ax[0, 0]
    for j, (tag, _r, _m, title) in enumerate(TASKS):
        off = j * 4
        for i in range(10):
            ys = [D[tag][a]["E"][i] for a in ARMS]
            a0.plot([off, off + 1, off + 2], ys, "-o", ms=3.5, lw=.9, alpha=.45,
                    color="gray")
        ms = [D[tag][a]["E"].mean() for a in ARMS]
        a0.plot([off, off + 1, off + 2], ms, "-s", ms=9, lw=2.6, color="k", zorder=5)
        for k, a in enumerate(ARMS):
            a0.scatter([off + k], [ms[k]], s=95, color=COL[a], zorder=6,
                       edgecolor="k", lw=1.1)
            a0.annotate(f"{ms[k]:.4f}", (off + k, ms[k]), fontsize=8.5,
                        textcoords="offset points", xytext=(6, 7))
        # 任务标题放到轴外(axes 坐标), 避免与数据/轴顶重叠
        a0.text((off + 1 + 0.5) / 7.5, 1.035, title, transform=a0.transAxes,
                ha="center", fontsize=9.5, fontweight="bold")
    a0.axvline(3, color="k", lw=1, ls=":", alpha=.5)          # 两任务分隔
    a0.set_xlim(-0.6, 6.6)
    a0.set_xticks([0, 1, 2, 4, 5, 6])
    a0.set_xticklabels(ARMS * 2, fontsize=9)
    a0.set_ylabel("gap 能量 / 帧")
    a0.set_title("① 逐集配对（灰线 = 单集，黑方 = 均值）", fontsize=10.5, pad=22)
    a0.grid(alpha=.3)

    # ② 效应量 + bootstrap CI
    a1 = ax[0, 1]
    xs, labs = [], []
    for j, (tag, _r, _m, _t) in enumerate(TASKS):
        for k, arm in enumerate(ARMS[1:]):
            x, y = D[tag]["A"]["E"], D[tag][arm]["E"]
            rel = (y.mean() / x.mean() - 1) * 100
            lo, hi = boot(x, y, rng)
            p = j * 3 + k
            a1.barh(p, rel, color=COL[arm], alpha=.85, edgecolor="k", lw=.8)
            a1.plot([lo, hi], [p, p], color="k", lw=2)
            a1.annotate(f"{rel:.1f}%  [{lo:.0f}, {hi:.0f}]", (min(rel, lo), p),
                        fontsize=8.5, va="center",
                        textcoords="offset points", xytext=(-6, 0), ha="right")
            xs.append(p); labs.append(f"{tag} {arm}")
    a1.axvline(0, c="k", lw=1)
    a1.set_yticks(xs); a1.set_yticklabels(labs, fontsize=9)
    a1.set_xlabel("相对 A 的 gap 变化 (%)")
    a1.set_xlim(-75, 12)
    a1.set_title("② 效应量 + bootstrap 95% CI（黑线）", fontsize=10.5)
    a1.grid(alpha=.3, axis="x")

    # ③ 运动段 vs 静止段
    a2 = ax[1, 0]
    w = 0.35
    for j, (tag, _r, _m, _t) in enumerate(TASKS):
        for k, arm in enumerate(["A", "B3b13"]):
            base = j * 3 + k * w - w / 2
            a2.bar(base, D[tag][arm]["Em"], w, color=COL[arm], alpha=.9,
                   edgecolor="k", lw=.8, label="运动段" if j + k == 0 else None)
            a2.bar(base + 1.2, D[tag][arm]["Es"], w, color=COL[arm], alpha=.45,
                   hatch="//", edgecolor="k", lw=.8,
                   label="静止段(悬停)" if j + k == 0 else None)
            for xx, vv in ((base, D[tag][arm]["Em"]), (base + 1.2, D[tag][arm]["Es"])):
                a2.annotate(f"{vv:.4f}", (xx, vv), ha="center", fontsize=8,
                            textcoords="offset points", xytext=(0, 3))
    for j, (tag, _r, _m, _t) in enumerate(TASKS):
        st = D[tag]["A"]["still"] * 100
        dm = (D[tag]["B3b13"]["Em"] / D[tag]["A"]["Em"] - 1) * 100
        ds = (D[tag]["B3b13"]["Es"] / D[tag]["A"]["Es"] - 1) * 100
        a2.text(j * 3 + 0.6, -0.012,
                f"{tag}　静止帧 {st:.0f}%\n运动段 {dm:+.0f}%　静止段 {ds:+.0f}%",
                ha="center", fontsize=9)
    a2.set_xticks([0, 1.2, 3, 4.2])
    a2.set_xticklabels(["pp2 运动", "pp2 静止", "ins 运动", "ins 静止"], fontsize=9)
    a2.set_ylabel("gap 能量 / 帧")
    a2.set_ylim(bottom=-0.022)
    a2.legend(fontsize=9, loc="upper right")
    a2.set_title("③ 运动段 vs 静止段（深 = A，浅斜线 = B3b13）\n"
                 "insert 过半帧是悬停 —— 该段的改善不等于任务性能", fontsize=10.5)
    a2.grid(alpha=.3, axis="y")

    # ④ chunk 内位置剖面
    a3 = ax[1, 1]
    for j, (tag, _r, _m, _t) in enumerate(TASKS):
        ls = "-" if tag == "pp2" else "--"
        for arm in ("A", "B3b13"):
            p = D[tag][arm]["prof"]
            a3.plot(p / D[tag]["A"]["prof"][30:90].mean(), ls, lw=1.4,
                    color=COL[arm], alpha=.9 if tag == "pp2" else .6,
                    label=f"{tag} {arm}")
    a3.axvspan(0, 10, color="gray", alpha=.15)
    a3.axhline(1, c="k", lw=.8)
    a3.set_yscale("log")
    a3.set_xlabel("chunk 内位置（帧）")
    a3.set_ylabel("gap² / 各任务 A 的中段均值")
    a3.set_title("④ chunk 内位置剖面（灰 = blend 作用面 0–9）\n"
                 "中段也降 ⇒ 接缝瞬态响铃到作用面之外", fontsize=10.5)
    a3.legend(fontsize=8.5, ncol=2); a3.grid(alpha=.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = ROOT / "outputs/ab_ins/FF_two_task_summary.png"
    fig.savefig(out, dpi=140)
    print(f"→ {out.relative_to(ROOT.parent)}\n")
    print(f"  {'任务':>6}{'臂':>8}{'gap/帧':>10}{'相对A':>9}{'静止帧':>8}"
          f"{'运动段':>10}{'静止段':>10}")
    for tag, _r, _m, _t in TASKS:
        for arm in ARMS:
            d = D[tag][arm]; A = D[tag]["A"]
            print(f"  {tag:>6}{arm:>8}{d['E'].mean():>10.5f}"
                  f"{d['E'].mean()/A['E'].mean()*100-100:>8.1f}%{d['still']*100:>7.0f}%"
                  f"{d['Em']:>10.5f}{d['Es']:>10.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
