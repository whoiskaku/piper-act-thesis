#!/usr/bin/env python3
"""直接回归 π 的真机对照图（2026-08-04）。

四联：
  ① 逐集分布 —— 六个臂，按会话分组；A2 是本次补采的**同会话裸 ACT 对照**
  ② 效应量 + bootstrap 95% CI（相对同会话 A2）
  ③ chunk 内位置剖面 —— **否掉"渲染阻塞"混淆项的那张图**：
     π 臂 render_ms_max≈330 ms（帧预算 50 ms 的 6.6 倍，且只有 π 臂渲染），
     若它是主因，超出应集中在 chunk 开头；实测超出在**中段**、开头反而最小
  ④ 离线拟合 vs 真机结果 —— 两次 π 的对照：离线大幅变好，真机反而更差

口径与 analyze_ab 一致：gap = bandpass(real−ideal) 陀螺三轴 0.5–9 Hz 能量/帧，
ideal 从 `chunks[].raw` 逐 chunk 渲染，未对齐；**统一用 pp2-deploy 的 R**（已发布口径）。

用法: python sim/plot_pi_direct.py
"""
from __future__ import annotations
import glob
import json
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
CH = 100
ARMS = [("A", "outputs/ab2", "7-27", "裸 ACT", "tab:blue"),
        ("B3b13", "outputs/ab2", "7-29", "FF", "tab:brown"),
        ("D", "outputs/ab2", "7-29", "π (灰盒G训练)", "tab:red"),
        ("A2", "outputs/ab3", "8-04", "裸 ACT 对照", "tab:cyan"),
        ("E", "outputs/ab3", "8-04", "π (直接回归)", "tab:purple"),
        ("F", "outputs/ab3", "8-04", "FF + π(直接)", "tab:olive")]


def eps(root, arm, mf=500):
    fs = sorted(glob.glob(f"{root}/{arm}_ep*.json"),
                key=lambda p: int(re.search(r"ep(\d+)", p).group(1)))
    return [f for f in fs
            if json.load(open(f))["summary"]["frames"] > mf][:10]


def main():
    from analyze_ab import load_run, _bp
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    from scipy import stats
    bp = _bp()
    R = np.array(json.load(open(ROOT / "outputs/p2_labels/pp2-deploy/manifest.json"
                                ))["R"], np.float32)
    rend = IdealChunkRenderer()

    E, PROF, RMS = {}, {}, {}
    for arm, root, _s, _l, _c in ARMS:
        fs = eps(str(ROOT / root), arm)
        E[arm] = np.array([(lambda r: r["energy"] / r["n"])(load_run(f, R, rend))
                           for f in fs])
        RMS[arm] = np.mean([json.load(open(f))["summary"].get("render_ms_max", 0)
                            for f in fs])
        if arm in ("A2", "E", "F"):
            pr = []
            for f in fs:
                d = json.load(open(f))
                imu = np.array([x["imu"] for x in d["frames"] if x["imu"] is not None],
                               np.float32)
                outs, prev = [], None
                for c in d["chunks"]:
                    raw = np.array(c["raw"], np.float32)
                    outs.append(rend.render(raw, prev)); prev = raw
                ide = np.concatenate(outs)[:len(imu)]; n = len(ide)
                g = (bp(imu[:n, 3:6] - ide[:, 3:6] @ R.T) ** 2).sum(1)
                pos = np.arange(n) % CH
                pr.append([g[pos == p].mean() for p in range(CH)])
            PROF[arm] = np.mean(pr, 0)

    A2 = E["A2"]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(2, 2, figsize=(14.5, 9.8))
    fig.suptitle("直接回归 π（去掉 Ĝ）的真机结果　"
                 "gap = real−ideal 陀螺 0.5–9 Hz 能量/帧\n"
                 "A2 = 本次补采的**同会话裸 ACT 对照** ⇒ 会话效应已排除（vs 旧 A +0.8%, p=0.881）",
                 fontsize=12.5)

    # ① 逐集分布
    a0 = ax[0, 0]
    for i, (arm, _r, ses, lab, col) in enumerate(ARMS):
        v = E[arm]
        a0.scatter(np.full(len(v), i) + rng.normal(0, .06, len(v)), v,
                   s=26, color=col, alpha=.55, zorder=2)
        a0.errorbar(i, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)),
                    fmt="s", ms=11, color=col, capsize=6, lw=2.2,
                    markeredgecolor="k", zorder=3)
        a0.annotate(f"{v.mean():.4f}", (i, v.mean()), fontsize=8.5,
                    textcoords="offset points", xytext=(11, 4))
    a0.axhline(A2.mean(), ls="--", c="tab:cyan", lw=1.5, zorder=1,
               label=f"同会话对照 A2 {A2.mean():.4f}")
    a0.axvline(2.5, color="k", ls=":", lw=1.2, alpha=.6)
    a0.text(1.0, a0.get_ylim()[1] * .97, "7-27 / 7-29 会话", ha="center", fontsize=9)
    a0.text(4.5, a0.get_ylim()[1] * .97, "8-04 会话", ha="center", fontsize=9)
    a0.set_xticks(range(len(ARMS)))
    a0.set_xticklabels([f"{a}\n{l}" for a, _r, _s, l, _c in ARMS], fontsize=8.5)
    a0.set_ylabel("gap 能量 / 帧"); a0.legend(fontsize=9)
    a0.set_title("① 逐集分布（点 = 单集，方 = 均值±SEM）", fontsize=10.5)
    a0.grid(alpha=.3, axis="y")

    # ② 效应量 vs A2
    a1 = ax[0, 1]
    order = [("B3b13", "FF 单独"), ("D", "π (灰盒G训练)"), ("E", "π (直接回归)"),
             ("F", "FF + π(直接)"), ("A", "旧 A（会话检验）")]
    for k, (arm, lab) in enumerate(order):
        x, y = A2, E[arm]
        rel = (y.mean() / x.mean() - 1) * 100
        bs = [(y[i].mean() / x[j].mean() - 1) * 100
              for i, j in zip(rng.integers(0, len(y), (4000, len(y))),
                              rng.integers(0, len(x), (4000, len(x))))]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        col = "tab:green" if rel < 0 else ("gray" if abs(rel) < 5 else "tab:red")
        a1.barh(k, rel, color=col, alpha=.85, edgecolor="k", lw=.8)
        a1.plot([lo, hi], [k, k], color="k", lw=2)
        a1.annotate(f"{rel:+.1f}%", (rel, k), fontsize=9.5, va="center",
                    fontweight="bold",
                    textcoords="offset points", xytext=(8 if rel > 0 else -8, 0),
                    ha="left" if rel > 0 else "right")
    a1.axvline(0, c="k", lw=1.2)
    a1.set_yticks(range(len(order)))
    a1.set_yticklabels([l for _a, l in order], fontsize=9.5)
    a1.set_xlabel("相对同会话对照 A2 的 gap 变化 (%)")
    a1.set_xlim(-55, 110)
    a1.set_title("② 效应量 + bootstrap 95% CI（黑线）", fontsize=10.5)
    a1.grid(alpha=.3, axis="x")

    # ③ chunk 内位置剖面 —— 否掉渲染阻塞
    a2 = ax[1, 0]
    for arm, col in (("A2", "tab:cyan"), ("E", "tab:purple"), ("F", "tab:olive")):
        a2.plot(PROF[arm], lw=1.5, color=col,
                label=f"{arm}  (render_max {RMS[arm]:.0f} ms)")
    a2.axvspan(0, 10, color="tab:red", alpha=.12)
    a2.annotate("若渲染阻塞是主因，\n超出应集中在这里", (5, PROF["E"].max() * .55),
                fontsize=9, ha="left", color="tab:red")
    a2.set_yscale("log")
    a2.set_xlabel("chunk 内位置（帧）"); a2.set_ylabel("gap² 均值/帧")
    seg = {"开头0-9": slice(0, 10), "中段30-89": slice(30, 90)}
    txt = "  相对 A2：" + "　".join(
        f"{arm} " + "/".join(f"{PROF[arm][s].mean()/PROF['A2'][s].mean():.2f}×"
                             for s in seg.values()) for arm in ("E", "F"))
    a2.set_title("③ chunk 内位置剖面 —— 否掉「渲染阻塞」\n"
                 + txt + "　（开头/中段）", fontsize=10.5)
    a2.legend(fontsize=9); a2.grid(alpha=.3)

    # ④ 离线拟合 vs 真机
    a3 = ax[1, 1]
    off = [-0.088, +0.484]                       # 对 δa* 的方向相关（留出集）
    real = [(E["D"].mean() / A2.mean() - 1) * 100,
            (E["E"].mean() / A2.mean() - 1) * 100]
    labs = ["v1  π (灰盒G训练)", "v2  π (直接回归)"]
    x = np.arange(2); w = 0.36
    b1 = a3.bar(x - w / 2, off, w, color="tab:blue", edgecolor="k", lw=.8,
                label="离线：对 δa* 的方向相关（留出集）")
    a3b = a3.twinx()
    b2 = a3b.bar(x + w / 2, real, w, color="tab:red", edgecolor="k", lw=.8,
                 label="真机：相对 A2 的 gap 变化 (%)")
    for xx, v in zip(x - w / 2, off):
        a3.annotate(f"{v:+.3f}", (xx, v), ha="center", fontsize=9.5, fontweight="bold",
                    textcoords="offset points", xytext=(0, 4 if v > 0 else -14))
    for xx, v in zip(x + w / 2, real):
        a3b.annotate(f"{v:+.1f}%", (xx, v), ha="center", fontsize=9.5, fontweight="bold",
                     textcoords="offset points", xytext=(0, 4))
    a3.axhline(0, c="k", lw=1)
    a3.set_xticks(x); a3.set_xticklabels(labs, fontsize=10)
    a3.set_ylabel("离线方向相关", color="tab:blue")
    a3b.set_ylabel("真机 gap 变化 (%)", color="tab:red")
    a3.set_ylim(-0.7, 0.7); a3b.set_ylim(-20, 100)
    a3.tick_params(axis="y", colors="tab:blue"); a3b.tick_params(axis="y", colors="tab:red")
    a3.legend(handles=[b1, b2], fontsize=9, loc="upper left")
    a3.set_title("④ 离线拟合大幅变好，真机反而更差\n"
                 "离线衡量的是「拟合标签多准」，失效点在「标签本身对不对」",
                 fontsize=10.5)
    a3.grid(alpha=.3, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = ROOT / "outputs/ab3/pi_direct_summary.png"
    fig.savefig(out, dpi=140)
    print(f"→ {out.relative_to(ROOT.parent)}\n")
    print(f"  {'臂':>7}{'会话':>7}{'集数':>5}{'gap/帧':>10}{'vs A2':>9}{'render_max':>12}")
    for arm, _r, ses, lab, _c in ARMS:
        v = E[arm]
        print(f"  {arm:>7}{ses:>7}{len(v):>5}{v.mean():>10.5f}"
              f"{v.mean()/A2.mean()*100-100:>8.1f}%{RMS[arm]:>11.0f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
