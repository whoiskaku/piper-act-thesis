#!/usr/bin/env python3
"""A vs X 的角速度 / gap / 频谱对比图（2026-07-29）。

**两张图用同一个配对序号**（默认 6），使 A 面板在 AB 与 AD 两图中**完全一致**，
从而可以跨图直接对照 B(FF) 与 D(π) 相对同一条 A 的表现。

注意：闭环策略 ⇒ 各臂轨迹**并非逐点相同**，同序号只表示同一采集次序的配对，
曲线对比是**分布性**的，不是逐点差分。

用法:
  python sim/plot_arm_compare.py --other B --ep 6
  python sim/plot_arm_compare.py --other D --ep 6
"""
from __future__ import annotations
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import sys
sys.path.insert(0, str(Path(__file__).parent))
from lerobot.common.policies.act.b_runtime import IdealChunkRenderer

ROOT = Path(__file__).resolve().parents[1]
FPS = 20.0
STYLE = {"A": ("tab:blue", "裸 ACT"), "B": ("tab:green", "FF lag=2 有效blend2"),
         "D": ("tab:red", "ACT + π (FF 关闭)"),
         "B3": ("tab:orange", "FF lag=3 有效blend2"),
         "B3b8": ("tab:purple", "FF lag=3 有效blend5"),
         "B3b13": ("tab:brown", "FF lag=3 有效blend10"),
         "A2": ("tab:cyan", "裸 ACT（同会话对照）"),
         "E": ("tab:purple", "π 直接回归（FF 关闭）"),
         "F": ("tab:olive", "FF + π 直接回归")}


def _bp():
    """**必须与统计口径同一个滤波器** —— analyze_ab._bp 用的是训练侧的 41 抽头 FIR
    (train_pi_e2.make_bp)。此前这里自定义了 scipy butter(3)+filtfilt, 通带形状不同,
    导致图上的绝对能量与报告数字对不上(A 0.0608 vs 0.05625)。2026-07-29 统一。"""
    from analyze_ab import _bp as _shared
    return _shared()


def eps(arm, root, min_frames=500, n=10):
    fs = sorted(Path(root).glob(f"{arm}_ep*.json"),
                key=lambda p: int(re.search(r"ep(\d+)", p.name).group(1)))
    return [f for f in fs
            if json.loads(f.read_text())["summary"]["frames"] > min_frames][:n]


def load(path, R):
    d = json.loads(Path(path).read_text())
    fr, ch = d["frames"], d["chunks"]
    imu = np.array([x["imu"] for x in fr if x["imu"] is not None], np.float32)
    rend = IdealChunkRenderer(); outs, prev = [], None
    for c in ch:
        raw = np.array(c["raw"], np.float32)
        outs.append(rend.render(raw, prev)); prev = raw
    ideal = np.concatenate(outs)[:len(imu)][:, 3:6] @ R.T
    return imu[:len(ideal), 3:6], ideal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--other", default="B", help="与 A 对比的臂: B / D / B3 / B3b13 …")
    ap.add_argument("--ep", type=int, default=6,
                    help="配对序号（两张图用同一个，使 A 面板一致）")
    ap.add_argument("--task", default="pp2", help="pp2 / ins —— 决定用哪个 manifest 的 R")
    ap.add_argument("--root", default="outputs/ab2")
    ap.add_argument("--min-frames", type=int, default=500)
    ap.add_argument("--base", default="A", help="基线臂名。ab3 用同会话对照 A2")
    ap.add_argument("--n", type=int, default=10,
                    help="每臂纳入的集数。⚠ 旧默认是 10，对 40 集的块必须显式给 40")
    ap.add_argument("--base-ep", type=int, default=None,
                    help="基线臂单独的集号（默认与 --ep 相同）。"
                         "**有会话漂移时必须用它**：同序号的两集采集位可能相差很远，"
                         "漂移会把对比放大或缩小。")
    a_ = ap.parse_args()
    O = a_.other
    AB2 = ROOT / a_.root
    R = np.array(json.loads((ROOT / f"outputs/p2_labels/{a_.task}-deploy/manifest.json"
                             ).read_text())["R"], np.float32)
    bp = _bp()
    B = a_.base
    fA = eps(B, AB2, a_.min_frames, a_.n)
    fO = eps(O, AB2, a_.min_frames, a_.n)
    STYLE.setdefault(B, ("tab:blue", B))
    STYLE.setdefault(O, ("tab:brown", O))       # 新臂名不必预先登记
    kB = a_.ep if a_.base_ep is None else a_.base_ep
    kO = a_.ep
    # ⚠ **越界要报错, 不能静默钳位**。原实现是 `k = min(--ep, len-1, len-1)`,
    #   而 `eps()` 默认只取前 10 集 ⇒ `--ep 34` 会被悄悄画成第 9 集, 图与标题全错。
    if not (0 <= kB < len(fA)):
        raise SystemExit(f"--base-ep {kB} 越界: {B} 臂只有 {len(fA)} 集"
                         f"（--n={a_.n}；40 集的块要显式 --n 40）")
    if not (0 <= kO < len(fO)):
        raise SystemExit(f"--ep {kO} 越界: {O} 臂只有 {len(fO)} 集"
                         f"（--n={a_.n}；40 集的块要显式 --n 40）")
    k = kO
    rA, iA = load(fA[kB], R)
    rO, iO = load(fO[kO], R)
    gA, gO = bp(rA - iA), bp(rO - iO)
    eA = float((gA ** 2).sum() / len(gA)); eO = float((gO ** 2).sum() / len(gO))

    # 全集均值（供标题）
    mA = np.mean([float((bp(np.subtract(*load(f, R))) ** 2).sum()
                        / len(load(f, R)[0])) for f in fA])
    mO = np.mean([float((bp(np.subtract(*load(f, R))) ** 2).sum()
                        / len(load(f, R)[0])) for f in fO])

    c_o, lab_o = STYLE[O]
    fig, ax = plt.subplots(4, 1, figsize=(13, 11))
    _same = "同一配对序号" if kB == kO else f"**{B}=第{kB}集 / {O}=第{kO}集（按采集位配对）**"
    ttl = (f"{B}({STYLE[B][1]}) vs {O}({lab_o}) —— {_same} 第{k}集"
           f"\n"
           f"陀螺 y 轴，gap 未对齐口径　|　闭环策略 ⇒ 两臂轨迹并非逐点相同，作分布对照")
    fig.suptitle(ttl, fontsize=11)

    t, t2 = np.arange(len(rA)) / FPS, np.arange(len(rO)) / FPS
    ax[0].plot(t, iA[:, 1], lw=.9, c="k", alpha=.7, label="ideal（ACT 指令渲染）")
    ax[0].plot(t, rA[:, 1], lw=.9, c=STYLE[B][0], alpha=.85, label=f"real {B}（{STYLE[B][1]}）")
    ax[0].set_title(f"① {B} 臂 real vs ideal　gap/帧 {eA:.4f}（全集均值 {mA:.4f}）", fontsize=10)
    ax[1].plot(t2, iO[:, 1], lw=.9, c="k", alpha=.7, label="ideal")
    ax[1].plot(t2, rO[:, 1], lw=.9, c=c_o, alpha=.85, label=f"real {O}（{lab_o}）")
    ax[1].set_title(f"② {O} 臂 real vs ideal　gap/帧 {eO:.4f}"
                    f"（全集均值 {mO:.4f}，相对 {B} {mO/mA*100-100:+.1f}%）", fontsize=10)
    for a2 in ax[:2]:
        a2.set_ylabel("ω_y (rad/s)"); a2.legend(fontsize=9); a2.grid(alpha=.3); a2.set_xlim(0, 17)

    ax[2].plot(t, gA[:, 1], lw=.9, c=STYLE[B][0], alpha=.85, label=f"{B} gap")
    ax[2].plot(t2, gO[:, 1], lw=.9, c=c_o, alpha=.85, label=f"{O} gap")
    ax[2].set_title("③ gap = real − ideal（0.5–9 Hz 带通）", fontsize=10)
    ax[2].set_ylabel("gap ω_y (rad/s)"); ax[2].set_xlabel("时间 (s)")
    ax[2].legend(fontsize=9); ax[2].grid(alpha=.3); ax[2].set_xlim(0, 17)

    P = {}
    for arm, fl in ((B, fA), (O, fO)):
        acc = []
        for f in fl:
            r, i = load(f, R)
            fq, p = welch(np.linalg.norm(bp(r - i), axis=1), fs=FPS, nperseg=256)
            acc.append(p)
        P[arm] = np.mean(acc, 0)
    ax[3].semilogy(fq, P[B], c=STYLE[B][0], lw=1.4, label=f"{B} {STYLE[B][1]}")
    ax[3].semilogy(fq, P[O], c=c_o, lw=1.4, label=f"{O} {lab_o}")
    ax[3].axvspan(0.5, 9, color="gray", alpha=.12); ax[3].set_xlim(0, 10)
    ax[3].set_xlabel("频率 (Hz)"); ax[3].set_ylabel("PSD")
    ax[3].set_title(f"④ gap 频谱（{B} {len(fA)} 集 / {O} {len(fO)} 集平均，灰 = 评分带）",
                    fontsize=10)
    ax[3].legend(fontsize=9); ax[3].grid(alpha=.3)

    fig.tight_layout(rect=[0, 0, 1, 0.945])
    out = AB2 / (f"{B}{O}_curves.png" if kB == kO else f"{B}{kB}_{O}{kO}_curves.png")
    fig.savefig(out, dpi=140)
    print(f"→ {out.relative_to(ROOT.parent)}   ({B}=第{kB}集 / {O}=第{kO}集)")
    print(f"  {O}/{B} 逐频段 PSD 比（<1 = 改善）:")
    for lo, hi in [(0.5, 2), (2, 4), (4, 6), (6, 8), (8, 9)]:
        m = (fq >= lo) & (fq < hi)
        print(f"    {lo}-{hi} Hz: {P[O][m].sum() / P[B][m].sum():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
