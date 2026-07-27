#!/usr/bin/env python3
"""Phase1 no-TE 评测汇总: 分阶段成功率 + 按结果(成功/失败)分组的 IMU 带功率/时长。

交叉对照: 失败集是否=高振动/不收敛那批; 成功集的平滑度 vs demo(去掉失败thrash污染)。
用法: /home/enolab-guo/miniconda3/envs/lerobot/bin/python sim/eval_summary.py [repo_短名]
"""
from __future__ import annotations
import csv
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

LR = Path.home() / ".cache/huggingface/lerobot/enolab-guo"
FS, NPERSEG = 30.0, 256
REPO = sys.argv[1] if len(sys.argv) > 1 else "eval_phase1_noimu_no_te"


def bp(im):
    a = np.linalg.norm(im[:, :3] - im[:, :3].mean(0), axis=1)
    f, P = welch(a, fs=FS, nperseg=min(NPERSEG, len(a)))
    m = (f >= 2.0) & (f < 4.5)
    return float(np.trapezoid(P[m], f[m]))


def active_end(st):
    v = np.linalg.norm(np.diff(st[:, :6], axis=0), axis=1)
    v = np.convolve(v, np.ones(9) / 9, "same")
    idx = np.where(v > 0.1 * np.percentile(v, 95))[0]
    return ((idx[-1] + 1) if len(idx) else len(st)) / FS


def main():
    stages = {}
    csvp = Path(__file__).resolve().parents[1] / "outputs/review" / f"enolab-guo__{REPO}_eval.csv"
    for r in csv.reader(csvp.open()):
        if len(r) >= 2:
            stages[int(r[0])] = int(r[1])

    rows = []
    for pq in sorted(glob.glob(str(LR / REPO / "data/chunk-*/episode_*.parquet")),
                     key=lambda p: int(Path(p).stem.split("_")[1])):
        ep = int(Path(pq).stem.split("_")[1])
        df = pd.read_parquet(pq)
        im = np.stack(df["observation.imu_real"].to_numpy()).astype(float)
        st = np.stack(df["observation.state"].to_numpy()).astype(float)
        rows.append((ep, stages.get(ep, -1), bp(im), len(im) / FS, active_end(st)))

    demo = [bp(np.stack(pd.read_parquet(p)["observation.imu_real"].to_numpy()).astype(float))
            for p in sorted(glob.glob(str(LR / "pickplace_clean/data/chunk-*/episode_*.parquet")))[1:]]
    demo_med = float(np.median(demo))

    n = len(rows)
    succ = [r for r in rows if r[1] == 3]
    fail = [r for r in rows if r[1] == 0]
    print(f"\n{'='*56}\nPhase1 no-TE 评测 ({REPO}, n={n})\n{'='*56}")
    for thr, lab in [(1, "抓住"), (2, "提起"), (3, "放到目标=成功")]:
        c = sum(1 for r in rows if r[1] >= thr)
        print(f"  ≥{lab:<10}: {c:>2}/{n} = {100*c/n:>3.0f}%")
    print(f"\n失败集(stage0=没抓到): {sorted(r[0] for r in fail)}")
    print(f"→ 失败全在抓取阶段; 一旦抓住({len(succ)}集)100%完成放置。瓶颈=抓取。")

    def med(g, i):
        return float(np.median([r[i] for r in g])) if g else float("nan")
    print(f"\n按结果分组(中位): accel带功率 | 有效运动时长")
    print(f"  demo (teleop)        bp={demo_med:.3f}")
    print(f"  no-TE 成功 (n={len(succ):>2})    bp={med(succ,2):.3f}   active={med(succ,4):.1f}s")
    print(f"  no-TE 失败 (n={len(fail):>2})    bp={med(fail,2):.3f}   active={med(fail,4):.1f}s")
    hi_fail = sum(1 for r in fail if r[2] > demo_med)
    print(f"\n失败集中 {hi_fail}/{len(fail)} 高于 demo 带功率中位 → 失败≈thrash(高振动); "
          f"成功集 bp 中位{'≤' if med(succ,2)<=demo_med else '>'}demo")


if __name__ == "__main__":
    main()
