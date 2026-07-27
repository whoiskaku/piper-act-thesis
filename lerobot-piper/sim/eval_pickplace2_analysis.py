#!/usr/bin/env python3
"""eval_pickplace2_100k 分析 (双相机 Phase1 基线, 20fps):
① 执行 vs 指令 保真度 (corr/滞后/|gap|) —— 复核"执行零偏离"在慢任务+双相机下是否成立;
② IMU 2-4.5Hz 带功率 (Welch, 长度不变) vs demo(pickplace2_train) —— 部署平滑度 vs 人工示教;
③ 每条时长 —— 提前结束(<30s封顶)≈用户按→(成功或明确失败), 供分阶段标注参考。

执行=state×1e-3(毫度→度), 指令=action×57.3(rad→度)。臂 j1/j2/j3。FS=20。
用法: python sim/eval_pickplace2_analysis.py [eval_repo] [demo_repo]
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
import pandas as pd
from scipy.signal import welch

LR = os.path.expanduser("~/.cache/huggingface/lerobot/enolab-guo")
EVAL = sys.argv[1] if len(sys.argv) > 1 else "eval_pickplace2_100k"
DEMO = sys.argv[2] if len(sys.argv) > 2 else "pickplace2_train"
FS, R2D, MAXLAG, ARM = 20.0, 180 / np.pi, 8, [0, 1, 2]


def load(repo):
    out = []
    for p in sorted(glob.glob(f"{LR}/{repo}/data/chunk-*/episode_*.parquet"),
                    key=lambda q: int(q.split("_")[-1][:-8])):
        df = pd.read_parquet(p)
        out.append((np.stack(df["action"].to_numpy()).astype(float) * R2D,
                    np.stack(df["observation.state"].to_numpy()).astype(float) * 1e-3,
                    np.stack(df["observation.imu_real"].to_numpy()).astype(float)))
    return out


def bp(im):
    a = np.linalg.norm(im[:, :3] - im[:, :3].mean(0), axis=1)
    f, P = welch(a, fs=FS, nperseg=min(256, len(a)))
    m = (f >= 2.0) & (f < 4.5)
    return float(np.trapezoid(P[m], f[m]))


def best_lag(e, c):
    bd, br = 0, -2.0
    for d in range(MAXLAG + 1):
        a, b = e[d:], c[:len(c) - d]
        if len(a) < 20 or a.std() < 1e-9 or b.std() < 1e-9:
            continue
        r = np.corrcoef(a, b)[0, 1]
        if r > br:
            br, bd = r, d
    return bd, br


def main():
    ev, dm = load(EVAL), load(DEMO)
    dbp = [bp(im) for _, _, im in dm]
    dmed = float(np.median(dbp))
    print(f"\n{'='*66}\n{EVAL} (N={len(ev)}, 无TE) vs demo {DEMO} (N={len(dm)})\n{'='*66}")
    print(f"demo IMU 带功率(2-4.5Hz) 中位 = {dmed:.3f}\n")
    print(f"{'ep':>3} {'时长s':>5} {'corr':>6} {'滞后ms':>6} {'|gap|°':>6} {'IMU bp':>7} {'vs demo':>8}")
    rows = []
    for i, (act, st, im) in enumerate(ev):
        n = len(act)
        corr = np.mean([best_lag(st[:, j], act[:, j])[1] for j in ARM])
        lag = np.mean([best_lag(st[:, j], act[:, j])[0] for j in ARM]) / FS * 1e3
        gap = np.mean([np.abs(st[:, j] - act[:, j]).mean() for j in ARM])
        b = bp(im)
        rows.append((n / FS, corr, lag, gap, b))
        print(f"{i:>3} {n/FS:>5.0f} {corr:>6.3f} {lag:>6.0f} {gap:>6.2f} {b:>7.3f} {b/dmed:>7.2f}×")
    r = np.array(rows)
    print(f"\n中位: 时长={np.median(r[:,0]):.0f}s corr={np.median(r[:,1]):.3f} "
          f"滞后={np.median(r[:,2]):.0f}ms |gap|={np.median(r[:,3]):.2f}° "
          f"IMU={np.median(r[:,4]):.3f} ({np.median(r[:,4])/dmed:.2f}× demo)")
    nearly = sum(1 for x in r[:, 0] if x < 28)
    print(f"提前结束(<28s) {nearly}/{len(ev)} 条 —— 分阶段成功率需人工标注(review_eval.py)")


if __name__ == "__main__":
    main()
