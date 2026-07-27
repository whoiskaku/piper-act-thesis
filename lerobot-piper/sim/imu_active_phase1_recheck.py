#!/usr/bin/env python3
"""旧 Phase1 (30Hz, 单相机) 活动段·全频段 IMU 复查 —— 检验"no-TE ≈ demo"是否是整条平均的稀释假象。

数据: eval_phase1_noimu_no_te (25条, 按 review CSV 分 成功17/失败8) vs demo pickplace_clean。
口径与 imu_active_broadband.py 相同: 活动段(关节速度门限) + 全频RMS + 分频段, 逐条→中位。FS=30。
用法: python sim/imu_active_phase1_recheck.py
"""
from __future__ import annotations
import csv, glob, os
import numpy as np
import pandas as pd
from scipy.signal import welch

LR = os.path.expanduser("~/.cache/huggingface/lerobot/enolab-guo")
REPO_ROOT = os.path.expanduser("~/claude-test-2/lerobot-piper")
FS = 30.0
BANDS = [(0.5, 2.0), (2.0, 4.5), (4.5, 10.0), (10.0, 15.0)]


def load(repo):
    out = []
    for p in sorted(glob.glob(f"{LR}/{repo}/data/chunk-*/episode_*.parquet"),
                    key=lambda q: int(q.split("_")[-1][:-8])):
        df = pd.read_parquet(p)
        st = np.stack(df["observation.state"].to_numpy()).astype(float)[:, :6] * 1e-3
        im = np.stack(df["observation.imu_real"].to_numpy()).astype(float)
        out.append((int(p.split("_")[-1][:-8]), st, im))
    return out


def active_mask(st, w=13):
    v = np.linalg.norm(np.diff(st, axis=0), axis=1)
    v = np.convolve(v, np.ones(w) / w, "same")
    m = v > 0.05 * np.percentile(v, 99)
    m = np.convolve(m.astype(float), np.ones(w), "same") > 0
    return np.append(m, m[-1])


def ep_metrics(st, im):
    m = active_mask(st)
    a = np.linalg.norm(im[:, :3] - im[:, :3].mean(0), axis=1)[m]
    if len(a) < 96:
        return None
    f, P = welch(a - a.mean(), fs=FS, nperseg=min(256, len(a)))
    rms = float(np.sqrt(np.mean((a - a.mean()) ** 2)))
    bps = [float(np.trapezoid(P[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)])) for lo, hi in BANDS]
    return rms, bps, m.sum() / FS


def main():
    stages = {}
    for r in csv.reader(open(f"{REPO_ROOT}/outputs/review/enolab-guo__eval_phase1_noimu_no_te_eval.csv")):
        if len(r) >= 2:
            stages[int(r[0])] = int(r[1])

    ev = load("eval_phase1_noimu_no_te")
    dm = load("pickplace_clean")
    groups = {
        "demo": [ep_metrics(st, im) for _, st, im in dm],
        "deploy-成功": [ep_metrics(st, im) for ep, st, im in ev if stages.get(ep) == 3],
        "deploy-失败": [ep_metrics(st, im) for ep, st, im in ev if stages.get(ep) == 0],
        "deploy-全部": [ep_metrics(st, im) for ep, st, im in ev],
    }
    groups = {k: [x for x in v if x] for k, v in groups.items()}

    print(f"\n{'='*84}\n旧 Phase1 复查 (30Hz): 活动段·全频段 IMU  [逐条→中位]\n{'='*84}")
    print(f"{'':14}{'n':>3}{'活动s':>7}{'全频RMS':>9}" + "".join(f"{lo}-{hi}Hz".rjust(12) for lo, hi in BANDS))
    med = {}
    for name, ms in groups.items():
        rms_m = np.median([x[0] for x in ms])
        bp_m = np.median([x[1] for x in ms], 0)
        dur = np.median([x[2] for x in ms])
        med[name] = (rms_m, bp_m)
        print(f"  {name:<12}{len(ms):>3}{dur:>7.1f}{rms_m:>9.3f}" + "".join(f"{b:>12.3f}" for b in bp_m))
    for name in ("deploy-成功", "deploy-失败", "deploy-全部"):
        rr = med[name][0] / med["demo"][0]
        br = med[name][1] / med["demo"][1]
        print(f"  → {name}/demo: 全频RMS={rr:.2f}x " +
              " ".join(f"{lo}-{hi}={x:.2f}x" for (lo, hi), x in zip(BANDS, br)))


if __name__ == "__main__":
    main()
