#!/usr/bin/env python3
"""#77 — 真机偏离审计: 执行(state) vs 理想=ACT命令(action)。

老师纠偏(2026-06-24): 目标=让真机执行逼近理想动作; 先量化"偏离理想多少/有没有过冲"。
理想 = ACT 命令(action)。偏离 = 执行关节(observation.state) − 命令。逐臂关节(j1/j2/j3):
  · 滞后 lag: 互相关找执行落后命令几帧(控制器响应延迟);
  · 跟踪 RMS(raw): RMS(执行 − 命令)(含滞后);
  · 滞后补偿后偏离, 拆: 低频(<2Hz)=轨迹失真/过冲(老师关心的"逼近理想") vs 高频(>2Hz)=振动(命令无关#74);
  · corr(执行,命令): 形状吻合度。
输出: 控制台分数据集表 + outputs/eval/real_deviation_audit.png(时域 命令vs执行 + 低/高频偏离条)。

用法: /home/enolab-guo/miniconda3/envs/lerobot/bin/python sim/real_deviation_audit.py
"""
from __future__ import annotations
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

LR = Path.home() / ".cache/huggingface/lerobot"
REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/eval/real_deviation_audit.png"
FS, ARM = 30.0, [0, 1, 2]
MDEG_TO_RAD = np.pi / 180.0 / 1000.0
R2D = 180.0 / np.pi
MAXLAG = 15                                  # ±0.5s
_LO = butter(4, 2.0 / 15.0, "low")           # <2Hz = 轨迹层
_HI = butter(4, 2.0 / 15.0, "high")          # >2Hz = 振动层
DATASETS = {"eval_s5tok_A (ACT部署)": "enolab-guo/eval_s5tok_A",
            "demo v3 (teleop plant)": "local/piper_pick_place_imu_v3"}


def load_eps(nsname, skip=1):
    eps = []
    for pq in sorted(glob.glob(str(LR / nsname / "data/chunk-*/episode_*.parquet")))[skip:]:
        df = pd.read_parquet(pq)
        act = np.stack(df["action"].to_numpy()).astype(np.float64)
        st = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
        st = st.copy(); st[:, :6] *= MDEG_TO_RAD
        eps.append((act, st))
    return eps


def best_lag(e, c):
    """执行 e 落后命令 c 几帧(0..MAXLAG): 最大化 corr(e[d:], c[:len-d])。"""
    best_d, best_r = 0, -2.0
    for d in range(0, MAXLAG + 1):
        a, b = e[d:], c[:len(c) - d]
        if len(a) < 30 or a.std() < 1e-9 or b.std() < 1e-9:
            continue
        r = np.corrcoef(a, b)[0, 1]
        if r > best_r:
            best_r, best_d = r, d
    return best_d


def audit(eps):
    raw, dev, lo, hi, mot = [], [], [], [], []      # 池化样本(rad)
    corrs, lags = [], []
    for act, st in eps:
        for j in ARM:
            c, e = act[:, j], st[:, j]
            if len(c) < 60 or c.std() < 1e-9:
                continue
            d = best_lag(e, c)
            ec, cc = e[d:], c[:len(c) - d]          # 滞后对齐
            dv = ec - cc                            # 滞后补偿后偏离
            raw.append(e - c); dev.append(dv)
            lo.append(filtfilt(*_LO, dv)); hi.append(filtfilt(*_HI, dv))
            mot.append(cc - cc.mean())
            corrs.append(np.corrcoef(ec, cc)[0, 1]); lags.append(d)
    cat = lambda L: np.concatenate(L)
    rms = lambda x: float(np.sqrt((x ** 2).mean()) * R2D)   # → 度
    return dict(corr=float(np.mean(corrs)), lag_ms=float(np.mean(lags)) / FS * 1e3,
                raw=rms(cat(raw)), dev=rms(cat(dev)), lo=rms(cat(lo)), hi=rms(cat(hi)),
                mot=rms(cat(mot)),
                lo_frac=float((cat(lo) ** 2).sum() / ((cat(dev) ** 2).sum() + 1e-18)))


def main():
    res = {}
    for name, ns in DATASETS.items():
        eps = load_eps(ns)
        if not eps:
            print(f"[skip] {name}: 无数据"); continue
        res[name] = audit(eps); res[name]["n"] = len(eps)

    print(f"\n{'='*92}\n真机偏离审计: 执行(state) vs 理想=ACT命令(action), 臂关节 j1/j2/j3 (单位: 度)\n{'='*92}")
    print(f"{'数据集':<24}{'n':>3} {'corr':>6} {'lag(ms)':>8} {'跟踪RMS':>8} {'对齐后':>7} "
          f"{'低频(轨迹)':>10} {'高频(振动)':>10} {'低频占比':>8} {'运动幅度':>8}")
    print("-" * 92)
    for name, r in res.items():
        print(f"{name:<24}{r['n']:>3} {r['corr']:>6.3f} {r['lag_ms']:>8.0f} {r['raw']:>7.2f}° {r['dev']:>6.2f}° "
              f"{r['lo']:>9.2f}° {r['hi']:>9.2f}° {100*r['lo_frac']:>7.0f}% {r['mot']:>7.2f}°")
    print("-" * 92)
    print("读法(回答老师⑥): lag=控制器响应延迟; 低频(轨迹)=滞后补偿后仍存在的轨迹失真/过冲(=老师'逼近理想'问题);")
    print("  高频=振动(命令无关#74)。若 低频(轨迹)≪运动幅度 且 低频占比小 → 真机其实跟得好、没大过冲(问题小);")
    print("  若 低频(轨迹)大 → 确有过冲/失真可补偿(前馈补偿有意义)。")

    # ---- 图: ACT 数据集 一段时域 命令vs执行 + 低/高频偏离条 ----
    actname = next((k for k in res if "ACT" in k), None)
    if actname:
        eps = load_eps(DATASETS[actname]); act, st = eps[0]
        j = 1                                        # j2 (运动较大)
        c, e = act[:, j] * R2D, st[:, j] * R2D
        # 选高运动窗口
        v = np.abs(np.diff(c, prepend=c[0]))
        W = 150; s0 = int(np.argmax([v[s:s+W].sum() for s in range(0, max(1, len(c)-W))]))
        sl = slice(s0, s0 + W); t = np.arange(sl.start, sl.stop) / FS
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
        ax[0].plot(t, c[sl], "C0", lw=1.4, label="ACT command (ideal)")
        ax[0].plot(t, e[sl], "C3", lw=1.2, label="real executed (state)")
        for b in range(((sl.start)//100+1)*100, sl.stop, 100):
            ax[0].axvline(b/FS, color="gray", ls=":", lw=.8)
        ax[0].set(xlabel="s", ylabel="j2 angle (deg)",
                  title=f"{actname}: command vs executed (灰线=chunk边界)")
        ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)
        # 偏离分解条
        names = list(res); x = np.arange(len(names)); w = 0.38
        ax[1].bar(x - w/2, [res[n]["lo"] for n in names], w, color="C1", label="低频 低频(轨迹/过冲)")
        ax[1].bar(x + w/2, [res[n]["hi"] for n in names], w, color="C4", label="高频(振动)")
        ax[1].plot(x, [res[n]["mot"] for n in names], "k_", ms=22, label="运动幅度(参考)")
        ax[1].set_xticks(x); ax[1].set_xticklabels([n.split(" (")[0] for n in names], fontsize=8)
        ax[1].set(ylabel="deviation RMS (deg)", title="偏离分解: 低频(轨迹) vs 高频(振动)")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis="y")
        fig.suptitle("真机执行 vs ACT命令(理想): 偏离主要是滞后/振动还是轨迹过冲?", fontsize=11)
        fig.tight_layout(); OUT.parent.mkdir(parents=True, exist_ok=True); fig.savefig(OUT, dpi=130)
        print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
