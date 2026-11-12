#!/usr/bin/env python3
"""Stage 5 — transfer-chain 诊断（A=纯ACT vs B=ACT+RefineNet, 各 8 eps）。

不追加实验，仅在已有 16 episode 上严谨重算。修正前一版的两个问题：
  (1) 主指标改 **绝对 hp_rms**（5Hz zero-phase Butterworth 高通后 RMS），不用 fraction
      （fraction 有分母陷阱）。
  (2) 补 **中间层 joint state**，把链路拆开定位瓶颈：
        commanded action → measured joint state(电机反馈) → 机械结构 → IMU
      判读：
        action↓ 但 state 不↓  → 低层控制器/量化/执行误差吞掉收益
        state↓ 但 IMU 不↓     → IMU hf 由结构共振/传感器噪声/接触主导
        IMU hp_rms↓ 但 fraction 不↓ → 只是 fraction 分母陷阱, 方法其实有效

分阶段（按 gripper 事件 + chunk 边界）：full / approach / carry / boundary。
谱诊断：PSD 0-5/5-10/10-15Hz；coherence action↔state、state↔IMU（5-15Hz 带均值）。
统计：unpaired（A/B 非配对录制）→ bootstrap 95% CI of mean diff；附 grasp-cycle proxy。

单位：action(rad) 与 state(mdeg→rad, 臂关节) 统一到 rad 后可比。active joints=j1/j2/j3/grip。

用法: python scripts/stage5_transfer_diagnosis.py
"""
from __future__ import annotations
import glob, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch, coherence

LR = Path.home() / ".cache/huggingface/lerobot/enolab-guo"
A_DIR, B_DIR = LR / "eval_refine_A", LR / "eval_refine_B"
OUT = Path(__file__).resolve().parents[1] / "outputs/eval/stage5_diag"
FS = 30.0
ACTIVE = [0, 1, 2, 6]            # j1/j2/j3/grip
MDEG_TO_RAD = math.pi / 180.0 / 1000.0
CHUNK = 100
_B, _A = butter(4, 5.0 / (FS / 2.0), btype="high")
RNG = np.random.default_rng(0)


def hp_rms(x):                   # x:(T,C) -> per-col RMS of 5Hz HP, then mean
    if len(x) < 13:
        return float("nan")
    y = filtfilt(_B, _A, x, axis=0)
    return float(np.sqrt((y ** 2).mean(axis=0)).mean())


def load_ep(pq):
    df = pd.read_parquet(pq)
    act = np.stack(df["action"].to_numpy()).astype(np.float64)              # (T,7) rad
    st = np.stack(df["observation.state"].to_numpy()).astype(np.float64)    # (T,7) mdeg
    st_rad = st.copy(); st_rad[:, :6] = st[:, :6] * MDEG_TO_RAD             # 臂→rad
    imu = np.stack(df["observation.imu_real"].to_numpy()).astype(np.float64)  # (T,6)
    return act, st_rad, imu


def grasp_events(act, ch=6):
    g = act[:, ch]; dg = np.abs(np.diff(g))
    if dg.max() <= 0:
        return []
    ev = np.where(dg > dg.max() * 0.3)[0]
    out = []
    for e in ev:
        if not out or e - out[-1] > 15:
            out.append(int(e))
    return out


def phase_slices(T, ev):
    if len(ev) >= 2:
        a, r = ev[0], ev[-1]
    else:
        a, r = T // 4, 3 * T // 4
    bm = np.zeros(T, bool)
    for b in range(CHUNK, T, CHUNK):
        bm[max(0, b - 4):b + 4] = True
    return {"full": slice(0, T), "approach": slice(0, a), "carry": slice(a, r),
            "boundary": bm}


def sig_hp(arr, sl, cols=None):
    seg = arr[sl] if not isinstance(sl, np.ndarray) else arr[sl]
    if cols is not None:
        seg = seg[:, cols]
    return hp_rms(seg)


def collect(d):
    pqs = sorted(glob.glob(str(d / "data/chunk-*/episode_*.parquet")))
    rows = []
    for pq in pqs:
        act, st, imu = load_ep(pq); T = len(act); ph = phase_slices(T, grasp_events(act))
        rec = {}
        for pname, sl in ph.items():
            rec[f"action/{pname}"] = sig_hp(act, sl, ACTIVE)
            rec[f"state/{pname}"] = sig_hp(st, sl, ACTIVE)
            rec[f"imu_accel/{pname}"] = sig_hp(imu, sl, [0, 1, 2])
            rec[f"imu_gyro/{pname}"] = sig_hp(imu, sl, [3, 4, 5])
        rows.append(rec)
    return rows, pqs


def coh_band(x, y, lo=5, hi=15):
    f, c = coherence(x, y, fs=FS, nperseg=min(256, len(x)))
    m = (f >= lo) & (f <= hi)
    return float(c[m].mean()) if m.any() else float("nan")


def psd_bands(x):
    f, p = welch(x, fs=FS, nperseg=min(256, len(x)))
    out = {}
    for lo, hi in [(0, 5), (5, 10), (10, 15)]:
        m = (f >= lo) & (f < hi); out[f"{lo}-{hi}"] = float(p[m].sum())
    return out


def boot_ci(a, b, n=10000):
    a, b = np.asarray(a), np.asarray(b)
    diffs = []
    for _ in range(n):
        diffs.append(RNG.choice(b, len(b)).mean() - RNG.choice(a, len(a)).mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(b) - np.mean(a)), float(lo), float(hi)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    A, pa = collect(A_DIR); B, pb = collect(B_DIR)
    keys = list(A[0].keys())
    Av = {k: np.array([r[k] for r in A]) for k in keys}
    Bv = {k: np.array([r[k] for r in B]) for k in keys}

    print(f"\n{'='*94}\nStage 5 transfer-chain (绝对 hp_rms, 5Hz Butterworth, active j1/j2/j3/grip; n=8 each)\n{'='*94}")
    print(f"{'signal/phase':>22} {'A mean':>10} {'B mean':>10} {'B/A':>6} {'Δ(B-A)':>9} {'95% CI':>20}")
    print("-" * 94)
    report = {}
    for sig in ["action", "state", "imu_accel", "imu_gyro"]:
        for ph in ["full", "approach", "carry", "boundary"]:
            k = f"{sig}/{ph}"; a, b = Av[k], Bv[k]
            d, lo, hi = boot_ci(a, b)
            ratio = b.mean() / a.mean() if a.mean() else float("nan")
            sig_mark = "" if (lo < 0 < hi) else " *"
            report[k] = {"A": float(a.mean()), "B": float(b.mean()), "ratio": ratio,
                         "diff": d, "ci": [lo, hi]}
            print(f"{k:>22} {a.mean():>10.5f} {b.mean():>10.5f} {ratio:>6.2f} {d:>+9.5f} "
                  f"[{lo:>+.4f},{hi:>+.4f}]{sig_mark}")
        print("-" * 94)

    # ---- 链路定位（full episode B/A ratio）----
    print("\n[链路定位] full-episode B/A ratio (越小=RefineNet 越降):")
    for sig in ["action", "state", "imu_accel", "imu_gyro"]:
        print(f"  {sig:>10}: {report[f'{sig}/full']['ratio']:.2f}")
    print("  判读: action↓但state≈1→控制器吞收益; state↓但IMU≈1→IMU非动作驱动; 都↓→传递成立")

    # ---- coherence + PSD (carry 段, 聚合所有 A+B episode) ----
    print("\n[coherence 5-15Hz] (mean over eps, active joints / IMU axes)")
    cohs = {"action-state": [], "state-imu_acc": []}
    psd_imu = {"0-5": [], "5-10": [], "10-15": []}
    for pq in pa + pb:
        act, st, imu = load_ep(pq); ev = grasp_events(act)
        a0, r0 = (ev[0], ev[-1]) if len(ev) >= 2 else (len(act)//4, 3*len(act)//4)
        sl = slice(a0, r0)
        for j in [0, 1, 2]:
            cohs["action-state"].append(coh_band(act[sl, j], st[sl, j]))
            cohs["state-imu_acc"].append(coh_band(st[sl, j], imu[sl, j]))
        for ax in range(3):
            pb_ = psd_bands(imu[sl, ax])
            for band in psd_imu:
                psd_imu[band].append(pb_[band])
    for k, v in cohs.items():
        print(f"  {k:>16}: {np.nanmean(v):.3f}  (低=弱关联)")
    tot = sum(np.sum(psd_imu[b]) for b in psd_imu)
    print("[IMU accel PSD 占比]", {b: f"{np.sum(psd_imu[b])/tot*100:.0f}%" for b in psd_imu})

    # ---- grasp-cycle proxy (gripper 完成 close→open 周期数) ----
    def cycles(pqs):
        n = []
        for pq in pqs:
            act, *_ = load_ep(pq); n.append(len(grasp_events(act)))
        return n
    print(f"\n[grasp proxy] gripper 事件数/ep  A={cycles(pa)}  B={cycles(pb)} (需人工确认真实成功率)")

    (OUT / "transfer_diagnosis.json").write_text(json.dumps(report, indent=2))
    print(f"\n[done] saved {OUT/'transfer_diagnosis.json'}")


if __name__ == "__main__":
    main()
