#!/usr/bin/env python3
"""P0.4 — 理想参照两种生成法的系统对比: 运动学渲染(kin) vs 伺服仿真同步(dyn)。

kin: 指令样条 → 规定(q,q̇,q̈) → 虚拟IMU (ideal_imu_kinematic, 零标定依赖)
dyn: 指令喂ctrl → kp/阻尼伺服真积分 → 虚拟IMU (dyn_replay_imu, 依赖标定)

维度: ①与real的对齐(逐通道最优滞后/corr) ②低频(<2Hz)跟踪RMS ③高频(2-9Hz)
带功率比 ideal/real ④chunk边界渲染(deploy: 边界±0.25s HP能量/段中) ⑤运行时。
口径: 全部 20Hz 抗混叠; 每通道去DC(TSND零偏在DC层, 不影响频带指标)。

用法: python sim/compare_ideal_methods.py [--eps 10]
输出: outputs/sim/ideal_imu/method_comparison.{json,png} + stdout 表
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np

from ideal_imu_kinematic import MODEL_PATH, load_episode, build_spline, ideal_imu_traj
from dyn_replay_imu import aa_downsample, dyn_replay, parse_jkp

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs/sim/ideal_imu"
DATASETS = [
    ("enolab-guo/pickplace2_train",     "pp2-demo"),
    ("enolab-guo/eval_pickplace2_100k", "pp2-deploy"),
    ("enolab-guo/insert_train",         "ins-demo"),
    ("enolab-guo/eval_insert_100k",     "ins-deploy"),
]
LF, HF = (0.3, 2.0), (2.0, 9.0)


def kin_ideal_20(repo: str, ep: int, smooth_cmd: float | None = None):
    """smooth_cmd: 意图版理想 — 渲染前对 20Hz 指令零相位低通(Hz), None=字面版。"""
    import mujoco
    d = load_episode(repo, ep)
    fps = d["fps"]
    T = len(d["action"])
    t20 = np.arange(T) / fps
    cmd = d["action"]
    if smooth_cmd:
        from scipy.signal import butter, filtfilt
        cmd = filtfilt(*butter(2, smooth_cmd / (fps / 2)), cmd, axis=0)
    sp = build_spline(t20, cmd, "cubic")
    stride = int(round(200.0 / fps))
    hi_fps = stride * fps
    t_hi = np.arange(int(round(t20[-1] * hi_fps)) + 1) / hi_fps
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site")
    t0 = time.perf_counter()
    _, imu_hi, _, _ = ideal_imu_traj(model, data, sp, t_hi, sid)
    rt = time.perf_counter() - t0
    return aa_downsample(imu_hi, hi_fps, fps, T), d, rt


def bandpower(x20: np.ndarray, fps: float, lo: float, hi: float) -> np.ndarray:
    X = np.fft.rfft((x20 - x20.mean(0)) * np.hanning(len(x20))[:, None], axis=0)
    f = np.fft.rfftfreq(len(x20), 1 / fps)
    m = (f >= lo) & (f <= hi)
    return (np.abs(X[m]) ** 2).sum(0)


def lag_corr_ch(ideal: np.ndarray, real: np.ndarray, maxlag=8):
    lags, corrs = [], []
    for c in range(ideal.shape[1]):
        best = (0, -2.0)
        a0, b0 = ideal[:, c], real[:, c]
        if a0.std() < 1e-9 or b0.std() < 1e-9:
            lags.append(0); corrs.append(np.nan); continue
        for L in range(maxlag + 1):
            a, b = (a0[:-L], b0[L:]) if L else (a0, b0)
            cc = float(np.corrcoef(a, b)[0, 1])
            if cc > best[1]:
                best = (L, cc)
        lags.append(best[0]); corrs.append(best[1])
    return np.array(lags), np.array(corrs)


def lf_rms(ideal: np.ndarray, real: np.ndarray, fps: float, lags: np.ndarray):
    from scipy.signal import butter, filtfilt
    ba = butter(3, LF[1] / (fps / 2))
    out = []
    for c in range(ideal.shape[1]):
        L = int(lags[c])
        a, b = (ideal[:-L, c], real[L:, c]) if L else (ideal[:, c], real[:, c])
        a, b = filtfilt(*ba, a - a.mean()), filtfilt(*ba, b - b.mean())
        out.append(float(np.sqrt(((a - b) ** 2).mean())))
    return np.array(out)


def boundary_ratio(x20: np.ndarray, fps: float, period=100, halfw=5, guard=20):
    from scipy.signal import butter, filtfilt
    hp = filtfilt(*butter(3, 2.0 / (fps / 2), "high"), x20 - x20.mean(0), axis=0)
    e = (hp ** 2).sum(1)
    n = len(x20)
    bmask = np.zeros(n, bool)
    gmask = np.ones(n, bool)
    for b in range(1, n // period + 1):
        i = b * period
        if i < n:
            bmask[max(0, i - halfw):min(n, i + halfw + 1)] = True
        gmask[max(0, i - guard):min(n, i + guard)] = False
    gmask[:guard] = False
    if not bmask.any() or not gmask.any():
        return np.nan
    return float(e[bmask].mean() / max(e[gmask].mean(), 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, default=10, help="每数据集最多集数")
    ap.add_argument("--j-kp", default="1:5000")
    args = ap.parse_args()
    j_kp = parse_jkp(args.j_kp)

    results = {}
    for repo, tag in DATASETS:
        root = Path.home() / ".cache/huggingface/lerobot" / repo
        if not root.exists():
            print(f"[skip] {repo} 不存在"); continue
        total = json.loads((root / "meta/info.json").read_text())["total_episodes"]
        eps = list(range(min(args.eps, total)))
        results[tag] = []
        for ep in eps:
            try:
                kin20, d, rt_k = kin_ideal_20(repo, ep)
            except Exception as e:
                print(f"[{tag} ep{ep}] kin失败: {e}"); continue
            if "imu_real" not in d:
                print(f"[{tag}] 无 imu_real, 跳过数据集"); results.pop(tag); break
            t0 = time.perf_counter()
            _, imu_hi, fs_hi, _ = dyn_replay(repo, ep, j_kp)
            rt_d = time.perf_counter() - t0
            fps = d["fps"]; T = len(d["action"])
            dyn20 = aa_downsample(imu_hi, fs_hi, fps, T)
            real = d["imu_real"][:T]
            n = min(len(real), len(kin20), len(dyn20))
            real, kin20, dyn20 = real[:n], kin20[:n], dyn20[:n]
            r = {"ep": ep, "rt_kin": rt_k, "rt_dyn": rt_d}
            for name, ideal in (("kin", kin20), ("dyn", dyn20)):
                lags, corrs = lag_corr_ch(ideal, real)
                r[f"{name}_lag_g"] = float(np.nanmean(lags[3:]))
                r[f"{name}_corr_g"] = float(np.nanmean(corrs[3:]))
                r[f"{name}_lfrms_g"] = float(lf_rms(ideal, real, fps, lags)[3:].mean())
                r[f"{name}_hf_a"] = float((bandpower(ideal[:, :3], fps, *HF) /
                                           bandpower(real[:, :3], fps, *HF)).mean())
                r[f"{name}_hf_g"] = float((bandpower(ideal[:, 3:], fps, *HF) /
                                           bandpower(real[:, 3:], fps, *HF)).mean())
                if "deploy" in tag:
                    r[f"{name}_bnd"] = boundary_ratio(ideal, fps)
            if "deploy" in tag:
                r["real_bnd"] = boundary_ratio(real, fps)
            results[tag].append(r)
            print(f"[{tag} ep{ep}] kin: corr_g={r['kin_corr_g']:.3f} lag={r['kin_lag_g']:.1f} | "
                  f"dyn: corr_g={r['dyn_corr_g']:.3f} lag={r['dyn_lag_g']:.1f}")

    # ---- 汇总表 ----
    def med(tag, key):
        v = [r[key] for r in results[tag] if key in r and np.isfinite(r[key])]
        return float(np.median(v)) if v else float("nan")

    print("\n================ 汇总(逐集中位数) ================")
    print(f"{'数据集':12s} | {'方法':4s} | 滞后(帧) | corr(gyro) | LF-RMS(gyro) | HF比 acc | HF比 gyro | 边界比 | 耗时/集")
    for tag in results:
        for name in ("kin", "dyn"):
            b = f"{med(tag, name+'_bnd'):6.1f}" if "deploy" in tag else "   -  "
            print(f"{tag:12s} | {name:4s} |   {med(tag,name+'_lag_g'):4.1f}   |   {med(tag,name+'_corr_g'):6.3f}   |"
                  f"    {med(tag,name+'_lfrms_g'):6.4f}   |  {med(tag,name+'_hf_a'):6.2f}  |  {med(tag,name+'_hf_g'):6.2f}   | {b} | {med(tag,'rt_'+name):5.2f}s")
        if "deploy" in tag:
            print(f"{tag:12s} | real |    -     |     -      |      -       |    -     |    -      | {med(tag,'real_bnd'):6.1f} |   -")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "method_comparison.json").write_text(json.dumps(results, indent=1))
    print(f"\n→ {OUT_DIR/'method_comparison.json'}")


if __name__ == "__main__":
    main()
