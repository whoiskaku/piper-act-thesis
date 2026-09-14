#!/usr/bin/env python3
"""③a 波形可行性检查（**固定预算，零机时**）—— 8 帧小波能否在 pp2 上同时满足 ① 与 ④？

背景（为什么不用单帧脉冲）
--------------------------
单帧脉冲在 20 Hz 上的加速度需求 = `2A/dt²`：8 mrad → 367°/s²，已达/超 `A_MAX`(307–391)。
而 **pp2 任务本身已占满 j2/j3 的加速度预算** —— 加"余量门"后这两个关节
**一帧可发位都没有**。实测注入不对称 `‖δa⁺+δa⁻‖/‖δa⁺‖`：
单帧 8 mrad 0.543 · 单帧 5 mrad 0.496 · 2 帧×4 mrad 0.354 · 3 帧×2.7 mrad 0.245，
而 ① 的信噪同时掉到 j2 1.8–2.1（门槛 3）⇒ **幅度够①就破④，够④就破①**。

⚠ 不做的事（用户 2026-09-02）
-----------------------------
**不**把仿真限幅后的扰动当"实际输入"、再用 H 扣掉不对称来让 ④ 通过。
限幅器是 pp2 命令→真机响应通道的一部分，H 也是从**命令**扰动辨识的；那样做会
**换掉正在检验的通道**，而且仿真值替代不了真机实际执行量。
若某波形在 pp2 上被非对称削波，**④ 不过就是该工况下的真实结果**。

本脚本只做**可行性筛查**：在仿真限幅器下估各幅度的 ①信噪 与 ④反对称度，
用来**锁定波形与幅度**，不用来判定 ③a。

固定预算（事前定死，不外延）
----------------------------
波形 = v2 的 8 帧 5 Hz 方波小波 `(1,1,-1,-1,1,1,-1,-1)`；幅度 {2,3,4,5,6} mrad。

判读
----
- 存在某幅度使**三关节**同时 ①信噪>3 且 ④反对称度<0.3 ⇒ 锁定该幅度，进入采集
- 不存在 ⇒ **如实收口**：现有 ③a 无法在该操作点验证线性增量模型，**不进 ③b**

用法:
    python -u sim/feas_3a_wavelet.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "sim/step2_g", ROOT / "sim/step3_canobs",
           ROOT / "sim/step4_gvnext"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

WAVELET = np.array([1, 1, -1, -1, 1, 1, -1, -1], float)   # v2：8 帧 5 Hz 方波
AMPS = (0.002, 0.003, 0.004, 0.005, 0.006)                # 固定预算，5 点
JOINTS = (1, 2, 4)
PERIOD = 40
LR = 24                     # 8 帧小波 + 16 帧尾巴（= H 的核长）
GATE_TH = 0.04
SNR_GATE, ASYM_GATE = 3.0, 0.3
N_EFF_K12 = 0.00545         # K=12 相干叠加后的 Δ_real 噪声（见 CHECKLIST_3a §5）
OUT = ROOT / "outputs/probe_3a/feasibility.json"


def main() -> int:
    import gc
    gc.disable()
    from data import load, to20, valid_files
    from optuna_ident_enc import accel_limited_dt
    from sim.step2_fidelity import A_MAX

    H = np.ascontiguousarray(np.load(ROOT / "outputs/step1_H/step1_H_v1.npz")["H"], float)
    dd = load(sorted(valid_files(ROOT / "outputs/step4_gvnext/s1"))[0])
    tr = np.asarray(dd["t_read"], float)
    dts = np.append(np.diff(tr), np.median(np.diff(tr)))
    cmd = np.degrees(np.asarray(dd["cmd"], float)[:, :6])
    T = len(cmd)
    g = np.linalg.norm(to20(dd["gyro100"]), axis=1)
    g = np.convolve(g, np.ones(5) / 5, "same")
    t20 = np.asarray(dd["tg_i"][::5], float)
    moving = np.interp(tr, t20[:len(g)], g) >= GATE_TH

    nw = len(WAVELET)
    print(f"波形 = 8 帧 5 Hz 方波小波 {WAVELET.astype(int).tolist()}   "
          f"响应窗 LR={LR}   固定预算 {len(AMPS)} 个幅度")
    print(f"加速度需求 2A/dt²（单帧脉冲同幅度作对照）· A_MAX {list(A_MAX.astype(int))}\n")
    print(f"  {'幅度':>7}{'关节':>5}{'实发':>6}{'注入反对称':>12}{'信噪':>8}"
          f"{'①>3':>7}{'④<0.3':>8}")
    res = {}
    for amp in AMPS:
        res[amp] = {}
        for j in JOINTS:
            da = np.zeros(T)
            idx = []
            k = 0
            for t in range(PERIOD, T - PERIOD - nw, PERIOD):
                if not moving[t]:
                    continue
                if k % len(JOINTS) == JOINTS.index(j):
                    da[t:t + nw] = amp * WAVELET
                    idx.append(t)
                k += 1
            if not idx:
                print(f"  {amp*1e3:>6.0f}m j{j+1:<4}{0:>6}{'—':>12}{'—':>8}")
                res[amp][j] = None
                continue
            a0 = accel_limited_dt(cmd[:, j], float(A_MAX[j]), dts)
            ap = np.radians(accel_limited_dt(cmd[:, j] + np.degrees(da),
                                             float(A_MAX[j]), dts) - a0)
            am = np.radians(accel_limited_dt(cmd[:, j] - np.degrees(da),
                                             float(A_MAX[j]), dts) - a0)
            w = np.zeros(T, bool)
            for t in idx:
                w[t:t + nw + 2] = True
            asym = float(np.linalg.norm(ap[w] + am[w]) / max(np.linalg.norm(ap[w]), 1e-30))
            # 响应用**奇分量** (ap−am)/2 经实测 H —— 与跨复放 ± 配对的分析口径一致
            odd = 0.5 * (ap - am)
            r = np.zeros((T, 3))
            for kk in range(len(H)):
                r[kk:] += np.outer(odd[:T - kk], H[kk][:, j])
            ww = np.zeros(T, bool)
            for t in idx:
                ww[t:t + LR] = True
            ne = N_EFF_K12 * np.sqrt(12 / max(4 * len(idx), 1))
            snr = float(np.sqrt(np.mean(r[ww] ** 2))) / ne
            res[amp][j] = {"n": len(idx), "asym": asym, "snr": snr}
            print(f"  {amp*1e3:>6.0f}m j{j+1:<4}{len(idx):>6}{asym:>12.3f}{snr:>8.2f}"
                  f"{'  ✓' if snr > SNR_GATE else '  ✗':>7}"
                  f"{'  ✓' if asym < ASYM_GATE else '  ✗':>8}")
        print()

    good = [a for a in AMPS if res[a] and all(
        res[a][j] and res[a][j]["snr"] > SNR_GATE and res[a][j]["asym"] < ASYM_GATE
        for j in JOINTS)]
    print("  ── 判读 ──")
    if good:
        pick = max(good, key=lambda a: min(res[a][j]["snr"] for j in JOINTS))
        print(f"  **存在可行幅度: {[f'{a*1e3:.0f}mrad' for a in good]} ⇒ "
              f"锁定 {pick*1e3:.0f} mrad，可进入采集**")
    else:
        best = min(AMPS, key=lambda a: max(
            (res[a][j]["asym"] if res[a][j] else 9e9) for j in JOINTS))
        print(f"  **不存在同时满足 ①信噪>3 与 ④反对称<0.3 的幅度**")
        print(f"  最接近的是 {best*1e3:.0f} mrad："
              + " · ".join(f"j{j+1} 信噪 {res[best][j]['snr']:.2f}/反对称 "
                           f"{res[best][j]['asym']:.3f}" for j in JOINTS
                           if res[best][j]))
        print(f"  ⇒ **如实收口：现有 ③a 无法在 pp2 这个操作点验证线性增量模型，不进 ③b。**")
        print(f"     **不**用 H 扣除限幅效应来改变这个判决（用户 2026-09-02）。")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"wavelet": WAVELET.tolist(), "amps_rad": list(AMPS), "lr": LR,
         "feasible_amps": [float(a) for a in good],
         "results": {str(a): {f"j{j+1}": v for j, v in d.items()} for a, d in res.items()},
         "gates": {"snr": SNR_GATE, "asym": ASYM_GATE},
         "caliber": ("仿真限幅器下的**可行性筛查**，用于锁定波形与幅度；"
                     "**不**用于判定 ③a。不得用 H 扣除限幅不对称来让 ④ 通过 —— "
                     "限幅器是 pp2 命令→真机响应通道的一部分。"),
         }, indent=1, ensure_ascii=False))
    print(f"\n  → {OUT}")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
