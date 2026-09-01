#!/usr/bin/env python3
"""**事前停止门**：高速电机速度/电流是否与腕部 3–9 Hz 残差稳定相干？

残差 = 真机 IMU(抗混叠) − G-q-v1 预测，带通 3–9 Hz，100 Hz 时基。

⚠ 多元相干的偏置必须扣
----------------------
q 路输入、n 段平均时，**即使输入与输出毫无关系**，估计出来的多元相干期望也是
`q/n`。6 路输入、11 段平均 ⇒ 零假设水平就有 0.55 —— 不扣偏置的话
「γ²=0.6」什么都不说明。故本脚本一律报告：
    原始 γ²  ·  零假设水平 q/n  ·  扣偏置后 (γ²−q/n)/(1−q/n)
门禁判据用**扣偏置后**的值。

分三组输入，因为它们的信息含量不同（2026-09-01 已测）：
  · speed  —— 与 d(q)/dt 同源(0.5–3 Hz ρ²=0.93–0.999)，基本是运动学，G 已有
  · current —— **唯一真正新增的通道**(力/力矩侧)
  · 两者合并

用法:
    python -u sim/step3_canobs/gate_coherence.py
"""
from __future__ import annotations
import gc, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g", ROOT / "sim/step3_canobs"):
    sys.path.insert(0, str(_p))

EPS = ("ep0", "ep1")
FS = 100.0
NPERSEG = 256
BAND = (3.0, 9.0)
OUT = ROOT / "outputs/step3_canobs"


def multi_coherence(X, y, fs, nperseg):
    """MISO 多元相干 γ²(f)。X:(N,q) 输入, y:(N,) 输出。→ (f, γ², n_seg)"""
    from scipy.signal import csd
    q = X.shape[1]
    f, _ = csd(y, y, fs=fs, nperseg=nperseg)
    nf = len(f)
    Gxx = np.zeros((nf, q, q), complex)
    Gxy = np.zeros((nf, q), complex)
    for i in range(q):
        for j in range(q):
            Gxx[:, i, j] = csd(X[:, i], X[:, j], fs=fs, nperseg=nperseg)[1]
        Gxy[:, i] = csd(X[:, i], y, fs=fs, nperseg=nperseg)[1]
    Gyy = csd(y, y, fs=fs, nperseg=nperseg)[1].real
    g2 = np.zeros(nf)
    for k in range(nf):
        try:
            sol = np.linalg.solve(Gxx[k] + 1e-18 * np.eye(q), Gxy[k])
        except np.linalg.LinAlgError:
            continue
        g2[k] = np.real(np.vdot(Gxy[k], sol)) / max(Gyy[k], 1e-30)
    n_seg = max(1, int(2 * len(y) / nperseg) - 1)          # 50% 重叠
    return f, np.clip(g2, 0, 1), n_seg


def main() -> int:
    gc.disable()
    from scipy.signal import butter, filtfilt, firwin
    from scipy.linalg import orthogonal_procrustes
    import g_q_v1 as GQ
    from ceiling_200hz import load_streams

    def lp(x, fc, fs):
        return filtfilt(*butter(4, fc / (fs / 2)), x, axis=0)

    def bp(x, lo, hi, fs):
        k = firwin(int(2 * fs) | 1, [lo, hi], fs=fs, pass_zero=False)
        x = np.atleast_2d(x.T).T
        return np.stack([np.convolve(x[:, j], k, mode="same")
                         for j in range(x.shape[1])], 1)

    print("事前停止门：电机速度/电流 vs 腕部 3–9 Hz 残差的多元相干\n")
    per_ep = {}
    for nm in EPS:
        d = json.loads((ROOT / f"outputs/probe/replay_{nm}_A1.json").read_text())
        g100, q200, tg_i, tg_q = load_streams(nm)
        off = d["can_hs_meta"]["clock"]["offset_realtime_minus_monotonic_s"]
        cmd = np.asarray(d["cmd"], float)
        ts, tr = np.asarray(d["t_send"], float), np.asarray(d["t"], float)
        _q, _g, hi, read, fs_hi = GQ._sim(cmd, t_send=ts, t_read=tr, return_hi=True)
        t_hi = tr[0] + (np.arange(len(hi)) - read[0]) / fs_hi
        gq100 = np.stack([np.interp(tg_i, t_hi, lp(hi, 45.0, fs_hi)[:, k])
                          for k in range(3)], 1)
        A = lp(g100, 45.0, FS)                               # 真机(抗混叠口径)
        R, _ = orthogonal_procrustes(gq100, A)
        res = A - gq100 @ R                                  # 腕部残差, 三轴
        # 电机侧输入 → 同一 100 Hz 网格
        H = np.asarray(d["can_hs"], float)
        SP = np.zeros((len(tg_i), 6)); CU = np.zeros((len(tg_i), 6))
        for j in range(1, 7):
            m = H[:, 1] == j
            t_ = H[m, 0] - off
            SP[:, j - 1] = np.interp(tg_i, t_, lp(H[m, 2] * 1e-3, 45.0, 200.0))
            CU[:, j - 1] = np.interp(tg_i, t_, lp(H[m, 3] * 1e-3, 45.0, 200.0))
        per_ep[nm] = {"res": bp(res, *BAND, FS), "SP": bp(SP, *BAND, FS),
                      "CU": bp(CU, *BAND, FS)}

    sel = lambda f: (f >= BAND[0]) & (f <= BAND[1])
    curves = {}
    print(f"  {'集':<6}{'输入':<12}{'原始 γ²':>10}{'零假设 q/n':>12}{'扣偏置后':>11}{'段数':>7}")
    for nm in EPS:
        P = per_ep[nm]
        for lbl, X in (("speed", P["SP"]), ("current", P["CU"]),
                       ("speed+cur", np.hstack([P["SP"], P["CU"]]))):
            g2s, raws, nulls = [], [], []
            for k in range(3):
                f, g2, nseg = multi_coherence(X, P["res"][:, k], FS, NPERSEG)
                w = np.abs(P["res"][:, k]).std() ** 2          # 按该轴能量加权
                g2s.append((g2, w)); raws.append(np.mean(g2[sel(f)]))
                nulls.append(X.shape[1] / nseg)
            W = sum(w for _, w in g2s)
            gm = sum(g * w for g, w in g2s) / W
            raw = float(np.average(raws, weights=[w for _, w in g2s]))
            null = float(np.mean(nulls))
            corr = max(0.0, (raw - null) / (1 - null))
            curves[(nm, lbl)] = (f, gm)
            print(f"  {nm:<6}{lbl:<12}{raw:>10.3f}{null:>12.3f}{corr:>11.3f}{nseg:>7}")
    # 两集的 γ²(f) 曲线一致性（在 3–9 Hz 带内）
    print()
    for lbl in ("speed", "current", "speed+cur"):
        a = curves[("ep0", lbl)][1][sel(curves[("ep0", lbl)][0])]
        b = curves[("ep1", lbl)][1][sel(curves[("ep1", lbl)][0])]
        r = float(np.corrcoef(a, b)[0, 1])
        print(f"  两集 γ²(f) 曲线相关 ({lbl}): {r:+.3f}   "
              f"⚠ **同一会话内**, 不是跨会话")
    np.savez(OUT / "gate_coherence.npz",
             **{f"{nm}_{lbl}": np.stack(v) for (nm, lbl), v in curves.items()})
    print(f"\n  → {OUT/'gate_coherence.npz'}")
    print("\n  口径: 2 条集同属 2026-09-01 一个会话 ⇒ 预注册要求的"
          "「两个会话各自独立 ≥ 0.35」与「两会话曲线相关 ≥ 0.5」**无法判定**。")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
