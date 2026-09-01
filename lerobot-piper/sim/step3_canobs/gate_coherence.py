#!/usr/bin/env python3
"""**事前停止门**：高速电机速度/电流是否与腕部 3–9 Hz 残差稳定相干？

残差 = 真机 IMU(抗混叠) − G-q-v1 预测，带通 3–9 Hz，100 Hz 时基。

⚠ 多元相干的偏置必须扣
----------------------
q 路输入、n 段平均时，**即使输入与输出毫无关系**，估计出来的多元相干期望也是
`q/n`。6 路输入、22 段平均 ⇒ 零假设水平就有 0.27 —— 不扣偏置的话
「γ²=0.6」什么都不说明。故一律报告：
    原始 γ²  ·  零假设水平 q/n  ·  扣偏置后 (γ²−q/n)/(1−q/n)
门禁判据用**扣偏置后**的值。

分三组输入，因为它们的信息含量不同（2026-09-01 已测）：
  · speed  —— 与 d(q)/dt 同源(0.5–3 Hz ρ²=0.93–0.999)，基本是运动学，G 已有
  · current —— **唯一真正新增的通道**(力/力矩侧)
  · 两者合并

时基口径：真机侧用 TSND **设备 ms**，CAN 用**内核时刻**。
诊断集的 20 Hz 控制环可能劣化，**落盘的 20 Hz 字段一律不用**。

用法:
    python -u sim/step3_canobs/gate_coherence.py                # 会话1 (A1)
    python -u sim/step3_canobs/gate_coherence.py --arms A1 A3   # 跨会话
"""
from __future__ import annotations
import argparse, gc, json, sys
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


def _lp(x, fc, fs):
    from scipy.signal import butter, filtfilt
    return filtfilt(*butter(4, fc / (fs / 2)), x, axis=0)


def _bp(x, lo, hi, fs):
    from scipy.signal import firwin
    k = firwin(int(2 * fs) | 1, [lo, hi], fs=fs, pass_zero=False)
    x = np.atleast_2d(x.T).T
    return np.stack([np.convolve(x[:, j], k, mode="same")
                     for j in range(x.shape[1])], 1)


def multi_coherence(X, y, fs=FS, nperseg=NPERSEG):
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
    return f, np.clip(g2, 0, 1), max(1, int(2 * len(y) / nperseg) - 1)


def episode_signals(nm: str, arm: str = "A1"):
    """→ {res, SP, CU}（均已带通到 3–9 Hz，100 Hz 网格）。"""
    from scipy.linalg import orthogonal_procrustes
    import g_q_v1 as GQ
    from ceiling_200hz import load_streams
    p = ROOT / f"outputs/probe/replay_{nm}_{arm}.json"
    d = json.loads(p.read_text())
    g100, q200, tg_i, tg_q = load_streams(nm, arm)
    off = d["can_hs_meta"]["clock"]["offset_realtime_minus_monotonic_s"]
    cmd = np.asarray(d["cmd"], float)
    ts, tr = np.asarray(d["t_send"], float), np.asarray(d["t"], float)
    _q, _g, hi, read, fs_hi = GQ._sim(cmd, t_send=ts, t_read=tr, return_hi=True)
    t_hi = tr[0] + (np.arange(len(hi)) - read[0]) / fs_hi
    gq100 = np.stack([np.interp(tg_i, t_hi, _lp(hi, 45.0, fs_hi)[:, k])
                      for k in range(3)], 1)
    A = _lp(g100, 45.0, FS)                          # 真机(抗混叠口径)
    res = A - gq100 @ orthogonal_procrustes(gq100, A)[0]
    H = np.asarray(d["can_hs"], float)
    SP = np.zeros((len(tg_i), 6)); CU = np.zeros((len(tg_i), 6))
    for j in range(1, 7):
        m = H[:, 1] == j
        t_ = H[m, 0] - off
        SP[:, j - 1] = np.interp(tg_i, t_, _lp(H[m, 2] * 1e-3, 45.0, 200.0))
        CU[:, j - 1] = np.interp(tg_i, t_, _lp(H[m, 3] * 1e-3, 45.0, 200.0))
    return {"res": _bp(res, *BAND, FS), "SP": _bp(SP, *BAND, FS),
            "CU": _bp(CU, *BAND, FS)}


def band_coh(X, res):
    """→ (原始 γ², 零假设 q/n, 扣偏置后, γ²(f) 能量加权曲线, f, nseg)。"""
    sel = None
    g2s, raws, nulls, ws = [], [], [], []
    for k in range(3):
        f, g2, nseg = multi_coherence(X, res[:, k])
        if sel is None:
            sel = (f >= BAND[0]) & (f <= BAND[1])
        w = float(res[:, k].std() ** 2)
        g2s.append(g2); ws.append(w)
        raws.append(float(np.mean(g2[sel]))); nulls.append(X.shape[1] / nseg)
    W = sum(ws)
    curve = sum(g * w for g, w in zip(g2s, ws)) / W
    raw = float(np.average(raws, weights=ws))
    null = float(np.mean(nulls))
    return raw, null, max(0.0, (raw - null) / (1 - null)), curve, f, nseg


def main() -> int:
    gc.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["A1"],
                    help="每个 arm = 一个会话（A1=会话1, A3=会话2）")
    a = ap.parse_args()
    print("事前停止门：电机速度/电流 vs 腕部 3–9 Hz 残差的多元相干\n")
    curves, corr_by = {}, {}
    print(f"  {'会话':<6}{'集':<6}{'输入':<12}{'原始 γ²':>10}{'零假设 q/n':>12}"
          f"{'扣偏置后':>11}{'段数':>7}")
    for arm in a.arms:
        for nm in EPS:
            P = episode_signals(nm, arm)
            for lbl, X in (("speed", P["SP"]), ("current", P["CU"]),
                           ("speed+cur", np.hstack([P["SP"], P["CU"]]))):
                raw, null, corr, curve, f, nseg = band_coh(X, P["res"])
                curves[(arm, nm, lbl)] = (f, curve)
                corr_by.setdefault((arm, lbl), []).append(corr)
                print(f"  {arm:<6}{nm:<6}{lbl:<12}{raw:>10.3f}{null:>12.3f}"
                      f"{corr:>11.3f}{nseg:>7}")
    sel = lambda f: (f >= BAND[0]) & (f <= BAND[1])
    print()
    for lbl in ("speed", "current", "speed+cur"):
        mins = {arm: min(corr_by[(arm, lbl)]) for arm in a.arms}
        print(f"  {lbl:<12} 各会话**最小**扣偏置 γ²: "
              + "  ".join(f"{k}={v:.3f}" for k, v in mins.items())
              + f"   门槛 0.35 ⇒ {'通过' if min(mins.values()) >= 0.35 else '未过'}")
    if len(a.arms) >= 2:
        A, B = a.arms[0], a.arms[1]
        print(f"\n  **跨会话** γ²(f) 曲线相关 ({A} vs {B}, 逐集配对):")
        for lbl in ("speed", "current", "speed+cur"):
            rs = []
            for nm in EPS:
                fa, ca = curves[(A, nm, lbl)]; fb, cb = curves[(B, nm, lbl)]
                rs.append(float(np.corrcoef(ca[sel(fa)], cb[sel(fb)])[0, 1]))
            print(f"    {lbl:<12}" + "  ".join(f"{nm}={r:+.3f}" for nm, r in zip(EPS, rs))
                  + f"   平均 {np.mean(rs):+.3f}   门槛 0.5 ⇒ "
                    f"{'通过' if np.mean(rs) >= 0.5 else '未过'}")
    else:
        print("\n  ⚠ 只给了一个会话 ⇒ 预注册的跨会话两条判据**无法判定**。"
              "补采后用 --arms A1 A3 重跑。")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "gate_coherence.npz",
             **{f"{k[0]}_{k[1]}_{k[2]}": np.stack(v) for k, v in curves.items()})
    print(f"\n  → {OUT/'gate_coherence.npz'}")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
