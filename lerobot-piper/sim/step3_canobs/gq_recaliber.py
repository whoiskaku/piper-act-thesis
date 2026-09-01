#!/usr/bin/env python3
"""**修正抗混叠口径后，G-q-v1 值多少** —— 不改模型，只改真机侧的抽取方式。

§28 查明的口径不对称：
    仿真侧 `g_q_v1.py:231` 从 1 kHz 经 9 Hz 抗混叠低通降到 20 Hz；
    真机侧 TSND 100 Hz 被 `get_latest()` **无抗混叠**抽到 20 Hz。
⇒ 一直在拿**抗混叠过的预测**去对**混叠过的目标**打分，而混叠成分在仿真侧
   原理上不存在对应物。本脚本只把真机侧改成同型抗混叠，**G 一个字节没动**。

    旧口径: A = 真机朴素抽到 20 Hz      S = G-q-v1 官方输出(已抗混叠)
    新口径: A = 真机先低通 9 Hz 再抽取   S = 同上, 逐字节相同

时基
----
这两条诊断集的 20 Hz 控制环是坏的(中位 51.2 ms, p99 84.5 ms)，故：
  · 真机侧用 TSND **设备 ms**(步长恒 10、零丢样)建均匀 100 Hz 网格；
  · G 侧取 `_sim(..., return_hi=True)` 的 1 kHz 原始陀螺，按**实际** t_send/t_read
    驱动，再抗混叠重采到同一网格 —— 完全绕开抖动的帧网格。

会话标定：逐集正交 Procrustes 拟合 3×3 旋转，两个口径同等自由度。

用法:
    python -u sim/step3_canobs/gq_recaliber.py
"""
from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g", ROOT / "sim/step3_canobs"):
    sys.path.insert(0, str(_p))

EPS = ("ep0", "ep1")
FS_I = 100.0
BANDS = ((0.5, 3.0), (3.0, 9.0), (0.5, 9.0))
OUT = ROOT / "outputs/step3_canobs"


def main() -> int:
    gc.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["A1"])
    ARMS = ap.parse_args().arms
    from scipy.signal import butter, filtfilt, firwin
    from scipy.linalg import orthogonal_procrustes
    import g_q_v1 as GQ
    from ceiling_200hz import load_streams, kin_gyro_fs

    def lp(x, fc, fs):
        return filtfilt(*butter(4, fc / (fs / 2)), x, axis=0)

    def bp(x, lo, hi, fs):
        k = firwin(int(2 * fs) | 1, [lo, hi], fs=fs, pass_zero=False)
        return np.stack([np.convolve(x[:, j], k, mode="same")
                         for j in range(x.shape[1])], 1)

    def score(A, S):
        aa = float((A * A).sum()); ss = float((S * S).sum()); as_ = float((A * S).sum())
        return ((aa + ss - 2 * as_) / max(aa, 1e-30),
                as_ ** 2 / max(aa * ss, 1e-30), as_ / max(ss, 1e-30))

    def evaluate(A_, S_, fs, maxshift):
        best = None
        for L in range(0, maxshift + 1):
            k = min(len(A_) - L, len(S_))
            A = A_[L:L + k]; S = S_[:k]
            R, _ = orthogonal_procrustes(S, A)
            Sr = S @ R
            v = sum(score(bp(A, *b, fs), bp(Sr, *b, fs))[0] for b in BANDS)
            if best is None or v < best[0]:
                best = (v, L, A, Sr)
        _, L, A, Sr = best
        return L, {b: score(bp(A, *b, fs), bp(Sr, *b, fs)) for b in BANDS}

    print("修正抗混叠口径后 G-q-v1 值多少（模型未改，只改真机侧抽取）\n")
    keys = ("gq_naive20", "gq_aa20", "kin_naive20", "kin_aa20")
    acc = {k: {b: np.zeros(3) for b in BANDS} for k in keys}
    saved = {}
    runs = [(arm, nm) for arm in ARMS for nm in EPS]
    for arm, nm in runs:
        d = json.loads((ROOT / f"outputs/probe/replay_{nm}_{arm}.json").read_text())
        g100, q200, tg_i, tg_q = load_streams(nm, arm)
        cmd = np.asarray(d["cmd"], float)
        ts, tr = np.asarray(d["t_send"], float), np.asarray(d["t"], float)
        # G 侧：1 kHz 原始陀螺 → 抗混叠 → 重采到真机的均匀 100 Hz 网格
        _q, _g20, hi, read, fs_hi = GQ._sim(cmd, t_send=ts, t_read=tr, return_hi=True)
        t_hi = tr[0] + (np.arange(len(hi)) - read[0]) / fs_hi
        hi_aa = lp(hi, 45.0, fs_hi)
        gq100 = np.stack([np.interp(tg_i, t_hi, hi_aa[:, k]) for k in range(3)], 1)
        kin100 = lp(kin_gyro_fs(q200, 200.0), 45.0, 200.0)[::2][:len(g100)]

        A_naive, A_aa = g100[::5], lp(g100, 9.0, FS_I)[::5]
        for tag, S100 in (("gq", gq100), ("kin", kin100)):
            S20 = lp(S100, 9.0, FS_I)[::5]
            for cal, A in (("naive20", A_naive), ("aa20", A_aa)):
                L, r = evaluate(A, S20, 20.0, 3)
                for b in BANDS:
                    acc[f"{tag}_{cal}"][b] += np.array(r[b])
                if tag == "gq":
                    print(f"  {arm}/{nm} G-q-v1 {cal:>8}  L={L}  "
                          f"ρ² = {' / '.join(f'{r[b][1]:.3f}' for b in BANDS)}")
        saved[f"{arm}_{nm}"] = {"A_naive": A_naive, "A_aa": A_aa,
                     "gq20": lp(gq100, 9.0, FS_I)[::5],
                     "kin20": lp(kin100, 9.0, FS_I)[::5]}

    print(f"\n  {'':<32}{'0.5–3 Hz':>22}{'3–9 Hz':>22}{'0.5–9 Hz':>22}")
    print(f"  {'':<32}" + "".join(f"{'ρ²':>10}{'残差':>7}{'增益':>5}" for _ in BANDS))
    for k, lbl in (("gq_naive20", "G-q-v1  旧口径(真机混叠)"),
                   ("gq_aa20", "G-q-v1  新口径(两侧抗混叠)"),
                   ("kin_naive20", "天花板  旧口径"),
                   ("kin_aa20", "天花板  新口径")):
        s = ""
        for b in BANDS:
            r, r2, g = acc[k][b] / len(runs)
            s += f"{r2:>10.3f}{r:>7.3f}{g:>5.2f}"
        print(f"  {lbl:<32}{s}")
    for b, nb in ((0.5, 3.0), (3.0, 9.0), (0.5, 9.0)) and (
            ((0.5, 3.0), "0.5–3"), ((3.0, 9.0), "3–9"), ((0.5, 9.0), "0.5–9")):
        d1 = (acc["gq_aa20"][b][1] - acc["gq_naive20"][b][1]) / len(runs)
        print(f"  {nb} Hz: G-q-v1 的 ρ² 因**口径修正**变化 {d1:+.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "recaliber_traces.npz",
             **{f"{nm}_{k}": v for nm, dd in saved.items() for k, v in dd.items()})
    print(f"\n  波形已存 → {OUT/'recaliber_traces.npz'}")
    print(f"  口径: pp2 诊断集 {len(runs)} 条 / {len(ARMS)} 个会话(2026-09-01)，**不可与 17 集开发基准集**"
          "的 §28 数字并列。")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
