#!/usr/bin/env python3
"""**关节层诊断**：G-q-v1 预测的关节轨迹 vs 真实 200 Hz 关节轨迹，逐关节分频。

`RESULT_CAN_OBSERVABILITY.md` §4 的归因：3–9 Hz 腕部陀螺的缺口里
**0.457 在 G 的关节轨迹预测**、0.308 在传感器层。本脚本直接看关节层本身：
到底哪几个关节、什么性质的误差。

§28 的 `where_lost.py` 做过类似的事，但**那是在 20 Hz 混叠编码器上做的** ——
当时 3–9 Hz 的"真值"本身就被混叠污染，结论不可用。现在有 200 Hz 干净关节角。

口径
----
两侧都在**实际帧时刻**上比较：真关节 200 Hz 先低通 9 Hz 再采到帧时刻，
G-q-v1 用官方 `predict_qpos(cmd, t_send=, t_read=)` 的原生输出。
两者共享同一（略不均匀的）网格，带通把它当均匀 20 Hz 处理 —— 失真对两者相同。
逐关节各自扫最优位移（G 整体比真关节晚约 50 ms，但逐关节可能不同）。
**只评价该集里真正被激励的关节**（行程 > 5°）。

用法:
    python -u sim/step3_canobs/joint_error_200hz.py --arms A1 A3
"""
from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g", ROOT / "sim/step3_canobs"):
    sys.path.insert(0, str(_p))

EPS = ("ep0", "ep1")
FS_Q, FPS = 200.0, 20.0
D2R = np.pi / 180 / 1000.0
BANDS = ((0.5, 3.0), (3.0, 9.0))
TRAVEL_DEG = 5.0


def main() -> int:
    gc.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["A1", "A3"])
    A = ap.parse_args()
    from scipy.signal import butter, filtfilt, firwin
    import g_q_v1 as GQ

    def lp(x, fc, fs):
        return filtfilt(*butter(4, fc / (fs / 2)), x, axis=0)

    def bp(x, lo, hi):
        k = firwin(41, [lo, hi], fs=FPS, pass_zero=False)
        return np.convolve(x, k, mode="same")

    def best(a_, s_):
        """→ (最优位移, ρ², 增益, 残差/实测)。"""
        out = None
        for L in range(0, 4):
            n = len(a_) - L
            a, s = a_[L:L + n], s_[:n]
            aa, ss, as_ = (a * a).sum(), (s * s).sum(), (a * s).sum()
            r2 = as_ ** 2 / max(aa * ss, 1e-30)
            if out is None or r2 > out[1]:
                out = (L, r2, as_ / max(ss, 1e-30), (aa + ss - 2 * as_) / max(aa, 1e-30))
        return out

    print("关节层：G-q-v1 预测关节轨迹 vs 真实 200 Hz 关节轨迹（抗混叠到帧时刻）\n")
    agg = {b: {j: [] for j in range(6)} for b in BANDS}
    for arm in A.arms:
        for nm in EPS:
            d = json.loads((ROOT / f"outputs/probe/replay_{nm}_{arm}.json").read_text())
            off = d["can_hs_meta"]["clock"]["offset_realtime_minus_monotonic_s"]
            tr = np.asarray(d["t"], float)
            ts = np.asarray(d["t_send"], float)
            cmd = np.asarray(d["cmd"], float)
            # 真关节 200 Hz → 均匀网格 → 低通 9 Hz → 采到帧时刻
            Q = np.asarray(d["can_q200"], float)
            tq, qv = [], []
            for gid in (1, 3, 5):
                m = Q[:, 1] == gid
                tq.append(Q[m, 0] - off); qv.append(Q[m, 2:4])
            t0 = max(tr[0], *[t[0] for t in tq]); t1 = min(tr[-1], *[t[-1] for t in tq])
            tg = np.arange(t0, t1, 1.0 / FS_Q)
            q200 = np.zeros((len(tg), 6))
            for gid, t_, v_ in zip((1, 3, 5), tq, qv):
                for c in (0, 1):
                    q200[:, gid - 1 + c] = np.interp(tg, t_, v_[:, c]) * D2R
            q200f = lp(q200, 9.0, FS_Q)
            sel = (tr >= t0) & (tr <= t1)
            q_true = np.stack([np.interp(tr[sel], tg, q200f[:, j]) for j in range(6)], 1)
            q_hat = np.radians(GQ.predict_qpos(cmd, t_send=ts, t_read=tr))[sel]
            # 指令本身(限幅前)作参照: 看伺服层到底改了多少
            q_cmd = cmd[sel, :6]
            trav = np.degrees(q_true.max(0) - q_true.min(0))
            print(f"=== {arm}/{nm}  ({sel.sum()} 帧) ===")
            print(f"  {'关节':<5}{'行程°':>8} | " + " | ".join(
                f"{f'{lo}-{hi}Hz  L  ρ²   增益  残差':<28}" for lo, hi in BANDS))
            for j in range(6):
                if trav[j] < TRAVEL_DEG:
                    continue
                row = ""
                for b in BANDS:
                    L, r2, g, res = best(bp(q_true[:, j], *b), bp(q_hat[:, j], *b))
                    agg[b][j].append((r2, g, res))
                    row += f" | {L:>1}  {r2:>5.3f}  {g:>5.2f}  {res:>6.3f}      "
                print(f"  j{j+1:<4}{trav[j]:>8.1f}{row}")
            print()
    print(f"  {'关节':<5}{'集数':>6}" + "".join(
        f"{f'  {lo}-{hi} Hz: ρ²   增益   残差':<30}" for lo, hi in BANDS))
    for j in range(6):
        if not agg[BANDS[0]][j]:
            continue
        s = ""
        for b in BANDS:
            v = np.array(agg[b][j])
            s += f"  {v[:,0].mean():>6.3f} {v[:,1].mean():>6.2f} {v[:,2].mean():>7.3f}   "
        print(f"  j{j+1:<5}{len(agg[BANDS[0]][j]):>6}{s}")
    print("\n  ρ² = 允许最优标量增益后能解释的方差比（纯形状）；增益 <1 ⇒ G 的幅度偏小。")
    print("  口径: pp2 诊断集，逐关节各自最优位移，仅评价行程 > 5° 的关节。")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
