#!/usr/bin/env python3
"""重复精度封顶验证的固定处理链 —— `reports/PLAN_REPEAT_PRECISION_CAPPED.md` §2（冻结前写死，不随数据调整）。

1. 时间基准：每次复放用 100 Hz 设备毫秒对主机时刻做线性拟合，得到 100 Hz 样本的时刻（去掉接收抖动）。
2. 抗混叠低通（在 100 Hz 上、取样之前）：`firwin(183, 9.0, window=("kaiser", β), fs=100)`，
   183 阶与 β 由 `kaiserord(60 dB, 过渡带 8–10 Hz)` 给出；系数 sha256 前 16 位须等于 `FIR_SHA16`（否则拒绝）。
   零相位：`filtfilt(b, [1.0], x, axis=0, padtype="odd", padlen=549)`（等效幅频 |H|²：0–8 Hz 在 ±0.02 dB 内，9 Hz −12.0 dB，10 Hz 以上不高于 −120 dB）。
3. 固定相位取样：在每帧 `t_send + OFFSET_S`（−5.0 ms）处，对第 2 步结果做三次样条插值
   （`CubicSpline(..., bc_type="not-a-knot", extrapolate=False)`）→ 20 Hz；任一取样点超出 100 Hz 覆盖范围 ⇒ 拒绝（该次复放无效）。
4. 之后由调用方套用项目计分口径 `P(·)`（去均值 → bandpass41 0.5–9 Hz → 丢首尾 20 帧）。
⚠ 固定相位插值本身不等于抗混叠，所以第 2 步在第 3 步之前。新处理链的数字与旧口径（朴素 20 Hz、或收口诊断里的线性插值）不能直接当同口径比较。

    python -u sim/precision_chain.py            # 冻结前的离线复算：pilot 已有记录按本处理链复算 σ、核对预期精度（不调阈值）
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE / "q1", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

ROOT = HERE.parent
PILOT = ROOT / "outputs/pilot_response14"
OUT = ROOT / "outputs/pilot_response14_analysis"
FS100 = 100.0
NUMTAPS = 183
CUTOFF_HZ = 9.0
PADLEN = 3 * NUMTAPS
OFFSET_S = -0.005
FIR_SHA16 = "f74b8d9914e4ee8d"


def fir_sha16(b) -> str:
    return hashlib.sha256(np.ascontiguousarray(b, np.float64).tobytes()).hexdigest()[:16]


def aa_fir() -> np.ndarray:
    from scipy.signal import firwin, kaiserord
    n, beta = kaiserord(60.0, (10.0 - 8.0) / (0.5 * FS100))
    n += (n + 1) % 2                                                    # 取奇数阶（线性相位 I 型）
    b = firwin(n, CUTOFF_HZ, window=("kaiser", beta), fs=FS100)
    if len(b) != NUMTAPS or fir_sha16(b) != FIR_SHA16:
        raise ValueError(f"抗混叠 FIR 与冻结设计不符（{len(b)} 阶，sha {fir_sha16(b)}）—— 检查 scipy 版本")
    return b


def aa_filter(x) -> np.ndarray:
    from scipy.signal import filtfilt
    return filtfilt(aa_fir(), [1.0], np.asarray(x, np.float64), axis=0, padtype="odd", padlen=PADLEN)


def time_base(i100) -> np.ndarray:
    """100 Hz 样本的时刻：主机时刻对设备毫秒的线性拟合。"""
    i100 = np.asarray(i100, np.float64)
    b, a = np.polyfit(i100[:, 1], i100[:, 0], 1)
    return a + b * i100[:, 1]


def frame_locked(rec, offset: float = OFFSET_S) -> np.ndarray:
    """一次复放 → 抗混叠后、在每帧 t_send + offset 处取样的 20 Hz 陀螺 (T, 3)。"""
    from scipy.interpolate import CubicSpline
    i100 = np.asarray(rec["imu100"], np.float64)
    t = time_base(i100)
    q = np.asarray(rec["t_send"], np.float64) + offset
    if q.min() < t[0] or q.max() > t[-1]:
        raise ValueError("取样时刻超出 100 Hz 流的覆盖范围 —— 该次复放无效")
    g = aa_filter(i100[:, 5:8])
    return CubicSpline(t, g, axis=0, bc_type="not-a-knot", extrapolate=False)(q)


# ── 冻结前的离线复算（用户 2026-10-05 同意；不据此调阈值）──────────────────────
def recompute_pilot(root=PILOT) -> dict:
    import analyze_pilot_response14 as A
    import q1_windows as W
    root = Path(root)
    att = json.loads((root / "session.json").read_text())["attempts"]
    A.load_runs(root, att)                                              # 记录哈希与账本一致
    recs = {a["run_id"]: json.loads((root / "records" / f"replay_{a['run_id']}.json").read_text())
            for a in att if a["run_id"] != "A0"}
    K = W.kernel()
    X = {k: W.P(frame_locked(r), K) for k, r in recs.items()}
    bare = [X[r] for r in A.BARE]
    s_bare, df_bare = A.pooled_sigma(bare)
    s_adj = float(np.sqrt(np.mean([A.pair_sigma(X[a], X[b]) ** 2 for a, b in A.ADJ_PAIRS])))
    ebar = np.mean(bare, 0)
    se_pilot = s_adj * np.sqrt(1 / 2 + 1 / 5)
    resp, s2 = {}, []
    for g, spec in A.PROBES.items():
        est = A.responses([X[r] for r in spec["plus"]], [X[r] for r in spec["minus"]], ebar)
        for sgn in ("plus", "minus"):
            r = A.rms(est[sgn])
            resp[f"{g}{'+' if sgn == 'plus' else '−'}"] = {"rms": r, "rms_over_se": r / se_pilot}
            s2.append(max(r ** 2 - se_pilot ** 2, 0.0))
    S2 = float(np.mean(s2))
    var_half = s_adj ** 2 * (1 / 2 + 1 / 4)                              # 本方案分半：每半 2 次同条件 + 4 次裸
    return {"chain": {"fir_sha16": FIR_SHA16, "numtaps": NUMTAPS, "cutoff_hz": CUTOFF_HZ, "padlen": PADLEN,
                      "offset_s": OFFSET_S},
            "sigma_bare_new_chain": s_bare, "df_bare": df_bare, "sigma_adj_new_chain": s_adj,
            "responses_new_chain": resp, "signal_power_S2": S2, "signal_rms_S": float(np.sqrt(S2)),
            "planned_design_n4_M8": {"se_full": s_adj * np.sqrt(1 / 4 + 1 / 8), "se_half": float(np.sqrt(var_half)),
                                     "expected_R": S2 / (S2 + var_half) if S2 > 0 else 0.0},
            "reference_scenarios_other_calibers": {"naive_20hz_latest_sample": 0.0353,
                                                   "diag_fixed_phase_linear_no_aa": 0.0275,
                                                   "note": "参考情景值，口径不同，不是新处理链的上下界"}}


def main(argv=None) -> int:
    gc.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=PILOT)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    res = recompute_pilot(a.root)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "recompute_new_chain.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
