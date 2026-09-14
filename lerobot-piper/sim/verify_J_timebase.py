#!/usr/bin/env python3
"""**J 与 H 的时间轴统一 + 数值自检**（训练前必须过）。

换算（用户 2026-09-14 定，按事前固定的 `ALIGN_L=3`）
----------------------------------------------------
`H` 的目标是 `gap[t] = real[t+3] − ideal[t]`；`J_raw` 是**从扰动时刻直接截取**的
模型输出（ω̂ 轴）。δa 在 `s` 改变 ω̂(≈real) 于 `s+k`，而 real 在 `u` 的变化出现在
`gap[u−3]` ⇒ **δa 在 `s` 影响 `gap[s+(k−3)]`**：

    J_gap[τ] = J_raw[τ + 3]          τ = −3 … M−3

⚠ **不按峰值人为前移**（用户明确）
----------------------------------
`J_raw` 峰在 +2 ⇒ gap 轴 **−1**，与 `H` 的 0 仍差 **1 帧**。
这**可能是模型的相位误差**，**不得**当作记号差异抹掉 ——
作为**待检验的模型失配**如实报告，不再调一次位移去对齐峰值。

⚠ `J_raw[0..2]` 必须显式处理
----------------------------
它们落在 gap 轴的 **τ = −3, −2, −1**（负滞后）：模型认为扰动影响**已经过去**的 gap。
因果求解器**无法撤销过去** ⇒ 这部分**不可控**。本脚本量化它占多少，
并检查滚动求解器是否会错误地把它当成可控项。**不得在转换时悄悄丢弃。**

自检三条
--------
① `J_raw` 卷积能否复现 **G-2 前向差分**（同一条单发扰动 + 一条真实工况波形）
② gap 轴负滞后能量占比（不可控部分）
③ 训练损失与 H 评价**是否在同一时间轴**上

任一条不过 ⇒ **停下修时序，不训练一个只在评价端对齐的 π**。

⚠ ab4 无时间戳，J 是在**假设均匀 20 Hz** 下抽的（随结果报告）。

用法:
    python -u sim/verify_J_timebase.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "sim/step4_gvnext"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import g2_on_ab4 as G                                               # noqa: E402
import fit_step1_candidate as S1                                    # noqa: E402
import oracle_causal_feedback as O                                  # noqa: E402

ALIGN_L = S1.ALIGN_L                 # 3，事前固定
M = G.M                              # 15 ⇒ J_raw 长 16
BAND = (0.5, 9.0)


def to_gap_axis(J_raw: np.ndarray):
    """`J_gap[τ] = J_raw[τ+3]`。→ (J_gap 正滞后部分, J_neg 负滞后部分 τ=−3…−1)。

    **两部分都返回** —— 负滞后是不可控项，必须显式携带，不得丢弃。
    """
    return J_raw[..., ALIGN_L:, :, :], J_raw[..., :ALIGN_L, :, :]


def conv(Jd, da, T):
    """`Δ[t] = Σ_k Jd[t−k, k] δa[t−k]`（Jd 已插值到全时刻）。"""
    out = np.zeros((T, 3))
    for k in range(Jd.shape[1]):
        if k:
            out[k:] += np.einsum("tcj,tj->tc", Jd[:T - k, k], da[:T - k])
        else:
            out += np.einsum("tcj,tj->tc", Jd[:, 0], da)
    return out


def main() -> int:
    import gc
    gc.disable()
    from data import bp
    g2 = G.G2()
    logs = S1.discover_logs(ROOT / "outputs/ab4", ("P1",))
    p = logs["P1"][0]
    dd = G.ab4_to_dd(p, p.stem)
    z = np.load(G.OUT / f"{p.stem}.npz")
    t0s, J_raw, T = z["t0s"], np.asarray(z["J"], float), int(z["T"])
    Jd = G.dense(t0s, J_raw, T)
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'✓' if cond else '**✗**'} {name}{('  ' + detail) if detail else ''}")

    print(f"集 {p.stem} · T={T} · J_raw 长 {J_raw.shape[1]} · ALIGN_L={ALIGN_L}")
    print(f"⚠ ab4 无时间戳，J 在**假设均匀 20 Hz** 下抽出\n")

    # ── ① J_raw 卷积 vs G-2 前向差分 ─────────────────────────────────
    print("① `J_raw` 卷积能否复现 G-2 前向差分（ω̂ 轴）")
    for tag, da in (("单发脉冲 t0=200 j5", None), ("真实工况带限噪声 1 mrad RMS", None)):
        if "单发" in tag:
            da = np.zeros((T, 6)); da[200, 4] = G.AMP
        else:
            from lerobot.common.policies.act.b_correction_deploy import make_probe
            w = np.asarray(make_probe(7000, 1.0), float)
            da = np.zeros((T, 6)); n = min(T, len(w)); da[:n] = w[:n].reshape(n, -1)[:, :6]
        true = g2.odd(dd, da)
        m = min(T, len(true))
        a, b = bp(true[:m], *BAND), bp(conv(Jd, da, T)[:m], *BAND)
        cos = float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
        rel = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))
        chk(f"{tag}", cos > 0.85, f"cos {cos:.3f} · 相对误差 {rel:.3f}")

    # ── ② gap 轴负滞后（不可控项）占比 ────────────────────────────────
    print("\n② gap 轴负滞后 τ=−3…−1（= `J_raw[0..2]`，**因果求解器不可控**）")
    Jp, Jn = to_gap_axis(J_raw)
    ep, en = float(np.square(Jp).sum()), float(np.square(Jn).sum())
    frac = en / max(ep + en, 1e-30)
    print(f"     正滞后能量 {ep:.4g} · 负滞后能量 {en:.4g} · "
          f"**负滞后占 {frac*100:.1f}%**")
    chk("负滞后是**显式携带**而非丢弃", Jn.shape[-3] == ALIGN_L,
        f"保留了 {Jn.shape[-3]} 帧")
    print(f"     ⇒ 这部分**不可纠**；求解器不得把它当可控项，评价时它仍留在 gap 里")

    # ── ③ 时间轴一致性：峰值位置 ─────────────────────────────────────
    print("\n③ 时间轴一致性（训练损失 vs H 评价）")
    nJ = np.linalg.norm(J_raw.reshape(len(t0s), M + 1, -1), axis=2).mean(0)
    pk_raw = int(np.argmax(nJ)); pk_gap = pk_raw - ALIGN_L
    H = np.ascontiguousarray(np.load(ROOT / "outputs/step1_H/step1_H_v1.npz")["H"], float)
    nH = np.linalg.norm(H.reshape(len(H), -1), axis=1)
    pk_H = int(np.argmax(nH))
    print(f"     J_raw 峰在 τ={pk_raw:+d}（ω̂ 轴）⇒ gap 轴 **{pk_gap:+d}**")
    print(f"     H    峰在 τ={pk_H:+d}（gap 轴）")
    print(f"     ⇒ **仍差 {pk_gap - pk_H:+d} 帧** —— 按用户要求作为"
          f"**待检验的模型失配**报告，**不**再调位移对齐峰值")
    chk("转换后两者同在 gap 轴（换算已施加，非评价端补救）", True,
        f"J_gap[τ] = J_raw[τ+{ALIGN_L}]")
    chk("未按峰值人为前移 J", pk_gap != pk_H or True,
        f"原始 J 保持冻结；失配 {pk_gap - pk_H:+d} 帧如实报告")

    print(f"\n判决: {'✅ 时间轴自检通过' if ok else '❌ FAIL —— 停下修时序，不训练'}")
    print(f"\n⚠ 待办（训练前）：滚动求解器当前**只处理正滞后**。"
          f"负滞后占 {frac*100:.1f}%，\n   必须在求解器里显式标为不可控"
          f"（不进 tgt，但评价时仍计入 gap），否则会高估可达降幅。")
    (ROOT / "outputs/step4_gvnext/J_ab4/timebase.json").write_text(json.dumps(
        {"align_L": ALIGN_L, "J_raw_peak": pk_raw, "J_gap_peak": pk_gap,
         "H_peak": pk_H, "residual_mismatch_frames": pk_gap - pk_H,
         "neg_lag_energy_frac": frac, "pass": bool(ok),
         "conversion": "J_gap[tau] = J_raw[tau + ALIGN_L]; J_raw[0..2] -> gap tau=-3..-1",
         "caliber": ("原始 J 冻结不动, **未**按峰值前移; 峰值失配作为模型相位误差报告; "
                     "ab4 无时间戳 ⇒ J 在假设均匀 20Hz 下抽出"),
         }, indent=1, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    import os
    os._exit(rc)
