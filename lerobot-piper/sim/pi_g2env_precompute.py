#!/usr/bin/env python3
"""预计算 π 的 **G-2 训练环境**（零机时）。

定位（用户 2026-09-16 定稿）
----------------------------
**π 是最终部署件；G-2 只是训练环境。** 部署时不需要 G-2 / H / 求解器。

    训练:  ACT 轨迹 → 冻结 G-2 环境 → π 出 δa → G-2 执行 u+δa
                                          → 用仿真 IMU 残差算损失 → 更新 π

    L = ‖ω_G2(u+δa) − ω_ideal‖² + λ_a‖δa‖² + λ_s‖Δδa‖²

本脚本负责**环境那一半**：把每集的 `ω_G2(u)`、`ideal`、以及 G-2 自己的雅可比 `J`
一次性算好存盘，训练时直接用（否则每个 batch 都要跑 MuJoCo，训练不可行）。

为什么用 J 而不是每步真跑 G-2
------------------------------
G-2 前向含 MuJoCo 伺服 rollout 与逐帧运动学，**不可微**。用 G-2 **自己的**雅可比
把损失线性化：`ω_G2(u+δa) ≈ ω_G2(u) + J⊛δa`。

**J 的适用性已实测**（`sim/verify_J_for_pi.py`，`outputs/p2_pi/verify_J_for_pi.json`）：
全带 cos 0.900 / 范数比 0.980 / 相对误差 0.444 —— **未过我事前定的 0.30 门禁**，
但用它求解在**真 G-2** 上实测能降 **+22.6%**（`/tmp` 试验，见 WORKLOG）
⇒ 作为梯度来源够用，**训练完必须用真 G-2 复核**，不能只信 J。

⚠ 时基（踩过两次）
------------------
- `Jd[t, k]` : δa[t] → **ω**[t+k]
- `gap[i] = ω[i+L] − ideal[i]`，`L = ALIGN_L = 3`
⇒ 两者**不在同一时间轴**。本脚本统一存成 **ω 时间轴**的目标：

    tgt_w[s] = ω_G2(u)[s] − ideal[s−L]        (s < L 处置零)

于是 δa[t] 的贡献直接落在 `tgt_w[t+k]`，与 `Jd[t,k]` 对齐，不再需要换算。

用法:
    python -u sim/pi_g2env_precompute.py [--roots outputs/ab5,outputs/ab4,outputs/ab2]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "sim/step4_gvnext"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import g2_on_ab4 as G                                                # noqa: E402
import fit_step1_candidate as S1                                     # noqa: E402
from freeze_pi_g2wall import G2v0                                    # noqa: E402

OUT = ROOT / "outputs/p2_pi/g2env"
L, M = S1.ALIGN_L, S1.M
DEFAULT_ROOTS = ("outputs/ab5", "outputs/ab4", "outputs/ab2")


def find_logs(roots) -> list[Path]:
    out = []
    for r in roots:
        d = ROOT / r
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*_ep*.json")):
            try:
                pay = json.loads(p.read_text())
            except Exception:
                continue
            if not pay.get("chunks") or len(pay.get("frames", [])) < 500:
                continue
            out.append(p)
    return out


def main() -> int:
    import gc
    gc.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default=",".join(DEFAULT_ROOTS))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    from lerobot.common.policies.act.step1_correction_runtime import calibrate_ideal_imu

    calib = json.loads(
        (ROOT / "outputs/step1_H/step1_H_v1.json").read_text())["calibration"]
    R = np.asarray(calib["R"], np.float32)
    b = np.asarray(calib.get("b", [0, 0, 0]), np.float32)
    rend = IdealChunkRenderer()
    g2 = G2v0()

    logs = find_logs([r.strip() for r in a.roots.split(",") if r.strip()])
    if a.limit:
        logs = logs[:a.limit]
    print(f"π 的 G-2 训练环境预计算 · {len(logs)} 集")
    print(f"目标存成 **ω 时间轴**: tgt_w[s] = ω_G2(u)[s] − ideal[s−{L}]")
    print(f"J: Jd[t,k] = δa[t] → ω[t+k]，与 tgt_w 同轴\n")
    OUT.mkdir(parents=True, exist_ok=True)

    t00 = time.perf_counter()
    kept = []
    for i, p in enumerate(logs):
        pay = json.loads(p.read_text())
        chunks = sorted(pay["chunks"], key=lambda c: int(c["start"]))
        u = np.concatenate([np.asarray(c["raw"], np.float64) for c in chunks])
        T = len(u)
        starts = [int(c["start"]) for c in chunks]
        ideal, prev = [], None
        for c in chunks:
            raw = np.asarray(c["raw"], np.float32)
            ideal.append(calibrate_ideal_imu(
                np.asarray(rend.render(raw, prev), np.float32), R, b)[:, 3:6])
            prev = raw
        ideal = np.concatenate(ideal)[:T].astype(np.float64)

        dd = {"name": p.stem, "cmd": u, "t_send": np.arange(T) / 20.0,
              "t_read": np.arange(T) / 20.0,
              "tg_i": np.arange(T * 5) / 100.0, "gyro100": np.zeros((T * 5, 3))}
        w = g2.omega(dd)
        w6 = np.zeros((len(w), 6), np.float32); w6[:, 3:] = w
        wcal = calibrate_ideal_imu(w6, R, b)[:, 3:].astype(np.float64)
        n = min(len(wcal), T)
        tgt = np.zeros((n, 3))
        for s in range(L, n):
            if s - L < len(ideal):
                tgt[s] = wcal[s] - ideal[s - L]
        t0s, J = G.extract(g2, dd, n)
        Jd = G.dense(t0s, J, n).astype(np.float32)

        # 接触门控掩码（与部署同口径，由夹爪指令决定，逐 chunk 带 previous_grip）
        mask = np.zeros(n, bool)
        pg = None
        for c in chunks:
            s0 = int(c["start"]); raw = np.asarray(c["raw"], np.float64)
            e0 = min(s0 + len(raw), n)
            if s0 >= n:
                break
            mask[s0:e0] = S1.command_contact_mask(
                raw[:e0 - s0, 6].astype(np.float32), grip_full=S1.GRIP_FULL,
                threshold_ratio=S1.CONTACT_THRESHOLD_RATIO,
                half_window=S1.CONTACT_HALF_WINDOW, previous_grip=pg)
            pg = float(raw[-1, 6])

        np.savez_compressed(
            OUT / f"{p.parent.name}__{p.stem}.npz",
            u=u[:n].astype(np.float32), tgt=tgt.astype(np.float32), J=Jd,
            contact=mask, starts=np.asarray(starts, np.int32), n=n)
        kept.append(f"{p.parent.name}/{p.stem}")
        if (i + 1) % 5 == 0 or i == len(logs) - 1:
            el = time.perf_counter() - t00
            print(f"  {i+1:>3}/{len(logs)}  {el:>5.0f}s"
                  f"（剩 ~{el/(i+1)*(len(logs)-i-1):.0f}s）  最新 {p.stem} T={n}")

    (OUT / "index.json").write_text(json.dumps(
        {"episodes": kept, "align_L": L, "M": M,
         "tgt_def": "tgt_w[s] = omega_G2(u)[s] - ideal[s-L]; s<L 置零 (**omega 时间轴**)",
         "J_def": "Jd[t,k] = delta_a[t] -> omega[t+k]; 由 g2_on_ab4.extract/dense(AMP=1mrad)",
         "contact": "与部署同口径的接触门控掩码(由夹爪指令决定, 逐 chunk 带 previous_grip)",
         "caliber": ("ideal 逐 chunk 渲染(filtfilt 非因果, 不可截断); "
                     "G-2 = 原始冻结 v=0; 时基假设均匀 20Hz(日志无时间戳); "
                     "**J 的相对误差 0.444 未过事前 0.30 门禁 ⇒ 训练后必须用真 G-2 复核**"),
         }, ensure_ascii=False, indent=1))
    print(f"\n  → {OUT}  （{len(kept)} 集）")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
