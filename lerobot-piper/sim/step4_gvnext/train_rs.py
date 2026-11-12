#!/usr/bin/env python3
"""**传感器层残差 `r_s`** —— 因果 FIR，`PREREG_G_VNEXT.md` §2/§4。

    ω̂ = kin(q̂) + r_s(u, q̂, q̂̇, history)
    目标 = IMU100_real − kin(q200_real)          ← **用真关节**，纯传感器层误差

为什么目标用真关节
------------------
这样 `r_s` 学的是**关节→IMU 写像**本身的缺口，与关节预测的好坏解耦。
留出实测：真关节 + 刚体渲染的天花板在 3–9 Hz 只有 **ρ²=0.649**，
即传感器层还有 **0.351** 的空间 —— 那正是 `r_s` 的目标。
（判据2 门槛 0.707 > 0.649 ⇒ **不可能靠关节层过**，必须 `r_s` 补。）

⚠ 训练泄漏
----------
`r_s` 的输入里 `q̂ = q_base + r_q`，训练时 `r_q` **只能用折外预测**
（`rq_fir.npz` 里的 `oof_*`）。用全量拟合的 `r_q` 会让 `r_s` 见到
它在推理时见不到的精度。

⚠ 坐标系
--------
真机 IMU 在绑带系、`kin` 在机体系，且绑带朝向**逐会话漂移**、无标定。
评价用**逐集正交 Procrustes**，故训练也必须同约定：
先把真机旋进机体系再作差，`r_s` 的输出也在机体系。

损失与 `r_q` 同规：`(1−w)·MSE/var + w·MSE(bp)/var(bp)`，**每项按自身尺度归一**。

用法:
    python -u sim/step4_gvnext/train_rs.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g", ROOT / "sim/step3_canobs", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data import load, to20, bp, q_at, valid_files, FPS      # noqa: E402
from train_rq import fir_design, stack_band, fit, BAND, W_BAND, KFOLD  # noqa: E402
from ceiling_200hz import kin_gyro_fs                        # noqa: E402

S1 = ROOT / "outputs/step4_gvnext/s1"
RQ = ROOT / "outputs/step4_gvnext/rq"
OUT = ROOT / "outputs/step4_gvnext/rs"
LAGS = (4, 8, 16)
LAMS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def sensor_target(dd: dict, t20: np.ndarray):
    """→ (A_body 真机旋进机体系, K_true kin(真关节), 目标 = A_body − K_true)。"""
    from scipy.linalg import orthogonal_procrustes
    A = to20(dd["gyro100"])[:len(t20)]
    K = kin_gyro_fs(q_at(dd, t20), FPS)
    n = min(len(A), len(K))
    A, K = A[:n], K[:n]
    R, _ = orthogonal_procrustes(A, K)          # 真机 → 机体系
    return A @ R, K, A @ R - K


def rs_features(dd: dict, t20: np.ndarray, qhat: np.ndarray) -> np.ndarray:
    """`r_s` 的每帧特征。**只用推理时可得的量**（q̂ = 底座 + r_q）。"""
    u = np.stack([np.interp(t20, dd["t_read"], dd["cmd"][:, j]) for j in range(6)], 1)
    g = np.interp(t20, dd["t_read"], dd["cmd"][:, 6])
    dq = np.gradient(qhat, 1 / FPS, axis=0)
    ddq = np.gradient(dq, 1 / FPS, axis=0)
    kin = kin_gyro_fs(qhat, FPS)                # 刚体渲染本身也是特征
    return np.concatenate([u, qhat, dq, ddq, kin,
                           np.abs(dq), g[:, None]], 1)


def band_err3(Yt, Yp, lo, hi):
    a, b = bp(Yt, lo, hi), bp(Yp, lo, hi)
    return float(((a - b) ** 2).sum()), float((a ** 2).sum())


def main() -> int:
    import gc
    gc.disable()
    import g_q_v1 as GQ
    z = np.load(RQ / "rq_fir.npz")
    sel = json.loads((RQ / "select.json").read_text())
    print(f"r_s 训练 —— q̂ 用 r_q 的**折外**预测 (nlag={sel['nlag']}, "
          f"lam={sel['lam']:.0e})\n")

    files = valid_files(S1)
    EP = []
    for f in files:
        dd = load(f)
        t20 = dd["tg_i"][::5]
        qh_f = np.radians(GQ.predict_qpos(dd["cmd"], t_send=dd["t_send"],
                                          t_read=dd["t_read"]))
        qb = np.stack([np.interp(t20, dd["t_read"], qh_f[:, j]) for j in range(6)], 1)
        key = f"oof_{dd['name']}"
        if key not in z:
            raise SystemExit(f"缺 {dd['name']} 的 r_q 折外预测 —— 先跑 train_rq.py")
        qhat = qb + z[key]                      # **折外** q̂，不是全量拟合
        Ab, K, Y = sensor_target(dd, t20)
        n = len(Y)
        EP.append({"name": dd["name"], "X": rs_features(dd, t20, qhat)[:n], "Y": Y,
                   "A": Ab, "K": K})
    print(f"  训练族 {len(EP)} 条 · 特征维 {EP[0]['X'].shape[1]} · 折外 K={KFOLD}")
    e0 = np.mean([band_err3(e["A"], e["K"], *BAND)[0] /
                  max(band_err3(e["A"], e["K"], *BAND)[1], 1e-30) for e in EP])
    print(f"  基线（kin(真关节) 对真机）3–9 Hz 残差/实测 = {e0:.3f}"
          f"   ⇒ 传感器层可得空间")

    folds = [list(range(i, len(EP), KFOLD)) for i in range(KFOLD)]
    print(f"\n  {'nlag':>5}{'lam':>8}{'折外 3-9Hz 削减':>16}{'折外 0.5-3Hz 削减':>18}")
    best = None
    for nlag in LAGS:
        D = [fir_design(e["X"], nlag) for e in EP]
        for lam in LAMS:
            n39 = d39 = n13 = d13 = 0.0
            for te in folds:
                tr = [i for i in range(len(EP)) if i not in te]
                W = fit([stack_band(D[i], EP[i]["Y"], W_BAND)[0] for i in tr],
                        [stack_band(D[i], EP[i]["Y"], W_BAND)[1] for i in tr], lam)
                for i in te:
                    P = D[i] @ W
                    a, b = band_err3(EP[i]["Y"], P, *BAND); n39 += a; d39 += b
                    a, b = band_err3(EP[i]["Y"], P, 0.5, 3.0); n13 += a; d13 += b
            r39 = 1 - n39 / max(d39, 1e-30)
            r13 = 1 - n13 / max(d13, 1e-30)
            print(f"  {nlag:>5}{lam:>8.0e}{r39*100:>15.1f}%{r13*100:>17.1f}%")
            if best is None or r39 > best[0]:
                best = (r39, r13, nlag, lam)
    r39, r13, nlag, lam = best
    print(f"\n  **选中 nlag={nlag}, lam={lam:.0e}** —— 折外 3–9 Hz 削减 {r39*100:.1f}%"
          f"，0.5–3 Hz {r13*100:.1f}%")
    D = [fir_design(e["X"], nlag) for e in EP]
    W_full = fit([stack_band(D[i], EP[i]["Y"], W_BAND)[0] for i in range(len(EP))],
                 [stack_band(D[i], EP[i]["Y"], W_BAND)[1] for i in range(len(EP))], lam)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "rs_fir.npz", W=W_full, nlag=nlag, lam=lam, w_band=W_BAND)
    (OUT / "select.json").write_text(json.dumps(
        {"nlag": int(nlag), "lam": float(lam), "w_band": W_BAND, "kfold": KFOLD,
         "oof_reduction_3_9": float(r39), "oof_reduction_0_5_3": float(r13),
         "baseline_resid_3_9": float(e0),
         "caliber": ("目标=IMU100_real − kin(q200_real), 真机先 Procrustes 旋进机体系; "
                     "q̂ 用 r_q 折外预测; 训练族=会话1 有效 17 条"),
         }, indent=1, ensure_ascii=False))
    print(f"\n  → {OUT/'rs_fir.npz'}")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
