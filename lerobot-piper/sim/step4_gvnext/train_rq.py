#!/usr/bin/env python3
"""**关节层残差 `r_q`** —— 因果 FIR，`PREREG_G_VNEXT.md` §2/§4。

    q̂ = G-q-v1(u) + r_q(u, q̂_base, history)
    目标 = q200_real − q_Gq

⚠ 为什么损失必须带带内项
------------------------
残差的能量与位置一样由**低频主导**（3–9 Hz 只占关节位置能量的 0.003%–0.12%）。
用朴素 MSE 拟合就会重蹈 `optuna_ident_enc` 的覆辙 —— 目标从没要求过 3–9 Hz，
参数自然动不了它（`RESULT_BAND_WEIGHTED_IDENT.md`）。故：

    L = 0.5 · MSE(残差) + 0.5 · MSE(bp_{3-9}(残差))

两项都是 FIR 系数的二次型 ⇒ 仍是闭式岭回归（把带通后的设计矩阵与目标**堆叠**）。

输入边界（严格执行，与旧实验同规）
----------------------------------
**允许**：指令 `u`、底座预测 `q̂_base` 及其一/二阶差分、跟踪误差 `u − q̂_base`、
夹爪指令及其变化、实际帧间隔。全部是**推理时可得**的量。
**禁止**：帧序号、任务相位、集 ID、未来指令、真机编码器/IMU 作为推理输入。

选参只用**训练族的折外损失**；留出族不参与。

用法:
    python -u sim/step4_gvnext/train_rq.py                 # 折外选参 + 报告
    python -u sim/step4_gvnext/train_rq.py --selftest      # 无真机数据的形状自检
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data import load, lp, bp, q_at, valid_files, FPS, FS_I   # noqa: E402

S1 = ROOT / "outputs/step4_gvnext/s1"
OUT = ROOT / "outputs/step4_gvnext/rq"
BAND = (3.0, 9.0)
W_BAND = 0.5                       # 与 optuna_ident_band 同规，事前定死
LAGS = (4, 8, 16, 24)              # 因果滞后窗候选（帧, 20 Hz ⇒ 0.2–1.2 s）
LAMS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
KFOLD = 6
MOVING = [0, 1, 2, 4, 5]           # j4 在所有 pp2 族里恒为 0


def features(dd: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """→ (X 每帧特征, Y 残差目标, t 均匀 20 Hz 时刻)。**只用推理时可得的量。**"""
    import g_q_v1 as GQ
    t20 = dd["tg_i"][::5]
    qh_f = np.radians(GQ.predict_qpos(dd["cmd"], t_send=dd["t_send"],
                                      t_read=dd["t_read"]))
    qh = np.stack([np.interp(t20, dd["t_read"], qh_f[:, j]) for j in range(6)], 1)
    u = np.stack([np.interp(t20, dd["t_read"], dd["cmd"][:, j]) for j in range(6)], 1)
    g = np.interp(t20, dd["t_read"], dd["cmd"][:, 6])
    qt = q_at(dd, t20)                                   # 真关节（**只作目标**）
    dq = np.gradient(qh, 1 / FPS, axis=0)
    ddq = np.gradient(dq, 1 / FPS, axis=0)
    err = u - qh                                          # 伺服正在追的跟踪误差
    X = np.concatenate([u, qh, dq, ddq, err,
                        g[:, None], np.gradient(g, 1 / FPS)[:, None]], 1)
    return X, qt - qh, t20


def fir_design(X: np.ndarray, nlag: int) -> np.ndarray:
    """因果 FIR 设计矩阵（零历史起始）+ 常数列。"""
    T, F = X.shape
    out = np.zeros((T, nlag * F + 1))
    for L in range(nlag):
        if L == 0:
            out[:, :F] = X
        elif L < T:
            out[L:, L * F:(L + 1) * F] = X[:-L]
    out[:, -1] = 1.0
    return out


def stack_band(D: np.ndarray, Y: np.ndarray, w: float):
    """把带内项堆进最小二乘 ⇒ 仍是闭式解。

    ⚠ **每项必须按自身目标尺度归一**：带通后的目标方差比宽带小约三个数量级，
    直接用 `0.5·MSE + 0.5·MSE(带通)` 会让带内项几乎不起作用 ——
    又是「目标没要求过 3–9 Hz」那个坑（`optuna_ident_band` 里已按自身尺度归一）。

        L = (1−w)·MSE(e)/var(Y) + w·MSE(bp e)/var(bp Y)
    """
    Db, Yb = bp(D, *BAND), bp(Y, *BAND)
    sy = float(Y.std()) + 1e-12
    sb = float(Yb.std()) + 1e-12
    a, b = np.sqrt(1.0 - w) / sy, np.sqrt(w) / sb
    return np.vstack([a * D, b * Db]), np.vstack([a * Y, b * Yb])


def fit(Ds, Ys, lam):
    D, Y = np.vstack(Ds), np.vstack(Ys)
    G = D.T @ D
    G += lam * (np.trace(G) / max(G.shape[0], 1)) * np.eye(G.shape[0])
    return np.linalg.solve(G, D.T @ Y)


def band_err(Yt, Yp, lo, hi):
    """→ (残差能量, 实测能量)，逐可动关节合计。"""
    a, b = bp(Yt, lo, hi), bp(Yp, lo, hi)
    return (float(((a - b)[:, MOVING] ** 2).sum()),
            float((a[:, MOVING] ** 2).sum()))


def main() -> int:
    import gc
    gc.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        X = np.random.randn(300, 10)
        D = fir_design(X, 4)
        assert D.shape == (300, 41) and np.allclose(D[0, 10:40], 0), "FIR 因果性坏了"
        assert np.allclose(D[5, 10:20], X[4]), "滞后 1 对错了"
        print("  ✓ FIR 设计矩阵：形状/因果性/滞后对齐 通过")
        return 0

    files = valid_files(S1)
    print(f"训练族（会话1）{len(files)} 条 · 折外 K={KFOLD} · 按**集**划分\n")
    EP = []
    for f in files:
        dd = load(f)
        X, Y, t = features(dd)
        EP.append({"name": dd["name"], "X": X, "Y": Y})
    print(f"  特征维 {EP[0]['X'].shape[1]}   每集帧数 {EP[0]['X'].shape[0]}")

    folds = [list(range(i, len(EP), KFOLD)) for i in range(KFOLD)]
    print(f"\n  {'nlag':>5}{'lam':>8}{'折外 3-9Hz 削减':>16}{'折外 宽带削减':>15}")
    best = None
    for nlag in LAGS:
        Dall = [fir_design(e["X"], nlag) for e in EP]
        for lam in LAMS:
            num39 = den39 = numb = denb = 0.0
            for te in folds:
                tr = [i for i in range(len(EP)) if i not in te]
                W = fit([stack_band(Dall[i], EP[i]["Y"], W_BAND)[0] for i in tr],
                        [stack_band(Dall[i], EP[i]["Y"], W_BAND)[1] for i in tr], lam)
                for i in te:
                    P = Dall[i] @ W
                    n_, d_ = band_err(EP[i]["Y"], P, *BAND); num39 += n_; den39 += d_
                    n_, d_ = band_err(EP[i]["Y"], P, 0.5, 9.0); numb += n_; denb += d_
            r39 = 1 - num39 / max(den39, 1e-30)
            rb = 1 - numb / max(denb, 1e-30)
            print(f"  {nlag:>5}{lam:>8.0e}{r39*100:>15.1f}%{rb*100:>14.1f}%")
            if best is None or r39 > best[0]:
                best = (r39, rb, nlag, lam)
    r39, rb, nlag, lam = best
    print(f"\n  **选中 nlag={nlag}, lam={lam:.0e}** —— 折外 3–9 Hz 削减 {r39*100:.1f}%"
          f"，宽带 {rb*100:.1f}%")
    print(f"  预注册的 FIR 不足判据: 折外 3–9 Hz 改善 < 5% ⇒ "
          + ("**不足，可试一个固定规模小网络**" if r39 < 0.05 else "**足够，不上网络**"))

    # 逐关节 + 折外预测落盘（供 r_s 用，避免训练泄漏）
    Dall = [fir_design(e["X"], nlag) for e in EP]
    oof = {}
    pj = {j: [0.0, 0.0] for j in MOVING}
    for te in folds:
        tr = [i for i in range(len(EP)) if i not in te]
        W = fit([stack_band(Dall[i], EP[i]["Y"], W_BAND)[0] for i in tr],
                [stack_band(Dall[i], EP[i]["Y"], W_BAND)[1] for i in tr], lam)
        for i in te:
            P = Dall[i] @ W
            oof[EP[i]["name"]] = P
            A, B = bp(EP[i]["Y"], *BAND), bp(P, *BAND)
            for j in MOVING:
                pj[j][0] += float(((A - B)[:, j] ** 2).sum())
                pj[j][1] += float((A[:, j] ** 2).sum())
    print(f"\n  逐关节折外 3–9 Hz 削减")
    for j in MOVING:
        print(f"    j{j+1}: {(1 - pj[j][0]/max(pj[j][1],1e-30))*100:>6.1f}%")
    W_full = fit([stack_band(Dall[i], EP[i]["Y"], W_BAND)[0] for i in range(len(EP))],
                 [stack_band(Dall[i], EP[i]["Y"], W_BAND)[1] for i in range(len(EP))], lam)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "rq_fir.npz", W=W_full, nlag=nlag, lam=lam, w_band=W_BAND,
             moving=MOVING, names=[e["name"] for e in EP],
             **{f"oof_{k}": v for k, v in oof.items()})
    (OUT / "select.json").write_text(json.dumps(
        {"nlag": int(nlag), "lam": float(lam), "w_band": W_BAND, "kfold": KFOLD,
         "oof_reduction_3_9": float(r39), "oof_reduction_broad": float(rb),
         "per_joint_3_9": {f"j{j+1}": float(1 - pj[j][0]/max(pj[j][1],1e-30))
                           for j in MOVING},
         "caliber": "训练族=会话1 18 条; 折外按集 K=6; 选参只用折外, 留出族未参与",
         }, indent=1, ensure_ascii=False))
    print(f"\n  → {OUT/'rq_fir.npz'}  (含折外预测, 供 r_s 用)")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
