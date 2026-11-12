"""分层模型的传感器层 —— `ω̂ = R·kin(q̂) + r_s`（+ 1b 的实测残差前瞻项）。§2.4。

`r_s`：因果 FIR（nlag 16、相对岭 1e-4 —— 沿用 G-2 `r_s` 的选定值，不调参），
特征 = G-2 `rs_features` 的 34 维 + 理想陀螺 3 维（§2.6 输入相同）= 37 维。
**折内、用实测关节闭式拟合**：目标 = `ω_real − R·kin(q_real)`（与 G-2「目标用真关节」同理），
特征也取自实测关节 ⇒ 不需要关节层的折外预测、无折内泄漏。
⚠ 代价 = **训练用实测关节、推理用 q̂**（训练—推理差异，首版保留，报告须注明）。
拟合在评分频带（0.5–9 Hz）内做。**不复用 G-2 已拟合的 `r_s`。**

无效关节样本（存为 0）：FIR 第 k 行用到 `q_{k−17 … k+2}`（16 个滞后 + 二阶中心差分），
只有这段支撑全有效的行才进拟合/目标/历史；带通拟合再要求 ±20 帧全干净。
1b 的附加项**无截距、无有效比例**：没有有效残差时精确为 0（用户 2026-10-02 订正）；
用**固定岭回归**拟合（用户 2026-10-02 定；与 AR 同一实现、同一强度 `q1_ar.RIDGE_REL`）。
"""
from __future__ import annotations

import numpy as np

import q1_data as D
import q1_windows as W

NLAG = 16
LAM = 1e-4
N_FEAT = 37
STENCIL_BACK = NLAG + 1        # FIR 第 k 行 ← 特征 k−15…k ← q_{k−17…k+2}
STENCIL_FWD = 2
SPAN_WARM = NLAG + 4
SUPPORT_BACK = STENCIL_BACK - D.ALIGN_L    # 首个计分输出（帧 t0+3）的支撑回到 t0−14


def kin_body(q) -> np.ndarray:
    from ceiling_200hz import kin_gyro_fs
    return kin_gyro_fs(np.asarray(q, np.float64), D.FS)


def features(u7, q, ideal, kin) -> np.ndarray:
    """→ (L,37) = [u6, q, dq, ddq, kin(机体系), |dq|, 夹爪, 理想陀螺]。"""
    dq = np.gradient(q, 1 / D.FS, axis=0)
    ddq = np.gradient(dq, 1 / D.FS, axis=0)
    return np.concatenate([np.asarray(u7)[:, :6], q, dq, ddq, kin, np.abs(dq),
                           np.asarray(u7)[:, 6:7], ideal], 1)


def fir(X, nlag: int = NLAG) -> np.ndarray:
    from train_rq import fir_design
    return fir_design(np.asarray(X, np.float64), nlag)


def bp(x) -> np.ndarray:
    from scipy.signal import firwin
    k = firwin(41, [0.5, 9.0], fs=D.FS, pass_zero=False)
    x = np.asarray(x, np.float64)
    return np.stack([np.convolve(x[:, j], k, mode="same") for j in range(x.shape[1])], 1)


def clean_rows(q_valid, w_valid) -> np.ndarray:
    """原始行 k「干净」⇔ `q_{k−17 … k+2}` 全有效且 `ω_k` 有效（无效样本不经差分/FIR 进入该行）。"""
    qv = np.asarray(q_valid, bool)
    T = len(qv)
    ok = np.zeros(T, bool)
    for k in range(STENCIL_BACK, T - STENCIL_FWD):
        ok[k] = bool(qv[k - STENCIL_BACK:k + STENCIL_FWD + 1].all() and w_valid[k])
    return ok


def check_window_support(ep, t0: int) -> None:
    """1b 推理：首个计分输出的支撑（帧 t0−14 … t0，来自起点前的实测历史）要么全有效、要么全是填充。
    混合 ⇒ 填充值会经差分/FIR 污染有效帧 ⇒ 按数据契约报错（本数据只有 t0 = 0 全填充）。"""
    k = np.arange(t0 - SUPPORT_BACK, t0 + 1)
    v = np.zeros(len(k), bool)
    ok = k >= 0
    v[ok] = ep.q_valid[k[ok]]
    if v.any() and not v.all():
        raise D.DataContractError(f"{ep.name} t0={t0}: 传感器层支撑内实测有效/无效混合")


def omega_span(q, u7, ideal, R, Wrs) -> np.ndarray:
    """一段跨度上的 `ω̂`（传感器系）= R·kin(q) + r_s；FIR 在跨度起点零历史。"""
    kb = kin_body(q)
    return kb @ np.asarray(R).T + fir(features(u7, q, ideal, kb)) @ Wrs


def fit_rs(eps, R) -> np.ndarray:
    """折内闭式拟合：带内岭回归，`λ = LAM·tr(G)/dim`（同 `train_rq.fit`）；只用干净行。"""
    G = B = None
    for e in eps:
        kb = kin_body(e.q)
        Dm = fir(features(e.cmd[:e.T], e.q, e.ideal[:e.T], kb))
        Y = e.w - kb @ np.asarray(R).T
        m = W.band_clean(clean_rows(e.q_valid, e.w_valid))
        Db, Yb = bp(Dm)[m], bp(Y)[m]
        G = Db.T @ Db if G is None else G + Db.T @ Db
        B = Db.T @ Yb if B is None else B + Db.T @ Yb
    G = G + LAM * (np.trace(G) / G.shape[0]) * np.eye(G.shape[0])
    return np.linalg.solve(G, B)


def residual_history(ep, t0: int, R, Wrs) -> np.ndarray:
    """1b：起点前 20 帧的实测传感器残差 `s_k = ω_k − (R·kin(q_k) + r_s)`，**只用 ≤ t0 的实测**
    （跨度止于 t0 ⇒ 末端单侧差分），且只保留支撑内实测全有效的样本，其余置 0。
    → (3·HIST,)；没有有效样本时全为 0 ⇒ 1b 的附加项精确为 0。"""
    lo = t0 - D.HIST + 1
    a = max(0, lo - SPAN_WARM)
    h = np.zeros((D.HIST, 3))
    if t0 - a + 1 > STENCIL_BACK + 1:
        qv, wv = ep.q_valid[a:t0 + 1], ep.w_valid[a:t0 + 1]
        s = ep.w[a:t0 + 1] - omega_span(ep.q[a:t0 + 1], ep.cmd[a:t0 + 1], ep.ideal[a:t0 + 1], R, Wrs)
        for i, k in enumerate(range(lo, t0 + 1)):
            j = k - a
            if j >= STENCIL_BACK and wv[j] and qv[j - STENCIL_BACK:min(j + STENCIL_FWD, t0 - a) + 1].all():
                h[i] = s[j]
    return h.ravel()


def sensor_residual(ep, R, Wrs) -> np.ndarray:
    """拟合目标：整集实测关节上的传感器残差，帧 0…T−1（不干净的帧 = NaN）。"""
    out = ep.w - omega_span(ep.q, ep.cmd[:ep.T], ep.ideal[:ep.T], R, Wrs)
    out[~clean_rows(ep.q_valid, ep.w_valid)] = np.nan
    return out


def fit_cmap(eps, R, Wrs, windows_of=W.chunk_windows) -> dict:
    """1b 的逐前瞻线性项：`s_{t0+h+3} ≈ x(t0)·C_h`（无截距），只用训练折，固定岭回归。"""
    from q1_ar import ridge
    rows: dict = {}
    for e in eps:
        y = sensor_residual(e, R, Wrs)
        for w in windows_of(e):
            x = residual_history(e, w.t0, R, Wrs)
            for h in range(w.n_scored):
                k = w.t0 + h + D.ALIGN_L
                if k < e.T and np.isfinite(y[k]).all():
                    X, Y = rows.setdefault(h, ([], []))
                    X.append(x)
                    Y.append(y[k])
    return {h: ridge(X, Y) for h, (X, Y) in rows.items()}
