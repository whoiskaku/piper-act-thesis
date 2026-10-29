"""折内拟合的线性 AR 参照（**不训练神经网络**）—— `DESIGN_G_PI_V2.md` §2.5、§5 #4。

目标 = gap `e_t`（t ∈ [t0, s1]）；输入 = 起点可得的最近 20 个**原始**实测 gap
`e_{t0−3−l}`（l = 0…19；`e_{t0−3}` 只用到 `ω_{t0}`），三轴联合；每个前瞻一组系数；
**固定岭回归**（用户 2026-10-02 定，替代普通最小二乘；不调超参）：`λ = RIDGE_REL·tr(XᵀX)/p`，
与 `train_rq.fit` 同一实现；强度 `RIDGE_REL = 1e-2` 是我定的默认值。历史不足（`t0−3−l < 0`）填 0。
**无截距、无有效比例列**：没有有效历史时预测精确为 0；逐前瞻截距会学到 chunk 相位的平均
模板，那不是「误差延续」（与 1b 附加项同一原则，用户 2026-10-02 订正 1b 时定）。
不复用 `oracle_causal_feedback.py` 的权重与残差目标。
"""
from __future__ import annotations

import numpy as np

import q1_data as D
import q1_windows as W

N_FEAT = 3 * D.HIST
RIDGE_REL = 1e-2             # 固定相对岭强度（默认值，可改；1b 前瞻项共用）


def ridge(X, Y) -> np.ndarray:
    """固定相对岭 `λ = RIDGE_REL·tr(XᵀX)/p`（`train_rq.fit` 同一实现）；不调超参。"""
    from train_rq import fit
    return fit([np.asarray(X, np.float64)], [np.asarray(Y, np.float64)], RIDGE_REL)


def design_row(ep, t0: int) -> np.ndarray:
    """→ (N_FEAT,) = [e_{t0−3}, …, e_{t0−22}]（各 3 轴；无效 = 0）。"""
    idx = t0 - D.ALIGN_L - np.arange(D.HIST)
    ok = idx >= 0
    h = np.zeros((D.HIST, 3))
    h[ok] = ep.gap[idx[ok]]
    return h.ravel()


def fit(eps, windows_of=W.chunk_windows) -> dict:
    """→ {前瞻 h: (N_FEAT, 3) 系数}。**只用传入的（训练折）集。**"""
    rows: dict = {}
    for e in eps:
        for w in windows_of(e):
            x = design_row(e, w.t0)
            for h in range(w.n_scored):
                X, Y = rows.setdefault(h, ([], []))
                X.append(x)
                Y.append(e.gap[w.t0 + h])
    return {h: ridge(X, Y) for h, (X, Y) in rows.items()}


def window_ehat(coef: dict, ep, w: W.Window) -> np.ndarray:
    missing = [h for h in range(w.n_scored) if h not in coef]
    if missing:
        raise ValueError(f"前瞻 {missing[:3]} 在训练折里没有样本")
    x = design_row(ep, w.t0)
    return np.stack([x @ coef[h] for h in range(w.n_scored)])
