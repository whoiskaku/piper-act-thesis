import copy

import numpy as np
import pytest

import q1_ar as AR
import q1_data as D
import q1_testing as T
import q1_windows as W


def test_design_row_uses_only_visible_history():
    e = T.synthetic_episode(0)
    x = AR.design_row(e, 100)
    assert x.shape == (AR.N_FEAT,) == (3 * D.HIST,)               # 只有历史数值：无截距、无有效比例
    assert np.allclose(x[:3], e.gap[97]) and np.allclose(x[57:60], e.gap[78])
    assert (AR.design_row(e, 0) == 0).all()


def test_no_history_predicts_exactly_zero():
    eps = [T.synthetic_episode(s) for s in range(6)]
    coef = AR.fit(eps[:5])
    assert (AR.window_ehat(coef, eps[5], W.chunk_windows(eps[5])[0]) == 0).all()


def test_prediction_ignores_gap_after_origin():
    eps = [T.synthetic_episode(s) for s in range(6)]
    coef = AR.fit(eps[:5])
    e = eps[5]
    w = W.chunk_windows(e)[2]
    base = AR.window_ehat(coef, e, w)
    e2 = copy.copy(e)
    g = e.gap.copy(); g[w.t0 - 2:] += 1.0; e2.gap = g          # e_{t0−2} 已用到 ω_{t0+1}
    assert np.array_equal(AR.window_ehat(coef, e2, w), base)


def _ridge(X, Y, lam_rel):
    G = X.T @ X
    return np.linalg.solve(G + lam_rel * np.trace(G) / G.shape[0] * np.eye(G.shape[0]), X.T @ Y)


def test_fit_is_fixed_relative_ridge():
    """用户 2026-10-02 定：AR 改用固定岭回归（不调）；λ = RIDGE_REL·tr(XᵀX)/p，同 `train_rq.fit`。"""
    eps = [T.synthetic_episode(s) for s in range(8)]
    assert AR.RIDGE_REL == 1e-2
    coef = AR.fit(eps)
    for h in (0, 50, 99):
        rows = [(AR.design_row(e, w.t0), e.gap[w.t0 + h]) for e in eps for w in W.chunk_windows(e)
                if h < w.n_scored]
        X = np.stack([r[0] for r in rows]); Y = np.stack([r[1] for r in rows])
        assert np.allclose(coef[h], _ridge(X, Y, 1e-2), rtol=1e-10, atol=1e-12)
        ols = np.linalg.lstsq(X, Y, rcond=None)[0]
        assert np.linalg.norm(coef[h]) < np.linalg.norm(ols)       # 确实收缩了


def test_missing_lead_raises():
    e = T.synthetic_episode(1)
    with pytest.raises(ValueError):
        AR.window_ehat({0: np.zeros((AR.N_FEAT, 3))}, e, W.chunk_windows(e)[0])
