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


def test_fit_is_exact_on_a_linear_target():
    rng = np.random.default_rng(0)
    eps = [T.synthetic_episode(s) for s in range(16)]
    M = rng.normal(0, 0.1, (AR.N_FEAT, 3))
    sel = lambda e: [w for w in W.chunk_windows(e) if w.t0 >= 100]
    for e in eps:
        for w in sel(e):
            e.gap[w.t0] = AR.design_row(e, w.t0) @ M          # 只改 h = 0 的目标
    coef = AR.fit(eps, windows_of=sel)
    w = sel(eps[0])[1]
    assert np.allclose(AR.window_ehat(coef, eps[0], w)[0], AR.design_row(eps[0], w.t0) @ M)


def test_missing_lead_raises():
    e = T.synthetic_episode(1)
    with pytest.raises(ValueError):
        AR.window_ehat({0: np.zeros((AR.N_FEAT, 3))}, e, W.chunk_windows(e)[0])
