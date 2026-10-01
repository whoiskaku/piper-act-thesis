import copy

import numpy as np
import pytest

import q1_data as D
import q1_sensor as S
import q1_testing as T
import q1_windows as W


def test_features_layout():
    e = T.synthetic_episode(0)
    q = e.q[2:60]
    X = S.features(e.cmd[2:60], q, e.ideal[2:60], S.kin_body(q))
    assert X.shape == (58, S.N_FEAT)
    assert np.allclose(X[:, :6], e.cmd[2:60, :6]) and np.allclose(X[:, 6:12], q)
    assert np.allclose(X[:, 33], e.cmd[2:60, 6]) and np.allclose(X[:, 34:], e.ideal[2:60])


def test_fir_is_causal():
    X = np.random.default_rng(0).normal(size=(50, 4))
    X2 = X.copy(); X2[30:] += 1.0
    assert np.array_equal(S.fir(X2)[:30], S.fir(X)[:30])


def test_fit_rs_uses_only_given_episodes_and_is_deterministic():
    eps = [T.synthetic_episode(s) for s in range(4)]
    R = np.eye(3)
    W1, W2, W3 = S.fit_rs(eps[:3], R), S.fit_rs(eps[:3], R), S.fit_rs(eps[1:], R)
    assert W1.shape == (S.NLAG * S.N_FEAT + 1, 3)
    assert np.array_equal(W1, W2) and not np.allclose(W1, W3)


def test_residual_history_uses_only_data_up_to_origin():
    eps = [T.synthetic_episode(s) for s in range(4)]
    R = np.eye(3)
    Wrs = S.fit_rs(eps[:3], R)
    e = eps[3]
    x = S.residual_history(e, 100, R, Wrs)
    e2 = copy.copy(e)
    q = e.q.copy(); q[101:] += 0.05; e2.q = q
    ww = e.w.copy(); ww[101:] += 1.0; e2.w = ww
    assert np.array_equal(S.residual_history(e2, 100, R, Wrs), x)
    assert x.shape == (3 * D.HIST,) and (x != 0).all()
    assert (S.residual_history(e, 0, R, Wrs) == 0).all()


def test_cmap_term_is_exactly_zero_without_valid_history():
    eps = [T.synthetic_episode(s) for s in range(4)]
    R = np.eye(3)
    Wrs = S.fit_rs(eps, R)
    C = S.fit_cmap(eps, R, Wrs)
    x0 = S.residual_history(eps[0], 0, R, Wrs)
    assert all(C[h].shape == (3 * D.HIST, 3) and (x0 @ C[h] == 0).all() for h in C)


def _dropout(e, k, value):
    e.q_valid = e.q_valid.copy(); e.q_valid[k] = False
    e.q = e.q.copy(); e.q[k] = value
    return e


def test_clean_rows_and_band_clean_cover_the_stencils():
    e = _dropout(T.synthetic_episode(0), 300, 0.0)
    ok = S.clean_rows(e.q_valid, e.w_valid)
    assert not ok[:S.STENCIL_BACK + D.Q_VALID_FROM].any() and ok[S.STENCIL_BACK + D.Q_VALID_FROM]
    assert not ok[298:318].any() and ok[297] and ok[318]           # k−17 ≤ 300 ≤ k+2
    bc = W.band_clean(ok)
    assert not bc[278:338].any() and bc[277] and bc[338]


def test_invalid_joint_values_never_enter_the_fit():
    """无效帧里存的数（0 或任意值）不得经中心差分 / FIR / 带通影响拟合与有效帧的残差。"""
    R = np.eye(3)
    a = [_dropout(T.synthetic_episode(s), 300, 0.0) for s in range(4)]
    b = [_dropout(T.synthetic_episode(s), 300, 5.0) for s in range(4)]
    Wa, Wb = S.fit_rs(a, R), S.fit_rs(b, R)
    assert np.array_equal(Wa, Wb)
    ya, yb = S.sensor_residual(a[0], R, Wa), S.sensor_residual(b[0], R, Wa)
    assert np.array_equal(np.isnan(ya), np.isnan(yb)) and np.array_equal(ya[~np.isnan(ya)], yb[~np.isnan(yb)])
    assert np.array_equal(S.residual_history(a[0], 310, R, Wa), S.residual_history(b[0], 310, R, Wa))


def test_window_support_must_not_mix_valid_and_invalid():
    e = T.synthetic_episode(0)
    S.check_window_support(e, 0)                                   # t0 = 0：支撑全是填充 ⇒ 允许
    S.check_window_support(e, 100)                                 # 全是有效实测 ⇒ 允许
    with pytest.raises(D.DataContractError):
        S.check_window_support(_dropout(e, 95, 0.0), 100)


def test_fit_cmap_covers_all_leads():
    eps = [T.synthetic_episode(s) for s in range(4)]
    R = np.eye(3)
    Wrs = S.fit_rs(eps, R)
    C = S.fit_cmap(eps, R, Wrs)
    assert sorted(C) == list(range(100)) and C[0].shape == (3 * D.HIST, 3)


@pytest.mark.slow
def test_measured_joint_sensor_layer_explains_gap(bare_eps, calib):
    """实现自检（**不是模型分数**：用了未来的实测关节）：带通后应解释大部分 gap。"""
    R, _ = calib
    tr = [e for e in bare_eps if e.sess != "ab5/A"]
    te = [e for e in bare_eps if e.sess == "ab5/A"][:10]
    Wrs = S.fit_rs(tr, R)
    K = W.kernel()
    num = den = 0.0
    for e in te:
        a = D.Q_VALID_FROM
        full = np.zeros((e.T, 3))
        full[a:] = S.omega_span(e.q[a:], e.cmd[a:e.T], e.ideal[a:e.T], R, Wrs)
        eh = full[D.ALIGN_L:D.ALIGN_L + e.n] - e.ideal[:e.n]
        n_, d_, _ = W.score(e.gap[30:], eh[30:], K)
        num += n_; den += d_
    assert 1 - num / den > 0.3
