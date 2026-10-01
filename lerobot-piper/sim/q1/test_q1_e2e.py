import copy

import numpy as np
import pytest
import torch

import q1_data as D
import q1_e2e as E
import q1_testing as T
import q1_windows as W

FAMS = [("tcn", E.make_tcn), ("tr", E.make_transformer)]


def _jit(m, s=0.05):
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * s)
    return m


def _pred(m, e, st, meas):
    ws = W.chunk_windows(e)
    X = torch.tensor(np.stack([E.window_input(e, w, st, meas) for w in ws]))
    with torch.no_grad():
        Ee, Qq = E.decode(m(X), e, ws, st)
    return [x.numpy() for x in Ee], [x.numpy() for x in Qq]


def test_input_shapes_and_future_has_no_measured_values():
    eps = [T.synthetic_episode(s) for s in range(3)]
    st = E.norm_stats(eps[:2])
    w = W.chunk_windows(eps[2])[1]
    xa = E.window_input(eps[2], w, st, False)
    xb = E.window_input(eps[2], w, st, True)
    assert xa.shape == (E.L_SEQ, E.CH_A) and xb.shape == (E.L_SEQ, E.CH_B)
    assert (xb[D.HIST:, E.CH_A:] == 0).all()


@pytest.mark.parametrize("name,mk", FAMS)
def test_zero_init_is_zero_prediction(name, mk):
    eps = [T.synthetic_episode(s) for s in range(3)]
    st = E.norm_stats(eps[:2])
    e = eps[2]
    torch.manual_seed(0)
    Ee, Qq = _pred(mk(E.CH_A), e, st, False)
    w = W.chunk_windows(e)[1]
    assert all((x == 0).all() for x in Ee)
    u_vis = W.visible(e.cmd[:, :6], w.t1, w.s1 + D.ALIGN_L)[np.arange(w.t0, w.s1 + 1) + D.ALIGN_L]
    assert np.allclose(Qq[1], u_vis, atol=1e-6)                 # q̂ 基线 = 钳位后的可见指令
    assert np.allclose(Qq[1][-3:], e.cmd[w.t1, :6], atol=1e-6)  # 末 3 帧落在 t1 之后 ⇒ 等于 u_{t1}


@pytest.mark.parametrize("name,mk", FAMS)
@pytest.mark.parametrize("meas", [False, True])
def test_no_leakage(name, mk, meas):
    eps = [T.synthetic_episode(s) for s in range(3)]
    st = E.norm_stats(eps[:2])
    e = eps[2]
    w = W.chunk_windows(e)[2]
    torch.manual_seed(0)
    m = _jit(mk(E.CH_B if meas else E.CH_A))
    bE, bQ = (x[2] for x in _pred(m, e, st, meas))
    e1 = copy.copy(e)
    c = e.cmd.copy(); c[w.t1 + 1:] += 0.05; e1.cmd = c
    i_ = e.ideal.copy(); i_[w.t1 + 1:] += 0.5; e1.ideal = i_
    E1, Q1 = (x[2] for x in _pred(m, e1, st, meas))
    assert np.array_equal(E1, bE) and np.array_equal(Q1, bQ)     # ê 与 q̂ 都不读下一 chunk
    e2 = copy.copy(e)
    q = e.q.copy(); q[w.t0 + 1:] += 0.05; e2.q = q
    ww = e.w.copy(); ww[w.t0 + 1:] += 0.5; e2.w = ww
    E2, Q2 = (x[2] for x in _pred(m, e2, st, meas))
    assert np.array_equal(E2, bE) and np.array_equal(Q2, bQ)
    e3 = copy.copy(e); e3.q = e.q + 0.01; e3.w = e.w + 0.1
    assert (not np.array_equal(_pred(m, e3, st, meas)[0][2], bE)) == meas


@pytest.mark.parametrize("name,mk", FAMS)
def test_causal_over_sequence(name, mk):
    torch.manual_seed(0)
    m = _jit(mk(E.CH_A))
    x = torch.randn(1, E.L_SEQ, E.CH_A)
    x2 = x.clone(); x2[:, 60:] += 1.0
    with torch.no_grad():
        assert torch.equal(m(x)[:, :60], m(x2)[:, :60])


def test_param_counts_are_comparable():
    for ch in (E.CH_A, E.CH_B):
        torch.manual_seed(0)
        a = sum(p.numel() for p in E.make_tcn(ch).parameters())
        b = sum(p.numel() for p in E.make_transformer(ch).parameters())
        assert abs(b / a - 1) <= 0.25, (a, b)


def test_norm_stats_use_only_given_episodes():
    eps = [T.synthetic_episode(s) for s in range(3)]
    assert not np.allclose(E.norm_stats(eps[:2])["u"][0], E.norm_stats(eps)["u"][0])
