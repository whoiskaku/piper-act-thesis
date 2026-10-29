import copy

import numpy as np
import pytest
import torch

import q1_data as D1
import q1_e2e as E
import q1_windows as W
import q2_models as Mo
import q2_testing as T


def _st(eps):
    return E.norm_stats(eps)


def _jitter(m, s=0.05):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n != "bases":
                p.add_(torch.randn_like(p) * s)
    return m


def test_da_scale_preserves_zero_and_uses_rms_not_std():
    e = T.synthetic(0)
    e.da = np.zeros_like(e.da); e.da[:, 0] = 2e-3                     # 常数补正：std = 0，RMS = 2e-3
    s = Mo.da_scale([e])
    assert np.isclose(s[0], 2e-3) and s[3] == Mo.SCALE_FLOOR           # j4 恒为 0 ⇒ 下限，0 仍是 0
    assert (np.asarray(e.da) / s)[:, 3].max() == 0


def test_window_inputs_shapes_and_visibility():
    eps = [T.synthetic(s) for s in range(3)]
    st, sc = _st(eps), Mo.da_scale(eps)
    e = eps[0]
    ws = W.chunk_windows(e)
    X, Dd = Mo.window_inputs(e, ws, st, sc)
    assert X.shape == (len(ws), E.L_SEQ, E.CH_A) and Dd.shape == (len(ws), E.L_SEQ, 6)
    w = ws[1]                                                          # 序列帧 = t0−19 … t0+102
    seq = np.arange(w.t0 - D1.HIST + 1, w.t0 + E.K_OUT + 1)
    inside = (seq >= 0) & (seq <= w.t1)
    assert np.allclose(Dd[1][inside], (e.da[seq[inside]] / sc).astype(np.float32))
    assert (Dd[1][~inside] == 0).all()


@pytest.mark.parametrize("form", Mo.FORMS)
def test_zero_init_is_zero_response(form):
    eps = [T.synthetic(s) for s in range(2)]
    st, sc = _st(eps), Mo.da_scale(eps)
    torch.manual_seed(0)
    m = Mo.make(form)
    X, Dd = Mo.window_inputs(eps[0], W.chunk_windows(eps[0]), st, sc)
    with torch.no_grad():
        assert (Mo.increment(m, form, X, Dd) == 0).all()


@pytest.mark.parametrize("form", Mo.FORMS)
def test_zero_correction_gives_exactly_zero_increment(form):
    eps = [T.synthetic(s) for s in range(2)]
    st, sc = _st(eps), Mo.da_scale(eps)
    torch.manual_seed(0)
    m = _jitter(Mo.make(form))
    X, Dd = Mo.window_inputs(eps[0], W.chunk_windows(eps[0]), st, sc)
    with torch.no_grad():
        assert (Mo.increment(m, form, X, np.zeros_like(Dd)) == 0).all()
        assert Mo.increment(m, form, X, Dd).abs().max() > 0


def test_condlinear_is_exactly_linear_in_the_correction():
    eps = [T.synthetic(s) for s in range(2)]
    st, sc = _st(eps), Mo.da_scale(eps)
    torch.manual_seed(0)
    m = _jitter(Mo.make("condlinear")).double()
    X, D1_ = Mo.window_inputs(eps[0], W.chunk_windows(eps[0]), st, sc)
    D2_ = np.random.default_rng(0).normal(size=D1_.shape)
    X = X.astype(np.float64)
    with torch.no_grad():
        f = lambda d: Mo.increment(m, "condlinear", X, d.astype(np.float64))
        assert torch.allclose(f(2.0 * D1_ - 3.0 * D2_), 2.0 * f(D1_) - 3.0 * f(D2_), atol=1e-10)


def test_condlinear_matches_the_explicit_kernel_formula_and_can_go_negative():
    torch.manual_seed(0)
    m = Mo.make("condlinear").double()
    with torch.no_grad():
        m.mix.bias.copy_(torch.tensor([-1.0, 0.5, 0.0, 2.0], dtype=torch.float64))   # 可正可负（不用 softmax）
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1, 40, E.CH_A)); Dd = rng.normal(size=(1, 40, 6))
    with torch.no_grad():
        got = Mo.increment(m, "condlinear", X, Dd)[0].numpy()
        a = m.mix(m.body(torch.as_tensor(X).transpose(1, 2)))[0].numpy()             # (R, L)
    B = m.bases.detach().numpy()
    i = 30
    want = sum(a[r, i] * sum(B[r, tau] @ Dd[0, i - tau] for tau in range(Mo.N_LAG)) for r in range(Mo.N_BASIS))
    assert np.allclose(got[i], want)


@pytest.mark.parametrize("form", Mo.FORMS)
def test_increment_is_causal_over_the_sequence(form):
    torch.manual_seed(0)
    m = _jitter(Mo.make(form))
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1, E.L_SEQ, E.CH_A)).astype(np.float32); Dd = rng.normal(size=(1, E.L_SEQ, 6)).astype(np.float32)
    X2, D2_ = X.copy(), Dd.copy(); X2[:, 60:] += 1.0; D2_[:, 60:] += 1.0
    with torch.no_grad():
        assert torch.equal(Mo.increment(m, form, X, Dd)[:, :60], Mo.increment(m, form, X2, D2_)[:, :60])


@pytest.mark.parametrize("form", Mo.FORMS)
def test_increment_ignores_next_chunk_commands_and_corrections(form):
    eps = [T.synthetic(s) for s in range(3)]
    st, sc = _st(eps), Mo.da_scale(eps)
    torch.manual_seed(0)
    m = _jitter(Mo.make(form))
    e = eps[2]
    ws = W.chunk_windows(e)
    w = ws[2]

    def pred(ep):
        X, Dd = Mo.window_inputs(ep, ws, st, sc)
        with torch.no_grad():
            dE, dQ = Mo.decode_increment(Mo.increment(m, form, X, Dd), ws, st)
        return dE[2].numpy(), dQ[2].numpy()

    bE, bQ = pred(e)
    e1 = copy.copy(e)
    c = e.cmd.copy(); c[w.t1 + 1:] += 0.05; e1.cmd = c
    d = e.da.copy(); d[w.t1 + 1:] += 0.01; e1.da = d
    i_ = e.ideal.copy(); i_[w.t1 + 1:] += 0.5; e1.ideal = i_
    E1, Q1 = pred(e1)
    assert np.array_equal(E1, bE) and np.array_equal(Q1, bQ)


def test_decode_increment_positions_and_nominal_scales():
    eps = [T.synthetic(s) for s in range(2)]
    st = _st(eps)
    ws = W.chunk_windows(eps[0])
    out = torch.zeros(len(ws), E.L_SEQ, 9)
    out[0, 2 + D1.HIST, :] = 1.0                                       # 第一个计分帧 t0（序列里是 t0+3）
    dE, dQ = Mo.decode_increment(out, ws, st)
    assert np.allclose(dE[0][0].numpy(), st["s_gap"]) and np.allclose(dQ[0][0].numpy(), st["s_eps"])
    assert (dE[0][1:] == 0).all() and dE[0].shape == (ws[0].n_scored, 3)


def test_parameter_counts():
    torch.manual_seed(0)
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(Mo.make("nonlinear")) == 68_361 and n(Mo.make("condlinear")) == 70_436
