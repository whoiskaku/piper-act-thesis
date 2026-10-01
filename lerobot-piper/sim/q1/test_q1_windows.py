import numpy as np
import pytest

import q1_data as D
import q1_testing as T
import q1_windows as W


def test_chunk_windows_cover_scored_frames_once_and_truncate_last_chunk():
    e = T.synthetic_episode(0, T=587)
    ws = W.chunk_windows(e)
    assert [(w.t0, w.t1) for w in ws] == [(0, 99), (100, 199), (200, 299),
                                          (300, 399), (400, 499), (500, 599)]
    assert ws[-1].s1 == e.n - 1 == 583
    W.stitch(e.n, ws, [np.zeros((w.n_scored, 3)) for w in ws])


def test_stitch_rejects_gaps_and_bad_shapes():
    e = T.synthetic_episode(0)
    ws = W.chunk_windows(e)
    with pytest.raises(ValueError):
        W.stitch(e.n, ws[:-1], [np.zeros((w.n_scored, 3)) for w in ws[:-1]])
    with pytest.raises(ValueError):
        W.stitch(e.n, ws, [np.zeros((w.n_scored + 1, 3)) for w in ws])


def test_visible_clamps_beyond_t1():
    a = np.arange(20, dtype=float)[:, None]
    v = W.visible(a, t1=9, upto=14)
    assert v.shape == (15, 1) and v[:10, 0].tolist() == list(range(10)) and (v[10:, 0] == 9).all()
    assert np.array_equal(W.visible(a, t1=9, upto=5), a[:6])


def test_span_pads_negative_frames():
    a = np.arange(10, dtype=float)[:, None] + 1
    assert W.span(a, -3, 2, fill=-1.0)[:, 0].tolist() == [-1, -1, -1, 1, 2, 3]


def test_past_block_padding_and_masks():
    e = T.synthetic_episode(0)
    pb = W.past_block(e, 0, measured=True)                 # t0 = 0：只有第 0 帧存在，且编码器无效
    assert pb["m_cmd"].tolist() == [0.0] * 19 + [1.0]
    assert not pb["m_q"].any() and pb["m_w"].tolist() == [0.0] * 19 + [1.0]
    assert np.allclose(pb["q"], e.cmd[0, :6]) and np.allclose(pb["u"][:19], e.cmd[0])
    assert (pb["ideal"][:19] == 0).all() and (pb["w"][:19] == 0).all()
    pb = W.past_block(e, 100, measured=True)
    assert pb["m_q"].all() and np.allclose(pb["q"], e.q[81:101])


def test_past_block_masks_mid_episode_dropout():
    e = T.synthetic_episode(0)
    e.q_valid = e.q_valid.copy(); e.q_valid[90] = False; e.q = e.q.copy(); e.q[90] = 0.0
    pb = W.past_block(e, 100, measured=True)
    i = 90 - 81
    assert pb["m_q"][i] == 0.0 and np.allclose(pb["q"][i], e.cmd[0, :6])


def test_score_matches_fabs_err_energy():
    import fit_step1_candidate as S1
    import train_fabs_g0 as A
    K = W.kernel()
    assert np.array_equal(K, A.bp_kernel(S1))
    e = T.synthetic_episode(1)
    eh = np.random.default_rng(0).normal(0, 0.01, e.gap.shape)
    num, den, f3 = W.score(e.gap, eh, K)
    a, c = A.err_energy(e.gap - eh, K)
    b, _ = A.err_energy(e.gap, K)
    assert np.isclose(num, a, rtol=1e-12) and np.isclose(den, b, rtol=1e-12) and f3 == c
    assert W.score(e.gap, np.zeros_like(e.gap), K)[0] == den
    assert W.score(e.gap, e.gap, K)[0] == 0.0


def test_mid_windows_and_overlay():
    e = T.synthetic_episode(0)
    mws = W.mid_windows(e)
    assert [w.t0 for w in mws] == [50, 150, 250, 350, 450, 550]
    out = W.overlay(np.zeros((e.n, 3)), mws, [np.ones((w.n_scored, 3)) for w in mws])
    assert out[49].sum() == 0 and out[50].sum() == 3 and out[100].sum() == 0


def test_band_clean_requires_full_support():
    ok = np.ones(200, bool); ok[100] = False
    bc = W.band_clean(ok)
    assert not bc[:D.EDGE].any() and not bc[-D.EDGE:].any()
    assert not bc[80:121].any() and bc[79] and bc[121]


def test_lead_bins_partition_scored_frames():
    e = T.synthetic_episode(2)
    ws = W.chunk_windows(e)
    K = W.kernel()
    tab = W.lead_score(e.gap, np.zeros_like(e.gap), ws, K)
    assert np.isclose(sum(v[1] for v in tab.values()), W.score(e.gap, 0 * e.gap, K)[1])
    lead = W.lead_of(e.n, ws)
    assert lead[0] == 3 and lead[99] == 102 and lead[100] == 3
