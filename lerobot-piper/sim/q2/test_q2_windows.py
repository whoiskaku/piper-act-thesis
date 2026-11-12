import copy

import numpy as np
import pytest

import q1_windows as W
import q2_testing as T
import q2_windows as W2


def test_da_visible_zeroes_rows_after_t1_and_after_the_episode_end():
    e = T.synthetic(0)
    w = W.chunk_windows(e)[1]                                        # t0 = 100, t1 = 199
    v = W2.da_visible(e, w, w.t1 + 5)
    assert np.array_equal(v[:w.t1 + 1], e.da[:w.t1 + 1]) and (v[w.t1 + 1:] == 0).all()
    last = W.chunk_windows(e)[-1]                                    # t1 = 599 > T − 1
    v = W2.da_visible(e, last, last.t1)
    assert v.shape == (last.t1 + 1, 6) and np.array_equal(v[:e.T], e.da) and (v[e.T:] == 0).all()


def test_da_span_pads_frames_before_the_episode_start():
    e = T.synthetic(1)
    w = W.chunk_windows(e)[0]
    s = W2.da_span(e, w, -3, 2)
    assert (s[:3] == 0).all() and np.array_equal(s[3:], e.da[:3])


def test_h_window_matches_the_gap_axis_formula_without_shift():
    e = T.synthetic(2)
    H = np.random.default_rng(0).normal(0, 1, (16, 3, 6))
    w = W.chunk_windows(e)[2]
    got = W2.h_window(H, e, w)
    t = w.t0 + 7
    want = sum(H[tau] @ e.da[t - tau] for tau in range(16))         # Δê_t = Σ_τ H[τ] δa_{t−τ}，不再平移 L=3
    assert got.shape == (w.n_scored, 3) and np.allclose(got[7], want)


def test_h_window_uses_past_corrections_and_never_future_ones():
    e = T.synthetic(3)
    H = np.random.default_rng(1).normal(0, 1, (16, 3, 6))
    w = W.chunk_windows(e)[2]
    base = W2.h_window(H, e, w)
    e1 = copy.copy(e); d = e.da.copy(); d[w.t1 + 1:] += 0.01; e1.da = d     # 改下一 chunk 的补正 ⇒ 不变
    assert np.array_equal(W2.h_window(H, e1, w), base)
    e2 = copy.copy(e); d = e.da.copy(); d[w.t0 - 3] += 0.01; e2.da = d      # 改过去的补正 ⇒ 窗口开头变（不清零重启）
    assert not np.allclose(W2.h_window(H, e2, w)[:3], base[:3])


def test_load_h_checks_the_frozen_fingerprint():
    H = W2.load_h()
    assert H.shape == (16, 3, 6)
