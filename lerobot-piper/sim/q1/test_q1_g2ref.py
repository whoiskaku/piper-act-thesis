import copy
import json

import numpy as np
import pytest

import q1_data as D
import q1_g2ref as G
import q1_testing as T
import q1_windows as W


@pytest.fixture(scope="module")
def g2():
    return G.load_g2()


def test_gyro100_is_not_read(g2):
    e = T.synthetic_episode(3)
    dd = G._dd(e.cmd[:150], "x")
    a = np.asarray(g2.omega(dd))
    dd["gyro100"] = np.full_like(dd["gyro100"], 9.0)
    assert np.array_equal(np.asarray(g2.omega(dd)), a)


def test_window_ignores_commands_after_t1(g2, calib):
    e = T.synthetic_episode(4)
    w = W.chunk_windows(e)[2]
    base = G.window_ehat(g2, e, w, *calib)
    e2 = copy.copy(e)
    c = e.cmd.copy(); c[w.t1 + 1:] += 0.05; e2.cmd = c
    assert np.array_equal(G.window_ehat(g2, e2, w, *calib), base)


@pytest.mark.slow
def test_whole_episode_reproduces_fabs_g0(bare_eps, calib, g2):
    """旧口径必须逐会话复现 `outputs/fabs_g0/fabs_g0_v1.json` 的 G-2 redu（实现自检）。"""
    old = {r["session"]: r["g2_redu"] for r in json.loads(
        (D.ROOT / "outputs/fabs_g0/fabs_g0_v1.json").read_text())["folds"]}
    K = W.kernel()
    acc = {}
    for e in bare_eps:
        n, d, _ = W.score(e.gap, G.whole_episode_ehat(g2, e, *calib), K)
        a = acc.setdefault(e.sess, [0.0, 0.0]); a[0] += n; a[1] += d
    for s, (n, d) in acc.items():
        assert abs((1 - n / d) - old[s]) < 1e-5, (s, 1 - n / d, old[s])


@pytest.mark.slow
def test_clamping_changes_only_last_scored_frames(bare_eps, calib, g2):
    e = bare_eps[0]
    whole = G.whole_episode_ehat(g2, e, *calib)
    for w in W.chunk_windows(e)[:-1]:                    # 末 chunk 的旧口径另有序列末端效应
        d = np.abs(G.window_ehat(g2, e, w, *calib) - whole[w.t0:w.s1 + 1]).max(1)
        changed = np.where(d > 0)[0] + w.t0
        assert len(changed) > 0 and changed.min() >= w.t1 - 5, (w, changed)


def test_mid_origin_equals_main_for_g2(g2, calib):
    """G-2 不读实测 ⇒ 中段起点（同一 t1）的预测与主窗口逐位相同（`refs` 据此令 aux = main）。"""
    e = T.synthetic_episode(5)
    w = W.chunk_windows(e)[1]
    m = W.mid_windows(e)[1]
    assert np.array_equal(G.window_ehat(g2, e, m, *calib), G.window_ehat(g2, e, w, *calib)[50:])
