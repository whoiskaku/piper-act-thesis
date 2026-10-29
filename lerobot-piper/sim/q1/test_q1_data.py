import numpy as np
import pytest

import q1_data as D
import q1_testing as T


def test_synthetic_episode_invariants():
    e = T.synthetic_episode(0)
    assert e.n == e.T - D.ALIGN_L
    assert e.cmd.shape == (600, 7) and e.ideal.shape == (600, 3) and e.caps.shape == (600,)
    assert not e.q_valid[:D.Q_VALID_FROM].any() and e.q_valid[D.Q_VALID_FROM:].all()
    assert np.allclose(e.gap, e.w[D.ALIGN_L:] - e.ideal[:e.n])


def test_measured_joints_masks_head_and_dropout():
    st = np.zeros((10, 7))
    st[2:, :6] = 1000.0
    st[6, :6] = 0.0                                   # 第 6 帧掉线（全零）
    q, valid = D.measured_joints(st)
    assert valid.tolist() == [False, False, True, True, True, True, False, True, True, True]
    assert np.allclose(q[2], np.radians(1.0)) and (q[6] == 0).all()


def test_measured_joints_rejects_nonzero_head():
    with pytest.raises(D.DataContractError):
        D.measured_joints(np.ones((10, 7)))


def test_chunk_contract():
    row = [0.0] * 7
    D.check_chunks([{"start": 100, "raw": [row] * 100}, {"start": 0, "raw": [row] * 100}])
    with pytest.raises(D.DataContractError):
        D.check_chunks([{"start": 0, "raw": [row] * 101}])
    with pytest.raises(D.DataContractError):
        D.check_chunks([{"start": 0, "raw": [row] * 100}, {"start": 120, "raw": [row] * 100}])


@pytest.mark.slow
def test_load_bare_matches_fabs_loader(bare_eps, calib):
    import fit_step1_candidate as S1
    import train_fabs_g0 as A
    assert {k: len(v) for k, v in D.by_session(bare_eps).items()} == D.EXPECTED_COUNTS
    old = A.load_all(S1, D.renderer(), *calib)
    assert [o.path for o in old] == [e.path for e in bare_eps]
    for e, o in zip(bare_eps, old):
        assert np.abs(e.gap - o.y).max() <= 2e-7                # o.y 存为 float32
        assert np.array_equal(e.ideal[:e.n], o.ideal)
        assert e.q_valid[D.Q_VALID_FROM:].all() and not e.q_valid[:D.Q_VALID_FROM].any()
        assert len(e.cmd) == 600 and set(e.caps.tolist()) == {99, 199, 299, 399, 499, 599}


def test_roster_is_frozen_with_the_expected_counts():
    """名单随代码冻结（q1_roster.json 在 CODE_FILES 里）：各会话 40/30/15/10/5、集名不重复。"""
    import q1_train as TR
    assert "q1_roster.json" in TR.CODE_FILES
    assert {s: len(v) for s, v in D.ROSTER.items()} == D.EXPECTED_COUNTS
    names = [n for v in D.ROSTER.values() for n in v]
    assert len(set(names)) == 100 and all(n.startswith(s + "_ep") for s, v in D.ROSTER.items() for n in v)


def test_check_roster_rejects_a_swap_that_keeps_the_total():
    """ab5 少一集、ab6 多一集：总数仍是 100，也必须拦下。"""
    from types import SimpleNamespace as NS
    eps = [NS(sess=s, name=n) for s, v in D.ROSTER.items() for n in v]
    D.check_roster(eps)
    swapped = [e for e in eps if e.name != "ab5/A_ep3"] + [NS(sess="ab6/A", name="ab6/A_ep30")]
    assert len(swapped) == 100
    with pytest.raises(D.DataContractError):
        D.check_roster(swapped)
    with pytest.raises(D.DataContractError):
        D.check_roster(eps[:-1])


@pytest.mark.slow
def test_real_data_matches_the_frozen_roster(bare_eps):
    D.check_roster(bare_eps)
