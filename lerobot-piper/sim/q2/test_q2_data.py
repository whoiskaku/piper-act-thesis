import json

import numpy as np
import pytest

import q1_data as D1
import q2_data as D2
import q2_testing as T

EXPECTED = {"ab5/A": 40, "ab6/A": 30, "ab7/Apost": 15, "ab5/C": 40, "ab5/Csafe": 3,
            "ab6/D": 30, "ab7/D": 15, "ab4/P2": 6, "ab4/P3": 4}


def test_roster_is_frozen_with_the_expected_groups():
    assert {k: len(v) for k, v in D2.ROSTER.items()} == EXPECTED
    names = [n for v in D2.ROSTER.values() for n in v]
    assert len(names) == len(set(names)) == 183
    assert all(n.startswith(k.split("/")[0] + "/" + k.split("/")[1] + "_ep") for k, v in D2.ROSTER.items() for n in v)


def test_synthetic_episode_fields():
    e = T.synthetic(0, "ab6", "D", "corr")
    assert (e.fold, e.arm, e.kind) == ("ab6", "D", "corr") and e.da.shape == (e.T, 6)
    assert np.abs(e.da).max() > 0 and (e.da[:, 3] == 0).all()
    b = T.synthetic(1, "ab6", "A", "bare")
    assert (b.da == 0).all()


def _log(tmp_path, kind, tamper=0.0, zero=False):
    rng = np.random.default_rng(0)
    T_ = 120
    raw = rng.normal(0, 0.1, (T_, 7))
    da = np.zeros((T_, 6)) if (kind == "bare" or zero) else rng.normal(0, 1e-3, (T_, 6))
    a = raw.copy(); a[:, :6] += da; a[5, 0] += tamper
    pay = {"frames": [{"a": a[k].tolist(), "da": da[k].tolist()} for k in range(T_)],
           "chunks": [{"start": 0, "raw": raw[:100].tolist()}, {"start": 100, "raw": raw[100:].tolist()}]}
    p = tmp_path / f"{kind}.json"
    p.write_text(json.dumps(pay))
    return p, T_, da


@pytest.mark.parametrize("kind", ["bare", "corr", "aux"])
def test_log_da_accepts_a_consistent_log(tmp_path, kind):
    p, T_, da = _log(tmp_path, kind)
    assert np.array_equal(D2.log_da(p, T_, kind), da)


def test_log_da_rejects_inconsistent_logs(tmp_path):
    p, T_, _ = _log(tmp_path, "corr", tamper=1e-4)                  # a ≠ u + δa
    with pytest.raises(D2.LogContractError):
        D2.log_da(p, T_, "corr")
    p, T_, _ = _log(tmp_path, "corr", zero=True)                    # 补正运行的 δa 全为 0
    with pytest.raises(D2.LogContractError):
        D2.log_da(p, T_, "corr")
    p, T_, _ = _log(tmp_path, "corr")                               # 当作裸跑读：δa 非零
    with pytest.raises(D2.LogContractError):
        D2.log_da(p, T_, "bare")
    assert issubclass(D2.LogContractError, D1.DataContractError)    # 不会被加载循环（只吞 ValueError）吞掉


def test_check_roster_catches_a_swap_that_keeps_the_total():
    eps = T.synthetic_set()
    roster = T.roster_of(eps)
    D2.check_roster(eps, roster=roster)
    swapped = [e for e in eps if e.name != eps[0].name] + [T.synthetic(99, eps[0].fold, "D" if eps[0].fold != "ab5" else "C", "corr")]
    with pytest.raises(D1.DataContractError):
        D2.check_roster(swapped, roster=roster)


def test_check_roster_only_checks_the_requested_groups():
    eps = T.synthetic_set()
    roster = T.roster_of(eps)
    part = [e for e in eps if e.fold == "ab6"]
    D2.check_roster(part, keys=["ab6/A", "ab6/D"], roster=roster)
    with pytest.raises(D1.DataContractError):
        D2.check_roster(part, keys=["ab6/A", "ab6/D", "ab7/D"], roster=roster)


def test_spec_for_selects_arms_by_fold_and_kind():
    assert D2.spec_for(("ab6",), ("bare", "corr")) == [({"ab6": ("A",)}, "bare"), ({"ab6": ("D",)}, "corr")]
    assert D2.spec_for(("ab5",), ("aux",)) == [({"ab4": ("P2", "P3")}, "aux")]


@pytest.mark.slow
def test_real_data_matches_the_roster_and_contracts():
    eps = D2.load()                                                 # 核对名单与日志契约
    assert {k: len(v) for k, v in D2.groups(eps).items()} == EXPECTED
    assert all(e.da.shape == (e.T, 6) for e in eps)
    assert all((e.da == 0).all() for e in eps if e.kind == "bare")
    for f in D2.FOLDS:                                              # 近期裸跑 = 阶段一冻结名单里的同一批集
        arm = D2.BARE[f][0]
        assert sorted(e.name for e in eps if e.fold == f and e.kind == "bare") == sorted(D1.ROSTER[f"{f}/{arm}"])
