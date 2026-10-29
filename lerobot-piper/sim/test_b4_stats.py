"""`b4_stats.py`（B4 的统计与判定，方案 `reports/PLAN_b4.md` §6、约束 3）的测试。纯计算，不读数据。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b4_stats as S  # noqa: E402

GOOD = {"effect": -3.0, "ci95": [-4.0, -2.0]}
FLAT = {"effect": 0.5, "ci95": [-1.0, 2.0]}


def test_rel_change_is_percent_of_reference():
    assert S.rel_change(9.0, 10.0) == pytest.approx(-10.0)


def test_hier2_effect_is_mean_of_group_medians():
    r = S.hier2({"a": [-10, -12, -11, -9], "b": [-2, -4, -3, -1]})
    assert r["effect"] == pytest.approx((-10.5 + -2.5) / 2)
    assert r["n_groups"] == 2 and r["per_group"]["a"] == {"median": -10.5, "n": 4}


def test_hier2_resamples_groups_not_only_rounds():
    """一组强改善、一组强变差：先抽组时区间跨 0；旧 hier() 只抽轮次、组固定，区间约 [−5.05, −4.95]，不跨 0。"""
    r = S.hier2({"a": [-20.0, -20.1, -19.9, -20.0], "b": [10.0, 10.1, 9.9, 10.0]})
    assert r["ci95"][0] < 0 < r["ci95"][1]


def test_hier2_resamples_units_within_groups():
    """只有一组时，区间的宽度只能来自组内重抽。"""
    r = S.hier2({"a": [-1.0, -2.0, -30.0, -1.5, -2.5]})
    assert r["ci95"][0] < r["ci95"][1]


def test_judge_uses_joint_only():
    """约束 3：只看关节偏差（加运动护栏）；末端变差只报告，不影响判定。"""
    assert S.judge({"joint_raw": GOOD, "path": FLAT, "speed": FLAT, "ee_raw": {"effect": 40.0, "ci95": [20, 60]}}) == "有效"
    assert S.judge({"joint_raw": {"effect": -1.0, "ci95": [-3.0, 0.5]}, "path": FLAT, "speed": FLAT}) != "有效"
    assert "护栏" in S.judge({"joint_raw": GOOD, "path": {"effect": 6.0, "ci95": [5, 7]}, "speed": FLAT})
    assert "护栏" in S.judge({"joint_raw": GOOD, "path": FLAT, "speed": {"effect": -6.0, "ci95": [-7, -5]}})


def test_strict_count():
    assert S.strict_count([(True, True)] * 19 + [(True, False)]) == (1, 0, False)
    assert S.strict_count([(True, False), (False, True)] + [(True, True)] * 18) == (1, 1, True)


def test_chance_fail_prob_values():
    assert S.chance_fail_prob(0.0, 20) == 0.0
    assert S.chance_fail_prob(0.02, 20) == pytest.approx(0.2386, abs=1e-3)
    assert S.chance_fail_prob(0.05, 20) == pytest.approx(0.3449, abs=1e-3)
    assert S.chance_fail_prob(0.05, 200) == pytest.approx(0.454, abs=1e-3)


def test_judge_closed_needs_20_pairs_and_extra_guards():
    r = {"joint_raw": GOOD, "path": FLAT, "speed": FLAT, "duration": FLAT, "cmd_speed": FLAT}
    assert S.judge_closed(r, [(True, True)] * 20) == "有效"
    assert "未完成" in S.judge_closed(r, [(True, True)] * 19)
    assert S.judge_closed({**r, "duration": {"effect": 6.0, "ci95": [5, 7]}}, [(True, True)] * 20) != "有效"
    assert S.judge_closed({**r, "cmd_speed": {"effect": -6.0, "ci95": [-7, -5]}}, [(True, True)] * 20) != "有效"
    assert S.judge_closed(r, [(True, False)] + [(True, True)] * 19) != "有效"
    assert S.judge_closed({**r, "joint_raw": {"effect": -1.0, "ci95": [-3.0, 0.5]}}, [(True, True)] * 20) != "有效"
