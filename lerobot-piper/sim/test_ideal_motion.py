"""老师定义的理想动作（`sim/ideal_motion.py`）的测试。合成数据。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ideal_motion as IM  # noqa: E402

FR = 0.05


def smooth_cmd(n=80, amp=0.2, hz=0.4):
    t = np.arange(n) * FR
    a = np.zeros((n, 7))
    a[:, :6] = amp * np.sin(2 * np.pi * hz * t)[:, None] * np.array([1, 0.5, -0.5, 0, 0.3, 0.2])
    return a                                                   # 加速度 ≤ amp·(2π·hz)² ≈ 1.3 rad/s²，在上限之内


def test_ideal_passes_through_feasible_commands():
    a = smooth_cmd()
    x = IM.ideal(a)
    assert x.shape == ((len(a) - 1) * IM.SUB + 1, 6)
    assert np.abs(x[::IM.SUB] - a[:, :6]).max() < 1e-7


def test_ideal_respects_the_acceleration_limit_at_a_step():
    a = np.zeros((40, 7))
    a[20:, 0] = 0.1                                            # 一帧之内跳 0.1 rad：物理上做不到
    x = IM.ideal(a)
    acc = np.abs(np.diff(x[:, 0], 2)) / IM.DTG ** 2
    assert acc.max() <= IM.A_MAX * (1 + 1e-6) and np.abs(x[::IM.SUB, 0] - a[:, 0]).max() > 1e-3


def test_lag_and_shape_recover_a_pure_delay():
    a = smooth_cmd()
    x = IM.ideal(a)
    ts = 10.0 + np.arange(len(a)) * FR
    tq = np.arange(ts[0], ts[-1], 0.005)
    cmd = IM.IdealCmd(x)
    q = cmd(np.clip(tq - ts[0] - 0.030, 0, None))[:, :6]       # 「实测」= 理想整体晚 30 ms
    ep = {"t_send": ts, "tq": tq, "q": q}
    free = tq >= ts[0] + 0.5
    r = IM.joint_ideal(ep, x, free)
    assert abs(r["lag_ms"] - 30) <= 2 and r["shape"] < 0.05 and r["dev"] > 1.0


def test_ideal_cmd_has_the_spline_interface():
    x = IM.ideal(smooth_cmd())
    cmd = IM.IdealCmd(x)
    t = np.array([0.5012, 1.0021, 1.5033])                     # 取在 5 ms 网格格子内部：格内位置是线性的
    p, v, acc = (cmd(t, nu) for nu in (0, 1, 2))
    assert p.shape == v.shape == acc.shape == (3, 7) and not p[:, 6].any()
    h = 5e-4
    assert np.allclose(v[:, :6], (cmd(t + h)[:, :6] - cmd(t - h)[:, :6]) / (2 * h), atol=1e-9)


# ── 真机验证脚本 b4_ideal_check（用户 11-12）────────────────────────────────────
def test_earlier_shifts_a_ramp_and_keeps_gripper_and_last_frame():
    t = np.arange(10) * FR
    cmd = np.zeros((10, 7))
    cmd[:, :6] = t[:, None] * np.arange(1, 7)
    cmd[:, 6] = 0.03
    e = IM.earlier(cmd, 0.024)
    assert np.allclose(e[:-1, :6], (t[:-1, None] + 0.024) * np.arange(1, 7)) and np.array_equal(e[-1], cmd[-1])
    assert np.array_equal(e[:, 6], cmd[:, 6])


def test_plan_reverses_the_condition_order_in_round_two():
    import b4_ideal_check as V
    assert V.plan(["p1"], V.SETS["ideal"][1]) == [("p1", "shift60", 1), ("p1", "shift84", 1), ("p1", "teacher24", 1),
                                                  ("p1", "teacher24", 2), ("p1", "shift84", 2), ("p1", "shift60", 2)]
    assert V.plan(["p1"], V.SETS["b4val"][1]) == [("p1", "shift84", 1), ("p1", "regen", 1), ("p1", "regen", 2), ("p1", "shift84", 2)]


def test_judge_uses_the_given_main_metric():                     # 用户 11-12：主指标改为相对理想的关节偏差
    import b4_stats as ST
    ok = {"effect": 0.0, "ci95": [-1.0, 1.0]}
    r = {"path": ok, "speed": ok, "joint_raw": {"effect": 5.0, "ci95": [2.0, 8.0]}, "joint_ideal": {"effect": -30.0, "ci95": [-40.0, -20.0]}}
    assert ST.judge(r) == "关节偏差没有显著下降" and ST.judge(r, main="joint_ideal") == "有效"


def test_fbval_runs_both_conditions_online_and_only_b4fb_turns_feedback_on(tmp_path, monkeypatch):   # 设计 PLAN_b3.md §9
    import b4_ideal_check as V
    calls = []
    monkeypatch.setattr(V.B, "check_session", lambda sess, calib: None)
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    monkeypatch.setattr(V.RR, "run_child", lambda args: calls.append(args) or 0)
    monkeypatch.setitem(V.SETS, "fbval", (("p07",), ("b4", "b4fb"), "f"))
    (tmp_path / "replay_p07_fb4fb2_r1.json").write_text("{}")                  # 已开始过的那一遍跳过
    assert V.cmd_replay("fbval", tmp_path) == 0
    assert [(a[a.index("--rid") + 1], "--fb" in a) for a in calls] == [("p07_fb41", False), ("p07_fb4fb1", True), ("p07_fb42", False)]
    assert all(Path(a[2]).name == "b4_regen_replay.py" and a[3] == "run" and a[a.index("--mode") + 1] == "regen"
               and a[a.index("--net") + 1] == str(V.R.REGEN / "frozen.json") and a[a.index("--tid") + 1] == "p07" for a in calls)
