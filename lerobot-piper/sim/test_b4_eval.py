"""`b4_eval.py`（回放评价，方案 `reports/PLAN_b4.md` §5.4、§6；实现计划任务 10）的测试。合成数据，不连机械臂。"""
from __future__ import annotations

import gc

gc.disable()                      # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）

import sys  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b4_eval as E  # noqa: E402
import b4_teacher as B  # noqa: E402
import ilc_route1 as I  # noqa: E402
import repeat_reversal as RR  # noqa: E402

from lerobot.common.policies.act.b4_deploy import B4Deploy  # noqa: E402
from tests.test_b4_deploy import CFG, CHUNKS, MODEL, ChunkReplayer  # noqa: E402


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    dep = B4Deploy(ChunkReplayer(CHUNKS), "shift", CFG, "x")
    for _ in range(len(CHUNKS)):
        dep.select_action({})
    p = tmp_path_factory.mktemp("b4") / "ep_b4.npz"
    dep.save_episode_log(p)
    log = dict(np.load(p))
    seq = B.build_seq(log, log["u"], CFG, mujoco.MjModel.from_xml_path(str(MODEL)))
    return SimpleNamespace(log=log, seq=seq, net=lambda x: 0.01 * np.tanh(x[:5]))


def test_net_command_equals_deploy_output(case, tmp_path):
    """约束 1、2：回放用的 Shift＋网络指令，和部署时在同一串整段预测上会发出的指令逐位相同。"""
    live = B4Deploy(ChunkReplayer(CHUNKS), "net", CFG, "x", net=case.net)
    for _ in range(len(CHUNKS)):
        live.select_action({})
    live.save_episode_log(tmp_path / "live.npz")
    u = np.load(tmp_path / "live.npz")["u"]
    cmd7 = E.net_command(case.log, case.seq, case.net, 8, 8, CFG)
    assert np.array_equal(cmd7[RR.HOLD:], u) and np.array_equal(cmd7[:RR.HOLD], np.repeat(u[:1], RR.HOLD, 0))
    assert not np.array_equal(cmd7, case.seq["s7"])                     # 网络确实加了补正


def test_val_plan_alternates_order():
    p = E.plan("val", ["a"])
    assert p == [("a", "shift", 1), ("a", "net", 1), ("a", "net", 2), ("a", "shift", 2)]


def test_test_plan_is_balanced():
    p = E.plan("test", ["a"])
    assert Counter(c for _, c, _ in p) == {"shift": 4, "net": 4, "teacher": 4}
    firsts = [next(c for _, c, r in p if r == rr) for rr in (1, 2, 3)]
    assert sorted(firsts) == ["net", "shift", "teacher"]


def results(n_traj=10, net=0.9, ee_net=1.4):
    """合成：每条 4 轮；网络关节偏差是 Shift 的 net 倍（末端角速度 ee_net 倍），教师 0.85 倍。"""
    rng = np.random.default_rng(0)
    out = {}
    for t in range(n_traj):
        d = {}
        for r in (1, 2, 3, 4):
            j = 5.0 + rng.normal(0, 0.05)
            m = lambda f, fe: {"joint_raw": j * f, "ee_raw": 0.06 * fe, "path": 1.0, "speed": 1.0}   # noqa: E731
            d[("shift", r)], d[("net", r)], d[("teacher", r)] = m(1, 1), m(net, ee_net), m(0.85, 1)
        out[f"t{t}"] = d
    return out


def test_analyze_never_calls_old_verdict(monkeypatch):
    """约束 3：两级自助、只看关节偏差；末端变差只报告；不调用旧版判定。"""
    for name in ("verdict", "compare", "hier"):
        monkeypatch.setattr(I, name, lambda *a, _n=name, **k: pytest.fail(f"调用了 ilc_route1.{_n}"))
    out = E.analyze_results(results(), ("shift", "net", "teacher"), 4)
    assert out["verdict"] == "有效" and out["joint_raw"]["n_groups"] == 10
    assert out["ee_raw"]["effect"] > 30                                 # 末端变差照样报告出来
    assert out["gain_ratio"] == pytest.approx(0.1 / 0.15, rel=0.05)     # 网络拿到 ILC 收益的约 2/3


def test_incomplete_trajectory_is_reported_not_judged():
    res = results()
    res["t3"][("net", 4)] = None
    out = E.analyze_results(res, ("shift", "net", "teacher"), 4)
    assert out["per_traj"]["t3"]["status"] == "未完成" and out["joint_raw"]["n_groups"] == 9
    assert "未完成" in out["verdict"] and "9/10" in out["verdict"]          # 没跑齐：整体不下「有效」，已有数据照常报告
    assert out["joint_raw"]["effect"] < 0


def test_planned_id_without_any_data_still_counts():
    res = results()
    res["t9"] = {}                                                      # 预定的一条一遍都没跑
    out = E.analyze_results(res, ("shift", "net", "teacher"), 4)
    assert "未完成" in out["verdict"] and out["per_traj"]["t9"]["status"] == "未完成"


def test_no_improvement_is_not_effective():
    assert E.analyze_results(results(net=1.0), ("shift", "net", "teacher"), 4)["verdict"] != "有效"


def test_session_ctx_uses_the_sessions_own_calibration(tmp_path):
    """跨天分析：每个会话按 session.json 记下的标定哈希找那一份标定（含备份），不用当前的。"""
    import json
    cal = tmp_path / "imu_calib"
    cal.mkdir()
    (cal / "imu_calib.json").write_text(json.dumps({"R": "today"}))
    (cal / "imu_calib_20261030.json").write_text(json.dumps({"R": "old"}))
    sess = tmp_path / "s"
    sess.mkdir()
    (sess / "session.json").write_text(json.dumps({"calib_sha": B._sha(cal / "imu_calib_20261030.json")}))
    ctx = B.session_ctx(sess, cal)
    assert ctx.calib == {"R": "old"} and ctx.calib_sha == B._sha(cal / "imu_calib_20261030.json")
    (sess / "session.json").write_text(json.dumps({"calib_sha": "0" * 16}))
    with pytest.raises(SystemExit):
        B.session_ctx(sess, cal)


def test_cached_result_is_bound_to_calibration(tmp_path):
    """缓存的结果只在标定哈希相同时复用；换了标定就重算（这里没有回放记录，重算会找不到文件）。"""
    import json
    e = np.zeros((3, 5))
    np.savez(tmp_path / "res_x.npz", met=json.dumps({"joint_raw": 1.0}), eq=e, ew=e, ee=e, qc=0.5, calib_sha="aaa")
    assert B.replay_result("x", tmp_path, {}, SimpleNamespace(calib_sha="aaa"))[0] == {"joint_raw": 1.0}
    with pytest.raises(FileNotFoundError):
        B.replay_result("x", tmp_path, {}, SimpleNamespace(calib_sha="bbb"))


def _fake_session(root, name, tid, rounds, net_sha, conds=("shift", "net")):
    import json
    s = root / name
    s.mkdir(parents=True)
    for c in conds:
        for r in rounds:
            rid = f"{tid}_e{c}{r}"
            (s / f"replay_{rid}_r1.json").write_text("{}")
            np.savez(s / f"seq_{rid}.npz", source=json.dumps({"net_sha": net_sha if c == "net" else ""}))
    return s


def test_pick_session_binds_network_and_whole_trajectory(tmp_path):
    """审阅 Important 2：一条轨迹的全部轮次必须来自同一会话，且网络那几遍用的是指定的网络；取最新的那个会话。"""
    old = _fake_session(tmp_path, "20261101", "a", (1, 2), "OLD")
    new = _fake_session(tmp_path, "20261103", "a", (1, 2), "NEW")
    _fake_session(tmp_path, "20261102", "b", (1,), "NEW")
    _fake_session(tmp_path, "20261104", "b", (2,), "NEW")
    ss = [old, tmp_path / "20261102", new, tmp_path / "20261104"]
    assert E.pick_session("a", ss, ("shift", "net"), 2, "NEW") == new
    assert E.pick_session("a", ss, ("shift", "net"), 2, "OLD") == old
    assert E.pick_session("b", ss, ("shift", "net"), 2, "NEW") is None          # 两轮跨了会话：不拼
