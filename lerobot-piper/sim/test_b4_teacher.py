"""`b4_teacher.py` 的测试（方案 `reports/PLAN_b4.md` §5.2；实现计划任务 6、7）：教师的回放序列由闭环 Shift 的记录逐字节构成（约束 2），
记录有问题就拒绝；序列文件不覆盖。记录由任务 3 的部署包装（shift 模式）跑合成整段预测得到；不连机械臂、不读数据集。"""
from __future__ import annotations

import gc

gc.disable()                      # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）

import dataclasses  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b4_teacher as B  # noqa: E402
import repeat_reversal as RR  # noqa: E402

from lerobot.common.policies.act.b4_deploy import B4Deploy  # noqa: E402
from tests.test_b4_deploy import CFG, CHUNKS, MODEL, ChunkReplayer  # noqa: E402

M = mujoco.MjModel.from_xml_path(str(MODEL))


@pytest.fixture(scope="module")
def shift_log(tmp_path_factory):
    dep = B4Deploy(ChunkReplayer(CHUNKS), "shift", CFG, "x")
    for _ in range(len(CHUNKS)):
        dep.select_action({})
    p = tmp_path_factory.mktemp("b4") / "ep_b4.npz"
    dep.save_episode_log(p)
    return dict(np.load(p))


def test_s7_is_logged_online_shift_bitwise(shift_log):
    """约束 2：教师的 Shift 基准就是录制时实发的在线 Shift（逐字节），窗就是录制时记下的窗；开头补的保持段窗为 0。"""
    q = B.build_seq(shift_log, shift_log["u"], CFG, M)
    assert np.array_equal(q["s7"][RR.HOLD:], shift_log["u"].astype(np.float32))
    assert np.array_equal(q["w"][RR.HOLD:], shift_log["w"]) and not q["w"][:RR.HOLD].any()
    assert np.array_equal(q["a7"][:RR.HOLD], np.repeat(shift_log["a_pi"][:1], RR.HOLD, 0))


def test_rejects_dataset_action_mismatch(shift_log):
    """Review Focus 4：数据集里记的动作与包装实发的不一致（发送时被改过）→ 不能当基准。"""
    act = shift_log["u"].copy()
    act[50, 1] += 1e-3
    with pytest.raises(SystemExit, match="数据集动作"):
        B.build_seq(shift_log, act, CFG, M)


def test_rejects_raw_or_incomplete_log(shift_log):
    for bad in ({**shift_log, "mode": np.array("raw")}, {**shift_log, "complete": np.array(False)}):
        with pytest.raises(SystemExit):
            B.build_seq(bad, bad["u"], CFG, M)


def test_rejects_shift_not_reproducible_from_chunks(shift_log):
    """约束 2：记录的实发指令必须能由整段预测复算出来（数据集动作也跟着改，只剩复算这一关）。"""
    u = shift_log["u"].copy()
    u[30, 0] += 1e-3
    with pytest.raises(SystemExit, match="复算"):
        B.build_seq({**shift_log, "u": u}, u, CFG, M)


def test_contact_frames_and_unsafe_count(shift_log):
    """疑似接触帧（v0.10 口径，评价与 ILC 误差用）落在夹爪切换附近；在线窗多留了 0.1 s，没有接触帧落在窗 > 0 处。"""
    q = B.build_seq(shift_log, shift_log["u"], CFG, M)
    assert q["marked"][RR.HOLD + 30] and q["marked"][RR.HOLD + 45] and not q["marked"][RR.HOLD + 5]
    assert q["n_unsafe"] == 0


def test_rejects_shift_outside_hard_bounds(shift_log):
    """Shift 指令本身离管线硬界不足 0.2° 时，Δ = 0 都不满足上下限 → 拒绝。"""
    mid = 0.5 * (np.asarray(CFG.jnt_lo) + np.asarray(CFG.jnt_hi))      # 合成轨迹在限位中点 ± 5 % 行程内摆动
    cfg = dataclasses.replace(CFG, jnt_hi=tuple(mid - 0.5))
    with pytest.raises(SystemExit, match="上下限"):
        B.build_seq(shift_log, shift_log["u"], cfg, M)


def test_save_seq_never_overwrites_different_content(shift_log, tmp_path):
    q = B.build_seq(shift_log, shift_log["u"], CFG, M)
    p = tmp_path / "t" / "seq_base.npz"
    h = B.save_seq(p, q, {"id": "t"})
    assert B.save_seq(p, q, {"id": "t"}) == h                       # 内容相同：不报错
    q2 = {**q, "s7": q["s7"] + np.float32(1e-3)}
    with pytest.raises(SystemExit, match="不覆盖"):
        B.save_seq(p, q2, {"id": "t"})


# ── 任务 7：教师运行与独立验收（合成回放结果，不连机械臂）───────────────────
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

import ilc_route1 as I  # noqa: E402

N = 120


def fake_seq():
    t = np.arange(N) * 0.05
    a7 = np.zeros((N, 7), np.float32)
    a7[:, :6] = 0.3 * np.sin(t[:, None] * np.arange(1, 7) * 0.5)
    w = np.ones((N, 5))
    w[:RR.HOLD] = 0
    return {"a7": a7, "s7": a7.copy(), "w": w, "lo": np.full((N, 5), -0.015), "hi": np.full((N, 5), 0.015),
            "marked": np.zeros(N, bool)}


class Fake:
    """假的回放：execute 只留下回放记录文件（attempted 就认为跑过了）；result 按 rid 给合成指标。
    trip 里的那几遍关节偏差 20 mrad（超过护栏 1.05 × 基准）；stop_at = 第几次执行时返回 0（模拟中途退出）。"""

    def __init__(self, sess, trip=(), stop_at=None, invalid=(), qc_bad=()):
        self.sess, self.q, self.trip, self.stop_at, self.calls = sess, fake_seq(), set(trip), stop_at, 0
        self.invalid, self.qc_bad = set(invalid), set(qc_bad)

    def execute(self, k, n, rid, seq, L):
        if self.stop_at is not None and self.calls + 1 == self.stop_at:
            return 0
        (self.sess / f"replay_{rid}_r1.json").write_text("{}")
        self.calls += 1
        return True

    def result(self, rid):
        tag = rid.split("_")[-1]
        lk, n = tag[:-2], int(tag[-2:])
        if tag in self.invalid:
            return None
        jq = 20.0 if tag in self.trip else {"b": 10.0, "c": 9.0 - 0.1 * n, "h": 8.5 - 0.05 * n, "as": 10.0, "at": 9.0}[lk]
        e = 0.002 * (1 + 0.01 * n) * np.ones((N, 5))
        return {"joint_raw": jq, "ee_raw": 0.06, "path": 1.0, "speed": 1.0}, (e, 0.5 * e, 0.4 * e), 3.0 if tag in self.qc_bad else 0.5


def walk(tmp_path, **kw):
    f = Fake(tmp_path, **kw)
    return B.walk_teacher("t", tmp_path, f.execute, f.result, f.q), f


def sent_delta(sess, rid, q):
    return (np.load(sess / f"seq_{rid}.npz")["cmd7"] - q["s7"])[:, list(B.O.ACTIVE)]


def test_schedule_is_12_learning_plus_6_acceptance():
    assert len(B.SCHEDULE) == 12 and [k for k, _ in B.SCHEDULE].count("h") == 6
    assert [k for k, _ in B.SCHEDULE].count("b") == 2 and [k for k, _ in B.SCHEDULE].count("c") == 4
    assert B.ACCEPT == [("as", 1), ("at", 1), ("at", 2), ("as", 2), ("as", 3), ("at", 3)]


def test_probes_and_confirmations_use_the_12(tmp_path):
    """预算记账：护栏触发后的确认、无效的一遍都占这 12 遍，不追加。"""
    st, f = walk(tmp_path, trip={"c02", "h03"}, invalid={"h05"})
    assert len(st["learning"]) == 12 and st["h"].left == 0 and st["c"].left == 0
    assert f.calls == 18


def test_orientation_stop_during_learning(tmp_path):
    """学习中某一遍 IMU 朝向偏差超过 2.5° → 停下（退出码 3），不冻结、不验收。"""
    st, f = walk(tmp_path, qc_bad={"c02"})
    assert st["stopped"] == 3 and f.calls == 4 and "delta" not in st
    assert not (tmp_path / "teacher_t.npz").exists()


def test_rerun_refuses_changed_sequence(tmp_path):
    """重跑时重建出的序列和当时回放的不同（代码或输入变了）→ 停下，不拿不一样的指令接着学。"""
    walk(tmp_path, stop_at=5)
    f = Fake(tmp_path)
    f.q["s7"] = f.q["s7"] + np.float32(1e-3)
    with pytest.raises(SystemExit, match="重建出的序列"):
        B.walk_teacher("t", tmp_path, f.execute, f.result, f.q)


def test_frozen_delta_is_last_accepted(tmp_path):
    """第 5 遍触发护栏、第 6 遍确认也超 → 退回：冻结的是第 4 遍时接受、实际发出过的那份补正；验收发的就是它。"""
    st, f = walk(tmp_path, trip={"h05", "h06"})
    assert np.allclose(st["delta"], sent_delta(tmp_path, "t_h04", f.q), atol=1e-7)
    assert not np.allclose(st["delta"], sent_delta(tmp_path, "t_h05", f.q), atol=1e-7)
    t = np.load(tmp_path / "teacher_t.npz")
    assert np.array_equal(np.load(tmp_path / "seq_t_at01.npz")["cmd7"], t["cmd7"])
    assert not sent_delta(tmp_path, "t_as01", f.q).any()


def test_acceptance_never_reaches_learners(tmp_path, monkeypatch):
    calls, orig = [], I.Learner.add
    monkeypatch.setattr(I.Learner, "add", lambda self, *a, **k: calls.append(self.kind) or orig(self, *a, **k))
    st, _ = walk(tmp_path)
    assert len(st["accept"]) == 6 and st["verdict"] == (True, [])
    assert Counter(calls) == {"c": 4, "h": 6}                         # 验收的 6 遍一遍也没进学习器
    assert st["h"].kind == "h" and st["h"].wr == 0.5                  # 加 IMU 支，IMU 独有部分权重 ½（用户 10-27）


def test_resume_reconstructs_without_rerunning(tmp_path):
    """中途退出后再跑：已跑过的遍不重跑，按记录确定性地重建，最后冻结的补正和一口气跑完的相同。"""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    st1, f1 = walk(a, stop_at=8)
    assert st1["stopped"] == 0 and f1.calls == 7
    st2, f2 = walk(a)
    full, _ = walk(b)
    assert f2.calls == 11 and np.array_equal(st2["delta"], full["delta"])


def acc(round2_worse=False, path_plus=0.0, invalid=None, qc=0.5):
    out = {}
    for r in (1, 2, 3):
        out[("as", r)] = ({"joint_raw": 10.0, "path": 1.0, "speed": 1.0}, qc)
        out[("at", r)] = ({"joint_raw": 10.5 if round2_worse and r == 2 else 9.0, "path": 1.0 + path_plus, "speed": 1.0}, qc)
    if invalid:
        out[invalid] = None
    return out


def test_accept_verdict_cases():
    assert B.accept_verdict(acc()) == (True, [])
    assert not B.accept_verdict(acc(round2_worse=True))[0]
    assert not B.accept_verdict(acc(path_plus=0.06))[0]
    assert not B.accept_verdict(acc(invalid=("at", 3)))[0]
    assert not B.accept_verdict(acc(qc=3.0))[0]


def test_session_rejects_other_or_stale_calibration(tmp_path):
    calib = tmp_path / "imu_calib.json"
    calib.write_text("{}")
    sess = tmp_path / "s"
    sess.mkdir()
    B.check_session(sess, calib)                                       # 今天标定、新会话：可以
    (sess / "session.json").write_text(json.dumps({"calib_sha": "0" * 16}))
    with pytest.raises(SystemExit):
        B.check_session(sess, calib)                                   # 标定和会话开始时不同
    (sess / "session.json").unlink()
    old = time.time() - 2 * 86400
    os.utime(calib, (old, old))
    with pytest.raises(SystemExit):
        B.check_session(sess, calib)                                   # 标定不是今天做的


def test_next_ids_picks_unfinished_train_and_val_in_order():
    """--next N：按录制顺序挑训练、验证里还没跑完（没有验收结论）的教师；测试集、备用不挑。"""
    order = ["p1", "p2", "p3", "p4", "p5", "p6"]
    split = {"p1": "train", "p2": "test", "p3": "val", "p4": "train", "p5": "spare", "p6": "train"}
    assert B.next_ids(2, order, split, done={"p1"}) == ["p3", "p4"]
    assert B.next_ids(9, order, split, done=set()) == ["p1", "p3", "p4", "p6"]
