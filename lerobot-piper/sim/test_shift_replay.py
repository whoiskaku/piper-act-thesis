"""`shift_replay.py` 的测试：提前后的序列、排程平衡、误差相对原始指令、离线预测的方向、补正量、配对与判据 b。不连机械臂、不读数据集。"""
from __future__ import annotations

import gc

gc.disable()                      # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）

import sys  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shift_replay as S  # noqa: E402


def test_shifted_zero_whole_frame_and_fraction():
    rng = np.random.default_rng(0)
    cmd7 = np.cumsum(rng.normal(0, 0.01, (200, 7)), 0).astype(np.float32).astype(float)   # 同 cmd_seq：float32 序列读成 float
    assert np.array_equal(S.shifted(cmd7, 0.0), cmd7.astype(np.float32))                  # 提前 0 = 原序列（逐位）
    assert np.allclose(S.shifted(cmd7, S.RR.DT), np.r_[cmd7[1:], cmd7[-1:]], atol=1e-6)   # 提前一整帧 = 后一帧，末尾保持
    x = S.shifted(cmd7, 0.015)                                                              # 不足一帧：帧间线性插值（0.3 帧）
    assert np.allclose(x[:-1], cmd7[:-1] + 0.3 * np.diff(cmd7, axis=0), atol=1e-6)
    assert np.allclose(x[-1], cmd7[-1], atol=1e-6)


def test_schedule_each_condition_once_per_round_and_balanced_positions():
    plan = S.schedule()
    assert len(plan) == len(set(plan)) == len(S.SEQS) * len(S.CONDS) * S.N_REP           # 36 遍，不重复
    pos = Counter()
    for k in range(0, len(plan), len(S.CONDS)):                                            # 每 3 遍 = 同一序列同一轮的三个条件
        block = plan[k:k + len(S.CONDS)]
        assert len({(s, r) for s, _, r in block}) == 1 and {c for _, c, _ in block} == set(S.CONDS)
        pos.update((c, i) for i, (_, c, _) in enumerate(block))
    assert len(pos) == len(S.CONDS) ** 2 and set(pos.values()) == {len(S.SEQS) * S.N_REP // len(S.CONDS)}   # 各位置各 4 次


def test_deviation_is_relative_to_the_original_command():
    import mujoco
    from ideal_imu_kinematic import MODEL_PATH, ideal_imu_traj
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site")
    calib = {"R": np.eye(3), "b": np.zeros(3), "gyro_bias": np.zeros(3), "dp": np.zeros(3), "delay_s": 0.0}
    cmd7 = np.zeros((120, 7))
    cmd7[:, 0] = np.r_[np.zeros(30), 0.01 * np.arange(1, 71), np.full(20, 0.7)]   # j1：停 1.5 s → 匀速 0.2 rad/s → 停
    sent = S.shifted(cmd7, 0.06).astype(float)                                     # 提前 1.2 帧：匀速段每帧多 0.012 rad
    ts = 100.0 + S.RR.DT * np.arange(len(cmd7))
    tq, ti = np.arange(ts[0], ts[-1], 1 / S.D.FS_Q), np.arange(ts[0], ts[-1], 1 / S.D.FS_I)

    def ep(follow):        # 实际完全跟着 follow 走：编码器 = 零阶保持，陀螺 = 线性插值经运动学渲染；ep["cmd"] 是发出的指令
        g = ideal_imu_traj(model, mujoco.MjData(model), S.LinCmd(ts, follow[:, :6]), ti, sid)[1][:, 3:]
        return {"t_send": ts, "cmd": sent[:, :6], "tq": tq, "q": S.zoh(ts, follow[:, :6], tq),
                "ti": ti, "imu": np.c_[np.zeros((len(ti), 3)), g]}

    fq = (tq >= ts[30]) & (tq < ts[98])                                            # 关节：匀速段内，偏差处处 0.012 rad
    fi = (ti >= ts[20]) & (ti < ts[110])                                           # 末端：含起止拐点
    same = S.metrics(ep(cmd7), cmd7[:, :6], fq, fi, calib, model, sid)
    moved = S.metrics(ep(sent), cmd7[:, :6], fq, fi, calib, model, sid)
    assert same["joint_raw"] == pytest.approx(0.0, abs=1e-9) and same["ee_raw"] == pytest.approx(0.0, abs=1e-9)
    assert moved["joint_raw"] == pytest.approx(12.0 / np.sqrt(5), rel=1e-4)       # 0.012 rad 只在 j1：五个活动关节 RMS（mrad）
    assert moved["ee_raw"] > 0.01                                                  # 跟着发出的指令走，相对原始指令不为 0


def test_shift_actual_moves_actual_motion_earlier():
    def f(t):
        return np.sin(np.pi * t)

    tq, ti = np.arange(0, 4, 1 / S.D.FS_Q), np.arange(0, 4, 1 / S.D.FS_I)
    out = S.shift_actual({"tq": tq, "q": np.repeat(f(tq)[:, None], 6, 1), "ti": ti, "imu": np.repeat(f(ti)[:, None], 6, 1)}, 0.06)
    assert np.allclose(out["q"][tq < 3.9], f(tq[tq < 3.9] + 0.06)[:, None], atol=1e-4)     # q'(t) = q(t + 0.06)：提前，不是推迟
    assert np.allclose(out["imu"][ti < 3.9], f(ti[ti < 3.9] + 0.06)[:, None], atol=1e-4)
    assert out["tq"] is tq and out["ti"] is ti                                             # 时间轴不变


def test_correction_counts_only_active_joints():
    cmd7 = np.zeros((100, 7))
    sent = cmd7.copy()
    sent[:, 0] += 0.003                                                            # j1 每帧 3 mrad
    sent[50, 5] += 0.02                                                            # j6 一帧 20 mrad
    sent[:, 3] += 0.5                                                              # j4 不是活动关节；夹爪不算
    sent[:, 6] += 1.0
    c = S.correction(sent, cmd7)
    assert c["rms"] == pytest.approx(np.sqrt((100 * 3.0 ** 2 + 20.0 ** 2) / 500)) and c["max"] == pytest.approx(20.0)


def test_paired_by_sequence_and_round_and_rule_b():
    rows = {}
    for s in S.SEQS:
        for r in range(1, S.N_REP + 1):
            rows[(s, "base", r)] = {"joint_raw": 10.0}
            rows[(s, "d60", r)] = {"joint_raw": 9.0 if s == "A" else 8.0}           # A 降 10%、B 降 20%
    del rows[("B", "base", 6)]                                                     # 缺基线的那一轮不配对
    j = S.paired(rows, "d60", "joint_raw")
    assert j["n"] == 11 and j["median"] == pytest.approx(-10.0) and j["ci95"][1] < 0 and j["p"] < 0.01

    def x(med, hi):
        return {"median": med, "ci95": [med - 3, hi], "p": 0.01, "n": 12}

    down, flat = x(-10, -6), x(0, 2)
    assert S.verdict_b(down, flat, flat, flat) == "有效（关节偏差显著下降）"
    assert S.verdict_b(flat, down, flat, flat) == "有效（末端角速度偏差显著下降）"
    assert S.verdict_b(down, x(4, 6), flat, flat) == "无效"                        # 末端的上界超过 +5%：算变差
    assert S.verdict_b(x(-4, -1), flat, flat, flat) == "无效"                      # 降幅不到 5%
    assert S.verdict_b(down, flat, x(-6, -2), flat).startswith("护栏未过")          # 运动量少了 6%
    assert S.verdict_b(down, None, flat, flat).startswith("无法判定")


def test_run_one_exception_never_leaves_destructor_armed(monkeypatch, tmp_path):
    import json
    import types

    class Robot:
        is_connected = True

    calls = []
    U = types.SimpleNamespace(make_robot=lambda *a, **k: Robot())
    RP = types.SimpleNamespace(staged_return=lambda robot, torch, grip: calls.append(robot.is_connected) or True)
    for name, mod in (("lerobot.common.robot_devices.robots.utils", U), ("replay_probe", RP), ("torch", types.ModuleType("torch"))):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(S, "OUT", tmp_path)
    np.savez(tmp_path / "seq_A_d15.npz", cmd7=np.zeros((5, 7), np.float32))
    rec = tmp_path / "replay_A_d15_r1.json"

    def premove_fails(*a, **k):                    # 预移动 / 静置阶段：连接后抛出，没有回放记录
        made.append(U.make_robot())
        raise KeyboardInterrupt

    made = []
    RP.run = premove_fails
    assert S.run_one("A", "d15", "1") == 1
    assert calls == [True] and made[0].is_connected is False                       # 连着时分段归位一次，之后析构不再归位
    assert json.loads(rec.read_text())["stopped"] == "KeyboardInterrupt: "        # 落了 stopped 记录：不补跑、分析排除

    def recording_fails(*a, **k):                  # 录制阶段：管线已落盘（stopped 为空）并归位过，然后抛出
        made.append(U.make_robot())
        rec.write_text(json.dumps({"stopped": None, "t_send": [1.0]}))
        raise RuntimeError("x")

    made, calls[:] = [], []
    RP.run = recording_fails
    assert S.run_one("A", "d15", "1") == 1
    assert calls == [] and made[0].is_connected is False                           # 不再归位第二次
    assert json.loads(rec.read_text()) == {"stopped": "RuntimeError: x", "t_send": [1.0]}   # 只补原因，数据不动

    def blocked(robot, torch, grip):               # 归位中又按 Ctrl-C：仍然解除析构
        raise KeyboardInterrupt

    rec.unlink()
    made, RP.run, RP.staged_return = [], premove_fails, blocked
    assert S.run_one("A", "d15", "1") == 1 and made[0].is_connected is False
    assert json.loads(rec.read_text())["return_error"] == "KeyboardInterrupt: "
