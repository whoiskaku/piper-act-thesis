"""`ilc_route1.py` 的测试：更新公式（窗、限位掩码、Δ 的上下限、两支等价、符号与对齐、角速度 → 关节位置、雅可比坐标）、
接受 / 确认 / 退回 / 试探与复测、排程、完整轮次的配对与共同重抽、纯时间平移分量、序列文件（Δ = 0 复现 Shift、跳变检查、重建核对）、
规划器被控对象上的收敛。不连机械臂、不读数据集。"""
from __future__ import annotations

import gc

gc.disable()                      # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）

import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ilc_route1 as I  # noqa: E402

NONE5 = lambda n: np.zeros((n, 5), bool)  # noqa: E731


def met(jq, jw):
    return {"joint_raw": jq, "ee_raw": jw}


def box(n):
    return (np.ones((n, 5)), -I.CLIP, I.CLIP)


def test_zero_update_keeps_accepted_delta_and_frozen_frames_stay_zero():
    n = 120
    marked = np.zeros(n, bool)
    marked[60:70] = True                                       # 疑似接触段（所有关节）
    sat = NONE5(n)
    sat[90:95, 1] = True                                       # j2 的限位饱和帧（只这一个关节）
    W = I.window(I.frozen(marked, sat))
    assert W[:I.RR.HOLD].max() == 0 and W[60:70].max() == 0 and W[-1].max() == 0
    assert W[90:95, 1].max() == 0 and W[90:95, 0].min() == 1  # 饱和只冻结那个关节
    rng = np.random.default_rng(0)
    delta = np.zeros((n, 5))
    for _ in range(5):                                         # 从 0 出发连续几次随机更新
        delta = I.step(delta, rng.normal(0, 0.01, (n, 5)), 0.5, W)
    assert np.all(delta[W == 0] == 0) and np.abs(delta).max() <= I.CLIP   # 冻结的帧严格为 0
    assert np.array_equal(I.step(delta, np.zeros((n, 5)), 0.5, W), delta)  # 零更新：已接受的 Δ 一点不变（过渡段也不被削弱）


def test_limit_mask_union_by_send_frame():
    n, lo, hi = 150, np.full(6, -1.0), np.full(6, 1.0)
    a6, s6 = np.zeros((n, 6)), np.zeros((n, 6))
    a6[50, 1] = -1.1                                           # 原始帧越界（j2）
    s6[80, 2] = 1.1                                            # 只有实际发令的 Shift 帧越界（j3）
    a6[101, 5] = 1.2                                           # j6 原始第 101 帧越界：第 100 帧的前瞻误差取的是它
    sat = I.saturated(a6, s6, lo, hi)
    expect = NONE5(n)
    expect[[49, 50], 1] = True                                 # 第 49 帧前瞻到第 50 帧
    expect[80, 2] = True                                       # Shift 帧越界不往前瞻扩展
    expect[[100, 101], 4] = True
    assert np.array_equal(sat, expect)


def test_delta_bounds_keep_margin_to_hard_limit():
    n = 60
    s6 = np.zeros((n, 6))
    lo_h, hi_h = np.full(6, -0.02), np.full(6, 1.0)
    s6[:, 1] = -0.02 + I.BOUND_MARGIN + 0.004                  # j2 离硬界只有 0.2° + 4 mrad
    lo, hi = I.bounds(s6, lo_h, hi_h)
    W = np.ones((n, 5))
    delta = I.step(np.zeros((n, 5)), np.full((n, 5), -0.05), 1.0, W, lo, hi)    # 大幅往下推
    assert np.allclose(delta[:, 1], -0.004) and np.allclose(delta[:, 0], -I.CLIP)  # j2 停在离硬界 0.2° 处；其余到 ±15 mrad
    assert np.all(s6[:, I.T.ACTIVE] + delta >= lo_h[I.T.ACTIVE] + I.BOUND_MARGIN - 1e-12)


def test_hard_bound_matches_replay_pipeline():
    import mujoco
    import replay_probe as RP
    from ideal_imu_kinematic import MODEL_PATH
    lo = mujoco.MjModel.from_xml_path(str(MODEL_PATH)).jnt_range[1, 0] - I.HARD_SLACK   # j2 的硬界
    cmd7 = np.zeros((40, 7), np.float32)
    cmd7[:, 1] = lo + np.radians(0.1)
    RP.safety_checks(cmd7, np.zeros((40, 6), np.float32))                  # 硬界内 0.1°：通过
    cmd7[:, 1] = lo - np.radians(0.1)
    with pytest.raises(AssertionError):                                    # 硬界外 0.1°：管线拒绝
        RP.safety_checks(cmd7, np.zeros((40, 6), np.float32))


def test_branch_candidates_identical_when_imu_term_is_zero():
    n, rng = 100, np.random.default_rng(1)
    acc, eq = rng.normal(0, 0.002, (n, 5)), rng.normal(0, 0.003, (n, 5))
    j = I.Learner("j", 6, acc, (eq, np.zeros((n, 5))), met(10, 0.05), box(n), ["bp"])
    i = I.Learner("i", 6, acc, (eq, np.zeros((n, 5))), met(10, 0.05), box(n), ["bp"])
    assert np.array_equal(j.next()["delta"], i.next()["delta"])            # 同一状态、同一步长，IMU 项为零 → 同一候选
    i2 = I.Learner("i", 6, acc, (eq, rng.normal(0, 0.003, (n, 5))), met(10, 0.05), box(n), ["bp"])
    assert not np.array_equal(j.next()["delta"], i2.next()["delta"])
    np.testing.assert_array_equal(j.next()["delta"], I.step(acc, 0.5 * eq, I.GAMMA0, *box(n)))   # 分叉后关节系数 ½


def test_joint_error_sign_and_time_alignment():
    ts = 100.0 + I.RR.DT * np.arange(120)
    tq = np.arange(ts[0], ts[-1], 1 / I.D.FS_Q)
    q = np.zeros((len(tq), 6))
    q[:, 0] = -0.002 * np.exp(-0.5 * ((tq - ts[60]) / 0.1) ** 2)          # 实测在指令（0）下方，峰在第 60 帧的时刻
    e = I.joint_err({"t_send": ts, "tq": tq, "q": q}, np.zeros((120, 6)), np.zeros(120, bool))
    assert e[:, 0].max() > 0.0018 and e[:, 0].argmax() in (58, 59)        # 目标 − 实测 > 0；第 k 帧取 t_send[k] + 60 ms 的误差
    assert np.abs(e[:, 1:]).max() == 0


def test_angular_error_maps_to_joint_position_correction():
    rng = np.random.default_rng(2)
    U, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    V, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    J = U @ np.diag([1.2, 0.9, 0.6]) @ V[:3]                               # 3×5，满秩
    t = np.arange(0, 10, 0.01)
    eps = np.column_stack([0.02 * np.sin(2 * np.pi * 2 * t), 0.015 * np.sin(2 * np.pi * 3 * t + 1), 0.01 * np.sin(2 * np.pi * 1.5 * t + 2)])
    dq = I.ang_to_joint(t, eps, np.repeat(J[None], len(t), 0))            # rad
    rec = np.gradient(dq, t, axis=0) @ J.T                                 # 这份位置修改按刚体运动学产生的角速度
    m = (t > 2) & (t < 8)
    assert np.sqrt(np.mean((rec - eps)[m] ** 2)) / np.sqrt(np.mean(eps[m] ** 2)) < 0.02


def test_site_jacobian_matches_kinematic_gyro():
    import mujoco
    from ideal_imu_kinematic import MODEL_PATH, ideal_imu_traj
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site")
    q6, qd6 = np.array([0.2, 0.9, -0.7, 0.0, 0.5, 0.3]), np.array([0.3, -0.2, 0.4, 0.0, -0.5, 0.6])   # j4 不动（不在活动关节里）

    def sp(t, nu=0):
        return np.r_[q6 if nu == 0 else qd6 if nu == 1 else np.zeros(6), 0.0][None]

    gyro = ideal_imu_traj(model, mujoco.MjData(model), sp, np.array([0.0]), sid)[1][0, 3:]   # 安装点坐标的角速度
    J = I.jacobians(model, sid, q6[None])[0]
    np.testing.assert_allclose(J @ qd6[I.T.ACTIVE], gyro, atol=1e-9)


def test_guard_confirmation_rejection_and_budget():
    n = 100
    eq, ew = np.full((n, 5), 0.001), np.zeros((n, 5))
    L = I.Learner("c", 4, np.zeros((n, 5)), (eq, ew), met(10, 0.05), box(n), ["b"])
    c1 = L.next()
    assert L.add(c1, met(11, 0.05), (eq, ew)).startswith("触发护栏") and L.next() is c1
    assert L.add(c1, met(10.8, 0.05), (eq, ew)).startswith("退回")        # 两遍平均 10.9 > 10.5
    assert L.gamma == 0.25 and not L.acc.any()
    c2 = L.next()
    np.testing.assert_array_equal(c2["delta"], I.step(np.zeros((n, 5)), eq, 0.25, *box(n)))
    assert L.add(c2, met(9.0, 0.09), (eq, ew)).startswith("接受")         # 共同关节阶段不看末端护栏
    assert np.array_equal(L.acc, c2["delta"])
    c3 = L.next()
    assert L.add(c3, met(11, 0.05), (eq, ew)).startswith("退回") and L.left == 0   # 没有预算确认 → 直接退回


def test_imu_branch_switches_to_probing_and_remeasures_last():
    n = 100
    eq, ew = np.full((n, 5), 0.001), np.full((n, 5), 0.002)
    L = I.Learner("i", 5, np.zeros((n, 5)), (eq, ew), met(10, 0.05), box(n), ["bp"])
    for _ in range(2):
        msg = L.add(L.next(), met(10, 0.05), (eq, ew))                     # 接受但目标值没有下降
    assert L.probing and "改为试探" in msg
    acc, p = L.acc.copy(), L.next()
    assert p["label"] == "试探+" and L.add(p, met(10.2, 0.051), (eq, ew)).startswith("等")
    m = L.next()
    assert m["label"] == "试探−"
    np.testing.assert_array_equal(m["delta"], I.step(acc, 0.5 * eq - 0.5 * ew, L.gamma, *box(n)))
    assert L.add(m, met(9.8, 0.049), (eq, ew)) == "接受试探−" and np.array_equal(L.acc, m["delta"])
    last = L.next()                                                        # 只剩 1 遍：复测已接受的补正，不再更新
    assert L.left == 1 and last["label"] == "复测" and np.array_equal(last["delta"], L.acc)
    acc = L.acc.copy()
    assert L.add(last, met(9.0, 0.040), (eq, ew)).startswith("复测") and np.array_equal(L.acc, acc) and L.met["joint_raw"] == 9.8
    L2 = I.Learner("i", 6, np.zeros((n, 5)), (eq, ew), met(10, 0.05), box(n), ["bp"])
    for _ in range(2):
        L2.add(L2.next(), met(10, 0.05), (eq, ew))
    L2.add(L2.next(), met(10.1, 0.05), (eq, ew))
    assert L2.add(L2.next(), met(10.1, 0.05), (eq, ew)).endswith("c → 0.5")   # 两个试探都没改善


def test_schedule_budget_and_balance():
    plan = I.schedule()
    assert len(plan) == len(set(plan)) == 68
    cnt = Counter((s, lk) for s, lk, _ in plan)
    for s in I.SEQS:
        assert all(cnt[(s, lk)] == b for lk, b in I.BUDGET.items())
        ev = [(n, lk[2:]) for x, lk, n in plan if x == s and lk.startswith("e_")]
        assert all({c for r, c in ev if r == n} == set(I.EVAL) for n in range(1, I.N_EVAL + 1))
        pos = Counter((c, [cc for r, cc in ev if r == n].index(c)) for n, c in ev)
        assert set(pos.values()) == {1}                                    # 拉丁方：每个条件在每个位置各一次
        br = [lk for x, lk, _ in plan if x == s and lk in ("j", "i")]
        assert sum(br[2 * k] == "j" for k in range(I.BUDGET["j"])) == I.BUDGET["j"] // 2   # 两支各占一半先手


def test_joint_bootstrap_keeps_round_pairing_and_verdict():
    r = I.hier({"A": [-10.0] * 4, "B": [-20.0] * 4})
    assert r["effect"] == pytest.approx(-15) and r["ci95"] == pytest.approx([-15, -15])
    r2 = I.hier({"A": [-8.0, -2, -6, 0], "B": [8.0, 2, 6, 0]})            # 同轮完全反向：共同重抽下每次都正好抵消
    assert r2["ci95"] == pytest.approx([0, 0])

    def x(eff, hi):
        return {"effect": eff, "ci95": [eff - 3, hi]}

    flat = x(0, 2)
    assert I.verdict({"joint_raw": x(-8, -1), "ee_raw": flat, "path": flat, "speed": flat}) == "有改善（关节偏差）"
    assert I.verdict({"joint_raw": x(-8, -1), "ee_raw": x(3, 6), "path": flat, "speed": flat}) == "没有改善"
    assert I.verdict({"joint_raw": x(-8, -1), "ee_raw": flat, "path": x(-6, -2), "speed": flat}).startswith("护栏未过")


def test_compare_judges_only_with_all_rounds():
    m = {"joint_raw": 10.0, "ee_raw": 0.05, "path": 1.0, "speed": 0.2}
    better = dict(m, joint_raw=9.0)
    e = {s: {(n, c): (better if c == "j" else m) for n in range(1, 5) for c in ("shift", "j")} for s in ("A", "B")}
    r = I.compare(e, "j", "shift")
    assert r["rounds"] == [1, 2, 3, 4] and r["verdict"] == "有改善（关节偏差）" and r["joint_raw"]["effect"] == pytest.approx(-10)
    del e["B"][(3, "j")]                                                   # B 第 3 轮缺一遍
    r = I.compare(e, "j", "shift")
    assert r["rounds"] == [1, 2, 4] and r["verdict"] == "只报告、未完成（完整轮次 3/4）"


def test_time_shift_component():
    t = I.RR.DT * np.arange(200)
    a6 = np.column_stack([0.3 * np.sin(2 * np.pi * f * t) for f in (0.3, 0.5, 0.2, 0.0, 0.4, 0.6)])
    W = I.window(I.frozen(np.zeros(200, bool), NONE5(200)))
    delta = 0.010 * np.gradient(a6, I.RR.DT, axis=0)[:, I.T.ACTIVE]       # 整体提前 10 ms 的一阶近似
    assert I.time_shift_ms(delta, a6, W) == pytest.approx(10.0)


def test_sequence_file_shift_reproduction_jump_check_and_rebuild_guard(monkeypatch, tmp_path):
    monkeypatch.setattr(I, "OUT", tmp_path)
    t = I.RR.DT * np.arange(50)
    s7 = np.column_stack([0.2 * np.sin(2 * np.pi * 0.5 * t + j) for j in range(7)]).astype(np.float32)
    z = np.load(I.write_seq("A_b01", s7, np.zeros((50, 5)), "Shift 基准"))
    assert np.array_equal(z["cmd7"], s7)                                    # Δ = 0 逐位复现 Shift
    assert json.loads(str(z["source"]))["cmd_hash"] == hashlib.sha256(s7.tobytes()).hexdigest()[:16]
    jump = np.zeros((50, 5))
    jump[25:, 0] = 0.015                                                   # 一帧里跳 15 mrad
    with pytest.raises(SystemExit):
        I.write_seq("A_c01", np.zeros((50, 7), np.float32), jump, "候选")   # Shift 静止时门槛只有 6 mrad
    (tmp_path / "replay_A_b01_r1.json").write_text("{}")                   # 回放过之后，重建出不同的序列就停
    with pytest.raises(SystemExit):
        I.write_seq("A_b01", s7, np.full((50, 5), 0.001), "Shift 基准")


def test_joint_ilc_converges_on_planner_plant():
    n = 160
    ts = 100.0 + I.RR.DT * np.arange(n)
    tq = np.arange(ts[0], ts[-1], 1 / I.D.FS_Q)
    a7 = np.zeros((n, 7))
    a7[20:, :6] = 0.25 * np.sin(2 * np.pi * 0.4 * (ts[20:, None] - ts[20])) * np.array([1, 0.6, -0.5, 0, 0.4, 0.3])
    s7 = I.S.shifted(a7, 0.06).astype(float)
    none = np.zeros(n, bool)
    W = I.window(I.frozen(none, NONE5(n)))

    def plant(delta):                                                      # 被控对象：加速度受限规划 + 30 ms 纯延迟（Shift 补不全）
        u = s7[:, :6].copy()
        u[:, I.T.ACTIVE] += delta
        return {"t_send": ts, "tq": tq, "q": I.T.PlanTrajectory(u, ts, q0=a7[0, :6])(tq - 0.03)[:, :6]}

    def low(ep):                                                           # <5 Hz 关节误差 RMS（保持段之后、末尾留白）
        e = I.D.lp((I.S.zoh(ts, a7[:, :6], tq) - ep["q"])[:, I.T.ACTIVE], 5, I.D.FS_Q)
        m = (tq >= ts[20]) & (tq <= ts[-1] - 0.3)
        return np.sqrt(np.mean(e[m] ** 2))

    delta, J = np.zeros((n, 5)), []
    for _ in range(5):
        ep = plant(delta)
        J.append(low(ep))
        delta = I.step(delta, I.joint_err(ep, a7[:, :6], none), 0.5, W)
    assert all(b < a for a, b in zip(J, J[1:])) and J[-1] < 0.5 * J[0], J
