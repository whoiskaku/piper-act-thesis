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


def test_v06_branch_candidates_identical_when_imu_term_is_zero():
    n, rng = 100, np.random.default_rng(1)
    acc, eq = rng.normal(0, 0.002, (n, 5)), rng.normal(0, 0.003, (n, 5))
    j = I.Learner("j", 6, acc, (eq, np.zeros((n, 5)), np.zeros((n, 5))), met(10, 0.05), box(n), ["bp"])
    i = I.Learner("i", 6, acc, (eq, np.zeros((n, 5)), np.zeros((n, 5))), met(10, 0.05), box(n), ["bp"])
    assert np.array_equal(j.next()["delta"], i.next()["delta"])            # 同一状态、同一步长，IMU 项为零 → 同一候选
    i2 = I.Learner("i", 6, acc, (eq, rng.normal(0, 0.003, (n, 5)), np.zeros((n, 5))), met(10, 0.05), box(n), ["bp"])
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
    L = I.Learner("c", 4, np.zeros((n, 5)), (eq, ew, 0 * ew), met(10, 0.05), box(n), ["b"])
    c1 = L.next()
    assert L.add(c1, met(11, 0.05), (eq, ew, 0 * ew)).startswith("触发护栏") and L.next() is c1
    assert L.add(c1, met(10.8, 0.05), (eq, ew, 0 * ew)).startswith("退回")        # 两遍平均 10.9 > 10.5
    assert L.gamma == 0.25 and not L.acc.any()
    c2 = L.next()
    np.testing.assert_array_equal(c2["delta"], I.step(np.zeros((n, 5)), eq, 0.25, *box(n)))
    assert L.add(c2, met(9.0, 0.09), (eq, ew, 0 * ew)).startswith("接受")         # 共同关节阶段不看末端护栏
    assert np.array_equal(L.acc, c2["delta"])
    c3 = L.next()
    assert L.add(c3, met(11, 0.05), (eq, ew, 0 * ew)).startswith("退回") and L.left == 0   # 没有预算确认 → 直接退回


def test_imu_branch_switches_to_probing_and_remeasures_last():
    n = 100
    eq, ew = np.full((n, 5), 0.001), np.full((n, 5), 0.002)
    L = I.Learner("i", 5, np.zeros((n, 5)), (eq, ew, 0 * ew), met(10, 0.05), box(n), ["bp"])
    for _ in range(2):
        msg = L.add(L.next(), met(10, 0.05), (eq, ew, 0 * ew))                     # 接受但目标值没有下降
    assert L.probing and "改为试探" in msg
    acc, p = L.acc.copy(), L.next()
    assert p["label"] == "试探+" and L.add(p, met(10.2, 0.051), (eq, ew, 0 * ew)).startswith("等")
    m = L.next()
    assert m["label"] == "试探−"
    np.testing.assert_array_equal(m["delta"], I.step(acc, 0.5 * eq - 0.5 * ew, L.gamma, *box(n)))
    assert L.add(m, met(9.8, 0.049), (eq, ew, 0 * ew)) == "接受试探−" and np.array_equal(L.acc, m["delta"])
    last = L.next()                                                        # 只剩 1 遍：复测已接受的补正，不再更新
    assert L.left == 1 and last["label"] == "复测" and np.array_equal(last["delta"], L.acc)
    acc = L.acc.copy()
    assert L.add(last, met(9.0, 0.040), (eq, ew, 0 * ew)).startswith("复测") and np.array_equal(L.acc, acc) and L.met["joint_raw"] == 9.8
    L2 = I.Learner("i", 6, np.zeros((n, 5)), (eq, ew, 0 * ew), met(10, 0.05), box(n), ["bp"])
    for _ in range(2):
        L2.add(L2.next(), met(10, 0.05), (eq, ew, 0 * ew))
    L2.add(L2.next(), met(10.1, 0.05), (eq, ew, 0 * ew))
    assert L2.add(L2.next(), met(10.1, 0.05), (eq, ew, 0 * ew)).endswith("c → 0.5")   # 两个试探都没改善


def test_schedule_budget_and_balance():
    plan = I.schedule()
    assert len(plan) == len(set(plan)) == 68
    cnt = Counter((s, lk) for s, lk, _ in plan)
    for s in I.SEQS:
        assert all(cnt[(s, lk)] == b for lk, b in I.BUDGET.items() if lk != "h")   # h 支只在 v0.10 的三支排程里
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


# ── v0.7：只关节支也带编码器预测的末端项（方案 v0.7 §2.4）──────────────────────
def rigid_case():
    """已知刚体运动的合成回放：解析关节轨迹 → 刚体运动学 → 按标定约定（R·ω_site(t − d) + 零偏）生成「实测」陀螺。
    标定故意取非单位旋转、非零零偏和 12 ms 延迟，IMU 时刻与编码器网格错开 3 ms。"""
    import mujoco
    from ideal_imu_kinematic import MODEL_PATH, ideal_imu_traj
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site")
    c, A, f = np.array([0, 0.8, -0.8, 0, 0.4, 0]), np.array([0.3, 0.2, 0.2, 0, 0.2, 0.3]), np.array([0.5, 0.8, 1.1, 0, 1.3, 0.7])

    def sp(t, nu=0):
        t = np.atleast_1d(t)[:, None]
        w = 2 * np.pi * f
        y = c + A * np.sin(w * t) if nu == 0 else A * w * np.cos(w * t) if nu == 1 else -A * w ** 2 * np.sin(w * t)
        return np.column_stack([y, np.zeros(len(t))])

    ax = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(0.5) * K + (1 - np.cos(0.5)) * K @ K
    calib = {"R": R, "b": np.zeros(3), "gyro_bias": np.array([0.01, -0.02, 0.005]), "dp": np.zeros(3), "delay_s": 0.012}
    ts = I.RR.DT * np.arange(120)
    tq, ti = np.arange(ts[0], ts[-1], 1 / I.D.FS_Q), np.arange(ts[0] + 0.003, ts[-1], 1 / I.D.FS_I)
    g = ideal_imu_traj(model, mujoco.MjData(model), sp, ti - calib["delay_s"], sid)[1][:, 3:] @ R.T + calib["gyro_bias"]
    ep = {"t_send": ts, "tq": tq, "q": sp(tq)[:, :6], "ti": ti, "imu": np.c_[np.zeros((len(ti), 3)), g]}
    return ep, sp(ts)[:, :6], calib, model, sid


def test_v07_branches_differ_only_by_imu_versus_encoder_prediction():
    n, rng = 100, np.random.default_rng(4)
    acc, eq, ee = (rng.normal(0, s, (n, 5)) for s in (0.001, 0.002, 0.002))   # ee = M[ω* − ω_enc]
    W = np.ones((n, 5))

    def mk(kind, ew):                                                      # 限幅设宽：只检验限幅与接受决策之前的更新量
        return I.Learner(kind, 6, acc, (eq, ew, ee), met(10, 0.05), (W, -1.0, 1.0), ["bp"], enc=True)

    assert np.array_equal(mk("j", ee).next()["delta"], mk("i", ee).next()["delta"])   # 实测 IMU = 编码器预测 → 两支候选相同
    ew = rng.normal(0, 0.002, (n, 5))                                      # ew = M[ω* − ω_meas]
    diff = mk("i", ew).next()["delta"] - mk("j", ew).next()["delta"]
    np.testing.assert_allclose(diff, W * I.qfilter(I.GAMMA0 * 0.5 * (ew - ee)), atol=1e-12)   # = Q[γ·½·M[ω_enc − ω_meas]]


def test_v07_imu_map_is_linear_on_the_same_data():
    ep, a6, calib, model, sid = rigid_case()
    rng = np.random.default_rng(5)
    x, y = rng.normal(0, 0.05, (2, len(ep["ti"]), 3))
    marked = np.zeros(len(a6), bool)
    marked[50:60] = True
    M = lambda e: I.ang_err(ep, e, a6, marked, calib, model, sid)          # noqa: E731
    np.testing.assert_allclose(M(x) - M(y), M(x - y), atol=1e-12)          # 差分恒等式不依赖刚体投影近似


def test_v07_rigid_motion_gives_no_spurious_residual():
    ep, a6, calib, model, sid = rigid_case()
    g, ge = ep["imu"][:, 3:], I.enc_gyro(ep, calib, model, sid)
    m = (ep["ti"] > ep["ti"][0] + 0.5) & (ep["ti"] < ep["ti"][-1] - 0.5)  # 避开滤波端点

    def rms(x):
        return np.sqrt(np.mean(np.sum(x[m] ** 2, 1)))

    def band(x):
        return I.D.lp(x, 5, I.D.FS_I) - I.D.lp(x, 0.5, I.D.FS_I)

    assert rms(ge - g) < 0.01 * rms(g - calib["gyro_bias"])               # 坐标、零偏、时延、IMU 时刻都对上
    assert rms(band(ge - g)) < 0.01 * rms(band(g))                         # 更新用的 0.5–5 Hz 带内也没有凭空的残差
    wrong = dict(calib, delay_s=0.0)                                       # 对照：时延处理错了就会出现残差
    assert rms(I.enc_gyro(ep, wrong, model, sid) - g) > 0.05 * rms(g - calib["gyro_bias"])


def test_v07_encoder_branch_ignores_imu_readings():
    ep, a6, calib, model, sid = rigid_case()
    noisy = dict(ep, imu=np.random.default_rng(6).normal(0, 1, ep["imu"].shape))   # 只改 IMU 读数（时间戳不变）
    assert np.array_equal(I.enc_gyro(ep, calib, model, sid), I.enc_gyro(noisy, calib, model, sid))
    marked = np.zeros(len(a6), bool)
    assert np.array_equal(I.joint_err(ep, a6, marked), I.joint_err(noisy, a6, marked))
    n, rng = 100, np.random.default_rng(7)
    err = tuple(rng.normal(0, 0.002, (n, 5)) for _ in range(3))
    runs = []
    for jw in (0.04, 0.09):                                                # 只关节支的接受与步长不看末端指标
        L = I.Learner("j", 6, np.zeros((n, 5)), err, met(10, 0.05), box(n), ["bp"], enc=True)
        msgs = [L.add(L.next(), met(jq, jw), err) for jq in (10.4, 11.0, 10.9, 10.2)]
        runs.append((msgs, L.gamma, L.acc.copy()))
    assert runs[0][0] == runs[1][0] and runs[0][1] == runs[1][1] and np.array_equal(runs[0][2], runs[1][2])



def test_calibration_found_by_session_hash(monkeypatch, tmp_path):
    cal, out = tmp_path / "imu_calib", tmp_path / "session"
    cal.mkdir(), out.mkdir()
    (cal / "imu_calib.json").write_text('{"v": "now"}')
    (cal / "imu_calib_20261019.json").write_text('{"v": "old"}')
    monkeypatch.setattr(I, "CALIB", cal / "imu_calib.json")
    monkeypatch.setattr(I, "OUT", out)
    assert I.calib_path() == cal / "imu_calib.json"                        # 新会话：用当前标定
    old = hashlib.sha256(b'{"v": "old"}').hexdigest()[:16]
    (out / "session.json").write_text(json.dumps({"calib_sha": old}))
    assert I.calib_path() == cal / "imu_calib_20261019.json"               # 旧会话：按哈希找到备份
    (out / "session.json").write_text(json.dumps({"calib_sha": "0" * 16}))
    with pytest.raises(SystemExit):                                       # 找不到就停
        I.calib_path()


# ── v0.8：窗过渡 0.5 s + 同会话对照（方案 v0.8）─────────────────────────────
def test_v08_rule_params_window_and_schedule():
    assert I.rule_params("v0.6") == (False, 5, I.EVAL, 0.5)                # 旧会话照原样重建
    assert I.rule_params("v0.7") == (True, 5, I.EVAL, 0.5)
    assert I.rule_params("v0.8") == (True, 10, I.EVAL + ("j07",), 0.5)
    zero = np.zeros((60, 5), bool)
    zero[:20] = True
    W = I.window(zero, 10)
    assert W[:20].max() == 0 and 0 < W[25, 0] < 1 and W[28, 0] < 1 and W[29:].min() == 1   # 距离 10 帧（0.5 s）才到 1
    assert I.window(zero, 5)[23, 0] < 1 and I.window(zero, 5)[24:].min() == 1           # 旧窗：距离 5 帧（0.25 s）
    evals = I.EVAL + ("j07",)
    plan = I.schedule(evals)
    assert len(plan) == len(set(plan)) == 76
    for s in I.SEQS:
        ev = [(n, lk[2:]) for x, lk, n in plan if x == s and lk.startswith("e_")]
        assert all({c for r, c in ev if r == n} == set(evals) for n in range(1, I.N_EVAL + 1))   # 每轮每个条件一次
        pos = Counter((c, [cc for r, cc in ev if r == n].index(c)) for n, c in ev)
        assert set(pos.values()) == {1}                                    # 每个条件 4 轮占 4 个不同位置


def test_v08_comparator_command_replayed_bytewise(monkeypatch, tmp_path):
    src, out = tmp_path / "v07", tmp_path / "v08"
    src.mkdir(), out.mkdir()
    rng = np.random.default_rng(8)
    s7 = np.cumsum(rng.normal(0, 0.002, (50, 7)), 0).astype(np.float32)
    old = s7.copy()
    old[:, I.T.ACTIVE] += rng.normal(0, 0.001, (50, 5)).astype(np.float32)   # v0.7 学出的补正（float32 写在指令里）
    h = hashlib.sha256(old.tobytes()).hexdigest()[:16]
    for r in range(1, I.N_EVAL + 1):
        np.savez(src / f"seq_A_e_j{r:02d}.npz", cmd7=old, source=json.dumps({"cmd_hash": h}))
    monkeypatch.setattr(I, "COMPARATOR", src)
    monkeypatch.setattr(I, "OUT", out)
    cmd, hh = I.comparator("A")
    assert hh == h
    z = np.load(I.write_seq("A_e_j0701", s7, None, "评价 j07", cmd7=cmd))
    assert np.array_equal(z["cmd7"], old) and json.loads(str(z["source"]))["cmd_hash"] == h   # 逐字节原样重放
    for bad in ({"cmd7": old, "source": json.dumps({"cmd_hash": "x"})},                  # 记下的哈希不一致 → 停
                {"cmd7": s7, "source": json.dumps({"cmd_hash": h})}):                    # 哈希照旧、内容被改 → 停
        np.savez(src / "seq_A_e_j03.npz", **bad)
        with pytest.raises(SystemExit):
            I.comparator("A")


def test_v08_band_metrics_recover_known_sines():
    ts = I.RR.DT * np.arange(200)
    tq, ti = np.arange(ts[0], ts[-1], 1 / I.D.FS_Q), np.arange(ts[0] + 0.003, ts[-1], 1 / I.D.FS_I)
    a6 = np.zeros((200, 6))
    q = np.zeros((len(tq), 6))
    q[:, 0] = -0.001 * np.sin(2 * np.pi * 3 * tq)                          # 关节误差：3 Hz、1 mrad，只在 j1
    g = np.zeros((len(ti), 3))
    g[:, 0], g[:, 1] = 0.01 * np.sin(2 * np.pi * 10 * ti), 0.02 * np.sin(2 * np.pi * 2 * ti)   # 末端：10 Hz 与 2 Hz
    ep = {"t_send": ts, "tq": tq, "q": q, "ti": ti, "imu": np.c_[np.zeros((len(ti), 3)), g]}
    fq, fi = (tq > 1) & (tq < 9), (ti > 1) & (ti < 9)
    zone = np.zeros(200, bool)
    zone[60:100] = True
    m = I.band_metrics(ep, a6, np.zeros((len(ti), 3)), fq, fi, zone, {"delay_s": 0.0})
    assert m["ee_5_15"] == pytest.approx(0.01 / np.sqrt(2), rel=0.03) and m["ee_05_5"] == pytest.approx(0.02 / np.sqrt(2), rel=0.03)
    assert m["ee_5_15_zone"] == pytest.approx(0.01 / np.sqrt(2), rel=0.05)
    assert m["joint_05_5"] == pytest.approx(1.0 / np.sqrt(2) / np.sqrt(5), rel=0.03) and m["joint_5_10"] < 0.02


# ── v0.9：加 IMU 支里 IMU 独有部分的权重 ½ → 1（方案 §11）────────────────────
def test_v09_rule_params_and_three_condition_schedule():
    assert I.rule_params("v0.9") == (True, 10, ("shift", "j", "i"), 1.0)
    plan = I.schedule(("shift", "j", "i"))
    assert len(plan) == len(set(plan)) == 60                               # 4 + 8 + 24 + 3 条件 × 4 轮 × 2 序列
    for s in I.SEQS:
        ev = [(n, lk[2:]) for x, lk, n in plan if x == s and lk.startswith("e_")]
        assert all({c for r, c in ev if r == n} == {"shift", "j", "i"} for n in range(1, I.N_EVAL + 1))


def test_v09_weights_only_the_imu_only_part_and_old_weight_is_bytewise():
    n, rng = 100, np.random.default_rng(9)
    acc, eq, ee, ew = (rng.normal(0, s, (n, 5)) for s in (0.001, 0.002, 0.002, 0.002))
    W = np.ones((n, 5))

    def cand(kind, wr, sign=1):                                            # 限幅设宽：只检验限幅与接受决策之前的更新量
        return I.Learner(kind, 6, acc, (eq, ew, ee), met(10, 0.05), (W, -1.0, 1.0), ["bp"], enc=True, wr=wr).cand(sign, "x")["delta"]

    for sign in (1, -1):                                                   # wr = ½：候选与试探都逐位等于 v0.6–v0.8 的旧式
        assert np.array_equal(cand("i", 0.5, sign), I.step(acc, 0.5 * eq + sign * 0.5 * ew, I.GAMMA0, W, -1.0, 1.0))
    np.testing.assert_allclose(cand("i", 1.0) - cand("j", 1.0), W * I.qfilter(I.GAMMA0 * 1.0 * (ew - ee)), atol=1e-12)   # 两支只差 1·M[ω_enc − ω_meas]
    assert np.array_equal(cand("j", 1.0), cand("j", 0.5))                 # 只关节支不受这个权重影响


# ── v0.10：两条开 CUDA graph 后的评测指令，三支 0 / ½ / 1（方案 §12）───────────────────
def test_v010_rule_params_three_branch_schedule_and_old_schedule_unchanged():
    assert I.rule_params("v0.10") == (True, 10, ("shift", "j", "h", "i"), 1.0)
    evals = ("shift", "j", "h", "i")
    plan = I.schedule(evals)
    assert len(plan) == len(set(plan)) == 80                               # 4 + 8 + 3 支 × 6 × 2 + 4 条件 × 4 轮 × 2
    for s in I.SEQS:
        br = [(lk, n) for x, lk, n in plan if x == s and lk in ("j", "h", "i")]
        assert Counter(lk for lk, _ in br) == {"j": 6, "h": 6, "i": 6}
        pos = Counter((lk, [l for l, m in br if m == n].index(lk)) for lk, n in br)
        assert set(pos.values()) == {2}                                    # 6 遍里每支在每个先后位置各 2 次
        ev = [(n, lk[2:]) for x, lk, n in plan if x == s and lk.startswith("e_")]
        assert all({c for r, c in ev if r == n} == set(evals) for n in range(1, I.N_EVAL + 1))
    old = [(s, "b", n) for n in (1, 2) for s in I.SEQS] + [(s, "c", n) for n in range(1, 5) for s in I.SEQS]
    for n in range(1, 7):
        f, g = ("j", "i") if n % 2 else ("i", "j")
        old += [("A", f, n), ("B", g, n), ("A", g, n), ("B", f, n)]
    assert I.schedule(I.EVAL)[:len(old)] == old and I.schedule(("shift", "j", "i"))[:len(old)] == old   # 两支的老排程不变


def test_v010_half_branch_equals_old_imu_branch_and_follows_imu_rules():
    n, rng = 100, np.random.default_rng(10)
    acc, eq, ee, ew = (rng.normal(0, s, (n, 5)) for s in (0.001, 0.002, 0.002, 0.002))
    W = np.ones((n, 5))
    bx = (W, -1.0, 1.0)
    st = {"c": I.Learner("c", 4, acc, (eq, ew, ee), met(10, 0.05), bx, ["c4"], True, 1.0)}
    h, i, j = (I.learner(st, lk, bx, True, 1.0) for lk in ("h", "i", "j"))
    assert (h.wr, i.wr) == (0.5, 1.0)                                       # h 固定 ½，i 取规则版本给的权重
    for sign in (1, -1):                                                   # h = v0.7 / v0.8 的加 IMU 支（逐位）
        assert np.array_equal(h.cand(sign, "x")["delta"], I.step(acc, 0.5 * eq + sign * 0.5 * ew, I.GAMMA0, W, -1.0, 1.0))
    assert not h.ok(met(10, 0.05 * 1.06)) and j.ok(met(10, 0.05 * 1.06))   # 末端护栏对 h 生效，只关节支不看末端
    for _ in range(2):                                                     # 两次「接受但目标值没有下降」→ h 也转入试探
        c = h.next()
        c["rids"].append("x")
        h.add(c, met(10, 0.05), (eq, ew, ee))
    assert h.probing


def test_v010_sequences_are_episode_command_and_60ms_shift(monkeypatch, tmp_path):
    cmd7 = np.cumsum(np.random.default_rng(11).normal(0, 0.002, (60, 7)), 0).astype(np.float32)
    monkeypatch.setattr(I.RR, "episode_cmd", lambda repo, i: cmd7 + np.float32(i * 1e-3))
    monkeypatch.setattr(I, "SR_V10", tmp_path)
    assert I.cmd_seq() == 0
    for s, (repo, i) in I.SRC_V10.items():
        src = (cmd7 + np.float32(i * 1e-3)).astype(float)
        b, d = (np.load(tmp_path / f"seq_{s}_{c}.npz")["cmd7"] for c in ("base", "d60"))
        assert np.array_equal(b, src.astype(np.float32)) and np.array_equal(d, I.S.shifted(src, 0.060))   # 基线 = 原指令；提前 60 ms
    assert I.cmd_seq() == 0                                                # 再跑：内容相同就跳过
    monkeypatch.setattr(I.RR, "episode_cmd", lambda repo, i: cmd7 + np.float32(1.0))
    with pytest.raises(SystemExit):
        I.cmd_seq()                                                        # 内容变了：不覆盖
