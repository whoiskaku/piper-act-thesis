"""`cmd_tracking.py` 的测试：规划器与参考实现逐位相同；各指标在合成数据上还原已知量。不连机械臂、不读数据集。"""
from __future__ import annotations

import gc

gc.disable()                      # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "step2_g")]
import cmd_tracking as T  # noqa: E402
from optuna_ident_enc import accel_limited_dt  # noqa: E402


def test_plan_substeps_frame_ends_equal_reference_bitwise():
    rng = np.random.default_rng(0)
    cmd = np.cumsum(rng.normal(0, 2.0, 300))             # 度
    dts = 0.05 + rng.uniform(-0.01, 0.02, 300)           # 不等的帧长，同真机 t_send
    q, v, a = T.plan_substeps(cmd, dts, 286.0)
    assert np.array_equal(q[T.SUB - 1::T.SUB], accel_limited_dt(cmd, 286.0, dts))
    assert np.abs(a).max() <= 286.0


def test_plan_trajectory_positions_velocities_accelerations_agree():
    t_send = 0.05 * np.arange(60)
    cmd = np.zeros((60, 6))
    cmd[10:, 1] = 0.3                                     # j2 在第 10 帧给一个 0.3 rad 的阶跃
    traj = T.PlanTrajectory(cmd, t_send)
    t = np.linspace(0.6, 2.5, 400)
    q, v, a = traj(t)[:, 1], traj(t, 1)[:, 1], traj(t, 2)[:, 1]
    assert traj(t).shape == (400, 7)
    assert np.allclose(np.gradient(q, t), v, atol=0.02)   # 速度 ≈ 位置的导数
    assert np.abs(a).max() == pytest.approx(T.A_MAX)      # 加速段正好是参考上限
    assert q[-1] == pytest.approx(0.3, abs=1e-9)          # 最后停在目标上


def test_plan_starts_from_the_given_initial_position():
    cmd = np.full((60, 6), 0.2)                            # 指令一直是 0.2 rad，机械臂实际从 0 出发
    traj = T.PlanTrajectory(cmd, 0.05 * np.arange(60), q0=np.zeros(6))
    q = traj(np.array([0.0, 2.9]))[:, 0]
    assert q[0] == pytest.approx(0.0, abs=1e-3) and q[1] == pytest.approx(0.2, abs=1e-9)


def test_best_delay_recovers_a_known_shift():
    t = np.arange(0, 20, 0.005)

    def f(x):
        return np.sin(2 * np.pi * 0.4 * x) + 0.3 * np.sin(2 * np.pi * 1.3 * x)

    tau, e0, e1 = T.best_delay(t, f(t - 0.06), f)
    assert tau == pytest.approx(0.06, abs=0.003) and e1 < 0.01 * e0


def test_band_rms_recovers_a_known_high_frequency_sine():
    t = np.arange(0, 30, 0.01)
    x = 0.2 * np.sin(2 * np.pi * 8 * t) + 1.0 * np.sin(2 * np.pi * 0.5 * t)   # 8 Hz 在带内，0.5 Hz 在带外
    assert T.rms(T.band(x, 100.0)[300:-300]) == pytest.approx(0.2 / np.sqrt(2), rel=0.03)


def test_speed_gain_and_overshoot_recover_known_values():
    t = np.arange(0, 10, 0.005)
    v_ideal = np.sin(2 * np.pi * 0.3 * t)
    assert T.speed_gain(0.8 * v_ideal, v_ideal) == pytest.approx(0.8)
    q_ideal = np.cumsum(v_ideal) * 0.005
    t_rev = 1 / 0.6                                        # 第一次换向（速度由正变负）
    q_act = q_ideal + 0.01 * np.exp(-((t - t_rev) / 0.05) ** 2)
    assert T.overshoot(t, q_act, q_ideal, v_ideal)[0] == pytest.approx(0.01, abs=1e-3)


def test_overshoot_only_at_turning_points_not_stop_and_go():
    # 慢速上行 2 s（每帧 0.003 rad：参考每帧降到零再走，峰速约 0.12 rad/s）再掉头下行；实际 = 指令的线性插值（匀速，不停）。
    # 旧口径（理想速度变号或降到零）会在每个走停处算一次「过冲」，值约为实际 0.3 s 内继续前进的距离
    cmd = np.r_[np.arange(41) * 0.003, 0.12 - np.arange(1, 41) * 0.003]
    t_send = 0.05 * np.arange(len(cmd))
    traj = T.PlanTrajectory(np.column_stack([cmd, np.zeros((len(cmd), 5))]), t_send)
    t = np.arange(0.0, t_send[-1], 0.005)
    ov = T.overshoot(t, np.interp(t, t_send, cmd), traj(t)[:, 0], traj(t, 1)[:, 0])
    assert len(ov) == 1 and abs(ov[0]) < 0.002                # 只有掉头那一次，且实际没有越过转折点


def test_overshoot_is_zero_when_actual_follows_reference_through_close_turns():
    # 上行 → 回落约 0.7° → 再上行超过旧极值，两次转折相隔不到 0.3 s；实际与参考完全相同，过冲必须为 0
    cmd = np.r_[np.arange(11) * 0.01, 0.094, 0.088, 0.088 + np.arange(1, 9) * 0.01, np.full(10, 0.168)]
    t_send = 0.05 * np.arange(len(cmd))
    traj = T.PlanTrajectory(np.column_stack([cmd, np.zeros((len(cmd), 5))]), t_send)
    t = np.arange(0.0, t_send[-1], 0.005)
    q = traj(t)[:, 0]
    k = T.turning_points(q, np.deg2rad(T.TURN_DEG))[0][0]
    assert q[(t > t[k]) & (t <= t[k] + 0.3)].max() > q[k]      # 固定 0.3 s 窗口里，参考自己就超过了旧极值
    ov = T.overshoot(t, q, q, traj(t, 1)[:, 0])
    assert len(ov) >= 1 and np.allclose(ov, 0.0)


def test_boundary_segments_only_without_te():
    t = np.arange(0, 25, 0.005)
    bnd = 0.05 * np.arange(100, 500, 100)                 # 第 100、200、300、400 帧
    m = T.near(t, bnd, T.BOUNDARY_S)
    assert m.sum() == pytest.approx(len(bnd) * 2 * T.BOUNDARY_S / 0.005, abs=2 * len(bnd))
    assert not T.near(t, np.array([]), T.BOUNDARY_S).any()  # 开 TE：没有边界段


def test_ideal_imu_static_pose_reads_gravity_plus_bias():
    import mujoco
    from ideal_imu_kinematic import MODEL_PATH
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site")
    cmd = np.tile(np.deg2rad([-4.9, 51.5, -25.2, 0.0, 9.4, 4.0]), (40, 1))
    traj = T.PlanTrajectory(cmd, 0.05 * np.arange(40))
    calib = {"R": np.eye(3), "b": [0.1, -0.2, 1.0], "gyro_bias": [0.01, 0.0, -0.02], "dp": [0.0, 0.0, 0.0]}
    acc, gyr = T.ideal_imu(traj, np.linspace(0.2, 1.8, 50), calib, model, sid)
    assert np.allclose(gyr, [0.01, 0.0, -0.02], atol=1e-9)
    assert np.allclose(np.linalg.norm(acc - np.array([0.1, -0.2, 1.0]), axis=1), 9.81, atol=1e-3)
