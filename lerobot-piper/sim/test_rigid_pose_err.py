"""编码器推算的刚体末端位姿误差（补充分析，sim/rigid_pose_err.py）的测试。"""
import gc
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
gc.disable()
import rigid_pose_err as P  # noqa: E402


def rot(u, t):
    """绕单位轴 u 转 t（Rodrigues）。"""
    u = np.asarray(u, float) / np.linalg.norm(u)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


@pytest.fixture(scope="module")
def mj():
    import mujoco
    from ideal_imu_kinematic import MODEL_PATH
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    return model, mujoco.MjData(model), mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site")


def test_pose_err_exact_on_known_rotations():
    th = np.array([0.0, 1e-5, 1e-3, 0.1, 1.0, 3.0])                      # 含很小的角度（数值稳定）与接近 π
    Rb = rot([1, 2, 3], 0.7)
    R, Rr = np.stack([Rb @ rot([0.3, -1, 0.5], t) for t in th]), np.stack([Rb] * len(th))
    p = np.zeros((len(th), 3))
    dp, dr = P.pose_err(p, R, p + [0.003, -0.004, 0.0], Rr)
    assert np.allclose(dp, 0.005) and np.allclose(dr, th, rtol=1e-6, atol=1e-9)


def test_fk_single_joint_offset_moves_site_by_lever_arm(mj):
    model, data, sid = mj
    import mujoco
    q0 = np.array([[0.2, 1.0, -1.2, 0.1, 0.4, 0.3]])
    p0, R0 = P.fk(model, data, sid, q0)
    d = 0.01
    for j in range(6):                                                   # 单个转动关节偏 d：姿态差 = d，位置差 = 2ρ·sin(d/2)
        data.qpos[:6] = q0[0]
        mujoco.mj_kinematics(model, data)
        a, u = data.xanchor[j], data.xaxis[j]
        v = p0[0] - a
        rho = np.linalg.norm(v - (v @ u) * u)                             # 末端站点到该关节轴线的距离
        q = q0.copy()
        q[0, j] += d
        dp, dr = P.pose_err(*P.fk(model, data, sid, q), p0, R0)
        assert dr[0] == pytest.approx(d, rel=1e-9) and dp[0] == pytest.approx(2 * rho * np.sin(d / 2), rel=1e-9, abs=1e-12)


def test_metrics_uses_raw_zoh_reference_and_mask(mj):
    model, data, sid = mj
    n = 120
    ts = 0.05 * np.arange(n) + np.random.default_rng(1).uniform(0, 0.004, n)
    cmd = np.array([0.2, 1.0, -1.2, 0.1, 0.4, 0.3]) + 0.1 * np.sin(2 * np.pi * 0.5 * ts)[:, None]
    tq = np.arange(ts[0], ts[-1], 1 / 200)
    q = P.S.zoh(ts, cmd, tq).copy()                                       # 实际 = 原始指令零阶保持
    bad = (tq > 2.0) & (tq < 3.0)
    q[bad, 1] += 0.05                                                     # 只在被排除的帧上有大偏差
    ep = {"tq": tq, "q": q, "t_send": ts}
    m = P.metrics(ep, cmd, ~bad, model, data, sid)
    assert m["pos_raw"] < 1e-9 and m["rot_raw"] < 1e-9                    # 这批帧上与参考完全一致
    assert np.isfinite(m["pos_ideal"]) and m["pos_ideal"] > 0             # 理想渲染一族与零阶保持不同
    assert P.metrics(ep, cmd, np.ones(len(tq), bool), model, data, sid)["rot_raw"] > 1.0   # 不排除时那段偏差会算进去（mrad）
