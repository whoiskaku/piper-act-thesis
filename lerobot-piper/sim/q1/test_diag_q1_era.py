import numpy as np
import pytest

import diag_q1_era as G
import q1_data as D
import q1_testing as T
import q1_train as TR
import q1_windows as W


def test_not_part_of_the_frozen_code_fingerprint():
    """事后诊断不得改变冻结实验的 code_sha。"""
    assert "diag_q1_era.py" not in TR.CODE_FILES


def test_cmd_stats_on_a_constant_velocity_ramp():
    e = T.synthetic_episode(0)
    e.cmd = e.cmd.copy()
    e.cmd[:, :6] = np.arange(len(e.cmd))[:, None] * 0.01 * D.FS / D.FS     # 每帧 +0.01 rad
    s = G.cmd_stats(e)
    assert np.isclose(s["speed_j1"], 0.01 * D.FS) and s["acc_j1"] < 1e-9
    assert np.isclose(s["boundary_jump_ratio"], 1.0)                      # 匀速：边界与块内一样


def test_boundary_jump_is_detected():
    e = T.synthetic_episode(0)
    e.cmd = e.cmd.copy()
    e.cmd[:, :6] = 0.0
    for t0 in (100, 200, 300, 400, 500):
        e.cmd[t0:, 0] += 0.1                                              # 每个 chunk 开头跳 0.1 rad
    e.cmd[:, 1] = np.arange(len(e.cmd)) * 1e-3                            # 块内小步，避免中位步长为 0
    s = G.cmd_stats(e)
    assert np.isclose(s["boundary_jump"], np.hypot(0.1, 1e-3)) and s["boundary_jump_ratio"] > 50


def test_rotation_fit_recovers_a_known_rotation_and_gain():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(2000, 3))
    c, s = np.cos(np.radians(3.0)), np.sin(np.radians(3.0))
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    ang, g = G.kabsch(A, 1.2 * A @ Rz)
    assert abs(ang - 3.0) < 1e-6 and abs(g - 1.2) < 1e-9


def test_tracking_fit_on_the_synthetic_first_order_lag():
    """合成集：q[k+1] = q[k] + 0.3·(u[k] − q[k])、u 为随机游走 ⇒ 速度互相关峰在 d = 1，幅值比 ≈ 0.3。"""
    eps = [T.synthetic_episode(s) for s in range(3)]
    d, a, r = G.tracking_fit(eps, 0)
    assert d == 1 and abs(a - 0.3) < 0.03 and r > 0.6


def _rz(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rotated(seed, deg):
    """合成集，令实测陀螺 = 理想陀螺 × Rz(deg)（行向量约定），按 ALIGN_L 对齐。"""
    e = T.synthetic_episode(seed)
    e.ideal = np.random.default_rng(seed).normal(0, 0.5, e.ideal.shape)
    e.w = np.zeros_like(e.w)
    e.w[D.ALIGN_L:] = e.ideal[:e.n] @ _rz(deg)
    e.gap = e.w[D.ALIGN_L:] - e.ideal[:e.n]
    return e


def test_rotation_calib_recovers_a_known_mount_rotation():
    """仓库惯例（imu_frame_calib）：未滤波陀螺、|理想| > 0.15 rad/s 的样本、滞后 3 帧做 Kabsch。"""
    eps = [_rotated(s, 4.0) for s in range(3)]
    ang, gain, Rd, n = G.rotation_calib(eps)
    assert abs(ang - 4.0) < 1e-6 and abs(gain - 1.0) < 1e-9 and n > 1000


def test_rotation_share_is_one_when_the_gap_is_pure_rotation():
    eps = [_rotated(s, 4.0) for s in range(3)]
    share = G.rotation_share(eps, _rz(4.0), W.kernel())
    assert abs(share - 1.0) < 1e-9
    assert G.rotation_share(eps, np.eye(3), W.kernel()) == 0.0


def test_ood_rate_is_zero_inside_and_one_outside():
    train = [T.synthetic_episode(s) for s in range(4)]
    inside = G.ood_rate(train, train, G.positions)
    assert inside <= 0.11                                                 # 训练集自身：每维约 2%（分位插值略多），5 维合计约 10%
    far = T.synthetic_episode(9)
    far.cmd = far.cmd.copy()
    far.cmd[:, :6] += 5.0
    assert G.ood_rate([far], train, G.positions) == 1.0


def test_separated_flag():
    assert G.separated({"a": 0.0, "b": 10.0}, {"c": 4.0, "d": 5.0, "e": 6.0})
    assert not G.separated({"a": 5.5, "b": 10.0}, {"c": 4.0, "d": 5.0, "e": 6.0})
