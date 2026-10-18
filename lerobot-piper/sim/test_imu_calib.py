"""`imu_calib.py` 标定动作序列的测试：只在 MuJoCo 模型里检查，不连机械臂。"""
from __future__ import annotations

import gc

gc.disable()                      # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import imu_calib as C  # noqa: E402


@pytest.fixture(scope="module")
def seq():
    cmd7, segs = C.make_sequence()
    return cmd7, segs, C.check_sequence(cmd7, segs)


def test_sequence_passes_model_safety_checks(seq):
    _, _, r = seq
    assert r["limit_ok"] and r["vel_ok"] and r["acc_ok"]
    assert r["z_ok"] and r["min_z"] >= 0.019


def test_starts_and_ends_at_home_with_gripper_fixed(seq):
    cmd7, _, _ = seq
    home = np.deg2rad(C.SQ.HOME_DEG)
    assert np.allclose(cmd7[0, :6], home) and np.allclose(cmd7[-1, :6], home)
    assert np.ptp(cmd7[:, 6]) == 0.0


def test_static_fit_windows_hold_still(seq):
    cmd7, segs, _ = seq
    static = [s for s in segs if s["kind"] == "static"]
    assert len(static) == len(C.STATIC)
    for s in static:
        assert s["end"] - s["fit_start"] == int(C.FIT_S * C.FPS)
        assert np.ptp(cmd7[s["fit_start"]:s["end"], :6], axis=0).max() == 0.0


def test_each_move_segment_rotates_exactly_one_joint(seq):
    cmd7, segs, _ = seq
    moved = []
    for s in segs:
        if s["kind"] == "move":
            span = np.ptp(cmd7[s["start"]:s["end"], :6], axis=0)
            assert (span > 0).sum() == 1
            moved.append(int(np.argmax(span)))
    assert moved == [0, 1, 2, 3, 4, 5]


def test_static_poses_see_gravity_from_spread_directions(seq):
    _, _, r = seq
    assert r["grav_cond"] < 5.0


def test_replay_entry_disables_gc_then_calls_pipeline(monkeypatch):
    import replay_probe
    seen = {}

    def fake_run(arm, dry_run=False, **kw):
        seen.update(arm=arm, dry_run=dry_run, gc_on=gc.isenabled(), **kw)
        return 0

    monkeypatch.setattr(replay_probe, "run", fake_run)
    gc.enable()                       # 模拟一个默认开着 GC 的新进程
    try:
        assert C.main(["replay", "--dry-run"]) == 0
    finally:
        gc.disable()
    assert seen == {"arm": "calib", "dry_run": True, "gc_on": False,
                    "seqfile": "outputs/imu_calib/calib_seq.npz",
                    "outdir": Path("outputs/imu_calib")}


def test_fit_recovers_a_known_mount_from_a_synthetic_replay(tmp_path):
    """合成回放：已知朝向 R、加计零偏 b、陀螺零偏、安装位置偏差 dp、延迟 → fit 应把它们解回来。
    「真实」IMU 用把 ee_imu_site 真的挪了 dp 之后的 MuJoCo 传感器算，不经过 fit 里的杆臂公式；
    IMU 主机时刻带 0–4 ms 随机到达抖动，关节角按 0.001° 量化、时间戳按 REALTIME 存（同真机记录）。"""
    import mujoco
    from scipy.interpolate import CubicSpline
    from scipy.spatial.transform import Rotation
    from ideal_imu_kinematic import MODEL_PATH, ideal_imu_traj

    cmd7, segs = C.make_sequence()
    cmd7 = cmd7.astype(np.float32).astype(float)             # 与序列文件里存的 float32 一致
    np.savez(tmp_path / "seq.npz", cmd7=cmd7.astype(np.float32), segs=json.dumps(segs))
    R0 = Rotation.from_euler("xyz", [3.0, -2.0, 5.0], degrees=True).as_matrix()
    b0, bg0 = np.array([0.3, -0.2, 1.1]), np.array([0.005, -0.004, 0.01])
    dp0, lat = np.array([0.012, -0.015, 0.02]), 0.030
    T0, OFF = 1000.0, 1.7e9                                   # monotonic 起点；REALTIME − MONOTONIC
    t_send = T0 + np.arange(len(cmd7)) / C.FPS
    sp = CubicSpline(t_send, cmd7)                            # 「实际」关节轨迹：过各帧指令的光滑曲线

    m = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site")
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, m.site_quat[sid])
    m.site_pos[sid] += rot.reshape(3, 3) @ dp0                # dp 在 site 系，site_pos 在 link6 系
    t_dev = np.arange(T0, t_send[-1], 0.01)                   # IMU 设备采样时刻 = 它测到的运动时刻
    true = ideal_imu_traj(m, mujoco.MjData(m), sp, t_dev, sid)[1]
    jit = np.random.default_rng(0).uniform(0.0, 0.004, len(t_dev))
    imu100 = np.column_stack([t_dev + lat + jit, 5_000_000 + 10 * np.arange(len(t_dev)),
                              true[:, :3] @ R0.T + b0, true[:, 3:] @ R0.T + bg0])
    t_q = np.arange(T0, t_send[-1], 0.005)
    qm = np.round(np.degrees(sp(t_q)[:, :6]) * 1000)         # 0.001° 量化
    can_q200 = [[t + OFF, g, qm[k, g - 1], qm[k, g]] for g in (1, 3, 5) for k, t in enumerate(t_q)]
    rec = {"cmd": cmd7.tolist(), "t_send": t_send.tolist(), "t": (t_send + 0.01).tolist(),
           "imu100": imu100.tolist(), "can_q200": can_q200,
           "can_hs_meta": {"clock": {"offset_realtime_minus_monotonic_s": OFF}}}
    (tmp_path / "replay_calib.json").write_text(json.dumps(rec))

    r = C.fit(tmp_path / "replay_calib.json", tmp_path / "seq.npz")

    assert C._deg(np.array(r["R"]) @ R0.T) < 0.05
    assert np.abs(np.array(r["gyro_bias"]) - bg0).max() < 2e-4
    assert np.abs(np.array(r["b"]) - b0).max() < 0.01
    assert np.abs(np.array(r["dp"]) - dp0).max() < 1e-3
    assert abs(r["delay_s"] - (lat + jit.mean())) < 1.5e-3   # 到达抖动的均值也算进延迟
    assert len(r["diag"]["static_resid"]) == len(C.STATIC)   # 首尾同名的 home 不能互相覆盖
