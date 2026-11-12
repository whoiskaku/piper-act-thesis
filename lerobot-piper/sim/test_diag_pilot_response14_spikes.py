"""收口诊断的测试：合成数据，不读真实记录。"""
import gc
import sys
from pathlib import Path

import numpy as np

gc.disable()
HERE = Path(__file__).resolve().parent
for _p in (HERE / "q1", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import diag_pilot_response14_spikes as S   # noqa: E402


def test_repeated_spikes_match_and_random_spikes_do_not():
    rng = np.random.default_rng(0)
    T = 600
    d1, d2 = rng.normal(0, 0.03, (T, 3)), rng.normal(0, 0.03, (T, 3))
    frames = [50, 120, 300, 450]
    for i, f in enumerate(frames):                                      # 同位置、同方向、相近幅度
        d1[f, i % 3] += 0.4 * (-1) ** i
        d2[f, i % 3] += 0.35 * (-1) ** i
    m = S.match_spikes(d1, d2, thr=0.14)
    assert m["n_spikes"] >= 8 and m["match_rate"] > 0.9 and m["sign_agree"] > 0.9
    assert 0.7 < m["amp_ratio_median"] < 1.2 and m["match_rate"] > 5 * m["chance_rate"]
    e1, e2 = rng.normal(0, 0.03, (T, 3)), rng.normal(0, 0.03, (T, 3))
    for i, f in enumerate(frames):                                      # 位置不同的尖峰
        e1[f, 0] += 0.4
        e2[f + 37, 0] += 0.4
    m2 = S.match_spikes(e1, e2, thr=0.14)
    assert m2["match_rate"] < 0.2


def test_sampling_phase_and_consistent_resampling_remove_jitter():
    """20 Hz 样本 = 读数时最新的 100 Hz 样本；每次复放的 100 Hz 网格相位不同 ⇒ 同一运动被取在不同时刻。
    按指令写入时刻的固定偏移在 100 Hz 流上重采样后，两次的差应明显变小。"""
    f = lambda t: np.stack([np.sin(2 * np.pi * 7.3 * t), np.cos(2 * np.pi * 3.1 * t), np.sin(2 * np.pi * 8.7 * t)], 1)

    def rec(seed):
        r = np.random.default_rng(seed)
        t_send = 0.5 + np.arange(587) * 0.05
        t_read = t_send + 0.0024
        phi = r.uniform(0, 0.01)
        t100 = phi + np.arange(3200) * 0.01                             # 真实采样时刻（网格相位各次不同）
        idx = np.searchsorted(t100, t_read) - 1                         # 读数时最新的样本
        ms = 40000 + np.arange(3200) * 10
        host = t100 + r.uniform(0, 0.0005, len(t100))                   # 主机时间戳带接收抖动
        i100 = np.column_stack([host, ms, np.zeros((len(t100), 3)), f(t100)])
        return {"t_send": t_send.tolist(), "t": t_read.tolist(),
                "imu": np.column_stack([np.zeros((587, 3)), f(t100)[idx]]).tolist(), "imu100": i100.tolist()}

    a, b = rec(2), rec(3)
    ph = S.sampling_phase(a)
    assert ph.shape == (587,) and np.all(np.abs(ph) < 0.012)
    raw_diff = np.sqrt(np.mean((np.asarray(a["imu"])[:, 3:6] - np.asarray(b["imu"])[:, 3:6]) ** 2))
    ca, cb = S.resample_consistent(a, -0.005), S.resample_consistent(b, -0.005)
    assert ca.shape == (587, 3) and np.sqrt(np.mean((ca - cb) ** 2)) < 0.3 * raw_diff


def test_timing_audit_flags_late_frames_and_imu_gaps():
    t_send = 0.5 + np.arange(100) * 0.05
    t_send[40:] += 0.02                                                 # 一次 70 ms 的间隔
    ms = np.arange(0, 6000, 10)
    ms = np.delete(ms, [200, 201])                                      # 丢两个 100 Hz 样本
    i100 = np.column_stack([ms / 1000.0, ms, np.zeros((len(ms), 6))])
    rec = {"t_send": t_send.tolist(), "t": (t_send + 0.002).tolist(), "imu": np.zeros((100, 6)).tolist(),
           "imu100": i100.tolist(), "t_send_span": [0.002] * 100}
    a = S.timing_audit(rec)
    assert a["late_frames_gt60ms"] == 1 and a["imu100_gaps"] == 1 and abs(a["send_interval_max_ms"] - 70.0) < 1e-6


def test_command_check_compares_recorded_actions_with_the_plan():
    cmd7 = np.zeros((10, 7))
    da = np.zeros((10, 6))
    da[3, 1] = 0.002
    rec_cmd = np.concatenate([cmd7[:, :6] + da, cmd7[:, 6:7]], 1)
    assert S.command_check(rec_cmd, cmd7, da, use_probe=True)["max_abs_rad"] < 1e-12
    assert S.command_check(rec_cmd, cmd7, da, use_probe=False)["max_abs_rad"] > 1e-3
