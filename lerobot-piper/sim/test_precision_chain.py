"""重复精度封顶验证的固定处理链（PLAN_REPEAT_PRECISION_CAPPED.md §2）测试：合成数据。"""
import gc
import sys
from pathlib import Path

import numpy as np
import pytest

gc.disable()
HERE = Path(__file__).resolve().parent
for _p in (HERE / "q1", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import precision_chain as C   # noqa: E402


def test_fir_is_the_frozen_design():
    b = C.aa_fir()
    assert len(b) == 183 and abs(b.sum() - 1.0) < 1e-12 and C.fir_sha16(b) == C.FIR_SHA16


def test_zero_phase_filter_keeps_the_band_without_delay_and_removes_aliasing_content():
    t = np.arange(0, 30, 0.01)
    keep = np.sin(2 * np.pi * 3.0 * t)
    x = np.stack([keep + 0.5 * np.sin(2 * np.pi * 15.0 * t)] * 3, 1)
    y = C.aa_filter(x)
    mid = slice(300, -300)
    assert np.max(np.abs(y[mid, 0] - keep[mid])) < 1e-3                 # 3 Hz 原样保留、零延迟
    assert np.std(y[mid, 1] - keep[mid]) < 1e-3                         # 15 Hz 被滤掉


def test_frame_locked_sampling_hits_the_fixed_phase_and_refuses_extrapolation():
    ms = 50000 + np.arange(3200) * 10
    host = 1.0 + (ms - ms[0]) / 1000.0 + np.random.default_rng(0).uniform(-0.00025, 0.00025, len(ms))   # 零均值接收抖动
    t_true = 1.0 + (ms - ms[0]) / 1000.0
    f = lambda t: np.stack([np.sin(2 * np.pi * 2.0 * t), np.cos(2 * np.pi * 4.0 * t), np.sin(2 * np.pi * 1.0 * t)], 1)
    t_send = 3.0 + np.arange(500) * 0.05 + np.random.default_rng(1).uniform(-0.002, 0.002, 500)
    rec = {"t_send": t_send.tolist(), "imu100": np.column_stack([host, ms, np.zeros((len(ms), 3)), f(t_true)]).tolist()}
    y = C.frame_locked(rec)
    want = f(t_send + C.OFFSET_S)
    assert y.shape == (500, 3) and np.max(np.abs(y - want)) < 5e-3
    rec_bad = dict(rec, t_send=(t_send + 40.0).tolist())                # 超出 100 Hz 覆盖范围
    with pytest.raises(ValueError):
        C.frame_locked(rec_bad)
