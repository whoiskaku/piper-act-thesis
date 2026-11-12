"""`b4_common.py` 与在线窗的测试：在线配置的生成与读回、正运动学与 cmd_tracking 一致、在线窗与 v0.10 窗的对照、多留 0.1 s 只会多置零。
读一条开 TE 的基线指令（eval_pickcube_base_TE 第 19 集）与示教数据集（目标中心）；不连机械臂。"""
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
import b4_common as C  # noqa: E402
import cmd_tracking as T  # noqa: E402
import ilc_route1 as I  # noqa: E402
import repeat_reversal as RR  # noqa: E402
import shift_replay as S  # noqa: E402
from ideal_imu_kinematic import MODEL_PATH  # noqa: E402

import lerobot.common.policies.act.b4_online as O  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(MODEL_PATH))


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    p = C.make_cfg(tmp_path_factory.mktemp("b4") / "online_cfg.json")
    assert C.make_cfg(p) == p                       # 再生成一次内容相同：不报错、不改文件
    return O.load_cfg(p)


@pytest.fixture(scope="module")
def a7():
    return RR.episode_cmd("eval_pickcube_base_TE", 19)[RR.HOLD:].astype(float)


def exact_predictions(a7):
    """预测 = 最终值：第 k 帧之后的部分融合值就是第 k+1 … 帧的最终指令（最后一帧之后用最后一帧）。"""
    return [a7[k + 1:k + 100] if k + 1 < len(a7) else a7[-1:] for k in range(len(a7))]


def v010_window(a7, model, cfg, n_c=10, float_times=False):
    """v0.10 的窗（ilc_route1.Ctx）：开头补 HOLD 帧，疑似接触（原始指令与提前 60 ms 的指令取并集）、限位饱和、保持段为 0，
    向外 10 帧余弦过渡，再去掉开头的 HOLD 帧。不加片尾那一帧（在线不可知）。
    float_times = True：夹爪切换用 v0.10 原样的 RR.contact_frames（浮点时刻比较）；否则用整数帧距 ±n_c。返回 (窗, 疑似接触帧)。"""
    a7h = np.concatenate([np.repeat(a7[:1], RR.HOLD, 0), a7])
    s7h = S.shifted(a7h, 0.060).astype(float)
    center = np.asarray(cfg.center)

    def contact(c):
        if float_times:
            return RR.contact_frames(c, model, center)
        ev, k = T.grip_events(c[:, 6]), np.arange(len(c))
        grip = (np.abs(k[:, None] - ev[None, :]) <= n_c).any(1) if len(ev) else np.zeros(len(c), bool)
        return grip | (np.linalg.norm(T.tcp(c[:, :6], model) - center, axis=1) <= T.ZONE_M)

    marked = contact(a7h) | contact(s7h)
    zero = np.repeat(marked[:, None], 5, 1) | I.saturated(a7h[:, :6], s7h[:, :6], np.asarray(cfg.jnt_lo), np.asarray(cfg.jnt_hi))
    zero[:RR.HOLD] = True
    return I.window(zero, 10)[RR.HOLD:], marked[RR.HOLD:]


def test_tcp_matches_cmd_tracking(model):
    q = np.random.default_rng(1).uniform(-1, 1, (20, 6))
    assert np.allclose(O.Tcp(str(MODEL_PATH))(q), T.tcp(q, model), atol=1e-12)


def test_online_window_equals_v010_rule_when_predictions_exact(model, cfg, a7):
    """预测 = 最终值、不多留 0.1 s 时，在线窗与 v0.10 的窗（夹爪切换按整数帧距 ±10）逐帧相同。
    v0.10 原样的浮点时刻比较只在「恰好离夹爪切换 10 帧」的帧上取舍不定（随舍入、随帧号起点变）：整数帧距在那里一律算接触。"""
    c0, tcp, ens = dataclasses.replace(cfg, contact_pad_s=0.0), O.Tcp(cfg.model_path), exact_predictions(a7)
    w_on = np.stack([O.online_window(a7, ens[k], k, c0, tcp) for k in range(len(a7))])
    w_off, m_int = v010_window(a7, model, cfg)
    assert np.allclose(w_on, w_off, atol=1e-12)
    _, m_float = v010_window(a7, model, cfg, float_times=True)
    assert not (m_float & ~m_int).any()                        # 整数帧距只会多置零
    a7h = np.concatenate([np.repeat(a7[:1], RR.HOLD, 0), a7])
    ev = np.concatenate([T.grip_events(a7h[:, 6]), T.grip_events(S.shifted(a7h, 0.060)[:, 6].astype(float))]) - RR.HOLD
    for k in np.flatnonzero(m_int & ~m_float):                 # 多置零的帧都恰好离某个夹爪切换 10 帧
        assert np.abs(ev - k).min() == 10


def test_contact_pad_only_adds_zero_frames(model, cfg, a7):
    """偏保守：夹爪切换前后多留 0.1 s，窗只会更小，而且确实有帧变小。"""
    tcp, ens = O.Tcp(cfg.model_path), exact_predictions(a7)
    w0 = np.stack([O.online_window(a7, ens[k], k, dataclasses.replace(cfg, contact_pad_s=0.0), tcp) for k in range(len(a7))])
    w1 = np.stack([O.online_window(a7, ens[k], k, cfg, tcp) for k in range(len(a7))])
    assert np.all(w1 <= w0) and np.any(w1 < w0)
