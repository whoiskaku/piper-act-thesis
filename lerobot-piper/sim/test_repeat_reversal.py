"""`repeat_reversal.py` 的测试：事件换算、换向与近静止的判定、回摆、留一法、分项判定。不连机械臂、不读数据集。"""
from __future__ import annotations

import gc

gc.disable()                      # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repeat_reversal as R  # noqa: E402


def ramp(*legs):
    """分段匀速的 j1 指令（rad）：legs = (帧数, 每帧步长)。"""
    return np.r_[0.0, np.concatenate([np.full(n, d) for n, d in legs])].cumsum()


def cmd6(j1):
    c = np.zeros((len(j1), 6))
    c[:, 0] = j1
    return c


def test_event_time_uses_each_replays_own_t_send():
    rng = np.random.default_rng(0)
    t_send = 100.0 + np.cumsum(0.05 + rng.uniform(-0.005, 0.01, 80))     # 一遍回放自己的写总线时刻（帧长有抖动）
    traj = R.T.PlanTrajectory(cmd6(np.zeros(80)), t_send)
    for k in (0, 7, 19, 20, 613, 1558):
        f, s = divmod(k, 20)
        assert R.event_time(t_send, k) == pytest.approx(t_send[f] + (s + 1) / 20 * (t_send[f + 1] - t_send[f]))
        assert R.event_time(t_send, k) == pytest.approx(traj.t[k])        # 与本遍参考的细分步网格是同一时刻


def test_reversal_is_a_turning_point_of_the_reference():
    # 静止 0.5 s → 上行 1 s（每帧 0.01 rad）→ 慢速同向 0.5 s（每帧 0.0025 rad：参考每帧降到零再走，峰速约 0.11 rad/s）
    # → 更慢 0.5 s（每帧 0.001 rad，峰速约 0.07 rad/s）→ 掉头下行 1.5 s → 停住 1.5 s（参考到位时有约 0.01° 的数值小过冲）
    j1 = ramp((10, 0.0), (20, 0.01), (10, 0.0025), (10, 0.001), (30, -0.01), (30, 0.0))
    rev = [e for e in R.find_events(cmd6(j1)) if e["kind"] == "reversal"]
    assert [(e["j"], e["sign"]) for e in rev] == [(0, 1)]     # 只有掉头那一次；之前的上行峰速 0.2 rad/s 照样算进「之前的运动」
    assert 2.5 <= rev[0]["t"] < 2.7                            # 指令在第 50 帧（2.5 s）掉头


def test_near_still_needs_half_second_below_threshold_after_motion():
    def still_events(jitter, hold):
        j1 = ramp((10, 0.0), (20, 0.01))                       # 开头静止 0.5 s（之前没有运动，不算）→ 上行 1 s
        c = j1[-1]
        j1 = np.r_[j1, c + jitter * (-1.0) ** np.arange(hold)]  # 停 hold 帧，指令每帧 ±jitter 抖动
        j1 = np.r_[j1, c + 0.01 * np.arange(1, 21), np.full(10, c + 0.2)]   # 再走 1 s → 停 0.5 s（窗口不完整，不算）
        return [e for e in R.find_events(cmd6(j1)) if e["kind"] == "still"]

    ev = still_events(0.0005, 16)                              # 停 0.8 s，指令抖动范围 0.06°
    assert len(ev) == 1 and 1.5 <= ev[0]["t"] < 1.8
    assert still_events(0.0015, 16) == []                      # 抖动范围 0.17° > 0.1°
    assert still_events(0.0005, 8) == []                       # 只停 0.4 s


def test_swing_back_ignores_normal_return_and_counts_crossing():
    t = np.arange(-0.2, 0.6 + 1e-9, 0.005)

    def ref(x):                                                # 参考在 t = 0 掉头（°）
        return 10.0 - 30.0 * x ** 2

    def g(x):
        return np.exp(-0.5 * (x / 0.02) ** 2)

    lag = ref(t - 0.05)                                        # 晚 50 ms 跟着参考：掉头后跌了 9° 多，但都在参考的这一侧
    assert lag.max() - lag[-1] > 9
    assert R.over_swing(t, lag, lag - ref(t), 10.0, 1)[:2] == pytest.approx((0.0, 0.0), abs=1e-9)
    act = ref(t) + 0.2 * g(t) - 0.3 * g(t - 0.3)               # 多走 0.2°；之后在 0.3 s 越到当时参考另一侧 0.3°
    assert R.over_swing(t, act, act - ref(t), 10.0, 1) == pytest.approx((0.2, 0.3, 0.0), abs=1e-3)
    assert R.over_swing(t, -act, ref(t) - act, -10.0, -1) == pytest.approx((0.2, 0.3, 0.0), abs=1e-3)   # 反方向同理


def test_over_swing_window_stops_at_adjacent_turns_including_unfiltered():
    # 静止 0.5 s → 快速上行 1 s → 慢慢回落 0.17°（峰速约 0.07 rad/s，这个转折没通过峰速筛选）→ 再上行超过旧极值。
    # 实际与参考完全相同：多走、回摆都应为 0；不在相邻转折处截断的话，再上行那段会被算成几度的多走
    j1 = ramp((10, 0.0), (20, 0.01), (3, -0.001), (10, 0.01), (30, 0.0))
    cmd7 = np.column_stack([cmd6(j1), np.zeros(len(j1))])
    t_send = 0.05 * np.arange(len(j1))
    traj = R.T.PlanTrajectory(cmd7[:, :6], t_send)
    ev = [e for e in R.find_events(cmd7[:, :6]) if e["kind"] == "reversal"]
    assert len(ev) == 1                                        # 回落那一下不是换向事件……
    assert len(R.T.turning_points(np.degrees(traj.Q[:, 0]), R.T.TURN_DEG)) == 2   # ……但它是转折点，要用来截窗
    ev = [e | {"turns": b} for e, b in zip(ev, R.turn_bounds(cmd7, ev))]
    tq, ti = np.arange(0, t_send[-1], 0.005), np.arange(0, t_send[-1], 0.01)
    ep = {"t_send": t_send, "tq": tq, "q": traj(tq)[:, :6], "ti": ti}
    w = R.windows(ep, traj, np.zeros((len(tq), 6)), np.zeros((len(ti), 3)), ev)[0]
    assert w["over"] == pytest.approx(0.0, abs=1e-3) and w["swing"] == 0.0
    te = R.event_time(t_send, ev[0]["k"])
    qa = np.degrees(np.interp(te + R.GQ, tq, ep["q"][:, 0]))
    assert R.over_swing(R.GQ, qa, w["enc"], np.degrees(traj(te)[0, 0]), 1)[0] > 1.0   # 用整个窗口：再上行被算成多走
    assert len(w["enc"]) == len(R.GQ)                           # 评分波形仍是整个 −0.2 / +0.6 s 窗口


def test_loo_r2_identical_waveform_with_noise_and_pure_noise():
    rng = np.random.default_rng(1)
    n, m = 8, 20000
    x = np.sin(np.linspace(0, 200 * np.pi, m))                 # 功率 0.5
    assert R.loo_r2(np.tile(x, (n, 1))) == pytest.approx(1.0)
    noisy = x + 0.5 * rng.standard_normal((n, m))              # 噪声功率 0.25：期望 (P − σ²/(n−1)) / (P + σ²)
    assert R.loo_r2(noisy) == pytest.approx((0.5 - 0.25 / (n - 1)) / 0.75, abs=0.01)
    assert R.loo_r2(rng.standard_normal((n, m))) == pytest.approx(-1 / (n - 1), abs=0.01)


def test_contact_flag_marks_windows_touching_gripper_switch_or_target_zone():
    import mujoco
    from ideal_imu_kinematic import MODEL_PATH
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    cmd7 = np.zeros((200, 7))                                  # 10 s
    cmd7[:, :6] = np.deg2rad([0, 30, -30, 0, 20, 0])
    cmd7[150:, 1] += np.deg2rad(10)                            # 7.5 s 起换一个姿态，目标区中心就放在这个姿态的末端
    cmd7[60:, 6] = 0.07                                        # 3.0 s 夹爪张开 ⇒ 疑似接触段 2.5–3.5 s
    center = R.T.tcp(cmd7[150:151, :6], model)[0]
    ev = [{"t": t} for t in (1.5, 2.0, 4.0, 6.5, 7.0)]         # 窗口 −0.2 / +0.6 s
    assert R.contact_flags(cmd7, ev, model, center) == [False, True, False, False, True]


def test_verdict_per_item_only_from_A_and_B():
    assert R.verdict({"A": 0.7, "B": 0.6, "C": -0.5}) == "支持针对这一项继续尝试补正"   # C 再差也不否决
    assert R.verdict({"A": 0.1, "B": 0.29, "C": 0.95}) == "这一项暂不推进"               # C 再好也不改变
    assert R.verdict({"A": 0.7, "B": 0.3, "C": 0.9}) == "无法判定"                       # 一条 ≥ 0.6、一条不到 0.6 也不 < 0.3
    assert R.verdict({"A": None, "B": 0.9, "C": 0.9}) == "无法判定"                      # A 不足 6 遍或该项没有事件
