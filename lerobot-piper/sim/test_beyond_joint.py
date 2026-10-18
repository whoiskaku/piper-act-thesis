"""`beyond_joint.py` 的测试：运动学预测、线性相干分解、按指令帧对齐、因果性（改未来、过去不变）、可预测度。不连机械臂、不读数据集。"""
from __future__ import annotations

import gc

gc.disable()                      # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）

import copy  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beyond_joint as B  # noqa: E402


def model_sid():
    import mujoco
    from ideal_imu_kinematic import MODEL_PATH
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    return model, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site")


def test_predicted_gyro_equals_single_joint_rate():
    # 只有 j1 转（绕固定轴）：安装点角速度的模 = |q̇1|（标定取单位阵、零偏 0、不延迟）
    model, sid = model_sid()
    tq, ti = np.arange(0, 4, 0.005), np.arange(0, 4, 0.01)
    q = np.tile(np.deg2rad([0, 40, -40, 0, 20, 0]), (len(tq), 1))
    q[:, 0] = 0.3 * np.sin(2 * np.pi * 0.5 * tq)
    calib = {"R": np.eye(3), "b": [0, 0, 0], "gyro_bias": [0, 0, 0], "dp": [0, 0, 0], "delay_s": 0.0}
    ep = {"tq": tq, "q": q, "ti": ti}
    w = B.at(ep, B.predict_on_grid(ep, calib, model, sid)[1], 0.0)
    rate = np.abs(0.3 * 2 * np.pi * 0.5 * np.cos(2 * np.pi * 0.5 * ti))
    mid = (ti > 0.5) & (ti < 3.5)
    assert np.allclose(np.linalg.norm(w, axis=1)[mid], rate[mid], atol=2e-3)


def test_spectra_split_puts_an_extra_oscillation_on_the_right_side():
    rng = np.random.default_rng(0)
    t = np.arange(0, 300, 0.01)
    common = B.lt15(np.cumsum(rng.standard_normal((len(t), 3)), axis=0) * 0.01)   # 两边共有的宽带运动
    extra = np.zeros_like(common)
    extra[:, 0] = 0.05 * np.sin(2 * np.pi * 8 * t)                             # 8 Hz，RMS 0.035
    s = B.spectra(common + extra, common)                                      # 只加在实测侧
    assert B.band_rms(s, "meas_incoh") == pytest.approx(0.05 / np.sqrt(2), rel=0.1)
    assert B.band_rms(s, "pred_incoh") < 0.005
    s = B.spectra(common, common + extra)                                      # 只加在预测侧
    assert B.band_rms(s, "pred_incoh") == pytest.approx(0.05 / np.sqrt(2), rel=0.1)
    assert B.band_rms(s, "meas_incoh") < 0.005


def test_to_nominal_aligns_replays_by_their_own_frame_times():
    rng = np.random.default_rng(1)
    grid = np.arange(1.0, 9.0, 0.01)
    reps = []
    for _ in range(3):                                                         # 三遍的帧长各不相同
        t_send = 100.0 + np.cumsum(0.05 + rng.uniform(-0.004, 0.006, 200))
        ti = np.arange(t_send[0], t_send[-1], 0.01)
        tau = np.interp(ti, t_send, 0.05 * np.arange(200))                     # 真正的名义时间
        x = np.column_stack([np.sin(2 * np.pi * 0.7 * tau), np.cos(2 * np.pi * 1.3 * tau), tau * 0])
        reps.append(B.to_nominal({"ti": ti, "t_send": t_send}, x, grid))
    assert B.RR.loo_r2(np.array(reps)) == pytest.approx(1.0, abs=1e-4)


def synthetic_record(rng, bump=0.0, t_cut=2.0):
    """合成一条回放记录（字段同 replay_probe）：三组 CAN 关节角 ~200 Hz（时刻有抖动）、IMU 100 Hz。bump ≠ 0 时只改 t_cut 之后的编码器。"""
    rows = []
    for g, cols in B.D.Q_GROUPS:
        t = np.cumsum(0.005 + rng.uniform(-0.001, 0.001, 900))
        for tk in t:
            q = [0.2 * np.sin(2 * np.pi * 0.7 * tk + c) + (bump * np.sin(2 * np.pi * 3 * tk) if tk > t_cut else 0.0) for c in cols]
            rows.append([tk, g, *(np.asarray(q) / B.D.D2R)])
    rows.sort(key=lambda x: x[0])
    ti = np.arange(0.3, 4.2, 0.01)
    imu = [[tk, tk * 1000, 0.0, 0.0, 9.8, *rng.normal(0, 0.05, 3)] for tk in ti]
    return {"t_send": list(np.arange(0.2, 4.3, 0.05)), "can_q200": rows, "imu100": imu,
            "can_hs_meta": {"clock": {"offset_realtime_minus_monotonic_s": 0.0}}}


def test_causal_branch_past_unchanged_when_future_encoder_data_change():
    # 端到端：同一条记录，只改 2 s 之后的编码器原始样本。决策时刻 2 s 能用的残差（IMU 时刻 ≤ 2 s − 10 ms，见 D_AVAIL）
    # 及其滤波输出必须逐位不变；离线分支（零相位低通 + 样条）在更早处就会变——说明这个测试能抓到未来信息混入。
    model, sid = model_sid()
    calib = {"R": np.eye(3), "b": [0, 0, 0], "gyro_bias": [0, 0, 0], "dp": [0, 0, 0], "delay_s": 0.002}
    h1 = synthetic_record(np.random.default_rng(3))
    h2 = synthetic_record(np.random.default_rng(3), bump=0.3)
    assert h1["imu100"] == h2["imu100"]                                        # IMU 两边相同
    act = np.zeros((len(h1["t_send"]), 7))
    ep1, ep2 = B.T.from_record(h1, act), B.T.from_record(copy.deepcopy(h2), act)
    avail = ep1["ti"] <= 2.0 - B.D_AVAIL * 0.01 + 1e-9
    r1, r2 = (np.subtract(*B.causal_parts(ep, calib, model, sid)) for ep in (ep1, ep2))
    assert np.array_equal(r1[avail], r2[avail])
    assert np.array_equal(B.causal_filter(r1, B.MAIN_BAND)[avail], B.causal_filter(r2, B.MAIN_BAND)[avail])
    assert np.abs(r1[ep1["ti"] > 2.05] - r2[ep1["ti"] > 2.05]).max() > 0.1   # 未来确实变了
    o1, o2 = (ep["imu"][:, 3:] - B.at(ep, B.predict_on_grid(ep, calib, model, sid)[1], 0.002) for ep in (ep1, ep2))
    assert np.abs(o1[avail] - o2[avail]).max() > 1e-3                         # 离线分支的「过去」被未来改变了


def test_causal_state_does_not_alias_high_frequency_encoder_content_into_band():
    # 编码器侧 95 Hz 的小幅成分（类似时间戳的规律性图样）：取到 100 Hz 的 IMU 时刻后不能折叠进 3–15 Hz；
    # 不先低通的做法（v1.2、v1.3）会把它折叠到 5 Hz
    tq, t = np.arange(0, 20, 0.005), np.arange(1, 19, 0.01) + 0.0013
    smooth = np.zeros((len(tq), 6))
    smooth[:, 0] = 0.3 * np.sin(2 * np.pi * 0.5 * tq)
    hf = smooth.copy()
    hf[:, 0] += 0.0005 * np.sin(2 * np.pi * 95 * tq)                          # 角速度幅值约 0.3 rad/s
    leak = B.nrms(B.T.band(B.causal_state(tq, hf, t)[1][:, :1] - B.causal_state(tq, smooth, t)[1][:, :1], B.FS))
    assert leak < 0.005

    def raw(q):                                                                # 对照：只差分、直接插值到 IMU 时刻
        return np.interp(t, tq, np.r_[0.0, np.diff(q[:, 0])] * 200)

    assert B.nrms(B.T.band((raw(hf) - raw(smooth))[:, None], B.FS)) > 0.03


def test_predictability_persistent_oscillation_vs_white_noise():
    rng = np.random.default_rng(2)
    n_rep, n = 8, 3000
    tau = np.arange(n) * 0.01
    common = np.sin(tau)[:, None] * np.ones(3)                                 # 每遍都一样（可复现部分）
    w, r = 2 * np.pi * 8 / 100, 0.99                                           # 8 Hz、轻阻尼的 AR(2)
    osc = np.zeros((n_rep, n, 3))
    eps = rng.standard_normal((n_rep, n, 3)) * 0.01
    for k in range(2, n):
        osc[:, k] = 2 * r * np.cos(w) * osc[:, k - 1] - r * r * osc[:, k - 2] + eps[:, k]

    def series(x):
        return [{"tau": tau, "tau_lin": tau, "main": common + x[s], "ok": np.ones(n, bool)} for s in range(n_rep)]

    res = B.predictability(series(osc))
    assert res[10][0] > 0.5 and res[10][0] > res[10][1]                        # 100 ms 后仍可预测，且好过「保持现值」
    res = B.predictability(series(rng.standard_normal((n_rep, n, 3))))
    assert abs(res[10][0]) < 0.05 and res[10][1] == pytest.approx(-1.0, abs=0.1)   # 白噪声：预测不了；保持现值约 −1
