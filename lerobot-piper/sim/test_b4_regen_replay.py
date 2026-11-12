"""重新生成网络 ② 的在线回放程序（`sim/b4_regen_replay.py`，实现计划任务 6）的测试。合成数据，不连机械臂。"""
from __future__ import annotations

import gc

gc.disable()                      # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）

import json  # noqa: E402
import struct  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b4_regen as R  # noqa: E402
import b4_regen_replay as RP2  # noqa: E402
from test_b4_regen import case, recording_fn, teacher_oracle  # noqa: E402,F401  case 是任务 2 的合成夹具

import lerobot.common.policies.act.b4_online as O  # noqa: E402

LO, HI = -np.ones(6) * 9, np.ones(6) * 9      # 位置界放宽：这里不测位置


def run_loop(case, mode, fn, source=None, return_rec=False, h=None, **kw):
    """loop 跑一遍合成回放：clock、send 都返回记录里第 k 帧的 t_send，截止时刻和离线完全相同。返回发出的指令列表。kw 原样传给 loop。"""
    h = h or case.replays[0]
    ts, k, sent = h["t_send"], [-1], []

    def clock():
        k[0] += 1
        return ts[k[0]]

    def send(u7):
        sent.append(u7)
        return ts[k[0]]

    off = h["can_hs_meta"]["clock"]["offset_realtime_minus_monotonic_s"]
    rec = RP2.loop(case.n, case.base, case.log, fn, case.calib, off, LO, HI, source or RP2.Recorded(h), mode, send, clock, **kw)
    return (sent, rec) if return_rec else sent


def jump_at(frame: int, by: float):
    """测试用的「网络」：输出 = 之后单元（ACT 指令），第 frame 帧整体加 by。"""
    k = iter(range(10_000))
    return lambda past, fut: fut + (by if next(k) == frame else 0.0)


def drop_can_after(case, t: float):
    """实测源：第 0 遍合成回放，到达时刻晚于 t 的编码器行全部丢掉（模拟 CAN 中断）。"""
    h = case.replays[0]
    off = h["can_hs_meta"]["clock"]["offset_realtime_minus_monotonic_s"]
    q = np.asarray(h["can_q200"], float)
    return RP2.Recorded({**h, "can_q200": q[q[:, 0] - off <= t]})


def rec_ok(case):
    """一遍正常跑完的 loop 结果（完美的教师网络、regen 模式）。"""
    return run_loop(case, "regen", teacher_oracle(case), return_rec=True)[1]


class _FakeIMU:
    """假的 imu_reader：drain_recording 交出到达时刻不晚于 now() 的样本 (主机时刻, 设备 ms, 加速度, 陀螺)。"""

    def __init__(self, rows, now):
        self.rows, self.now, self.i = rows, now, 0

    def drain_recording(self):
        out = []
        while self.i < len(self.rows) and self.rows[self.i][0] <= self.now():
            r = self.rows[self.i]
            out.append((r[0], r[1], r[2:5], r[5:8]))
            self.i += 1
        return out


class _FakeCAN:
    """假的 CanHighSpdRecorder：drain_recording 交出到达时刻不晚于 now() 的原始帧 (t_realtime, can_id, 字节)。"""

    def __init__(self, q200, off, now):
        ids = {1: 0x2A5, 3: 0x2A6, 5: 0x2A7}
        self.frames = [(r[0], ids[int(r[1])], struct.pack(">ii", int(r[2]), int(r[3]))) for r in q200]
        self.off0, self.now, self.i = off, now, 0

    def drain_recording(self):
        out = []
        while self.i < len(self.frames) and self.frames[self.i][0] - self.off0 <= self.now():
            out.append(self.frames[self.i])
            self.i += 1
        return out


def test_dry_inputs_equal_the_offline_assembly(case):           # 设计 §6；Review Focus 1
    seen = []
    run_loop(case, mode="shadow", fn=recording_fn(seen, out=lambda p, f: f))   # Shift 回放 + shadow ⇒ 已发出的指令和记录相同
    d = R.assemble_regen(case.replays[0], case.base, case.log, case.teacher7, case.calib)
    for i, k in enumerate(d["k"]):
        assert np.array_equal(seen[k][0], d["past"][i]) and np.array_equal(seen[k][1], d["fut"][i])


def test_live_source_matches_recorded_bitwise(case):            # Review Focus 1：在线逐帧取实测，与读录好的记录逐位相同
    h = dict(case.replays[1])
    h["can_q200"] = np.c_[h["can_q200"][:, :2], np.round(h["can_q200"][:, 2:])]   # CAN 上传的是整数（0.001°）
    off, now = h["can_hs_meta"]["clock"]["offset_realtime_minus_monotonic_s"], [0.0]
    live = RP2.Live(SimpleNamespace(imu_reader=_FakeIMU(h["imu100"], lambda: now[0])), _FakeCAN(h["can_q200"], off, lambda: now[0]))

    class Source:                                                 # 到本帧开始为止到达的都已交出（含截止时刻之后的，Live 自己筛掉）
        def rows(self, cutoff):
            now[0] = cutoff + O.CUT_S
            return live.rows(cutoff)

    a, b = [], []
    run_loop(case, "regen", recording_fn(a, lambda p, f: f + 0.001), h=h)
    run_loop(case, "regen", recording_fn(b, lambda p, f: f + 0.001), source=Source(), h=h)
    assert len(a) == len(b) == case.n and all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(a, b))


def test_only_the_first_output_frame_is_sent(case):             # 设计 §6
    a = run_loop(case, mode="regen", fn=teacher_oracle(case))
    b = run_loop(case, mode="regen", fn=teacher_oracle(case, noise_after_first=0.05))
    assert np.array_equal(np.array(a), np.array(b))


def test_shadow_sends_shift_bitwise(case):
    sent = run_loop(case, mode="shadow", fn=lambda p, f: f + 0.01)
    assert np.array_equal(np.array(sent), case.base["s7"].astype(np.float32))


def test_perfect_teacher_net_sends_the_teacher_commands(case):
    sent = run_loop(case, mode="regen", fn=teacher_oracle(case))
    assert np.array_equal(np.array(sent), case.teacher7.astype(np.float32))


def test_regen_stops_before_sending_a_violating_frame(case):     # Review Focus 5
    sent, rec = run_loop(case, mode="regen", fn=jump_at(frame=40, by=0.2), return_rec=True)
    assert len(sent) == 40 and "每帧变化" in rec["stopped"]


def test_stale_measurements_stop_both_modes(case):              # Review Focus 5
    for mode in ("regen", "shadow"):
        sent, rec = run_loop(case, mode=mode, fn=teacher_oracle(case), source=drop_can_after(case, t=1.5), return_rec=True)
        assert rec["stopped"] and "实测中断" in rec["stopped"] and max(rec["t_send"]) < 1.5 + 0.031 + 0.05


def test_written_files_pass_replay_result_checks(case, tmp_path):
    p_rec, p_seq = RP2.write_outputs(tmp_path, "p1_gregen1", rec_ok(case), mode="regen", net_sha="abc")
    h, z = json.loads(p_rec.read_text()), np.load(p_seq)
    assert h["cmd_hash"] == json.loads(str(z["source"]))["cmd_hash"] and len(h["t_send"]) == len(z["cmd7"])


# ── B：B4 + 写死的比例反馈（设计 `reports/PLAN_b3.md` §9）────────────────────────────────────
A0 = np.array([0.1, 0.5, -0.4, 0.0, 0.1, 0.0, 0.06])
ALPHA = 0.005                     # rad/s：实测相对理想的漂移速度


@pytest.fixture(scope="module")
def drift():
    """ACT 指令恒为 A0（理想 = 常数）；编码器 = A0 − ALPHA·(t − 1.0)，所以「理想 − 实测」= ALPHA·(t − 1.0)，线性插值下精确。
    j3 的窗恒为 0，其余为 1。网络（测试里给的）输出 ACT 指令段，所以 B4 的指令就是 A0。"""
    n, off = 60, 1000.0
    a7 = np.tile(A0, (n, 1))
    chunk = np.tile(A0, (n - 20, 12, 1))
    a_pi, _ = O.te_states(chunk)
    w = np.ones((n, 5))
    w[:, 2] = 0
    t_send = 1.0 + 0.05 * np.arange(n)
    te = np.arange(t_send[0] - 0.02, t_send[-1] + 0.05, 0.005) + 0.0025      # 采样和截止时刻错开 2.5 ms：免得取不取由浮点舍入决定
    q = A0[:6] - ALPHA * (te - 1.0)[:, None]
    q200 = np.concatenate([np.c_[te + off, np.full(len(te), g), np.degrees(q[:, list(cols)]) * 1000] for g, cols in O.CAN_GROUPS])
    ti = np.arange(t_send[0] - 0.02, t_send[-1] + 0.05, 0.01)
    imu = np.c_[ti, np.round((ti - ti[0]) * 1000), np.tile([0, 0, 9.8, 0, 0, 0], (len(ti), 1))]
    h = {"t_send": t_send, "cmd": a7.astype(np.float32), "can_q200": q200[np.argsort(q200[:, 0], kind="stable")],
         "can_hs_meta": {"clock": {"offset_realtime_minus_monotonic_s": off}}, "imu100": imu, "stopped": None, "cmd_hash": "h"}
    calib = {"R": np.eye(3).tolist(), "gyro_bias": [0.0, 0.0, 0.0], "b": [0.0, 0.0, 0.0]}
    ideal = np.tile(A0[:6], ((n - 1) * 10 + 1, 1))
    return SimpleNamespace(n=n, base={"a7": a7, "s7": a7.copy(), "w": w}, log={"a_pi": a_pi, "chunk": chunk}, calib=calib,
                           replays=[h], ideal=ideal)


def law(n: int) -> np.ndarray:
    """手推的期望补正 (n,)：第 k 帧截止前 50 ms 的 10 个网格点，前 9 个在两个采样之间（线性插值，误差精确）；最后一个正好在截止时刻，
    晚于截止前最后一个采样（早 2.5 ms），按约定保持它的值 ⇒ 平均时刻 = t_send[k] − 0.010 − 0.0225 − 0.00025，
    误差平均 = ALPHA·(0.05·k − 0.03275)（第 0 帧还没发过指令，取 0）；过 1 Hz 一阶低通（每帧 ρ = e^(−2π·1·0.05)），乘增益 0.7。"""
    rho, lp, out = np.exp(-2 * np.pi * 1.0 * 0.05), 0.0, []
    for k in range(n):
        lp = rho * lp + (1 - rho) * (ALPHA * (0.05 * k - 0.03275) if k else 0.0)
        out.append(0.7 * lp)
    return np.array(out)


def test_feedback_follows_the_fixed_law(drift):
    sent, rec = run_loop(drift, "regen", lambda p, f: f, return_rec=True, fb=drift.ideal)
    c, want = np.asarray(rec["fb_c"]), law(drift.n)
    assert c.shape == (drift.n, 5) and want.max() > 0.005
    assert np.allclose(c[:, [0, 1, 3, 4]], want[:, None], atol=1e-9) and not c[:, 2].any()      # j3 的窗为 0：不加
    u = np.array(sent, float)
    assert np.allclose(u[:, [0, 1, 4, 5]] - A0[[0, 1, 4, 5]], want[:, None], atol=1e-6) and np.allclose(u[:, [2, 3, 6]], A0[[2, 3, 6]])
    assert np.allclose(np.asarray(rec["net_u"], float), A0)                                       # 记录的 net_u 是加反馈之前的


def test_shadow_records_feedback_but_sends_shift(drift):
    sent, rec = run_loop(drift, "shadow", lambda p, f: f, return_rec=True, fb=drift.ideal)
    assert np.array_equal(np.array(sent), drift.base["s7"].astype(np.float32))
    assert np.allclose(np.asarray(rec["fb_c"])[:, 0], law(drift.n), atol=1e-9)


def test_without_feedback_nothing_is_added(drift):
    sent, rec = run_loop(drift, "regen", lambda p, f: f, return_rec=True)
    assert not np.asarray(rec["fb_c"]).any() and np.allclose(np.array(sent, float), A0)
