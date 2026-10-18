"""`HighRateLog`（ACT 运行时的高频记录）及其在 `record_episode` 里的接入的契约测试。

单元测试按文件路径加载模块，不导入 LeRobot 顶层包；接入测试导入 `control_utils`
（需要 lerobot 环境）。IMU、CAN 记录器、机器人、数据集、策略都用假的，不需要硬件。
"""
from __future__ import annotations

import importlib.util
import json
import math
import struct
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "_high_rate_log_test", _ROOT / "lerobot/common/robot_devices/high_rate_log.py")
hrl = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = hrl
_SPEC.loader.exec_module(hrl)


class FakeIMU:
    def __init__(self):
        self.calls, self.pending = [], []

    def start_recording(self):
        self.calls.append("start")

    def drain_recording(self):
        out, self.pending = self.pending, []
        return out

    def stop_recording(self):
        self.calls.append("stop")


class FakeCan:
    made = []

    def __init__(self, channel="can0"):
        self.err, self.buf, self.stopped = None, [], False
        FakeCan.made.append(self)

    def start_recording(self):
        return True

    def wait_first(self, timeout=1.5):
        return True

    def drain_recording(self):
        out, self.buf = self.buf, []
        return out

    def stop_recording(self):
        self.stopped = True

    def meta(self, piper=None):
        return {"channel": "can0"}


class FakeArm:
    last_jointctrl_t = None


class FakeRobot:
    def __init__(self, imu):
        self.imu_reader, self.arm = imu, FakeArm()


def _imu_row(t, ms, gz):
    return (t, ms, [0.0, 0.0, 9.8], [0.0, 0.0, gz])


@pytest.fixture
def fake_can(monkeypatch):
    FakeCan.made = []
    monkeypatch.setattr(hrl.can_hs_recorder, "CanHighSpdRecorder", FakeCan)


# ── HighRateLog 本身 ─────────────────────────────────────────────────────

def test_frames_imu_and_can_are_recorded_and_saved(tmp_path, fake_can):
    imu = FakeIMU()
    robot = FakeRobot(imu)
    log = hrl.HighRateLog(robot)
    assert imu.calls == ["start"]

    robot.arm.last_jointctrl_t = (10.000, 10.002)
    imu.pending = [_imu_row(9.99, 1000, 0.1)]
    log.can.buf = [(1e9 + 0.001, 0x2A5, struct.pack(">ii", 1000, -2000))]
    log.frame()
    robot.arm.last_jointctrl_t = (10.050, 10.051)
    imu.pending = [_imu_row(10.04, 1010, 0.2)]
    log.can.buf = [(1e9 + 0.05, 0x251, struct.pack(">hH", 5, 7) + bytes(4))]
    log.frame()
    imu.pending = [_imu_row(10.09, 1020, 0.3)]      # 最后一帧之后才到的样本，save 时也要排空
    can = log.can

    out = tmp_path / "episode_000000.json"
    log.save(out)
    d = json.loads(out.read_text())

    assert d["complete"] is True and "error" not in d
    assert len(d["t"]) == 2
    assert d["t_send"] == pytest.approx([10.001, 10.0505])
    assert d["t_send_span"] == pytest.approx([0.002, 0.001])
    assert [r[1] for r in d["imu100"]] == [1000, 1010, 1020]
    assert d["imu100"][0] == pytest.approx([9.99, 1000, 0.0, 0.0, 9.8, 0.0, 0.0, 0.1])
    assert d["can_q200"] == [[pytest.approx(1e9 + 0.001), 1, 1000, -2000]]
    assert d["can_hs"] == [[pytest.approx(1e9 + 0.05), 1, 5, 7]]
    assert d["can_hs_meta"] == {"channel": "can0"}
    assert imu.calls == ["start", "stop"] and can.stopped


def test_can_without_frames_is_stopped_and_reported_but_imu_still_recorded(tmp_path, monkeypatch):
    class SilentCan(FakeCan):            # 接口 DOWN：start 成功但收不到帧
        def wait_first(self, timeout=1.5):
            self.err = "1.5s 内没有收到帧"
            return False

    FakeCan.made = []
    monkeypatch.setattr(hrl.can_hs_recorder, "CanHighSpdRecorder", SilentCan)
    imu = FakeIMU()
    log = hrl.HighRateLog(FakeRobot(imu))
    assert log.can is None and FakeCan.made[0].stopped

    imu.pending = [_imu_row(1.0, 1, 0.0)]
    log.frame()
    log.save(tmp_path / "x.json")
    d = json.loads((tmp_path / "x.json").read_text())

    assert d["can_err"] == "1.5s 内没有收到帧"
    assert "can_q200" not in d
    assert len(d["imu100"]) == 1


def test_missing_imu_reader_and_send_time_give_nan_not_crash(tmp_path, fake_can):
    log = hrl.HighRateLog(FakeRobot(None))      # 没有 IMU；arm 也没有发令时刻
    log.frame()
    log.save(tmp_path / "y.json")
    d = json.loads((tmp_path / "y.json").read_text())

    assert d["imu100"] == []
    assert math.isnan(d["t_send"][0]) and math.isnan(d["t_send_span"][0])


def test_save_with_error_marks_incomplete(tmp_path, fake_can):
    imu = FakeIMU()
    log = hrl.HighRateLog(FakeRobot(imu))
    log.frame()
    log.save(tmp_path / "z.json", error="KeyboardInterrupt: ")
    d = json.loads((tmp_path / "z.json").read_text())

    assert d["complete"] is False and d["error"] == "KeyboardInterrupt: "
    assert imu.calls == ["start", "stop"] and FakeCan.made[0].stopped


def test_write_failure_still_stops_recorders(tmp_path, fake_can):
    imu = FakeIMU()
    log = hrl.HighRateLog(FakeRobot(imu))
    blocker = tmp_path / "blocker"
    blocker.write_text("")                      # 让「父目录」其实是个文件 ⇒ 写不进去
    with pytest.raises(OSError):
        log.save(blocker / "w.json")
    assert imu.calls == ["start", "stop"] and FakeCan.made[0].stopped


# ── 接入 record_episode（需要 lerobot 环境）──────────────────────────────

def _record(cu, robot, root):
    """以「数据集第 7 集」调一次 record_episode；策略是原生的，没有补正 wrapper 的任何属性。"""
    dataset = SimpleNamespace(root=root, meta=SimpleNamespace(total_episodes=7))
    policy = SimpleNamespace(reset=lambda: None)
    cu.record_episode(robot=robot, dataset=dataset, events={}, episode_time_s=1,
                      display_cameras=False, policy=policy, device="cpu", use_amp=False,
                      fps=20, single_task="t")


@pytest.fixture
def cu(fake_can):                              # fake_can：靠它先把 CAN 记录器换成假的
    from lerobot.common.robot_devices import control_utils
    return control_utils


def test_native_policy_is_logged_under_dataset_episode_index(tmp_path, monkeypatch, cu):
    def loop(**kw):
        for _ in range(3):
            kw["high_rate_log"].frame()

    monkeypatch.setattr(cu, "control_loop", loop)
    imu = FakeIMU()
    _record(cu, FakeRobot(imu), tmp_path)
    d = json.loads((tmp_path / "high_rate" / "episode_000007.json").read_text())

    assert d["complete"] is True and len(d["t"]) == 3
    assert imu.calls == ["start", "stop"] and FakeCan.made[-1].stopped


def test_interrupt_saves_incomplete_log_closes_recorders_and_reraises(tmp_path, monkeypatch, cu):
    def loop(**kw):
        kw["high_rate_log"].frame()
        raise KeyboardInterrupt

    monkeypatch.setattr(cu, "control_loop", loop)
    imu = FakeIMU()
    with pytest.raises(KeyboardInterrupt):
        _record(cu, FakeRobot(imu), tmp_path)
    d = json.loads((tmp_path / "high_rate" / "episode_000007.json").read_text())

    assert d["complete"] is False and d["error"].startswith("KeyboardInterrupt")
    assert len(d["t"]) == 1
    assert imu.calls == ["start", "stop"] and FakeCan.made[-1].stopped


def test_control_loop_logs_stage_times_per_frame(cu):
    """真的 control_loop：每帧记一行各段耗时；推理段量到的是 predict_action；各段相加 = 整圈。"""
    import torch
    events = {"exit_early": False}

    class LoopRobot(FakeRobot):
        is_connected, robot_type, n = True, "piper", 0

        def capture_observation(self):
            return {"observation.state": torch.zeros(7)}

        def send_action(self, action):
            self.n += 1
            events["exit_early"] = self.n >= 3        # 跑 3 帧就结束
            return action

    class SlowPolicy:                                  # 推理固定花 20 ms
        def select_action(self, batch):
            time.sleep(0.02)
            return torch.zeros(1, 7)

    robot = LoopRobot(FakeIMU())
    log = hrl.HighRateLog(robot)
    cu.control_loop(robot=robot, events=events, policy=SlowPolicy(), device="cpu",
                    use_amp=False, high_rate_log=log)

    rows = log.rec["stage_ms"]
    assert len(rows) == len(log.rec["t"]) == 3
    for obs, infer, send, record, show, body in rows:
        assert min(obs, infer, send, record, show) >= 0
        assert infer >= 20
        assert body == pytest.approx(obs + infer + send + record + show, abs=0.05)


def test_save_failure_during_interrupt_does_not_hide_the_interrupt(tmp_path, monkeypatch, cu):
    def loop(**_):
        raise KeyboardInterrupt

    monkeypatch.setattr(cu, "control_loop", loop)
    root = tmp_path / "not_a_dir"
    root.write_text("")                         # 数据集目录其实是个文件 ⇒ 存盘失败
    imu = FakeIMU()
    with pytest.raises(KeyboardInterrupt):      # 抛出来的仍是原来的 Ctrl+C，不是存盘错误
        _record(cu, FakeRobot(imu), root)
    assert imu.calls == ["start", "stop"] and FakeCan.made[-1].stopped
