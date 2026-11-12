"""ACT 运行时的高频记录：IMU 100 Hz、关节角 200 Hz，以及每帧指令写进总线的时刻。

老师 2026-10 的计划要求：真机运行时 IMU 100 Hz、编码器 200 Hz，时间对齐地记录。
做法照搬固定指令回放（`sim/replay_probe.py`），用到的三个部件都是现成的：
- IMU：`robot.imu_reader` 的 start/drain/stop_recording。
  每条 = [主机 monotonic 时刻, 设备 ms, ax, ay, az, gx, gy, gz]（m/s², rad/s）。
- 关节角：`sim/step3_canobs/can_hs_recorder.py`，被动只读 CAN。
  帧时刻是内核 REALTIME，和 monotonic 的偏移记在 `can_hs_meta`，对齐时必须用它。
- 发令时刻：`robot.arm.last_jointctrl_t` = JointCtrl 写总线前后的 monotonic，取中点。
控制回路里每帧只搬原始数据、不解码；CAN 解码放到集结束后的 `save()`。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "sim/step3_canobs"))
import can_hs_recorder  # noqa: E402


class HighRateLog:
    """一集的高频记录：创建即开始；每帧发令后调 `frame()`；集结束调 `save(path)`。

    由 `control_utils.record_episode` 在**每次策略运行**时创建（与是否挂补正 wrapper、
    是否开 TE 无关），存到 `<数据集目录>/high_rate/episode_<编号>.json`，编号 = 数据集的集编号。
    """

    def __init__(self, robot):
        self.robot = robot
        self.imu = getattr(robot, "imu_reader", None)
        self.rec = {"clock": "time.monotonic", "t": [], "t_send": [], "t_send_span": [],
                    "imu100": [],
                    "fields_imu100": ["host_t", "device_ms", "ax", "ay", "az", "gx", "gy", "gz"],
                    "stage_ms": [],
                    "fields_stage_ms": ["obs", "infer", "send", "record", "show", "body"]}
        if self.imu is not None:
            self.imu.start_recording()
        self.can, self.can_raw = can_hs_recorder.CanHighSpdRecorder(), []
        # wait_first 不能省：CAN 接口 DOWN 时 start 也会成功，之后静默录到零帧
        if not (self.can.start_recording() and self.can.wait_first()):
            self.can.stop_recording()
            self.rec["can_err"] = self.can.err
            self.can = None

    def frame(self):
        """每帧在 `robot.send_action()` 之后调用一次。"""
        span = getattr(getattr(self.robot, "arm", None), "last_jointctrl_t", None)
        self.rec["t"].append(time.monotonic())
        self.rec["t_send"].append(0.5 * (span[0] + span[1]) if span else float("nan"))
        self.rec["t_send_span"].append(span[1] - span[0] if span else float("nan"))
        self._drain()

    def stages(self, *seconds):
        """每帧各段耗时（存为 ms）：读观测、推理、发指令、存数据集、显示、整圈（不含等待）。
        用来查回路为什么跑不满 fps（2026-10-18 开 TE 时真机只有约 13 Hz）。"""
        self.rec["stage_ms"].append([round(s * 1e3, 2) for s in seconds])

    def _drain(self):
        if self.imu is not None:
            for ht, ms, ac, gy in self.imu.drain_recording():
                self.rec["imu100"].append([float(ht), int(ms), *map(float, ac), *map(float, gy)])
        if self.can is not None:
            self.can_raw.extend(self.can.drain_recording())

    def save(self, path, error=None):
        """排空剩余样本、停止记录、解码 CAN，写成一个 JSON。

        `error` 非空表示这一集中途出错（含 Ctrl+C），已有数据照存，标为不完整。
        记录器在写文件**之前**一定先停掉，所以写文件失败也不会留下线程和 CAN socket。
        """
        try:
            self._drain()
        finally:
            if self.imu is not None:
                self.imu.stop_recording()
            if self.can is not None:
                self.can.stop_recording()
                self.can_raw.extend(self.can.drain_recording())    # 停止前后到达的最后几帧
        self.rec["complete"] = error is None
        if error is not None:
            self.rec["error"] = error
        if self.can is not None:
            self.rec["can_hs"], self.rec["can_q200"] = can_hs_recorder.decode(self.can_raw)
            self.rec["can_hs_meta"] = self.can.meta(
                getattr(getattr(self.robot, "arm", None), "piper", None))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.rec))
