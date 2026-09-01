"""**六关节高速 CAN 反馈的被动记录器**（0x251–0x256）。

接口刻意与 `robot.imu_reader` 的 100 Hz 记录一致(`start_recording` /
`drain_recording` / `stop_recording`)，好让 `replay_probe.py` 用同一套写法接入。

为什么被动监听而不是轮询 `GetArmHighSpdInfoMsgs()`
--------------------------------------------------
六个电机是**六条独立 CAN 帧**，SDK 把它们写进**同一个结构体**，`time_stamp`
被最后到达的那一帧覆盖(`piper_interface_v2.py:1836` 起)。在 20 Hz 控制回路里
轮询这个接口 = 对六路异步流做无抗混叠抽取，**正是我们在 TSND 上踩过的坑**
(`step2_ceiling.py`)。被动监听拿到每帧自己的内核收包时刻，且不与控制回路争用。

SocketCAN 允许同一接口上多个 socket 并存，因此本记录器与 SDK 的接收线程互不干扰，
**只读，不发任何帧**。

时钟口径
--------
帧时间戳 = `rx.timestamp` = 内核 `SO_TIMESTAMPNS` = **CLOCK_REALTIME**；
管线(`t`/`t_send`/TSND `host_t`)用 **CLOCK_MONOTONIC**。
`meta()` 返回二者偏移，对齐时必须用它，不能直接混。

effort 口径
-----------
落盘**原始 current**；`effort` 由 SDK 用固定系数换算(J1–J3 ×1.18125,
J4–J6 ×0.95844)，**与固件版本无关**，而旧固件(1.8-2 及以前) J1–J3 另有倍率规则。
故 effort **只作激励代理，不称真实关节力矩**。
"""
from __future__ import annotations
import threading, time
from collections import deque

HS_IDS = {0x251: 1, 0x252: 2, 0x253: 3, 0x254: 4, 0x255: 5, 0x256: 6}
# 关节角反馈也是 200 Hz(2026-09-01 实测)。管线一直用 GetArmJointMsgs() 在 20 Hz
# 轮询它 —— **无抗混叠**, 与 TSND 同一个缺陷。一起录, 成本为零。
Q_IDS = {0x2A5: ("joint_1", "joint_2"), 0x2A6: ("joint_3", "joint_4"),
         0x2A7: ("joint_5", "joint_6")}


def clock_offset() -> tuple[float, float]:
    """→ (REALTIME − MONOTONIC, 读取不确定度)。"""
    m0 = time.monotonic(); r = time.time(); m1 = time.monotonic()
    return r - 0.5 * (m0 + m1), m1 - m0


class CanHighSpdRecorder:
    """六关节高速反馈的后台记录器。**只读**。"""

    def __init__(self, channel: str = "can0", maxlen: int = 400_000):
        self.channel = channel
        self._buf: deque = deque(maxlen=maxlen)
        self._qbuf: deque = deque(maxlen=maxlen)
        self._th: threading.Thread | None = None
        self._stop = threading.Event()
        self._bus = None
        self.n_dropped = 0          # maxlen 溢出时丢弃的条数
        self.err: str | None = None
        self.off0 = self.unc0 = None
        self.off1 = None

    # ---- 与 imu_reader 对齐的三个方法 ---------------------------------
    def start_recording(self) -> bool:
        if self._th is not None:
            return True
        try:
            import can
            from piper_sdk.protocol.protocol_v2.piper_protocol_v2 import C_PiperParserV2
            from piper_sdk.piper_msgs.msg_v2.arm_messages import PiperMessage
        except Exception as e:                                   # noqa: BLE001
            self.err = f"import 失败: {type(e).__name__}: {e}"
            return False
        try:
            self._bus = can.interface.Bus(
                channel=self.channel, interface="socketcan",
                can_filters=[{"can_id": i, "can_mask": 0x7FF}
                             for i in (*HS_IDS, *Q_IDS)])
        except Exception as e:                                   # noqa: BLE001
            self.err = f"打开 {self.channel} 失败: {type(e).__name__}: {e}"
            return False
        self.off0, self.unc0 = clock_offset()
        parser, Msg = C_PiperParserV2(), PiperMessage

        def loop():
            while not self._stop.is_set():
                try:
                    rx = self._bus.recv(timeout=0.2)
                except Exception:                                # noqa: BLE001
                    break
                if rx is None:
                    continue
                cid = rx.arbitration_id
                j = HS_IDS.get(cid)
                if j is None and cid not in Q_IDS:
                    continue
                m = Msg()
                if not parser.DecodeMessage(rx, m):
                    continue
                if j is not None:
                    fb = getattr(m, f"arm_high_spd_feedback_{j}")
                    # pos 不记: 0x251-6 的 byte4-7 恒为 0, 固件根本不传(2026-09-01 实测)。
                    # effort 也不记: = current x 固定系数, 不含新信息。
                    if len(self._buf) == self._buf.maxlen:
                        self.n_dropped += 1
                    self._buf.append((float(rx.timestamp), j,
                                      int(fb.motor_speed), int(fb.current)))
                else:
                    js = m.arm_joint_feedback
                    if len(self._qbuf) == self._qbuf.maxlen:
                        self.n_dropped += 1
                    a, b = Q_IDS[cid]
                    self._qbuf.append((float(rx.timestamp), int(a[-1]),
                                       int(getattr(js, a)), int(getattr(js, b))))

        self._stop.clear()
        self._th = threading.Thread(target=loop, name="can_hs_rec", daemon=True)
        self._th.start()
        return True

    def wait_first(self, timeout: float = 1.5) -> bool:                # noqa: D401
        """等第一帧到达。

        **必须调用**: SocketCAN 允许 bind 到 state DOWN 的接口，`start_recording()`
        照样返回 True，然后**静默录到零帧**。只有真收到帧才算这一路可用。
        """
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self._buf:
                return True
            time.sleep(0.02)
        self.err = (f"{timeout:.1f}s 内没有收到 0x251–0x256 帧 —— "
                    f"检查 {self.channel} 是否 up、机械臂是否上电")
        return False

    def drain_recording(self) -> list:
        """取出并清空**电机高速**缓冲。逐帧排空，防环形覆盖（与 imu100 同策略）。"""
        out = []
        b = self._buf
        while b:
            out.append(b.popleft())
        return out

    def drain_q200(self) -> list:
        """取出并清空 **200 Hz 关节角**缓冲 → [(t_realtime, 首关节号, qa, qb)]。"""
        out = []
        b = self._qbuf
        while b:
            out.append(b.popleft())
        return out

    def stop_recording(self) -> None:
        self.off1, _ = clock_offset()
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=1.0)
            self._th = None
        if self._bus is not None:
            try:
                self._bus.shutdown()
            except Exception:                                    # noqa: BLE001
                pass
            self._bus = None

    # ---- 口径元数据 ----------------------------------------------------
    def meta(self, piper=None) -> dict:
        d = {
            "channel": self.channel,
            "fields_can_hs": ["t_realtime", "joint(1-6)",
                              "motor_speed_0.001rad_s(电机侧, 减速前)",
                              "current_0.001A"],
            "fields_can_q200": ["t_realtime", "首关节号(1/3/5)",
                                "q_a_0.001deg", "q_b_0.001deg"],
            "note_q200_provenance": (
                "**未定**: 0x2A5-7 的关节角是电机编码器除以减速比得来的, "
                "还是独立的输出侧编码器, 静止测试无法区分(静止时它零变化)。"
                "运动集上可判: 若 d(q)/dt x N 与 motor_speed 在量化精度内一致 "
                "=> 电机侧推算, **看不见传动柔性**; 否则是独立输出侧。"
                "在判定前不得声称它是连杆侧"),
            "note_pos": ("0x251-6 的 byte4-7 恒为 0, 固件不传 pos; "
                         "关节位置来自 0x2A5-7, 200 Hz, 连杆侧"),
            "note_effort": "effort = current x 固定系数, 不含新信息, 故不落盘",
            "clock": {"can": "CLOCK_REALTIME (SO_TIMESTAMPNS)",
                      "pipeline": "CLOCK_MONOTONIC",
                      "offset_realtime_minus_monotonic_s": self.off0,
                      "offset_read_uncertainty_s": self.unc0,
                      "offset_at_stop_s": self.off1},
            "effort_caliber": ("落盘原始 current；effort = SDK 固定系数换算"
                               "(J1-J3 x1.18125, J4-J6 x0.95844)，与固件版本无关。"
                               "旧固件(<=1.8-2) J1-J3 另有倍率规则 ⇒ "
                               "**effort 只作激励代理，不称真实关节力矩**"),
            "n_dropped_overflow": self.n_dropped,
            "err": self.err,
        }
        if piper is not None:
            for k, fn in (("firmware", "GetPiperFirmwareVersion"),
                          ("sdk", "GetCurrentSDKVersion"),
                          ("protocol", "GetCurrentProtocolVersion"),
                          ("can_fps", "GetCanFps")):
                try:
                    v = getattr(piper, fn)()
                    d[k] = v if isinstance(v, (int, float, str)) else str(v)
                except Exception as e:                           # noqa: BLE001
                    d[k] = f"<{type(e).__name__}>"
        return d
