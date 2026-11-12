"""**六关节高速 CAN 反馈的被动记录器**（电机 0x251–0x256 + 关节角 0x2A5–0x2A7）。

接口与 `robot.imu_reader` 的 100 Hz 记录一致（`start_recording` /
`drain_recording` / `stop_recording`），好让 `replay_probe.py` 用同一套写法接入。

为什么被动监听而不是轮询 `GetArmHighSpdInfoMsgs()`
--------------------------------------------------
六个电机是**六条独立 CAN 帧**，SDK 把它们写进**同一个结构体**，`time_stamp`
被最后到达的那一帧覆盖(`piper_interface_v2.py:1836` 起)。在 20 Hz 控制回路里
轮询这个接口 = 对六路异步流做无抗混叠抽取，**正是我们在 TSND 上踩过的坑**
(`step2_ceiling.py`)。被动监听拿到每帧自己的内核收包时刻，**只读，不发任何帧**。

⚠ 热路径必须便宜 —— 2026-09-01 的教训
--------------------------------------
初版在后台线程里直接调 SDK 的 `PiperMessage()` + `DecodeMessage()`，
**26.4 µs/帧 × 1800 帧/s**，抢 GIL 把 20 Hz 主控制环拖成 19.3 Hz
（帧间隔 p99 从 ~54 ms 恶化到 ~85 ms，最大 103 ms）。
证据：同一序列 07-27/28 无本记录器时 7 条全部 19.91–19.92 Hz、p99 ≤ 55 ms。

现在线程只存 `(时刻, can_id, 原始字节)` —— **0.21 µs/帧，快 125 倍**，
解码全部推迟到 `stop_recording()` 之后（`decode()`）。
字节布局已与 SDK 逐位核对：`>hH`(0x251–6)、`>ii`(0x2A5–7)。

时钟口径
--------
帧时间戳 = `rx.timestamp` = 内核 `SO_TIMESTAMPNS` = **CLOCK_REALTIME**；
管线(`t`/`t_send`/TSND `host_t`)用 **CLOCK_MONOTONIC**。
`meta()` 返回二者偏移，对齐时必须用它，不能直接混。

effort 口径
-----------
落盘**原始 current**。`effort` 是 SDK 用固定系数换算的(J1–J3 ×1.18125,
J4–J6 ×0.95844)，**与固件版本无关**，而实测本机固件 S-V1.5-8 属于
「旧固件 J1–J3 另有倍率」的范围 ⇒ effort **只作激励代理，不称真实关节力矩**。
且 effort = current × 常数，不含新信息，故不落盘。
"""
from __future__ import annotations
import struct, threading, time
from collections import deque

HS_IDS = {0x251: 1, 0x252: 2, 0x253: 3, 0x254: 4, 0x255: 5, 0x256: 6}
# 关节角反馈也是 200 Hz(2026-09-01 实测)。管线一直用 GetArmJointMsgs() 在 20 Hz
# 轮询它 —— **无抗混叠**, 与 TSND 同一个缺陷。一起录, 成本为零。
Q_IDS = {0x2A5: 1, 0x2A6: 3, 0x2A7: 5}          # → 该帧承载的**首**关节号
_HS = struct.Struct(">hH")                       # motor_speed(int16), current(uint16)
_Q = struct.Struct(">ii")                        # 两个关节角, 单位 0.001 deg
# 0x251–6 的 byte4–7 恒为 0（固件不传 pos），故只解前 4 字节。


def clock_offset() -> tuple[float, float]:
    """→ (REALTIME − MONOTONIC, 读取不确定度)。"""
    m0 = time.monotonic(); r = time.time(); m1 = time.monotonic()
    return r - 0.5 * (m0 + m1), m1 - m0


def decode(raw) -> tuple[list, list]:
    """原始帧 → (can_hs, can_q200)。**离线调用**，不在控制回路里。

    can_hs   : [t_realtime, 关节 1-6, motor_speed(0.001rad/s), current(0.001A)]
    can_q200 : [t_realtime, 首关节号(1/3/5), q_a(0.001deg), q_b(0.001deg)]
    """
    hs, q2 = [], []
    for t, cid, data in raw:
        j = HS_IDS.get(cid)
        if j is not None:
            sp, cu = _HS.unpack_from(data, 0)
            hs.append((t, j, sp, cu))
        else:
            g = Q_IDS.get(cid)
            if g is not None:
                a, b = _Q.unpack_from(data, 0)
                q2.append((t, g, a, b))
    return hs, q2


class CanHighSpdRecorder:
    """电机高速 + 关节角 200 Hz 的后台记录器。**只读**。"""

    def __init__(self, channel: str = "can0", maxlen: int = 600_000):
        self.channel = channel
        self._buf: deque = deque(maxlen=maxlen)
        self._th: threading.Thread | None = None
        self._stop = threading.Event()
        self._bus = None
        self.n_dropped = 0          # maxlen 溢出时丢弃的条数
        self.err: str | None = None
        self.off0 = self.unc0 = self.off1 = None

    # ---- 与 imu_reader 对齐的三个方法 ---------------------------------
    def start_recording(self) -> bool:
        if self._th is not None:
            return True
        try:
            import can
        except Exception as e:                                   # noqa: BLE001
            self.err = f"import can 失败: {type(e).__name__}: {e}"
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
        buf, recv, stop = self._buf, self._bus.recv, self._stop
        maxlen = buf.maxlen

        def loop():
            # 热路径只做三件事: 收、判满、存原始字节。**不解码。**
            while not stop.is_set():
                try:
                    rx = recv(0.2)
                except Exception:                                # noqa: BLE001
                    break
                if rx is not None:
                    if len(buf) == maxlen:
                        self.n_dropped += 1
                    buf.append((rx.timestamp, rx.arbitration_id, bytes(rx.data)))

        self._stop.clear()
        self._th = threading.Thread(target=loop, name="can_hs_rec", daemon=True)
        self._th.start()
        return True

    def wait_first(self, timeout: float = 1.5) -> bool:
        """等第一帧到达。

        **必须调用**: SocketCAN 允许 bind 到 state DOWN 的接口，`start_recording()`
        照样返回 True，然后**静默录到零帧**。只有真收到帧才算这一路可用。
        """
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self._buf:
                return True
            time.sleep(0.02)
        self.err = (f"{timeout:.1f}s 内没有收到 0x251–0x256 / 0x2A5–0x2A7 帧 —— "
                    f"检查 {self.channel} 是否 up、机械臂是否上电")
        return False

    def drain_recording(self) -> list:
        """取出并清空**原始**缓冲 → [(t, can_id, bytes)]。逐帧排空，防环形覆盖。

        故意不在这里解码：本方法在 20 Hz 控制回路里每帧调用。解码用 `decode()`。
        """
        out = []
        b = self._buf
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
                              "motor_speed_0.001rad_s", "current_0.001A"],
            "fields_can_q200": ["t_realtime", "首关节号(1/3/5)",
                                "q_a_0.001deg", "q_b_0.001deg"],
            "note_pos": ("0x251-6 的 byte4-7 恒为 0, 固件不传 pos; "
                         "关节位置来自 0x2A5-7, 200 Hz"),
            "note_effort": ("effort = current x 固定系数, 不含新信息, 故不落盘; "
                            "本机固件 S-V1.5-8 属旧固件 J1-J3 另有倍率的范围 => "
                            "只作激励代理, 不称真实关节力矩"),
            "note_q200_provenance": (
                "2026-09-01 实测: motor_speed 与 d(q200)/dt 在 0.5-3 Hz 的 rho^2 "
                "为 0.93-0.999、3-9 Hz 在充分激励关节上 0.82-0.95 => **同源**, "
                "没有独立输出侧编码器的证据 => 200 Hz 关节角消掉的是帧间混叠, "
                "**不是**「编码器≠连杆角」"),
            "clock": {"can": "CLOCK_REALTIME (SO_TIMESTAMPNS)",
                      "pipeline": "CLOCK_MONOTONIC",
                      "offset_realtime_minus_monotonic_s": self.off0,
                      "offset_read_uncertainty_s": self.unc0,
                      "offset_at_stop_s": self.off1},
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
