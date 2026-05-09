#!/usr/bin/env python3
"""TSND151 real IMU reader.

两用：
  1) 作为模块：`from tsnd151_reader import TSND151Reader`；LeRobot 录制循环里每 tick 调 `get_latest()`。
  2) 作为脚本：录制 N 秒 acc+gyro 存 npz，快速验证硬件 + 采样率 + 量纲。

硬件约定：
  - 有线 USB：/dev/ttyACM0（设备路径可 CLI 改）
  - 采样：interval_in_ms=10 → 100Hz（3× 过采样 30Hz 主循环）
  - 只用 acc_gyro（不开 quaternion；量纲最直接，后面需要四元数再加）

单位：
  - 设备原始：acc=0.1mg, gyro=0.01dps
  - 这里统一转 SI：m/s², rad/s
"""

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path
from queue import Queue, Empty

import numpy as np

# tsnd 库：位于 third_party 下
_THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party"
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))
from tsnd import TSND151  # noqa: E402

ACC_UNIT_TO_MS2 = 0.0001 * 9.80665        # 0.1mg -> m/s^2
GYRO_UNIT_TO_RADS = 0.01 * np.pi / 180.0  # 0.01dps -> rad/s


class TSND151Reader:
    """后台线程把 TSND151 流式数据收进 ring buffer，主线程随时 get_latest。"""

    def __init__(self, device_path: str = "/dev/ttyACM0",
                 interval_ms: int = 10, buffer_size: int = 2000):
        self.device_path = device_path
        self.interval_ms = interval_ms
        self.buffer = deque(maxlen=buffer_size)  # tuples: (host_t, device_ms, acc[3], gyro[3])
        self._buffer_lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread = None
        self._sensor = None
        self._queue = Queue()

    def start(self):
        self._sensor = TSND151.open(self.device_path)
        self._sensor.stop_recording()
        self._sensor.set_time()
        ok = self._sensor.set_acc_and_gyro_interval(
            interval_in_ms=self.interval_ms, avg_num_for_send=1, avg_num_for_save=0)
        if not ok:
            raise RuntimeError("set_acc_and_gyro_interval failed")
        self._sensor.set_response_queue("acc_gyro_data", self._queue)
        self._sensor.start_recording(force_restart=True)
        self._thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._thread.start()
        return self

    def _drain_loop(self):
        while not self._stop_evt.is_set():
            try:
                raw = self._queue.get(timeout=0.1)
            except Empty:
                continue
            ms, acc_raw, gyro_raw = TSND151.parse_acc_gyro(raw)
            host_t = time.monotonic()
            acc = np.array(acc_raw, dtype=np.float32) * ACC_UNIT_TO_MS2
            gyro = np.array(gyro_raw, dtype=np.float32) * GYRO_UNIT_TO_RADS
            with self._buffer_lock:
                self.buffer.append((host_t, ms, acc, gyro))

    def get_latest(self):
        """最新一条 sample；无数据返回 None。"""
        with self._buffer_lock:
            if not self.buffer:
                return None
            return self.buffer[-1]

    def get_history(self):
        """当前 ring buffer 的全部快照（拷贝）。"""
        with self._buffer_lock:
            return list(self.buffer)

    def stop(self):
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._sensor is not None:
            try:
                self._sensor.stop_recording()
                time.sleep(0.2)
            finally:
                self._sensor.close()
                self._sensor = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()


def _record_cli():
    ap = argparse.ArgumentParser(description="Standalone TSND151 recorder (validation)")
    ap.add_argument("--device", default="/dev/ttyACM0")
    ap.add_argument("--duration", type=float, default=10.0, help="seconds")
    ap.add_argument("--interval-ms", type=int, default=10, help="device interval in ms (10 -> 100Hz)")
    ap.add_argument("--output", default="./tsnd_test.npz")
    args = ap.parse_args()

    print(f"[tsnd] open {args.device}, interval={args.interval_ms}ms "
          f"(expected {1000.0/args.interval_ms:.1f} Hz)")

    with TSND151Reader(args.device, interval_ms=args.interval_ms) as rd:
        t_start = time.monotonic()
        next_report = t_start + 1.0
        while True:
            now = time.monotonic()
            if now - t_start >= args.duration:
                break
            if now >= next_report:
                latest = rd.get_latest()
                n = len(rd.buffer)
                if latest is not None:
                    _, ms, a, g = latest
                    print(f"[{now-t_start:5.1f}s] buf={n:4d} last ms={ms} "
                          f"acc={a.round(2).tolist()} gyro={g.round(3).tolist()}")
                else:
                    print(f"[{now-t_start:5.1f}s] buf=0 (no data yet)")
                next_report = now + 1.0
            time.sleep(0.05)

        history = rd.get_history()

    if not history:
        print("[tsnd] no samples collected — check device path / wiring", file=sys.stderr)
        sys.exit(1)

    host_t = np.array([h[0] for h in history], dtype=np.float64)
    device_ms = np.array([h[1] for h in history], dtype=np.int64)
    accel = np.stack([h[2] for h in history]).astype(np.float32)
    gyro = np.stack([h[3] for h in history]).astype(np.float32)

    # 实际采样率（用设备 ms）
    if len(device_ms) > 1:
        dt_ms = np.diff(device_ms.astype(np.float64))
        # 处理可能的 ms 回绕（不应该，但安全起见）
        dt_ms = dt_ms[dt_ms > 0]
        actual_hz = 1000.0 / np.mean(dt_ms) if len(dt_ms) > 0 else float("nan")
    else:
        actual_hz = float("nan")

    out_path = Path(args.output).expanduser().resolve()
    np.savez(out_path,
             host_t=host_t, device_ms=device_ms, accel=accel, gyro=gyro,
             interval_ms=args.interval_ms)
    print(f"\n[tsnd] collected {len(history)} samples in {args.duration:.1f}s")
    print(f"[tsnd] actual rate (by device ms) = {actual_hz:.2f} Hz")
    print(f"[tsnd] accel range: x=[{accel[:,0].min():.2f},{accel[:,0].max():.2f}] "
          f"y=[{accel[:,1].min():.2f},{accel[:,1].max():.2f}] "
          f"z=[{accel[:,2].min():.2f},{accel[:,2].max():.2f}]  (m/s^2, gravity≈9.81 应出现在某一轴)")
    print(f"[tsnd] gyro  range: x=[{gyro[:,0].min():.3f},{gyro[:,0].max():.3f}] "
          f"y=[{gyro[:,1].min():.3f},{gyro[:,1].max():.3f}] "
          f"z=[{gyro[:,2].min():.3f},{gyro[:,2].max():.3f}]  (rad/s)")
    print(f"[tsnd] saved -> {out_path}")


if __name__ == "__main__":
    _record_cli()
