#!/usr/bin/env python3
"""订阅 /imu/data 记录到 npz；时长到或收到 SIGINT 退出。"""

import argparse
import signal
import sys
import numpy as np
import rospy
from sensor_msgs.msg import Imu


class ImuRecorder:
    def __init__(self, topic: str, output: str, duration: float, target_rate: float):
        self.output = output
        self.duration = duration
        self.samples = []
        self.min_dt = 1.0 / target_rate if target_rate > 0 else 0.0
        self._last_stamp = None
        self._start = None

        rospy.init_node("mujoco_imu_recorder", anonymous=True)
        rospy.Subscriber(topic, Imu, self._cb, queue_size=200, tcp_nodelay=True)

    def _cb(self, msg: Imu):
        t = msg.header.stamp.to_sec()
        if self._start is None:
            self._start = t
        if self._last_stamp is not None and (t - self._last_stamp) < self.min_dt:
            return
        self._last_stamp = t
        self.samples.append([
            t - self._start,
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
        ])
        if t - self._start >= self.duration:
            rospy.signal_shutdown("duration reached")

    def run(self):
        rospy.spin()
        self._save()

    def _save(self):
        if not self.samples:
            print(f"[recorder] no samples received; output not written", file=sys.stderr)
            return
        arr = np.array(self.samples, dtype=np.float64)
        np.savez(self.output, t=arr[:, 0], accel=arr[:, 1:4], gyro=arr[:, 4:7])
        print(f"[recorder] saved {len(arr)} samples to {self.output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/imu/data")
    ap.add_argument("--output", required=True)
    ap.add_argument("--duration", type=float, default=10.0, help="seconds")
    ap.add_argument("--rate", type=float, default=30.0, help="target downsample rate; 0 = no downsample")
    args = ap.parse_args()

    rec = ImuRecorder(args.topic, args.output, args.duration, args.rate)
    signal.signal(signal.SIGINT, lambda *_: rospy.signal_shutdown("sigint"))
    rec.run()


if __name__ == "__main__":
    main()
