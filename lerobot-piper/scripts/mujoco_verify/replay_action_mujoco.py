#!/usr/bin/env python3
"""读 .npy 的 action 序列 (T, J)，以 --rate Hz 发布 JointState 到 --topic。

action 列顺序默认：joint1..joint6, joint7。joint8 由 MuJoCo 端镜像 joint7，不必提供。
允许 J=6（只臂）或 J=7（带 gripper）。
"""

import argparse
import numpy as np
import rospy
from sensor_msgs.msg import JointState


JOINT_NAMES_6 = [f"joint{i}" for i in range(1, 7)]
JOINT_NAMES_7 = JOINT_NAMES_6 + ["joint7"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", required=True, help=".npy file, shape (T, 6) or (T, 7)")
    ap.add_argument("--topic", default="/joint_states",
                    help="launch 文件里 remap 过，默认发 /joint_states；裸节点用 /mujoco_joint_states_ctrl")
    ap.add_argument("--rate", type=float, default=30.0)
    ap.add_argument("--hold-end-s", type=float, default=0.5, help="finished 后保持末态秒数，便于 IMU 收尾采样")
    args = ap.parse_args()

    arr = np.load(args.actions)
    assert arr.ndim == 2 and arr.shape[1] in (6, 7), f"actions shape must be (T,6) or (T,7); got {arr.shape}"
    names = JOINT_NAMES_6 if arr.shape[1] == 6 else JOINT_NAMES_7

    rospy.init_node("replay_action_mujoco", anonymous=True)
    pub = rospy.Publisher(args.topic, JointState, queue_size=10, tcp_nodelay=True)
    rate = rospy.Rate(args.rate)

    # 等订阅者上线
    t0 = rospy.Time.now()
    while pub.get_num_connections() == 0 and (rospy.Time.now() - t0).to_sec() < 3.0:
        rospy.sleep(0.05)

    print(f"[replay] publishing {arr.shape[0]} frames @ {args.rate} Hz to {args.topic}")
    for i, row in enumerate(arr):
        if rospy.is_shutdown():
            break
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = names
        msg.position = list(row.astype(float))
        pub.publish(msg)
        rate.sleep()

    # 末态保持
    if args.hold_end_s > 0 and not rospy.is_shutdown():
        end_time = rospy.Time.now() + rospy.Duration.from_sec(args.hold_end_s)
        while rospy.Time.now() < end_time and not rospy.is_shutdown():
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = names
            msg.position = list(arr[-1].astype(float))
            pub.publish(msg)
            rate.sleep()

    print("[replay] done")


if __name__ == "__main__":
    main()
