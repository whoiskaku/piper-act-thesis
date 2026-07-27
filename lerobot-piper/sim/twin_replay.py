#!/usr/bin/env python3
"""P0.3 — 数字孪生回放: 把一条 episode 的指令流按真实时间在 MuJoCo 里播放。

显示的是「理想执行」(指令的运动学实现) — 即补正模块要把真机拉向的目标动作。
用法(在你自己的带图形界面的终端, lerobot env):
    python sim/twin_replay.py --repo enolab-guo/pickplace2_train --episode 0
    python sim/twin_replay.py --repo enolab-guo/eval_pickplace2_100k --episode 0 --speed 0.5

按键: 窗口关闭即退出; --loop 循环播放; --speed 倍速(0.5=慢放)。
"""
from __future__ import annotations
import argparse
import time

import numpy as np

from ideal_imu_kinematic import MODEL_PATH, build_spline, full_qpos, load_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--interp", choices=["cubic", "pchip"], default="cubic")
    ap.add_argument("--source", choices=["action", "state"], default="action",
                    help="action=理想(指令, 默认); state=真机实测编码器(对照用)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--headless-smoke", action="store_true",
                    help="无窗口跑 2s 验证逻辑(CI/无显示环境)")
    args = ap.parse_args()

    import mujoco
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    d = load_episode(args.repo, args.episode)
    if args.source == "action":
        cmd = d["action"]
    else:
        cmd = d["state_mdeg"] * (np.pi / 180 / 1000)
    t20 = np.arange(len(cmd)) / d["fps"]
    sp = build_spline(t20, cmd, args.interp)
    t_end = float(t20[-1])
    print(f"[twin] {args.repo} ep{args.episode} source={args.source} "
          f"T={len(cmd)}@{d['fps']:g}fps ({t_end:.1f}s) speed×{args.speed}")

    def pose_at(t_virtual: float):
        q = sp(np.clip(t_virtual, 0.0, t_end))
        data.qpos[:] = full_qpos(q)
        mujoco.mj_fwdPosition(model, data)

    if args.headless_smoke:
        n = 0
        t0 = time.monotonic()
        while (el := time.monotonic() - t0) < 2.0:
            pose_at((el * args.speed) % t_end if args.loop else el * args.speed)
            n += 1
        print(f"[twin] headless smoke OK: 2s 内步进 {n} 次 ({n/2:.0f} fps)")
        return

    import mujoco.viewer
    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            t0 = time.monotonic()
            while v.is_running():
                t_virt = (time.monotonic() - t0) * args.speed
                if t_virt > t_end:
                    break
                pose_at(t_virt)
                v.sync()
                time.sleep(0.01)  # ~100Hz 播放
            if not args.loop:
                break
        print("[twin] 播放结束/窗口关闭")


if __name__ == "__main__":
    main()
