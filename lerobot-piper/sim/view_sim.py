#!/usr/bin/env python3
"""仿真可视化 / 截图工具 —— 展示 object-aware MuJoCo 环境(Piper + 胶带圆环 + 桌面)。

三种模式：
  交互查看(默认): 弹出 MuJoCo viewer, 鼠标旋转/缩放, 截图后关闭窗口退出。
      python sim/view_sim.py
  动画回放(--replay): 回放某 demo 动作, weld 抓取代理使夹爪拾起胶带并搬运(可录屏)。
      python sim/view_sim.py --replay --ep 24
  离屏存图(--save PATH): 无需 GUI, 渲染一帧 PNG(grasp 姿态)。
      python sim/view_sim.py --save sim_grasp.png
  其它: --no-object 只看机械臂; --ep 选 episode; --pose start|grasp 静态姿态。
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import mujoco

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sim.build_object_model import (
    load_config, derive_placement, write_object_model, grasp_interval, BASE_XML)
from sim.replay_demo_validation import load_real_demo, state_mdeg_to_rad
from sim.noisy_piper_sim import action_7_to_ctrl_8


def build(ep, with_object):
    if not with_object:
        return mujoco.MjModel.from_xml_path(str(BASE_XML)), None, None
    cfg = load_config(); demo = load_real_demo(ep)
    ring, ttop = derive_placement(BASE_XML, demo, cfg)
    out = REPO / "sim/_view_tmp.xml"; write_object_model(cfg, ring, ttop, out)
    m = mujoco.MjModel.from_xml_path(str(out)); out.unlink(missing_ok=True)
    return m, demo, cfg


def set_pose(m, d, demo, pose):
    if demo is None:
        return
    sm = demo["state_mdeg"]
    fr = 0 if pose == "start" else (grasp_interval(demo)[0])
    d.qpos[:8] = state_mdeg_to_rad(sm[fr])
    mujoco.mj_forward(m, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=24)
    ap.add_argument("--no-object", action="store_true")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--pose", choices=["start", "grasp"], default="grasp")
    args = ap.parse_args()

    m, demo, cfg = build(args.ep, not args.no_object)
    d = mujoco.MjData(m)

    # ---- 离屏存图 ----
    if args.save:
        set_pose(m, d, demo, args.pose)
        rnd = mujoco.Renderer(m, height=480, width=640)
        rnd.update_scene(d, camera=-1)
        img = rnd.render()
        from PIL import Image
        Image.fromarray(img).save(args.save)
        print(f"[saved] {args.save}  (model nq={m.nq} nbody={m.nbody})")
        return

    from mujoco import viewer as mjv
    # ---- 动画回放(weld 抓取代理) ----
    if args.replay and demo is not None:
        act = demo["action"]; ctrl8 = action_7_to_ctrl_8(act)
        at, rel, _ = grasp_interval(demo)
        d.qpos[:8] = state_mdeg_to_rad(demo["state_mdeg"][0]); mujoco.mj_forward(m, d)
        eqid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")
        l6 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link6")
        tp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tape")
        with mjv.launch_passive(m, d) as v:
            for t in range(len(act)):
                if t == at and eqid >= 0:
                    q6i = np.zeros(4); mujoco.mju_negQuat(q6i, d.xquat[l6])
                    rp = np.zeros(3); mujoco.mju_rotVecQuat(rp, d.xpos[tp]-d.xpos[l6], q6i)
                    rq = np.zeros(4); mujoco.mju_mulQuat(rq, q6i, d.xquat[tp])
                    m.eq_data[eqid, 3:6] = rp; m.eq_data[eqid, 6:10] = rq
                    d.eq_active[eqid] = 1
                if t == rel and eqid >= 0:
                    d.eq_active[eqid] = 0
                d.ctrl[:] = ctrl8[t]
                for _ in range(int(round((1/30) / m.opt.timestep))):
                    mujoco.mj_step(m, d)
                v.sync(); time.sleep(1/30)
                if not v.is_running():
                    break
        return

    # ---- 静态交互查看 ----
    set_pose(m, d, demo, args.pose)
    print("[viewer] 鼠标拖拽旋转 / 滚轮缩放；截图后关闭窗口退出。")
    mjv.launch(m, d)


if __name__ == "__main__":
    main()
