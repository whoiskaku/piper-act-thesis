#!/usr/bin/env python3
"""Stage 4.2 — 快速接触标定 harness（不载 ACT，直接回放 demo 动作）。

把某个 demo 的真机动作序列(= A_clean，真机上成功抓起了胶带)回放进 object-aware
sim，报告：|accel| 峰值、抓取窗内 accel、tape 抬升量(grasp_lift)、tape 是否随夹爪
移动。用来迭代接触参数(solref/solimp/condim/friction)直到：接触稳定(peak 不爆)、
grasp_lift>0(真抓起)。几秒一轮，可 CLI 覆盖参数做 sweep。

用法:
    python sim/calib_contact.py --ep 24 --solref 0.05 1 --condim 4 --friction 1.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sim.noisy_piper_sim import NoisyPiperSim, action_7_to_ctrl_8, ctrl_8_to_action_7
from sim.replay_demo_validation import load_real_demo, state_mdeg_to_rad
from sim.build_object_model import (
    load_config, derive_placement, write_object_model, detect_grasp_frames, BASE_XML,
)

NOISE = REPO / "outputs/sim/noise_params.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=24)
    ap.add_argument("--solref", type=float, nargs=2, default=None)
    ap.add_argument("--solimp", type=float, nargs=3, default=None)
    ap.add_argument("--condim", type=int, default=None)
    ap.add_argument("--friction", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config()
    if args.solref:   cfg["contact"]["solref"] = args.solref
    if args.solimp:   cfg["contact"]["solimp"] = args.solimp
    if args.condim:   cfg["contact"]["condim"] = args.condim
    if args.friction: cfg["contact"]["friction"][0] = args.friction

    demo = load_real_demo(args.ep)
    act7 = demo["action"]                                   # (T,7) 真机动作
    qpos0 = state_mdeg_to_rad(demo["state_mdeg"][0])
    gframes = detect_grasp_frames(demo)
    ring_center, table_top = derive_placement(BASE_XML, demo, cfg)

    obj_xml = Path(REPO / "sim/_calib_tmp.xml")
    write_object_model(cfg, ring_center, table_top, obj_xml)
    sim = NoisyPiperSim(obj_xml, NOISE, seed=0)
    qadr = sim.model.jnt_qposadr[
        mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "tape_free")]

    sim.rng = np.random.default_rng(0)
    sim.reset(init_qpos=qpos0)
    tape_z0 = float(sim.data.qpos[qadr + 2])
    tape_xy0 = sim.data.qpos[qadr:qadr + 2].copy()
    imu = np.zeros((len(act7), 6)); tape_z = np.zeros(len(act7))
    ctrl8 = action_7_to_ctrl_8(act7)
    for t in range(len(act7)):
        o = sim.step(ctrl8[t], noise=False)
        imu[t] = o["imu_clean"]; tape_z[t] = float(sim.data.qpos[qadr + 2])

    an = np.linalg.norm(imu[:, :3] - imu[:, :3].mean(0), axis=1)
    lift = float(tape_z.max() - tape_z0)
    final_disp = float(np.linalg.norm(sim.data.qpos[qadr:qadr + 2] - tape_xy0))
    gwin = np.concatenate([an[max(0, g - 3):g + 10] for g in gframes]) if gframes else an

    obj_xml.unlink(missing_ok=True)
    print(f"ep{args.ep}  solref={cfg['contact']['solref']} solimp={cfg['contact']['solimp']} "
          f"condim={cfg['contact']['condim']} fric={cfg['contact']['friction'][0]}")
    print(f"  |accel| peak={an.max():.2f}  (真机靶 ~7.5)   grasp-window mean={gwin.mean():.2f} max={gwin.max():.2f}")
    print(f"  tape lift={lift*1000:.1f}mm   final xy-disp={final_disp*1000:.1f}mm   "
          f"{'✓抓起+移动' if lift>0.01 and final_disp>0.02 else '✗未成功抓取'}")


if __name__ == "__main__":
    main()
