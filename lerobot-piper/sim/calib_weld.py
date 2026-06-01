#!/usr/bin/env python3
"""Stage 4.2 — weld 约束快速标定(demo 回放, 扫 weld solref)。

relpose 修对后, weld 真刚性握住 payload, 但 solref 太刚会高频振铃污染 sustained accel_hp。
本脚本回放 demo 动作(=A_clean, 真机成功抓取轨迹), 在 attach/release 帧通断 weld,
报告 sustained accel_hp/gyro_hp(排除事件±MARGIN) + attach/release peak + 是否稳定(NaN)。
目标: 找到 weld solref 使 sustained 稳定(无振铃)且 payload 真被握住。

用法: python sim/calib_weld.py --ep 24 --weld-solref 0.02 1
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sim.noisy_piper_sim import NoisyPiperSim
from sim.replay_demo_validation import load_real_demo, state_mdeg_to_rad
from sim.build_object_model import (
    load_config, derive_placement, write_object_model, grasp_interval, BASE_XML,
)
from sim.transfer_audit_weld import _ids, _attach_weld, _sync_tape_to_gripper, execute, sustained_mask, peak_accel, MARGIN
from sim.transfer_audit import hp_rms_cols

NOISE = REPO / "outputs/sim/noise_params.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=24)
    ap.add_argument("--weld-solref", type=float, nargs=2, default=None)
    ap.add_argument("--weld-solimp", type=float, nargs=3, default=None)
    args = ap.parse_args()
    cfg = load_config()
    if args.weld_solref: cfg["weld"]["solref"] = args.weld_solref
    if args.weld_solimp: cfg["weld"]["solimp"] = args.weld_solimp

    demo = load_real_demo(args.ep)
    act7 = demo["action"]
    qpos0 = state_mdeg_to_rad(demo["state_mdeg"][0])
    attach, release, events = grasp_interval(demo)
    ring, ttop = derive_placement(BASE_XML, demo, cfg)

    obj_xml = Path(REPO / "sim/_weldcalib_tmp.xml")
    write_object_model(cfg, ring, ttop, obj_xml)
    sim = NoisyPiperSim(obj_xml, NOISE, seed=0)
    ids = _ids(sim)

    nan = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imu, tz = execute(sim, qpos0, act7, noise=False, ids=ids, attach=attach, release=release)
    if not np.isfinite(imu).all():
        nan = True
    smask = sustained_mask(len(act7), attach, release, events)
    a_hp = hp_rms_cols(imu[smask, :3]); g_hp = hp_rms_cols(imu[smask, 3:])
    ap_ = peak_accel(imu, attach - 3, attach + 10)
    rp_ = peak_accel(imu, release - 3, release + 10)
    lift = float(tz.max() - tz[0]) if tz is not None else None
    obj_xml.unlink(missing_ok=True)
    print(f"weld_solref={cfg['weld']['solref']} solimp={cfg['weld']['solimp']}  "
          f"{'⚠NaN' if nan else 'stable'}")
    print(f"  sustained accel_hp={a_hp:.4f} gyro_hp={g_hp:.4f}  "
          f"(no-object 参照 accel_hp≈0.36)   attach_peak={ap_:.2f} release_peak={rp_:.2f}  "
          f"carry_lift={lift*1000:.0f}mm")


if __name__ == "__main__":
    main()
