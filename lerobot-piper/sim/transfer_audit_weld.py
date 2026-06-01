#!/usr/bin/env python3
"""Stage 4.0 — Object payload transfer audit（weld attachment proxy）。

⚠️ weld = oracle grasp ATTACHMENT 代理，**不是**真实抓取接触物理。用它稳定、可复现地
测量 61g 偏心负载(环心 COM)对 link6 IMU 的影响，以及 action smoothing 在带载下的 transfer。

四条约束(按拍板)：
  1. attach 前同步 tape pose+velocity(匹配 gripper 速度)再启用 weld → 避免修正力假 spike。
  2. 固定 grasp/release 帧(demo 事件)，ACT 与 refined 共用同一区间。
  3. sustained-load 窗(排除 attach/release/中间事件 ±MARGIN)测偏心负载；
     event 窗单独报 attach/release spike(不当真实接触峰值)。
  4. weld 的 solref/solimp 单独配置(见 config.weld)。

条件：no / yes+weld(payload) / yes+free(失败基线)。
核心：ΔIMU_clean & ΔIMU_noisy (sustained, ACT−refined) ；object=yes(weld) vs no ；event peak 单列。

用法:  python sim/transfer_audit_weld.py [--n-eps 3]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.act.refine_net import RefineNet
from sim.noisy_piper_sim import NoisyPiperSim, action_7_to_ctrl_8, ctrl_8_to_action_7
from sim.replay_demo_validation import load_real_demo, state_mdeg_to_rad
from sim.precompute_act_output import load_frozen_act
from sim.deploy_eval_phase1 import C1_CKPT, DATASET_REPO, REFINE_CKPT, NOISE_PARAMS, MODEL_XML
from sim.transfer_audit import build_policy_trajs, hp_rms_cols
from sim.build_object_model import (
    load_config, derive_placement, write_object_model, grasp_interval, BASE_XML,
)

OUT_DIR = REPO / "outputs/eval/stage4_transfer_audit"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MARGIN = 8          # 帧；sustained 窗排除事件附近 ±MARGIN
SIM_SEED = 12345


def _ids(sim):
    m = sim.model
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "tape_free")
    return {
        "qadr": m.jnt_qposadr[jid], "dofadr": m.jnt_dofadr[jid],
        "link6": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link6"),
        "tape": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tape"),
        "weld": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld"),
    }


def _attach_weld(sim, ids):
    """把 weld relpose 设为 attach 瞬间的当前相对位姿(否则默认锁 reset 位姿→假 spike)。"""
    m, d = sim.model, sim.data
    q6inv = np.zeros(4); mujoco.mju_negQuat(q6inv, d.xquat[ids["link6"]])
    relpos = np.zeros(3)
    mujoco.mju_rotVecQuat(relpos, d.xpos[ids["tape"]] - d.xpos[ids["link6"]], q6inv)
    relquat = np.zeros(4); mujoco.mju_mulQuat(relquat, q6inv, d.xquat[ids["tape"]])
    eqid = ids["weld"]
    m.eq_data[eqid, 0:3] = 0.0          # anchor = body1 origin
    m.eq_data[eqid, 3:6] = relpos       # relpose 位置 = 当前相对
    m.eq_data[eqid, 6:10] = relquat     # relpose 朝向 = 当前相对


def _sync_tape_to_gripper(sim, ids):
    """attach 前：把 tape 的速度同步到 gripper 在 tape 处的速度(避免 weld 修正冲量)。"""
    m, d = sim.model, sim.data
    vel6 = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, ids["link6"], vel6, 0)
    w, v = vel6[:3], vel6[3:]                                  # world: [rot, lin]
    r = d.qpos[ids["qadr"]:ids["qadr"] + 3] - d.xpos[ids["link6"]]
    v_com = v + np.cross(w, r)
    dof = ids["dofadr"]
    d.qvel[dof:dof + 3] = v_com
    d.qvel[dof + 3:dof + 6] = w                                # tape 初始未旋转 → local≈world


def execute(sim, qpos0_8, traj_7, noise, ids=None, attach=None, release=None):
    """执行 (T,7) 指令轨迹。weld 模式: 传 ids+attach+release(yes+weld); 否则自由/无物体。
    返回 imu(T,6), tape_z(T or None)。"""
    sim.rng = np.random.default_rng(SIM_SEED)
    sim.reset(init_qpos=qpos0_8)
    if ids is not None and attach is not None:
        sim.data.eq_active[ids["weld"]] = 0
    ctrl8 = action_7_to_ctrl_8(traj_7)
    imu = np.zeros((len(traj_7), 6), dtype=np.float32)
    tape_z = np.zeros(len(traj_7), dtype=np.float32) if ids is not None else None
    for t in range(len(traj_7)):
        if ids is not None and attach is not None:
            if t == attach:
                _attach_weld(sim, ids)            # relpose = 当前相对位姿
                _sync_tape_to_gripper(sim, ids)   # 速度匹配 gripper
                sim.data.eq_active[ids["weld"]] = 1
            elif t == release:
                sim.data.eq_active[ids["weld"]] = 0
        obs = sim.step(ctrl8[t], noise=noise)
        imu[t] = obs["imu_noisy"] if noise else obs["imu_clean"]
        if tape_z is not None:
            tape_z[t] = float(sim.data.qpos[ids["qadr"] + 2])
    return imu, tape_z


def sustained_mask(T, attach, release, events):
    m = np.zeros(T, dtype=bool)
    m[attach + MARGIN: release - MARGIN] = True
    for e in events:
        m[max(0, e - MARGIN): e + MARGIN] = False
    return m


def peak_accel(imu, lo, hi):
    seg = imu[max(0, lo):hi, :3]
    if len(seg) < 2:
        return 0.0
    return float(np.linalg.norm(seg - seg.mean(0), axis=1).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eps", type=int, default=3)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    from sim.build_object_model import object_mass_kg
    print(f"[cfg] tape {cfg['object']['d_outer_mm']}/{cfg['object']['d_inner_mm']}mm "
          f"mass={object_mass_kg(cfg)*1000:.0f}g  weld solref={cfg['weld']['solref']}  "
          f"contact solref={cfg['contact']['solref']}")
    print("[note] weld = oracle attachment proxy, NOT physically-simulated grasp contact.")

    policy, tcfg = load_frozen_act(C1_CKPT, DATASET_REPO, DEVICE)
    dataset = make_dataset(tcfg)
    ep_from = dataset.episode_data_index["from"].numpy()
    ckpt = torch.load(REFINE_CKPT, map_location=DEVICE, weights_only=False)
    refine = RefineNet().to(DEVICE).eval(); refine.load_state_dict(ckpt["model_state"])
    smean = np.asarray(ckpt["state_mean"], np.float32); sstd = np.asarray(ckpt["state_std"], np.float32)

    sim_base = NoisyPiperSim(MODEL_XML, NOISE_PARAMS, seed=SIM_SEED)
    POLICIES = ["ACT", "refined+blend"]
    NOISES = [("clean", False), ("noisy", True)]
    # accumulators: (cond, noise, policy) -> list of sustained accel_hp+gyro_hp ; event peaks
    acc = {}
    ev = {}

    for ep in range(24, 24 + args.n_eps):
        demo = load_real_demo(ep)
        attach, release, events = grasp_interval(demo)
        T = len(demo["action"])
        _, qpos0, traj_act, traj_blend = build_policy_trajs(
            policy, dataset, ep_from, refine, smean, sstd, ep)
        trajs = {"ACT": traj_act, "refined+blend": traj_blend}
        smask = sustained_mask(T, attach, release, events)

        # object sim for this episode
        cube_xyz, table_top = derive_placement(MODEL_XML, demo, cfg)
        obj_xml = Path(tempfile.mktemp(suffix=".xml", dir=str(MODEL_XML.parent)))
        write_object_model(cfg, cube_xyz, table_top, obj_xml)
        sim_obj = NoisyPiperSim(obj_xml, NOISE_PARAMS, seed=SIM_SEED)
        ids = _ids(sim_obj)

        for noise_name, noise in NOISES:
            for pol in POLICIES:
                traj = trajs[pol]
                # no object
                imu_n, _ = execute(sim_base, qpos0, traj, noise)
                # yes + weld payload
                imu_w, tz_w = execute(sim_obj, qpos0, traj, noise, ids, attach, release)
                # yes + free (failure baseline) — 仅 clean+ACT 跑一次省时
                for cond, imu in [("no", imu_n), ("weld", imu_w)]:
                    k = (cond, noise_name, pol)
                    acc.setdefault(k, []).append(
                        (hp_rms_cols(imu[smask, :3]), hp_rms_cols(imu[smask, 3:])))
                # event peaks (weld only)
                ke = (noise_name, pol)
                ev.setdefault(ke, []).append(
                    (peak_accel(imu_w, attach - 3, attach + 10),
                     peak_accel(imu_w, release - 3, release + 10)))
        obj_xml.unlink(missing_ok=True)
        print(f"  [ep{ep}] attach={attach} release={release} events={events} done")

    def m(k, idx):
        return float(np.mean([x[idx] for x in acc[k]]))

    print(f"\n{'='*96}")
    print("Stage 4.0 — payload transfer audit (weld proxy)  sustained-load 窗 (排除事件±%d帧)" % MARGIN)
    print(f"{'='*96}")
    print(f"{'cond':>6} {'noise':>6} {'policy':>14} {'accel_hp':>9} {'gyro_hp':>9}")
    print("-" * 96)
    rep = {}
    for cond in ["no", "weld"]:
        for nn, _ in NOISES:
            for pol in POLICIES:
                k = (cond, nn, pol)
                a, g = m(k, 0), m(k, 1)
                rep["/".join(k)] = {"accel_hp": a, "gyro_hp": g}
                print(f"{cond:>6} {nn:>6} {pol:>14} {a:>9.5f} {g:>9.5f}")
    print("-" * 96)
    # 核心判据
    for nn, _ in NOISES:
        a_act = rep[f"weld/{nn}/ACT"]; a_ref = rep[f"weld/{nn}/refined+blend"]
        di_act = a_act["accel_hp"] + a_act["gyro_hp"]
        di_ref = a_ref["accel_hp"] + a_ref["gyro_hp"]
        print(f"  [weld {nn}] ΔIMU sustained = ACT−refined = {di_act-di_ref:+.5f} "
              f"(ACT={di_act:.5f} refined={di_ref:.5f}, ratio={di_ref/di_act:.3f})")
    for nn, _ in NOISES:
        no = rep[f"no/{nn}/ACT"]; yes = rep[f"weld/{nn}/ACT"]
        ni = no["accel_hp"] + no["gyro_hp"]; yi = yes["accel_hp"] + yes["gyro_hp"]
        print(f"  [object yes(weld) vs no, {nn}, ACT] IMU = {yi:.5f} vs {ni:.5f}  "
              f"(ratio={yi/ni:.3f}) {'← 物体几乎不改变IMU(臂主导)' if abs(yi/ni-1)<0.1 else ''}")
    print("\nattach/release event peaks (weld 激活/解除瞬间, 单列, 非真实接触峰):")
    for ke in sorted(ev):
        ap_, rp_ = np.mean([x[0] for x in ev[ke]]), np.mean([x[1] for x in ev[ke]])
        print(f"  {ke[0]:>6} {ke[1]:>14}  attach_peak={ap_:.2f}  release_peak={rp_:.2f}")

    (OUT_DIR / "transfer_audit_weld.json").write_text(json.dumps(
        {"config_weld": cfg["weld"], "mass_g": object_mass_kg(cfg) * 1000,
         "margin": MARGIN, "report": rep}, indent=2))
    print(f"\n[done] saved {OUT_DIR/'transfer_audit_weld.json'}")


if __name__ == "__main__":
    main()
