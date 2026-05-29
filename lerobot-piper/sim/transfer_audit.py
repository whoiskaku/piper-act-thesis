#!/usr/bin/env python3
"""Stage 4.0 — Object-aware transfer audit (2×2×2).

唯一目标：测 action smoothing → object/contact IMU smoothing 的 transfer 是否存在。
不做 fine-tune / L1 / L2 / surrogate。

设计矩阵：
  object : {no, yes}        — base sim vs cube+table 接触
  noise  : {clean, noisy}   — NoisyPiperSim.step(noise=False/True)
  policy : {ACT, refined+blend}

每格度量：
  action_hp_rms      — 指令动作的 5Hz 高通 RMS（仅随 policy 变）
  imu_accel_hp_rms   — 执行后 IMU accel(3ch) 高通 RMS
  imu_gyro_hp_rms    — 执行后 IMU gyro(3ch) 高通 RMS
  imu_peak_accel     — |accel| 峰值
  contact_hp_rms     — 抓取窗内 IMU(6ch) 高通 RMS
  grasp_success      — object 模式下 cube 是否被抬离桌面

关键判据（详见 RESEARCH_PLAN §Stage4.0）：
  ΔIMU_clean = imu_hp_rms[obj,clean,ACT] − imu_hp_rms[obj,clean,refined+blend]
  ΔIMU_noisy = 同上(noisy)
  clean 明显降→transfer 成立; clean 降 noisy 不降→被 noise floor 淹没; clean 不降→解耦。

用法:  python sim/transfer_audit.py [--n-eps 3]
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.act.refine_net import RefineNet
from sim.noisy_piper_sim import (
    NoisyPiperSim, action_7_to_ctrl_8, ctrl_8_to_action_7, qpos_8_to_state_7,
)
from sim.replay_demo_validation import load_real_demo, state_mdeg_to_rad
from sim.precompute_act_output import load_frozen_act
from sim.deploy_eval_phase1 import (
    act_chunk, refine_chunk, blend_boundaries, _HP_B, _HP_A,
    C1_CKPT, DATASET_REPO, REFINE_CKPT, NOISE_PARAMS, MODEL_XML,
    CHUNK, SIM_SEED,
)
from sim.build_object_model import load_config, detect_grasp_frames, derive_placement, write_object_model
from scipy.signal import filtfilt

OUT_DIR = REPO_ROOT / "outputs/eval/stage4_transfer_audit"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def hp_rms_cols(x):
    """Per-column 5Hz high-pass RMS, return mean over given columns. x:(T,C)."""
    y = filtfilt(_HP_B, _HP_A, x, axis=0)
    return float(np.sqrt((y ** 2).mean(axis=0)).mean())


def execute(sim, qpos0_8, traj_7, noise, cube_qadr=None):
    """Execute commanded (T,7) traj; return imu(T,6), exec(T,7), cube_lift(float)."""
    sim.rng = np.random.default_rng(SIM_SEED)
    sim.reset(init_qpos=qpos0_8)
    cube_z0 = float(sim.data.qpos[cube_qadr + 2]) if cube_qadr is not None else None
    ctrl8 = action_7_to_ctrl_8(traj_7)
    imu = np.zeros((len(traj_7), 6), dtype=np.float32)
    exe = np.zeros_like(traj_7)
    cube_zmax = -1e9
    for t in range(len(traj_7)):
        obs = sim.step(ctrl8[t], noise=noise)
        imu[t] = obs["imu_noisy"] if noise else obs["imu_clean"]
        exe[t] = ctrl_8_to_action_7(obs["action_executed"])
        if cube_qadr is not None:
            cube_zmax = max(cube_zmax, float(sim.data.qpos[cube_qadr + 2]))
    lift = (cube_zmax - cube_z0) if cube_qadr is not None else None
    return imu, exe, lift


def build_policy_trajs(policy, dataset, ep_from, refine, smean, sstd, ep):
    """Reproduce deploy trajectories: ACT-only and refined+blend over a full episode."""
    real = load_real_demo(ep)
    demo7 = real["action"]; state_mdeg = real["state_mdeg"]; T = len(demo7)
    g0 = int(ep_from[ep]); starts = list(range(0, T, CHUNK))
    qpos0_8 = state_mdeg_to_rad(state_mdeg[0])

    traj_act = np.zeros((T, 7), np.float32)
    traj_ref = np.zeros((T, 7), np.float32)
    for s in starts:
        n = min(CHUNK, T - s)
        sm = dataset[g0 + s]
        ac = act_chunk(policy, sm)
        cur = qpos_8_to_state_7(state_mdeg_to_rad(state_mdeg[s]))
        rf = refine_chunk(refine, ac, sm["observation.history_action"].numpy(),
                          sm["observation.history_imu"].numpy(),
                          sm["history_pad_mask"].numpy(), cur, smean, sstd)
        traj_act[s:s + n] = ac[:n]
        traj_ref[s:s + n] = rf[:n]
    traj_blend = blend_boundaries(traj_ref, m=10)
    return demo7, qpos0_8, traj_act, traj_blend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eps", type=int, default=3)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    print(f"[cfg] cube {cfg['cube']['size_mm']}mm {cfg['cube']['mass_g']}g  "
          f"friction={cfg['contact']['friction'][0]}  (占位={cfg['cube'].get('_todo') is not None})")

    policy, tcfg = load_frozen_act(C1_CKPT, DATASET_REPO, DEVICE)
    dataset = make_dataset(tcfg)
    ep_from = dataset.episode_data_index["from"].numpy()
    ckpt = torch.load(REFINE_CKPT, map_location=DEVICE, weights_only=False)
    refine = RefineNet().to(DEVICE).eval(); refine.load_state_dict(ckpt["model_state"])
    smean = np.asarray(ckpt["state_mean"], np.float32); sstd = np.asarray(ckpt["state_std"], np.float32)
    print(f"[refine] ep{ckpt['epoch']}  policies = ACT, refined+blend(m=10)")

    sim_base = NoisyPiperSim(MODEL_XML, NOISE_PARAMS, seed=SIM_SEED)

    val_eps = list(range(24, 24 + args.n_eps))
    # accumulators: (object, noise, policy) -> list of metric dicts
    cells = {}
    act_hp = {"ACT": [], "refined+blend": []}

    for ep in val_eps:
        demo7, qpos0_8, traj_act, traj_blend = build_policy_trajs(
            policy, dataset, ep_from, refine, smean, sstd, ep)
        real = load_real_demo(ep)
        gframes = detect_grasp_frames(real)
        act_hp["ACT"].append(hp_rms_cols(traj_act[:, [0, 1, 2, 6]]))
        act_hp["refined+blend"].append(hp_rms_cols(traj_blend[:, [0, 1, 2, 6]]))

        # object sim for this episode (cube at demo grasp)
        cube_xyz, table_top_z = derive_placement(MODEL_XML, real, cfg)
        # 必须写在 sim/ 下，否则 meshes/ 相对路径解析不到
        obj_xml = Path(tempfile.mktemp(suffix=".xml", dir=str(MODEL_XML.parent)))
        write_object_model(cfg, cube_xyz, table_top_z, obj_xml)
        sim_obj = NoisyPiperSim(obj_xml, NOISE_PARAMS, seed=SIM_SEED)
        cube_qadr = sim_obj.model.jnt_qposadr[
            mujoco.mj_name2id(sim_obj.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")]

        for obj, sim, qadr in [("no", sim_base, None), ("yes", sim_obj, cube_qadr)]:
            for noise_name, noise in [("clean", False), ("noisy", True)]:
                for pol, traj in [("ACT", traj_act), ("refined+blend", traj_blend)]:
                    imu, _, lift = execute(sim, qpos0_8, traj, noise, qadr)
                    cwin = np.concatenate([imu[max(0, g - 3):g + 10] for g in gframes]) \
                        if gframes else imu
                    key = (obj, noise_name, pol)
                    cells.setdefault(key, []).append({
                        "imu_accel_hp": hp_rms_cols(imu[:, :3]),
                        "imu_gyro_hp": hp_rms_cols(imu[:, 3:]),
                        "imu_peak_accel": float(np.linalg.norm(imu[:, :3] - imu[:, :3].mean(0), axis=1).max()),
                        "contact_hp": hp_rms_cols(cwin),
                        "grasp_lift": lift,
                    })
        obj_xml.unlink(missing_ok=True)
        print(f"  [ep{ep}] done  (grasp frames={gframes})")

    # ---- aggregate + report ----
    def agg(key, m):
        vals = [d[m] for d in cells[key] if d[m] is not None]
        return float(np.mean(vals)) if vals else None

    print(f"\n{'='*100}")
    print("Stage 4.0 — 2×2×2 transfer audit  (占位 cube 参数，仅跑通；结论需实测 cube)")
    print(f"  action hp_rms:  ACT={np.mean(act_hp['ACT']):.5f}  refined+blend={np.mean(act_hp['refined+blend']):.5f}")
    print(f"{'='*100}")
    hdr = (f"{'object':>7} {'noise':>6} {'policy':>14} {'accel_hp':>9} {'gyro_hp':>9} "
           f"{'peak_acc':>9} {'contact_hp':>10} {'grasp_lift':>10}")
    print(hdr); print("-" * 100)
    report = {}
    for obj in ["no", "yes"]:
        for noise_name in ["clean", "noisy"]:
            for pol in ["ACT", "refined+blend"]:
                k = (obj, noise_name, pol)
                r = {m: agg(k, m) for m in ["imu_accel_hp", "imu_gyro_hp", "imu_peak_accel", "contact_hp", "grasp_lift"]}
                report["/".join(k)] = r
                gl = f"{r['grasp_lift']:.4f}" if obj == "yes" else "  --"
                print(f"{obj:>7} {noise_name:>6} {pol:>14} {r['imu_accel_hp']:>9.5f} "
                      f"{r['imu_gyro_hp']:>9.5f} {r['imu_peak_accel']:>9.3f} {r['contact_hp']:>10.5f} {gl:>10}")
    print("-" * 100)
    # transfer 判据（object 模式，accel+gyro 合并看）
    def imu_hp(obj, noise):
        a = report[f"{obj}/{noise}/ACT"]; b = report[f"{obj}/{noise}/refined+blend"]
        ai = a["imu_accel_hp"] + a["imu_gyro_hp"]; bi = b["imu_accel_hp"] + b["imu_gyro_hp"]
        return ai, bi
    for obj in ["no", "yes"]:
        for noise in ["clean", "noisy"]:
            ai, bi = imu_hp(obj, noise)
            print(f"  [object={obj} {noise}] ΔIMU = ACT−refined = {ai-bi:+.5f}  "
                  f"(ACT={ai:.5f} refined={bi:.5f}, ratio={bi/ai:.3f})")
    print("\n判读: object=yes 的 clean 行 ΔIMU>0 且明显 → transfer 成立; "
          "clean 降但 noisy 不降 → noise floor; clean 不降 → 解耦(需 L2)。")
    (OUT_DIR / "transfer_audit.json").write_text(json.dumps({
        "config": cfg, "n_eps": args.n_eps, "action_hp": {k: float(np.mean(v)) for k, v in act_hp.items()},
        "report": report,
    }, indent=2))
    print(f"\n[done] saved {OUT_DIR/'transfer_audit.json'}")


if __name__ == "__main__":
    main()
