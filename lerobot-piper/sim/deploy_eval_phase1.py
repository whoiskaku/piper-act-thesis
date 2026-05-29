#!/usr/bin/env python3
"""Stage 3b — Phase 1 闭环 deploy 评估：ACT vs ACT+RefineNet。

约束（见 STAGE3B_RESULTS.md）：
  - C1-v3 部署是 receding-horizon 开环：n_action_steps=100=chunk_size，无 temporal
    ensemble。每 100 步用当前观测预测一整 chunk、全部执行，再预测。
  - ACT 有图像 backbone，无物体 sim 渲染严重 OOD → ACT 观测 teacher-force 真机
    episode（真图像+真 state），ACT chunk 可复现（= precompute 的 ACT_output）。
  - Phase 1 无物体 → 测不了 task success，只测平滑度 + 轨迹保真。

闭环新增的验证点：
  1. chunk 边界拼接：整条 episode 跨 100 步边界的 hf（监督式从没测过）。
  2. 执行噪声鲁棒性：refined 过 NoisyPiperSim 电机噪声后 executed-action hf 是否仍低。

RefineNet history 两种模式（都跑做对比）：
  - real : history 取真机 dataset wrapper 帧（= 训练分布，隔离 RefineNet 效果）。
  - cl   : history 由 RefineNet 自己跨 chunk 的 refined-executed 动作+IMU 累积
           （seed 用 episode 起点的真机 600 帧，之后滑入 refined）。

指标（active joints j1/j2/j3/grip）：commanded hf、executed hf、chunk-boundary jump、
       轨迹保真 |traj − real_demo_action|。聚合 6 个 val episode（eps 24–29）。

用法:  python sim/deploy_eval_phase1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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
from sim.train_refine_net import hf_fraction_per_joint, hf_power, ACTIVE_JOINTS

C1_CKPT = REPO_ROOT / "outputs/train/2026-05-08/C1_v3_history_seed42/checkpoints/070000/pretrained_model"
DATASET_REPO = "local/piper_pick_place_imu_v3"
REFINE_CKPT = REPO_ROOT / "outputs/train/refine_net/phase1_runB_hf0.10_dl0.1_lr0.0003_bs64_ep200/ckpt_best.pt"
NOISE_PARAMS = REPO_ROOT / "outputs/sim/noise_params.json"
MODEL_XML = REPO_ROOT / "sim/piper_model.xml"
OUT_DIR = REPO_ROOT / "outputs/eval/stage3b_deploy"

VAL_EPS = list(range(24, 30))
CHUNK = 100              # = n_action_steps
HISTORY_LEN = 600
SIM_SEED = 12345         # same noise draws across ACT / refined for fairness
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
JOINT_NAMES = ["j1", "j2", "j3", "grip"]


# ─── inference helpers ─────────────────────────────────────────────
def act_chunk(policy, sample) -> np.ndarray:
    """Run frozen ACT on one teacher-forced sample → (CHUNK, 7) numpy."""
    batch = {k: v.unsqueeze(0).to(DEVICE) for k, v in sample.items()
             if isinstance(v, torch.Tensor)}
    batch.pop("action", None); batch.pop("action_is_pad", None)
    with torch.no_grad():
        b = policy.normalize_inputs(batch)
        if policy.config.image_features:
            b = dict(b)
            b["observation.images"] = torch.stack(
                [b[k] for k in policy.config.image_features], dim=-4)
        b.pop("action", None); b.pop("action_is_pad", None)
        out = policy.model(b)
        actions = out[0] if isinstance(out, (list, tuple)) else out
        actions = policy.unnormalize_outputs({"action": actions})["action"]
    return actions[0].cpu().numpy().astype(np.float32)   # (CHUNK, 7)


def refine_chunk(model, act_out, hist_a, hist_i, hist_pad, cur_state,
                 smean, sstd) -> np.ndarray:
    """RefineNet refine one chunk → (CHUNK, 7) numpy."""
    state = (cur_state - smean) / (sstd + 1e-8)
    with torch.no_grad():
        refined, _ = model(
            torch.from_numpy(act_out)[None].to(DEVICE),
            torch.from_numpy(hist_a)[None].float().to(DEVICE),
            torch.from_numpy(hist_i)[None].float().to(DEVICE),
            torch.from_numpy(state.astype(np.float32))[None].to(DEVICE),
            history_pad_mask=torch.from_numpy(hist_pad)[None].to(DEVICE),
        )
    return refined[0].cpu().numpy().astype(np.float32)


# ─── sim execution ─────────────────────────────────────────────────
def execute_traj(sim, qpos0_8, traj_7):
    """Execute a full (T,7) commanded trajectory; return executed (T,7) + imu (T,6)."""
    sim.rng = np.random.default_rng(SIM_SEED)
    sim.reset(init_qpos=qpos0_8)
    ctrl_8 = action_7_to_ctrl_8(traj_7)
    exec_7 = np.zeros_like(traj_7)
    imu = np.zeros((len(traj_7), 6), dtype=np.float32)
    for t in range(len(traj_7)):
        obs = sim.step(ctrl_8[t], noise=True)
        exec_7[t] = ctrl_8_to_action_7(obs["action_executed"])
        imu[t] = obs["imu_noisy"]
    return exec_7, imu


# ─── metrics ───────────────────────────────────────────────────────
def hf_active(traj_7) -> np.ndarray:
    """Per-active-joint hf fraction over the FULL episode (bulk-motion-dominated;
    kept only to show why it washes out jitter)."""
    t = torch.from_numpy(traj_7.astype(np.float32))[None]   # (1,T,7)
    return hf_fraction_per_joint(t)[0, ACTIVE_JOINTS].numpy()


def hf_windowed(traj_7, win: int = 100, stride: int = 50) -> np.ndarray:
    """Per-active-joint hf fraction averaged over sliding windows — matches the
    training (per-100-step-chunk) scale, so it re-exposes local jitter."""
    T = len(traj_7)
    if T < win:
        return hf_active(traj_7)
    fracs = []
    for s in range(0, T - win + 1, stride):
        t = torch.from_numpy(traj_7[s:s + win].astype(np.float32))[None]
        fracs.append(hf_fraction_per_joint(t)[0, ACTIVE_JOINTS].numpy())
    return np.stack(fracs).mean(axis=0)


def hf_rms(traj_7) -> np.ndarray:
    """Per-active-joint absolute high-freq amplitude (rad), bulk-motion-independent.
    sqrt(high-freq power / T) from the Hann-windowed FFT."""
    t = torch.from_numpy(traj_7.astype(np.float32))[None]
    E_high, _ = hf_power(t)                                  # (1,7)
    return torch.sqrt(E_high[0, ACTIVE_JOINTS] / len(traj_7)).numpy()


def boundary_jump(traj_7) -> np.ndarray:
    """Mean |a[b]-a[b-1]| at chunk boundaries (b=100,200,...), per active joint."""
    bs = [b for b in range(CHUNK, len(traj_7), CHUNK)]
    if not bs:
        return np.zeros(len(ACTIVE_JOINTS))
    jumps = np.abs(np.stack([traj_7[b] - traj_7[b - 1] for b in bs]))  # (nb,7)
    return jumps.mean(axis=0)[ACTIVE_JOINTS]


def fidelity(traj_7, demo_7) -> float:
    """Mean |traj - real demo action| over active joints (BC-style guardrail)."""
    return float(np.abs(traj_7 - demo_7)[:, ACTIVE_JOINTS].mean())


# ─── main ──────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[device] {DEVICE}")
    print(f"[load] ACT  {C1_CKPT.parent.parent.name}")
    policy, cfg = load_frozen_act(C1_CKPT, DATASET_REPO, DEVICE)
    dataset = make_dataset(cfg)
    ep_from = dataset.episode_data_index["from"].numpy()
    ep_to = dataset.episode_data_index["to"].numpy()

    print(f"[load] RefineNet {REFINE_CKPT.parent.name}")
    ckpt = torch.load(REFINE_CKPT, map_location=DEVICE, weights_only=False)
    refine = RefineNet().to(DEVICE).eval()
    refine.load_state_dict(ckpt["model_state"])
    smean = np.asarray(ckpt["state_mean"], dtype=np.float32)
    sstd = np.asarray(ckpt["state_std"], dtype=np.float32)
    print(f"  loaded ckpt epoch {ckpt['epoch']}  "
          f"(train val hf_ratio={ckpt['val_stats']['hf_ratio']:.3f})")

    sim = NoisyPiperSim(MODEL_XML, NOISE_PARAMS, seed=SIM_SEED)

    # accumulators: condition -> list of per-episode (hf_cmd, hf_exec, bnd, fid)
    conds = ["ACT", "refined_real", "refined_cl"]
    keys = ["hf_full", "hf_win", "hf_aln", "hf_rms_cmd", "hf_rms_exec", "bnd", "fid"]
    agg = {c: {k: [] for k in keys} for c in conds}

    for ep in VAL_EPS:
        real = load_real_demo(ep)
        demo_7 = real["action"]                            # (T,7) = A_clean target
        state_mdeg = real["state_mdeg"]
        T = len(demo_7)
        qpos0_8 = state_mdeg_to_rad(state_mdeg[0])
        g0 = int(ep_from[ep])
        starts = list(range(0, T, CHUNK))

        # ---- ACT chunks (teacher-forced; shared by all conditions) ----
        act_chunks = {}        # start -> (CHUNK,7)
        samples = {}
        for s in starts:
            sample = dataset[g0 + s]
            samples[s] = sample
            act_chunks[s] = act_chunk(policy, sample)

        # ---- ACT-only & refined_real trajectories (commanded) ----
        traj_act = np.concatenate([act_chunks[s][:min(CHUNK, T - s)] for s in starts])

        traj_rr = np.zeros((T, 7), dtype=np.float32)
        for s in starts:
            sm = samples[s]
            cur_state = qpos_8_to_state_7(state_mdeg_to_rad(state_mdeg[s]))
            ref = refine_chunk(
                refine, act_chunks[s],
                sm["observation.history_action"].numpy(),
                sm["observation.history_imu"].numpy(),
                sm["history_pad_mask"].numpy(),
                cur_state, smean, sstd)
            traj_rr[s:s + min(CHUNK, T - s)] = ref[:min(CHUNK, T - s)]

        # ---- refined_cl: closed-loop refined-executed history ----
        seed = samples[0]
        tl_a = list(seed["observation.history_action"].numpy())   # 600 seed frames
        tl_i = list(seed["observation.history_imu"].numpy())
        tl_p = list(seed["history_pad_mask"].numpy())
        sim.rng = np.random.default_rng(SIM_SEED)
        sim.reset(init_qpos=qpos0_8)
        traj_cl = np.zeros((T, 7), dtype=np.float32)
        for s in starts:
            n = min(CHUNK, T - s)
            hist_a = np.asarray(tl_a[-HISTORY_LEN:], dtype=np.float32)
            hist_i = np.asarray(tl_i[-HISTORY_LEN:], dtype=np.float32)
            hist_p = np.asarray(tl_p[-HISTORY_LEN:], dtype=bool)
            cur_state = qpos_8_to_state_7(state_mdeg_to_rad(state_mdeg[s]))
            ref = refine_chunk(refine, act_chunks[s], hist_a, hist_i, hist_p,
                               cur_state, smean, sstd)
            # execute refined chunk step-by-step (closed-loop IMU feedback)
            ctrl_8 = action_7_to_ctrl_8(ref[:n])
            for t in range(n):
                obs = sim.step(ctrl_8[t], noise=True)
                traj_cl[s + t] = ref[t]
                tl_a.append(ctrl_8_to_action_7(obs["action_executed"]).astype(np.float32))
                tl_i.append(obs["imu_noisy"].astype(np.float32))
                tl_p.append(False)

        # ---- execute ACT-only & refined_real through sim (executed hf) ----
        exec_act, _ = execute_traj(sim, qpos0_8, traj_act)
        exec_rr, _ = execute_traj(sim, qpos0_8, traj_rr)
        exec_cl, _ = execute_traj(sim, qpos0_8, traj_cl)

        for c, cmd, ex in [("ACT", traj_act, exec_act),
                           ("refined_real", traj_rr, exec_rr),
                           ("refined_cl", traj_cl, exec_cl)]:
            agg[c]["hf_full"].append(hf_active(cmd))
            agg[c]["hf_win"].append(hf_windowed(cmd, win=CHUNK, stride=CHUNK // 2))
            agg[c]["hf_aln"].append(hf_windowed(cmd, win=CHUNK, stride=CHUNK))
            agg[c]["hf_rms_cmd"].append(hf_rms(cmd))
            agg[c]["hf_rms_exec"].append(hf_rms(ex))
            agg[c]["bnd"].append(boundary_jump(cmd))
            agg[c]["fid"].append(fidelity(cmd, demo_7))
        print(f"  [ep{ep}] T={T} done")

    # ---- aggregate + report ----
    def mean_joints(c, key):
        return np.stack(agg[c][key]).mean(axis=0)   # (4,)
    def mean_scalar(c, key):
        return float(np.mean(agg[c][key]))

    print(f"\n{'='*92}")
    print("Stage 3b — Phase 1 闭环 deploy eval  (6 val eps, active joints j1/j2/j3/grip)")
    print("  hf_win = 滑窗100 hf_fraction(训练scale); hf_rms = 绝对高频幅值(rad);")
    print("  hf_full = full-ep fraction(被bulk motion稀释,仅作对照)")
    print(f"{'='*92}")
    hdr = (f"{'condition':>13} {'hf_win50':>9} {'hf_aln':>8} {'hf_rms_cmd':>11} {'hf_rms_exec':>12} "
           f"{'bnd_jump':>9} {'fidelity':>9}")
    print(hdr); print("-" * 92)
    report = {}
    for c in conds:
        hw = mean_joints(c, "hf_win"); ha = mean_joints(c, "hf_aln")
        hr = mean_joints(c, "hf_rms_cmd")
        hre = mean_joints(c, "hf_rms_exec"); bn = mean_joints(c, "bnd")
        hf = mean_joints(c, "hf_full"); fd = mean_scalar(c, "fid")
        report[c] = {
            "hf_win_mean": float(hw.mean()), "hf_win_joints": hw.tolist(),
            "hf_aln_mean": float(ha.mean()), "hf_aln_joints": ha.tolist(),
            "hf_rms_cmd_mean": float(hr.mean()), "hf_rms_cmd_joints": hr.tolist(),
            "hf_rms_exec_mean": float(hre.mean()),
            "bnd_jump_mean": float(bn.mean()), "bnd_jump_joints": bn.tolist(),
            "hf_full_mean": float(hf.mean()), "fidelity": fd,
        }
        print(f"{c:>13} {hw.mean():>9.4f} {ha.mean():>8.4f} {hr.mean():>11.5f} {hre.mean():>12.5f} "
              f"{bn.mean():>9.4f} {fd:>9.4f}")
    print("-" * 92)
    print("  hf_win50=滑窗stride50(含跨边界); hf_aln=chunk对齐窗(不跨边界,=监督scale)")
    base = report["ACT"]
    for c in ["refined_real", "refined_cl"]:
        print(f"  {c} vs ACT:  hf_win ratio={report[c]['hf_win_mean']/(base['hf_win_mean']+1e-12):.3f}"
              f"  hf_rms_cmd ratio={report[c]['hf_rms_cmd_mean']/(base['hf_rms_cmd_mean']+1e-12):.3f}"
              f"  hf_rms_exec ratio={report[c]['hf_rms_exec_mean']/(base['hf_rms_exec_mean']+1e-12):.3f}")
    print("\nper-joint hf_win (j1/j2/j3/grip):")
    for c in conds:
        print(f"  {c:>13}: " + " ".join(f"{x:.4f}" for x in mean_joints(c, "hf_win")))
    print("per-joint hf_rms_cmd (j1/j2/j3/grip):")
    for c in conds:
        print(f"  {c:>13}: " + " ".join(f"{x:.5f}" for x in mean_joints(c, "hf_rms_cmd")))

    (OUT_DIR / "deploy_eval.json").write_text(json.dumps({
        "refine_ckpt": str(REFINE_CKPT), "refine_epoch": int(ckpt["epoch"]),
        "val_eps": VAL_EPS, "report": report,
    }, indent=2))
    print(f"\n[done] saved {OUT_DIR/'deploy_eval.json'}")


if __name__ == "__main__":
    main()
