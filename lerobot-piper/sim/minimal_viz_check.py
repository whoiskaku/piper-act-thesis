#!/usr/bin/env python3
"""#76 — 最小可视化检查 (不改 wrapper): A=原始ACT vs B=ACT+Hattn cut2.0 残差。

离线: A=h5 重建 ep 命令(非重叠 chunk); B=A 过 Hattn cut2.0 残差。6 面板时域对比:
  1 joint position  2 joint velocity  3 joint acceleration  4 joint jerk  (命令, 直接=残差效果)
  5 IMU accel  6 IMU gyro  (把 A/B 各喂 NoisyPiperSim 执行取 sim IMU)
残差小 → position 几乎重合; 微分阶越高(vel→accel→jerk) HF 越放大, 残差平滑越明显。
诚实: sim action→IMU 通路弱(Stage4), 面板5/6 的 A vs B 很可能接近(=sim测不到IMU收益, 非bug)。

用法: /home/enolab-guo/miniconda3/envs/lerobot/bin/python sim/minimal_viz_check.py [--ep 24] [--joint 1]
"""
from __future__ import annotations
from pathlib import Path
import argparse

import h5py
import numpy as np
import torch

from sim.eval_action_smoothness import reconstruct, run_model, RND, H5, DEV
from sim.noisy_piper_sim import NoisyPiperSim, action_7_to_ctrl_8, ctrl_8_to_action_7

REPO = Path(__file__).resolve().parents[1]
MODEL_XML = REPO / "sim/piper_model.xml"
NOISE = REPO / "outputs/sim/noise_params.json"
OUT = REPO / "outputs/eval/minimal_viz_ACT_vs_refined.png"
DEFAULT = "phase1_dense_Hattn+absboth_cut2.0_hf1.0_dl0.1_lr0.0003_bs64_ep150"
FS = 30.0
CHUNK = 100
JN = {0: "j1", 1: "j2", 2: "j3"}


def execute(sim, qpos0_8, traj_7, noise=False):
    sim.rng = np.random.default_rng(123)
    sim.reset(init_qpos=qpos0_8)
    ctrl = action_7_to_ctrl_8(traj_7)
    imu = np.zeros((len(traj_7), 6))
    for t in range(len(traj_7)):
        o = sim.step(ctrl[t], noise=noise)
        imu[t] = o["imu_clean"] if not noise else o["imu_noisy"]
    return imu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=24)
    ap.add_argument("--joint", type=int, default=1, help="关节面板用哪个臂关节(0/1/2=j1/j2/j3)")
    ap.add_argument("--ckpt", type=Path, default=RND / DEFAULT / "ckpt_best.pt")
    ap.add_argument("--win", type=int, default=150, help="可视窗口帧数(自动选高运动段)")
    cli = ap.parse_args()

    with h5py.File(H5, "r") as f:
        g = f["val"]
        ep = g["src_ep"][:]; st = g["src_start"][:]
        sel = []
        for s in range(0, 100000, CHUNK):
            idx = np.where((ep == cli.ep) & (st == s))[0]
            if len(idx) == 0:
                break
            sel.append(int(idx[0]))
        ACTc = g["ACT_output"][:][sel].astype(np.float32)         # (nch,100,7)
        IMG = g["act_img_tokens"][:][sel]
        q0 = g["start_state_7"][:][sel[0]].astype(np.float64)     # 7-dim 起始(rad)
    A = ACTc.reshape(-1, 7)                                       # 原始 ACT 命令
    ckpt = torch.load(cli.ckpt, map_location=DEV, weights_only=False)
    m, _ = reconstruct(ckpt)
    B = run_model(m, ACTc, IMG).reshape(-1, 7)                    # +Hattn cut2.0 残差
    print(f"[viz] ep{cli.ep} {len(A)} 帧; ckpt ep{ckpt.get('epoch')}")

    # sim 执行取 IMU (clean, 同 seed) —— A 与 B 各跑一次
    sim = NoisyPiperSim(MODEL_XML, NOISE, seed=123)
    q0_8 = action_7_to_ctrl_8(q0)
    imuA = execute(sim, q0_8, A); imuB = execute(sim, q0_8, B)

    # 单 chunk 内窗口(不跨 100 步边界): 每个 chunk 内取 [margin:margin+win], 选运动最大的 chunk
    j = cli.joint
    win = min(cli.win, CHUNK - 10)
    margin = 5
    vel_full = np.abs(np.diff(A[:, j], prepend=A[0, j])) * FS
    nch = max(1, len(A) // CHUNK)
    best_k = max(range(nch), key=lambda k: vel_full[k*CHUNK+margin: k*CHUNK+margin+win].sum())
    e = best_k * CHUNK + margin
    sl = slice(e, e + win)
    print(f"[viz] 单chunk内窗口: chunk{best_k} frames[{e}:{e+win}] (无边界)")
    t = np.arange(sl.start, sl.stop) / FS

    def d(x, n):  # n 阶差分 ×fps^n
        for _ in range(n):
            x = np.diff(x) * FS
        return x

    def rms24(x):
        from scipy.signal import butter, filtfilt
        b, a = butter(4, [2/15, 4.5/15], "band")
        return float(np.sqrt((filtfilt(b, a, x)**2).mean())) if len(x) > 30 else float("nan")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(6, 1, figsize=(12, 13), sharex=True)
    panels = [
        (f"{JN[j]} position (cmd, rad)", A[:, j], B[:, j], 0),
        (f"{JN[j]} velocity (rad/s)", A[:, j], B[:, j], 1),
        (f"{JN[j]} acceleration (rad/s^2)", A[:, j], B[:, j], 2),
        (f"{JN[j]} jerk (rad/s^3)", A[:, j], B[:, j], 3),
        ("IMU accel |a-mean| (sim, m/s^2)", None, None, -1),
        ("IMU gyro |w| (sim, rad/s)", None, None, -2),
    ]
    print(f"  {'panel':>26} {'A rms(2-4.5Hz)':>15} {'B':>11} {'B/A':>6}")
    for r, (lab, sa, sb, order) in enumerate(panels):
        if order >= 0:
            ya = d(sa, order)[sl]; yb = d(sb, order)[sl]   # n 阶差分后取窗口
        elif order == -1:
            ya = np.linalg.norm(imuA[:, :3] - imuA[:, :3].mean(0), axis=1)[sl]
            yb = np.linalg.norm(imuB[:, :3] - imuB[:, :3].mean(0), axis=1)[sl]
        else:
            ya = np.linalg.norm(imuA[:, 3:], axis=1)[sl]; yb = np.linalg.norm(imuB[:, 3:], axis=1)[sl]
        ra, rb = rms24(ya), rms24(yb)
        print(f"  {lab[:26]:>26} {ra:>15.4g} {rb:>11.4g} {rb/(ra+1e-12):>6.2f}")
        xa = (np.arange(sl.start, sl.start + len(ya))) / FS
        ax[r].plot(xa, ya, color="C0", lw=.9, label=f"A:ACT (2-4.5Hz rms {ra:.4g})")
        ax[r].plot(xa, yb, color="C3", lw=.9, label=f"B:+residual ({rb:.4g}, ×{rb/(ra+1e-12):.2f})")
        ax[r].set_ylabel(lab, fontsize=9); ax[r].legend(fontsize=8, loc="upper right"); ax[r].grid(alpha=.3)
    ax[-1].set_xlabel("s")
    fig.suptitle(f"Minimal check: A=ACT vs B=ACT+Hattn cut2.0  (ep{cli.ep}, {JN[j]}; 越高阶微分残差越明显; sim IMU弱)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT, dpi=130)
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
