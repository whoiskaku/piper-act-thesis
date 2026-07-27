#!/usr/bin/env python3
"""#75 — MuJoCo 并排视觉回放视频 (demo 用): ACT vs RefineNet 命令动作。

从 h5 val 用 (src_ep, src_start) 重建某 episode 的部署轨迹 (非重叠 chunk 0/100/200..
= receding-horizon 部署的 ACT 命令), refined 用 Hattn cut2.0 跑同样 chunk。kinematic 回放
(qpos=命令, 不过物理) → 左 ACT / 右 refined 并排渲染 → mp4。

诚实说明: HF 抖动真尺度 ~0.09°(亚毫米), 渲染肉眼看不出 → 默认 --amplify 6 把每个 chunk 内
>2Hz 抖动放大 6× 让差异可见, 视频角标注明"jitter ×N for visibility"。真平滑数值以频谱图为准。

用法: /home/enolab-guo/miniconda3/envs/lerobot/bin/python sim/replay_video.py [--ep 24]
      [--ckpt <ckpt_best.pt>] [--amplify 6] [--fps 30]
"""
from __future__ import annotations
from pathlib import Path
import argparse

import h5py
import numpy as np
import torch
import cv2
import imageio.v2 as imageio
import mujoco
from scipy.signal import butter, filtfilt

from sim.eval_action_smoothness import reconstruct, run_model, RND, H5, DEV
from sim.noisy_piper_sim import action_7_to_ctrl_8

REPO = Path(__file__).resolve().parents[1]
MODEL_XML = REPO / "sim/piper_model.xml"
OUT = REPO / "outputs/eval/replay_video_ACT_vs_refined.mp4"
DEFAULT = "phase1_dense_Hattn+absboth_cut2.0_hf1.0_dl0.1_lr0.0003_bs64_ep150"
CHUNK = 100


def amplify_jitter(traj, amp, filt):
    """逐 chunk 放大【RefineNet 作用带 2-amp_hi Hz】抖动(不放大边界跳变, 也不放大两臂共有的
    >5Hz 不可学噪声): out = traj + (amp-1)*bandpass(traj)。traj (T,7)。"""
    if amp == 1.0:
        return traj
    out = traj.copy()
    for s in range(0, len(traj), CHUNK):
        seg = traj[s:s+CHUNK]
        if len(seg) < 18:
            continue
        bp = filtfilt(*filt, seg, axis=0)
        out[s:s+len(seg)] = seg + (amp - 1.0) * bp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=24)
    ap.add_argument("--ckpt", type=Path, default=RND / DEFAULT / "ckpt_best.pt")
    ap.add_argument("--amplify", type=float, default=6.0, help="可视化放大 RefineNet 作用带抖动(真尺度看不见); 1=真尺度")
    ap.add_argument("--amp-hi", type=float, default=4.5, help="放大带上限 Hz(默认 4.5=RefineNet 作用带; 设 15=放大全部>2Hz旧行为)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--w", type=int, default=480)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--out", type=Path, default=OUT, help="输出 mp4 路径(避免覆盖不同 amplify 版)")
    cli = ap.parse_args()

    # ---- 从 h5 重建 episode 的部署轨迹 (非重叠 chunk) ----
    with h5py.File(H5, "r") as f:
        g = f["val"]
        ep = g["src_ep"][:]; st = g["src_start"][:]
        sel = []
        for s in range(0, 100000, CHUNK):
            idx = np.where((ep == cli.ep) & (st == s))[0]
            if len(idx) == 0:
                break
            sel.append(int(idx[0]))
        ACT = g["ACT_output"][:][sel].astype(np.float32)        # (nchunk,100,7)
        IMG = g["act_img_tokens"][:][sel]
    nch = len(sel)
    print(f"[replay] ep{cli.ep}: {nch} 非重叠 chunk → {nch*CHUNK} 帧 ({nch*CHUNK/cli.fps:.1f}s @ {cli.fps}fps)")

    ckpt = torch.load(cli.ckpt, map_location=DEV, weights_only=False)
    m, a = reconstruct(ckpt)
    REF = run_model(m, ACT, IMG)                                # (nchunk,100,7)
    traj_act = ACT.reshape(-1, 7)
    traj_ref = REF.reshape(-1, 7)
    print(f"[replay] ckpt={cli.ckpt.parent.name[:42]} ep{ckpt.get('epoch')}  amplify={cli.amplify}×")

    filt = butter(4, [2.0 / 15.0, min(cli.amp_hi, 14.9) / 15.0], "band")
    traj_act_v = amplify_jitter(traj_act, cli.amplify, filt)
    traj_ref_v = amplify_jitter(traj_ref, cli.amplify, filt)
    T = len(traj_act_v)

    # ---- MuJoCo kinematic 渲染 ----
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 135.0, -20.0, 1.1
    cam.lookat[:] = [0.05, 0.0, 0.30]
    rnd = mujoco.Renderer(model, height=cli.h, width=cli.w)

    def frame_of(traj, t):
        data.qpos[:8] = action_7_to_ctrl_8(traj[t])
        mujoco.mj_forward(model, data)
        rnd.update_scene(data, camera=cam)
        return rnd.render()

    def label(img, txt, color):
        img = np.ascontiguousarray(img)
        cv2.rectangle(img, (0, 0), (cli.w, 34), (0, 0, 0), -1)
        cv2.putText(img, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return img

    amp_note = (f"2-{cli.amp_hi:g}Hz jitter x{cli.amplify:g} for visibility"
                if cli.amplify != 1.0 else "true scale")
    frames = []
    for t in range(T):
        ia = label(frame_of(traj_act_v, t), "ACT (raw)", (90, 160, 255))
        ir = label(frame_of(traj_ref_v, t), "RefineNet cut2.0", (120, 255, 120))
        both = np.hstack([ia, ir])
        cv2.putText(both, amp_note, (10, cli.h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (220, 220, 220), 1, cv2.LINE_AA)
        frames.append(both)
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(cli.out, frames, fps=cli.fps, quality=8, macro_block_size=1)
    print(f"[saved] {cli.out}  ({T} frames, {model.nq} dof)")


if __name__ == "__main__":
    main()
