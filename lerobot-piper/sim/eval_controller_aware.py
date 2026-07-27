#!/usr/bin/env python3
"""#71 — controller-aware 残差 vs 命令空间残差 的诚实对比。

固定口径(不依赖训练时 cutoff): 分带能量 [0-2 / 2-4.5 / 5-15 Hz](Hann-FFT, 去DC),
active joints j1/j2/j3/grip。对每个 ckpt 在 **命令空间** 与 **控制器透传空间 C(·)**
(以及代理 IMU = C 后二阶差分)都算 refined/ACT 带比 + L1 保真。

要回答:
  · 透传带(2-4.5Hz) C 空间比: controller-aware 是否 < 命令空间基线(#66 放大被消除)?
  · 代理 IMU 高频: 是否真降?
  · L1: 保真是否守住(护栏 ACT~clean 的 1.1×)?

用法: python sim/eval_controller_aware.py
"""
from __future__ import annotations
import json
from pathlib import Path

import h5py
import numpy as np
import torch

import sim.train_refine_net as T
from sim.train_refine_net import ACTIVE_JOINTS as AJ, FPS
from lerobot.common.policies.act.refine_net import RefineNet

REPO = Path(__file__).resolve().parents[1]
H5 = REPO / "outputs/sim/phase1_dense.h5"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
RND = REPO / "outputs/train/refine_net"
BANDS = [(0.0, 2.0), (2.0, 4.5), (5.0, 15.0)]

CKPTS = {
    "cmd-space (基线)": RND / "phase1_dense_Hattn+absboth_cut2.0_hf1.0_dl0.1_lr0.0003_bs64_ep150/ckpt_best.pt",
    "ctrl-aware":       RND / "phase1_S0+Hattn+absboth+cut2+ctrl_in-act_output_hf1.0_dl0.1_lr0.0003_bs64_ep150/ckpt_best.pt",
    "ctrl+imu+cmb":     RND / "phase1_S0+Hattn+absboth+cut2+ctrlimu+cmb_in-act_output_hf1.0_dl0.1_lr0.0003_bs64_ep150/ckpt_best.pt",
}


def band_energy(x):
    """x (N,T,A) → 每带 active-joint 能量和(标量/带)。Hann, 去DC。"""
    N, Tn, A = x.shape
    xac = x - x.mean(1, keepdims=True)
    w = np.hanning(Tn)[None, :, None]
    X = np.fft.rfft(xac * w, axis=1)
    P = (np.abs(X) ** 2)[:, :, AJ]                       # (N,F,|AJ|)
    f = np.fft.rfftfreq(Tn, d=1.0 / FPS)
    return {b: float(P[:, (f >= b[0]) & (f < b[1])].sum()) for b in BANDS}


def accel(x):
    return x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]


def apply_C(x):
    t = T.apply_controller(torch.from_numpy(x.astype(np.float32)).to(DEV)).cpu().numpy()
    return t


def reconstruct(ckpt):
    a = ckpt.get("args", {}) or {}
    V = {"S0": (False, False, False), "S1": (True, False, False),
         "S2": (True, True, False), "S3": (True, True, True), "full": (True, True, True)}
    us, uha, uhi = V.get(a.get("variant", "full"), (True, True, True))
    m = RefineNet(use_state=us, use_history_action=uha, use_history_imu=uhi,
                  use_act_hidden=a.get("use_act_hidden", False),
                  use_act_img_attn=a.get("use_act_img_attn", False),
                  n_img_readout=a.get("n_img_readout", 1),
                  use_act_rlt=a.get("use_act_rlt", False)).to(DEV).eval()
    sd = ckpt["model_state"]; te = sd["type_embed.weight"]
    if te.shape[0] < m.type_embed.weight.shape[0]:
        sd["type_embed.weight"] = torch.cat([te, m.type_embed.weight.data[te.shape[0]:].clone()], 0)
    m.load_state_dict(sd)
    return m, a


def run_model(m, ACT, IMG):
    out = []
    z = torch.zeros
    with torch.no_grad():
        for b in range(0, len(ACT), 128):
            ao = torch.from_numpy(ACT[b:b+128]).to(DEV)
            ai = torch.from_numpy(IMG[b:b+128].astype(np.float32)).to(DEV)
            B = ao.shape[0]
            r, _ = m(ao, z(B, 600, 7, device=DEV), z(B, 600, 6, device=DEV),
                     z(B, 7, device=DEV), history_pad_mask=None, act_img=ai)
            out.append(r.cpu().numpy())
    return np.concatenate(out, 0)


def ratios(refined, ACT, clean, space):
    """命令/控制器/代理IMU 三空间的分带比 + L1。space ∈ {cmd, ctrl, ctrl_imu}。"""
    if space == "cmd":
        r, a = refined, ACT
    elif space == "ctrl":
        r, a = apply_C(refined), apply_C(ACT)
    else:  # ctrl_imu
        r, a = accel(apply_C(refined)), accel(apply_C(ACT))
    Er, Ea = band_energy(r), band_energy(a)
    return {f"{b[0]:g}-{b[1]:g}Hz": Er[b] / (Ea[b] + 1e-12) for b in BANDS}


def main():
    T.load_controller()
    with h5py.File(H5, "r") as f:
        g = f["val"]
        ACT = g["ACT_output"][:].astype(np.float32)
        CLN = g["A_clean"][:].astype(np.float32)
        IMG = g["act_img_tokens"][:]
    print(f"[data] val n={len(ACT)}")
    L1_ACT = float(np.abs(ACT[:, :, AJ] - CLN[:, :, AJ]).mean())
    print(f"[guard] L1(ACT,clean) active = {L1_ACT:.5f}  (护栏=1.1× = {1.1*L1_ACT:.5f})\n")

    report = {}
    for name, path in CKPTS.items():
        if not path.exists():
            print(f"[skip] {name}: ckpt 不存在 {path}"); continue
        ckpt = torch.load(path, map_location=DEV, weights_only=False)
        m, a = reconstruct(ckpt)
        refined = run_model(m, ACT, IMG)
        L1 = float(np.abs(refined[:, :, AJ] - CLN[:, :, AJ]).mean())
        rec = {"epoch": ckpt.get("epoch"), "L1": L1,
               "cmd": ratios(refined, ACT, CLN, "cmd"),
               "ctrl": ratios(refined, ACT, CLN, "ctrl"),
               "ctrl_imu": ratios(refined, ACT, CLN, "ctrl_imu")}
        report[name] = rec
        print(f"=== {name}  (ep{rec['epoch']}, L1={L1:.5f}{' ✗破护栏' if L1>1.1*L1_ACT else ' ✓'}) ===")
        for sp, lab in [("cmd", "命令空间"), ("ctrl", "控制器透传 C(·)"), ("ctrl_imu", "代理IMU C后accel")]:
            s = "  ".join(f"{k}={v:.3f}" for k, v in rec[sp].items())
            print(f"  {lab:>16}: {s}")
        print()

    print("[判读] 透传 C(·) 与代理IMU 的 2-4.5Hz 比: controller-aware 应 < 命令空间基线")
    print("       (命令空间 #66 在 2-4.5Hz 放大>1 → 真机 IMU 变差; ctrl-aware 把损失放这带 → 应≤1)")
    out = REPO / "outputs/eval/controller_aware_eval.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
