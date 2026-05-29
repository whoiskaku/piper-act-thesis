#!/usr/bin/env python3
"""Stage 3b 预备 — 量化每个候选 loss 项在真实数据上的数值范围。

目的：在改 loss / 定 λ 之前，先知道
    L_BC, L_delta, L_acc, L_hf_fraction, jerk_RMS
分别在 ACT_output / A_clean / identity(refined=ACT_output) 上的真实量级，
这样 λ 可以按 "正则项加权后 ≈ L_BC 的 10~30%" 来推，而不是拍脑袋。

各项定义（与拟用训练 loss 一致）：
    L_BC(X)   = mean |X - A_clean|                         (rad)
    L_delta   = mean |refined - ACT_output| = mean |ΔA|    (rad)  identity 下 = 0
    L_acc(X)  = mean |X[t+1] - 2X[t] + X[t-1]|             (rad, 不除 dt²)
    L_hf(X)   = 5Hz 以上能量占比 ∈ [0,1]                    (无量纲)
    jerk(X)   = RMS of 三阶差分 / dt³                       (rad/s³)  仅作 eval 参照

用法:
    python sim/analyze_loss_scales.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sim.refine_net_dataset import RefinePhase1Dataset
from sim.train_refine_net import jerk_rms, hf_fraction_per_joint, ACTIVE_JOINTS


def hf_energy_fraction(actions):
    """Active-joint mean hf fraction (scalar per call) for scale analysis."""
    return hf_fraction_per_joint(actions)[:, ACTIVE_JOINTS].mean(dim=1)

H5_PATH = REPO_ROOT / "outputs/sim/phase1_paired_data.h5"


def l_acc(x: torch.Tensor) -> torch.Tensor:
    """Mean |second difference| over time. x: (B,T,A). 不除 dt²，rad 量级。"""
    d2 = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
    return d2.abs().mean()


def per_joint(x: torch.Tensor, ref: torch.Tensor) -> np.ndarray:
    """逐关节 mean |x - ref|，看哪几个关节贡献 L_BC。"""
    return (x - ref).abs().mean(dim=(0, 1)).cpu().numpy()


def summarize(name: str, X: torch.Tensor, A_clean: torch.Tensor):
    bc = (X - A_clean).abs().mean().item()
    acc = l_acc(X).item()
    hf = hf_energy_fraction(X).mean().item()
    jk = jerk_rms(X).item()
    print(f"  {name:<14} L_BC={bc:.5f}  L_acc={acc:.5f}  "
          f"L_hf_frac={hf:.4f}  jerk_RMS={jk:.1f}")
    return dict(bc=bc, acc=acc, hf=hf, jerk=jk)


def main():
    for split in ("train", "val"):
        ds = RefinePhase1Dataset(H5_PATH, split)
        ACT = torch.from_numpy(ds.ACT_output)        # (N,100,7)
        CLEAN = torch.from_numpy(ds.A_clean)
        N = ACT.shape[0]
        print(f"\n[{split}] N={N}  chunk={ACT.shape[1]}  action_dim={ACT.shape[2]}")

        # 各项在两种序列上的数值
        s_act = summarize("ACT_output", ACT, CLEAN)
        s_cln = summarize("A_clean", CLEAN, CLEAN)   # L_BC(A_clean)=0 by def

        # identity 情形：refined = ACT_output → L_delta = 0, L_BC = |ACT-CLEAN|
        print(f"  {'identity':<14} L_delta=0.00000 (refined=ACT_output), "
              f"L_BC={s_act['bc']:.5f}")

        # 逐关节 L_BC（ACT vs clean），看活动关节
        pj = per_joint(ACT, CLEAN)
        print(f"  per-joint |ACT-clean|: "
              + " ".join(f"j{i+1}={v:.4f}" for i, v in enumerate(pj)))

        if split == "val":
            # ── 基于 val 量级推 λ：令各正则项加权后 ≈ L_BC 的 10~30% ──
            L_BC = s_act["bc"]
            print(f"\n[λ 推导 @ val]  目标: λ·term ∈ [{0.1*L_BC:.5f}, {0.3*L_BC:.5f}]  "
                  f"(= L_BC 的 10~30%, L_BC={L_BC:.5f})")
            # 参照值：用 ACT_output 上的项值（refined 初始就在这里）
            # L_delta 用 |ACT-clean| 作为 "合理 refinement 位移" 的代表量级
            ref = {
                "λ_delta (ref |ACT-clean|)": s_act["bc"],
                "λ_acc   (ref L_acc(ACT))":  s_act["acc"],
                "λ_hf    (ref hf_frac(ACT))": s_act["hf"],
            }
            for label, term_val in ref.items():
                lo = 0.1 * L_BC / term_val
                hi = 0.3 * L_BC / term_val
                print(f"  {label:<28} term≈{term_val:.5f}  →  λ ∈ [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
