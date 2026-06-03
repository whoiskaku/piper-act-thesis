#!/usr/bin/env python3
"""#62 — 输入信号对照: 在指定输入信号下评估已训 RefineNet ckpt。

加载 ckpt(从 ckpt['args']['variant'] 还原输入开关), 在 val 上以 --input-signal
指定的信号(act_output 或 a_noisy)作为"待精修输入", 报告:
  hf(input)        输入本身的 active-joint hf
  hf(refined)      精修后 active-joint hf
  hf_ratio         refined/input
  hf_abs_refined   refined 的绝对 active hf(跨模型可比)
  L_BC             L1(refined, A_clean)  (任务保真)

用于对照: N0(训act_output) @ act_output ; N1(训a_noisy) @ a_noisy ; N1 @ act_output(部署分布)。

用法: python sim/eval_input_signal.py --ckpt <path> --input-signal act_output|a_noisy
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from lerobot.common.policies.act.refine_net import RefineNet
from sim.refine_net_dataset import RefinePhase1Dataset
from sim.train_refine_net import hf_fraction_per_joint, ACTIVE_JOINTS, H5_PATH

VARIANTS = {"S0": (False, False, False), "S1": (True, False, False),
            "S2": (True, True, False), "S3": (True, True, True), "full": (True, True, True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input-signal", choices=["act_output", "a_noisy"], default="act_output")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=dev, weights_only=False)
    variant = ckpt.get("args", {}).get("variant", "full")
    us, uha, uhi = VARIANTS.get(variant, (True, True, True))
    model = RefineNet(use_state=us, use_history_action=uha, use_history_imu=uhi).to(dev).eval()
    model.load_state_dict(ckpt["model_state"])
    smean = np.asarray(ckpt["state_mean"], np.float32); sstd = np.asarray(ckpt["state_std"], np.float32)

    ds = RefinePhase1Dataset(H5_PATH, "val", smean, sstd)
    inp = torch.from_numpy(ds.A_noisy if args.input_signal == "a_noisy" else ds.ACT_output).to(dev)
    A_clean = torch.from_numpy(ds.A_clean).to(dev)
    ha = torch.from_numpy(ds.history_action).to(dev)
    hi = torch.from_numpy(ds.history_imu).to(dev)
    pad = torch.from_numpy(ds.history_pad_mask).to(dev)
    st = (torch.from_numpy(ds.start_state_7).to(dev) - torch.from_numpy(smean).to(dev)) / (torch.from_numpy(sstd).to(dev) + 1e-8)

    with torch.no_grad():
        refined, delta = model(inp, ha, hi, st, history_pad_mask=pad)
    hf_in = hf_fraction_per_joint(inp.cpu())[:, ACTIVE_JOINTS].mean().item()
    hf_ref = hf_fraction_per_joint(refined.cpu())[:, ACTIVE_JOINTS].mean().item()
    l_bc = (refined - A_clean).abs().mean().item()
    print(f"ckpt={Path(args.ckpt).parent.parent.name}  variant={variant}  输入={args.input_signal}")
    print(f"  hf(input)={hf_in:.4f}  hf(refined)={hf_ref:.4f}  hf_ratio={hf_ref/hf_in:.3f}  "
          f"hf_abs_refined={hf_ref:.4f}  L_BC={l_bc:.4f}  |Δ|={delta.abs().mean().item():.4f}")


if __name__ == "__main__":
    main()
