#!/usr/bin/env python3
"""Stage 3a 验证 — 负消融对比。

在 val H5 样本上跑 frozen C1-v3 ACT 与 fine-tuned C2-denoising ACT 的输出，比较：
  - hf_fraction (active joints, chunk-level) —— 平滑度（与 Stage 3b 同指标）
  - L1(output, A_clean) —— 任务保真（BC）

预期负结果：C2 的 hf 比 C1 低（更平滑），但 L_BC 比 C1 高（任务保真劣化）。
对照 Stage 3b 的 ACT+RefineNet：hf↓到 0.25× 且 L_BC 不升反降 —— 反衬 frozen+RefineNet 的优势。

用法:  python sim/eval_stage3a.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lerobot.common.datasets.factory import make_dataset
from sim.precompute_act_output import load_frozen_act, gather_batch
from sim.refine_net_dataset import RefinePhase1Dataset
from sim.train_refine_net import hf_fraction_per_joint, ACTIVE_JOINTS

C1_CKPT = REPO / "outputs/train/2026-05-08/C1_v3_history_seed42/checkpoints/070000/pretrained_model"
C2_CKPT = REPO / "outputs/train/C2_denoising_act/lam0.1/checkpoints/020000/pretrained_model"
DATASET_REPO = "local/piper_pick_place_imu_v3"
H5 = REPO / "outputs/sim/phase1_paired_data.h5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BS = 32


def run_policy_on_val(policy, dataset, ep_from, src_ep, src_start, n):
    outs = []
    for b in range(0, n, BS):
        idxs = [int(ep_from[src_ep[i]] + src_start[i]) for i in range(b, min(b + BS, n))]
        samples = [dataset[i] for i in idxs]
        batch = gather_batch(samples)
        with torch.no_grad():
            bi = policy.normalize_inputs(batch)
            if policy.config.image_features:
                bi = dict(bi)
                bi["observation.images"] = torch.stack(
                    [bi[k] for k in policy.config.image_features], dim=-4)
            bi.pop("action", None); bi.pop("action_is_pad", None)
            out = policy.model(bi)
            act = out[0] if isinstance(out, (list, tuple)) else out
            act = policy.unnormalize_outputs({"action": act})["action"]
        outs.append(act.cpu().numpy().astype(np.float32))
    return np.concatenate(outs)


def metrics(out, A_clean):
    """active-joint hf_fraction (chunk mean) + L1 to A_clean."""
    t = torch.from_numpy(out)
    hf = hf_fraction_per_joint(t)[:, ACTIVE_JOINTS].mean().item()
    l1 = np.abs(out - A_clean).mean()
    return hf, float(l1)


def main():
    print(f"[load] C1 frozen + dataset")
    c1, cfg = load_frozen_act(C1_CKPT, DATASET_REPO, DEVICE)
    dataset = make_dataset(cfg)
    ep_from = dataset.episode_data_index["from"].numpy()
    print(f"[load] C2 denoise")
    c2, _ = load_frozen_act(C2_CKPT, DATASET_REPO, DEVICE)

    vds = RefinePhase1Dataset(H5, "val")
    import h5py
    with h5py.File(H5, "r") as f:
        src_ep = f["val"]["src_ep"][:]; src_start = f["val"]["src_start"][:]
    A_clean = vds.A_clean
    n = len(src_ep)
    print(f"[val] n={n}")

    out_c1 = run_policy_on_val(c1, dataset, ep_from, src_ep, src_start, n)
    out_c2 = run_policy_on_val(c2, dataset, ep_from, src_ep, src_start, n)

    hf1, l1_1 = metrics(out_c1, A_clean)
    hf2, l1_2 = metrics(out_c2, A_clean)
    hf_clean, _ = metrics(A_clean, A_clean)

    print(f"\n{'='*72}")
    print("Stage 3a 负消融 — val (active joints j1/j2/j3/grip)")
    print(f"{'='*72}")
    print(f"{'model':>22} {'hf_fraction':>12} {'hf_ratio/ACT':>13} {'L_BC':>9}")
    print("-" * 72)
    print(f"{'C1 frozen ACT':>22} {hf1:>12.4f} {1.000:>13.3f} {l1_1:>9.4f}")
    print(f"{'C2 denoise-ACT(λ0.1)':>22} {hf2:>12.4f} {hf2/hf1:>13.3f} {l1_2:>9.4f}")
    print(f"{'A_clean(floor)':>22} {hf_clean:>12.4f} {hf_clean/hf1:>13.3f} {0.0:>9.4f}")
    print(f"{'(3b ACT+RefineNet)':>22} {'~0.0085':>12} {'~0.25':>13} {'0.0046':>9}")
    print("-" * 72)
    dhf = (hf1 - hf2) / hf1 * 100
    dbc = (l1_2 - l1_1) / l1_1 * 100
    print(f"C2 vs C1:  hf {('-' if dhf>0 else '+')}{abs(dhf):.0f}%  |  L_BC {('+' if dbc>0 else '-')}{abs(dbc):.0f}%")
    verdict = ("负结果成立: 改 ACT 换来平滑↓但任务保真↓"
               if (hf2 < hf1 and l1_2 > l1_1 * 1.05)
               else "需检视: 未出现明显 平滑↑/保真↓ 取舍")
    print(f"判读: {verdict}")


if __name__ == "__main__":
    main()
