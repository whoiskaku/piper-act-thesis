#!/usr/bin/env python3
"""Stage 2.5 — Precompute frozen C1-v3 ACT_output for every paired sample.

For each (src_ep, src_start) tuple stored in phase1_paired_data.h5, look up
the matching frame in the v3 LeRobotDataset (with history wrapping applied,
as during C1-v3 training), run a frozen ACT forward pass, and store the
predicted 100-step action chunk back into the H5 as the new dataset
`ACT_output` (shape N, 100, 7).

This makes Stage 3b's RefineNet training a simple supervised problem:
  RefineNet(ACT_output, history_action, history_imu, current_state) → ΔA
  refined = ACT_output + ΔA   (residual)
  loss = L1(refined, A_clean) + λ × jerk_RMS(refined)

Per RESEARCH_PLAN_v4.3 Principle 2, ACT stays frozen throughout.
"""
import json
import sys
import time
import tempfile
from pathlib import Path

import draccus
import h5py
import numpy as np
import torch

# Make the in-repo lerobot package importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.factory import make_policy
from lerobot.configs.train import TrainPipelineConfig


C1_CKPT = REPO_ROOT / "outputs/train/2026-05-08/C1_v3_history_seed42/checkpoints/070000/pretrained_model"
DATASET_REPO = "local/piper_pick_place_imu_v3"
H5_PATH = REPO_ROOT / "outputs/sim/phase1_paired_data.h5"

BATCH_SIZE = 32   # tune by GPU memory
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_frozen_act(ckpt_path: Path, dataset_repo: str, device: str):
    """Mirror load_actor_with_value_head from phase2_train.py but without VH."""
    train_cfg_dict = json.load(open(ckpt_path / "train_config.json"))
    # Keep value head off — we don't need it for inference.
    if "enable_value_head" in train_cfg_dict["policy"]:
        train_cfg_dict["policy"]["enable_value_head"] = False
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(train_cfg_dict, f)
        tmp = f.name
    cfg = draccus.parse(TrainPipelineConfig, tmp, args=[])
    cfg.policy.pretrained_path = str(ckpt_path)

    ds_meta = LeRobotDatasetMetadata(dataset_repo, root=None)
    policy = make_policy(cfg=cfg.policy, ds_meta=ds_meta, device=device)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad = False
    return policy, cfg


def build_ep_frame_to_global(dataset):
    """Build (ep_idx, frame_in_ep) → global_idx for fast lookup."""
    ep_from = dataset.episode_data_index["from"].numpy()
    ep_to = dataset.episode_data_index["to"].numpy()
    # For each ep, global_idx ranges [ep_from[ep], ep_to[ep]).
    # frame_in_ep starts at 0; global_idx = ep_from[ep] + frame_in_ep.
    return ep_from, ep_to


def gather_batch(samples):
    """Stack a list of per-sample dicts into a batched dict of tensors on device."""
    # Each sample is a dict from dataset[i]; we keep only the keys ACT needs.
    keys_needed = []
    sample0 = samples[0]
    for k, v in sample0.items():
        if isinstance(v, torch.Tensor):
            keys_needed.append(k)
    # Exclude GT action so VAE encoder takes the eval path (z = 0).
    keys_needed = [k for k in keys_needed if k != "action"]

    batch = {}
    for k in keys_needed:
        batch[k] = torch.stack([s[k] for s in samples], dim=0).to(DEVICE, non_blocking=True)
    return batch


def main():
    print(f"[device] {DEVICE}")
    print(f"[load] policy from {C1_CKPT}")
    policy, cfg = load_frozen_act(C1_CKPT, DATASET_REPO, DEVICE)
    print(f"  enable_history_input = {getattr(cfg.policy, 'enable_history_input', False)}")
    print(f"  chunk_size = {cfg.policy.chunk_size}")

    print(f"[load] dataset {DATASET_REPO}")
    dataset = make_dataset(cfg)
    ep_from, ep_to = build_ep_frame_to_global(dataset)
    print(f"  total samples: {len(dataset)}, episodes: {len(ep_from)}")

    # Open H5 for read + append
    print(f"\n[h5] open {H5_PATH}")
    with h5py.File(H5_PATH, "a") as f:
        for split in ("train", "val"):
            grp = f[split]
            n = grp.attrs.get("n", None) or grp["A_clean"].shape[0]
            src_ep = grp["src_ep"][:]
            src_start = grp["src_start"][:]
            chunk_size = grp["A_clean"].shape[1]

            print(f"\n[{split}] N = {n}, chunk_size = {chunk_size}")

            # Resolve global dataset indices for every sample
            global_idxs = ep_from[src_ep] + src_start
            # Safety: must be within ep_to[ep] - chunk_size + 1
            ep_lengths = ep_to[src_ep] - ep_from[src_ep]
            assert (src_start + chunk_size <= ep_lengths).all(), \
                "chunk extends past episode end"

            # Allocate output
            if "ACT_output" in grp:
                del grp["ACT_output"]
            act_out = grp.create_dataset(
                "ACT_output", shape=(n, chunk_size, 7),
                dtype="float32", compression="gzip", compression_opts=4,
            )

            t0 = time.time()
            written = 0
            for b_start in range(0, n, BATCH_SIZE):
                b_end = min(b_start + BATCH_SIZE, n)
                samples = [dataset[int(global_idxs[i])] for i in range(b_start, b_end)]
                batch = gather_batch(samples)

                with torch.no_grad():
                    # Mirror select_action's input prep (modeling_act.py:120-135):
                    batch_in = policy.normalize_inputs(batch)
                    if policy.config.image_features:
                        batch_in = dict(batch_in)
                        batch_in["observation.images"] = torch.stack(
                            [batch_in[k] for k in policy.config.image_features],
                            dim=-4,
                        )
                    # Drop "action" if present — keeps model in eval/inference path
                    batch_in.pop("action", None)
                    batch_in.pop("action_is_pad", None)

                    out = policy.model(batch_in)
                    actions = out[0] if isinstance(out, (list, tuple)) else out
                    actions = policy.unnormalize_outputs(
                        {"action": actions}
                    )["action"]
                actions = actions.detach().cpu().numpy().astype(np.float32)
                act_out[b_start:b_end] = actions
                written += b_end - b_start

                if b_start // BATCH_SIZE % 5 == 0:
                    elapsed = time.time() - t0
                    rate = written / max(elapsed, 1e-9)
                    eta = (n - written) / max(rate, 1e-9)
                    print(f"  [{split}] {written}/{n}  "
                          f"({rate:.1f}/s, eta {eta:.0f}s)")

            elapsed = time.time() - t0
            print(f"  [{split}] done in {elapsed:.1f}s, mean {act_out[:].mean():+.4f}, std {act_out[:].std():.4f}")

    print(f"\n[done] ACT_output saved into {H5_PATH}")
    # Report file size
    print(f"  file size: {H5_PATH.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
