#!/usr/bin/env python3
"""Stage 2.6 — Precompute history_action / history_imu / history_pad_mask to H5.

For each H5 sample at (src_ep, src_start), use the same EpisodeHistoryDataset
wrapper that C1-v3 was trained with to extract the right-aligned, padded
600-frame history (action 7-dim, IMU 6-dim, pad mask 600-dim). Save back to
phase1_paired_data.h5 so Stage 3b can load history directly without redoing
the wrapping at every batch.

Per RESEARCH_PLAN_v4.3 Principle 3 (sim-train / real-deploy):
  Deploy time → history_buffer provides the previous episode's full action/IMU.
  Train time → here we reuse the dataset's "previous episode" history selection
  so distribution matches deploy.

Adds to each split:
  history_action   (N, 600, 7)
  history_imu      (N, 600, 6)
  history_pad_mask (N, 600)       True = padding, False = real frame
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.factory import make_policy
from lerobot.configs.train import TrainPipelineConfig

C1_CKPT = REPO_ROOT / "outputs/train/2026-05-08/C1_v3_history_seed42/checkpoints/070000/pretrained_model"
DATASET_REPO = "local/piper_pick_place_imu_v3"
H5_PATH = REPO_ROOT / "outputs/sim/phase1_paired_data.h5"


def load_cfg_only(ckpt_path: Path):
    train_cfg_dict = json.load(open(ckpt_path / "train_config.json"))
    if "enable_value_head" in train_cfg_dict["policy"]:
        train_cfg_dict["policy"]["enable_value_head"] = False
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(train_cfg_dict, f)
        tmp = f.name
    cfg = draccus.parse(TrainPipelineConfig, tmp, args=[])
    cfg.policy.pretrained_path = str(ckpt_path)
    return cfg


def main():
    print(f"[load] cfg from {C1_CKPT}")
    cfg = load_cfg_only(C1_CKPT)
    print(f"  history_max_length = {cfg.policy.history_max_length}")
    print(f"  history_use_action = {cfg.policy.history_use_action}")
    print(f"  history_use_imu    = {cfg.policy.history_use_imu}")

    print(f"[load] dataset {DATASET_REPO}")
    dataset = make_dataset(cfg)
    ep_from = dataset.episode_data_index["from"].numpy()
    ep_to = dataset.episode_data_index["to"].numpy()
    print(f"  total samples: {len(dataset)}, episodes: {len(ep_from)}")

    L = cfg.policy.history_max_length

    # Detect which keys EpisodeHistoryDataset provides for history
    sample0 = dataset[0]
    h_action_key = "observation.history_action"
    h_imu_key    = "observation.history_imu"
    h_mask_key   = "history_pad_mask"
    for k in (h_action_key, h_imu_key, h_mask_key):
        if k not in sample0:
            raise RuntimeError(f"Expected key '{k}' in dataset sample, got keys: {list(sample0.keys())}")
    action_dim = sample0[h_action_key].shape[-1]
    imu_dim    = sample0[h_imu_key].shape[-1]
    print(f"  history_action shape: ({L}, {action_dim})")
    print(f"  history_imu    shape: ({L}, {imu_dim})")

    print(f"\n[h5] open {H5_PATH}")
    with h5py.File(H5_PATH, "a") as f:
        for split in ("train", "val"):
            grp = f[split]
            src_ep = grp["src_ep"][:]
            src_start = grp["src_start"][:]
            n = len(src_ep)
            print(f"\n[{split}] N = {n}")

            # Allocate output datasets (replace if exist)
            for key, shape, dtype in [
                ("history_action",   (n, L, action_dim), "float32"),
                ("history_imu",      (n, L, imu_dim),    "float32"),
                ("history_pad_mask", (n, L),              "bool"),
            ]:
                if key in grp:
                    del grp[key]
                grp.create_dataset(
                    key, shape=shape, dtype=dtype,
                    compression="gzip", compression_opts=4,
                )

            t0 = time.time()
            for i in range(n):
                global_idx = int(ep_from[int(src_ep[i])] + int(src_start[i]))
                sample = dataset[global_idx]
                grp["history_action"][i]   = sample[h_action_key].numpy().astype(np.float32)
                grp["history_imu"][i]      = sample[h_imu_key].numpy().astype(np.float32)
                grp["history_pad_mask"][i] = sample[h_mask_key].numpy().astype(bool)
                if i % 100 == 0 and i > 0:
                    elapsed = time.time() - t0
                    rate = i / elapsed
                    eta = (n - i) / rate
                    print(f"  [{split}] {i}/{n}  ({rate:.1f}/s, eta {eta:.0f}s)")

            elapsed = time.time() - t0
            print(f"  [{split}] done in {elapsed:.1f}s")
            # Quick stats
            mask = grp["history_pad_mask"][:]
            print(f"    pad fraction: {mask.mean():.3f}  "
                  f"(0=no pad full L=600 real, 1=all pad)")

    print(f"\n[done] {H5_PATH}")
    print(f"  file size: {H5_PATH.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
