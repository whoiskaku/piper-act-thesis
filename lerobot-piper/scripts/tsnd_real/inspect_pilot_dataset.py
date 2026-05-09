#!/usr/bin/env python3
"""Pilot 录制后校验：observation.imu_real 字段是否正确写入 + 量纲是否合理 + 帧数对齐。"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def inspect_episode(parquet_path: Path):
    df = pd.read_parquet(parquet_path)
    n = len(df)
    cols = set(df.columns)
    print(f"\n--- {parquet_path.name} ---")
    print(f"frames: {n}   columns: {sorted(cols)}")

    for req in ("action", "observation.state", "observation.imu_real"):
        if req not in cols:
            print(f"  MISSING column: {req}")
            return

    action = np.stack(df["action"].to_numpy())
    state = np.stack(df["observation.state"].to_numpy())
    imu = np.stack(df["observation.imu_real"].to_numpy())

    if not (len(action) == len(state) == len(imu) == n):
        print(f"  LENGTH MISMATCH: action={len(action)} state={len(state)} imu={len(imu)} n={n}")
        return

    print(f"  action shape: {action.shape}   state shape: {state.shape}   imu shape: {imu.shape}")
    if imu.shape[1] != 6:
        print(f"  WRONG IMU DIM: {imu.shape[1]} (expected 6)")
        return

    acc = imu[:, :3]
    gyro = imu[:, 3:]
    g_norm = np.linalg.norm(acc, axis=1)
    print(f"  acc  mean={acc.mean(axis=0).round(2).tolist()}  std={acc.std(axis=0).round(2).tolist()}")
    print(f"  gyro mean={gyro.mean(axis=0).round(3).tolist()}  std={gyro.std(axis=0).round(3).tolist()}")
    print(f"  |acc|  mean={g_norm.mean():.2f}  std={g_norm.std():.2f}  "
          f"(合格: mean 在 9.5~11.5, std>0.3)")

    if (acc == 0).all(axis=1).any():
        n_zero = (acc == 0).all(axis=1).sum()
        print(f"  !! {n_zero} all-zero IMU frames — 本不应出现，检查 reader")

    if g_norm.max() > 30 or g_norm.min() < 5:
        print(f"  !! |acc| 超出合理范围 [5, 30]: min={g_norm.min():.2f} max={g_norm.max():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", required=True,
                    help="LeRobot dataset root, e.g. ~/.cache/huggingface/lerobot/local/piper_pick_place_imu_pilot")
    args = ap.parse_args()

    root = Path(args.dataset_path).expanduser().resolve()
    parquets = sorted(root.glob("data/chunk-*/episode_*.parquet"))
    if not parquets:
        print(f"no episode parquet under {root}")
        return

    print(f"found {len(parquets)} episodes under {root}")
    for pq in parquets:
        inspect_episode(pq)


if __name__ == "__main__":
    main()
