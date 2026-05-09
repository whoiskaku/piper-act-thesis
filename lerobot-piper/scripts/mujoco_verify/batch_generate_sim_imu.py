#!/usr/bin/env python3
"""P1 Stream 1：批量把 LeRobot 数据集里每条 episode 的 action 喂给 MuJoCo，采集 sim IMU 存成 npz。

前置：终端 1 已起 `roslaunch piper_mujoco piper_mujoco.launch`。

用法：
  python3 batch_generate_sim_imu.py \\
    --dataset-path ~/.cache/huggingface/lerobot/piper_pick_place \\
    --output-dir ./sim_imu_out

输出：
  sim_imu_out/
    episode_000000_sim_imu.npz   # t(N,) accel(N,3) gyro(N,3)
    episode_000001_sim_imu.npz
    ...
    summary.json                  # 成功/失败列表 + 每条长度
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def find_episodes(dataset_path: Path):
    parquets = sorted(dataset_path.glob("data/chunk-*/episode_*.parquet"))
    if not parquets:
        # 容错：有些数据集把 parquet 放在根下
        parquets = sorted(dataset_path.glob("**/episode_*.parquet"))
    return parquets


def load_action(parquet_path: Path) -> np.ndarray:
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    if "action" not in df.columns:
        raise KeyError(f"'action' column missing in {parquet_path}")
    arr = np.stack(df["action"].to_numpy())
    # 只保留 6/7 维（MuJoCo replay 支持），超出切掉
    if arr.shape[1] > 7:
        arr = arr[:, :7]
    return arr.astype(np.float32)


def run_episode(action: np.ndarray, tmp_action: Path, imu_out: Path,
                ctrl_topic: str, imu_topic: str, fs: float, hold_end_s: float) -> bool:
    np.save(tmp_action, action)
    duration = len(action) / fs + hold_end_s + 1.0

    rec_cmd = [
        sys.executable, str(HERE / "mujoco_imu_recorder.py"),
        "--topic", imu_topic,
        "--output", str(imu_out),
        "--duration", f"{duration:.3f}",
        "--rate", f"{fs}",
    ]
    rec = subprocess.Popen(rec_cmd)
    time.sleep(1.0)

    rep_cmd = [
        sys.executable, str(HERE / "replay_action_mujoco.py"),
        "--actions", str(tmp_action),
        "--topic", ctrl_topic,
        "--rate", f"{fs}",
        "--hold-end-s", f"{hold_end_s}",
    ]
    subprocess.call(rep_cmd)

    try:
        rec.wait(timeout=duration + 10.0)
    except subprocess.TimeoutExpired:
        rec.kill()
        return False

    return imu_out.exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", default="~/.cache/huggingface/lerobot/piper_pick_place")
    ap.add_argument("--output-dir", default=str(HERE / "sim_imu_out"))
    ap.add_argument("--imu-topic", default="/imu/data")
    ap.add_argument("--ctrl-topic", default="/joint_states")
    ap.add_argument("--fs", type=float, default=30.0)
    ap.add_argument("--hold-end-s", type=float, default=0.3)
    ap.add_argument("--max-episodes", type=int, default=-1, help="-1 = all")
    ap.add_argument("--skip-existing", action="store_true", help="跳过 output-dir 里已存在的 episode")
    args = ap.parse_args()

    ds_path = Path(args.dataset_path).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    if not ds_path.exists():
        print(f"[batch] dataset path does not exist: {ds_path}", file=sys.stderr)
        # 搜索 ~/.cache/huggingface/lerobot 下可能的候选
        cache_root = Path("~/.cache/huggingface/lerobot").expanduser()
        if cache_root.exists():
            candidates = [p for p in cache_root.iterdir() if p.is_dir()]
            if candidates:
                print(f"[batch] found candidates under {cache_root}:", file=sys.stderr)
                for c in candidates:
                    print(f"  {c}", file=sys.stderr)
        sys.exit(1)

    episodes = find_episodes(ds_path)
    if not episodes:
        print(f"[batch] no episode_*.parquet found under {ds_path}", file=sys.stderr)
        sys.exit(1)

    if args.max_episodes > 0:
        episodes = episodes[:args.max_episodes]

    print(f"[batch] found {len(episodes)} episodes under {ds_path}")
    print(f"[batch] output -> {out_dir}")

    summary = {"total": len(episodes), "ok": [], "fail": [], "skipped": []}
    t_start = time.time()

    for i, pq in enumerate(episodes):
        ep_id = pq.stem  # episode_000000
        imu_out = out_dir / f"{ep_id}_sim_imu.npz"
        if args.skip_existing and imu_out.exists():
            print(f"[{i+1}/{len(episodes)}] {ep_id}: skip (exists)")
            summary["skipped"].append(ep_id)
            continue

        try:
            action = load_action(pq)
        except Exception as e:
            print(f"[{i+1}/{len(episodes)}] {ep_id}: load failed - {e}", file=sys.stderr)
            summary["fail"].append({"ep": ep_id, "reason": f"load: {e}"})
            continue

        tmp_action = tmp_dir / f"{ep_id}_action.npy"
        t0 = time.time()
        ok = run_episode(
            action, tmp_action, imu_out,
            args.ctrl_topic, args.imu_topic, args.fs, args.hold_end_s,
        )
        elapsed = time.time() - t0

        if not ok:
            print(f"[{i+1}/{len(episodes)}] {ep_id}: FAIL ({elapsed:.1f}s)", file=sys.stderr)
            summary["fail"].append({"ep": ep_id, "reason": "replay/record failed"})
            continue

        try:
            d = np.load(imu_out)
            n = len(d["t"])
        except Exception as e:
            n = -1
        summary["ok"].append({"ep": ep_id, "n_samples": n, "n_action": len(action)})
        print(f"[{i+1}/{len(episodes)}] {ep_id}: OK (action={len(action)}, imu={n}, {elapsed:.1f}s)")

    total_elapsed = time.time() - t_start
    summary["elapsed_s"] = total_elapsed
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== BATCH SUMMARY ===")
    print(f"ok={len(summary['ok'])}  fail={len(summary['fail'])}  skipped={len(summary['skipped'])}")
    print(f"total wall time: {total_elapsed:.1f}s")
    print(f"summary written to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
