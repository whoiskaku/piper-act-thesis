#!/usr/bin/env python3
"""验证实验编排。

前置：另起终端 roslaunch piper_mujoco piper_mujoco.launch 已跑起来，/imu/data 正在发。
本脚本：
  1) 拿一条 base action 序列（从 LeRobot 数据集读；读不到则合成）
  2) 生成 A(raw) / B(smooth) / C(jitter) 三个版本
  3) 依次跑 (recorder 后台, replayer 前台) 三遍，得到 imu_A.npz / B / C
  4) 调 analyze_mujoco_response.py 产出判据

用法：
  python run_verification_experiment.py \\
      --dataset-path ~/.cache/huggingface/lerobot/piper_pick_place \\
      --output-dir ./out_verify
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


HERE = Path(__file__).resolve().parent


def load_base_action(dataset_path: str, num_frames: int) -> np.ndarray:
    """尝试读 LeRobot 数据集一条 episode 的 action；失败则合成。
    返回 shape (T, 6)。"""
    p = Path(dataset_path).expanduser()
    try:
        if p.exists():
            # LeRobot 数据集：data/chunk-000/episode_000000.parquet
            parquets = sorted(p.glob("data/chunk-*/episode_*.parquet"))
            if parquets:
                import pandas as pd
                df = pd.read_parquet(parquets[0])
                if "action" in df.columns:
                    actions = np.stack(df["action"].to_numpy())
                    actions = actions[:, :6] if actions.shape[1] >= 6 else actions
                    T = min(num_frames, len(actions))
                    print(f"[orchestrator] loaded real demo action from {parquets[0]} ({T} frames)")
                    return actions[:T].astype(np.float32)
    except Exception as e:
        print(f"[orchestrator] dataset load failed ({e}); falling back to synthetic", file=sys.stderr)

    # 合成：两个 joint 做慢正弦，其它保持中位
    print(f"[orchestrator] synthesizing base action ({num_frames} frames)")
    t = np.linspace(0.0, num_frames / 30.0, num_frames)
    base = np.zeros((num_frames, 6), dtype=np.float32)
    base[:, 0] = 0.5 * np.sin(2 * np.pi * 0.3 * t)         # joint1 慢扫
    base[:, 1] = 1.2 + 0.3 * np.sin(2 * np.pi * 0.25 * t)  # joint2 慢扫（避开下限0）
    base[:, 2] = -1.0 + 0.2 * np.sin(2 * np.pi * 0.3 * t)  # joint3
    base[:, 3] = 0.0
    base[:, 4] = 0.0
    base[:, 5] = 0.0
    return base


def make_smoothed(a: np.ndarray) -> np.ndarray:
    return savgol_filter(a, window_length=15, polyorder=3, axis=0)


def make_jittered(a: np.ndarray, freq_hz: float = 8.0, amp_rad: float = 0.02,
                  joint_idx: int = 3, fs: float = 30.0) -> np.ndarray:
    out = a.copy()
    t = np.arange(len(a)) / fs
    out[:, joint_idx] += amp_rad * np.sin(2 * np.pi * freq_hz * t)
    return out


def run_one(tag: str, action: np.ndarray, out_dir: Path, imu_topic: str,
            ctrl_topic: str, fs: float, hold_end_s: float):
    action_path = out_dir / f"action_{tag}.npy"
    imu_path = out_dir / f"imu_{tag}.npz"
    np.save(action_path, action)

    duration = len(action) / fs + hold_end_s + 0.5
    print(f"\n[orchestrator] === run {tag}: duration={duration:.2f}s ===")

    rec_cmd = [
        sys.executable, str(HERE / "mujoco_imu_recorder.py"),
        "--topic", imu_topic,
        "--output", str(imu_path),
        "--duration", f"{duration:.3f}",
        "--rate", f"{fs}",
    ]
    rec = subprocess.Popen(rec_cmd)
    time.sleep(1.0)  # 让 recorder 先上

    rep_cmd = [
        sys.executable, str(HERE / "replay_action_mujoco.py"),
        "--actions", str(action_path),
        "--topic", ctrl_topic,
        "--rate", f"{fs}",
        "--hold-end-s", f"{hold_end_s}",
    ]
    rep_ret = subprocess.call(rep_cmd)
    if rep_ret != 0:
        print(f"[orchestrator] replayer exit code {rep_ret}", file=sys.stderr)

    rec.wait(timeout=duration + 5.0)
    if not imu_path.exists():
        raise RuntimeError(f"IMU recording failed for {tag}: {imu_path} not produced")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", default="~/.cache/huggingface/lerobot/piper_pick_place")
    ap.add_argument("--output-dir", default=str(HERE / "out_verify"))
    ap.add_argument("--num-frames", type=int, default=120, help="4s @ 30Hz")
    ap.add_argument("--imu-topic", default="/imu/data")
    ap.add_argument("--ctrl-topic", default="/joint_states",
                    help="launch 文件 remap 过，默认 /joint_states；裸节点用 /mujoco_joint_states_ctrl")
    ap.add_argument("--fs", type=float, default=30.0)
    ap.add_argument("--hold-end-s", type=float, default=0.5)
    ap.add_argument("--jitter-freq", type=float, default=8.0)
    ap.add_argument("--jitter-amp", type=float, default=0.02)
    ap.add_argument("--jitter-joint", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base = load_base_action(args.dataset_path, args.num_frames)
    a_raw = base
    a_smooth = make_smoothed(base)
    a_jitter = make_jittered(base, args.jitter_freq, args.jitter_amp, args.jitter_joint, args.fs)

    for tag, act in [("A", a_raw), ("B", a_smooth), ("C", a_jitter)]:
        run_one(tag, act, out_dir, args.imu_topic, args.ctrl_topic, args.fs, args.hold_end_s)

    analyze_cmd = [
        sys.executable, str(HERE / "analyze_mujoco_response.py"),
        "--imu-a", str(out_dir / "imu_A.npz"),
        "--imu-b", str(out_dir / "imu_B.npz"),
        "--imu-c", str(out_dir / "imu_C.npz"),
        "--output-dir", str(out_dir),
        "--fs", f"{args.fs}",
    ]
    subprocess.call(analyze_cmd)


if __name__ == "__main__":
    main()
