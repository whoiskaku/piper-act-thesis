#!/usr/bin/env python3
"""P0.4-D1 — IMU 坐标系逐数据集自动标定(Kabsch 旋转 + 加计零偏)。

背景: 绑带式 TSND 的安装朝向逐会话漂移(实测 pp2-demo 4.85° vs pp2-deploy 6.68°,
euler 主分量一致=真实偏差), XML site quat 无唯一真值 → 不改模型,
在【对比层】对每个数据集拟合 (R, b): real ≈ R @ ideal + b(仅acc)。

方法: gyro 对(真旋转样本, 免重力/零偏) Kabsch 解 R; acc 残差均值给 b。
用法: python sim/imu_frame_calib.py --repo enolab-guo/pickplace2_train --lag 2
输出: outputs/sim/ideal_imu/<repo>/frame_calib.json  (R 3x3, b 3, 诊断量)
P1 用法: real_aligned = real; ideal_aligned = ideal @ R.T + [b,0] 后再算差。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from compare_ideal_methods import kin_ideal_20

OUT_ROOT = Path(__file__).resolve().parents[1] / "outputs/sim/ideal_imu"


def calibrate(repo: str, eps: list[int], lag: int, gyro_min: float = 0.15):
    Gi, Gr, Ai, Ar = [], [], [], []
    for ep in eps:
        kin20, d, _ = kin_ideal_20(repo, ep)
        T = min(len(d["imu_real"]), len(kin20))
        ideal, real = kin20[:T], d["imu_real"][:T]
        gi, gr = ideal[:T - lag, 3:], real[lag:, 3:]
        m = np.linalg.norm(gi, axis=1) > gyro_min
        Gi.append(gi[m]); Gr.append(gr[m])
        Ai.append(ideal[:T - lag, :3]); Ar.append(real[lag:, :3])
    gi, gr, ai, ar = (np.concatenate(x) for x in (Gi, Gr, Ai, Ar))
    H = gi.T @ gr
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T                      # real ≈ R @ ideal
    b = (ar - ai @ R.T).mean(0)             # acc 零偏(含重力标定误差)
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    diag = {
        "n_gyro_pairs": int(len(gi)), "rot_deg": ang,
        "gyro_rms_before": float(np.sqrt(((gr - gi) ** 2).mean())),
        "gyro_rms_after": float(np.sqrt(((gr - gi @ R.T) ** 2).mean())),
        "acc_rms_before": float(np.sqrt(((ar - ai) ** 2).mean())),
        "acc_rms_after": float(np.sqrt(((ar - ai @ R.T - b) ** 2).mean())),
    }
    return R, b, diag


def apply_calib(ideal6: np.ndarray, R: np.ndarray, b: np.ndarray) -> np.ndarray:
    """把标定应用到 ideal(与 real 对齐同一坐标系): (T,6) → (T,6)。"""
    out = np.empty_like(ideal6)
    out[:, :3] = ideal6[:, :3] @ R.T + b
    out[:, 3:] = ideal6[:, 3:] @ R.T
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--eps", type=int, default=10)
    ap.add_argument("--lag", type=int, required=True, help="real 相对 ideal 的滞后帧数(demo≈2, deploy≈3)")
    args = ap.parse_args()
    R, b, diag = calibrate(args.repo, list(range(args.eps)), args.lag)
    out_dir = OUT_ROOT / args.repo.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"repo": args.repo, "lag": args.lag, "R": R.tolist(), "b": b.tolist(), **diag}
    (out_dir / "frame_calib.json").write_text(json.dumps(payload, indent=1))
    print(f"旋转修正 {diag['rot_deg']:.2f}°  acc零偏 {np.round(b,3)}")
    print(f"gyro RMS {diag['gyro_rms_before']:.4f}→{diag['gyro_rms_after']:.4f}   "
          f"acc RMS {diag['acc_rms_before']:.3f}→{diag['acc_rms_after']:.3f}")
    print(f"→ {out_dir/'frame_calib.json'}")


if __name__ == "__main__":
    main()
