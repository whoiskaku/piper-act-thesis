#!/usr/bin/env python3
"""Stage 1.4b — IMU coordinate-frame calibration.

The clean replay validation showed sim physics is qualitatively correct
(|corr|>0.7 on az, gy) but IMU axes are rotated relative to the real
TSND151 mounting on Piper's wrist. This script finds an orthogonal rotation
R such that  R @ sim_imu  ≈  real_imu, separately for accel and gyro
(they should share R if the IMU is a rigid body, but solving separately
lets us cross-check).

Approach: Procrustes orthogonal fit over the same 5 replay episodes from
replay_demo_validation.py.

Outputs:
  outputs/sim/imu_calibration.json   — rotations + residual stats
  outputs/sim/stage1_replay/calibration_compare.png  — pre/post correlations
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
import json

from replay_demo_validation import (
    SIM_DIR, MODEL_PATH, DEMO_DIR, CONTROL_DT,
    load_real_demo, replay_in_sim,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs/sim"
REPLAY_OUT = OUT_DIR / "stage1_replay"


def orthogonal_procrustes_proper(X, Y):
    """Find proper rotation R (det = +1) minimizing ||X R - Y||_F.

    X, Y shape (N, 3). Returns R (3, 3) such that  X @ R  approximates Y.
    Equivalently, real ≈ sim @ R.
    """
    M = X.T @ Y               # (3, 3)
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1] *= -1
        R = U @ Vt
    return R


def fit_quality(X, Y, R):
    """Per-channel correlation and RMS error after rotation."""
    Y_pred = X @ R
    out = {}
    for i, name in enumerate(["x", "y", "z"]):
        if Y[:, i].std() > 1e-9 and Y_pred[:, i].std() > 1e-9:
            corr = float(np.corrcoef(Y[:, i], Y_pred[:, i])[0, 1])
        else:
            corr = 0.0
        rms_err = float(np.sqrt(((Y[:, i] - Y_pred[:, i]) ** 2).mean()))
        out[name] = {"correlation": corr, "rms_error": rms_err}
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPLAY_OUT.mkdir(parents=True, exist_ok=True)

    print(f"[load] {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    ep_indices = [0, 7, 15, 23, 29]
    sim_accel_all, real_accel_all = [], []
    sim_gyro_all, real_gyro_all = [], []

    print(f"\n[collect] replaying {len(ep_indices)} eps")
    for ep_idx in ep_indices:
        real = load_real_demo(ep_idx)
        sim = replay_in_sim(model, data, real, set_initial_pose=True)
        n = min(len(sim["imu"]), len(real["imu"]))
        # Skip first 30 frames (controller startup transient)
        skip = 30
        sim_accel_all.append(sim["imu"][skip:n, :3])
        real_accel_all.append(real["imu"][skip:n, :3])
        sim_gyro_all.append(sim["imu"][skip:n, 3:])
        real_gyro_all.append(real["imu"][skip:n, 3:])
        print(f"  ep {ep_idx}: {n - skip} frames (after 30-frame transient skip)")

    sim_accel = np.concatenate(sim_accel_all, axis=0)
    real_accel = np.concatenate(real_accel_all, axis=0)
    sim_gyro = np.concatenate(sim_gyro_all, axis=0)
    real_gyro = np.concatenate(real_gyro_all, axis=0)
    print(f"\n[fit] total {sim_accel.shape[0]} frames")

    # ----- Accel calibration -----
    # Remove per-channel mean from both (gravity bias differs between sim and real
    # because real IMU may have software-level gravity removal or its mount
    # orientation gives a different DC).
    sim_accel_ac = sim_accel - sim_accel.mean(axis=0, keepdims=True)
    real_accel_ac = real_accel - real_accel.mean(axis=0, keepdims=True)
    R_accel = orthogonal_procrustes_proper(sim_accel_ac, real_accel_ac)
    print(f"\n[accel] R =\n{R_accel}")
    print(f"  det(R) = {np.linalg.det(R_accel):+.4f}")

    quality_accel_pre = fit_quality(sim_accel_ac, real_accel_ac, np.eye(3))
    quality_accel_post = fit_quality(sim_accel_ac, real_accel_ac, R_accel)
    print(f"  pre-cal  channel corrs: {[round(quality_accel_pre[c]['correlation'], 3) for c in 'xyz']}")
    print(f"  post-cal channel corrs: {[round(quality_accel_post[c]['correlation'], 3) for c in 'xyz']}")

    # ----- Gyro calibration -----
    sim_gyro_ac = sim_gyro - sim_gyro.mean(axis=0, keepdims=True)
    real_gyro_ac = real_gyro - real_gyro.mean(axis=0, keepdims=True)
    R_gyro = orthogonal_procrustes_proper(sim_gyro_ac, real_gyro_ac)
    print(f"\n[gyro] R =\n{R_gyro}")
    print(f"  det(R) = {np.linalg.det(R_gyro):+.4f}")

    quality_gyro_pre = fit_quality(sim_gyro_ac, real_gyro_ac, np.eye(3))
    quality_gyro_post = fit_quality(sim_gyro_ac, real_gyro_ac, R_gyro)
    print(f"  pre-cal  channel corrs: {[round(quality_gyro_pre[c]['correlation'], 3) for c in 'xyz']}")
    print(f"  post-cal channel corrs: {[round(quality_gyro_post[c]['correlation'], 3) for c in 'xyz']}")

    # ----- Check whether accel and gyro share rotation -----
    R_diff = R_accel - R_gyro
    max_diff = float(np.abs(R_diff).max())
    print(f"\n[consistency] max |R_accel - R_gyro| = {max_diff:.4f}")
    if max_diff < 0.05:
        print("  ★ R_accel and R_gyro are consistent (rigid IMU body).")
    else:
        print("  ⚠ R_accel and R_gyro differ — using separate rotations for now.")

    # ----- Save -----
    output = {
        "source": "Procrustes orthogonal fit on 5 demo replay eps (skip first 30 transient)",
        "n_frames": int(sim_accel.shape[0]),
        "accel": {
            "R": R_accel.tolist(),
            "det_R": float(np.linalg.det(R_accel)),
            "pre_calibration": quality_accel_pre,
            "post_calibration": quality_accel_post,
        },
        "gyro": {
            "R": R_gyro.tolist(),
            "det_R": float(np.linalg.det(R_gyro)),
            "pre_calibration": quality_gyro_pre,
            "post_calibration": quality_gyro_post,
        },
        "max_abs_R_difference_accel_vs_gyro": max_diff,
        "note": (
            "Apply as: real_imu_predicted = (sim_imu - sim_imu_mean) @ R + real_imu_mean. "
            "The mean offset captures DC (gravity bias) which differs between sim and real "
            "and should be subtracted before rotation."
        ),
    }
    out_path = OUT_DIR / "imu_calibration.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n[save] {out_path}")

    # ----- Plot pre vs post correlation bars -----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    chans = ["x", "y", "z"]
    x = np.arange(3)
    width = 0.35

    for ax, name, pre, post in [
        (axes[0], "Accel", quality_accel_pre, quality_accel_post),
        (axes[1], "Gyro",  quality_gyro_pre,  quality_gyro_post),
    ]:
        pre_corrs = [pre[c]["correlation"] for c in chans]
        post_corrs = [post[c]["correlation"] for c in chans]
        ax.bar(x - width/2, pre_corrs,  width, label="pre-cal",  color="#888")
        ax.bar(x + width/2, post_corrs, width, label="post-cal", color="#2ca02c")
        ax.set_xticks(x); ax.set_xticklabels(chans)
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(f"{name}: sim vs real correlation per channel")
        ax.set_ylabel("Pearson r")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(pre_corrs):
            ax.text(i - width/2, v + 0.04 * np.sign(v) - 0.02, f"{v:+.2f}",
                    ha="center", fontsize=8)
        for i, v in enumerate(post_corrs):
            ax.text(i + width/2, v + 0.04 * np.sign(v) - 0.02, f"{v:+.2f}",
                    ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(REPLAY_OUT / "calibration_compare.png", dpi=130)
    plt.close(fig)
    print(f"[plot] {REPLAY_OUT / 'calibration_compare.png'}")


if __name__ == "__main__":
    main()
