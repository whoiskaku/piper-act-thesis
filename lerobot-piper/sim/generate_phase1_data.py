#!/usr/bin/env python3
"""Stage 2 — Generate Phase 1 sim paired training data.

Per RESEARCH_PLAN_v4.3 §5 Stage 2 (Phase 1 sim, no object). Seeds chunks from
v3 demos with optional amplitude perturbations, runs each through
NoisyPiperSim.execute_action_chunk() to get 6-tuple paired data, saves as HDF5.

Split:
  train: 24 v3 demos (eps 0..23)
  val:    6 v3 demos (eps 24..29)

For each demo:
  - sliding window over action sequence (stride = chunk_size // 2 = 50)
  - per resulting chunk, generate K_PERTURB variants:
      variant 0: identity (no perturbation, only sim noise)
      variants 1..K-1: per-joint amplitude scaling, factor U[1-AMP, 1+AMP]

Output (single HDF5 file per RESEARCH_PLAN_v4.3 §5):
  outputs/sim/phase1_paired_data.h5

Datasets per split (train/, val/):
  A_clean:    (N, 100, 8)  perturbed demo action (target)
  A_noisy:    (N, 100, 8)  motor-noise-executed action in sim
  IMU_clean:  (N, 100, 6)  sim IMU without sensor noise / vibration
  IMU_noisy:  (N, 100, 6)  sim IMU with all 3 noise layers
  qpos_clean: (N, 100, 8)  joint trajectory under clean execution
  qpos_noisy: (N, 100, 8)  joint trajectory under noisy execution
  start_qpos: (N, 8)       initial joint configuration
  src_ep:     (N,)          source v3 episode index
  src_start:  (N,)          start frame index within source episode
  perturb_id: (N,)          0 = identity, k>0 = perturbed variant id
  perturb_scale: (N, 7)     per-joint amplitude multiplier (1.0 if perturb_id==0)
"""
from pathlib import Path

import h5py
import numpy as np

from replay_demo_validation import (
    DEMO_DIR, load_real_demo, state_mdeg_to_rad,
)
from test_noisy_sim import real_action_to_ctrl
from noisy_piper_sim import NoisyPiperSim


SIM_DIR = Path(__file__).resolve().parent
MODEL_PATH = SIM_DIR / "piper_model.xml"
NOISE_PARAMS = Path(__file__).resolve().parents[1] / "outputs/sim/noise_params.json"
OUT_PATH = Path(__file__).resolve().parents[1] / "outputs/sim/phase1_paired_data.h5"

CHUNK_SIZE = 100
STRIDE = CHUNK_SIZE // 2     # 50 = overlapping windows
K_PERTURB = 4                 # 1 identity + 3 perturbed
AMP_RANGE = 0.10              # per-joint scale ∈ U[0.90, 1.10]

TRAIN_EPS = list(range(0, 24))
VAL_EPS = list(range(24, 30))


def slice_chunks(real_demo, chunk_size: int, stride: int):
    """Yield (start_idx, action_chunk, start_qpos_8) tuples from a demo."""
    action_7 = real_demo["action"]          # (T, 7)
    state_mdeg = real_demo["state_mdeg"]    # (T, 7)
    T = len(action_7)
    for start in range(0, T - chunk_size + 1, stride):
        action_chunk_7 = action_7[start:start + chunk_size]
        ctrl_chunk_8 = real_action_to_ctrl(action_chunk_7)        # (100, 8)
        start_qpos = state_mdeg_to_rad(state_mdeg[start])         # (8,)
        yield start, ctrl_chunk_8, start_qpos


def perturb_action(ctrl_chunk_8: np.ndarray, rng: np.random.Generator):
    """Per-joint amplitude scaling around the chunk's own mean.

    A_perturbed[t, j] = mean_j + scale_j × (A[t, j] - mean_j)
    This preserves the "rest position" and only perturbs the dynamic range.

    Returns (perturbed_chunk, scale_per_joint_7), where scale 7-dim corresponds
    to the v3 joints (j1..j6 + grip). Gripper finger 8 mirrors finger 7.
    """
    scale_7 = rng.uniform(1.0 - AMP_RANGE, 1.0 + AMP_RANGE, size=7).astype(np.float32)
    out = ctrl_chunk_8.copy()
    means = out.mean(axis=0)
    # arm joints 0..5 use scale_7[0..5]
    for j in range(6):
        out[:, j] = means[j] + scale_7[j] * (out[:, j] - means[j])
    # gripper actuator 6 uses scale_7[6]; actuator 7 = -actuator 6
    out[:, 6] = means[6] + scale_7[6] * (out[:, 6] - means[6])
    # Clip gripper to valid range [0, 0.035] AFTER perturbation
    out[:, 6] = np.clip(out[:, 6], 0.0, 0.035)
    out[:, 7] = -out[:, 6]
    return out, scale_7


def generate_split(sim, demo_eps, chunk_size, stride, k_perturb,
                    base_seed: int, label: str):
    """Process a list of demo episodes and return concatenated arrays."""
    A_clean_list, A_noisy_list = [], []
    IMU_clean_list, IMU_noisy_list = [], []
    qpos_clean_list, qpos_noisy_list = [], []
    start_qpos_list = []
    src_ep_list, src_start_list = [], []
    perturb_id_list, perturb_scale_list = [], []

    rng = np.random.default_rng(base_seed)

    for ep_idx in demo_eps:
        real = load_real_demo(ep_idx)
        chunks_in_ep = 0
        for start, ctrl_chunk_8, start_qpos in slice_chunks(real, chunk_size, stride):
            for variant in range(k_perturb):
                if variant == 0:
                    perturbed = ctrl_chunk_8.copy()
                    scale_7 = np.ones(7, dtype=np.float32)
                else:
                    perturbed, scale_7 = perturb_action(ctrl_chunk_8, rng)

                # Common seed combines ep + start + variant for reproducibility
                common_seed = base_seed + ep_idx * 10000 + start * 10 + variant
                result = sim.execute_action_chunk(
                    start_qpos, perturbed, common_seed=common_seed
                )
                A_clean_list.append(result["A_clean"].astype(np.float32))
                A_noisy_list.append(result["A_noisy"].astype(np.float32))
                IMU_clean_list.append(result["IMU_clean"].astype(np.float32))
                IMU_noisy_list.append(result["IMU_noisy"].astype(np.float32))
                qpos_clean_list.append(result["qpos_clean"].astype(np.float32))
                qpos_noisy_list.append(result["qpos_noisy"].astype(np.float32))
                start_qpos_list.append(start_qpos.astype(np.float32))
                src_ep_list.append(ep_idx)
                src_start_list.append(start)
                perturb_id_list.append(variant)
                perturb_scale_list.append(scale_7)
                chunks_in_ep += 1

        print(f"  [{label}] ep {ep_idx:02d}: {chunks_in_ep} samples")

    pack = {
        "A_clean":     np.stack(A_clean_list),
        "A_noisy":     np.stack(A_noisy_list),
        "IMU_clean":   np.stack(IMU_clean_list),
        "IMU_noisy":   np.stack(IMU_noisy_list),
        "qpos_clean":  np.stack(qpos_clean_list),
        "qpos_noisy":  np.stack(qpos_noisy_list),
        "start_qpos":  np.stack(start_qpos_list),
        "src_ep":      np.array(src_ep_list, dtype=np.int32),
        "src_start":   np.array(src_start_list, dtype=np.int32),
        "perturb_id":  np.array(perturb_id_list, dtype=np.int32),
        "perturb_scale": np.stack(perturb_scale_list),
    }
    return pack


def write_split(group, pack):
    for key, val in pack.items():
        group.create_dataset(key, data=val, compression="gzip", compression_opts=4)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[init] sim model = {MODEL_PATH}")
    print(f"[init] noise params = {NOISE_PARAMS}")

    sim = NoisyPiperSim(MODEL_PATH, NOISE_PARAMS, seed=0)
    print(f"[sim] chunk_size={CHUNK_SIZE}, stride={STRIDE}, k_perturb={K_PERTURB}, "
          f"amp_range=±{AMP_RANGE}")

    print(f"\n[generate] train split ({len(TRAIN_EPS)} eps)")
    import time
    t0 = time.time()
    train_pack = generate_split(sim, TRAIN_EPS, CHUNK_SIZE, STRIDE, K_PERTURB,
                                  base_seed=1000, label="train")
    print(f"  train total: {train_pack['A_clean'].shape[0]} samples "
          f"({time.time() - t0:.1f}s)")

    print(f"\n[generate] val split ({len(VAL_EPS)} eps)")
    t0 = time.time()
    val_pack = generate_split(sim, VAL_EPS, CHUNK_SIZE, STRIDE, K_PERTURB,
                                base_seed=2000, label="val")
    print(f"  val total: {val_pack['A_clean'].shape[0]} samples "
          f"({time.time() - t0:.1f}s)")

    print(f"\n[save] {OUT_PATH}")
    with h5py.File(OUT_PATH, "w") as f:
        # Top-level attrs
        f.attrs["version"] = "v4.3-stage2-phase1"
        f.attrs["chunk_size"] = CHUNK_SIZE
        f.attrs["stride"] = STRIDE
        f.attrs["k_perturb"] = K_PERTURB
        f.attrs["amp_range"] = AMP_RANGE
        f.attrs["fps"] = 30
        f.attrs["control_dt"] = 1.0 / 30.0
        f.attrs["n_train"] = train_pack["A_clean"].shape[0]
        f.attrs["n_val"] = val_pack["A_clean"].shape[0]

        train_g = f.create_group("train")
        val_g = f.create_group("val")
        write_split(train_g, train_pack)
        write_split(val_g, val_pack)

    # Report file size
    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"\n[done] file size: {size_mb:.1f} MB")
    print(f"       train: {train_pack['A_clean'].shape[0]} samples")
    print(f"       val:   {val_pack['A_clean'].shape[0]} samples")

    # Quick sanity stats from train split
    print("\nQuick sanity (train split, mean across all samples):")
    for key in ["A_clean", "A_noisy", "IMU_clean", "IMU_noisy"]:
        arr = train_pack[key]
        print(f"  {key:11s} shape {arr.shape}, mean {arr.mean():+.4f}, std {arr.std():.4f}")


if __name__ == "__main__":
    main()
