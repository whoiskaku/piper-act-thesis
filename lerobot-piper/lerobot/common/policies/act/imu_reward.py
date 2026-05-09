"""IMURewardComputer: static state→IMU GT lookup table for Phase 2 RL reward.

Built once offline (scripts/d2_setup_imu_reward.py) and loaded at training /
deployment time. Holds:
  - KDTree over normalized v2 demo states
  - Cached per-node IMU window (next chunk_size frames, normalized)
  - Per-demo tagging so we can exclude same-demo neighbors for intra-NN stats
  - Normalization stats (state, IMU)
  - OOD distance threshold τ

At inference time:
  state ──► query_chunk_gt(state)
            ├─► normalized state
            ├─► KDTree.query(k=1)
            └─► return (gt_imu_window_normalized, nn_distance, is_valid)
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np


@dataclass
class IMURewardComputer:
    """Static lookup table. All numpy arrays. Pickleable."""
    # KDTree (sklearn.neighbors.KDTree)
    kdtree: object
    # Per-node IMU window, shape (n_nodes, chunk_size, imu_dim), normalized
    imu_windows: np.ndarray
    # Per-node demo id (for diagnostics; not used at inference)
    demo_ids: np.ndarray
    # Per-node frame_idx within its demo (for diagnostics)
    frame_indices: np.ndarray
    # Normalization stats
    state_mean: np.ndarray
    state_std: np.ndarray
    imu_mean: np.ndarray
    imu_std: np.ndarray
    # OOD threshold
    tau: float
    # Metadata
    chunk_size: int
    n_demos: int
    state_dim: int
    imu_dim: int
    # Stats summary (for monitoring)
    nn_dist_stats: dict  # {"v2_inter_p95": ..., "c1_mean": ..., "c1_p95": ..., ...}

    def query_chunk_gt(self, state: np.ndarray) -> tuple[np.ndarray, float, bool]:
        """At chunk boundary: state -> (gt_imu_window, nn_dist, is_valid).

        Args:
            state: (state_dim,) raw (un-normalized) joint angles.
        Returns:
            gt_imu_window: (chunk_size, imu_dim) NORMALIZED IMU window
                          (use .normalize_imu(real_imu) to compare)
            nn_dist: float, Euclidean distance in normalized state space
            is_valid: True if nn_dist <= tau
        """
        q = (state - self.state_mean) / (self.state_std + 1e-8)
        dist, idx = self.kdtree.query(q.reshape(1, -1), k=1)
        dist = float(dist[0][0])
        idx = int(idx[0][0])
        is_valid = (dist <= self.tau)
        return self.imu_windows[idx].copy(), dist, is_valid

    def normalize_imu(self, imu: np.ndarray) -> np.ndarray:
        """Normalize raw IMU using v2 stats. imu shape (..., imu_dim)."""
        return (imu - self.imu_mean) / (self.imu_std + 1e-8)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "IMURewardComputer":
        with open(path, "rb") as f:
            return pickle.load(f)


class RewardCalculator:
    """Stateful reward computer. Wraps IMURewardComputer + Phase2Config.

    Maintains EMA running stats of raw rewards for standardization.
    Use via:
        rc = RewardCalculator(computer, cfg)
        # at chunk boundary:
        gt_window, nn_dist, valid = rc.fetch_chunk_gt(state)
        # per step within chunk:
        R, valid_step = rc.step_reward(real_imu_t, gt_window[t], task_success_t,
                                       chunk_valid=valid)
    """

    def __init__(self, computer: IMURewardComputer, cfg):
        self.computer = computer
        self.cfg = cfg
        # EMA stats over raw R (before normalization & clip)
        self._r_mean = 0.0
        self._r_var = 1.0
        self._n_seen = 0

    def fetch_chunk_gt(self, state: np.ndarray):
        return self.computer.query_chunk_gt(state)

    def _update_ema(self, raw_r: float):
        a = self.cfg.reward_ema_alpha
        # Welford-like update with EMA
        delta = raw_r - self._r_mean
        self._r_mean = a * self._r_mean + (1 - a) * raw_r
        self._r_var = a * self._r_var + (1 - a) * (delta ** 2)
        self._n_seen += 1

    def step_reward(
        self,
        real_imu: np.ndarray,
        gt_imu: np.ndarray,
        task_success: bool,
        chunk_valid: bool,
    ) -> tuple[float, bool]:
        """Per-step reward. Returns (R_normalized_clipped, valid_flag).

        valid_flag = chunk_valid (whether this whole chunk passed OOD gate).
        Caller uses valid_flag to mask GAE/PPO loss.
        """
        # Normalize real IMU using v2 stats (same as gt_imu was normalized)
        r_norm = self.computer.normalize_imu(real_imu)
        l2 = -float(((r_norm - gt_imu) ** 2).sum())
        bonus = self.cfg.success_bonus if task_success else 0.0
        raw_R = l2 + bonus

        self._update_ema(raw_R)

        # Standardize then clip (only after some warmup samples)
        if self._n_seen > 50:
            R_std = (raw_R - self._r_mean) / (np.sqrt(self._r_var) + 1e-8)
            R = float(np.clip(R_std, self.cfg.reward_clip_min, self.cfg.reward_clip_max))
        else:
            # Pre-warmup: pass raw value (RL won't use it anyway during critic warmup)
            R = float(np.clip(raw_R, -10, 10))

        return R, chunk_valid

    def stats(self) -> dict:
        return {
            "r_mean": self._r_mean,
            "r_std": float(np.sqrt(self._r_var)),
            "n_seen": self._n_seen,
        }
