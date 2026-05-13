"""IMURewardComputer: static state→IMU GT lookup table for Phase 2 RL reward.

Built once offline (scripts/d2_setup_imu_reward.py) and loaded at training /
deployment time. Holds:
  - KDTree over normalized v2 demo states
  - Cached per-node IMU window (next chunk_size frames, normalized)
  - Per-demo tagging so we can exclude same-demo neighbors for intra-NN stats
  - Normalization stats (state, IMU)
  - OOD distance threshold τ

At inference time (Round C — cross-demo k=5 NN averaging):
  state ──► query_chunk_gt(state, k_demos=5)
            ├─► normalized state
            ├─► KDTree.query(k=50) → filter to 1 NN per demo, take top k_demos
            ├─► average their imu_windows → gt_avg
            └─► return (gt_avg_normalized, mean_nn_distance, is_valid)
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

    def query_chunk_gt(
        self,
        state: np.ndarray,
        k_demos: int = 5,
        k_query: int = 50,
    ) -> tuple[np.ndarray, float, bool]:
        """At chunk boundary: state -> (gt_imu_window, mean_nn_dist, is_valid).

        Round C: cross-demo k=5 NN averaging. Decouples gt_imu noise from
        any single demo's high-frequency content by averaging k_demos
        nearest neighbors taken from DISTINCT demos. Gt becomes smoother
        than any individual demo by factor √k_demos (assuming independent
        demo-specific noise), which indirectly drives policy smoothness.

        Args:
            state: (state_dim,) raw joint angles.
            k_demos: number of distinct demos to average. Round C = 5.
            k_query: how many KDTree neighbors to fetch before filtering to
                     k_demos unique demos. Must be >= k_demos. Higher values
                     more robust when same-demo frames cluster nearby.
        Returns:
            gt_imu_window: (chunk_size, imu_dim) averaged NORMALIZED window
                          (use .normalize_imu(real_imu) to compare)
            mean_nn_dist: float, mean Euclidean distance over selected NNs
            is_valid: True if mean_nn_dist <= tau

        Edge cases:
            - If fewer than k_demos distinct demos found in k_query, average
              over what we have (degrades gracefully toward k=1 in worst case).
            - If all selected nodes are OOD individually we still average and
              return mean dist; gate is on the mean.
        """
        q = (state - self.state_mean) / (self.state_std + 1e-8)
        # Over-fetch then filter by demo.
        k = min(int(k_query), int(self.kdtree.data.shape[0]))
        dists, idxs = self.kdtree.query(q.reshape(1, -1), k=k)
        dists = dists[0]
        idxs = idxs[0]

        # Filter: keep the nearest hit from each distinct demo, up to k_demos.
        seen_demos: set[int] = set()
        selected_idxs: list[int] = []
        selected_dists: list[float] = []
        for d, i in zip(dists, idxs):
            demo_id = int(self.demo_ids[int(i)])
            if demo_id in seen_demos:
                continue
            seen_demos.add(demo_id)
            selected_idxs.append(int(i))
            selected_dists.append(float(d))
            if len(selected_idxs) >= k_demos:
                break

        # Degrade gracefully if k_query was too small.
        if not selected_idxs:
            # Should never happen unless KDTree is empty.
            idx0 = int(idxs[0])
            return (self.imu_windows[idx0].copy(),
                    float(dists[0]),
                    bool(float(dists[0]) <= self.tau))

        selected_windows = self.imu_windows[selected_idxs]  # (k, chunk, imu_dim)
        gt_avg = selected_windows.mean(axis=0)              # (chunk, imu_dim)
        mean_dist = float(np.mean(selected_dists))
        is_valid = bool(mean_dist <= self.tau)
        return gt_avg, mean_dist, is_valid

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
        # Normalize real IMU using v3 stats (same normalization as gt_imu).
        # Round C: change sum→mean over IMU dims so reward magnitude is
        # dim-independent (cleaner scaling; same gradient direction up to a
        # constant factor of 1/imu_dim, absorbed by EMA standardization).
        r_norm = self.computer.normalize_imu(real_imu)
        l2 = -float(((r_norm - gt_imu) ** 2).mean())
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
