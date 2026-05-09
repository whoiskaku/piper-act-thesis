"""Rolling history buffer for autoregressive ACT deployment.

Records each completed episode's full (action, IMU) sequence and exposes the
last completed one as `observation.history_action` / `observation.history_imu`
plus a `history_pad_mask` aligned to the right (newest at index L-1).

First episode: returns all-zero pad with mask all True (matches training-time
zero-pad scenario).

Subsequent episodes: returns last completed episode's data, right-aligned and
zero-padded if shorter than `history_max_length`.

IMU rolling-mean smoothing matches the training-time wrapper (window=3).

Statistics for normalization come from the loaded policy's training stats so
the layout exactly matches what the model saw during BC training.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _rolling_mean(x: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1 or x.shape[0] < window:
        return x.clone()
    x_in = x.T.unsqueeze(0)
    left = (window - 1) // 2
    right = window - 1 - left
    x_padded = F.pad(x_in, (left, right), mode="replicate")
    smoothed = F.avg_pool1d(x_padded, kernel_size=window, stride=1)
    return smoothed.squeeze(0).T.contiguous()


class HistoryBuffer:
    def __init__(
        self,
        history_max_length: int,
        action_dim: int,
        imu_dim: int,
        use_action: bool,
        use_imu: bool,
        action_mean: Optional[torch.Tensor],
        action_std: Optional[torch.Tensor],
        imu_mean: Optional[torch.Tensor],
        imu_std: Optional[torch.Tensor],
        imu_smooth_window: int = 3,
        device: str | torch.device = "cpu",
    ):
        self.L = history_max_length
        self.action_dim = action_dim
        self.imu_dim = imu_dim
        self.use_action = use_action
        self.use_imu = use_imu
        self.imu_smooth_window = imu_smooth_window
        self.device = torch.device(device)

        self.action_mean = action_mean.to(self.device) if action_mean is not None else None
        self.action_std = action_std.to(self.device) if action_std is not None else None
        self.imu_mean = imu_mean.to(self.device) if imu_mean is not None else None
        self.imu_std = imu_std.to(self.device) if imu_std is not None else None

        # Current episode (being recorded right now)
        self._cur_action: list[torch.Tensor] = []
        self._cur_imu: list[torch.Tensor] = []
        # Last completed episode (returned as history input)
        self._last_action: Optional[torch.Tensor] = None
        self._last_imu: Optional[torch.Tensor] = None

    def append_step(self, action: torch.Tensor, imu: Optional[torch.Tensor]) -> None:
        """Call once per control-loop tick after action is sent."""
        if self.use_action and action is not None:
            self._cur_action.append(action.detach().to(self.device).flatten())
        if self.use_imu and imu is not None:
            self._cur_imu.append(imu.detach().to(self.device).flatten())

    def end_episode(self) -> None:
        """Move current → last so next episode's get_history() returns this episode's data."""
        if len(self._cur_action) > 0:
            self._last_action = torch.stack(self._cur_action)
        if len(self._cur_imu) > 0:
            self._last_imu = torch.stack(self._cur_imu)
        self._cur_action = []
        self._cur_imu = []

    def get_history(self) -> dict[str, torch.Tensor]:
        """Return dict with observation.history_action, observation.history_imu, history_pad_mask.
        Each tensor is shape (L, dim) or (L,) — caller must add batch dim."""
        L = self.L
        out: dict[str, torch.Tensor] = {}

        # First-episode case: no prior episode → full pad
        if self._last_action is None and self._last_imu is None:
            if self.use_action:
                out["observation.history_action"] = torch.zeros(L, self.action_dim, device=self.device)
            if self.use_imu:
                out["observation.history_imu"] = torch.zeros(L, self.imu_dim, device=self.device)
            out["history_pad_mask"] = torch.ones(L, dtype=torch.bool, device=self.device)
            return out

        # Normalize ONLY the real (non-pad) portion, keep pad as raw zeros to
        # match the training-time wrapper layout (which fills then leaves pad=0).
        n_real = 0
        if self.use_action and self._last_action is not None:
            seq = self._last_action.float()
            n_real = min(seq.shape[0], L)
            real_part = seq[-n_real:]
            if self.action_mean is not None:
                real_part = (real_part - self.action_mean) / (self.action_std + 1e-8)
            ha = torch.zeros(L, self.action_dim, device=self.device)
            ha[L - n_real:] = real_part
            out["observation.history_action"] = ha
        if self.use_imu and self._last_imu is not None:
            seq = self._last_imu.float()
            seq = _rolling_mean(seq, self.imu_smooth_window)
            n_real = min(seq.shape[0], L)
            real_part = seq[-n_real:]
            if self.imu_mean is not None:
                real_part = (real_part - self.imu_mean) / (self.imu_std + 1e-8)
            hi = torch.zeros(L, self.imu_dim, device=self.device)
            hi[L - n_real:] = real_part
            out["observation.history_imu"] = hi
        elif self.use_imu:
            # use_imu requested but no IMU recorded → pad zeros
            out["observation.history_imu"] = torch.zeros(L, self.imu_dim, device=self.device)

        mask = torch.zeros(L, dtype=torch.bool, device=self.device)
        if n_real < L:
            mask[: L - n_real] = True
        out["history_pad_mask"] = mask
        return out

    @classmethod
    def from_policy_and_dataset_meta(cls, policy, dataset_meta, device="cpu"):
        """Construct from a trained ACT policy + LeRobotDatasetMetadata.

        Pulls the dimensions and normalization stats so deployment matches
        the exact training-time layout.
        """
        cfg = policy.config
        action_dim = cfg.action_feature.shape[0]
        imu_dim = cfg.imu_feature.shape[0] if cfg.imu_feature is not None else 0

        # Action stats come from the model's normalize_targets buffer (loaded from ckpt).
        a_buf = policy.normalize_targets.buffer_action
        action_mean = a_buf["mean"].squeeze() if "mean" in a_buf else None
        action_std = a_buf["std"].squeeze() if "std" in a_buf else None

        # IMU stats: prefer the policy's normalize_inputs buffer (was input feature at train time);
        # fall back to dataset_meta.stats.
        imu_mean = imu_std = None
        imu_buf_name = "buffer_observation_imu_real"
        if hasattr(policy.normalize_inputs, imu_buf_name):
            buf = getattr(policy.normalize_inputs, imu_buf_name)
            imu_mean = buf["mean"].squeeze() if "mean" in buf else None
            imu_std = buf["std"].squeeze() if "std" in buf else None
        elif "observation.imu_real" in getattr(dataset_meta, "stats", {}):
            stats = dataset_meta.stats["observation.imu_real"]
            imu_mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
            imu_std = torch.as_tensor(stats["std"], dtype=torch.float32)

        return cls(
            history_max_length=cfg.history_max_length,
            action_dim=action_dim,
            imu_dim=imu_dim,
            use_action=cfg.history_use_action,
            use_imu=cfg.history_use_imu,
            action_mean=action_mean,
            action_std=action_std,
            imu_mean=imu_mean,
            imu_std=imu_std,
            device=device,
        )
