"""PyTorch Dataset wrapping the Stage 2/2.5/2.6 H5 for RefineNet training.

H5 file has groups "train" and "val". Each group provides arrays:
    A_clean          (N, 100, 7)
    A_noisy          (N, 100, 7)
    ACT_output       (N, 100, 7)
    history_action   (N, 600, 7)
    history_imu      (N, 600, 6)
    history_pad_mask (N, 600)  bool
    start_state_7    (N, 7)
    src_ep, src_start, perturb_id, perturb_scale, qpos_*, IMU_*

Loaded fully into memory (~20 MB total), per-worker safe.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class RefinePhase1Dataset(Dataset):
    KEYS_NEEDED = [
        "A_clean", "ACT_output",
        "history_action", "history_imu", "history_pad_mask",
        "start_state_7",
    ]

    def __init__(
        self,
        h5_path: str | Path,
        split: str,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
    ):
        """
        Args:
            h5_path: path to phase1_paired_data.h5
            split:   "train" or "val"
            state_mean / state_std: per-channel z-score stats (computed on train).
                                    If None, returns raw start_state_7.
        """
        super().__init__()
        self.h5_path = str(h5_path)
        self.split = split
        with h5py.File(self.h5_path, "r") as f:
            grp = f[split]
            self.A_clean = grp["A_clean"][:].astype(np.float32)
            self.ACT_output = grp["ACT_output"][:].astype(np.float32)
            self.A_noisy = grp["A_noisy"][:].astype(np.float32)  # #62 输入信号对照用
            self.act_hidden_enc = (grp["act_hidden_enc"][:].astype(np.float32)
                                   if "act_hidden_enc" in grp else None)  # #63 ACT encoder 摘要
            self.act_img_tokens = (grp["act_img_tokens"][:].astype(np.float32)
                                   if "act_img_tokens" in grp else None)  # #63(a) ACT image tokens
            self.history_action = grp["history_action"][:].astype(np.float32)
            self.history_imu = grp["history_imu"][:].astype(np.float32)
            self.history_pad_mask = grp["history_pad_mask"][:].astype(bool)
            self.start_state_7 = grp["start_state_7"][:].astype(np.float32)
        self.n = self.A_clean.shape[0]
        self.state_mean = (
            np.asarray(state_mean, dtype=np.float32) if state_mean is not None else None
        )
        self.state_std = (
            np.asarray(state_std, dtype=np.float32) if state_std is not None else None
        )

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        state = self.start_state_7[i]
        if self.state_mean is not None and self.state_std is not None:
            state = (state - self.state_mean) / (self.state_std + 1e-8)
        out = {
            "ACT_output":       torch.from_numpy(self.ACT_output[i]),
            "A_noisy":          torch.from_numpy(self.A_noisy[i]),
            "history_action":   torch.from_numpy(self.history_action[i]),
            "history_imu":      torch.from_numpy(self.history_imu[i]),
            "history_pad_mask": torch.from_numpy(self.history_pad_mask[i]),
            "current_state":    torch.from_numpy(state.astype(np.float32)),
            "A_clean":          torch.from_numpy(self.A_clean[i]),
        }
        if self.act_hidden_enc is not None:
            out["act_hidden"] = torch.from_numpy(self.act_hidden_enc[i])
        if self.act_img_tokens is not None:
            out["act_img"] = torch.from_numpy(self.act_img_tokens[i])
        return out

    def compute_state_stats(
        self, min_std: float = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-channel mean and std of start_state_7 (used for z-score).

        Joints with std below min_std are treated as constants and assigned
        std=1.0 to avoid division-by-zero blow-ups. (E.g., v3 j4/j5/j6 are
        essentially stationary in the pick-place task.)
        """
        mean = self.start_state_7.mean(axis=0)
        std = self.start_state_7.std(axis=0)
        std = np.where(std < min_std, 1.0, std)
        return mean, std
