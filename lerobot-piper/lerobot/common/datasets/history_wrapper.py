"""Episode-level history wrapper for ACT autoregressive Phase 1 training.

Wraps a LeRobotDataset and, on every __getitem__, pairs the current frame with
a random *different* episode whose full action and IMU sequences are exposed as
new keys: observation.history_action, observation.history_imu, history_pad_mask.

The wrapper is data-preprocessing only and does not modify the underlying dataset.
At deployment time the model's select_action receives history from the runtime
buffer instead — this wrapper is for offline training.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def rolling_mean(x: torch.Tensor, window: int) -> torch.Tensor:
    """Rolling-mean smoothing along time axis 0 with edge replicate padding.

    Args:
        x: (T, D) tensor.
        window: kernel size; window <= 1 returns a clone of x.

    Returns:
        (T, D) smoothed tensor of the same shape.
    """
    if window <= 1 or x.shape[0] < window:
        return x.clone()
    # F.avg_pool1d needs (B, C, L)
    x_in = x.T.unsqueeze(0)
    left = (window - 1) // 2
    right = window - 1 - left
    x_padded = F.pad(x_in, (left, right), mode="replicate")
    smoothed = F.avg_pool1d(x_padded, kernel_size=window, stride=1)
    return smoothed.squeeze(0).T.contiguous()


class EpisodeHistoryDataset(Dataset):
    """LeRobotDataset wrapper that injects previous-episode history features.

    Args:
        base_dataset: LeRobotDataset instance.
        history_max_length: zero-pad history sequences to this length (frames).
        use_action: include observation.history_action.
        use_imu: include observation.history_imu (requires observation.imu_real
            in the base dataset features).
        imu_smooth_window: rolling-mean window over IMU history before padding;
            applied only when use_imu=True. 1 disables.
        action_key: column name for actions in the base dataset.
        imu_key: column name for IMU samples in the base dataset.
        seed: optional seed for the pairing RNG. When None, uses the global
            python random which inherits DataLoader worker seeds.
    """

    def __init__(
        self,
        base_dataset,
        history_max_length: int = 400,
        use_action: bool = True,
        use_imu: bool = True,
        imu_smooth_window: int = 3,
        normalize_history: bool = True,
        action_key: str = "action",
        imu_key: str = "observation.imu_real",
        seed: int | None = None,
    ):
        self.base = base_dataset
        self.history_max_length = history_max_length
        self.use_action = use_action
        self.use_imu = use_imu
        self.imu_smooth_window = imu_smooth_window
        self.normalize_history = normalize_history
        self.action_key = action_key
        self.imu_key = imu_key
        self._rng = random.Random(seed) if seed is not None else random

        # Sanity: required columns must exist on the underlying hf_dataset.
        cols = set(self.base.hf_dataset.column_names)
        if self.use_action and self.action_key not in cols:
            raise KeyError(f"history wrapper: '{self.action_key}' not in dataset columns.")
        if self.use_imu and self.imu_key not in cols:
            raise KeyError(
                f"history wrapper: '{self.imu_key}' not in dataset columns. "
                "Re-collect dataset with enable_imu=True or disable history_use_imu."
            )

        # Episode list (these are the *selected* episodes the base dataset exposes).
        self._episode_indices = sorted(set(int(e) for e in self.base.episodes)) \
            if self.base.episodes is not None \
            else list(range(self.base.meta.total_episodes))

        # Cache full-episode tensors lazily: {ep_idx: {"action": ..., "imu": ...}}
        self._cache: dict[int, dict[str, torch.Tensor]] = {}

        # Resolve dims for zero-pad fallback
        self._action_dim = (
            int(self.base.meta.features[self.action_key]["shape"][0])
            if self.use_action else 0
        )
        self._imu_dim = (
            int(self.base.meta.features[self.imu_key]["shape"][0])
            if self.use_imu else 0
        )

        # Cache mean/std tensors for history normalization, matching the dataset's
        # MEAN_STD policy. Pad zeros stay zero pre-normalization; mask hides them anyway.
        self._action_mean = self._action_std = None
        self._imu_mean = self._imu_std = None
        if self.normalize_history:
            if self.use_action:
                stats = self.base.meta.stats[self.action_key]
                self._action_mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
                self._action_std = torch.as_tensor(stats["std"], dtype=torch.float32)
            if self.use_imu:
                stats = self.base.meta.stats[self.imu_key]
                self._imu_mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
                self._imu_std = torch.as_tensor(stats["std"], dtype=torch.float32)

    def __len__(self):
        return len(self.base)

    def _load_full_episode(self, ep_idx: int) -> dict[str, torch.Tensor]:
        if ep_idx in self._cache:
            return self._cache[ep_idx]
        ep_from = int(self.base.episode_data_index["from"][ep_idx])
        ep_to = int(self.base.episode_data_index["to"][ep_idx])
        out: dict[str, torch.Tensor] = {}
        # Slice columns from hf_dataset. Each column already returns torch tensors.
        if self.use_action:
            action_slice = self.base.hf_dataset[ep_from:ep_to][self.action_key]
            # The hf datasets backend may return list of tensors or a single tensor.
            action_t = (
                action_slice if torch.is_tensor(action_slice) else torch.stack(list(action_slice))
            ).float()
            if self.normalize_history:
                action_t = (action_t - self._action_mean) / (self._action_std + 1e-8)
            out["action"] = action_t
        if self.use_imu:
            imu_slice = self.base.hf_dataset[ep_from:ep_to][self.imu_key]
            imu_t = imu_slice if torch.is_tensor(imu_slice) else torch.stack(list(imu_slice))
            imu_t = rolling_mean(imu_t.float(), self.imu_smooth_window)
            if self.normalize_history:
                imu_t = (imu_t - self._imu_mean) / (self._imu_std + 1e-8)
            out["imu"] = imu_t
        self._cache[ep_idx] = out
        return out

    def _pick_prev_episode(self, current_ep_idx: int) -> int | None:
        """Random pairing: any episode different from current. None if impossible."""
        candidates = [e for e in self._episode_indices if e != current_ep_idx]
        if not candidates:
            return None
        return self._rng.choice(candidates)

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]
        current_ep_idx = int(item["episode_index"].item())
        prev_ep_idx = self._pick_prev_episode(current_ep_idx)

        L = self.history_max_length
        if prev_ep_idx is None:
            n_real = 0
            if self.use_action:
                item["observation.history_action"] = torch.zeros(L, self._action_dim)
            if self.use_imu:
                item["observation.history_imu"] = torch.zeros(L, self._imu_dim)
            item["history_pad_mask"] = torch.ones(L, dtype=torch.bool)
            return item

        prev = self._load_full_episode(prev_ep_idx)
        # Right-aligned layout: newest action of prev_ep always sits at index L-1.
        # If prev_ep is shorter than L, pad zeros at the front (positions [0, L-n_real)).
        # If prev_ep is longer, the oldest frames are dropped (we keep last L).
        n_real = 0
        if self.use_action:
            seq = prev["action"]
            n_real = min(seq.shape[0], L)
            hist_a = torch.zeros(L, self._action_dim)
            hist_a[L - n_real:] = seq[-n_real:]
            item["observation.history_action"] = hist_a
        if self.use_imu:
            seq = prev["imu"]
            n_real = min(seq.shape[0], L)
            hist_i = torch.zeros(L, self._imu_dim)
            hist_i[L - n_real:] = seq[-n_real:]
            item["observation.history_imu"] = hist_i

        mask = torch.zeros(L, dtype=torch.bool)
        if n_real < L:
            mask[: L - n_real] = True
        item["history_pad_mask"] = mask
        return item

    # Forward common attributes so trainers can introspect base metadata.
    def __getattr__(self, name):
        if name in ("base", "_cache", "_rng", "_episode_indices",
                    "history_max_length", "use_action", "use_imu",
                    "imu_smooth_window", "action_key", "imu_key",
                    "_action_dim", "_imu_dim"):
            raise AttributeError(name)
        return getattr(self.base, name)
