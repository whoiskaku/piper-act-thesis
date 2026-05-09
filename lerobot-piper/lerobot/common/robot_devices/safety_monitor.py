"""Phase 2 safety utilities — stateless functions + stateful trackers.

Used in:
  - phase2_rollout.py: clamp action, velocity limit, NaN check before send_action
  - phase2_actor_critic.py: NaN reward filter before GAE
  - phase2_train.py: consecutive failure pause logic + emergency stop integration

All pure functions are torch / numpy compatible (use whichever the caller passes).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────
# Stateless pure utilities
# ─────────────────────────────────────────────────────────────────

def clamp_action(action: torch.Tensor, min_vec: torch.Tensor,
                 max_vec: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """Clip action element-wise to [min_vec, max_vec].

    Returns:
        clamped: same shape as action, values in range
        exceeded: True if any element was clipped
    """
    if action.shape[-1] != min_vec.shape[-1]:
        raise ValueError(
            f"action dim {action.shape[-1]} != range dim {min_vec.shape[-1]}"
        )
    min_vec = min_vec.to(action.device).to(action.dtype)
    max_vec = max_vec.to(action.device).to(action.dtype)
    clamped = torch.clamp(action, min_vec, max_vec)
    exceeded = bool(((action != clamped).any()).item())
    return clamped, exceeded


def velocity_clip(a_t: torch.Tensor, a_prev: torch.Tensor,
                  max_delta: float) -> Tuple[torch.Tensor, bool]:
    """Clip step-to-step delta. a_t and a_prev are same-shape (..., A).

    Returns:
        a_clipped = a_prev + clip(a_t - a_prev, -max_delta, +max_delta)
        exceeded: True if any element's delta was clipped
    """
    delta = a_t - a_prev
    delta_clipped = torch.clamp(delta, -max_delta, +max_delta)
    a_clipped = a_prev + delta_clipped
    exceeded = bool(((delta != delta_clipped).any()).item())
    return a_clipped, exceeded


def is_finite_action(action: torch.Tensor) -> Tuple[bool, str]:
    """Check action has no NaN/Inf. Returns (is_safe, message_if_not)."""
    if not torch.all(torch.isfinite(action)):
        n_bad = (~torch.isfinite(action)).sum().item()
        return False, f"action contains {n_bad} non-finite values"
    return True, ""


def chunk_boundary_ramp(mu_new: torch.Tensor, mu_last_step: torch.Tensor,
                        n_ramp: int = 5) -> torch.Tensor:
    """Linearly interpolate the first n_ramp steps of mu_new toward mu_last_step.

    At step t (0-indexed):
        smoothed[t] = (1 - α_t) × mu_last_step + α_t × mu_new[t]
        α_t = (t + 1) / (n_ramp + 1)
    For t >= n_ramp: smoothed[t] = mu_new[t]

    Args:
        mu_new: (T, A) — new chunk's predicted means
        mu_last_step: (A,) — last action of previous chunk (post-clamp)
        n_ramp: number of leading steps to ramp

    Returns:
        smoothed: (T, A)
    """
    if n_ramp <= 0 or mu_new.shape[0] < n_ramp:
        return mu_new.clone()
    smoothed = mu_new.clone()
    for t in range(n_ramp):
        alpha = (t + 1) / (n_ramp + 1)
        smoothed[t] = (1 - alpha) * mu_last_step + alpha * mu_new[t]
    return smoothed


def imu_health_check(imu: torch.Tensor,
                     max_accel_norm: float = 50.0) -> Tuple[bool, str]:
    """IMU sanity check — accel norm in range + no NaN.

    Args:
        imu: shape (..., 6) where last 3 are gyro, first 3 are accel
        max_accel_norm: m/s², above this is sensor error / collision
    """
    if not torch.all(torch.isfinite(imu)):
        return False, "IMU has NaN or Inf"
    accel_norm = imu[..., :3].norm(dim=-1).max().item()
    if accel_norm > max_accel_norm:
        return False, f"accel norm {accel_norm:.1f} > {max_accel_norm}"
    return True, ""


def compute_action_clamp_range(
    action_stats: dict, margin: float = 1.2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """From dataset action stats (with 'min', 'max'), build clamp range.

    Returns (min_vec, max_vec), each shape (A,).
    """
    a_min = torch.as_tensor(action_stats["min"], dtype=torch.float32)
    a_max = torch.as_tensor(action_stats["max"], dtype=torch.float32)
    span = a_max - a_min
    margin_extra = (margin - 1.0) * span / 2.0
    return a_min - margin_extra, a_max + margin_extra


# ─────────────────────────────────────────────────────────────────
# Stateful tracker classes
# ─────────────────────────────────────────────────────────────────

class ConsecutiveFailureTracker:
    """Track consecutive episode failures. Pause training if streak ≥ N."""

    def __init__(self, max_streak: int = 3):
        self.max_streak = max_streak
        self.streak = 0
        self.total_failures = 0
        self.total_episodes = 0

    def record(self, success: bool) -> None:
        self.total_episodes += 1
        if success:
            self.streak = 0
        else:
            self.streak += 1
            self.total_failures += 1

    def should_pause(self) -> bool:
        return self.streak >= self.max_streak

    def reset(self) -> None:
        self.streak = 0

    def stats(self) -> dict:
        return {
            "consecutive_failures": self.streak,
            "total_failures": self.total_failures,
            "total_episodes": self.total_episodes,
            "failure_rate": self.total_failures / max(self.total_episodes, 1),
        }


class SafetyEventLog:
    """Aggregate counter of safety-relevant events for monitoring."""

    def __init__(self):
        self.counts = {
            "action_clamp_exceeded": 0,
            "velocity_clamp_exceeded": 0,
            "nan_action_detected": 0,
            "imu_health_failed": 0,
            "emergency_stop": 0,
        }

    def increment(self, key: str, n: int = 1) -> None:
        if key not in self.counts:
            self.counts[key] = 0
        self.counts[key] += n

    def snapshot(self) -> dict:
        return dict(self.counts)

    def reset(self) -> None:
        for k in self.counts:
            self.counts[k] = 0
