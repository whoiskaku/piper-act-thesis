"""Phase 2 actor-critic utilities for PPO fine-tuning of ACT.

Layered structure (most modules use the lower-level pure functions):

  Pure functions (no state, easy to test):
    - factorized_gaussian_log_prob
    - kl_gaussian_fixed_sigma
    - compute_gae
    - ppo_clip_loss
    - value_loss

  Helper class for the rollout↔update lifecycle:
    - StochasticActor — wraps ACTPolicy, applies σ-noise + chunk-boundary
                         smoothing, exposes (rollout, re_evaluate) entry points
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from lerobot.common.robot_devices.safety_monitor import chunk_boundary_ramp


# ════════════════════════════════════════════════════════════════
# Pure functions
# ════════════════════════════════════════════════════════════════

def factorized_gaussian_log_prob(action: torch.Tensor, mu: torch.Tensor,
                                  sigma_per_dim: torch.Tensor) -> torch.Tensor:
    """log N(action | mu, diag(σ²)) factorized over time × action_dim.

    Args:
        action: (B, T, A)
        mu:     (B, T, A) — Gaussian mean (μ_smoothed for stochastic actor)
        sigma_per_dim: (A,) — per-dim std (constant per dim, time-invariant)

    Returns:
        log_prob: (B,) summed over (T, A)
    """
    sigma = sigma_per_dim.view(1, 1, -1).to(action.device).to(action.dtype)
    var = sigma ** 2
    log_var = (var + 1e-12).log()
    # per-element: -0.5 [(a-μ)²/σ² + log(2π σ²)]
    diff_sq = (action - mu) ** 2
    per_elem = -0.5 * (diff_sq / var + log_var + math.log(2 * math.pi))
    return per_elem.sum(dim=(1, 2))


def kl_gaussian_fixed_sigma(mu_new: torch.Tensor, mu_old: torch.Tensor,
                             sigma_per_dim: torch.Tensor) -> torch.Tensor:
    """KL(N(μ_new, σ²) ‖ N(μ_old, σ²)) for fixed-σ diagonal Gaussian.

    For same σ in both, KL = 0.5 × Σ ((μ_new - μ_old) / σ)²

    Args:
        mu_new, mu_old: (B, T, A)
        sigma_per_dim:  (A,)

    Returns:
        kl_per_batch: (B,) summed over (T, A)
    """
    sigma = sigma_per_dim.view(1, 1, -1).to(mu_new.device).to(mu_new.dtype)
    diff_sq = (mu_new - mu_old) ** 2
    kl_per_elem = 0.5 * diff_sq / (sigma ** 2 + 1e-12)
    return kl_per_elem.sum(dim=(1, 2))


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    valid_masks: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation with valid-mask handling (D6 spec).

    Invalid chunks have their reward replaced with bootstrap value
    (V(s_t) - γ V(s_{t+1})), making delta_t = 0 — they contribute zero
    learning signal but don't pollute neighbors.

    Args:
        rewards:     (T,) per-chunk reward
        values:      (T+1,) V(s_0), V(s_1), ..., V(s_T) including final bootstrap
                     (typically V(s_T) = 0 for terminated episode, or model output)
        valid_masks: (T,) bool — chunk's NN dist <= τ (D2 OOD gate)
        gamma:       discount factor across chunks
        gae_lambda:  GAE λ

    Returns:
        advantages:  (T,) — explicit zero on invalid steps
        returns:     (T,) — advantages + values[:-1]
    """
    T = rewards.shape[0]
    assert values.shape[0] == T + 1, f"values must have length T+1, got {values.shape[0]}"
    assert valid_masks.shape[0] == T

    valid_f = valid_masks.float()
    # For invalid: reward = V(s_t) - γV(s_{t+1}) → delta = 0
    bootstrap = values[:-1] - gamma * values[1:]
    effective_rewards = torch.where(valid_masks, rewards, bootstrap)

    advantages = torch.zeros(T, dtype=rewards.dtype, device=rewards.device)
    gae = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(T)):
        delta = effective_rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae

    # Explicit zero on invalid (double safety)
    advantages = advantages * valid_f
    returns = advantages + values[:-1]
    # Returns also masked for value loss
    returns = torch.where(valid_masks, returns, values[:-1])  # invalid: V predicts itself → loss=0
    return advantages, returns


def ppo_clip_loss(
    log_prob_new: torch.Tensor,
    log_prob_old: torch.Tensor,
    advantages: torch.Tensor,
    valid_masks: torch.Tensor,
    clip_eps: float,
) -> tuple[torch.Tensor, dict]:
    """PPO clip surrogate with valid-mask.

    Args:
        log_prob_new, log_prob_old: (T,) — log π_θ(A_t|s_t)
        advantages: (T,) — already valid-masked, optionally normalized
        valid_masks: (T,) bool
        clip_eps: e.g., 0.2

    Returns:
        loss: scalar tensor — mean over valid steps (excludes invalid from divisor)
        metrics: dict with {"ppo_loss", "ratio_mean", "ratio_clip_frac", "n_valid"}
    """
    ratio = torch.exp(log_prob_new - log_prob_old)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    raw = -torch.min(surr1, surr2)

    valid_f = valid_masks.float()
    n_valid = valid_f.sum().clamp(min=1.0)
    loss = (raw * valid_f).sum() / n_valid

    clip_frac = ((ratio > 1 + clip_eps) | (ratio < 1 - clip_eps)).float()
    clip_frac = (clip_frac * valid_f).sum() / n_valid

    return loss, {
        "ppo_loss": float(loss.item()),
        "ratio_mean": float(ratio.mean().item()),
        "ratio_clip_frac": float(clip_frac.item()),
        "n_valid": int(n_valid.item()),
    }


def value_loss(
    V_pred: torch.Tensor,
    returns: torch.Tensor,
    valid_masks: torch.Tensor,
) -> torch.Tensor:
    """MSE value loss with valid-mask + valid-step mean."""
    valid_f = valid_masks.float()
    n_valid = valid_f.sum().clamp(min=1.0)
    raw = (V_pred - returns) ** 2
    return (raw * valid_f).sum() / n_valid


def standardize_advantages(advantages: torch.Tensor,
                            valid_masks: torch.Tensor) -> torch.Tensor:
    """Standardize advantages by valid-step mean/std (PPO best practice)."""
    valid_f = valid_masks.float()
    n_valid = valid_f.sum().clamp(min=1.0)
    mean = (advantages * valid_f).sum() / n_valid
    var = (((advantages - mean) ** 2) * valid_f).sum() / n_valid
    std = (var + 1e-8).sqrt()
    return ((advantages - mean) / std) * valid_f


# ════════════════════════════════════════════════════════════════
# StochasticActor — rollout + re-evaluate lifecycle
# ════════════════════════════════════════════════════════════════

class StochasticActor:
    """Wraps ACTPolicy for Phase 2 stochastic rollout + PPO update.

    Lifecycle:
        # Rollout
        out = actor.rollout_chunk(batch, prev_last_action, update_step)
        action_normalized = out["action"]                      # (1, T, A)
        log_prob_old = out["log_prob"]                          # (1,)
        mu_old = out["mu_smoothed"]                             # (1, T, A)
        V_old = out["value"]                                    # (1,)
        sigma = out["sigma"]                                    # (A,)
        # → unnormalize action_normalized to send to robot

        # Update (later, with stored data)
        out_new = actor.re_evaluate(batch, action_normalized,
                                     prev_last_action, update_step)
        log_prob_new = out_new["log_prob"]
        V_new = out_new["value"]
        mu_new = out_new["mu_smoothed"]
    """

    def __init__(self, act_policy, cfg, device: str | torch.device = "cuda"):
        self.policy = act_policy
        self.cfg = cfg
        self.device = torch.device(device)
        self._sigma_base = torch.as_tensor(cfg.sigma_per_dim, dtype=torch.float32,
                                            device=self.device)

    def get_sigma(self, update_step: int) -> torch.Tensor:
        """Effective per-dim σ at given update step (after schedule)."""
        return self._sigma_base * float(self.cfg.get_sigma_multiplier(update_step))

    def _normalize_batch(self, batch: dict) -> dict:
        """Apply ACTPolicy's normalize_inputs + image stacking + normalize_targets."""
        batch = self.policy.normalize_inputs(batch)
        if self.policy.config.image_features:
            batch = dict(batch)
            batch["observation.images"] = torch.stack(
                [batch[k] for k in self.policy.config.image_features], dim=-4
            )
        if "action" in batch:
            batch = self.policy.normalize_targets(batch)
        return batch

    def _forward_mu_value(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """ACT forward, returning (mu, V) in normalized space."""
        batch = self._normalize_batch(batch)
        mu, _, _, V = self.policy.model(batch, return_value=True)
        return mu, V

    def _smooth_chunk_boundary(self, mu: torch.Tensor,
                                prev_last_action: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply chunk-boundary μ ramp if previous chunk's last action provided."""
        if prev_last_action is None:
            return mu
        n_ramp = self.cfg.chunk_boundary_ramp_steps
        smoothed = mu.clone()
        for b in range(mu.shape[0]):
            smoothed[b] = chunk_boundary_ramp(mu[b], prev_last_action.to(mu.device),
                                                 n_ramp=n_ramp)
        return smoothed

    @torch.no_grad()
    def rollout_chunk(
        self,
        batch: dict,
        prev_last_action: Optional[torch.Tensor],
        update_step: int,
    ) -> dict:
        """Rollout-time decision: sample stochastic chunk action + log_prob.

        Args:
            batch: dict with model inputs (observation.state, images, history, etc.).
                   Should be on self.device with batch dim 1.
            prev_last_action: (A,) previous chunk's last NORMALIZED action, or None
                              for first chunk of episode.
            update_step: current PPO update step (for σ schedule).

        Returns:
            dict with keys: action (normalized), mu_smoothed, log_prob, value, sigma, eps
        """
        mu, V = self._forward_mu_value(batch)             # (1, T, A), (1,)
        mu_smoothed = self._smooth_chunk_boundary(mu, prev_last_action)
        sigma = self.get_sigma(update_step)               # (A,)

        eps = torch.randn_like(mu_smoothed)
        action = mu_smoothed + sigma.view(1, 1, -1) * eps
        log_prob = factorized_gaussian_log_prob(action, mu_smoothed, sigma)

        return {
            "action": action,           # (1, T, A) normalized
            "mu_smoothed": mu_smoothed,  # for later re_evaluate match
            "log_prob": log_prob,        # (1,)
            "value": V,                  # (1,)
            "sigma": sigma,              # (A,)
            "eps": eps,                  # (1, T, A)
        }

    def re_evaluate(
        self,
        batch: dict,
        action_normalized: torch.Tensor,
        prev_last_action: Optional[torch.Tensor],
        update_step: int,
    ) -> dict:
        """PPO update-time: re-evaluate stored action's log_prob + V under current policy.

        Used K_epoch times per stored rollout chunk.
        """
        mu, V = self._forward_mu_value(batch)
        mu_smoothed = self._smooth_chunk_boundary(mu, prev_last_action)
        sigma = self.get_sigma(update_step)
        log_prob = factorized_gaussian_log_prob(action_normalized, mu_smoothed, sigma)

        return {
            "log_prob": log_prob,
            "value": V,
            "mu_smoothed": mu_smoothed,
            "sigma": sigma,
        }


# ════════════════════════════════════════════════════════════════
# Critic warmup helper
# ════════════════════════════════════════════════════════════════

def critic_warmup_loss(
    actor: StochasticActor,
    batch: dict,
    real_returns: torch.Tensor,
) -> torch.Tensor:
    """V(s) MSE loss for critic warmup (Stage B).

    During warmup, only value_head params have requires_grad=True.
    """
    _, V = actor._forward_mu_value(batch)        # (B,)
    return ((V - real_returns) ** 2).mean()
