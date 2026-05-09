"""Phase 2 real-robot rollout — single episode.

One episode = T chunks × chunk_size steps (typically 6 × 100 = 600 frames @ 30 fps).
At each chunk boundary (every 100 steps): ACT forward → sample stochastic action chunk
                                          → query GT IMU window via state-NN.
Per step: send action with safety guards, record real IMU, compute reward.

Returns EpisodeRolloutData containing:
  - per-chunk: state batch, action_normalized, mu_smoothed, log_prob_old, V_old,
               nn_distance, valid mask, aggregated chunk_reward
  - episode-level: task_success, abort flag, n_steps, safety event counts

Notes:
  - This bypasses ACTPolicy.select_action's action queue / temporal ensembling
    because Phase 2 needs explicit chunk-level granularity for PPO.
  - Stores per-chunk RAW input batch (CPU tensors) so PPO update can re-evaluate
    log_prob under updated policy.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from lerobot.common.policies.act.phase2_actor_critic import (
    StochasticActor, factorized_gaussian_log_prob,
)
from lerobot.common.robot_devices.history_buffer import HistoryBuffer
from lerobot.common.robot_devices.safety_monitor import (
    clamp_action, velocity_clip, is_finite_action, imu_health_check,
    SafetyEventLog,
)


@dataclass
class ChunkRolloutData:
    """Per-chunk data captured during real-robot rollout."""
    chunk_idx: int
    # Input batch at chunk start (CPU, no batch dim removed). Used for re-evaluate.
    state_batch: dict
    prev_last_mu_smoothed: Optional[torch.Tensor]   # (A,) or None for first chunk
    # Stochastic policy output at rollout time (frozen)
    action_normalized: torch.Tensor   # (T, A) — what we sent (modulo safety clamps)
    mu_smoothed: torch.Tensor          # (T, A) — μ AFTER chunk-boundary smoothing
    log_prob_old: float
    value_old: float
    sigma: torch.Tensor                # (A,)
    # Reward / OOD info
    nn_distance: float
    valid: bool                        # NN_dist <= τ
    chunk_reward: float                # mean of valid step rewards (or 0 if invalid chunk)
    raw_rewards: list[float] = field(default_factory=list)
    # Per-step real IMU (for diagnostics + reward computation)
    real_imus: list[np.ndarray] = field(default_factory=list)
    # Safety counters within chunk
    n_action_clamp_exceeded: int = 0
    n_velocity_clipped: int = 0


@dataclass
class EpisodeRolloutData:
    chunks: list[ChunkRolloutData] = field(default_factory=list)
    task_success: bool = False
    n_steps: int = 0
    aborted: bool = False
    abort_reason: str = ""

    def chunk_rewards_tensor(self) -> torch.Tensor:
        return torch.tensor([c.chunk_reward for c in self.chunks], dtype=torch.float32)

    def chunk_values_tensor(self) -> torch.Tensor:
        return torch.tensor([c.value_old for c in self.chunks], dtype=torch.float32)

    def chunk_log_probs_tensor(self) -> torch.Tensor:
        return torch.tensor([c.log_prob_old for c in self.chunks], dtype=torch.float32)

    def chunk_valid_mask(self) -> torch.Tensor:
        return torch.tensor([c.valid for c in self.chunks], dtype=torch.bool)


def _batchify_observation(obs: dict, device: torch.device) -> dict:
    """Add batch dim to observation, move tensors to device. Mirrors predict_action's logic."""
    out = {}
    for k, v in obs.items():
        if torch.is_tensor(v):
            t = v
        else:
            t = torch.as_tensor(v)
        if "image" in k and t.dtype == torch.uint8:
            t = t.float() / 255.0
            if t.shape[-1] in (1, 3) and t.dim() == 3:
                t = t.permute(2, 0, 1).contiguous()
        out[k] = t.unsqueeze(0).to(device)
    return out


def rollout_episode(
    robot,
    actor: StochasticActor,
    history_buffer: HistoryBuffer,
    reward_calculator,                    # imu_reward.RewardCalculator
    cfg,                                  # Phase2Config
    update_step: int,
    action_min_raw: torch.Tensor,         # (A,) raw-space lower clamp
    action_max_raw: torch.Tensor,         # (A,) raw-space upper clamp
    safety_log: SafetyEventLog,
    deterministic: bool = False,
    task_success_fn: Optional[Callable[[], bool]] = None,
    fps: Optional[int] = None,
    log_per_step: bool = False,
) -> EpisodeRolloutData:
    """One real-robot Phase 2 episode.

    Args:
        robot: Piper robot interface (.capture_observation, .send_action).
        actor: StochasticActor (wraps ACT + value head).
        history_buffer: maintains across-episode history.
        reward_calculator: RewardCalculator (D2 reward).
        cfg: Phase2Config.
        update_step: PPO update counter (for σ schedule).
        action_min_raw, action_max_raw: clamp bounds in raw action units.
        safety_log: shared event log.
        deterministic: if True, σ=0 (eval gate mode).
        task_success_fn: callback returning bool. Defaults to manual prompt.
        fps: control fps (defaults to cfg.fps).
        log_per_step: print step trace.

    Returns:
        EpisodeRolloutData
    """
    fps = fps or cfg.fps
    expected_n_chunks = cfg.get_n_chunks_per_episode()
    chunk_size = cfg.chunk_size

    # Pre-resolve action mean/std from policy buffers for raw↔norm conversion
    a_mean_buf = actor.policy.normalize_targets.buffer_action
    a_mean = a_mean_buf["mean"].squeeze().detach().cpu()
    a_std = a_mean_buf["std"].squeeze().detach().cpu()

    ep = EpisodeRolloutData()
    prev_last_mu_smoothed: Optional[torch.Tensor] = None
    prev_action_raw: Optional[torch.Tensor] = None

    # Active chunk state
    active_chunk: Optional[ChunkRolloutData] = None
    active_action_norm_chunk: Optional[torch.Tensor] = None
    active_gt_window: Optional[np.ndarray] = None

    start_t = time.perf_counter()

    for global_step in range(expected_n_chunks * chunk_size):
        chunk_idx = global_step // chunk_size
        step_in_chunk = global_step % chunk_size

        # ── Time budget check ───────────────────────────────
        elapsed = time.perf_counter() - start_t
        if elapsed > cfg.episode_time_s:
            ep.aborted = True
            ep.abort_reason = "episode_time_exceeded"
            break

        # ── Capture observation ─────────────────────────────
        try:
            obs = robot.capture_observation()
        except Exception as e:
            ep.aborted = True
            ep.abort_reason = f"capture_observation failed: {e}"
            break

        # ── At chunk boundary: run ACT forward, get GT IMU window ──
        if step_in_chunk == 0:
            # Inject history fields
            obs_with_hist = dict(obs)
            obs_with_hist.update(history_buffer.get_history())
            batch = _batchify_observation(obs_with_hist, actor.device)

            if deterministic:
                # σ=0 for eval gate. Also SKIP chunk-boundary smoothing —
                # smoothing was added to dampen σ-noise discontinuities. Without
                # noise, the smoothing introduces phase lag that hurts the task
                # (causes 5/5 → 1/4 regression vs C3 baseline).
                with torch.no_grad():
                    mu, V = actor._forward_mu_value(batch)
                    mu_smoothed = mu  # NO ramp in deterministic mode
                    sigma_eff = actor.get_sigma(update_step).clamp(min=1e-6)
                    action_chunk_norm = mu_smoothed.clone()
                    log_prob = factorized_gaussian_log_prob(
                        action_chunk_norm, mu_smoothed, sigma_eff
                    )
                rollout_out = {
                    "action": action_chunk_norm,
                    "mu_smoothed": mu_smoothed,
                    "log_prob": log_prob,
                    "value": V,
                    "sigma": sigma_eff,
                }
            else:
                rollout_out = actor.rollout_chunk(batch, prev_last_mu_smoothed, update_step)

            active_action_norm_chunk = rollout_out["action"][0].detach().cpu()  # (T, A)
            mu_smoothed_cpu = rollout_out["mu_smoothed"][0].detach().cpu()

            # GT IMU window from state-NN
            state_raw_np = (
                obs["observation.state"].detach().cpu().numpy()
                if torch.is_tensor(obs["observation.state"])
                else np.asarray(obs["observation.state"])
            )
            gt_window, nn_dist, valid = reward_calculator.fetch_chunk_gt(state_raw_np)
            active_gt_window = gt_window  # (chunk_size, imu_dim)

            # Init ChunkRolloutData. Store the BATCHIFIED batch (image already in
            # CHW float format after _batchify_observation) so re-evaluate during
            # PPO update can call normalize_inputs without re-permuting.
            state_batch_cpu = {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                                for k, v in batch.items()}
            active_chunk = ChunkRolloutData(
                chunk_idx=chunk_idx,
                state_batch=state_batch_cpu,
                prev_last_mu_smoothed=(prev_last_mu_smoothed.detach().cpu()
                                         if prev_last_mu_smoothed is not None else None),
                action_normalized=active_action_norm_chunk,
                mu_smoothed=mu_smoothed_cpu,
                log_prob_old=float(rollout_out["log_prob"].item()),
                value_old=float(rollout_out["value"].item()),
                sigma=rollout_out["sigma"].detach().cpu(),
                nn_distance=float(nn_dist),
                valid=bool(valid),
                chunk_reward=0.0,   # filled at chunk end
            )

        # ── Per-step: get action, safety, send, record ─────
        action_t_norm = active_action_norm_chunk[step_in_chunk].clone()
        action_t_raw = action_t_norm * a_std + a_mean

        # NaN check
        ok, msg = is_finite_action(action_t_raw)
        if not ok:
            ep.aborted = True
            ep.abort_reason = f"NaN action @ step {global_step}: {msg}"
            safety_log.increment("nan_action_detected")
            break

        # Clamp to action range
        action_t_raw_c, ex_clamp = clamp_action(action_t_raw, action_min_raw, action_max_raw)
        if ex_clamp:
            active_chunk.n_action_clamp_exceeded += 1
            safety_log.increment("action_clamp_exceeded")

        # Velocity clip vs prev step — only in stochastic mode where σ noise can
        # cause large deltas. In deterministic mode the C1 actions are smooth by
        # design and clipping just introduces phase artifacts (regression vs C3).
        if (not deterministic) and prev_action_raw is not None:
            action_t_raw_c, ex_vel = velocity_clip(
                action_t_raw_c, prev_action_raw, cfg.velocity_max_delta
            )
            if ex_vel:
                active_chunk.n_velocity_clipped += 1
                safety_log.increment("velocity_clamp_exceeded")

        # Send
        try:
            sent_raw = robot.send_action(action_t_raw_c)
        except Exception as e:
            ep.aborted = True
            ep.abort_reason = f"send_action failed @ step {global_step}: {e}"
            break

        # IMU sanity check + reward
        imu_t_tensor = obs.get("observation.imu_real")
        if imu_t_tensor is None:
            ep.aborted = True
            ep.abort_reason = "robot did not return IMU observation"
            break

        imu_ok, imu_msg = imu_health_check(imu_t_tensor, cfg.imu_safety_max_accel_norm)
        if not imu_ok:
            ep.aborted = True
            ep.abort_reason = f"IMU health failed: {imu_msg}"
            safety_log.increment("imu_health_failed")
            break

        imu_np = imu_t_tensor.detach().cpu().numpy() if torch.is_tensor(imu_t_tensor) else np.asarray(imu_t_tensor)
        active_chunk.real_imus.append(imu_np)

        gt_imu_t = active_gt_window[step_in_chunk]
        # Determine if THIS step is the last step of the episode
        is_last_step = (global_step == expected_n_chunks * chunk_size - 1)
        task_success_now = False  # will be filled at episode end (or by callback)

        R, _ = reward_calculator.step_reward(
            imu_np, gt_imu_t,
            task_success=task_success_now, chunk_valid=active_chunk.valid
        )
        active_chunk.raw_rewards.append(R)

        # Update history buffer (use sent action, NORMALIZED for consistency
        # with v2 dataset stats — wait: history buffer normalizes itself.
        # Pass raw action; HistoryBuffer normalizes using v2 stats internally.)
        history_buffer.append_step(sent_raw, imu_t_tensor)

        # Update prev for next step
        prev_action_raw = sent_raw if torch.is_tensor(sent_raw) else torch.as_tensor(sent_raw)

        # ── End-of-chunk bookkeeping ──────────────────────
        if step_in_chunk == chunk_size - 1:
            # Aggregate chunk reward (mean over valid steps inside chunk)
            if active_chunk.valid and active_chunk.raw_rewards:
                active_chunk.chunk_reward = float(np.mean(active_chunk.raw_rewards))
            else:
                active_chunk.chunk_reward = 0.0
            ep.chunks.append(active_chunk)
            prev_last_mu_smoothed = active_chunk.mu_smoothed[-1].clone()

        ep.n_steps += 1

        # FPS limit
        if fps is not None and fps > 0:
            target_dt = 1.0 / fps
            cur_dt = (time.perf_counter() - start_t) - global_step * target_dt
            sleep_t = target_dt - max(cur_dt, 0)
            if sleep_t > 0:
                time.sleep(sleep_t)

        if log_per_step and global_step % 30 == 0:
            print(f"  step {global_step:>3d}  chunk={chunk_idx} "
                  f"valid={active_chunk.valid}  R={R:+.3f}")

    # ── End of episode ────────────────────────────────────
    history_buffer.end_episode()

    # Determine task success
    if ep.aborted:
        ep.task_success = False
    elif task_success_fn is not None:
        try:
            ep.task_success = bool(task_success_fn())
        except Exception:
            ep.task_success = False
    else:
        # Default: manual prompt
        try:
            ans = input(f"Episode finished ({ep.n_steps} steps). Task success? [y/N]: ").strip().lower()
            ep.task_success = ans in ("y", "yes", "1")
        except (EOFError, KeyboardInterrupt):
            ep.task_success = False

    return ep
