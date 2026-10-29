"""Central configuration for Phase 2 RL fine-tuning of autoregressive ACT.

All Phase 2 hyperparameters live here. Every Phase 2 module (reward setup,
modified ACT model, safety monitor, PPO actor-critic, rollout, train loop)
should read from a `Phase2Config` instance instead of hard-coding values.

Schedule fields use parallel tuples (steps, values) instead of list-of-tuples
because draccus / dataclasses serialize them more cleanly.

Decisions per D2-D6 spec (see THESIS_PROGRESS_REPORT.md §6):
  - Reward: L2 IMU distance + state-NN GT alignment + OOD skip + success bonus
  - Trajectory: chunk-as-trajectory, T≈6 chunks per episode
  - RL: PPO with σ_fixed Gaussian noise (no σ-head)
  - Loss: L_BC + α(curriculum) × L_RL + 0.5 × L_V (V detached from encoder)
  - Safety: action clamp + velocity limit + boundary ramp + emergency stop
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Phase2Config:
    """All Phase 2 RL fine-tuning hyperparameters."""

    # ============================================================
    # Stochastic policy (D4)
    # ============================================================
    # Per-dim σ in normalized action space. 7 dims = 6 joints + 1 gripper.
    # Gripper has narrow std in v2 → bigger normalized σ to keep absolute noise
    # comparable to joint noise.
    sigma_per_dim: tuple[float, ...] = (
        0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.20
    )
    # Step schedule: at each step value, σ multiplier changes (multiplied
    # against sigma_per_dim). Discrete jumps for simplicity.
    # Round B (v4b): σ taper aligned to update_step. Round A bug had
    # (0, 25K, 50K) which never advanced past stage 0 in 100 updates, so
    # σ stayed at 1.0× the whole run (§6.9 cause #2). Now σ tapers to 0.7
    # at update 100 and 0.5 at update 200 across the 300-update budget.
    sigma_schedule_steps: tuple[int, ...] = (0, 100, 200)
    sigma_schedule_values: tuple[float, ...] = (1.0, 0.7, 0.5)
    # → effective σ at step t is sigma_per_dim × sigma_schedule_values[stage(t)]

    # ============================================================
    # Reward (D2)
    # ============================================================
    # Threshold for state-NN OOD rejection. "auto" = max(v2_p95, c1_mean+3σ).
    nn_threshold_method: str = "auto"
    # If not None, overrides auto. Set after running d2_setup_imu_reward.py.
    nn_threshold_value: float | None = None
    chunk_size_for_gt: int = 100
    imu_smooth_window: int = 3  # rolling-mean over IMU before reward calc
    success_bonus: float = 10.0  # +10 at last step IF task_success
    # Reward standardization + clipping
    reward_ema_alpha: float = 0.99  # EMA decay for reward stats
    reward_clip_min: float = -3.0
    reward_clip_max: float = 3.0
    # Sim-to-real IMU offset correction (None = trust v2 stats)
    real_imu_calib_episodes: int = 0  # >0 = calibrate at episode 0..N-1
    # Number of distinct demos to average for cross-demo NN reward.
    # k=1 → Round A style (single nearest demo).
    # k=5 → Round C style (cross-demo averaging, decouples per-demo noise).
    # Round B (v4b) uses k=1 to keep reward identical to Round A for clean
    # single-axis (schedule) ablation.
    reward_k_demos: int = 1

    # ============================================================
    # Trajectory / GAE (D3)
    # ============================================================
    gamma_inter: float = 0.99  # discount across chunks
    gamma_intra: float = 1.0   # within-chunk reward aggregation (no discount)
    gae_lambda: float = 0.95

    # ============================================================
    # PPO (D4)
    # ============================================================
    ppo_clip_eps: float = 0.2
    ppo_k_epoch: int = 4
    ppo_target_kl: float = 50.0  # KL early stop. With σ=0.05, KL = 0.5×Σ(Δμ/σ)²
                                   # is hypersensitive: tiny μ shifts → huge KL.
                                   # 0.05 (LLM-RLHF default) was unrealistic;
                                   # 50.0 matches the σ=0.05 numeric scale.
    k_episodes_per_update: int = 4
    minibatch_size_chunks: int = 12
    actor_lr: float = 1e-5
    critic_lr: float = 1e-4
    grad_clip_norm: float = 1.0

    # ============================================================
    # Loss coordination (D5)
    # ============================================================
    # α schedule: weight of RL loss in total loss
    # v4c (Round D / α sweet-spot search): constant α=0.012, midpoint between
    # Round A's 0.005 (RL too weak, no smoothness change) and Round B's 0.025
    # (RL too strong, task collapsed in 8 updates). Single-value schedule
    # makes the α effect isolable.
    alpha_schedule_steps: tuple[int, ...] = (0,)
    alpha_schedule_values: tuple[float, ...] = (0.012,)
    # Reference schedules:
    #   Round A: (0,100,300)/(0.005,0.025,0.05) → 100 updates
    #   Round B: (0,50,150)/(0.025,0.05,0.10)   → 300 updates, collapsed @ 8
    value_loss_coef: float = 0.5  # coefficient of L_V (V detached from encoder)
    bc_batch_size_chunks: int = 12  # B=12 × L=600 ≈ 14GB; B=24 was OOM on RTX 3090
    # KL penalty (off by default, enable on failure)
    use_kl_penalty: bool = False
    kl_beta: float = 0.01

    # ============================================================
    # Safety (D6)
    # ============================================================
    # Action range = v2 stats min/max × this margin
    # Round B (v4b): middle ground after v4a (1.2 collision) and v4a_safe (0.95
    # over-clamping caused PPO ratio mismatch + KL explosion to 1000-10000).
    # 1.05 gives ACT 5% explore room while staying close to demo range.
    action_clamp_margin: float = 1.05
    # Step-to-step velocity limit (rad). Only enforced in stochastic mode.
    # Round B: middle of v4a (0.10) and v4a_safe (0.05).
    velocity_max_delta: float = 0.08
    # Chunk boundary μ smoothing
    chunk_boundary_ramp_steps: int = 5
    # IMU sanity check
    imu_safety_max_accel_norm: float = 50.0  # m/s²
    # Consecutive episode failure threshold for auto-pause
    consecutive_failure_threshold: int = 3

    # ============================================================
    # Critic warmup (D4 + D6)
    # ============================================================
    warmup_episodes: int = 20  # deterministic rollouts for V data
    warmup_gpu_steps: int = 2000  # value head fitting iterations

    # ============================================================
    # Main loop (D6)
    # ============================================================
    n_updates: int = 150  # v4c: shorter budget for α sweet-spot test; abort fast if no improvement.
    save_freq: int = 20  # save ckpt every N updates

    # ============================================================
    # Inter-episode reset prompt
    # ============================================================
    # If True, skip the "press Enter to start next episode" prompt between
    # consecutive rollouts. Useful for unattended dry-run mode; in real
    # deployment leave as False so operator can manually reset object positions.
    skip_inter_episode_reset: bool = False

    # ============================================================
    # Eval gate (D6)
    # ============================================================
    eval_freq: int = 25
    eval_n_episodes: int = 5
    eval_warn_threshold: int = 4    # ≥4/5 OK; <4 = warn
    eval_rollback_threshold: int = 3  # =3/5 → rollback + reduce α/σ
    eval_terminate_threshold: int = 2  # ≤2/5 → terminate training

    # ============================================================
    # Drift detection (D6 monitoring)
    # ============================================================
    # L_BC absolute value early warning. C1-v3 step 70K had l1≈0.019.
    drift_warn_ratio: float = 2.0   # WARN if L_BC > 2.0 × C1_baseline
    drift_alert_ratio: float = 3.0  # ALERT if L_BC > 3.0 × C1_baseline
    c1_l1_baseline: float = 0.019

    # ============================================================
    # Paths (updated 2026-05-09 to v3 — fixed IMU/camera mount)
    # ============================================================
    # Round B (v4b): fresh start from C1-v3 70K (clean BC baseline) for
    # single-axis ablation vs Round A.
    c1_ckpt_path: str = (
        "/home/enolab-guo/claude-test-2/lerobot-piper/outputs/train/"
        "2026-05-08/C1_v3_history_seed42/checkpoints/070000/pretrained_model"
    )
    v2_dataset_repo: str = "local/piper_pick_place_imu_v3"
    output_dir: str = "outputs/train/phase2_v4c"
    reward_computer_pickle: str = "outputs/eval/d2_imu_reward_computer.pkl"

    # ============================================================
    # Episode timing (must match training-time ACT chunk_size and fps)
    # ============================================================
    fps: int = 30
    episode_time_s: float = 20.0
    reset_time_s: float = 10.0
    chunk_size: int = 100  # ACT chunk_size, must match C1 ckpt config

    # ============================================================
    # Logging
    # ============================================================
    wandb_project: str = "lerobot-piper-phase2"
    wandb_enable: bool = True
    log_freq_updates: int = 1  # log every N PPO updates

    # ============================================================
    # Debugging / dry-run flags
    # ============================================================
    dry_run: bool = False  # if True, skip real-robot rollouts (for code testing)
    seed: int = 42

    # ─── Helper functions for schedules ──────────────────────────

    def get_alpha(self, update_step: int) -> float:
        """Step-schedule lookup for α (RL loss weight)."""
        return _step_schedule(self.alpha_schedule_steps,
                              self.alpha_schedule_values,
                              update_step)

    def get_sigma_multiplier(self, update_step: int) -> float:
        """Step-schedule lookup for global σ multiplier."""
        return _step_schedule(self.sigma_schedule_steps,
                              self.sigma_schedule_values,
                              update_step)

    def get_sigma_per_dim(self, update_step: int) -> tuple[float, ...]:
        """Effective per-dim σ at given update step."""
        m = self.get_sigma_multiplier(update_step)
        return tuple(s * m for s in self.sigma_per_dim)

    def get_n_chunks_per_episode(self) -> int:
        """Number of ACT decisions per episode (= T in trajectory terms)."""
        return int(self.episode_time_s * self.fps / self.chunk_size)


def _step_schedule(steps: tuple[int, ...], values: tuple[float, ...],
                   t: int) -> float:
    """Discrete step-schedule lookup. `steps` must be ascending."""
    assert len(steps) == len(values), "steps and values must have same length"
    val = values[0]
    for s, v in zip(steps, values):
        if t >= s:
            val = v
        else:
            break
    return val


# ─── Sanity helpers ──────────────────────────────────────────

def validate_config(cfg: Phase2Config) -> None:
    """Raise ValueError if config has an obviously broken combination."""
    if len(cfg.sigma_per_dim) != 7:
        raise ValueError(f"sigma_per_dim must have length 7, got {len(cfg.sigma_per_dim)}")
    if cfg.eval_warn_threshold < cfg.eval_rollback_threshold:
        raise ValueError("eval_warn_threshold should be >= eval_rollback_threshold")
    if cfg.eval_rollback_threshold < cfg.eval_terminate_threshold:
        raise ValueError("eval_rollback_threshold should be >= eval_terminate_threshold")
    if cfg.alpha_schedule_steps[0] != 0:
        raise ValueError("alpha_schedule_steps must start with step 0")
    if cfg.sigma_schedule_steps[0] != 0:
        raise ValueError("sigma_schedule_steps must start with step 0")
    n_chunks = cfg.get_n_chunks_per_episode()
    if n_chunks < 2:
        raise ValueError(f"episode_time_s × fps / chunk_size = {n_chunks} < 2; too few chunks/episode for GAE")
    expected_chunks = int(cfg.episode_time_s * cfg.fps / cfg.chunk_size)
    if expected_chunks != n_chunks:
        raise ValueError("episode_time_s/chunk_size mismatch")
