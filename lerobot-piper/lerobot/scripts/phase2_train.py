#!/usr/bin/env python3
"""Phase 2 main training loop — PPO fine-tuning of ACT with IMU smoothness reward.

Implements D2-D6 spec end-to-end:
  1. Loads C1 ckpt + adds value head
  2. Critic warmup (deterministic rollouts → V head fitting)
  3. Main PPO loop:
       rollout K episodes (stochastic) → GAE → K_epoch PPO update with BC anchor
  4. Eval gate every N updates (5 deterministic episodes)

Run:
    python -m lerobot.scripts.phase2_train [--phase2.<field>=value ...]
"""
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

import draccus
import numpy as np
import torch

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.configuration_phase2 import Phase2Config, validate_config
from lerobot.common.policies.act.imu_reward import IMURewardComputer, RewardCalculator
from lerobot.common.policies.act.phase2_actor_critic import (
    StochasticActor, compute_gae, ppo_clip_loss, value_loss, standardize_advantages,
    factorized_gaussian_log_prob, kl_gaussian_fixed_sigma,
)
from lerobot.common.policies.act.phase2_rollout import (
    rollout_episode, EpisodeRolloutData,
)
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.history_buffer import HistoryBuffer
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.robots.configs import RobotConfig
from lerobot.common.robot_devices.safety_monitor import (
    SafetyEventLog, ConsecutiveFailureTracker, compute_action_clamp_range,
)
from lerobot.common.utils.utils import init_logging
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# MockRobot — for dry-run mode (test code path without hardware)
# ════════════════════════════════════════════════════════════════

class MockRobot:
    """Drop-in replacement for PiperRobot in dry-run mode.

    Returns synthetic observations sampled from v2 statistics, accepts any
    send_action call, never blocks. Used to verify the full Phase 2 pipeline
    integration before connecting real hardware.
    """

    def __init__(self, ds_meta):
        self.is_connected = True
        self.cameras = {"two": None}  # placeholder for camera key check
        # Pull stats for realistic observations
        self.state_mean = torch.as_tensor(ds_meta.stats["observation.state"]["mean"], dtype=torch.float32)
        self.state_std = torch.as_tensor(ds_meta.stats["observation.state"]["std"], dtype=torch.float32)
        self.imu_mean = torch.as_tensor(ds_meta.stats["observation.imu_real"]["mean"], dtype=torch.float32)
        self.imu_std = torch.as_tensor(ds_meta.stats["observation.imu_real"]["std"], dtype=torch.float32)

    def connect(self):
        logger.info("[MockRobot] connect() — no-op")

    def disconnect(self):
        logger.info("[MockRobot] disconnect() — no-op")

    def capture_observation(self):
        # Sample from v2 distribution
        state = self.state_mean + 0.1 * self.state_std * torch.randn_like(self.state_std)
        imu = self.imu_mean + 0.5 * self.imu_std * torch.randn_like(self.imu_std)
        # Random image
        img = torch.randint(0, 256, (480, 640, 3), dtype=torch.uint8)
        return {
            "observation.state": state,
            "observation.imu_real": imu,
            "observation.images.two": img,
        }

    def send_action(self, action: torch.Tensor):
        # Just echo back (with tiny perturbation to simulate motor noise)
        return action + 0.001 * torch.randn_like(action)


@dataclass
class Phase2RunConfig:
    """Top-level run config combining Phase2Config + robot."""
    phase2: Phase2Config = draccus.field(default_factory=Phase2Config)
    robot: RobotConfig | None = None
    dry_run: bool = False  # if True, skip real-robot ops (use mock for testing)


# ════════════════════════════════════════════════════════════════
# Setup helpers
# ════════════════════════════════════════════════════════════════

def load_actor_with_value_head(c1_ckpt_path: str, dataset_repo: str, device: str):
    """Load C1 ckpt and grant it a value head (random init)."""
    train_cfg_path = Path(c1_ckpt_path) / "train_config.json"
    train_cfg_dict = json.load(open(train_cfg_path))
    train_cfg_dict["policy"]["enable_value_head"] = True
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(train_cfg_dict, f)
        tmp = f.name
    cfg = draccus.parse(TrainPipelineConfig, tmp, args=[])
    cfg.policy.pretrained_path = c1_ckpt_path

    ds_meta = LeRobotDatasetMetadata(dataset_repo, root=None)
    policy = make_policy(cfg=cfg.policy, ds_meta=ds_meta, device=device)
    return policy, cfg, ds_meta


def setup_optimizers(policy, cfg: Phase2Config):
    """Two parameter groups: actor (everything except value_head) + critic (value_head)."""
    value_params = list(policy.model.value_head.parameters())
    value_param_ids = {id(p) for p in value_params}
    actor_params = [p for p in policy.parameters() if id(p) not in value_param_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": actor_params, "lr": cfg.actor_lr},
            {"params": value_params, "lr": cfg.critic_lr},
        ],
        weight_decay=1e-4,
    )
    return optimizer


def freeze_actor_for_critic_warmup(policy):
    """Freeze everything except value_head."""
    for p in policy.parameters():
        p.requires_grad = False
    for p in policy.model.value_head.parameters():
        p.requires_grad = True


def unfreeze_all(policy):
    for p in policy.parameters():
        p.requires_grad = True


# ════════════════════════════════════════════════════════════════
# Inter-episode reset prompt
# ════════════════════════════════════════════════════════════════

def _wait_for_reset(cfg) -> None:
    """Pause between episodes so operator can manually reset workspace.
    Press Enter to start the next episode (or Ctrl+C to abort training)."""
    if getattr(cfg, "skip_inter_episode_reset", False):
        return
    try:
        input("  >>> Reset workspace (place object back), press Enter to start next episode...")
    except (EOFError, KeyboardInterrupt):
        raise


# ════════════════════════════════════════════════════════════════
# Critic warmup
# ════════════════════════════════════════════════════════════════

def critic_warmup(
    actor: StochasticActor,
    robot,
    history_buffer,
    reward_calculator,
    cfg: Phase2Config,
    optimizer: torch.optim.Optimizer,
    safety_log: SafetyEventLog,
    action_min, action_max,
    device,
    wandb_logger,
):
    """Stage A: collect deterministic rollouts. Stage B: train V head."""
    logger.info("== Critic warmup Stage A: deterministic rollouts ==")
    freeze_actor_for_critic_warmup(actor.policy)

    warmup_buffer = []  # list of (state_batch, return_value)
    for ep in range(cfg.warmup_episodes):
        logger.info(f"  warmup ep {ep+1}/{cfg.warmup_episodes}")
        if ep > 0:
            _wait_for_reset(cfg)
        ep_data = rollout_episode(
            robot, actor, history_buffer, reward_calculator, cfg,
            update_step=0,
            action_min_raw=action_min, action_max_raw=action_max,
            safety_log=safety_log,
            deterministic=True,
        )
        # Compute Monte-Carlo return per chunk for V target
        rewards = ep_data.chunk_rewards_tensor()
        T = len(rewards)
        returns = torch.zeros(T)
        running = 0.0
        for t in reversed(range(T)):
            running = float(rewards[t]) + cfg.gamma_inter * running
            returns[t] = running
        for chunk, R in zip(ep_data.chunks, returns):
            warmup_buffer.append((chunk.state_batch, float(R)))

    logger.info(f"== Critic warmup Stage B: train V head ({len(warmup_buffer)} samples) ==")
    actor.policy.eval()  # VAE assertion gated on training mode; gradients still flow in eval
    for step in range(cfg.warmup_gpu_steps):
        # Sample mini-batch of (state_batch, return_value) — one per gradient step is fine
        idx = np.random.randint(0, len(warmup_buffer))
        state_batch_cpu, target_R = warmup_buffer[idx]
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                  for k, v in state_batch_cpu.items()}
        target = torch.tensor([target_R], device=device, dtype=torch.float32)

        _, V = actor._forward_mu_value(batch)
        L_V = ((V - target) ** 2).mean()
        L_V.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            logger.info(f"  warmup V step {step}: L_V={L_V.item():.4f}")
            if wandb_logger:
                wandb_logger.log(
                    {"warmup/V_loss": L_V.item(), "warmup/step": step},
                    step=step,
                )

    unfreeze_all(actor.policy)
    logger.info("Critic warmup complete.")


# ════════════════════════════════════════════════════════════════
# Single PPO update step
# ════════════════════════════════════════════════════════════════

def collect_rollouts_for_update(
    actor: StochasticActor, robot, history_buffer, reward_calculator,
    cfg: Phase2Config, update_step: int,
    action_min, action_max, safety_log,
) -> list[EpisodeRolloutData]:
    """Collect K_episodes_per_update stochastic rollouts."""
    rollouts = []
    for ep_idx in range(cfg.k_episodes_per_update):
        logger.info(f"  rollout ep {ep_idx+1}/{cfg.k_episodes_per_update}")
        if ep_idx > 0:
            _wait_for_reset(cfg)
        ep = rollout_episode(
            robot, actor, history_buffer, reward_calculator, cfg, update_step,
            action_min_raw=action_min, action_max_raw=action_max,
            safety_log=safety_log, deterministic=False,
        )
        rollouts.append(ep)
    return rollouts


def compute_per_episode_advantages(
    episodes: list[EpisodeRolloutData], cfg: Phase2Config, device
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Compute per-episode (advantages, returns) with GAE + valid_mask."""
    all_advs, all_rets = [], []
    for ep in episodes:
        rewards = ep.chunk_rewards_tensor().to(device)
        values = ep.chunk_values_tensor().to(device)
        valid = ep.chunk_valid_mask().to(device)
        # Append terminal V=0 (episode ends at fixed time)
        values_full = torch.cat([values, torch.zeros(1, device=device)])
        adv, ret = compute_gae(rewards, values_full, valid,
                                cfg.gamma_inter, cfg.gae_lambda)
        all_advs.append(adv)
        all_rets.append(ret)
    return all_advs, all_rets


def ppo_update_step(
    actor: StochasticActor,
    rollouts: list[EpisodeRolloutData],
    advantages_per_ep: list[torch.Tensor],
    returns_per_ep: list[torch.Tensor],
    bc_loader_iter,                # cycle iterator over v2 BC dataloader
    cfg: Phase2Config,
    optimizer,
    update_step: int,
    device,
) -> dict:
    """Single PPO update across K_epochs.
    Returns dict of aggregated metrics.
    """
    # Flatten chunks across episodes; track per-chunk advantage/return
    flat_chunks = []
    flat_advs, flat_rets = [], []
    for ep, adv, ret in zip(rollouts, advantages_per_ep, returns_per_ep):
        for ck, a, r in zip(ep.chunks, adv, ret):
            flat_chunks.append(ck)
            flat_advs.append(a.item())
            flat_rets.append(r.item())

    if not flat_chunks:
        logger.warning("no rollout chunks collected, skipping update")
        return {"skipped": True}

    flat_advs_t = torch.tensor(flat_advs, dtype=torch.float32, device=device)
    flat_rets_t = torch.tensor(flat_rets, dtype=torch.float32, device=device)
    valid_mask = torch.tensor(
        [c.valid for c in flat_chunks], dtype=torch.bool, device=device
    )

    # Standardize advantages by valid-step stats
    flat_advs_t = standardize_advantages(flat_advs_t, valid_mask)

    α = cfg.get_alpha(update_step)
    metrics_accum = {
        "L_BC": 0.0, "L_RL": 0.0, "L_V": 0.0,
        "ratio_clip_frac": 0.0, "ratio_mean": 0.0,
        "kl_new_old": 0.0,
    }
    n_minibatches = 0
    early_stopped_epochs = 0

    actor.policy.eval()  # VAE assertion gated on training mode; gradients still flow in eval
    for ppo_epoch in range(cfg.ppo_k_epoch):
        # Shuffle chunks
        n = len(flat_chunks)
        perm = np.random.permutation(n)

        for start in range(0, n, cfg.minibatch_size_chunks):
            mb_idx = perm[start : start + cfg.minibatch_size_chunks]
            if len(mb_idx) == 0:
                continue

            # Memory-aware gradient accumulation:
            # call .backward() per chunk (immediate) so each chunk's activations
            # are freed before the next forward pass. Without this, all 12 chunk
            # graphs + the BC batch sit in GPU memory simultaneously → OOM on 3090.
            mb_kl = 0.0
            mb_loss_RL_sum, mb_loss_V_sum = 0.0, 0.0
            mb_ratio_clip_frac_sum = 0.0
            mb_ratio_mean_sum = 0.0
            n_in_mb = len(mb_idx)
            inv_n = 1.0 / max(n_in_mb, 1)

            for j in mb_idx:
                ck = flat_chunks[j]
                batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                          for k, v in ck.state_batch.items()}
                action_norm = ck.action_normalized.unsqueeze(0).to(device)
                prev_last = (ck.prev_last_mu_smoothed.to(device)
                              if ck.prev_last_mu_smoothed is not None else None)

                eval_out = actor.re_evaluate(batch, action_norm, prev_last, update_step)
                lp_new = eval_out["log_prob"]
                V_new = eval_out["value"]
                mu_new = eval_out["mu_smoothed"]

                advs_one = flat_advs_t[j:j+1]
                ret_one = flat_rets_t[j:j+1]
                valid_one = valid_mask[j:j+1]
                lp_old_one = torch.tensor([ck.log_prob_old], device=device, dtype=torch.float32)

                L_RL_one, m_one = ppo_clip_loss(
                    lp_new, lp_old_one, advs_one, valid_one, cfg.ppo_clip_eps
                )
                L_V_one = value_loss(V_new, ret_one, valid_one)

                with torch.no_grad():
                    sigma = eval_out["sigma"]
                    mu_old = ck.mu_smoothed.unsqueeze(0).to(device)
                    kl_one = kl_gaussian_fixed_sigma(mu_new, mu_old, sigma).mean().item()

                # Per-chunk loss (mean over mini-batch via inv_n) scaled by α
                chunk_loss = (L_RL_one + cfg.value_loss_coef * L_V_one) * inv_n * α
                chunk_loss.backward()  # IMMEDIATE backward, frees activations

                mb_loss_RL_sum += L_RL_one.item()
                mb_loss_V_sum += L_V_one.item()
                mb_ratio_clip_frac_sum += m_one["ratio_clip_frac"]
                mb_ratio_mean_sum += m_one["ratio_mean"]
                mb_kl += kl_one
                # free graph references
                del chunk_loss, L_RL_one, L_V_one, eval_out, batch
            mb_kl /= max(n_in_mb, 1)

            # BC anchor loss — separate forward+backward so its activations don't
            # coexist with the PPO chunk graphs.
            try:
                bc_batch = next(bc_loader_iter)
            except StopIteration:
                bc_batch = None
            if bc_batch is not None:
                bc_batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                              for k, v in bc_batch.items()}
                bc_loss_total, bc_loss_dict = actor.policy.forward(bc_batch)
                bc_loss_total.backward()
                metrics_accum["L_BC"] += bc_loss_dict.get("l1_loss", 0.0)
                del bc_loss_total, bc_batch

            torch.nn.utils.clip_grad_norm_(actor.policy.parameters(),
                                             cfg.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad()

            metrics_accum["L_RL"] += mb_loss_RL_sum / max(n_in_mb, 1)
            metrics_accum["L_V"] += mb_loss_V_sum / max(n_in_mb, 1)
            metrics_accum["ratio_clip_frac"] += mb_ratio_clip_frac_sum / max(n_in_mb, 1)
            metrics_accum["ratio_mean"] += mb_ratio_mean_sum / max(n_in_mb, 1)
            metrics_accum["kl_new_old"] += mb_kl
            n_minibatches += 1

            # KL early stop within epoch
            if mb_kl > cfg.ppo_target_kl:
                logger.info(f"  PPO epoch {ppo_epoch}: KL={mb_kl:.3f} > {cfg.ppo_target_kl}, early stop")
                early_stopped_epochs += 1
                break

    if n_minibatches > 0:
        for k in metrics_accum:
            metrics_accum[k] /= n_minibatches
    metrics_accum["alpha"] = α
    metrics_accum["n_minibatches"] = n_minibatches
    metrics_accum["early_stopped_epochs"] = early_stopped_epochs
    return metrics_accum


# ════════════════════════════════════════════════════════════════
# Main entry
# ════════════════════════════════════════════════════════════════

@parser.wrap()
def main(cfg: Phase2RunConfig):
    init_logging()
    p2 = cfg.phase2
    validate_config(p2)
    logger.info("Phase 2 RUN CONFIG:\n" + pformat(p2.__dict__))

    if cfg.robot is None and not cfg.dry_run:
        raise ValueError("--robot.<...> required (or set --dry_run=true)")

    # WandB — direct init (avoids the lerobot WandBLogger shim which expects
    # many TrainPipelineConfig-specific attrs).
    wandb_logger = None
    if p2.wandb_enable:
        try:
            import wandb
            wandb.init(
                project=p2.wandb_project,
                name=Path(p2.output_dir).name,
                config={k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in p2.__dict__.items()},
                dir=str(Path(p2.output_dir)),
                tags=["phase2", "piper"],
            )
            class _DirectWandB:
                def log(self, data, step=None):
                    wandb.log(data, step=step)
            wandb_logger = _DirectWandB()
            logger.info(f"WandB initialised: project={p2.wandb_project}, run={Path(p2.output_dir).name}")
        except Exception as e:
            logger.warning(f"WandB init failed: {e}; logging locally only")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(p2.seed)
    np.random.seed(p2.seed)

    # ── Load actor ──────────────────────────────────────────
    logger.info(f"Loading C1 ckpt + value head from {p2.c1_ckpt_path}")
    policy, train_cfg, ds_meta = load_actor_with_value_head(
        p2.c1_ckpt_path, p2.v2_dataset_repo, device
    )
    actor = StochasticActor(policy, p2, device=device)
    optimizer = setup_optimizers(policy, p2)

    # ── Load IMU reward computer ───────────────────────────
    logger.info(f"Loading IMURewardComputer from {p2.reward_computer_pickle}")
    computer = IMURewardComputer.load(p2.reward_computer_pickle)
    reward_calculator = RewardCalculator(computer, p2)
    logger.info(f"  τ = {computer.tau:.3f}")

    # ── Build robot (real or mock) ────────────────────────
    if cfg.dry_run:
        logger.info("[DRY-RUN] Using MockRobot (no real hardware)")
        robot = MockRobot(ds_meta)
        # Reduce expensive params for fast dry-run iteration
        p2.warmup_episodes = min(p2.warmup_episodes, 2)
        p2.warmup_gpu_steps = min(p2.warmup_gpu_steps, 50)
        p2.k_episodes_per_update = min(p2.k_episodes_per_update, 2)
        p2.n_updates = min(p2.n_updates, 3)
        p2.eval_freq = max(p2.eval_freq, 100)  # don't trigger eval in dry-run
        logger.info(f"[DRY-RUN] Reduced budget: {p2.warmup_episodes} warmup ep, "
                    f"{p2.n_updates} updates, K={p2.k_episodes_per_update}")
    else:
        cfg.robot.inference_time = True  # type: ignore
        robot = make_robot_from_config(cfg.robot)
    if not robot.is_connected:
        robot.connect()
    history_buffer = HistoryBuffer.from_policy_and_dataset_meta(policy, ds_meta, device=device)

    # ── Action clamp range from v2 stats ───────────────────
    action_stats = ds_meta.stats["action"]
    action_min, action_max = compute_action_clamp_range(action_stats, p2.action_clamp_margin)
    action_min = action_min.to(device); action_max = action_max.to(device)

    # ── BC dataloader ──────────────────────────────────────
    logger.info("Building BC dataloader (v2)")
    bc_dataset = make_dataset(train_cfg)
    bc_loader = torch.utils.data.DataLoader(
        bc_dataset, batch_size=p2.bc_batch_size_chunks,
        shuffle=True, num_workers=2, drop_last=True,
    )

    def cycle(it):
        while True:
            for x in it:
                yield x
    bc_iter = cycle(bc_loader)

    # ── Misc state ──────────────────────────────────────────
    safety_log = SafetyEventLog()
    failure_tracker = ConsecutiveFailureTracker(p2.consecutive_failure_threshold)

    # ── Critic warmup ──────────────────────────────────────
    critic_warmup(
        actor, robot, history_buffer, reward_calculator, p2,
        optimizer, safety_log, action_min, action_max, device, wandb_logger,
    )

    # ── Main PPO loop ──────────────────────────────────────
    ckpt_dir = Path(p2.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for update_step in range(p2.n_updates):
        logger.info("=" * 60)
        logger.info(f"PPO update {update_step+1}/{p2.n_updates}  α={p2.get_alpha(update_step):.3f}  σ_mult={p2.get_sigma_multiplier(update_step):.2f}")
        logger.info("=" * 60)

        rollouts = collect_rollouts_for_update(
            actor, robot, history_buffer, reward_calculator, p2, update_step,
            action_min, action_max, safety_log,
        )

        for ep in rollouts:
            failure_tracker.record(ep.task_success and not ep.aborted)
        if failure_tracker.should_pause():
            logger.error(
                f"{p2.consecutive_failure_threshold} consecutive failures, pausing for human review"
            )
            input("Press Enter to continue or Ctrl+C to abort: ")
            failure_tracker.reset()

        advs, rets = compute_per_episode_advantages(rollouts, p2, device)
        ppo_metrics = ppo_update_step(
            actor, rollouts, advs, rets, bc_iter, p2,
            optimizer, update_step, device,
        )

        # ── Logging ────────────────────────────────────────
        ep_metrics = {
            "task_success_rate": np.mean([ep.task_success for ep in rollouts]),
            "mean_reward": float(np.mean([
                c.chunk_reward for ep in rollouts for c in ep.chunks
            ]) if any(ep.chunks for ep in rollouts) else 0.0),
            "ood_skip_rate": float(1 - np.mean([
                c.valid for ep in rollouts for c in ep.chunks
            ]) if any(ep.chunks for ep in rollouts) else 0.0),
            "mean_nn_dist": float(np.mean([
                c.nn_distance for ep in rollouts for c in ep.chunks
            ]) if any(ep.chunks for ep in rollouts) else 0.0),
            "n_episodes": len(rollouts),
            "n_aborted": sum(ep.aborted for ep in rollouts),
        }

        # Drift detection: L_BC vs C1 baseline
        L_BC = ppo_metrics.get("L_BC", 0.0)
        drift_ratio = L_BC / max(p2.c1_l1_baseline, 1e-6)
        if drift_ratio > p2.drift_alert_ratio:
            logger.error(f"⚠⚠⚠ ALERT: L_BC drift {drift_ratio:.2f}× C1 baseline (>{p2.drift_alert_ratio})")
        elif drift_ratio > p2.drift_warn_ratio:
            logger.warning(f"⚠ WARN: L_BC drift {drift_ratio:.2f}× C1 baseline (>{p2.drift_warn_ratio})")

        all_log = {
            **ep_metrics, **ppo_metrics,
            **{f"safety/{k}": v for k, v in safety_log.snapshot().items()},
            "L_BC_ratio_to_C1": drift_ratio,
            "update_step": update_step,
        }
        logger.info("metrics: " + json.dumps({k: f"{v:.3f}" if isinstance(v, float) else v
                                                for k, v in all_log.items()}, indent=2))

        if wandb_logger:
            try:
                # Convert numpy types to native Python (wandb older versions choke on np.bool_, np.int64).
                clean_log = {}
                for k, v in all_log.items():
                    if hasattr(v, "item"):  # numpy scalar
                        v = v.item()
                    clean_log[k] = v
                # Offset main-loop step past the warmup steps to keep wandb monotonic.
                wandb_step = p2.warmup_gpu_steps + 1 + update_step
                wandb_logger.log(clean_log, step=wandb_step)
                logger.info(f"  wandb logged {len(clean_log)} keys at step {wandb_step}")
            except Exception as e:
                logger.error(f"  wandb log FAILED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()

        # ── Eval gate ──────────────────────────────────────
        if (update_step + 1) % p2.eval_freq == 0 and update_step > 0:
            logger.info(f"== Eval gate @ update {update_step+1} ==")
            eval_successes = 0
            for k in range(p2.eval_n_episodes):
                eval_ep = rollout_episode(
                    robot, actor, history_buffer, reward_calculator, p2, update_step,
                    action_min_raw=action_min, action_max_raw=action_max,
                    safety_log=safety_log, deterministic=True,
                )
                eval_successes += int(eval_ep.task_success)
            success_rate = eval_successes / p2.eval_n_episodes
            logger.info(f"  Eval result: {eval_successes}/{p2.eval_n_episodes}")
            if eval_successes <= p2.eval_terminate_threshold:
                logger.error("Task severely degraded → TERMINATE")
                break
            elif eval_successes <= p2.eval_rollback_threshold:
                logger.warning("Task degrading → ROLLBACK + reduce α/σ (manual)")
            elif eval_successes < p2.eval_warn_threshold:
                logger.warning("Task starting to degrade → WARN")

        # ── Checkpoint ─────────────────────────────────────
        if (update_step + 1) % p2.save_freq == 0:
            ckpt_path = ckpt_dir / f"step_{update_step+1:04d}"
            ckpt_path.mkdir(exist_ok=True)
            actor.policy.save_pretrained(ckpt_path / "pretrained_model")
            torch.save({"optimizer": optimizer.state_dict(),
                          "update_step": update_step,
                          "phase2_cfg": p2.__dict__},
                         ckpt_path / "training_state.pt")
            logger.info(f"  saved ckpt → {ckpt_path}")

    # Final
    final_dir = ckpt_dir / "final"
    final_dir.mkdir(exist_ok=True)
    actor.policy.save_pretrained(final_dir / "pretrained_model")
    logger.info(f"Phase 2 training complete. Final ckpt → {final_dir}")
    if robot.is_connected:
        robot.disconnect()


if __name__ == "__main__":
    main()
