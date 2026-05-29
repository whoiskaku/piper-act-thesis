#!/usr/bin/env python3
"""Stage 3b — Train RefineNet on Phase 1 sim paired data.

Per RESEARCH_PLAN_v4.3 §5 Stage 3b and decisions Q6-Q12:
  - Output: raw rad (residual on top of ACT_output)
  - current_state input: z-scored externally (stats from train split)
  - jerk loss in raw rad space (matches demo jerk RMS baseline 113.7)
  - AdamW, lr=3e-4, batch=64, epochs=200
  - λ_smooth scan: {0.01, 0.1, 1.0}; default 0.1
  - val: loss + jerk metrics; sim deploy smoothness optional add-on

Usage:
    python sim/train_refine_net.py --lambda-smooth 0.1 --epochs 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lerobot.common.policies.act.refine_net import RefineNet
from sim.refine_net_dataset import RefinePhase1Dataset


H5_PATH = REPO_ROOT / "outputs/sim/phase1_paired_data.h5"
FPS = 30.0
DT = 1.0 / FPS


# ─── Losses ────────────────────────────────────────────────────────
def jerk_rms(actions: torch.Tensor, dt: float = DT) -> torch.Tensor:
    """Per-batch jerk RMS over time. actions shape (B, T, A)."""
    v = (actions[:, 1:] - actions[:, :-1]) / dt
    a = (v[:, 1:] - v[:, :-1]) / dt
    j = (a[:, 1:] - a[:, :-1]) / dt
    return (j ** 2).mean().sqrt()


def hf_energy_fraction(actions: torch.Tensor, fs: float = FPS,
                        cutoff: float = 5.0) -> torch.Tensor:
    """Per-batch fraction of action power above cutoff Hz."""
    B, T, A = actions.shape
    x_ac = actions - actions.mean(dim=1, keepdim=True)
    w = torch.hann_window(T, device=actions.device).view(1, T, 1)
    X = torch.fft.rfft(x_ac * w, dim=1)
    P = (X.abs() ** 2)  # (B, F, A)
    freqs = torch.fft.rfftfreq(T, d=1.0 / fs).to(actions.device)
    hf_mask = (freqs > cutoff).view(1, -1, 1)
    return (P * hf_mask).sum(dim=(1, 2)) / (P.sum(dim=(1, 2)) + 1e-12)


# ─── Train / Eval loops ────────────────────────────────────────────
def run_epoch(model, loader, optimizer, lambda_smooth, device, train=True):
    model.train(train)
    total_bc = 0.0
    total_smooth = 0.0
    total_loss = 0.0
    total_jerk_refined = 0.0
    total_jerk_act = 0.0
    n_samples = 0

    grad_context = torch.enable_grad() if train else torch.no_grad()
    with grad_context:
        for batch in loader:
            B = batch["ACT_output"].shape[0]
            act_output       = batch["ACT_output"].to(device, non_blocking=True)
            history_action   = batch["history_action"].to(device, non_blocking=True)
            history_imu      = batch["history_imu"].to(device, non_blocking=True)
            history_pad_mask = batch["history_pad_mask"].to(device, non_blocking=True)
            current_state    = batch["current_state"].to(device, non_blocking=True)
            A_clean          = batch["A_clean"].to(device, non_blocking=True)

            refined, delta = model(
                act_output, history_action, history_imu, current_state,
                history_pad_mask=history_pad_mask,
            )

            l_bc = nn.functional.l1_loss(refined, A_clean)
            l_smooth = jerk_rms(refined)
            loss = l_bc + lambda_smooth * l_smooth

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_bc += l_bc.item() * B
            total_smooth += l_smooth.item() * B
            total_loss += loss.item() * B
            total_jerk_refined += jerk_rms(refined).item() * B
            total_jerk_act += jerk_rms(act_output).item() * B
            n_samples += B

    return {
        "l_bc": total_bc / n_samples,
        "l_smooth": total_smooth / n_samples,
        "loss": total_loss / n_samples,
        "jerk_refined": total_jerk_refined / n_samples,
        "jerk_act_baseline": total_jerk_act / n_samples,
    }


# ─── Main ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-smooth", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--h5", type=Path, default=H5_PATH)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "outputs/train/refine_net")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = args.out_dir / f"phase1_lam{args.lambda_smooth}_lr{args.lr}_bs{args.batch_size}_ep{args.epochs}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[device] {device}")
    print(f"[out_dir] {out_dir}")

    # ---- Data ----
    print(f"[load] {args.h5}")
    train_ds = RefinePhase1Dataset(args.h5, "train")
    state_mean, state_std = train_ds.compute_state_stats()
    print(f"  state_mean: {state_mean}")
    print(f"  state_std:  {state_std}")
    train_ds.state_mean = state_mean
    train_ds.state_std = state_std
    val_ds = RefinePhase1Dataset(args.h5, "val", state_mean, state_std)
    print(f"  train n = {len(train_ds)}, val n = {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, args.num_workers // 2), pin_memory=True,
    )

    # ---- Model ----
    model = RefineNet().to(device)
    n_params = model.num_parameters()
    print(f"[model] RefineNet n_params = {n_params:,} ({n_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # Cosine LR schedule with linear warmup
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Save run config ----
    cfg_dump = {
        "lambda_smooth": args.lambda_smooth, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": args.weight_decay, "seed": args.seed,
        "state_mean": state_mean.tolist(), "state_std": state_std.tolist(),
        "n_train": len(train_ds), "n_val": len(val_ds),
        "n_params": n_params,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2))

    # ---- Training loop ----
    history = []
    best_val_loss = float("inf")
    t_start = time.time()

    for epoch in range(args.epochs):
        t_ep = time.time()
        train_stats = run_epoch(model, train_loader, optimizer,
                                  args.lambda_smooth, device, train=True)
        val_stats = run_epoch(model, val_loader, None,
                                args.lambda_smooth, device, train=False)
        scheduler.step()

        elapsed_ep = time.time() - t_ep
        elapsed_total = time.time() - t_start
        cur_lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch,
            "lr": cur_lr,
            "train": train_stats,
            "val": val_stats,
            "time_s": elapsed_ep,
        })

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"[ep {epoch:3d}/{args.epochs}] "
                  f"lr={cur_lr:.2e}  "
                  f"train L_BC={train_stats['l_bc']:.4f} "
                  f"L_smooth={train_stats['l_smooth']:.2f}  "
                  f"val L_BC={val_stats['l_bc']:.4f} "
                  f"L_smooth={val_stats['l_smooth']:.2f}  "
                  f"jerk_refined={val_stats['jerk_refined']:.1f} "
                  f"(act_baseline={val_stats['jerk_act_baseline']:.1f})  "
                  f"[{elapsed_ep:.1f}s, total {elapsed_total/60:.1f}m]")

        # Save best ckpt
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "state_mean": state_mean, "state_std": state_std,
                "args": vars(args), "val_stats": val_stats,
            }, out_dir / "ckpt_best.pt")

        # Periodic save
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "state_mean": state_mean, "state_std": state_std,
                "args": vars(args), "val_stats": val_stats,
            }, out_dir / f"ckpt_ep{epoch:04d}.pt")

    # Final save
    torch.save({
        "epoch": args.epochs - 1, "model_state": model.state_dict(),
        "state_mean": state_mean, "state_std": state_std,
        "args": vars(args),
    }, out_dir / "ckpt_final.pt")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\n[done] total time {(time.time() - t_start) / 60:.1f}m")
    print(f"[done] best val_loss = {best_val_loss:.4f}")
    print(f"[done] outputs in {out_dir}")


if __name__ == "__main__":
    main()
