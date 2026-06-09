#!/usr/bin/env python3
"""Stage 3b — Train RefineNet on Phase 1 sim paired data.

Per RESEARCH_PLAN_v4.3 §5 Stage 3b + Run #1 diagnosis (see STAGE3B_RESULTS.md):
  - Output: raw rad (residual ΔA on top of frozen ACT_output)
  - current_state input: z-scored externally (stats from train split)
  - HEADLINE metric: active-joint (j1/j2/j3/grip) mean hf_fraction.
    jerk_RMS is kept ONLY as a secondary/continuity metric (proven unreliable).
  - Loss:  L_BC + λ_hf·L_hf_fraction + λ_delta·|ΔA| (+ optional λ_acc·L_acc)
       L_hf_fraction uses a stopgrad denominator (no low-freq-inflation cheat).
  - AdamW, lr=3e-4, batch=64, epochs=200

Run plan:
  Run A  L_BC only:                 --lambda-hf 0 --lambda-delta 0 --lambda-acc 0
  Run B  main path:                 --lambda-hf 0.05 --lambda-delta 0.2
  (λ ranges from sim/analyze_loss_scales.py: λ_hf∈[0.024,0.072], λ_delta∈[0.10,0.30])

Usage:
    python sim/train_refine_net.py --lambda-hf 0 --lambda-delta 0 --epochs 200   # Run A
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
HF_CUTOFF = 5.0          # Hz; energy above this = "jitter" we want to remove
EPS = 1e-8

# Active joints in the v3 pick-place task (j4/j5/j6 are stationary → excluded
# from the headline hf metric so their 0-energy doesn't dilute the mean).
ACTIVE_JOINTS = [0, 1, 2, 6]   # j1, j2, j3, grip(j7)


# ─── Spectral helpers ──────────────────────────────────────────────
def hf_power(actions: torch.Tensor, fs: float = FPS, cutoff: float = HF_CUTOFF):
    """Per-(B, joint) high-freq power and total AC power. actions: (B, T, A).

    Returns (E_high, E_total), each shaped (B, A). DC removed, Hann-windowed.
    """
    B, T, A = actions.shape
    x_ac = actions - actions.mean(dim=1, keepdim=True)
    w = torch.hann_window(T, device=actions.device).view(1, T, 1)
    X = torch.fft.rfft(x_ac * w, dim=1)
    P = X.abs() ** 2                                       # (B, F, A)
    freqs = torch.fft.rfftfreq(T, d=1.0 / fs).to(actions.device)
    hf_mask = (freqs > cutoff).view(1, -1, 1)
    E_high = (P * hf_mask).sum(dim=1)                      # (B, A)
    E_total = P.sum(dim=1)                                 # (B, A)
    return E_high, E_total


def hf_fraction_per_joint(actions: torch.Tensor) -> torch.Tensor:
    """Per-(B, joint) hf-energy fraction in [0,1] (no stopgrad — eval metric)."""
    E_high, E_total = hf_power(actions)
    return E_high / (E_total + EPS)                        # (B, A)


def hf_fraction_loss(actions: torch.Tensor, active=ACTIVE_JOINTS) -> torch.Tensor:
    """Smoothness loss = mean active-joint hf fraction with stopgrad denominator.

    Detaching E_total stops the model lowering the *fraction* by inflating
    low-freq energy; gradient flows only through the high-freq numerator.
    """
    E_high, E_total = hf_power(actions)
    frac = E_high / (E_total.detach() + EPS)               # (B, A)
    return frac[:, active].mean()


# ─── Time-domain regularizers ──────────────────────────────────────
def l_delta_term(delta: torch.Tensor) -> torch.Tensor:
    """Residual-shrinkage / trust-region: mean |ΔA|."""
    return delta.abs().mean()


def l_acc_term(actions: torch.Tensor) -> torch.Tensor:
    """Mean |second difference| (no /dt²). Optional, weak discriminator."""
    d2 = actions[:, 2:] - 2 * actions[:, 1:-1] + actions[:, :-2]
    return d2.abs().mean()


# ─── Secondary / continuity metric ─────────────────────────────────
def jerk_rms(actions: torch.Tensor, dt: float = DT) -> torch.Tensor:
    """Per-batch jerk RMS — SECONDARY metric only, NOT in the loss.

    Shown in Run #1 diagnosis to be dominated by bulk-motion amplitude rather
    than the high-freq jitter we care about. See STAGE3B_RESULTS.md.
    """
    v = (actions[:, 1:] - actions[:, :-1]) / dt
    a = (v[:, 1:] - v[:, :-1]) / dt
    j = (a[:, 1:] - a[:, :-1]) / dt
    return (j ** 2).mean().sqrt()


# ─── Train / Eval loops ────────────────────────────────────────────
def run_epoch(model, loader, optimizer, lambdas, device, train=True, input_key="ACT_output",
              zero_act_img=False):
    """lambdas = dict(hf=..., delta=..., acc=...)."""
    model.train(train)
    n = 0
    acc = dict(  # running sums (× batch size)
        l_bc=0.0, l_hf=0.0, l_delta=0.0, l_acc=0.0, loss=0.0,
        jerk_refined=0.0, jerk_act=0.0,
        l_bc_act=0.0,            # guardrail ref: |ACT_output - A_clean|
    )
    # per-active-joint hf fraction sums for refined / act / clean
    hf_ref = np.zeros(len(ACTIVE_JOINTS))
    hf_act = np.zeros(len(ACTIVE_JOINTS))
    hf_cln = np.zeros(len(ACTIVE_JOINTS))

    grad_context = torch.enable_grad() if train else torch.no_grad()
    with grad_context:
        for batch in loader:
            B = batch["ACT_output"].shape[0]
            act_output       = batch[input_key].to(device, non_blocking=True)  # 待精修输入(ACT_output 或 A_noisy, #62)
            history_action   = batch["history_action"].to(device, non_blocking=True)
            history_imu      = batch["history_imu"].to(device, non_blocking=True)
            history_pad_mask = batch["history_pad_mask"].to(device, non_blocking=True)
            current_state    = batch["current_state"].to(device, non_blocking=True)
            A_clean          = batch["A_clean"].to(device, non_blocking=True)
            act_hidden = (batch["act_hidden"].to(device, non_blocking=True)
                          if "act_hidden" in batch else None)
            act_img = (batch["act_img"].to(device, non_blocking=True)
                       if "act_img" in batch else None)
            if zero_act_img and act_img is not None:
                act_img = torch.zeros_like(act_img)        # 零-image 对照: 同架构无真实感知信息

            refined, delta = model(
                act_output, history_action, history_imu, current_state,
                history_pad_mask=history_pad_mask, act_hidden=act_hidden, act_img=act_img,
            )

            l_bc = nn.functional.l1_loss(refined, A_clean)
            l_hf = hf_fraction_loss(refined)
            l_dl = l_delta_term(delta)
            l_ac = l_acc_term(refined)
            loss = (l_bc
                    + lambdas["hf"] * l_hf
                    + lambdas["delta"] * l_dl
                    + lambdas["acc"] * l_ac)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            acc["l_bc"]    += l_bc.item() * B
            acc["l_hf"]    += l_hf.item() * B
            acc["l_delta"] += l_dl.item() * B
            acc["l_acc"]   += l_ac.item() * B
            acc["loss"]    += loss.item() * B
            acc["jerk_refined"] += jerk_rms(refined).item() * B
            acc["jerk_act"]     += jerk_rms(act_output).item() * B
            acc["l_bc_act"]     += nn.functional.l1_loss(act_output, A_clean).item() * B

            with torch.no_grad():
                fr = hf_fraction_per_joint(refined).mean(dim=0).cpu().numpy()
                fa = hf_fraction_per_joint(act_output).mean(dim=0).cpu().numpy()
                fc = hf_fraction_per_joint(A_clean).mean(dim=0).cpu().numpy()
            hf_ref += fr[ACTIVE_JOINTS] * B
            hf_act += fa[ACTIVE_JOINTS] * B
            hf_cln += fc[ACTIVE_JOINTS] * B
            n += B

    out = {k: v / n for k, v in acc.items()}
    out["hf_refined_joints"] = (hf_ref / n).tolist()
    out["hf_act_joints"]     = (hf_act / n).tolist()
    out["hf_clean_joints"]   = (hf_cln / n).tolist()
    out["hf_refined"] = float((hf_ref / n).mean())   # headline metric
    out["hf_act"]     = float((hf_act / n).mean())
    out["hf_clean"]   = float((hf_cln / n).mean())
    out["hf_ratio"]   = out["hf_refined"] / (out["hf_act"] + EPS)  # primary criterion
    return out


def grade(ratio: float) -> str:
    if ratio < 0.1:  return "EXCELLENT"
    if ratio < 0.25: return "good"
    if ratio < 0.5:  return "pass"
    return "FAIL"


# ─── Main ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-hf", type=float, default=0.0,
                    help="weight on active-joint hf_fraction loss (main smoothness term)")
    ap.add_argument("--lambda-delta", type=float, default=0.0,
                    help="weight on |ΔA| residual-shrinkage / trust-region")
    ap.add_argument("--lambda-acc", type=float, default=0.0,
                    help="weight on |2nd-diff| (optional, weak; default off)")
    ap.add_argument("--tag", type=str, default=None,
                    help="run name suffix; auto-built from λ if omitted")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--variant", choices=["S0", "S1", "S2", "S3", "full"],
                    default="full", help="输入消融: S0=仅ACT, S1=+state, S2=+state+hist_action, S3/full=全部")
    ap.add_argument("--input-signal", choices=["act_output", "a_noisy"], default="act_output",
                    help="#62 待精修输入: act_output(现状) 或 a_noisy(去噪对照)")
    ap.add_argument("--use-act-hidden", action="store_true",
                    help="#63 加 ACT encoder 摘要作 context token(变体 S0+H)")
    ap.add_argument("--use-act-img-attn", action="store_true",
                    help="#63(a) 对 ACT image tokens 做注意力读出作 context token(变体 S0+Hattn)")
    ap.add_argument("--zero-act-img", action="store_true",
                    help="#63(a) 对照: act_img 喂零(同架构无真实感知信息) → 隔离 信息 vs 参数容量")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--h5", type=Path, default=H5_PATH)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "outputs/train/refine_net")
    args = ap.parse_args()

    lambdas = {"hf": args.lambda_hf, "delta": args.lambda_delta, "acc": args.lambda_acc}

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 输入消融变体 → RefineNet 输入开关
    VARIANTS = {  # (use_state, use_history_action, use_history_imu)
        "S0": (False, False, False),
        "S1": (True,  False, False),
        "S2": (True,  True,  False),
        "S3": (True,  True,  True),
        "full": (True, True, True),
    }
    use_state, use_ha, use_hi = VARIANTS[args.variant]

    input_key = "A_noisy" if args.input_signal == "a_noisy" else "ACT_output"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _hsfx = ('+H' if args.use_act_hidden else '') + ('+Hattn' if args.use_act_img_attn else '')
    _hsfx += ('+zero' if args.zero_act_img else '')
    _ssfx = '' if args.seed == 42 else f'_s{args.seed}'   # seed42 保持原 tag(兼容已有目录)
    tag = args.tag or (
        f"{args.variant}{_hsfx}_in-{args.input_signal}_hf{args.lambda_hf}_dl{args.lambda_delta}{_ssfx}"
    )
    out_dir = args.out_dir / f"phase1_{tag}_lr{args.lr}_bs{args.batch_size}_ep{args.epochs}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[device] {device}")
    print(f"[out_dir] {out_dir}")
    print(f"[loss] L_BC + {args.lambda_hf}·hf + {args.lambda_delta}·|ΔA| + {args.lambda_acc}·acc")

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
    model = RefineNet(use_state=use_state, use_history_action=use_ha,
                      use_history_imu=use_hi, use_act_hidden=args.use_act_hidden,
                      use_act_img_attn=args.use_act_img_attn).to(device)
    print(f"[variant] {args.variant}{'+H' if args.use_act_hidden else ''}: "
          f"use_state={use_state} use_hist_action={use_ha} use_hist_imu={use_hi} use_act_hidden={args.use_act_hidden}")
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
        "lambda_hf": args.lambda_hf, "lambda_delta": args.lambda_delta,
        "lambda_acc": args.lambda_acc, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": args.weight_decay, "seed": args.seed,
        "active_joints": ACTIVE_JOINTS, "hf_cutoff": HF_CUTOFF,
        "state_mean": state_mean.tolist(), "state_std": state_std.tolist(),
        "n_train": len(train_ds), "n_val": len(val_ds), "n_params": n_params,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2))

    # ---- Training loop ----
    history = []
    # Model-selection criterion: lowest hf_ratio subject to BC guardrail.
    # (val loss is dominated by λ choices, so select on the headline metric.)
    best_hf_ratio = float("inf")
    t_start = time.time()

    for epoch in range(args.epochs):
        t_ep = time.time()
        train_stats = run_epoch(model, train_loader, optimizer, lambdas, device, train=True, input_key=input_key, zero_act_img=args.zero_act_img)
        val_stats = run_epoch(model, val_loader, None, lambdas, device, train=False, input_key=input_key, zero_act_img=args.zero_act_img)
        scheduler.step()

        elapsed_ep = time.time() - t_ep
        elapsed_total = time.time() - t_start
        cur_lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch, "lr": cur_lr,
            "train": train_stats, "val": val_stats, "time_s": elapsed_ep,
        })

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"[ep {epoch:3d}/{args.epochs}] lr={cur_lr:.2e}  "
                  f"L_BC={val_stats['l_bc']:.4f}(act {val_stats['l_bc_act']:.4f})  "
                  f"|ΔA|={val_stats['l_delta']:.4f}  "
                  f"hf_ref={val_stats['hf_refined']:.4f} hf_act={val_stats['hf_act']:.4f} "
                  f"hf_cln={val_stats['hf_clean']:.4f}  "
                  f"ratio={val_stats['hf_ratio']:.3f} [{grade(val_stats['hf_ratio'])}]  "
                  f"[{elapsed_ep:.1f}s, tot {elapsed_total/60:.1f}m]")

        # Save best ckpt by headline metric, with BC guardrail
        # (refined BC must not exceed the ACT-baseline BC by more than 10%).
        bc_ok = val_stats["l_bc"] <= 1.1 * val_stats["l_bc_act"]
        if bc_ok and val_stats["hf_ratio"] < best_hf_ratio:
            best_hf_ratio = val_stats["hf_ratio"]
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "state_mean": state_mean, "state_std": state_std,
                "args": vars(args), "val_stats": val_stats,
            }, out_dir / "ckpt_best.pt")

        if (epoch + 1) % args.save_every == 0:
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "state_mean": state_mean, "state_std": state_std,
                "args": vars(args), "val_stats": val_stats,
            }, out_dir / f"ckpt_ep{epoch:04d}.pt")

    torch.save({
        "epoch": args.epochs - 1, "model_state": model.state_dict(),
        "state_mean": state_mean, "state_std": state_std, "args": vars(args),
    }, out_dir / "ckpt_final.pt")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    final = history[-1]["val"]
    print(f"\n[done] total time {(time.time() - t_start) / 60:.1f}m")
    print(f"[done] best hf_ratio (BC-guarded) = {best_hf_ratio:.3f} [{grade(best_hf_ratio)}]")
    print(f"[done] final val: hf_ref={final['hf_refined']:.4f} hf_act={final['hf_act']:.4f} "
          f"ratio={final['hf_ratio']:.3f}  L_BC={final['l_bc']:.4f}(act {final['l_bc_act']:.4f})")
    print(f"[done] final per-joint hf refined {['%.4f'%x for x in final['hf_refined_joints']]} "
          f"vs act {['%.4f'%x for x in final['hf_act_joints']]} (j1,j2,j3,grip)")
    print(f"[done] outputs in {out_dir}")


if __name__ == "__main__":
    main()
