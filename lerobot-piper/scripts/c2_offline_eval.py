#!/usr/bin/env python3
"""C2 offline eval: load ckpt, run forward on v2 dataset, capture l1_loss + smoothness.

Uses the same factory + dataloader path as train.py to avoid batch-shape mismatches.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.factory import make_policy
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.policies import PreTrainedConfig


def jerk_rms(action_chunk_b, dt=1.0 / 30.0):
    """RMS jerk over (B, T, A) batch, only T>=4 valid."""
    if action_chunk_b.shape[1] < 4:
        return 0.0
    v = np.diff(action_chunk_b, axis=1) / dt
    a = np.diff(v, axis=1) / dt
    j = np.diff(a, axis=1) / dt
    return float(np.sqrt((j ** 2).mean()))


def hf_pct(action_chunk_b, fs=30.0, cutoff=5.0):
    """Mean over batch of: % energy above cutoff Hz."""
    if action_chunk_b.shape[1] < 4:
        return 0.0
    fs_pct = []
    for b in range(action_chunk_b.shape[0]):
        x = action_chunk_b[b] - action_chunk_b[b].mean(axis=0, keepdims=True)
        N = x.shape[0]
        w = np.hanning(N).reshape(-1, 1)
        X = np.fft.rfft(x * w, axis=0)
        p = np.abs(X) ** 2
        f = np.fft.rfftfreq(N, d=1.0 / fs)
        fs_pct.append(p[f > cutoff].sum() / (p.sum() + 1e-30) * 100)
    return float(np.mean(fs_pct))


def eval_ckpt(ckpt_dir, n_batches=20, batch_size=8, device="cuda"):
    """Reconstruct policy + dataset from ckpt's saved configs, run forward."""
    ckpt_dir = Path(ckpt_dir)
    train_cfg_path = ckpt_dir / "pretrained_model" / "train_config.json"
    train_cfg = json.load(open(train_cfg_path))

    print(f"  ckpt: {ckpt_dir.parent.parent.name} / {ckpt_dir.name}")
    print(f"  enable_history_input = {train_cfg['policy'].get('enable_history_input', False)}")

    # Decode the saved config dict directly (skip CLI parsing)
    import tempfile, draccus
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(train_cfg, f)
        tmp_path = f.name
    cfg = draccus.parse(TrainPipelineConfig, tmp_path, args=[])
    # Override policy.pretrained_path so make_policy loads weights
    cfg.policy.pretrained_path = str(ckpt_dir / "pretrained_model")

    dataset = make_dataset(cfg)
    base_meta = dataset.base.meta if hasattr(dataset, "base") else dataset.meta
    policy = make_policy(cfg=cfg.policy, ds_meta=base_meta, device=device)
    policy.eval()

    # Resolve action mean/std for unnormalization
    a_mean = torch.as_tensor(base_meta.stats["action"]["mean"], dtype=torch.float32, device=device)
    a_std = torch.as_tensor(base_meta.stats["action"]["std"], dtype=torch.float32, device=device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    metrics = {"l1_norm": [], "l1_unnorm_deg": [], "jerk_unnorm": [], "hf_pct": []}

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            # Replicate ACTPolicy.forward up through model() to capture normalized predictions
            batch = policy.normalize_inputs(batch)
            if policy.config.image_features:
                batch = dict(batch)
                batch["observation.images"] = torch.stack(
                    [batch[k] for k in policy.config.image_features], dim=-4
                )
            batch = policy.normalize_targets(batch)
            actions_hat, _, _ = policy.model(batch)

            # L1 in normalized space
            mask = ~batch["action_is_pad"].unsqueeze(-1)
            l1n = (F.l1_loss(batch["action"], actions_hat, reduction="none") * mask).mean().item()
            metrics["l1_norm"].append(l1n)

            # Unnormalize for smoothness metrics
            unnorm = actions_hat * a_std + a_mean
            unnorm_np = unnorm.cpu().numpy()
            metrics["jerk_unnorm"].append(jerk_rms(unnorm_np))
            metrics["hf_pct"].append(hf_pct(unnorm_np))

            # L1 in degrees (per-joint mean)
            gt_unnorm = batch["action"] * a_std + a_mean
            l1_rad = (gt_unnorm - unnorm).abs().mean().item()
            metrics["l1_unnorm_deg"].append(np.rad2deg(l1_rad))

    return {k: float(np.mean(v)) for k, v in metrics.items()}, len(metrics["l1_norm"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1", default="/home/enolab-guo/claude-test-2/outputs/train/2026-04-29/C1_v2_history_seed42/checkpoints/100000")
    ap.add_argument("--vanilla", default="/home/enolab-guo/claude-test-2/lerobot-piper/outputs/train/2026-04-27/17-44-36_p1v2_vanilla/checkpoints/050000")
    ap.add_argument("--n-batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    print("=" * 72)
    print("C2 OFFLINE EVAL: history-aug ACT vs vanilla")
    print(f"  n_batches={args.n_batches}, batch_size={args.batch_size}, ~{args.n_batches*args.batch_size} chunks each")
    print("=" * 72)

    print("\n[Vanilla v2 baseline]")
    vn_metrics, n_vn = eval_ckpt(args.vanilla, args.n_batches, args.batch_size)

    print("\n[C1: history-augmented]")
    c1_metrics, n_c1 = eval_ckpt(args.c1, args.n_batches, args.batch_size)

    print()
    print("=" * 72)
    print(f"{'metric':<22s} {'Vanilla v2':>14s} {'C1 (history)':>14s} {'Δ (rel %)':>14s}")
    print("-" * 72)
    for k in ["l1_norm", "l1_unnorm_deg", "jerk_unnorm", "hf_pct"]:
        vv = vn_metrics[k]; vc = c1_metrics[k]
        rel = (vc - vv) / abs(vv) * 100 if vv != 0 else 0.0
        sign = "↓" if rel < 0 else "↑"
        print(f"{k:<22s} {vv:>14.5f} {vc:>14.5f} {sign}{abs(rel):>11.1f}%")
    print("=" * 72)
    print()
    print("legend:  l1_norm = action L1 normalized space (lower=better fit to GT)")
    print("         l1_unnorm_deg = ditto in joint-degree units")
    print("         jerk_unnorm = RMS d³action/dt³ (lower = smoother)")
    print("         hf_pct = % action energy >5Hz (lower = smoother)")
    print()
    print("[Reminder] Both ckpts trained on v2 → in-sample eval.")
    print("           Real validation = C3 real-robot deployment.")

    # Save JSON
    out_path = Path("/home/enolab-guo/claude-test-2/lerobot-piper/outputs/eval/c2_history_vs_vanilla.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"vanilla_v2": vn_metrics, "c1_history": c1_metrics,
               "n_vn_batches": n_vn, "n_c1_batches": n_c1}, open(out_path, "w"), indent=2)
    print(f"\nresults saved -> {out_path}")


if __name__ == "__main__":
    main()
