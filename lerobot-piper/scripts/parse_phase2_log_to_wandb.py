#!/usr/bin/env python3
"""Parse Phase 2 training terminal log and back-fill metrics to a new WandB run.

Used because phase2_train.py had a `cfg.warmup_gpu_steps` AttributeError that
silently swallowed the wandb.log() call for main-loop metrics. Terminal log
(via `tee`) captured all metrics dicts; this script replays them to wandb.

Usage:
    python scripts/parse_phase2_log_to_wandb.py \
        --log outputs/phase2_v3_terminal.log \
        --project lerobot-piper-phase2 \
        --name phase2_v3_backfill
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import wandb


# Match a multi-line "metrics: { ... }" block in the log.
METRICS_RE = re.compile(r"metrics:\s*(\{.*?\n\})", re.DOTALL)
# Match warmup V step lines.
WARMUP_RE = re.compile(r"warmup V step (\d+):\s*L_V=([\d.eE+-]+)")


def parse_metrics_blocks(log_text: str) -> list[dict]:
    """Extract every 'metrics: { ... }' JSON block."""
    blocks = []
    for m in METRICS_RE.finditer(log_text):
        raw = m.group(1)
        # Convert "0.020" string-quoted floats to actual floats by re-parsing
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            print(f"  skip malformed block: {raw[:80]}...")
            continue
        out = {}
        for k, v in d.items():
            if isinstance(v, str):
                try:
                    if v.endswith("%"):
                        v = float(v.rstrip("%"))
                    else:
                        v = float(v)
                except ValueError:
                    pass
            out[k] = v
        blocks.append(out)
    return blocks


def parse_warmup_blocks(log_text: str) -> list[tuple[int, float]]:
    """Extract (step, V_loss) pairs from critic warmup Stage B."""
    return [(int(s), float(lv)) for s, lv in WARMUP_RE.findall(log_text)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="path to tee-captured terminal log")
    ap.add_argument("--project", default="lerobot-piper-phase2")
    ap.add_argument("--name", default="phase2_v3_backfill")
    ap.add_argument("--warmup-gpu-steps", type=int, default=2000,
                    help="must match training cfg; main metrics get step=warmup+1+update_step")
    args = ap.parse_args()

    log_text = Path(args.log).read_text()
    main_blocks = parse_metrics_blocks(log_text)
    warmup_blocks = parse_warmup_blocks(log_text)

    print(f"Parsed {len(warmup_blocks)} warmup V steps, {len(main_blocks)} main metric blocks")

    if not main_blocks:
        print("no main metrics found, exiting")
        return

    wandb.init(project=args.project, name=args.name,
                config={"backfilled_from": args.log,
                          "n_main_blocks": len(main_blocks),
                          "n_warmup_blocks": len(warmup_blocks)})
    try:
        # Replay warmup Stage B
        for step, lv in warmup_blocks:
            wandb.log({"warmup/V_loss": lv, "warmup/step": step}, step=step)

        # Replay main loop
        for blk in main_blocks:
            update_step = int(blk.get("update_step", 0))
            wandb.log(blk, step=args.warmup_gpu_steps + 1 + update_step)

        print(f"Logged {len(warmup_blocks)} warmup + {len(main_blocks)} main blocks to wandb run '{args.name}'")
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
