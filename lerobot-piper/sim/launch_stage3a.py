#!/usr/bin/env python3
"""Stage 3a launcher — fine-tune C1-v3 ACT with the hf smoothness penalty.

CLI override 无法注入 saved config.json 里不存在的新字段(enable_denoise_smooth)，
故走 parser.wrap 的 `train(cfg)` 直传路径：draccus 解析 C1 的 train_config.json →
在对象上设 fine-tune 覆盖 → 直接调 train()。

负消融：解冻 ACT，loss = L1 + kld + λ_hf·hf_fraction(actions_hat, j1/j2/j3/grip)。
与 Stage 3b 同一 hf 指标，结果直接可比。

用法:  python sim/launch_stage3a.py [--lambda-hf 0.1] [--steps 20000]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import draccus

# 诊断用：STAGE3A_WATCHDOG=<秒> → 该秒后 dump 主线程 Python 栈并退出（定位挂起点）。
if os.environ.get("STAGE3A_WATCHDOG"):
    import faulthandler
    faulthandler.dump_traceback_later(int(os.environ["STAGE3A_WATCHDOG"]), exit=True)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.train import train

CKPT = REPO / "outputs/train/2026-05-08/C1_v3_history_seed42/checkpoints/070000/pretrained_model"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-hf", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--save-freq", type=int, default=5000)
    ap.add_argument("--log-freq", type=int, default=200)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    cfg = draccus.parse(TrainPipelineConfig, str(CKPT / "train_config.json"), args=[])

    # fine-tune overrides
    cfg.policy.pretrained_path = str(CKPT)          # make_policy loads C1-v3 weights
    cfg.policy.enable_denoise_smooth = True
    cfg.policy.denoise_lambda_hf = args.lambda_hf
    if getattr(cfg.policy, "enable_value_head", False):
        cfg.policy.enable_value_head = False        # clean BC ACT, no critic

    cfg.steps = args.steps
    cfg.save_freq = args.save_freq
    cfg.log_freq = args.log_freq
    cfg.batch_size = 16
    cfg.num_workers = args.num_workers   # 0 避免 CUDA-init 后 fork dataloader worker 死锁
    cfg.use_amp = False                  # diag 证明无 AMP 路径稳定；train 循环 AMP 触发 native crash
    cfg.resume = False
    cfg.output_dir = REPO / f"outputs/train/C2_denoising_act/lam{args.lambda_hf}"
    cfg.job_name = f"C2_denoise_lam{args.lambda_hf}"
    cfg.wandb.enable = not args.no_wandb

    print(f"[stage3a] λ_hf={args.lambda_hf} steps={args.steps} "
          f"out={cfg.output_dir} wandb={cfg.wandb.enable}")
    print(f"[stage3a] enable_denoise_smooth={cfg.policy.enable_denoise_smooth} "
          f"pretrained={cfg.policy.pretrained_path}")
    train(cfg)


if __name__ == "__main__":
    main()
