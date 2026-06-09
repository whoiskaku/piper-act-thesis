#!/usr/bin/env python3
"""#63 — Precompute ACT encoder 隐状态摘要到 H5(供 S0+H 变体用)。

对每个 paired 样本对应的真机帧, 在 frozen ACT 上挂 forward hook 捕获主 encoder 输出
encoder_out (S, B, 512), 对 seq 维 mean-pool 得 (B, 512) 摘要, 追加写入
phase1_paired_data.h5 的 train/val 组, 键名 `act_hidden_enc (N, 512)`。

ACT 权重不动(冻结), 只读 encoder 中间输出。复用 precompute_act_output 的加载/批处理框架,
仅追加新键, 不动 ACT_output / A_clean 等现有数据。

用法:  python sim/precompute_act_hidden.py
"""
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lerobot.common.datasets.factory import make_dataset
from sim.precompute_act_output import (
    load_frozen_act, gather_batch, build_ep_frame_to_global,
    C1_CKPT, DATASET_REPO, H5_PATH, BATCH_SIZE, DEVICE,
)


def main():
    print(f"[device] {DEVICE}")
    print(f"[load] policy from {C1_CKPT}")
    policy, cfg = load_frozen_act(C1_CKPT, DATASET_REPO, DEVICE)
    print(f"[load] dataset {DATASET_REPO}")
    dataset = make_dataset(cfg)
    ep_from, ep_to = build_ep_frame_to_global(dataset)

    # ---- hook 主 encoder 输出 (S, B, 512) ----
    captured = {}

    def hook(_m, _i, out):
        captured["enc"] = out      # ACTEncoder forward 返回 (S, B, dim_model)

    handle = policy.model.encoder.register_forward_hook(hook)

    enc_dim = cfg.policy.dim_model
    print(f"[h5] open {H5_PATH}  (encoder dim={enc_dim})")
    with h5py.File(H5_PATH, "a") as f:
        for split in ("train", "val"):
            grp = f[split]
            n = grp["A_clean"].shape[0]
            src_ep = grp["src_ep"][:]
            src_start = grp["src_start"][:]
            global_idxs = ep_from[src_ep] + src_start
            print(f"\n[{split}] N = {n}")

            if "act_hidden_enc" in grp:
                del grp["act_hidden_enc"]
            out_ds = grp.create_dataset(
                "act_hidden_enc", shape=(n, enc_dim), dtype="float32",
                compression="gzip", compression_opts=4,
            )

            t0 = time.time()
            for b_start in range(0, n, BATCH_SIZE):
                b_end = min(b_start + BATCH_SIZE, n)
                samples = [dataset[int(global_idxs[i])] for i in range(b_start, b_end)]
                batch = gather_batch(samples)
                with torch.no_grad():
                    batch_in = policy.normalize_inputs(batch)
                    if policy.config.image_features:
                        batch_in = dict(batch_in)
                        batch_in["observation.images"] = torch.stack(
                            [batch_in[k] for k in policy.config.image_features], dim=-4)
                    batch_in.pop("action", None)
                    batch_in.pop("action_is_pad", None)
                    policy.model(batch_in)                 # 触发 hook
                    enc = captured["enc"]                  # (S, B, 512)
                    summary = enc.mean(dim=0)              # (B, 512)
                out_ds[b_start:b_end] = summary.detach().cpu().numpy().astype(np.float32)
                if b_start // BATCH_SIZE % 5 == 0:
                    el = time.time() - t0
                    print(f"  [{split}] {b_end}/{n}  ({b_end/max(el,1e-9):.1f}/s)")
            print(f"  [{split}] done in {time.time()-t0:.1f}s  "
                  f"mean {out_ds[:].mean():+.4f} std {out_ds[:].std():.4f} "
                  f"finite={np.isfinite(out_ds[:]).all()}")

    handle.remove()
    print(f"\n[done] act_hidden_enc 写入 {H5_PATH}")


if __name__ == "__main__":
    main()
