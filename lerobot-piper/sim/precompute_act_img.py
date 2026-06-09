#!/usr/bin/env python3
"""#63(a) — Precompute ACT encoder 的 image-feature tokens 到 H5(供 attention-readout 用)。

#63 的 mean-pool 摘要是对全部 1502 个 encoder token 平均,其中 1200 个是历史 token(#61 已证无用),
感知信息被严重稀释。本脚本只取 encoder 序列里的 **image 特征 token**(末尾 ~300),
供 RefineNet 用 learned query 做 RLT 式注意力读出。

encoder token 顺序(modeling_act):[latent, state, (env), (imu), history_action(L), history_imu(L), image(H*W)]
→ image tokens = encoder_out[offset:], offset = n_prefix + n_history。

存为 `act_img_tokens (N, n_img, 512)` float16(省空间)到 train/val 组。ACT 权重不动。
用法: python sim/precompute_act_img.py
"""
import sys, time
from pathlib import Path
import h5py, numpy as np, torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from lerobot.common.datasets.factory import make_dataset
from sim.precompute_act_output import (
    load_frozen_act, gather_batch, build_ep_frame_to_global,
    C1_CKPT, DATASET_REPO, H5_PATH, BATCH_SIZE, DEVICE)


def main():
    policy, cfg = load_frozen_act(C1_CKPT, DATASET_REPO, DEVICE)
    dataset = make_dataset(cfg)
    ep_from, _ = build_ep_frame_to_global(dataset)
    p = cfg.policy

    # encoder 前缀 + 历史 token 数 → image token 起始 offset
    n_prefix = 1
    n_prefix += int(getattr(p, "robot_state_feature", None) is not None)
    n_prefix += int(getattr(p, "env_state_feature", None) is not None)
    n_prefix += int(getattr(p, "enable_imu_input", False) and getattr(p, "imu_feature", None) is not None)
    hL = p.history_max_length
    n_hist = (hL if p.history_use_action else 0) + (hL if p.history_use_imu else 0)
    offset = n_prefix + n_hist
    print(f"[offset] n_prefix={n_prefix} n_hist={n_hist} → image tokens 从 {offset} 起")

    cap = {}
    h = policy.model.encoder.register_forward_hook(lambda m, i, o: cap.__setitem__("e", o))

    with h5py.File(H5_PATH, "a") as f:
        for split in ("train", "val"):
            grp = f[split]
            n = grp["A_clean"].shape[0]
            src_ep = grp["src_ep"][:]; src_start = grp["src_start"][:]
            gidx = ep_from[src_ep] + src_start
            n_img = None; out_ds = None
            t0 = time.time()
            for b in range(0, n, BATCH_SIZE):
                e = min(b + BATCH_SIZE, n)
                samples = [dataset[int(gidx[i])] for i in range(b, e)]
                batch = gather_batch(samples)
                with torch.no_grad():
                    bi = policy.normalize_inputs(batch)
                    if policy.config.image_features:
                        bi = dict(bi)
                        bi["observation.images"] = torch.stack(
                            [bi[k] for k in policy.config.image_features], dim=-4)
                    bi.pop("action", None); bi.pop("action_is_pad", None)
                    policy.model(bi)
                    img = cap["e"][offset:].permute(1, 0, 2)   # (B, n_img, 512)
                if out_ds is None:
                    n_img = img.shape[1]
                    print(f"[{split}] N={n}  n_img={n_img}  (S={cap['e'].shape[0]})")
                    if "act_img_tokens" in grp: del grp["act_img_tokens"]
                    out_ds = grp.create_dataset("act_img_tokens", shape=(n, n_img, 512),
                                                dtype="float16", compression="gzip", compression_opts=4)
                out_ds[b:e] = img.detach().cpu().numpy().astype(np.float16)
                if b // BATCH_SIZE % 5 == 0:
                    print(f"  [{split}] {e}/{n} ({e/max(time.time()-t0,1e-9):.1f}/s)")
            print(f"  [{split}] done {time.time()-t0:.1f}s finite={np.isfinite(out_ds[:5].astype(np.float32)).all()}")
    h.remove()
    print(f"[done] act_img_tokens 写入 {H5_PATH}")


if __name__ == "__main__":
    main()
