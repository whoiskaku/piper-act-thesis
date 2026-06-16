#!/usr/bin/env python3
"""#69 RLT-AE — 在冻结 ACT image token 上重构预训练的压缩器(RLT 论文式)。

我们 #63a 的 token 提取 = 单 learned query 一层 cross-attention 读出(仅端到端
平滑梯度驱动)。RLT 论文的关键差异是 **encoder-decoder 压缩器 + 重构损失**:
压缩 token 必须能重建回源 embedding → 强制信息保留, 独立于稀疏的下游梯度
(论文的 anti-overfit 论点)。

本脚本在 **冻结的** act_img_tokens (N,300,512, 已预计算) 上离线训练一个自编码器:
  encoder: 300 image token --(M learned query cross-attn)--> K 压缩 token (RLT token)
  decoder: K token --(300 pos query cross-attn)--> 重建 300×512
  loss: MSE(recon, img)  (per-channel z-score 空间)
训练后用 encoder 把每个 chunk 的 300 token 压成 K 个, 写回 H5 的 act_rlt_tokens
(N,K,d), 供 RefineNet 的 use_act_rlt 分支当冻结上下文用("压缩后再用", 最忠实 RLT)。

用法: python sim/train_rlt_compressor.py --h5 outputs/sim/phase1_dense.h5 [--k 8 --d 256 --epochs 80]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RLTCompressor(nn.Module):
    """encoder(300→K) + decoder(K→300) 自编码器; encoder 输出即 RLT token。"""

    def __init__(self, img_dim=512, d=256, k=8, n_img=300, nhead=8):
        super().__init__()
        self.in_proj = nn.Linear(img_dim, d)
        self.enc_query = nn.Parameter(torch.randn(1, k, d) * 0.02)
        self.enc_attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.enc_norm = nn.LayerNorm(d)
        # decoder: 300 pos query cross-attend K tokens → reconstruct
        self.dec_pos = nn.Parameter(torch.randn(1, n_img, d) * 0.02)
        self.dec_attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.dec_norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, img_dim)
        self.k = k; self.d = d

    def encode(self, img):                      # img: (B,300,d_in normalized)
        x = self.in_proj(img)                   # (B,300,d)
        q = self.enc_query.expand(img.shape[0], -1, -1)
        z, _ = self.enc_attn(q, x, x)           # (B,K,d)
        return self.enc_norm(z)

    def forward(self, img):
        z = self.encode(img)                    # (B,K,d)
        q = self.dec_pos.expand(img.shape[0], -1, -1)
        h, _ = self.dec_attn(q, z, z)           # (B,300,d)
        recon = self.out_proj(self.dec_norm(h)) # (B,300,d_in)
        return recon, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=Path, default=REPO_ROOT / "outputs/sim/phase1_dense.h5")
    ap.add_argument("--k", type=int, default=8, help="RLT token 数")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    torch.manual_seed(42); np.random.seed(42)
    with h5py.File(args.h5, "r") as f:
        tr = torch.from_numpy(f["train"]["act_img_tokens"][:].astype(np.float32))  # (Ntr,300,512)
        va = torch.from_numpy(f["val"]["act_img_tokens"][:].astype(np.float32))
    print(f"[load] train {tuple(tr.shape)}  val {tuple(va.shape)}")
    # per-channel z-score(train)
    mean = tr.mean(dim=(0, 1)); std = tr.std(dim=(0, 1)).clamp_min(1e-6)
    trn = (tr - mean) / std; van = (va - mean) / std
    img_dim = tr.shape[-1]; n_img = tr.shape[1]

    model = RLTCompressor(img_dim=img_dim, d=args.d, k=args.k, n_img=n_img).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] RLTCompressor K={args.k} d={args.d} n_params={n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def run(X, train):
        model.train(train)
        idx = torch.randperm(len(X)) if train else torch.arange(len(X))
        tot = 0.0; n = 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for b in range(0, len(X), args.batch_size):
                xb = X[idx[b:b + args.batch_size]].to(DEVICE)
                recon, _ = model(xb)
                loss = nn.functional.mse_loss(recon, xb)
                if train:
                    opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(xb); n += len(xb)
        return tot / n

    # 归一化空间 MSE 基线 = 1.0(方差); explained var = 1 - mse
    t0 = time.time(); best = 1e9
    for ep in range(args.epochs):
        tl = run(trn, True); vl = run(van, False)
        best = min(best, vl)
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"[ep {ep:3d}] train_mse={tl:.4f} val_mse={vl:.4f} "
                  f"val_explained_var={1 - vl:.3f}  [{time.time()-t0:.0f}s]")

    # 预计算 RLT token 写回 H5
    model.eval()
    with h5py.File(args.h5, "a") as f:
        for split, Xn in (("train", trn), ("val", van)):
            zs = []
            with torch.no_grad():
                for b in range(0, len(Xn), args.batch_size):
                    zs.append(model.encode(Xn[b:b + args.batch_size].to(DEVICE)).cpu())
            z = torch.cat(zs).numpy().astype(np.float16)         # (N,K,d)
            grp = f[split]
            if "act_rlt_tokens" in grp:
                del grp["act_rlt_tokens"]
            grp.create_dataset("act_rlt_tokens", data=z, compression="gzip", compression_opts=4)
            print(f"[{split}] act_rlt_tokens {z.shape} 写入")
    out = args.h5.parent / f"rlt_compressor_k{args.k}_d{args.d}.pt"
    torch.save({"model_state": model.state_dict(), "mean": mean, "std": std,
                "k": args.k, "d": args.d, "val_explained_var": 1 - best}, out)
    print(f"[done] best val explained_var={1-best:.3f}  saved {out}")


if __name__ == "__main__":
    main()
