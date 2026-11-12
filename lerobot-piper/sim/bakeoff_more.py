#!/usr/bin/env python3
"""P2.3c — 扩编 bake-off: Transformer / GRU / Mamba-lite / TCN-deep / MLP-window。
协议与 bakeoff_detailed.py 完全一致(lr扫描×1500 → 最优lr×3种子×6000, 同数据同头同损失)。
输出: outputs/p2_bakeoff/detailed_more.json
"""
from __future__ import annotations
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bakeoff_backbones import EPS_B, Head, OUT, TRAIN_EPS, VAL_EPS, batches, load_split
from bakeoff_detailed import EVAL_EVERY, FULL_STEPS, LRS, SWEEP_STEPS, load_masks, val_eval

torch.set_num_threads(4)


class XFormer(nn.Module):
    """因果小Transformer。长序列用滑窗评估(窗64=训练分布, 暖机16帧)——
    这也是它部署时的真实模式(每步看最近一窗)。"""
    WIN, WARM = 64, 16

    def __init__(self, d=45, h=48, nlayer=2):
        super().__init__()
        self.proj = nn.Linear(d, h)
        self.pos = nn.Parameter(torch.randn(1, self.WIN, h) * 0.02)
        layer = nn.TransformerEncoderLayer(h, 4, 96, dropout=0.1, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayer)
        self.head = Head(h)

    def _fwd(self, u):
        T = u.shape[1]
        x = self.proj(u) + self.pos[:, :T]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=u.device), 1)
        return self.head(self.enc(x, mask=mask))

    def forward(self, u):
        T = u.shape[1]
        if T <= self.WIN:
            return self._fwd(u)
        ys = [self._fwd(u[:, :self.WIN])]
        pos = self.WIN
        while pos < T:
            s = max(0, min(pos - self.WARM, T - self.WIN))
            y = self._fwd(u[:, s:s + self.WIN])
            ys.append(y[:, pos - s:])
            pos = s + self.WIN
        return torch.cat(ys, 1)


class GRUNet(nn.Module):
    def __init__(self, d=45, h=48):
        super().__init__()
        self.gru = nn.GRU(d, h, num_layers=2, batch_first=True)
        self.head = Head(h)

    def forward(self, u):
        y, _ = self.gru(u)
        return self.head(y)


class MambaLite(nn.Module):
    """简化选择性SSM(S6核心): dt/B/C 随输入变化, A 对数参数化, python递归。"""

    def __init__(self, d=45, H=32, N=8):
        super().__init__()
        self.enc = nn.Linear(d, H)
        self.A_log = nn.Parameter(torch.log(torch.rand(H, N) * 3 + 0.5))
        self.W_dt = nn.Linear(H, H)
        self.W_B = nn.Linear(H, N)
        self.W_C = nn.Linear(H, N)
        self.D = nn.Parameter(torch.ones(H))
        self.head = Head(H)

    def forward(self, u):
        x = F.gelu(self.enc(u))                       # (B,T,H)
        Bsz, T, H = x.shape
        N = self.A_log.shape[1]
        A = -torch.exp(self.A_log)                    # (H,N)
        st = x.new_zeros(Bsz, H, N)
        ys = []
        for t in range(T):
            xt = x[:, t]
            dt = F.softplus(self.W_dt(xt))            # (B,H)
            Bt, Ct = self.W_B(xt), self.W_C(xt)       # (B,N)
            st = st * torch.exp(A[None] * dt[..., None]) + \
                 dt[..., None] * Bt[:, None, :] * xt[..., None]
            ys.append((st * Ct[:, None, :]).sum(-1) + self.D * xt)
        return self.head(torch.stack(ys, 1))


class TCNDeep(nn.Module):
    def __init__(self, d=45, h=48):
        super().__init__()
        dil = [1, 2, 4, 8, 16]
        chs = [d] + [h] * len(dil)
        self.convs = nn.ModuleList(nn.Conv1d(chs[i], chs[i+1], 3, dilation=dl)
                                   for i, dl in enumerate(dil))
        self.dil = dil
        self.head = Head(h)

    def forward(self, u):
        x = u.transpose(1, 2)
        for i, (c, dl) in enumerate(zip(self.convs, self.dil)):
            y = F.gelu(c(F.pad(x, (2 * dl, 0))))
            x = y if i == 0 else x + y                # 残差(形状对齐后)
        return self.head(x.transpose(1, 2))


class MLPWin(nn.Module):
    def __init__(self, d=45, w=16):
        super().__init__()
        self.w = w
        self.mlp = nn.Sequential(nn.Linear(d * w, 128), nn.GELU(),
                                 nn.Linear(128, 64), nn.GELU())
        self.head = Head(64)

    def forward(self, u):
        x = F.pad(u, (0, 0, self.w - 1, 0))
        xu = x.unfold(1, self.w, 1)                   # (B,T,45,w)
        h = self.mlp(xu.reshape(u.shape[0], u.shape[1], -1))
        return self.head(h)


MODELS = {"Transformer": XFormer, "GRU": GRUNet, "Mamba-lite": MambaLite,
          "TCN-deep": TCNDeep, "MLP-window": MLPWin}


def train_one(cls, lr, seed, steps, xtr, ytr, atr, evalf):
    torch.manual_seed(seed); np.random.seed(seed)
    model = cls()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    curve = []
    model.train()
    for i, (xb, yb) in enumerate(batches(xtr, ytr, atr, steps=steps)):
        loss = F.huber_loss(model(xb), yb, delta=0.005)
        opt.zero_grad(); loss.backward(); opt.step()
        if (i + 1) % EVAL_EVERY == 0:
            curve.append({"step": i + 1, **evalf(model)})
            model.train()
    return model, curve


def main():
    xtr, ytr, atr = load_split(TRAIN_EPS)
    xva, yva, _ = load_split(VAL_EPS)
    mact, mct, mpp2 = load_masks(VAL_EPS)
    evalf = lambda m: val_eval(m, xva, yva, mact, mct, mpp2)
    out = {"sweep": {}, "runs": {}}
    for name, cls in MODELS.items():
        nparam = sum(p.numel() for p in cls().parameters())
        best = (None, 1e9)
        out["sweep"][name] = {"params": nparam}
        for lr in LRS:
            t0 = time.time()
            _, curve = train_one(cls, lr, 0, SWEEP_STEPS, xtr, ytr, atr, evalf)
            v = curve[-1]["all"]
            out["sweep"][name][str(lr)] = v
            print(f"[sweep {name}] lr={lr:g} → {v:.3f} mrad ({time.time()-t0:.0f}s)", flush=True)
            if v < best[1]:
                best = (lr, v)
        out["runs"][name] = {"best_lr": best[0], "params": nparam, "seeds": []}
        for seed in (0, 1, 2):
            t0 = time.time()
            _, curve = train_one(cls, best[0], seed, FULL_STEPS, xtr, ytr, atr, evalf)
            out["runs"][name]["seeds"].append({"seed": seed, "curve": curve})
            f = curve[-1]
            print(f"[full {name}] seed{seed} → all {f['all']:.3f} act {f['act']:.3f} "
                  f"contact {f['contact']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        (OUT / "detailed_more.json").write_text(json.dumps(out, indent=1))
    print("done")


if __name__ == "__main__":
    main()
