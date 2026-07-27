#!/usr/bin/env python3
"""P2.3d — 架构轮收官批: LSTM / vanilla-RNN / TCN-Attention。
协议与 bakeoff_detailed 完全一致。预注册: 本批后架构问题关闭。
输出: outputs/p2_bakeoff/detailed_final3.json
"""
from __future__ import annotations
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bakeoff_backbones import Head, OUT, TRAIN_EPS, VAL_EPS, batches, load_split
from bakeoff_detailed import EVAL_EVERY, FULL_STEPS, LRS, SWEEP_STEPS, load_masks, val_eval

torch.set_num_threads(4)


class LSTMNet(nn.Module):
    def __init__(self, d=45, h=48):
        super().__init__()
        self.rnn = nn.LSTM(d, h, num_layers=2, batch_first=True)
        self.head = Head(h)

    def forward(self, u):
        y, _ = self.rnn(u)
        return self.head(y)


class RNNNet(nn.Module):
    def __init__(self, d=45, h=48):
        super().__init__()
        self.rnn = nn.RNN(d, h, num_layers=2, batch_first=True, nonlinearity="tanh")
        self.head = Head(h)

    def forward(self, u):
        y, _ = self.rnn(u)
        return self.head(y)


class TCNAttn(nn.Module):
    """赢家TCN + 单层因果自注意力(残差) — 隔离注意力的边际价值。长序列滑窗评估。"""
    WIN, WARM = 64, 16

    def __init__(self, d=45, h=32):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(d, h, 3, dilation=1), nn.Conv1d(h, h, 3, dilation=2),
            nn.Conv1d(h, h, 3, dilation=4)])
        self.pads = [2, 4, 8]
        self.attn = nn.MultiheadAttention(h, 4, batch_first=True)
        self.norm = nn.LayerNorm(h)
        self.head = Head(h)

    def _fwd(self, u):
        x = u.transpose(1, 2)
        for c, p in zip(self.convs, self.pads):
            x = F.gelu(c(F.pad(x, (p, 0))))
        x = x.transpose(1, 2)
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=u.device), 1)
        a, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        return self.head(self.norm(x + a))

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


MODELS = {"LSTM": LSTMNet, "RNN": RNNNet, "TCN-Attn": TCNAttn}


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
            print(f"[sweep {name}] lr={lr:g} → {v:.3f} ({time.time()-t0:.0f}s)", flush=True)
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
        (OUT / "detailed_final3.json").write_text(json.dumps(out, indent=1))
    print("done")


if __name__ == "__main__":
    main()
