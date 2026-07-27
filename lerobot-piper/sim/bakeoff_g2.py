#!/usr/bin/env python3
"""Ĝ bake-off v2 第二批(主流增补): Transformer / LSTM / RNN / Mamba-lite / TCN-deep。
评分/协议与 bakeoff_g.py v2 完全一致(带号cos裁判/双R²/双极性线性度/权重落盘)。
输出: outputs/p2_bakeoff/g_bakeoff2.json
"""
from __future__ import annotations
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bakeoff_g import (LRS, MDL, OUT, dual_r2, ep_J, evaluate_generic, fit_c_enc,
                       make_f, train_nn)
from g_verdict import KIN, LAGS, TRAIN_EPS, VAL_EPS, load

torch.set_num_threads(4)


class XFormerG(nn.Module):
    WIN, WARM = 64, 16

    def __init__(self, h=48, nlayer=2):
        super().__init__()
        self.proj = nn.Linear(6, h)
        self.pos = nn.Parameter(torch.randn(1, self.WIN, h) * 0.02)
        layer = nn.TransformerEncoderLayer(h, 4, 96, dropout=0.1, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayer)
        self.out = nn.Linear(h, 3)

    def _fwd(self, u):
        T = u.shape[1]
        x = self.proj(u) + self.pos[:, :T]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=u.device), 1)
        return self.out(self.enc(x, mask=mask))

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


class LSTMG(nn.Module):
    def __init__(self, h=48):
        super().__init__()
        self.rnn = nn.LSTM(6, h, num_layers=2, batch_first=True)
        self.out = nn.Linear(h, 3)

    def forward(self, u):
        y, _ = self.rnn(u)
        return self.out(y)


class RNNG(nn.Module):
    def __init__(self, h=48):
        super().__init__()
        self.rnn = nn.RNN(6, h, num_layers=2, batch_first=True, nonlinearity="tanh")
        self.out = nn.Linear(h, 3)

    def forward(self, u):
        y, _ = self.rnn(u)
        return self.out(y)


class MambaG(nn.Module):
    def __init__(self, H=32, N=8):
        super().__init__()
        self.enc = nn.Linear(6, H)
        self.A_log = nn.Parameter(torch.log(torch.rand(H, N) * 3 + 0.5))
        self.W_dt = nn.Linear(H, H)
        self.W_B = nn.Linear(H, N)
        self.W_C = nn.Linear(H, N)
        self.D = nn.Parameter(torch.ones(H))
        self.out = nn.Linear(H, 3)

    def forward(self, u):
        x = F.gelu(self.enc(u))
        Bsz, T, H = x.shape
        A = -torch.exp(self.A_log)
        st = x.new_zeros(Bsz, H, self.A_log.shape[1])
        ys = []
        for t in range(T):
            xt = x[:, t]
            dt = F.softplus(self.W_dt(xt))
            Bt, Ct = self.W_B(xt), self.W_C(xt)
            st = st * torch.exp(A[None] * dt[..., None]) + dt[..., None] * Bt[:, None, :] * xt[..., None]
            ys.append((st * Ct[:, None, :]).sum(-1) + self.D * xt)
        return self.out(torch.stack(ys, 1))


class TCNDeepG(nn.Module):
    def __init__(self, h=48):
        super().__init__()
        dil = [1, 2, 4, 8, 16]
        chs = [6] + [h] * len(dil)
        self.convs = nn.ModuleList(nn.Conv1d(chs[i], chs[i + 1], 3, dilation=d)
                                   for i, d in enumerate(dil))
        self.dil = dil
        self.out = nn.Linear(h, 3)

    def forward(self, u):
        x = u.transpose(1, 2)
        for i, (c, d) in enumerate(zip(self.convs, self.dil)):
            y = F.gelu(c(F.pad(x, (2 * d, 0))))
            x = y if i == 0 else x + y
        return self.out(x.transpose(1, 2))


def main():
    train, val = load(TRAIN_EPS, with_gap=True), load(VAL_EPS, with_gap=True)
    _, S = fit_c_enc(train)
    J_cache = {id(e): ep_J(e) for e in train + val}
    rng = np.random.default_rng(0)
    site_map = {id(e): rng.integers(KIN, len(e["a"]) - LAGS - 1, 10) for e in val}
    res = {}
    for name, cls in [("Transformer", XFormerG), ("LSTM", LSTMG), ("RNN", RNNG),
                      ("Mamba-lite", MambaG), ("TCN-deep", TCNDeepG)]:
        best = (None, -1e9)
        for lr in LRS:
            m = train_nn(cls, lr, 0, 1500, train, J_cache)
            v = np.mean([dual_r2(make_f(m, ep, J_cache[id(ep)])(ep["a"]), ep)[1] for ep in val])
            if v > best[1]:
                best = (lr, v)
        t0 = time.time()
        models = []
        for s in (0, 1, 2):
            m = train_nn(cls, best[0], s, 4000, train, J_cache)
            torch.save(m.state_dict(), MDL / f"{name}_s{s}.pt")
            models.append(m)
        evaluate_generic(name, models, val, J_cache, site_map, S, res)
        res[name]["best_lr"] = best[0]
        print(f"    (best_lr={best[0]:g}, {time.time()-t0:.0f}s)", flush=True)
        (OUT / "g_bakeoff2.json").write_text(json.dumps(res, indent=1))
    print("done")


if __name__ == "__main__":
    main()
