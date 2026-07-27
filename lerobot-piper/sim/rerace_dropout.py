#!/usr/bin/env python3
"""E-2 重赛 — 起跑线拉平: 全员 dropout(0.1), 其余与上场逐字同。

2026-07-21 二刷(race3): apply_G 修复三处(补 D 直通项 / 核改数值探针消除 1 帧错位与
非因果丢失 / 模态改两段式使 alpha 夹在激励与谐振之间)。修复后 apply_G 与训练路径
逐点等价(verify_apply_G.py: 最坏 0.22%, 相关 1.00000)。race2 及之前全部作废。

预注册(先于结果):
  协议: 每模型 lr∈{1e-3,3e-3} 各800步探针(seed0)取优 → 3种子×6000步, β=1000/γ=0;
        第二阶段: 与榜首带交叠或差距<2pp 的模型加跑 seed 3,4(→5种子)后终判。
  dropout 规则: 每个隐层输出处 p=0.1 (Transformer 原生已带, 不动)。
  判决规则(与上场同): ①须胜 δa≡0 底线 ②误差带交叠/小胜 → 参数少者胜
        ③最差成员不塌方 ④饱和率>5% 红旗。
输出: outputs/p2_pi/race2.json, pi2_<name>_s<seed>.pt
"""
from __future__ import annotations
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_pi_e2 as T
from bakeoff_backbones import S4DNet, TCN
from bakeoff_final3 import LSTMNet
from bakeoff_more import GRUNet, MLPWin, TCNDeep, XFormer

P = 0.1


class TCN_D(TCN):
    def forward(self, u):
        x = u.transpose(1, 2)
        for c, p in zip(self.convs, self.pads):
            x = F.dropout(F.gelu(c(F.pad(x, (p, 0)))), P, self.training)
        return self.head(x.transpose(1, 2))


class TCNDeep_D(TCNDeep):
    def forward(self, u):
        x = u.transpose(1, 2)
        for i, (c, dl) in enumerate(zip(self.convs, self.dil)):
            y = F.dropout(F.gelu(c(F.pad(x, (2 * dl, 0)))), P, self.training)
            x = y if i == 0 else x + y
        return self.head(x.transpose(1, 2))


class GRU_D(GRUNet):
    def __init__(self, d=45, h=48):
        super().__init__(d, h)
        self.gru = nn.GRU(d, h, num_layers=2, batch_first=True, dropout=P)

    def forward(self, u):
        y, _ = self.gru(u)
        return self.head(F.dropout(y, P, self.training))


class LSTM_D(LSTMNet):
    def __init__(self, d=45, h=48):
        super().__init__(d, h)
        self.rnn = nn.LSTM(d, h, num_layers=2, batch_first=True, dropout=P)

    def forward(self, u):
        y, _ = self.rnn(u)
        return self.head(F.dropout(y, P, self.training))


class S4D_D(S4DNet):
    def forward(self, u):
        x = F.dropout(F.gelu(self.enc(u)), P, self.training)
        x = self.n1(x + F.dropout(F.gelu(self.s1(x)), P, self.training))
        x = self.n2(x + F.dropout(F.gelu(self.s2(x)), P, self.training))
        return self.head(x)


class MLP_D(MLPWin):
    def __init__(self, d=45, w=16):
        super().__init__(d, w)
        self.mlp = nn.Sequential(nn.Linear(d * w, 128), nn.GELU(), nn.Dropout(P),
                                 nn.Linear(128, 64), nn.GELU(), nn.Dropout(P))


ZOO = {"TCN": TCN_D, "TCN-deep": TCNDeep_D, "GRU": GRU_D, "LSTM": LSTM_D,
       "S4D": S4D_D, "Transformer": XFormer, "MLP-window": MLP_D}
_pick = json.loads((T.OUT / "sweep.json").read_text())["_pick"]     # 膝点随对练场重扫
BETA = float(_pick.split("_")[0][1:])
GAMMA = float(_pick.split("_g")[1])
RACE, PREF = "race3.json", "pi3"


def run_seeds(cls, lr, seeds, eps_tr, eps_va, G, name):
    recs = []
    for seed in seeds:
        m = T.train_one(cls, lr, seed, 6000, eps_tr, G, BETA, GAMMA, log_every=10**9)
        torch.save(m.state_dict(), T.OUT / f"{PREF}_{name}_s{seed}.pt")
        recs.append(T.evaluate(m, eps_va, G))
    return recs


def agg(recs):
    return {k: [float(np.mean([r[k] for r in recs])), float(np.std([r[k] for r in recs]))]
            for k in recs[0]}


def main():
    T.BP = T.make_bp()
    G = T.GEnsemble()
    eps_tr, eps_va = T.load_eps(T.TRAIN_EPS, G), T.load_eps(T.VAL_EPS, G)
    res, all_recs = {}, {}
    print(f"膝点(读自 sweep.json): β={BETA:g} γ={GAMMA:g}\n", flush=True)
    for name, cls in ZOO.items():
        best = (None, -1e9)
        for lr in (1e-3, 3e-3):
            m = T.train_one(cls, lr, 0, 800, eps_tr, G, BETA, GAMMA, log_every=10**9)
            r = T.evaluate(m, eps_va, G)
            if r["red"] > best[1]:
                best = (lr, r["red"])
        t0 = time.time()
        recs = run_seeds(cls, best[0], (0, 1, 2), eps_tr, eps_va, G, name)
        all_recs[name] = recs
        res[name] = agg(recs)
        res[name]["best_lr"] = best[0]
        res[name]["n_params"] = sum(p.numel() for p in cls().parameters())
        r = res[name]
        print(f"[{name}] 消减={r['red'][0]*100:.1f}±{r['red'][1]*100:.1f}% "
              f"运动段={r['red_motion'][0]*100:.1f}% 最差={r['worst'][0]*100:.1f}% "
              f"‖δa‖={r['da_rms_mrad'][0]:.2f} sat={r['sat_pct'][0]:.0f}% "
              f"params={r['n_params']} (lr={best[0]:g}, {time.time()-t0:.0f}s)", flush=True)
        (T.OUT / RACE).write_text(json.dumps(res, indent=1))

    # ---- 第二阶段: 榜首邻域加种子 → 5种子终判 ----
    leader = max(res, key=lambda n: res[n]["red"][0])
    lm, ls = res[leader]["red"]
    finalists = [n for n in res
                 if res[n]["red"][0] + res[n]["red"][1] >= lm - ls
                 or lm - res[n]["red"][0] < 0.02]
    print(f"\n第二阶段(5种子终判): {finalists}", flush=True)
    for name in finalists:
        recs = all_recs[name] + run_seeds(ZOO[name], res[name]["best_lr"], (3, 4),
                                          eps_tr, eps_va, G, name)
        res[name] = {**agg(recs), "best_lr": res[name]["best_lr"],
                     "n_params": res[name]["n_params"], "n_seeds": 5}
        r = res[name]
        print(f"[终判 {name}] 消减={r['red'][0]*100:.1f}±{r['red'][1]*100:.1f}% "
              f"最差={r['worst'][0]*100:.1f}% params={r['n_params']}", flush=True)
        (T.OUT / RACE).write_text(json.dumps(res, indent=1))
    print("race2 done")


if __name__ == "__main__":
    main()
