#!/usr/bin/env python3
"""E-2 前置 — Ĝ 集成: K=5 份 L2(成员0=全量数据, 1-4=episode bootstrap)。
用途: π 对练的防剥削训练(loss 对 K 份取均值, 评估另看最差成员)。
每份 = c_enc(编码器LS) + FlexL2(位形调度模态, M=3, 激励修复口径), 与终稿同构。
输出: outputs/p2_bakeoff/g_ensemble.npz (K 维堆叠的全部参数 + 特征标准化统计)
"""
from __future__ import annotations
import numpy as np
import torch

from bakeoff_g import fit_c_enc
from compare_ideal_methods import load_episode
from fit_g_level2 import FlexL2, contact_mask, excitation, fit_flex, raw_feats
from fit_g_production import BP, OUT, bp_t, rigid_parts
from g_verdict import DATASETS, DT, TRAIN_EPS, load

torch.set_num_threads(4)
K_ENS = 5


def main():
    train = load(TRAIN_EPS, with_gap=True)
    i = 0
    for repo, tag in DATASETS:
        for ei in TRAIN_EPS:
            d = load_episode(repo, ei)
            train[i]["a7"] = d["action"][:len(train[i]["a"])].astype(np.float32)
            train[i]["_ep"] = ei
            i += 1

    packs = {k: [] for k in ["c_enc", "log_om", "raw_ze", "W", "B", "D", "theta", "mu", "sd"]}
    rng = np.random.default_rng(0)
    for k in range(K_ENS):
        idx = list(range(len(train))) if k == 0 else \
            rng.integers(0, len(train), len(train)).tolist()
        boot = [train[j] for j in idx]
        c_enc, _ = fit_c_enc(boot)
        F_all = np.concatenate([raw_feats(e["a7"]) for e in boot])
        mu, sd = F_all.mean(0), F_all.std(0) + 1e-8
        mu[0], sd[0] = 0.0, 1.0
        uts, rts, phis, mks = [], [], [], []
        for e in boot:
            _, _, w, _ = rigid_parts(e, c_enc)
            u = excitation(e["a"], c_enc)
            r = bp_t((e["g"] @ e["R"]).astype(np.float32) - w, BP).astype(np.float32)
            uts.append(torch.from_numpy(u.copy()))
            rts.append(torch.from_numpy(r.copy()))
            phis.append(torch.from_numpy(((raw_feats(e["a7"]) - mu) / sd).astype(np.float32)))
            mks.append(torch.from_numpy(~contact_mask(e["tag"], e["_ep"], len(e["a"]))))
        # FIR 谱峰初始化(与 L2 同法, 简化: 直接用终稿极点邻域)
        torch.manual_seed(k)
        fx = FlexL2([3.1, 5.1, 5.7])
        fx = fit_flex(fx, uts, rts, phis, mks, iters=500)
        poles = np.sort(torch.exp(fx.log_om).detach().numpy() / (2 * np.pi))
        print(f"[成员{k}] {'全量' if k==0 else 'bootstrap'}  极点={np.round(poles,2)}Hz", flush=True)
        packs["c_enc"].append(c_enc)
        packs["mu"].append(mu); packs["sd"].append(sd)
        for name, val in [("log_om", fx.log_om), ("raw_ze", fx.raw_ze), ("W", fx.W),
                          ("B", fx.B), ("D", fx.D), ("theta", fx.theta)]:
            packs[name].append(val.detach().numpy())

    np.savez(OUT / "g_ensemble.npz", **{k: np.stack(v) for k, v in packs.items()})
    print(f"→ {OUT/'g_ensemble.npz'}  (K={K_ENS})")


if __name__ == "__main__":
    main()
