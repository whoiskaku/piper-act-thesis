#!/usr/bin/env python3
"""回归测试(2026-07-21建, 因发现 apply_G 漏 D 直通项) —
apply_G 必须逐点等于训练路径算出的真实增量 Ĝ(a+δa)−Ĝ(a)。

权威真值 = fit_g_level2.FlexL2 + fit_g_production.rigid_parts(训练时的同一份代码),
名义调度(J, phi)冻结, 与 apply_G 的设计口径一致。
判据: 相对误差 <1% 且方向相关 >0.999(全成员×全关节)。
用法: python sim/verify_apply_G.py
"""
from __future__ import annotations
import json
import sys

import numpy as np
import torch

import train_pi_e2 as TP
from fit_g_level2 import FlexL2, excitation, raw_feats
from fit_g_production import rigid_parts

TAG, EP, TAU, AMP = "pp2-deploy", 8, 200, 0.005


def main():
    G = TP.GEnsemble()
    z = np.load(TP.GROOT / "g_ensemble.npz")
    zl = np.load(TP.LROOT / TAG / f"ep{EP:03d}.npz")
    R = np.array(json.loads((TP.LROOT / TAG / "manifest.json").read_text())["R"], np.float32)
    a = zl["a_arm"].astype(np.float32)
    grip = zl["grip"].astype(np.float32)
    a7 = np.concatenate([a, grip[:, None]], 1)
    S = len(a)
    epJ, alpha, RB, RD = G.ep_context(a, grip, R)

    worst_err, worst_cor = 0.0, 1.0
    print(f"{'成员':>4} {'关节':>4} {'相对误差':>10} {'方向相关':>10} {'幅度比':>8}")
    for k in range(G.K):
        c = z["c_enc"][k]
        fx = FlexL2([3.1, 5.1, 5.7])
        with torch.no_grad():
            for n in ["log_om", "raw_ze", "W", "B", "D", "theta"]:
                getattr(fx, n).copy_(torch.from_numpy(z[n][k]))
        phi = torch.from_numpy(((raw_feats(a7) - z["mu"][k]) / z["sd"][k]).astype(np.float32))
        _, _, _, Jn = rigid_parts({"a": a}, c)             # 名义雅可比(冻结)

        def model_out(cmd):
            qd = np.zeros((len(cmd), 6), np.float32)
            ad = np.vstack([np.zeros((1, 6)), np.diff(cmd, axis=0)]) / TP.DT
            for kk in range(c.shape[1]):
                qd[kk:] += c[:, kk][None, :] * ad[:len(cmd) - kk]
            with torch.no_grad():
                fl = fx(torch.from_numpy(excitation(cmd, c)), phi).numpy()
            return np.einsum("tcj,tj->tc", Jn, qd) + fl    # sim系

        base = model_out(a)
        for jj in range(6):
            d1 = np.zeros_like(a)
            d1[TAU, jj] = AMP
            truth = np.roll((model_out(a + d1) - base) @ R.T, -TP.LSHIFT, 0)
            delta = torch.zeros(1, S, 6, device=TP.DEV)
            delta[0, TAU, jj] = AMP
            with torch.no_grad():
                got = TP.apply_G(delta, epJ.unsqueeze(0), alpha.unsqueeze(0),
                                 RB.unsqueeze(0), RD.unsqueeze(0), G)[0, k].cpu().numpy()
            sl = slice(TAU - 5, TAU + TP.LR_RESP)
            g, t = got[sl], truth[sl]
            if np.abs(t).max() < 1e-9:
                continue
            err = np.sqrt(((g - t) ** 2).mean()) / np.sqrt((t ** 2).mean())
            cor = float(np.sum(g * t) / np.sqrt(np.sum(g ** 2) * np.sum(t ** 2)))
            worst_err, worst_cor = max(worst_err, err), min(worst_cor, cor)
            if k == 0 or err > 0.01:
                print(f"{k:>4} {f'j{jj+1}':>4} {err*100:>9.3f}% {cor:>10.5f} "
                      f"{np.abs(g).max()/np.abs(t).max():>8.3f}")

    # δa≡0 恒等
    zero = torch.zeros(1, S, 6, device=TP.DEV)
    with torch.no_grad():
        dz = TP.apply_G(zero, epJ.unsqueeze(0), alpha.unsqueeze(0),
                        RB.unsqueeze(0), RD.unsqueeze(0), G)
    print(f"\nδa≡0 恒等检查: |Δe|max = {float(dz.abs().max()):.2e}")
    print(f"最坏: 相对误差={worst_err*100:.3f}%  方向相关={worst_cor:.5f}")
    ok = worst_err < 0.01 and worst_cor > 0.999 and float(dz.abs().max()) < 1e-12
    print("判决:", "✅ PASS" if ok else "❌ FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
