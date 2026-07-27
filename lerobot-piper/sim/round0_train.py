#!/usr/bin/env python3
"""P2 — B 模块 round-0 正式训练 (TCN 定案配置)。

相对 bake-off 的四项部署化改动:
  ① boundary-hold: 预览特征(ideal/a 的 t+4, t+8)不越 chunk 边界(部署时新chunk
     未重规划前不可知) — 越界处 hold 在边界前最后一帧;
  ② 谱归一化(所有 conv + 头 Linear) — Lipschitz 有界, 回答"学习模块会不会自激";
  ③ 输出平滑正则 λs·mean(Δδ̂²) — 部署侧不引入新毛刺;
  ④ 逐-eval 最优 checkpoint 保存。
配置: main(②③开, 3种子) + ref(②③关, 1种子, 量化安全层代价), 各10000步。
输出: outputs/p2_round0/{tcn_main_s*.pt, tcn_ref_s0.pt, curves.json, manifest.json}
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from bakeoff_backbones import EPS_B, ROOT, TAGS, TCN, batches

torch.set_num_threads(4)
OUT = Path(__file__).resolve().parents[1] / "outputs/p2_round0"
TRAIN_EPS, VAL_EPS = list(range(8)), [8, 9]
STEPS, EVAL_EVERY, LR, LS = 10000, 500, 5e-3, 0.1
CHUNK = 100


def load(eps_list):
    """因果特征(2026-07-14 修正 L 帧泄漏): 单一真源 b_runtime.build_offline。"""
    from lerobot.common.policies.act.b_runtime import build_offline
    xs, ys, acts = [], [], []
    for tag in TAGS:
        for ep in eps_list:
            f, y, m = build_offline(np.load(ROOT / tag / f"ep{ep:03d}.npz"), L=3)
            xs.append(f); ys.append(y); acts.append(m)
    return xs, ys, acts


def make_model(sn: bool):
    m = TCN()
    if sn:
        for i, c in enumerate(m.convs):
            m.convs[i] = spectral_norm(c)
        m.head.lin = spectral_norm(m.head.lin)
    return m


def evaluate(model, xs, ys, acts):
    model.eval()
    se = sa = n = na = 0.0
    rough = []
    with torch.no_grad():
        for x, y, a in zip(xs, ys, acts):
            p = model(torch.from_numpy(x).unsqueeze(0))[0].numpy()
            d2 = (p - y) ** 2
            se += d2.sum(); n += d2.size
            sa += d2[a].sum(); na += d2[a].size
            rough.append(np.abs(np.diff(p, axis=0)).mean())
    return (float(np.sqrt(se / n) * 1000), float(np.sqrt(sa / max(na, 1)) * 1000),
            float(np.mean(rough) * 1000))


def train(tag, sn, ls, seed, xtr, ytr, atr, xva, yva, ava):
    torch.manual_seed(seed); np.random.seed(seed)
    model = make_model(sn)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best = (1e9, None)
    curve = []
    t0 = time.time()
    model.train()
    for i, (xb, yb) in enumerate(batches(xtr, ytr, atr, steps=STEPS)):
        pred = model(xb)
        loss = F.huber_loss(pred, yb, delta=0.005)
        if ls:
            loss = loss + ls * ((pred[:, 1:] - pred[:, :-1]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (i + 1) % EVAL_EVERY == 0:
            v_all, v_act, rough = evaluate(model, xva, yva, ava)
            curve.append({"step": i + 1, "all": v_all, "act": v_act, "rough": rough})
            if v_all < best[0]:
                best = (v_all, {k: v.detach().clone() for k, v in model.state_dict().items()})
            model.train()
    model.load_state_dict(best[1])
    v_all, v_act, rough = evaluate(model, xva, yva, ava)
    torch.save(model.state_dict(), OUT / f"tcn_{tag}_s{seed}.pt")
    print(f"[{tag} s{seed}] best val all={v_all:.3f} act={v_act:.3f} "
          f"粗糙度={rough:.3f} mrad/帧 ({time.time()-t0:.0f}s)", flush=True)
    return {"seed": seed, "curve": curve, "best": {"all": v_all, "act": v_act, "rough": rough}}


def main():
    """正式配置(2026-07-14 用户选项C): SN关+λs关(消融归因后数据/用户裁决);
    安全=ε硬界+gate+测量式认证(经验增益+谱范数报告)。消融史见 memory/曲线json。"""
    OUT.mkdir(parents=True, exist_ok=True)
    xtr, ytr, atr = load(TRAIN_EPS)
    xva, yva, ava = load(VAL_EPS)
    print(f"train {sum(len(x) for x in xtr)} / val {sum(len(x) for x in xva)} 帧 (boundary-hold 已启用)")
    res = {"final": []}
    for seed in (0, 1, 2):
        res["final"].append(train("final", False, 0.0, seed, xtr, ytr, atr, xva, yva, ava))
    bests = [r["best"]["all"] for r in res["final"]]
    pick = int(np.argmin(bests))
    manifest = {
        "arch": "TCN(45→32,k3,d1/2/4)+tanh×ε", "eps_rad": EPS_B, "lr": LR,
        "config": "C: 无SN/无λs(消融: SN-43%, λs有偏—标签粗糙度1.602≈无正则输出1.598)",
        "safety": "ε=10mrad结构硬界 + 静止gate + 夹爪直通 + 测量式认证(见gain字段)",
        "spectral_norm": False, "lambda_smooth": 0.0, "steps": STEPS,
        "boundary_hold": True, "feature_dim": 45, "chunk": CHUNK,
        "deploy_pick": f"tcn_final_s{res['final'][pick]['seed']}.pt",
        "final_val": bests,
        "final_rough": [r["best"]["rough"] for r in res["final"]],
        "ablation": {"SN+ls": 1.031, "SN_only": 0.971, "ls_only": 0.831, "plain": 0.681},
        "lipschitz_sigma_product": 34.5,
    }
    (OUT / "curves.json").write_text(json.dumps(res, indent=1))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nfinal(3种子) val={np.mean(bests):.3f}±{np.std(bests):.3f}")
    print(f"部署候选: {manifest['deploy_pick']}")


if __name__ == "__main__":
    main()
