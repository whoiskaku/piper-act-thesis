#!/usr/bin/env python3
"""判决: 抬高 ε 硬界有用吗? (2026-07-22 用户提问)

假设: ε 当前不起约束作用(实际 ‖δa‖=界的11%, 饱和0%), 抬高它应无效;
     唯一可能起作用的是 β 低到 ε 开始被顶(β=30 时饱和1%)。
设计: 2×2 因子 —— β∈{300, 30} × ε∈{10, 20} mrad, 各3种子, 其余同定档配置。
"""
import json

import numpy as np
import torch

import bakeoff_backbones as BB
import train_pi_e2 as TP
from rerace_dropout import TCN_D
from train_pi_default import GAMMA, LR, STEPS, load_new25

TP.BP = TP.make_bp()
G = TP.GEnsemble()
tr, va, ppf_va = load_new25(G)
vp = [e for e in va if e["tag"] == "pp2-deploy"]
vi = [e for e in va if e["tag"] == "ins-deploy"]

res = {}
for beta in (3e2, 3e1):
    for eps in (0.010, 0.020):
        BB.EPS_B = eps                       # Head 每次 forward 读全局, 改这里即生效
        TP.EPS_B = eps                       # evaluate 的饱和判据也跟着走
        recs = []
        for sd in (0, 1, 2):
            m = TP.train_one(TCN_D, LR, sd, STEPS, tr, G, beta, GAMMA, log_every=10**9)
            rp, ri = TP.evaluate(m, vp, G), TP.evaluate(m, vi, G)
            recs.append({"pp2": rp["red"] * 100, "ins": ri["red"] * 100,
                         "da": rp["da_rms_mrad"], "sat": rp["sat_pct"],
                         "damax": float(m(vp[0]["X"].unsqueeze(0)).abs().max()) * 1e3})
        k = f"b{beta:g}_e{eps*1000:g}"
        res[k] = {q: [float(np.mean([x[q] for x in recs])),
                      float(np.std([x[q] for x in recs]))] for q in recs[0]}
        r = res[k]
        print(f"[β={beta:g} ε={eps*1000:g}mrad] pp2={r['pp2'][0]:5.1f}±{r['pp2'][1]:.1f}% "
              f"ins={r['ins'][0]:5.1f}±{r['ins'][1]:.1f}% ‖δa‖={r['da'][0]:.2f} "
              f"峰={r['damax'][0]:.1f}mrad sat={r['sat'][0]:.1f}%", flush=True)
        (TP.OUT / "eps_probe.json").write_text(json.dumps(res, indent=1))

print("\n== 判决 ==")
for beta in ("300", "30"):
    a, b = res[f"b{beta}_e10"], res[f"b{beta}_e20"]
    dp, di = b["pp2"][0] - a["pp2"][0], b["ins"][0] - a["ins"][0]
    sd = np.sqrt((a["pp2"][1] ** 2 + b["pp2"][1] ** 2) / 3)
    print(f"β={beta}: ε 10→20mrad 使 pp2 {dp:+.1f}pp ({abs(dp)/max(sd,1e-9):.1f}σ) "
          f"ins {di:+.1f}pp;  ‖δa‖ {a['da'][0]:.2f}→{b['da'][0]:.2f}")
