#!/usr/bin/env python3
"""β 完整曲线 · 新口径(逐chunk ideal, π v1=TCN, new25) — 补齐 {1000, 100} 并汇总。
已有: β=300(pi_v1, 5种子) / β∈{300,30}(probe_eps, 3种子)。
输出 outputs/p2_pi/beta_curve_final.json
"""
import json

import numpy as np
import torch

import train_pi_e2 as TP
from rerace_dropout import TCN_D
from train_pi_default import GAMMA, LR, STEPS, load_new25

TP.BP = TP.make_bp()
G = TP.GEnsemble()
tr, va, ppf_va = load_new25(G)
vp = [e for e in va if e["tag"] == "pp2-deploy"]
vi = [e for e in va if e["tag"] == "ins-deploy"]

out = {}
for beta in (1e3, 1e2):
    recs = []
    for sd in (0, 1, 2):
        m = TP.train_one(TCN_D, LR, sd, STEPS, tr, G, beta, GAMMA, log_every=10**9)
        rp, ri, rf = (TP.evaluate(m, x, G) for x in (vp, vi, ppf_va))
        recs.append({"pp2": rp["red"] * 100, "ins": ri["red"] * 100,
                     "ppf": rf["red"] * 100, "da": rp["da_rms_mrad"],
                     "sat": rp["sat_pct"], "pp2_worst": rp["worst"] * 100,
                     "damax": float(m(vp[0]["X"].unsqueeze(0)).abs().max()) * 1e3})
    out[f"b{beta:g}"] = {k: [float(np.mean([x[k] for x in recs])),
                             float(np.std([x[k] for x in recs]))] for k in recs[0]}
    r = out[f"b{beta:g}"]
    print(f"[β={beta:g}] pp2={r['pp2'][0]:5.1f}±{r['pp2'][1]:.1f}% "
          f"ins={r['ins'][0]:5.1f}±{r['ins'][1]:.1f}% ppf={r['ppf'][0]:5.1f}% "
          f"‖δa‖={r['da'][0]:.2f} 峰={r['damax'][0]:.1f} sat={r['sat'][0]:.1f}%", flush=True)
    (TP.OUT / "beta_curve_final.json").write_text(json.dumps(out, indent=1))
print("done")
