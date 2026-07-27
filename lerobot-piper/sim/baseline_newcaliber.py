#!/usr/bin/env python3
"""补: 原基线(old16 + β=1000)在**新口径**(逐chunk ideal)下的数值 —— 说明会第5节需同口径对比。"""
import json

import numpy as np
import torch

import train_pi_e2 as TP
from rerace_dropout import TCN_D

TP.BP = TP.make_bp()
G = TP.GEnsemble()
tr = TP.load_eps(TP.TRAIN_EPS, G)                 # old16 = pp2×8 + ins×8
va = TP.load_eps(TP.VAL_EPS, G)
vp = [e for e in va if e["tag"] == "pp2-deploy"]
vi = [e for e in va if e["tag"] == "ins-deploy"]

recs = []
for sd in (0, 1, 2):
    m = TP.train_one(TCN_D, 1e-3, sd, 6000, tr, G, 1e3, 0.0, log_every=10**9)
    rp, ri = TP.evaluate(m, vp, G), TP.evaluate(m, vi, G)
    recs.append({"pp2": rp["red"] * 100, "ins": ri["red"] * 100,
                 "da": rp["da_rms_mrad"]})
    print(f"  s{sd}: pp2={recs[-1]['pp2']:5.1f}% ins={recs[-1]['ins']:5.1f}%", flush=True)

agg = {k: [float(np.mean([x[k] for x in recs])), float(np.std([x[k] for x in recs]))]
       for k in recs[0]}
(TP.OUT / "baseline_newcaliber.json").write_text(json.dumps(agg, indent=1))
print(f"\n原基线(新口径): pp2={agg['pp2'][0]:.1f}±{agg['pp2'][1]:.1f}%  "
      f"ins={agg['ins'][0]:.1f}±{agg['ins'][1]:.1f}%")
