#!/usr/bin/env python3
"""骨干复核(2026-07-22): race3 在 old16+β1000 下判的, 新默认 new25+β300 下是否仍成立?
TCN / S4D / Transformer ×3种子 vs 已训好的 TCN-deep(pi_default, 5种子)。"""
import json
import numpy as np
import torch
import train_pi_e2 as TP
from rerace_dropout import TCN_D, S4D_D
from bakeoff_more import XFormer
from train_pi_default import BETA, GAMMA, STEPS, LR, load_new25

TP.BP = TP.make_bp(); G = TP.GEnsemble()
tr, va, ppf_va = load_new25(G)
va_pp = [e for e in va if e["tag"] == "pp2-deploy"]
va_ins = [e for e in va if e["tag"] == "ins-deploy"]
res = {}
for name, cls in [("TCN", TCN_D), ("S4D", S4D_D), ("Transformer", XFormer)]:
    recs = []
    for sd in (0, 1, 2):
        m = TP.train_one(cls, LR, sd, STEPS, tr, G, BETA, GAMMA, log_every=10**9)
        rp, ri, rf = (TP.evaluate(m, x, G) for x in (va_pp, va_ins, ppf_va))
        recs.append({"pp2": rp["red"]*100, "ins": ri["red"]*100, "ppf": rf["red"]*100,
                     "pp2_worst": rp["worst"]*100, "da": rp["da_rms_mrad"]})
        print(f"  [{name} s{sd}] pp2={recs[-1]['pp2']:5.1f}% ins={recs[-1]['ins']:5.1f}% "
              f"ppf={recs[-1]['ppf']:5.1f}%", flush=True)
    res[name] = {k: [float(np.mean([x[k] for x in recs])), float(np.std([x[k] for x in recs]))]
                 for k in recs[0]}
    res[name]["n_params"] = sum(p.numel() for p in cls(d=TP.FEAT).parameters())
    (TP.OUT / "backbone_recheck.json").write_text(json.dumps(res, indent=1))
d = json.loads((TP.OUT / "pi_default.json").read_text())
print("\n== 新默认(new25, β=300)下的骨干复核 ==")
print(f"{'TCN-deep(5种子)':>18}: pp2={d['pp2'][0]:5.1f}±{d['pp2'][1]:.1f}  "
      f"ins={d['ins'][0]:5.1f}±{d['ins'][1]:.1f}  ppf={d['ppf'][0]:5.1f}  34662参数")
for n, v in res.items():
    print(f"{n+'(3种子)':>18}: pp2={v['pp2'][0]:5.1f}±{v['pp2'][1]:.1f}  "
          f"ins={v['ins'][0]:5.1f}±{v['ins'][1]:.1f}  ppf={v['ppf'][0]:5.1f}  {v['n_params']}参数")
