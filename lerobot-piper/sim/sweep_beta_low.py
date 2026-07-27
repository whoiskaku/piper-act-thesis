#!/usr/bin/env python3
"""β 低段扫描(2026-07-22 用户令): β∈{30,100} ×3种子, new25 数据, TCN-deep 6000步。
与 ab_expand 的 β∈{300,1000} 拼成完整曲线。输出 outputs/p2_pi/beta_low.json"""
import json
import numpy as np
import torch
import train_pi_e2 as TP
from rerace_dropout import TCNDeep_D

TP.BP = TP.make_bp()
G = TP.GEnsemble()
tags0 = TP.TAGS
eps = TP.load_eps(TP.TRAIN_EPS, G)
va = TP.load_eps(TP.VAL_EPS, G)
TP.TAGS = ["pp2f-deploy"]
eps += TP.load_eps(list(range(6)), G)
ppf_va = TP.load_eps([6], G)
TP.TAGS = ["pp2f60-deploy"]
eps += TP.load_eps([0, 1, 2], G)
TP.TAGS = tags0
va_pp = [e for e in va if e["tag"] == "pp2-deploy"]
va_ins = [e for e in va if e["tag"] == "ins-deploy"]

res = {}
for beta in (1e2, 3e1):
    recs = []
    for sd in (0, 1, 2):
        m = TP.train_one(TCNDeep_D, 1e-3, sd, 6000, eps, G, beta, 0.0, log_every=10**9)
        rp, ri, rf = (TP.evaluate(m, x, G) for x in (va_pp, va_ins, ppf_va))
        r = {"pp2": rp["red"]*100, "ins": ri["red"]*100, "ppf": rf["red"]*100,
             "da": rp["da_rms_mrad"], "sat": rp["sat_pct"],
             "worst_pp2": rp["worst"]*100, "worst_ins": ri["worst"]*100}
        recs.append(r)
        print(f"  [β={beta:g} s{sd}] pp2={r['pp2']:5.1f}% ins={r['ins']:5.1f}% "
              f"ppf={r['ppf']:5.1f}% ‖δa‖={r['da']:.2f} sat={r['sat']:.0f}%", flush=True)
    res[f"b{beta:g}"] = {k: [float(np.mean([x[k] for x in recs])),
                             float(np.std([x[k] for x in recs]))] for k in recs[0]}
    (TP.OUT / "beta_low.json").write_text(json.dumps(res, indent=1))
print("\n完整曲线(new25, 3种子):")
prev = json.loads((TP.OUT / "ab_expand.json").read_text())
for k, lbl in [("b30", "β=30"), ("b100", "β=100")]:
    if k in res:
        v = res[k]
        print(f"  {lbl:>8}: pp2={v['pp2'][0]:5.1f}±{v['pp2'][1]:.1f}  ins={v['ins'][0]:5.1f}±{v['ins'][1]:.1f}"
              f"  ppf={v['ppf'][0]:5.1f}  ‖δa‖={v['da'][0]:.2f}  最差成员 pp2={v['worst_pp2'][0]:.1f}/ins={v['worst_ins'][0]:.1f}")
for k, lbl in [("new25_b300", "β=300"), ("new25_b1000", "β=1000")]:
    v = prev[k]
    print(f"  {lbl:>8}: pp2={v['pp2'][0]:5.1f}±{v['pp2'][1]:.1f}  ins={v['ins'][0]:5.1f}±{v['ins'][1]:.1f}"
          f"  ppf={v['ppf'][0]:5.1f}  ‖δa‖={v['da'][0]:.2f}")
