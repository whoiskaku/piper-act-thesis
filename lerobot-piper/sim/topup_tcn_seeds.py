#!/usr/bin/env python3
"""按预注册协议补种子: TCN 与 TCN-deep 接近 → TCN 补 s3,s4 凑满 5 种子再判。"""
import json
import numpy as np
import torch
import train_pi_e2 as TP
from rerace_dropout import TCN_D
from train_pi_default import BETA, GAMMA, STEPS, LR, load_new25

TP.BP = TP.make_bp(); G = TP.GEnsemble()
tr, va, ppf_va = load_new25(G)
va_pp = [e for e in va if e["tag"] == "pp2-deploy"]
va_ins = [e for e in va if e["tag"] == "ins-deploy"]
prev = json.loads((TP.OUT / "backbone_recheck.json").read_text())["TCN"]
recs = [{"pp2": 14.6, "ins": 43.2, "ppf": 11.4}]  # 占位, 下面用实跑覆盖
recs = []
for sd in range(5):
    m = TP.train_one(TCN_D, LR, sd, STEPS, tr, G, BETA, GAMMA, log_every=10**9)
    torch.save(m.state_dict(), TP.OUT / f"pi_tcn_s{sd}.pt")
    rp, ri, rf = (TP.evaluate(m, x, G) for x in (va_pp, va_ins, ppf_va))
    recs.append({"pp2": rp["red"]*100, "ins": ri["red"]*100, "ppf": rf["red"]*100,
                 "pp2_worst": rp["worst"]*100, "ins_worst": ri["worst"]*100,
                 "da": rp["da_rms_mrad"], "sat": rp["sat_pct"]})
    print(f"  [TCN s{sd}] pp2={recs[-1]['pp2']:5.1f}% ins={recs[-1]['ins']:5.1f}% "
          f"ppf={recs[-1]['ppf']:5.1f}% 最差pp2={recs[-1]['pp2_worst']:5.1f}%", flush=True)
agg = {k: [float(np.mean([x[k] for x in recs])), float(np.std([x[k] for x in recs]))]
       for k in recs[0]}
(TP.OUT / "pi_tcn5.json").write_text(json.dumps(agg, indent=1))
d = json.loads((TP.OUT / "pi_default.json").read_text())
print("\n== 5种子对5种子 (new25, β=300) ==")
for nm, v, p in [("TCN-deep", d, 34662), ("TCN", agg, 11334)]:
    se = lambda k: v[k][1] / np.sqrt(5)
    print(f"{nm:>10}: pp2={v['pp2'][0]:5.2f}±{v['pp2'][1]:.2f}(SE{se('pp2'):.2f})  "
          f"ins={v['ins'][0]:5.2f}±{v['ins'][1]:.2f}(SE{se('ins'):.2f})  "
          f"ppf={v['ppf'][0]:5.2f}  最差pp2={v['pp2_worst'][0]:5.2f}  {p}参数")
for k in ("pp2", "ins", "ppf"):
    dm = d[k][0] - agg[k][0]
    sd_ = np.sqrt((d[k][1]**2 + agg[k][1]**2) / 5)
    print(f"  {k}: TCN-deep − TCN = {dm:+.2f}pp  ({abs(dm)/max(sd_,1e-9):.2f}σ)")
