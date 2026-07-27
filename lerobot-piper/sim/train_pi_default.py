#!/usr/bin/env python3
"""定档 π（新默认, 2026-07-22 用户采纳）: 数据 new25 + β=300 + γ=0, TCN, 6000步 ×5种子。

new25 = pp2-deploy×8 + ins-deploy×8 + pp2f-deploy×6 + pp2f60-deploy×3
β=300 的理由: 因子化实验中 (new25,β=300) 相对原基线 (old16,β=1000) 使 pp2 9.5→16.1%
且种子方差 2.9→0.4; 未取模型内更优的 β=100/30 是因 ε 尺度体检测得真机 HF 增益为
Ĝ 的 1.3-1.5×, 力度越大该放大伤害越重 → 上机默认留余量, 真机 A/B 后再调。

输出: outputs/p2_pi/pi_v1_s{0..4}.pt + pi_v1.json
"""
import json

import numpy as np
import torch

import train_pi_e2 as TP
from rerace_dropout import TCN_D

BETA, GAMMA, STEPS, LR = 3e2, 0.0, 6000, 1e-3


def load_new25(G):
    tags0 = TP.TAGS
    tr = TP.load_eps(TP.TRAIN_EPS, G)
    va = TP.load_eps(TP.VAL_EPS, G)
    TP.TAGS = ["pp2f-deploy"]
    tr += TP.load_eps(list(range(6)), G)
    ppf_va = TP.load_eps([6], G)
    TP.TAGS = ["pp2f60-deploy"]
    tr += TP.load_eps([0, 1, 2], G)
    TP.TAGS = tags0
    return tr, va, ppf_va


def main():
    TP.BP = TP.make_bp()
    G = TP.GEnsemble()
    tr, va, ppf_va = load_new25(G)
    va_pp = [e for e in va if e["tag"] == "pp2-deploy"]
    va_ins = [e for e in va if e["tag"] == "ins-deploy"]
    print(f"训练集 {len(tr)} 集 / 验证 pp2×{len(va_pp)} ins×{len(va_ins)} fork×{len(ppf_va)}",
          flush=True)
    recs = []
    for sd in range(5):
        m = TP.train_one(TCN_D, LR, sd, STEPS, tr, G, BETA, GAMMA, log_every=10**9)
        torch.save(m.state_dict(), TP.OUT / f"pi_v1_s{sd}.pt")
        rp, ri, rf = (TP.evaluate(m, x, G) for x in (va_pp, va_ins, ppf_va))
        r = {"pp2": rp["red"] * 100, "ins": ri["red"] * 100, "ppf": rf["red"] * 100,
             "pp2_worst": rp["worst"] * 100, "ins_worst": ri["worst"] * 100,
             "da": rp["da_rms_mrad"], "sat": rp["sat_pct"]}
        recs.append(r)
        print(f"  s{sd}: pp2={r['pp2']:5.1f}% ins={r['ins']:5.1f}% ppf={r['ppf']:5.1f}% "
              f"‖δa‖={r['da']:.2f} sat={r['sat']:.0f}%", flush=True)
    agg = {k: [float(np.mean([x[k] for x in recs])), float(np.std([x[k] for x in recs]))]
           for k in recs[0]}
    agg["_config"] = {"data": "new25", "beta": BETA, "gamma": GAMMA,
                      "backbone": "TCN(dropout0.1)", "steps": STEPS, "lr": LR,
                      "n_train_eps": len(tr)}
    (TP.OUT / "pi_v1.json").write_text(json.dumps(agg, indent=1))
    print(f"\n定档: pp2={agg['pp2'][0]:.1f}±{agg['pp2'][1]:.1f}%  "
          f"ins={agg['ins'][0]:.1f}±{agg['ins'][1]:.1f}%  ppf={agg['ppf'][0]:.1f}%  "
          f"最差成员 pp2={agg['pp2_worst'][0]:.1f}/ins={agg['ins_worst'][0]:.1f}")


if __name__ == "__main__":
    main()
