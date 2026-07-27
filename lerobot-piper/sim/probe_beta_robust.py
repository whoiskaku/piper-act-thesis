#!/usr/bin/env python3
"""加大修正是否降低对 Ĝ 增益误差的容忍度? β∈{300,100,30} 各测鲁棒性曲线。"""
import numpy as np, torch
import train_pi_e2 as TP
from rerace_dropout import TCN_D
from train_pi_default import GAMMA, LR, STEPS, load_new25

TP.BP = TP.make_bp(); G = TP.GEnsemble()
tr, va, ppf = load_new25(G)
grp = {"pp2": [e for e in va if e["tag"]=="pp2-deploy"],
       "ins": [e for e in va if e["tag"]=="ins-deploy"]}
gs = np.array([1.0, 1.3, 1.5, 1.8, 2.0, 2.3, 2.6, 3.0])

def curve(models):
    out = {k: np.zeros(len(gs)) for k in grp}
    cnt = 0
    for m in models:
        for name, eps in grp.items():
            for ep in eps:
                n = len(ep["e"]); msk = ep["w"] > 0
                with torch.no_grad():
                    de = TP.apply_G(m(ep["X"].unsqueeze(0)), ep["epJ"], ep["alpha"],
                                    ep["RB"], ep["RD"], G)
                    b0 = TP.BP(ep["e"].view(1,1,n,3))[0,0]
                    den = (b0[msk]**2).sum()
                    for gi, g in enumerate(gs):
                        b1 = TP.BP(ep["e"].unsqueeze(0).unsqueeze(1) + float(g)*de)[0]
                        out[name][gi] += float(1 - (b1[:,msk]**2).sum((1,2)).mean()/den)*100
        cnt += 1
    return {k: v/(cnt*2) for k, v in out.items()}

print(f"{'β':>5}{'‖δa‖':>7}" + "".join(f"{g:>7.1f}" for g in gs) + "   零点g")
for beta, seeds in [(3e2, None), (1e2, (0,1)), (3e1, (0,1))]:
    if seeds is None:
        ms = []
        for sd in range(5):
            m = TCN_D(d=TP.FEAT).to(TP.DEV)
            m.load_state_dict(torch.load(TP.OUT/f"pi_v1_s{sd}.pt", map_location=TP.DEV))
            m.eval(); ms.append(m)
    else:
        ms = [TP.train_one(TCN_D, LR, sd, STEPS, tr, G, beta, GAMMA, log_every=10**9)
              for sd in seeds]
    da = np.mean([TP.evaluate(m, grp["pp2"], G)["da_rms_mrad"] for m in ms])
    c = curve(ms)
    for name in ("pp2", "ins"):
        y = c[name]
        z = np.nan
        for i in range(len(gs)-1):
            if y[i] > 0 >= y[i+1]:
                z = gs[i] + (gs[i+1]-gs[i]) * y[i]/(y[i]-y[i+1]); break
        lbl = f"{beta:.0f}" if name == "pp2" else ""
        d = f"{da:.2f}" if name == "pp2" else ""
        print(f"{lbl:>5}{d:>7}" + "".join(f"{v:>7.1f}" for v in y) +
              f"   {z:.2f}  ({name})")
