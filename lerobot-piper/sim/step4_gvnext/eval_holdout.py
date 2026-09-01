#!/usr/bin/env python3
"""**未见轨迹 + 跨会话留出**上的正式评价（`PREREG_G_VNEXT.md` §6）。

留出 = 会话2 的 pp2f(5) + pp2f60(3)，轨迹未参与训练、会话也未参与训练。
r_q 用训练族折外选出的 FIR（nlag/lam 已由 `rq/select.json` 定死）。
"""
import sys, json, numpy as np
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
for p in (ROOT, ROOT/"sim", ROOT/"sim/step2_g", ROOT/"sim/step3_canobs", HERE): sys.path.insert(0,str(p))
from data import load, to20, bp, q_at, best_fit, score, valid_files, FPS
from train_rq import features, fir_design, band_err, MOVING, BAND
from ceiling_200hz import kin_gyro_fs
import gc; gc.disable()

S2=ROOT/"outputs/step4_gvnext/s2"
sel=json.loads((ROOT/"outputs/step4_gvnext/rq/select.json").read_text())
W=np.load(ROOT/"outputs/step4_gvnext/rq/rq_fir.npz")["W"]
nlag=sel["nlag"]
HOLD=valid_files(S2, "replay_pp2f*.json")
print(f"留出 {len(HOLD)} 条（会话2, 未见轨迹）  r_q: FIR nlag={nlag} lam={sel['lam']:.0e}\n")

# ── 判据 4/5/6: r_q 的关节层 ──────────────────────────────────────────
n39=d39=nb=db=0.0; pj={j:[0.,0.] for j in MOVING}; rms=[0.,0.]
EPS=[]
for f in HOLD:
    dd=load(f); X,Y,t20=features(dd)
    P=fir_design(X,nlag)@W
    EPS.append((dd,X,Y,P,t20))
    a,b=band_err(Y,P,*BAND); n39+=a; d39+=b
    a,b=band_err(Y,P,0.5,9.0); nb+=a; db+=b
    A,B=bp(Y,*BAND),bp(P,*BAND)
    for j in MOVING:
        pj[j][0]+=float(((A-B)[:,j]**2).sum()); pj[j][1]+=float((A[:,j]**2).sum())
    rms[0]+=float((Y[:,MOVING]**2).sum()); rms[1]+=float(((Y-P)[:,MOVING]**2).sum())
r39=1-n39/max(d39,1e-30)
print(f"  判据4  r_q 留出 3–9 Hz 关节误差下降  {r39*100:>7.1f}%   门槛 ≥20%   "
      f"{'✓' if r39>=0.20 else '**✗**'}")
imp=[j for j in MOVING if 1-pj[j][0]/max(pj[j][1],1e-30) > 0]
print(f"  判据5  逐关节改善 {len(imp)}/5 " + " ".join(
    f"j{j+1}={100*(1-pj[j][0]/max(pj[j][1],1e-30)):+.1f}%" for j in MOVING)
    + f"   门槛 ≥4/5   {'✓' if len(imp)>=4 else '**✗**'}")
dr=np.sqrt(rms[1]/max(rms[0],1e-30))-1
print(f"  判据6  宽带关节 RMS 变化 {dr*100:>+7.1f}%   门槛 ≤+5%   "
      f"{'✓' if dr<=0.05 else '**✗**'}")

# ── 判据 1/2: 腕部 IMU（G-q-v1 单独 vs +r_q）──────────────────────────
print(f"\n  {'':<26}{'0.5–3 Hz ρ²':>14}{'3–9 Hz ρ²':>12}")
acc={"base":{}, "rq":{}, "ceil":{}}
for k in acc: acc[k]={(0.5,3.0):[], (3.0,9.0):[]}
for dd,X,Y,P,t20 in EPS:
    import g_q_v1 as GQ
    qh_f=np.radians(GQ.predict_qpos(dd["cmd"],t_send=dd["t_send"],t_read=dd["t_read"]))
    qh=np.stack([np.interp(t20,dd["t_read"],qh_f[:,j]) for j in range(6)],1)
    A20=to20(dd["gyro100"])[:len(t20)]
    qt=q_at(dd,t20)
    for tag,q in (("base",qh),("rq",qh+P),("ceil",qt)):
        S=kin_gyro_fs(q,FPS)
        for b in ((0.5,3.0),(3.0,9.0)):
            acc[tag][b].append(best_fit(A20,S,*b)[1])
FRZ={(0.5,3.0):0.789,(3.0,9.0):0.707}
for tag,lbl in (("base","G-q-v1 单独"),("rq","G-q-v1 + r_q"),("ceil","天花板(真关节)")):
    print(f"  {lbl:<26}" + "".join(f"{np.median(acc[tag][b]):>14.3f}" for b in ((0.5,3.0),(3.0,9.0))))
print(f"  {'冻结门槛':<26}" + "".join(f"{FRZ[b]:>14.3f}" for b in ((0.5,3.0),(3.0,9.0))))
for b in ((0.5,3.0),(3.0,9.0)):
    v=np.median(acc["rq"][b])
    print(f"  判据{1 if b[0]<1 else 2}  {b[0]}–{b[1]} Hz  {v:.3f} vs {FRZ[b]:.3f}   "
          f"{'✓' if v>=FRZ[b] else '**✗**'}")
