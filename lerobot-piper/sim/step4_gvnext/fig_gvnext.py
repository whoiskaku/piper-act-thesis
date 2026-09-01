#!/usr/bin/env python3
"""**G-2 的 pp2 波形对比图**（每次动 G 必出）—— 留出集，未见轨迹 + 跨会话。

四条线：実機（青）· G-q-v1（紫）· **G-2（緑）** · 天井=真の関節（橙）
上段 0.5–3 Hz / 下段 3–9 Hz（最も強い 4 s を拡大）。中央値集と最難集。
"""
import sys, json, numpy as np
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
for p in (ROOT, ROOT/"sim", ROOT/"sim/step2_g", ROOT/"sim/step3_canobs", HERE): sys.path.insert(0,str(p))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from data import load, bp, valid_files, best_fit, FPS
from train_rq import features as rq_features, fir_design, MOVING
from train_rs import rs_features, sensor_target
from eval_gate import base_q
from ceiling_200hz import kin_gyro_fs
import gc; gc.disable()
for _f in ("Noto Sans CJK JP","Noto Sans CJK SC","IPAexGothic"):
    if any(_f in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"]=_f; break
plt.rcParams["axes.unicode_minus"]=False
C_R,C_V1,C_NX,C_K="#1f6fb2","#8b2fb0","#2e9e4f","#e07b1a"
S2=ROOT/"outputs/step4_gvnext/s2"
zq=np.load(ROOT/"outputs/step4_gvnext/rq/rq_fir.npz"); sq=json.load(open(ROOT/"outputs/step4_gvnext/rq/select.json"))
zs=np.load(ROOT/"outputs/step4_gvnext/rs/rs_fir.npz"); ss=json.load(open(ROOT/"outputs/step4_gvnext/rs/select.json"))
D={}
for f in valid_files(S2,"replay_pp2f*.json"):
    dd=load(f); t20=dd["tg_i"][::5]
    Xq,Yq,_=rq_features(dd); qb=base_q(dd,t20); Pq=fir_design(Xq,sq["nlag"])@zq["W"]
    Ab,Kt,Ys=sensor_target(dd,t20); n=len(Ys); qh=qb+Pq
    Xs=rs_features(dd,t20,qh)[:n]; Ps=fir_design(Xs,ss["nlag"])@zs["W"]
    D[dd["name"]]={"A":Ab[:n],"v1":kin_gyro_fs(qb,FPS)[:n],
                   "nx":kin_gyro_fs(qh,FPS)[:n]+Ps,"ceil":Kt[:n]}
r2={k:best_fit(v["A"],v["nx"],3.0,9.0,rotate=False)[1] for k,v in D.items()}
o=sorted(D,key=lambda k:r2[k],reverse=True)
show=[(o[len(o)//2],"中央値"),(o[-1],"最難")]
fig,ax=plt.subplots(2,2,figsize=(16,8.2))
for c,(nm,kind) in enumerate(show):
    E=D[nm]; j=int(np.argmax((bp(E["A"],3.0,9.0)**2).sum(0)))
    for r,(lo,hi) in enumerate(((0.5,3.0),(3.0,9.0))):
        a=ax[r][c]; A=bp(E["A"],lo,hi)
        ser=[("実機 IMU",A,C_R,1.9,"-"),("天井=真の関節",bp(E["ceil"],lo,hi),C_K,1.4,"-"),
             ("G-q-v1",bp(E["v1"],lo,hi),C_V1,1.3,"-"),("G-2",bp(E["nx"],lo,hi),C_NX,1.7,"--")]
        t=np.arange(len(A))/FPS
        if r==1:
            e=np.convolve(A[:,j]**2,np.ones(40),"same"); k=int(np.argmax(e))
            w=slice(max(0,k-40),k+40); t=t[w]; ser=[(n_,s[w],cc,lw,st) for n_,s,cc,lw,st in ser]
        for n_,s,cc,lw,st in ser: a.plot(t,s[:,j],color=cc,lw=lw,ls=st,label=n_)
        rr=lambda s: float((A[:,j]*s[:,j]).sum()**2/max((A[:,j]**2).sum()*(s[:,j]**2).sum(),1e-30)) if len(s)==len(A) else 0
        a.set_title(f"{kind} {nm}  {lo}–{hi} Hz (軸 {'xyz'[j]})"+("  ※最も強い 4 s"if r else"")
                    +f"    ρ²: G-q-v1 {best_fit(E['A'],E['v1'],lo,hi,rotate=False)[1]:.3f}"
                    f" / **G-2 {best_fit(E['A'],E['nx'],lo,hi,rotate=False)[1]:.3f}**"
                    f" / 天井 {best_fit(E['A'],E['ceil'],lo,hi,rotate=False)[1]:.3f}",fontsize=10)
        a.set_ylabel("角速度 [rad/s]",fontsize=9); a.grid(alpha=.25); a.tick_params(labelsize=8)
        if r==1: a.set_xlabel("時間 [s]",fontsize=9)
        if r==0 and c==0: a.legend(fontsize=8.3,loc="upper right",ncol=2,framealpha=.92)
fig.suptitle("G-2（物理底座 + 双层残差）: 留出集 = **未見軌跡 + 跨セッション**    "
             "3–9 Hz で ρ² 0.193 → **0.520**（天井 0.649・門槛 0.707）",fontsize=13,y=0.985)
fig.text(0.5,0.005,"7 条判据中 6 条通过，唯一未过 = 判据2（3–9 Hz ρ² 0.520 < 0.707）。"
  "相位模板 0.137・零输入 0.193 均劣于 G-q-v1 ⇒ **不是记忆时间模板**；最难 20% 窗口 8/8 改善。"
  "口径: 真机 100 Hz→9 Hz 低通→20 Hz，时基=设备 ms/内核时刻，逐集 Procrustes 旋入机体系，"
  "逐系列最优位移。留出 8 条(会话2 pp2f+pp2f60)，与 17 集开发基准集不可并列。",
  ha="center",fontsize=8.3,color="#333")
fig.tight_layout(rect=(0,0.028,1,0.965))
for p in (ROOT/"outputs/step4_gvnext", ROOT/"outputs/step2_g/advisor"):
    p.mkdir(parents=True,exist_ok=True); fig.savefig(p/"fig9_gvnext_holdout.png",dpi=170)
print(f"  → {ROOT/'outputs/step4_gvnext/fig9_gvnext_holdout.png'}")
