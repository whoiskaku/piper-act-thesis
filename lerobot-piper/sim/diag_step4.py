#!/usr/bin/env python3
"""诊断第4步(用户设计) — 同一批 A 日志, 变回灌响应模型做 autoregressive 展开。
控制了 rollout 差异(物体位置/ACT随机性), 只变"π 看到的反馈"。

响应模型:
  ①原始Ĝ  ②1.4×Ĝ(真机总增益)  ③探针校准Ĝ(逐关节g, 含j3×2.27)  ④K=5悲观成员
判据: 只有 ③探针校准 的 AR 把 TF 的~4.7% 压到接近 C 真机的 0.3%,
      才能确认"Ĝ反馈失配"是第二根因。否则 A→C 下降归因于自然 rollout 差异。
"""
import json, numpy as np, torch
from pathlib import Path
import train_pi_e2 as TP
from compare_ideal_methods import load_episode
from lerobot.common.policies.act.b_runtime import (FeatureAssembler, IdealChunkRenderer,
                                                   load_tcn, CHUNK, WIN, FEAT_DIM)
ROOT=Path(__file__).resolve().parents[1]
TP.BP=TP.make_bp(); G=TP.GEnsemble()
man=json.loads((ROOT/"outputs/p2_labels/pp2-deploy/manifest.json").read_text())
R=np.array(man["R"],np.float32); b=np.array(man["b"],np.float32); MD=np.pi/180/1000
pi=load_tcn(ROOT/"outputs/p2_pi/pi_v1_s0.pt").to(TP.DEV); rend=IdealChunkRenderer()
FAIL={6}; L=3
# 探针逐关节增益(v2: j2/j3; v1: j5; 未测 j1/j4/j6 用总体1.40)
GJ=np.array([1.40, 1.108, 2.271, 1.40, 1.326, 1.40], np.float32)

def energy(x):
    n=len(x); return float((TP.BP(torch.from_numpy(np.ascontiguousarray(x,np.float32)).view(1,1,n,3).to(TP.DEV))[0,0]**2).sum())

def prep(cmd7,imu,st):
    T=min(len(cmd7),len(imu),len(st)); cmd7,imu,st=cmd7[:T],imu[:T],st[:T]
    asm=FeatureAssembler(R,b); prev=None; ci=0; starts=list(range(0,T,CHUNK))+[T]; X=np.zeros((T,FEAT_DIM),np.float32)
    for t in range(T):
        if ci<len(starts)-1 and t==starts[ci]:
            seg=cmd7[starts[ci]:starts[ci+1]]; asm.on_new_chunk(seg,rend.render(seg,prev)); prev=seg; ci+=1
        w,_=asm.step(imu[t],(st[t,:6]*MD).astype(np.float32)); X[t]=w[-1]
    ideal_g=np.array(asm.ideal)[:,3:]; n=T-L
    e=(imu[L:L+n,3:6]-ideal_g[:n]).astype(np.float32)
    epj,al,RB,RD=G.ep_context(cmd7[:n,:6],cmd7[:n,6],R)
    return X[:n],e,epj.unsqueeze(0),al.unsqueeze(0),RB.unsqueeze(0),RD.unsqueeze(0)

def resp_full(da, ctx, model, worst_k=None):
    """da(1,n,6) → 响应(n,3), 按 model 变换。"""
    epj,al,RB,RD=ctx
    de=TP.apply_G(da, epj,al,RB,RD, G)[0]                 # (K,n,3)
    if model=="pess": return de[worst_k].cpu().numpy()
    out=de.mean(0).cpu().numpy()
    if model=="1.4x": out*=1.4
    return out

def ar_expand(X,e,ctx,model,worst_k=None):
    n=len(e); Xar=X.copy(); esim=e.copy(); de_cum=np.zeros((n,3),np.float32)
    for t in range(n):
        if t-L>=0: Xar[t,3:6]=esim[t-L]
        lo=max(0,t-WIN+1); win=Xar[lo:t+1]
        pad=np.zeros((WIN,FEAT_DIM),np.float32); pad[-len(win):]=win
        with torch.no_grad(): da_t=pi(torch.from_numpy(pad).unsqueeze(0).to(TP.DEV))[0,-1].cpu().numpy()
        dd=np.zeros((1,n,6),np.float32)
        dd[0,t]= (da_t*GJ) if model=="calib" else da_t
        contrib=resp_full(torch.from_numpy(dd).to(TP.DEV),ctx,"raw" if model=="calib" else model,worst_k)
        esim=e+de_cum+contrib; de_cum=de_cum+contrib
    return (1-energy(esim)/energy(e))*100

MODELS=["raw","1.4x","calib","pess"]
res={m:[] for m in MODELS}; tf_res={m:[] for m in MODELS}
for ep in range(10):
    if ep in FAIL: continue
    d=load_episode("enolab-guo/eval_ab_A",ep)
    X,e,epj,al,RB,RD=prep(d["action"].astype(np.float32),d["imu_real"].astype(np.float32),d["state_mdeg"].astype(np.float32))
    ctx=(epj,al,RB,RD); n=len(e); e0=energy(e)
    # 一次性 TF: π 看原始X
    with torch.no_grad(): da_tf=pi(torch.from_numpy(X).unsqueeze(0).to(TP.DEV))
    de_K=TP.apply_G(da_tf,epj,al,RB,RD,G)[0]              # (K,n,3)
    worst_k=int(np.argmax([energy(e+de_K[k].cpu().numpy()) for k in range(de_K.shape[0])]))
    for m in MODELS:
        if m=="calib":
            dd=(da_tf.cpu().numpy()*GJ); de_tf=TP.apply_G(torch.from_numpy(dd).to(TP.DEV),epj,al,RB,RD,G)[0].mean(0).cpu().numpy()
        elif m=="1.4x": de_tf=de_K.mean(0).cpu().numpy()*1.4
        elif m=="pess": de_tf=de_K[worst_k].cpu().numpy()
        else: de_tf=de_K.mean(0).cpu().numpy()
        tf_res[m].append((1-energy(e+de_tf)/e0)*100)
        res[m].append(ar_expand(X,e,ctx,m,worst_k))
    print(f"A-ep{ep} done", flush=True)
print(f"\n{'响应模型':>12}{'TF':>9}{'AR':>9}{'AR相对TF':>10}")
for m,lbl in [("raw","①原始Ĝ"),("1.4x","②1.4×Ĝ"),("calib","③探针校准"),("pess","④悲观成员")]:
    tf,ar=np.mean(tf_res[m]),np.mean(res[m])
    print(f"{lbl:>12}{tf:>8.1f}%{ar:>8.1f}%{ar-tf:>+9.1f}pp")
print(f"\n对照: C 真机闭环 0.3%")
print("判据: ③探针校准 AR 接近 0.3% → Ĝ反馈失配是次因; 否则 A→C下降=自然rollout差异")
