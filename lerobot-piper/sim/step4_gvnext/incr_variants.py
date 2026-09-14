#!/usr/bin/env python3
"""**G 的增量通道：三个变体 vs 实测 H**（零机时）——「Ĝ 只需增量准」的直接检验。

背景
----
G-2 的 0.5-3Hz rho^2=0.836 / 3-9Hz 0.520 是**绝对**保真度，
对 `delta_a -> IMU` 的**增量**响应什么都没说。而老师的判据正是「Ĝ 只需增量准」。
旧 Ĝ 训出的 pi 真机失败时，实测就显示其增量方向是错的。

做法：在 G 上做**外生 ± 配对**数值扰动（探针外部生成、不读任何实测量），
取奇分量（任务基线是偶的，精确抵消），与实测 H 的响应对标。

    Delta_var = [w(u+da) - w(u-da)] / 2
    Delta_H   = causal_conv(H_body, da)
    cos = <Delta_H, Delta_var> / (|Delta_H||Delta_var|)     门槛 > 0.6 (PROTOCOL 3a 2)

三个变体逐层剥掉学出来的残差层：
    A  全 G-2      = kin(q_base + r_q) + r_s
    B  冻结 r_s    = kin(q_base + r_q)
    C  纯物理底座  = kin(q_base)          <- G-q-v1 的增量

坐标系（**必须夹住报**）
------------------------
H 在 ab4 真机传感器系；G 的 w 在机体系。冻结标定 R 来自 **另一个会话**
(pp2a-deploy)，而 step4 数据绑带重装过、无标定 => 冻结 R 很可能是错的旋转。
故同时报：
    cos  = 用冻结 R      -> **下界**（旋转可能错）
    cos* = 逐集拟合正交旋转 -> **上界**（旋转是自由参数，取最优）
真值夹在两者之间。位移固定 +3 帧（H 的定义是 real[t+3] - ideal[t]）。

⚠ 这是**离线对标**（G 的增量 vs H 的增量），H 本身是全局 LTI 算子，
   **不等于** PROTOCOL 3a 的 A1/B/A2 真机直测。

用法:
    python -u sim/step4_gvnext/incr_variants.py
"""
import gc; gc.disable()
import hashlib, json, sys
import numpy as np
for p in (".","sim","sim/step2_g","sim/step3_canobs","sim/step4_gvnext"): sys.path.insert(0,p)
from data import load, bp, valid_files, FPS
from train_rq import features as rq_features, fir_design
from train_rs import rs_features
from eval_gate import base_q
from ceiling_200hz import kin_gyro_fs
from fit_step1_candidate import causal_conv
from pathlib import Path

ROOT=Path("."); AMP=0.003; ALIGN_L=3
FREQS=np.array([0.7,1.3,2.1,3.4,5.2,7.1,8.6])
H_ART=ROOT/"outputs/step1_H/step1_H_v1.npz"
S1,S2=ROOT/"outputs/step4_gvnext/s1",ROOT/"outputs/step4_gvnext/s2"
RQ,RS=ROOT/"outputs/step4_gvnext/rq",ROOT/"outputs/step4_gvnext/rs"

def schroeder(fr,n,fps,seed):
    rng=np.random.default_rng(seed); k=np.arange(len(fr))
    ph=-np.pi*k*(k-1)/len(fr)+rng.uniform(0,2*np.pi); t=np.arange(n)/fps
    s=np.sum(np.cos(2*np.pi*fr[:,None]*t[None,:]+ph[:,None]),0)
    return s/max(np.abs(s).max(),1e-12)
def probe(n,seed): return np.stack([AMP*schroeder(FREQS,n,FPS,seed*97+j) for j in range(6)],1)
def perturb(dd,da):
    c=np.array(dd["cmd"],float,copy=True); m=min(len(c),len(da)); c[:m,:6]+=da[:m]
    return {**dd,"cmd":c}
def rotcos(A,B,shift=ALIGN_L):
    from scipy.linalg import orthogonal_procrustes
    a,b=(A[:len(A)-shift or None],B[shift:]) if shift>=0 else (A[-shift:],B[:len(B)+shift])
    n=min(len(a),len(b)); a,b=a[:n],b[:n]
    R_,_=orthogonal_procrustes(b,a); br=b@R_
    return (float((a*br).sum()/max(np.linalg.norm(a)*np.linalg.norm(br),1e-30)),
            float(np.degrees(np.arccos(np.clip((np.trace(R_)-1)/2,-1,1)))))
def compare(A,B,shift=ALIGN_L):
    a,b=(A[:len(A)-shift or None],B[shift:]) if shift>=0 else (A[-shift:],B[:len(B)+shift])
    n=min(len(a),len(b)); a,b=a[:n],b[:n]; dot=float((a*b).sum())
    return (dot/max(float((b*b).sum()),1e-30),
            dot/max(float(np.linalg.norm(a)*np.linalg.norm(b)),1e-30))

H=np.ascontiguousarray(np.load(H_ART)["H"],float)
meta=json.loads(H_ART.with_suffix(".json").read_text())
assert hashlib.sha256(H.tobytes()).hexdigest()==meta["H"]["sha256"]
R=np.asarray(meta["calibration"]["R"],float)
H_body=np.einsum("lak,ab->lbk",H,R)
zq,zs=np.load(RQ/"rq_fir.npz"),np.load(RS/"rs_fir.npz")
sq,ss=(json.loads((p/"select.json").read_text()) for p in (RQ,RS))

def omegas(dd,t20,ddp):
    qb=base_q(ddp,t20); qh=qb+fir_design(rq_features(ddp)[0],sq["nlag"])@zq["W"]
    n=len(t20); rs=fir_design(rs_features(ddp,t20,qh)[:n],ss["nlag"])@zs["W"]
    return {"A 全 G-2 = kin(q̂)+r_s":kin_gyro_fs(qh,FPS)[:n]+rs,
            "B 冻结 r_s = kin(q̂)":  kin_gyro_fs(qh,FPS)[:n],
            "C 再冻结 r_q = kin(q_base)":kin_gyro_fs(qb,FPS)[:n]}
VAR=["A 全 G-2 = kin(q̂)+r_s","B 冻结 r_s = kin(q̂)","C 再冻结 r_q = kin(q_base)"]
BANDS=((0.5,3.0),(3.0,9.0),(0.5,9.0))
cosr={v:{b:[] for b in BANDS} for v in VAR}; ang={v:[] for v in VAR}
cos={v:{b:[] for b in BANDS} for v in VAR}; mag={v:{b:[0.,0.] for b in BANDS} for v in VAR}
files=list(valid_files(S1))+list(valid_files(S2,"replay_pp2f*.json"))
for f in files:
    dd=load(f); t20=dd["tg_i"][::5]; n=len(t20)
    seed=int.from_bytes(hashlib.sha256(dd["name"].encode()).digest()[:4],"little")
    da20=probe(n,seed); tr=np.asarray(dd["t_read"],float)
    dac=np.stack([np.interp(tr,t20,da20[:,j]) for j in range(6)],1)
    daa=np.stack([np.interp(t20,tr,dac[:,j]) for j in range(6)],1)
    wp=omegas(dd,t20,perturb(dd,dac)); wm=omegas(dd,t20,perturb(dd,-dac))
    m=min(len(wp[VAR[0]]),n); d_h=causal_conv(H_body,daa[:m])
    for v in VAR:
        d=0.5*(wp[v][:m]-wm[v][:m])
        for b in BANDS:
            a,bb=bp(d_h,*b),bp(d,*b); cos[v][b].append(compare(a,bb)[1])
            rc,ag=rotcos(a,bb); cosr[v][b].append(rc)
            if b==(0.5,9.0): ang[v].append(ag)
            mag[v][b][0]+=float((bb**2).sum()); mag[v][b][1]+=float((a**2).sum())
print(f"{len(files)} 集（训练族 17 + 开发验证集 8）· 冻结 R · 固定 +3 帧 · 门槛 cos>0.6\n")
hdr="".join(f"{'cos '+f'{b[0]}-{b[1]}Hz':>15}" for b in BANDS)
print(f"  {'增量变体':<28}{hdr}{'幅值比(0.5-9)':>15}")
for v in VAR:
    r=np.sqrt(mag[v][(0.5,9.0)][0]/max(mag[v][(0.5,9.0)][1],1e-30))
    print(f"  {v:<28}"+"".join(f"{np.median(cos[v][b]):>15.3f}" for b in BANDS)+f"{r:>14.1f}x")
print(f"\n  逐变体的**低频**幅值比（0.5-3 Hz）：")
for v in VAR:
    r=np.sqrt(mag[v][(0.5,3.0)][0]/max(mag[v][(0.5,3.0)][1],1e-30))
    print(f"    {v:<28}{r:>8.2f}x")
print(f"\n  ── 逐集拟合旋转（**上界**，自由参数取最优）──")
print(f"\n  ── 逐集拟合旋转（**上界**：旋转是自由参数，取最优）──")
hdr2="".join(f"{'cos* '+f'{b[0]}-{b[1]}Hz':>15}" for b in BANDS)
print(f"  {'增量变体':<28}{hdr2}{'旋转角°中位':>14}")
for v in VAR:
    print(f"  {v:<28}"+"".join(f"{np.median(cosr[v][b]):>15.3f}" for b in BANDS)+f"{np.median(ang[v]):>13.1f}")
print("\n  幅值比 = ‖Δ_变体‖/‖Δ_H‖，1.0 理想；>1 高估。cos* 是上界，真值夹在 cos 与 cos* 之间。")
sys.stdout.flush(); import os; os._exit(0)
