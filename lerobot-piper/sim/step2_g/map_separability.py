#!/usr/bin/env python3
"""kp↔c 可分性**图** —— 局部灵敏度筛选不能锚在单点。

为什么需要这张图: 在基线(手调)参数上, kp 与 c 对 j1/j4/j6 都共线到 |cos|>0.99。
我起初判断这是「kp 手调得太高、共振跑出激励带」的假象, 并预测降低 kp 即可分开。
**该预测只对 j4 成立**(0.990→0.101), 对 j1/j6 反而更糟(0.993→0.998)。

真正的机理是**过阻尼**: 现模型 c1=500、c6=5 使两个极点一个远在带下(~0.02 Hz)、
一个远在带上(117–1894 Hz), 带内响应只由 **kp/c 的比值**决定 ⇒ 二者只以商出现。
而 `c1 = 500 N·m·s/rad` 几乎肯定不物理(1 rad/s 要 500 N·m, 对负载 1.5 kg 的臂
太离谱) ⇒ **基线不只是 kp 错, c 也错**, 在这种点上判「可辨识性」不可靠。

本图在 (kp, c) 平面上扫, 结论: **可分性由 c 控制, 不由 kp**。
c 降到物理合理量级即干净分开(j1 0.098 / j4 0.009); j6 全网格最低仅 0.677
—— 它惯量太小(0.00042), 要 ζ≈1 需 c6≈0.03, 已在网格外, 须单独处理。

⇒ **激励设计无需改动**; 需要改的是「不要在错误起点上下可辨识性结论」。
   正确做法: 拟合后在**拟合到的 θ** 上重跑筛选, 并报告逐参数的置信度而非点估计。

用法: python sim/step2_g/map_separability.py
"""
import sys
from pathlib import Path
import numpy as np, mujoco
ROOT=Path('/home/enolab-guo/claude-test-2/lerobot-piper'); sys.path.insert(0,str(ROOT))
from sim.step2_g import params as PA
from sim.step2_g.build_suite import build, simulate, MODEL
m=mujoco.MjModel.from_xml_path(str(MODEL))
allq=build(m); th0=PA.baseline(m)
KP=np.array([5000.,200.,200.,100.,50.,50.]); C0=np.array([500.,5.,20.,5.,5.,5.])
print("kp↔c 可分性图 —— 在(kp,c)平面上扫，看是否存在能分开的区域")
print("(|cos| 越小越好; <0.9 视为可分)\n",flush=True)
for j,nm in ((0,"j1"),(5,"j6"),(3,"j4")):
    seqs=[s for s in allq if s.target[0]==j][::2]
    print(f"── {nm}  ({len(seqs)} 条序列) ──")
    hdr="        " + "".join(f"{f'c×{cs:g}':>9}" for cs in (0.02,0.1,0.5,1.0))
    print(hdr,flush=True)
    for ks in (0.01,0.1,1.0):
        row=[]
        for cs in (0.02,0.1,0.5,1.0):
            th=th0.copy(); th[j]=KP[j]*ks; th[6+j]=C0[j]*cs
            st=PA.perturb_scale(th); S={}
            for s in seqs:
                b0=simulate(m,s.cmd,th)
                for i in (j,6+j):
                    t2=th.copy(); t2[i]+=st[i]
                    S.setdefault(i,[]).append((simulate(m,s.cmd,t2)-b0).ravel())
            V={i:np.concatenate(v) for i,v in S.items()}
            V={i:v/max(np.linalg.norm(v),1e-30) for i,v in V.items()}
            row.append(abs(float(V[j]@V[6+j])))
        print(f"kp×{ks:<5g}" + "".join(f"{c:>9.3f}" for c in row),flush=True)
    print(flush=True)
