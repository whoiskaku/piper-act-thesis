#!/usr/bin/env python3
"""Ĝ 结构审计（2026-07-28）—— 位形依赖本该从哪里进来，为什么没进来。

B′ 三轮测出: 真机 6-8Hz 响应在 w170/w280 之间差 3.92×, 而 Ĝ 预测只差 0.80×。
但 Ĝ **结构上本来就有位形调度**:

    α(t) = softplus(φ(t) @ θᵀ)          φ = [1, q2, q3, q2², q3², q2·q3, 速度, 夹爪]
    模态激励 = (W·u) * α(t)              ← α 乘在**输入侧**(进振子之前)
    输出     = 模态铃声 @ B  +  u @ Dᵀ    ← D 是**直通项, 位形无关**

所以这不必然是结构缺陷。四个候选:
  (a) θ 没学到 —— 非常数列停在 0 附近 ⇒ α 实际是常数(辨识失败, 不是结构失败)
  (b) θ 学到了, 但两个位形上 α 差得不够(≈0.8× 而非 3.92×)
  (c) α 只调制模态路径, 而 6-8Hz 上**位形无关的 D 项主导** ⇒ 调度对该频段无效
  (d) 模态频率都远低于 6-8Hz ⇒ 该频段模态已滚降, 同 (c)

本审计逐项量化, 判定该重辨识(a/b) 还是该改结构(c/d)。

用法: python sim/audit_G_structure.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch

import analyze_probe_b2 as A
import train_pi_e2 as TP
from g_model import Rigid, load_ensemble, pose_features, modal_impulse
from make_probe_b2 import NEXC

ROOT = Path(__file__).resolve().parents[1]
FEAT = ["1", "q2", "q3", "q2²", "q3²", "q2·q3", "速度", "夹爪"]
WINS = {"w170": 170, "w280": 280}
FREQS = np.array([6.0, 6.5, 7.0, 7.5])


def main():
    rig = Rigid()
    models = load_ensemble(rigid=rig)
    K = len(models)
    man = json.loads((TP.LROOT / "pp2-deploy" / "manifest.json").read_text())
    R = np.array(man["R"], np.float32)
    z = np.load(TP.LROOT / "pp2-deploy" / "ep000.npz")
    a_arm = z["a_arm"].astype(np.float32)
    grip = z["grip"].astype(np.float32)
    cmd7 = np.concatenate([a_arm, grip[:, None]], 1)

    # ---------- ① 模态频率与阻尼 ----------
    print("=== ① 辨识出的模态 (K=5 成员) ===")
    print(f"  {'成员':>6}" + "".join(f"{f'模态{i+1}':>22}" for i in range(3)))
    fs = []
    for k, g in enumerate(models):
        om = torch.exp(g.flex.log_om).detach().numpy() / (2 * np.pi)
        ze = (torch.sigmoid(g.flex.raw_ze) * 0.5).detach().numpy()
        fs.append(om)
        print(f"  {k:>6}" + "".join(f"{f'{o:.2f}Hz ζ={zz:.3f}':>22}" for o, zz in zip(om, ze)))
    fs = np.array(fs)
    print(f"  模态频率范围 {fs.min():.2f} – {fs.max():.2f} Hz;  "
          f"6–7.5Hz 内的模态数 = {int(((fs >= 6) & (fs <= 7.5)).sum())} / {fs.size}")
    if not ((fs >= 6) & (fs <= 7.5)).any():
        print(f"  ⚠ **没有任何模态落在 6–7.5Hz** ⇒ 该频段模态项已滚降 (候选 d)")

    # ---------- ② θ 是否真在用 q2/q3 ----------
    print("\n=== ② 学到的 θ: 位形调度真的被激活了吗 ===")
    print(f"  (初始化: 只有 '1' 列 = 0.5413, 其余 0 ⇒ α≡1 '开局无调度')")
    print(f"  {'成员':>6}{'模态':>5}" + "".join(f"{n:>9}" for n in FEAT))
    for k, g in enumerate(models):
        th = g.flex.theta.detach().numpy()
        for m in range(th.shape[0]):
            print(f"  {k:>6}{m+1:>5}" + "".join(f"{v:>9.3f}" for v in th[m]))
        if k == 0:
            pass
    th_all = np.array([g.flex.theta.detach().numpy() for g in models])
    nonconst = np.abs(th_all[:, :, 1:]).mean()
    print(f"  非常数列 |θ| 均值 = {nonconst:.4f}  (若 ≈0 则 α 实际是常数 ⇒ 候选 a)")

    # ---------- ③ 两个位形上 α 实际差多少 ----------
    print("\n=== ③ α 在两个位形上的实际比值 (要对上真机的 3.92×) ===")
    al = {}
    for wn, w in WINS.items():
        seg = slice(w, w + NEXC)
        phi = pose_features(cmd7[seg])
        rows = []
        for k, g in enumerate(models):
            p = ((phi - g.mu) / g.sd).astype(np.float32)
            rows.append(g.flex.gains(torch.from_numpy(p)).detach().numpy().mean(0))
        al[wn] = np.array(rows)                       # (K,M)
    print(f"  {'成员':>6}" + "".join(f"{f'模态{i+1} w170/w280':>18}" for i in range(3)))
    for k in range(K):
        print(f"  {k:>6}" + "".join(
            f"{al['w170'][k,m]/max(al['w280'][k,m],1e-9):>18.2f}" for m in range(3)))
    rat = al["w170"] / np.maximum(al["w280"], 1e-9)
    print(f"  α 比值 中位 {np.median(rat):.2f}  范围 {rat.min():.2f}–{rat.max():.2f}"
          f"   (真机需 3.92×)")

    # ---------- ④ 6-8Hz 上 模态路径 vs 直通 D 谁主导 ----------
    print("\n=== ④ 6–7.5Hz 上 模态铃声项 vs 位形无关直通项 D 的占比 ===")
    print("  (D 位形无关 ⇒ 若 D 主导, α 调度对该频段无效, 属结构问题 = 候选 c)")
    tg = np.arange(0, A.NANA / A.FPS, 0.01)
    print(f"  {'窗':>6}{'关节':>5}{'模态项':>11}{'直通D项':>11}{'合计':>11}{'D 占比':>9}")
    dom = []
    for wn, w in WINS.items():
        for jn, j in (("j2", 1), ("j3", 2)):
            seq = np.load(A.PB / f"seq_ep0_{jn}_4mrad_p.npz", allow_pickle=True)
            if w not in [int(x) for x in seq["wins"]]:
                continue
            da = np.zeros_like(a_arm)
            da[:, j] = seq["da"][:, j]
            mo, dt_, to = [], [], []
            for g in models:
                g.R = R
                J, phi = g.context(cmd7)
                # 全响应
                full = g.increment_fast(cmd7, da)
                # 只关模态(B=0) → 只剩 D; 只关 D → 只剩模态
                B0 = g.flex.B.detach().clone(); D0 = g.flex.D.detach().clone()
                with torch.no_grad():
                    g.flex.B.zero_()
                    only_d = g.increment_fast(cmd7, da)
                    g.flex.B.copy_(B0); g.flex.D.zero_()
                    only_m = g.increment_fast(cmd7, da)
                    g.flex.D.copy_(D0)
                lk = lambda y: np.mean([np.linalg.norm(A.lockin_at(
                    A._bl_eval(y, w / A.FPS + tg), tg, f)) for f in FREQS])
                mo.append(lk(only_m)); dt_.append(lk(only_d)); to.append(lk(full))
            m_, d_, t_ = np.median(mo), np.median(dt_), np.median(to)
            frac = d_ / max(m_ + d_, 1e-12)
            dom.append(frac)
            print(f"  {wn:>6}{jn:>5}{m_*1e3:>11.2f}{d_*1e3:>11.2f}{t_*1e3:>11.2f}"
                  f"{frac*100:>8.0f}%")
    print(f"  D 项平均占比 {np.mean(dom)*100:.0f}%")

    # ---------- 判决 ----------
    print("\n=== 判决 ===")
    a_fail = nonconst < 0.05
    b_fail = abs(np.median(rat) - 3.92) / 3.92 > 0.4
    c_fail = np.mean(dom) > 0.5
    d_fail = not ((fs >= 6) & (fs <= 7.5)).any()
    print(f"  (a) θ 未学到位形调度            : {'✗ 成立' if a_fail else '✓ 不成立'}"
          f"  (非常数列 |θ| = {nonconst:.4f})")
    print(f"  (b) α 学到了但比值不够          : {'✗ 成立' if b_fail else '✓ 不成立'}"
          f"  (α 比 {np.median(rat):.2f} vs 需 3.92)")
    print(f"  (c) 6–8Hz 由位形无关的 D 主导   : {'✗ 成立' if c_fail else '✓ 不成立'}"
          f"  (D 占比 {np.mean(dom)*100:.0f}%)")
    print(f"  (d) 无模态落在 6–7.5Hz          : {'✗ 成立' if d_fail else '✓ 不成立'}"
          f"  (模态 {fs.min():.2f}–{fs.max():.2f}Hz)")
    print()
    # α 的成员间离散 = 该参数被数据约束的程度
    spread = rat.max() / max(rat.min(), 1e-9)
    print(f"\n  α 比值的成员间离散 = {rat.min():.2f}–{rat.max():.2f} "
          f"({'跨零, 完全未约束' if rat.min() < 0.05 else f'{spread:.0f}×'})")
    if c_fail or d_fail:
        print("  ⇒ **结构问题**: 位形调度只作用于模态路径, 而该频段不由模态路径决定。")
        print("     需改结构: D → D(φ), 或增加落在 6–8Hz 且受 q2/q3 调度的模态。")
    elif a_fail:
        print("  ⇒ **辨识失败(未激活)**: θ 停在初值 ⇒ α 实际是常数。")
    elif b_fail:
        print("  ⇒ **辨识问题, 但不是能量加权造成的**(6–7.5Hz 占监督能量 19.5%, 每Hz与低频持平)。")
        print("     两条更根本的原因:")
        print("     ① **口径不同**: Ĝ 的柔性层是拟合**墙**(real−ideal, 由完整大幅值指令驱动)")
        print("        辨识出来的; 而 π 用的是**增量**响应 Ĝ(a+δa)−Ĝ(a)。二者的位形依赖")
        print("        无需一致 —— 且已有独立证据表明系统对幅值不是纯线性(偶分量随 δa¹ 缩放)。")
        print("        **B′ 探针是第一次直接测量 π 真正需要的那个增量量。**")
        print(f"     ② **α 的位形依赖本就没被约束住**: 成员间比值 {rat.min():.2f}–{rat.max():.2f}, ")
        print("        墙残差的宽带拟合下 (W, B, α) 有大量等效组合。")
        print("     ⇒ C 的受控工作点(直接测增量)正是缺的那一块 —— 重辨识, 不必改结构;")
        print(f"        但 D 占比 {np.mean(dom)*100:.0f}% 位形无关, 压掉约一半杠杆,")
        print("        D → D(φ) 值得作为次要结构改进。")
    else:
        print("  ⇒ 四个候选都不成立 —— 需要另找原因。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
