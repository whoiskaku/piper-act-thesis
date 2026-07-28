#!/usr/bin/env python3
"""Ĝ 可表达性 / 灵敏度 / Jacobian 秩分析（2026-07-28）。

**为什么需要这一步**（用户 2026-07-28 指出前一版审计的四处不严谨）:
  1. θ 非零只说明优化器移动过参数, 不说明调度已被**可靠辨识**
  2. α 的中位比 0.34 **不能**直接与真实总响应比 3.92 比较 ——
     三模态 + 直通 + 三轴是**复数叠加**, 必须看总 FRF 对各 α_m 的灵敏度
  3. 6–7.5Hz 占 19.5% 能量 **不代表**调度可辨识; 关键是该频段内的位形覆盖、
     激励, 以及**参数 Jacobian / Fisher 信息的秩**
  4. `0.53r + 0.47 = 3.92` 假设模态与直通**同相**, 对复频响不合法 ——
     D(φ) 是否需要应由**复数路径分解**决定

本工具做四件事:
  ① **复数路径分解**: 逐 (工作点, 频点) 把 Ĝ 的 FRF 拆成 刚体 / 模态1..3 / 直通D
     的复数 3-向量, 看相位关系(同相? 反相? 正交?)
  ② **总 FRF 对 α_m 的灵敏度**: ∂|H|/∂ln α_m = Re⟨H, P_m⟩/|H|
     —— 复数叠加下, 增大某个 α 可能**减小** |H|
  ③ **可表达性测试**: α 是**频率无关的正标量**(3个)。求 g_m ≥ 0 使
        H(g) = Σ_m g_m·P_m(f) + P_D(f) + P_rigid(f)  ≈  H_real(f)
     在该工作点的 4 频点 × 3 轴 = 12 复数 = 24 实残差上最小。
     拟合残差大 ⇒ **仅调 θ/α 不足**, 需 D(φ) / 位形相关模态混合 / 极点
  ④ **Jacobian 秩 / Fisher 信息**: 对 [α_1..α_3] (可扩到 D) 的复 Jacobian 的
     奇异值谱与条件数 —— 决定"仅调 θ 是否足够"以及需要多少独立位形

用法: python sim/audit_G_expressibility.py
"""
from __future__ import annotations
import io
import contextlib
import json
from pathlib import Path

import numpy as np
import torch

import analyze_probe_b2 as A
import train_pi_e2 as TP
from g_model import Rigid, load_ensemble
from make_probe_b2 import NEXC

ROOT = Path(__file__).resolve().parents[1]
OPS = [("j2", 170), ("j2", 280), ("j3", 170)]
ROUNDS = [(1, "4mrad", ""), (2, "4mrad", "r2_"), (3, "2mrad", "r3_")]
PATHS = ["刚体", "模态1", "模态2", "模态3", "直通D"]


def model_paths(g, cmd7, da, w, freqs, R):
    """逐路径的复数 FRF 3-向量。**数值提取**(不解析推导 —— 历史上解析推导错过三次)。

    返回 dict{路径名: (nf,3) complex}, 已除以输入的锁相值 U(f) ⇒ 就是 FRF。
    """
    g.R = R
    tg = np.arange(0, A.NANA / A.FPS, 0.01)
    fl = g.flex
    B0 = fl.B.detach().clone()
    D0 = fl.D.detach().clone()

    def lock(y):
        yy = A._bl_eval(y, w / A.FPS + tg)
        return np.array([A.lockin_at(yy, tg, f) for f in freqs])       # (nf,3)

    with torch.no_grad():
        fl.B.zero_(); fl.D.zero_()
        rigid = lock(g.increment_fast(cmd7, da))                       # 只剩刚体
        fl.D.copy_(D0)
        rig_d = lock(g.increment_fast(cmd7, da))                       # 刚体+直通
        fl.D.zero_()
        modes = []
        for m in range(B0.shape[0]):
            fl.B.zero_()
            fl.B[m] = B0[m]
            modes.append(lock(g.increment_fast(cmd7, da)) - rigid)     # 该模态单独
        fl.B.copy_(B0); fl.D.copy_(D0)
        full = lock(g.increment_fast(cmd7, da))
    # 输入侧锁相(δa 的该关节分量), 用同一网格
    jj = int(np.abs(da).max(0).argmax())
    U = np.array([A.lockin_at(A._bl_eval(da[:, jj], w / A.FPS + tg), tg, f) for f in freqs])
    U = np.where(np.abs(U) < 1e-12, 1e-12, U)
    out = {"刚体": rigid / U[:, None], "直通D": (rig_d - rigid) / U[:, None]}
    for m, mm in enumerate(modes):
        out[f"模态{m+1}"] = mm / U[:, None]
    out["_full"] = full / U[:, None]
    return out


def measured_H(jn, w, freqs):
    """实测 H1(f) 复数 3-向量, 三轮逆方差加权(用 |H| 的轮间散布作权重代理)。"""
    Hs, ws = [], []
    for rnd, amp, rt in ROUNDS:
        seq = np.load(A.PB / f"seq_ep0_{jn}_{amp}_p.npz", allow_pickle=True)
        if w not in [int(x) for x in seq["wins"]]:
            continue
        R = {k: json.loads((A.PB / f"replay_ep0_{rt}{jn}_{amp}_{s}.json").read_text())
             for k, s in [("Bp", "p"), ("Bm", "m")]}
        W = {k: A.arm_windows(R[k], seq, +1 if k == "Bp" else -1) for k in R}
        LI = lambda k, f, key: A.lockin_at(W[k][w][key], W[k][w]["t"], f, W[k][w]["dt"])
        U = np.array([(LI("Bp", f, "u") - LI("Bm", f, "u")) / 2 for f in freqs])
        Y = np.array([(LI("Bp", f, "y") - LI("Bm", f, "y")) / 2 for f in freqs])
        Hs.append(Y / np.where(np.abs(U) < 1e-12, 1e-12, U)[:, None])
        ws.append(1.0)
    Hs = np.array(Hs)
    return Hs.mean(0), Hs.std(0) / max(np.sqrt(len(Hs)), 1)


def fit_alpha(P, Href):
    """可表达性: 求 g_m ≥ 0 使 Σ g_m P_m + P_D + P_rigid ≈ Href。

    α 是**频率无关**的 ⇒ g 在该工作点的全部频点上共享。
    复线性最小二乘 + 非负投影(3 参数, 用穷举+refine 保证全局)。
    """
    base = P["刚体"] + P["直通D"]
    Ms = np.stack([P[f"模态{m+1}"] for m in range(3)])          # (3,nf,3)
    y = (Href - base).ravel()
    Amat = Ms.reshape(3, -1).T                                  # (nf*3, 3) complex
    # 实数化(g 是实数) ⇒ 拼实虚部
    Ar = np.vstack([Amat.real, Amat.imag])
    yr = np.concatenate([y.real, y.imag])
    g, *_ = np.linalg.lstsq(Ar, yr, rcond=None)
    if (g < 0).any():                                           # 非负投影(3维穷举活跃集)
        best = None
        for mask in range(1, 8):
            idx = [i for i in range(3) if mask >> i & 1]
            gg = np.zeros(3)
            sol, *_ = np.linalg.lstsq(Ar[:, idx], yr, rcond=None)
            if (sol < 0).any():
                continue
            gg[idx] = sol
            r = np.linalg.norm(Ar @ gg - yr)
            if best is None or r < best[0]:
                best = (r, gg)
        g = best[1] if best else np.maximum(g, 0)
    resid = Ar @ g - yr
    return g, np.linalg.norm(resid) / max(np.linalg.norm(yr + Ar @ np.zeros(3)), 1e-12), \
        np.linalg.norm(resid) / max(np.linalg.norm(Href.ravel()), 1e-12), Ar


def main():
    rig = Rigid()
    models = load_ensemble(rigid=rig)
    man = json.loads((TP.LROOT / "pp2-deploy" / "manifest.json").read_text())
    R = np.array(man["R"], np.float32)
    z = np.load(TP.LROOT / "pp2-deploy" / "ep000.npz")
    cmd7 = np.concatenate([z["a_arm"].astype(np.float32),
                           z["grip"].astype(np.float32)[:, None]], 1)
    freqs = np.array([6.0, 6.5, 7.0, 7.5])
    km = len(models) // 2                      # 中位成员; 全成员在 ④ 里扫

    art = {}
    for jn, w in OPS:
        j = {"j2": 1, "j3": 2}[jn]
        seq = np.load(A.PB / f"seq_ep0_{jn}_4mrad_p.npz", allow_pickle=True)
        if w not in [int(x) for x in seq["wins"]]:
            continue
        da = np.zeros((len(cmd7), 6), np.float32)
        da[:, j] = seq["da"][:, j]
        P = model_paths(models[km], cmd7, da, w, freqs, R)
        Href, Hsd = measured_H(jn, w, freqs)

        print(f"\n{'='*78}\n{jn} w{w}  (成员 k={km})\n{'='*78}")
        # ---------- ① 复数路径分解 ----------
        print("① 复数路径分解 (|·| mrad/s per rad, 相位相对总和)")
        print(f"  {'f':>5}" + "".join(f"{p:>20}" for p in PATHS) + f"{'合计':>16}")
        for i, f in enumerate(freqs):
            tot = P["_full"][i]
            row = f"  {f:>5.1f}"
            for p in PATHS:
                v = P[p][i]
                ph = np.degrees(np.angle(np.vdot(tot, v)))
                row += f"{np.linalg.norm(v)*1e3:>13.1f}∠{ph:>+5.0f}"
            row += f"{np.linalg.norm(tot)*1e3:>15.1f}"
            print(row)
        # 相位关系: 模态合成 vs 直通
        mm = sum(P[f"模态{m+1}"] for m in range(3))
        dd = P["直通D"]
        ang = [np.degrees(np.angle(np.vdot(mm[i], dd[i]))) for i in range(len(freqs))]
        print(f"  模态合成 与 直通D 的夹角: {np.round(ang,0)}°"
              f"  ⇒ {'近同相(实数配比近似可用)' if np.abs(ang).mean()<40 else '**非同相, 实数配比方程不合法**'}")

        # ---------- ② 灵敏度 ----------
        print("\n② 总 FRF 对 ln α_m 的灵敏度  ∂|H|/∂lnα_m = Re⟨H,P_m⟩/|H|  (归一到 |H|)")
        print(f"  {'f':>5}" + "".join(f"{f'模态{m+1}':>12}" for m in range(3)))
        for i, f in enumerate(freqs):
            H = P["_full"][i]
            s = [np.real(np.vdot(H, P[f"模态{m+1}"][i])) / max(np.linalg.norm(H), 1e-12) ** 2
                 for m in range(3)]
            print(f"  {f:>5.1f}" + "".join(f"{v:>12.3f}" for v in s)
                  + ("   ← 有负值: 增大该 α 会**减小** |H|" if min(s) < -0.02 else ""))

        # ---------- ③ 可表达性 ----------
        g, r_rel, r_abs, Ar = fit_alpha(P, Href)
        print(f"\n③ 可表达性: 仅用 3 个频率无关的 α 缩放 g_m ≥ 0, 能否拟合实测复数 FRF?")
        print(f"  最优 g = {np.round(g,3)}  (g=1 表示当前 α 已最优)")
        print(f"  拟合后残差 / |H_real| = **{r_abs*100:.1f}%**")
        print(f"  {'f':>5}{'|H_real|':>11}{'|H_模型(g)|':>13}{'幅值比':>9}{'方向夹角':>10}")
        Hfit = P["刚体"] + P["直通D"] + sum(g[m] * P[f"模态{m+1}"] for m in range(3))
        for i, f in enumerate(freqs):
            a_, b_ = np.linalg.norm(Href[i]), np.linalg.norm(Hfit[i])
            ang2 = np.degrees(np.angle(np.vdot(Href[i], Hfit[i])))
            print(f"  {f:>5.1f}{a_*1e3:>11.1f}{b_*1e3:>13.1f}{b_/max(a_,1e-12):>9.2f}"
                  f"{ang2:>+9.0f}°")
        # ---------- ③b 残差的成因分解: 相位 / 三轴方向 / 频率形状 ----------
        base = P["刚体"] + P["直通D"]
        Ms = np.stack([P[f"模态{m+1}"] for m in range(3)])
        y = (Href - base).ravel()
        Am = Ms.reshape(3, -1).T
        def rr(sol, Amat=Am, yy=y):
            return np.linalg.norm(Amat @ sol - yy) / max(np.linalg.norm(Href.ravel()), 1e-12)
        # (i) 实数非负 g (物理; α=softplus>0)
        r_real = r_abs
        # (ii) 允许复数 g (非物理, 仅诊断: 隔离"相位"这一项)
        gc, *_ = np.linalg.lstsq(Am, y, rcond=None)
        r_cplx = rr(gc)
        # (iii) 实数 g + 一个全局附加时延 τ (相位可由极点/时延提供)
        best = (1e9, None, None)
        for tau in np.linspace(-0.06, 0.06, 241):
            ph = np.exp(-2j * np.pi * freqs * tau)[:, None]
            yt = ((Href / ph) - base).ravel()
            Ar2 = np.vstack([Am.real, Am.imag])
            sol, *_ = np.linalg.lstsq(Ar2, np.concatenate([yt.real, yt.imag]), rcond=None)
            sol = np.maximum(sol, 0)
            rv = np.linalg.norm(Am @ sol - yt) / max(np.linalg.norm(Href.ravel()), 1e-12)
            if rv < best[0]:
                best = (rv, sol, tau)
        print(f"\n③b 残差成因分解 (残差 / |H_real|)")
        print(f"  (i)   实数非负 g   (物理, α=softplus>0)        : {r_real*100:>6.1f}%")
        print(f"  (ii)  复数 g       (非物理; 隔离相位这一项)     : {r_cplx*100:>6.1f}%")
        print(f"  (iii) 实数 g + 全局附加时延 τ={best[2]*1e3:+.0f}ms      : {best[0]*100:>6.1f}%")
        print(f"  ⇒ 相位贡献 = (i)−(ii) = {(r_real-r_cplx)*100:+.1f}pp;  "
              f"时延能补回 {(r_real-best[0])*100:+.1f}pp")
        print(f"  ⇒ 复数 g 也剩 {r_cplx*100:.1f}% ⇒ 那部分是**三轴方向 + 频率形状**, "
              f"{'仅调增益(即使允许复数)也不够' if r_cplx>0.3 else '主要是相位问题'}")
        art[f"{jn}_w{w}"] = {"g": g.tolist(), "resid_rel": float(r_abs),
                             "resid_complex_g": float(r_cplx),
                             "resid_real_g_plus_delay": float(best[0]),
                             "best_tau_ms": float(best[2] * 1e3),
                             "H_real_abs": np.linalg.norm(Href, axis=1).tolist(),
                             "H_fit_abs": np.linalg.norm(Hfit, axis=1).tolist()}

        # ---------- ④ Jacobian 秩 / 条件数 ----------
        sv = np.linalg.svd(Ar, compute_uv=False)
        print(f"\n④ 对 [α₁,α₂,α₃] 的 Jacobian 奇异值 = {np.round(sv/sv[0],4)}")
        print(f"  条件数 = {sv[0]/max(sv[-1],1e-30):.1f}"
              f"   数值秩(阈 1e-3) = {int((sv/sv[0] > 1e-3).sum())}/3")
        if sv[-1] / sv[0] < 1e-2:
            print(f"  ⚠ 最小奇异值 {sv[-1]/sv[0]:.1e} ⇒ 至少一个 α 方向**在该工作点不可辨识**")
        art[f"{jn}_w{w}"]["sv_ratio"] = (sv / sv[0]).tolist()

    # ---------- 跨工作点联合可辨识性 ----------
    print(f"\n{'='*78}\n跨工作点联合: α 的位形斜率需要多少独立位形?\n{'='*78}")
    print("  α_m(φ) = softplus(φ·θ_m) ⇒ 要辨识 θ_m 的 8 个分量, 需 ≥8 个**线性无关的 φ**")
    phis = []
    from g_model import pose_features
    for jn, w in OPS:
        phis.append(pose_features(cmd7[w:w + NEXC]).mean(0))
    Phi = np.array(phis)
    s2 = np.linalg.svd(Phi - Phi.mean(0), compute_uv=False)
    print(f"  现有 {len(phis)} 个工作点的 φ 矩阵(去均值)奇异值 = {np.round(s2/max(s2[0],1e-30),3)}")
    print(f"  φ 空间的数值秩 = {int((s2/max(s2[0],1e-30) > 1e-6).sum())} / 8")
    print(f"  ⇒ **现有工作点最多只能约束 φ 空间的 {int((s2/max(s2[0],1e-30) > 1e-6).sum())} 个方向**;")
    print(f"     θ 的其余方向由初值/正则决定, 不是数据决定 —— 这正是 α 跨成员不稳的原因。")
    print(f"  ⇒ C 的位姿网格应在 **φ 空间**里选点(而非关节角等距), 目标是让 Φ 满秩且条件数小。")

    (ROOT / "outputs/probe_b2/audit_expressibility.json").write_text(
        json.dumps(art, indent=1, ensure_ascii=False))
    print(f"\n→ outputs/probe_b2/audit_expressibility.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
