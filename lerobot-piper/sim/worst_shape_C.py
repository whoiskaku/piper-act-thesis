#!/usr/bin/env python3
"""最不利 shape error 搜索（2026-07-28，用户规格）。

**致命退化的防护（用户 2026-07-28 明确要求）**
  不能允许 shape error 含**任意的跨位姿共享线性相位斜率** —— 否则它可以直接
  伪装成另一个 τ 分支, 搜索必然把问题做成不可辨识。故:
    · **共享线性相位归入 τ**: 对 shape error 施加**零全局线性相位斜率**的规范
      (逐位姿斜率 a_p 强制 Σ_p w_p a_p = 0)
    · 允许**有界的位姿相关**相位斜率
    · 相位的**非线性**部分(曲率 f², f³)全局允许 —— 它无法伪装成纯时延

**参数化**（低阶多项式 ⇒ 频率二阶差分天然有界, 满足"不能逐频点任意跳变"）
    S_p(f) = exp(g_p(f)) · exp(−j·ψ_p(f)) · Rot(θ_p)
    g_p(f) = c0 + c1·ν + c2·ν²  + d·u_p            (ν = f − f̄, u_p = φ_p 投影)
    ψ_p(f) = a_p·2πν  + b1·2πν² + b2·2πν³          (Σ_p a_p = 0 ⇒ 无全局线性斜率)
    θ_p    = e0 + e1·u_p                            (三轴方向偏差)

**允许集合** = B′ 实测误差构成的**膨胀置信椭球**: ‖x/σ‖₂ ≤ r
    σ 由 B′ 实测量级给出(log-gain ln3, 相位斜率 20ms, 方向 15°)。

**其他口径**
  · shape error 表示**固定的真实系统失配**, **不对每个噪声 realization 单独优化**
  · 用一批 **common-random-number** seed 搜索, 另一批**独立** seed 验证

用法: python sim/worst_shape_C.py [--nsearch 400] [--trials 60]
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import analyze_probe_b2 as A
import train_pi_e2 as TP
from g_model import Rigid, load_ensemble
import select_poses_C as S
import power_analysis_C as P

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/probe_c"
F_WIDE = np.arange(2.0, 8.01, 0.5)
F_NARROW = np.array([6.0, 6.5, 7.0, 7.5])
# 允许集合的尺度(B′ 实测量级) —— 椭球半轴
SIG = {"c0": np.log(3.0), "c1": np.log(3.0) / 3, "c2": np.log(3.0) / 9,
       "d": np.log(3.0) / 2, "b1": 0.020, "b2": 0.010,
       "e0": np.radians(15.0), "e1": np.radians(10.0), "a": 0.020}


GAUGE = True   # False = **对照组**: 放开跨位姿共享线性相位斜率(预期选对率崩溃)


def unpack(x, nP):
    """x → 参数字典。a_p 部分**强制均值为零**(零全局线性相位斜率规范)。"""
    k = 0
    q = {}
    for nm in ("c0", "c1", "c2", "d", "b1", "b2", "e0", "e1"):
        q[nm] = x[k] * SIG[nm]; k += 1
    a = x[k:k + nP] * SIG["a"]
    if GAUGE:
        a = a - a.mean()                   # **规范**: Σ a_p = 0
    q["a"] = a
    return q


def apply_shape(H, F, q, pi, u_p):
    """把 shape error 作用到该位姿的 FRF 上。"""
    nu = F - F.mean()
    g = q["c0"] + q["c1"] * nu + q["c2"] * nu ** 2 + q["d"] * u_p
    psi = (q["a"][pi] * 2 * np.pi * nu + q["b1"] * 2 * np.pi * nu ** 2
           + q["b2"] * 2 * np.pi * nu ** 3)
    ang = q["e0"] + q["e1"] * u_p
    c, s = np.cos(ang), np.sin(ang)
    Hr = np.stack([c * H[:, 0] - s * H[:, 1], s * H[:, 0] + c * H[:, 1], H[:, 2]], 1)
    return (np.exp(g) * np.exp(-1j * psi))[:, None] * Hr


def model_bank(Hm, F_of):
    """预计算 Hb(τ, k) —— 它**不依赖 seed, 也不依赖 shape error**, 全设计只算一次。

    并用闭式投影残差 r² = ‖obs‖² − |⟨Hb,obs⟩|²/‖Hb‖², 把整个 τ 扫描化为一次矩阵乘。
    """
    bank = {}
    for k, H in Hm.items():
        Fp = F_of[k[0]]
        ph = np.exp(-2j * np.pi * np.outer(P.TAU_GRID, Fp))          # (nτ, nf)
        Hb = (ph[:, :, None] * H[None, :, :]).reshape(len(P.TAU_GRID), -1)
        bank[k] = (Hb, np.maximum((np.abs(Hb) ** 2).sum(1), 1e-30))
    return bank


def eval_shape(x, ctx, seeds, shape_on=True, noise_on=True):
    """给定 shape error, 在给定 seed 批上算 Q5%[ΔSSE] 与选对率(两分支取最坏)。"""
    Hm, Um, F_of, poses, nb_of, uvec, bank, Uinv = ctx
    q = unpack(x, len(poses))
    keys = sorted(Hm)
    io_of = {br: int(np.abs(P.TAU_GRID - P.TAU_TRUE[
        "-52ms" if br == "+102ms" else "+102ms"]).argmin()) for br in P.TAU_TRUE}
    worst, per = np.inf, {}
    for br, tt in P.TAU_TRUE.items():
        base = {}
        for k in keys:
            Fp = F_of[k[0]]
            H = apply_shape(Hm[k], Fp, q, k[0], uvec[k[0]]) if shape_on else Hm[k]
            base[k] = (H * np.exp(-2j * np.pi * Fp * tt)[:, None]).ravel()
        ds, hit = [], 0
        for sd in seeds:
            rr = np.random.default_rng(sd)
            curve = np.zeros(len(P.TAU_GRID))
            for k in keys:
                o = base[k]
                if noise_on:
                    nb = nb_of[tuple(F_of[k[0]])]
                    o = o + (nb[rr.integers(len(nb))] * Uinv[k][:, None]).ravel()
                Hb, hn = bank[k]
                curve += (np.abs(o) ** 2).sum() - np.abs(Hb.conj() @ o) ** 2 / hn
            hit += abs(P.TAU_GRID[int(curve.argmin())] - tt) < 0.030
            io = io_of[br]
            ds.append(curve[max(0, io - 6):io + 7].min() - curve.min())
        per[br] = {"hit": hit / len(seeds), "med": float(np.median(ds)),
                   "p05": float(np.percentile(ds, 5))}
        worst = min(worst, per[br]["p05"])
    return worst, per


def make_ctx(design, poses, g, R, nb_cache):
    F_of = design
    E_star = min(P.seq_energy(F_WIDE), P.seq_energy(F_NARROW))
    amp = {pi: P.amp_for(F_of[pi], "energy", E_star) for pi in range(len(poses))}
    Hm, Um = P.model_and_U(g, poses, R, F_of, amp)
    uvec = {pi: float(np.dot(S.phi_of(poses[pi])[1:6],
                             np.array([1.0, -1.0, 0.5, -0.5, 0.25]))) for pi in range(len(poses))}
    mu = np.mean(list(uvec.values())); sd = np.std(list(uvec.values())) + 1e-12
    uvec = {k: (v - mu) / sd for k, v in uvec.items()}
    Uinv = {k: 1.0 / np.where(np.abs(U) < 1e-15, 1e-15, U) for k, U in Um.items()}
    return (Hm, Um, F_of, poses, nb_cache, uvec, model_bank(Hm, F_of), Uinv)


def search_worst(ctx, seeds, nsearch, r=1.0, seed=0):
    """在椭球 ‖x‖₂ ≤ r 内搜最不利。随机采样(含边界) + 坐标下降精修。"""
    rng = np.random.default_rng(seed)
    nP = len(ctx[3])
    dim = 8 + nP
    best = (np.inf, None)
    for it in range(nsearch):
        v = rng.normal(size=dim)
        v /= np.linalg.norm(v)
        rad = r if it % 2 == 0 else r * rng.random() ** (1 / dim)   # 一半在边界上
        x = v * rad
        val, _ = eval_shape(x, ctx, seeds)
        if val < best[0]:
            best = (val, x.copy())
    # 坐标下降精修(投影回椭球)
    x = best[1].copy()
    for _ in range(2):
        for i in range(dim):
            for st in (0.25 * r, -0.25 * r):
                y = x.copy(); y[i] += st
                if np.linalg.norm(y) > r:
                    y *= r / np.linalg.norm(y)
                val, _ = eval_shape(y, ctx, seeds)
                if val < best[0]:
                    best = (val, y.copy()); x = y
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsearch", type=int, default=200)
    ap.add_argument("--trials", type=int, default=40, help="CRN 搜索 seed 数")
    ap.add_argument("--nvalid", type=int, default=300, help="独立验证 seed 数")
    ap.add_argument("--r", type=float, default=1.0, help="椭球膨胀因子")
    ap.add_argument("--control", action="store_true", help="跑关闭规范的对照组")
    a = ap.parse_args()

    cands, _, _ = S.candidate_poses()
    Ph = np.array([S.phi_of(c) for c in cands])
    tr = S.greedy_select(cands, 16, Ph)
    poses = [cands[i] for i in tr]
    grp = {}
    for k, i in enumerate(tr):
        grp.setdefault(S.key_qq(cands[i]), []).append(k)
    gkeys = list(grp)
    rig = Rigid(); models = load_ensemble(rigid=rig)
    man = json.loads((TP.LROOT / "pp2-deploy" / "manifest.json").read_text())
    R = np.array(man["R"], np.float32)
    g = models[len(models) // 2]
    nb_cache = {tuple(F_WIDE): P.noise_blocks(F_WIDE),
                tuple(F_NARROW): P.noise_blocks(F_NARROW)}

    SEED_SEARCH = list(range(20000, 20000 + a.trials))     # CRN: 所有候选共用
    SEED_VALID = list(range(80000, 80000 + a.nvalid))      # **独立**验证批(供 Wilson)

    TOP = [(1, 2, 5, 6), (0, 1, 2, 5), (0, 2, 4, 5)]
    cfgs = {}
    for t_ in TOP:
        anc = {grp[gkeys[gi]][0] for gi in t_}
        cfgs[f"混合{t_}"] = {pi: (F_WIDE if pi in anc else F_NARROW)
                             for pi in range(len(poses))}
    cfgs["D1 全宽带"] = {pi: F_WIDE for pi in range(len(poses))}
    cfgs["D2 全窄带"] = {pi: F_NARROW for pi in range(len(poses))}

    print(f"=== 最不利 shape error 搜索 ===")
    print(f"  规范: **零全局线性相位斜率**(Σ_p a_p = 0) ⇒ 共享线性相位归入 τ,")
    print(f"        shape error **无法伪装成另一个 τ 分支**")
    print(f"  参数: log-gain(c0,c1,c2,d) + 非线性相位曲率(b1,b2) + 方向(e0,e1)")
    print(f"        + 逐位姿线性斜率 a_p(均值为零)  共 {8+len(poses)} 维")
    print(f"  允许集合: ‖x/σ‖₂ ≤ {a.r}  (σ 为 B′ 实测量级; 低阶多项式 ⇒ 频率二阶差分天然有界)")
    print(f"  搜索 {a.nsearch} 点 × {a.trials} CRN seed;  验证用 {a.trials} **独立** seed\n")

    print(f"  {'设计':>16}{'无shape Q5%':>13}{'最坏Q5%(搜)':>13}{'最坏Q5%(验)':>13}"
          f"{'+102ms':>9}{'-52ms':>9}{'最坏选对':>10}{'Wilson下界':>11}{'顶边界':>7}")
    res = {}
    nx = 8 + len(poses)
    for nm, des in cfgs.items():
        ctx = make_ctx(des, poses, g, R, nb_cache)
        rnd0, _ = eval_shape(np.zeros(nx), ctx, SEED_SEARCH)
        val, x = search_worst(ctx, SEED_SEARCH, a.nsearch, a.r)
        vval, per = eval_shape(x, ctx, SEED_VALID)              # **独立** seed 验证
        so, _ = eval_shape(x, ctx, SEED_VALID, noise_on=False)  # shape-only
        no, _ = eval_shape(np.zeros(nx), ctx, SEED_VALID)       # noise-only
        hits = min(v["hit"] for v in per.values())
        n = len(SEED_VALID)
        lo, _ = P.wilson(int(round(hits * n)), n)
        onb = np.linalg.norm(x) > 0.98 * a.r
        res[nm] = {"rand_q05": rnd0, "worst_search": val, "worst_valid": vval,
                   "shape_only": so, "noise_only": no, "norm": float(np.linalg.norm(x)),
                   "hit_p102": per["+102ms"]["hit"], "hit_m52": per["-52ms"]["hit"],
                   "med_p102": per["+102ms"]["med"], "med_m52": per["-52ms"]["med"],
                   "worst_hit": hits, "wilson_lo": lo, "on_boundary": bool(onb),
                   "x": x.tolist()}
        print(f"  {nm:>16}{rnd0:>13.3e}{val:>13.3e}{vval:>13.3e}"
              f"{per['+102ms']['hit']*100:>8.1f}%{per['-52ms']['hit']*100:>8.1f}%"
              f"{hits*100:>9.1f}%{lo*100:>10.1f}%{'是' if onb else '否':>7}")

    print(f"\n  【最坏误差 x* 的能量分布(占 ‖x‖², 判断是靠增益还是靠相位)】")
    print(f"  {'设计':>16}{'log-gain':>10}{'相位曲率':>10}{'方向':>8}{'位姿斜率':>10}{'‖x‖':>8}")
    for nm, v in res.items():
        xx = np.array(v["x"]); n2 = max((xx ** 2).sum(), 1e-30)
        print(f"  {nm:>16}{(xx[:4]**2).sum()/n2:>9.0%}{(xx[4:6]**2).sum()/n2:>9.0%}"
              f"{(xx[6:8]**2).sum()/n2:>7.0%}{(xx[8:]**2).sum()/n2:>9.0%}{v['norm']:>8.2f}")

    print(f"\n  【最坏 ΔSSE 中位数(两分支)】")
    for nm, v in res.items():
        print(f"    {nm:>16}: +102ms 中位 {v['med_p102']:.3e} / 5% {v['worst_valid']:.3e}"
              f"   −52ms 中位 {v['med_m52']:.3e}")
    print(f"\n  【shape-only / noise-only 分解(最坏误差, 验证 seed)】")
    print(f"  {'设计':>16}{'shape-only':>13}{'noise-only':>13}{'联合':>13}{'shape占比':>11}")
    for nm, v in res.items():
        print(f"  {nm:>16}{v['shape_only']:>13.3e}{v['noise_only']:>13.3e}"
              f"{v['worst_valid']:>13.3e}{v['shape_only']/max(v['worst_valid'],1e-30):>10.0%}")

    order_rand = sorted(res, key=lambda k: -res[k]["rand_q05"])
    order_worst = sorted(res, key=lambda k: -res[k]["worst_valid"])
    print(f"\n  排名(随机 shape): {' > '.join(order_rand)}")
    print(f"  排名(最坏 shape): {' > '.join(order_worst)}")
    print(f"  ⇒ 最坏条件下排名{'**改变**' if order_rand != order_worst else '未变'}")
    print(f"\n  放行门槛: 两分支最坏选对率的 95%CI 下界 ≥95%")
    for nm, v in res.items():
        print(f"    {nm:>16}: {'✓' if v['wilson_lo'] >= 0.95 else '✗'}"
              f"  (下界 {v['wilson_lo']*100:.1f}%)")
    # ---- 退化距离: 伪装成另一分支所需的**共享**线性斜率, 折算成椭球半径 ----
    dtau = abs(P.TAU_TRUE["+102ms"] - P.TAU_TRUE["-52ms"])
    r_deg = np.sqrt(len(poses)) * dtau / SIG["a"]
    print(f"\n  【退化距离】伪装另一分支需**共享**斜率 {dtau*1e3:.0f} ms")
    print(f"    折算椭球半径 = √{len(poses)}·{dtau*1e3:.0f}/{SIG['a']*1e3:.0f} = "
          f"**{r_deg:.1f}**  (当前允许集合 r={a.r})")
    print(f"    ⇒ 该退化在允许集合外 **{r_deg/a.r:.0f}×**; 选对率稳健**不只靠规范**,")
    print(f"      也因为 B′ 尺度的相位误差远小于 154 ms 分支间距。")

    # ---- 定向退化探针: **只**放开共享线性斜率, 其余分量全零 ----
    # 目的: 隔离"共享斜率伪装 τ 分支"这一条退化路径本身,
    #       避免与 log-gain 炸毁式破坏混为一谈。
    if a.control:
        global GAUGE
        print(f"\n  【定向退化探针: 只放开共享线性斜率 s, 其余分量=0】")
        print(f"  {'设计':>16}{'s(ms)':>8}{'椭球r':>8}{'选对率':>9}{'Q5% ΔSSE':>13}")
        GAUGE = False
        nx2 = 8 + len(poses)
        for nm in ("D1 全宽带", f"混合{TOP[0]}"):
            ctx = make_ctx(cfgs[nm], poses, g, R, nb_cache)
            for s_ms in (0, 20, 40, 77, 110, 154):
                xx = np.zeros(nx2)
                xx[8:] = (s_ms * 1e-3) / SIG["a"]          # 全部位姿同一斜率
                v, pr = eval_shape(xx, ctx, SEED_VALID)
                h = min(u["hit"] for u in pr.values())
                print(f"  {nm:>16}{s_ms:>8d}{np.linalg.norm(xx):>8.1f}"
                      f"{h*100:>8.1f}%{v:>13.3e}")
        GAUGE = True

    # ---- 对照: 关掉规范, 逐级放大 r, 验证搜索确实能找到破坏性误差 ----
    if a.control:
        print(f"\n  【对照: 关闭零全局线性斜率规范】(验证搜索器有能力破坏选对率)")
        print(f"  {'设计':>16}{'r':>7}{'选对率':>9}{'Q5% ΔSSE':>13}")
        GAUGE = False
        for nm in ("D1 全宽带", f"混合{TOP[0]}"):
            ctx = make_ctx(cfgs[nm], poses, g, R, nb_cache)
            for rr_ in (1.0, 4.0, 16.0, r_deg):
                v, xx = search_worst(ctx, SEED_SEARCH, max(a.nsearch // 2, 20), rr_)
                _, pr = eval_shape(xx, ctx, SEED_VALID)
                h = min(u["hit"] for u in pr.values())
                print(f"  {nm:>16}{rr_:>7.1f}{h*100:>8.1f}%{v:>13.3e}")
        GAUGE = True

    (OUT / "worst_shape_C.json").write_text(json.dumps(
        {"r": a.r, "nsearch": a.nsearch, "trials": a.trials, "res": res},
        indent=1, ensure_ascii=False))
    print(f"\n→ {(OUT / 'worst_shape_C.json').relative_to(ROOT.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
