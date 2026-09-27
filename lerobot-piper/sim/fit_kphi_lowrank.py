#!/usr/bin/env python3
"""连续位形调度的低维核修正（③）—— 严格按 `reports/PREREG_KPHI_LOWRANK.md`。

要回答的**唯一**一句话：
    **相比固定核，增加这 12 个随位形变化的系数，有没有折外收益？**

    ĝap[t] = Σ_τ ( H₀[τ] + g(s[t−τ])·ΔK[τ] ) δa[t−τ]
    ΔK[τ,a,j] = Σ_{p=1..3} Σ_{q=1..4} c_pq · L_p[τ] · M_q[a,j]      ⇒ **12 个系数**

`c = 0` ⇒ 退化为固定核 ①。**该分支显式计算，不用大有限 λ 近似。**

⚠ 三条口径（预注册 §2bis / §2ter / L_p 范围）
---------------------------------------------
1. **两个 ridge 尺度分别算**：`λ_H = m_H × ridge_scale(固定核设计)`，
   `λ_Δ = m_Δ × ridge_scale(12 列修正设计)`。
   **不得**用拼接后的整个设计矩阵统一算 —— 否则加修正列会连带改变固定核的惩罚。
2. **基准 ① 独立选参**：① 在 5 个 `c≡0` 分支里选 `m_H`；③ 在全部 30 个组合里选。
   **不得**拿 ③ 选中的 `m_H` 临时构造较弱的 ①。
3. **`L_p` 是前 3 个 DCT-II 基，覆盖整个 16 帧滞后，代表平滑的核变化** ——
   **不是**「短滞后集中」。**阴性结果只针对这个预定基空间。**

折内三步（顺序写死，先固定基底再联合拟合 ⇒ 凸）
    ① 拟临时核 H̃（multiplier 1e-3，写死不选）
    ② H̃ 展开为 16×18 → SVD → 前 4 个右奇异向量 → 还原 3×6 得 M_1..M_4，归一化
       **此后不再更新**
    ③ 固定 M_q / L_p，对 (H₀, c) 联合 ridge —— 对参数线性 ⇒ 凸，无局部极小

⚠ 临时核正则、基底归一化、g 的 s̄/σ_s **全部只用该折的训练种子**。

用法:
    python -u sim/fit_kphi_lowrank.py --self-test
    python -u sim/fit_kphi_lowrank.py
"""
from __future__ import annotations
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = ROOT / "outputs/step1_H/kphi_lowrank_v1.json"
THR_UNUSED = None
NLAG, NAX, NJ = 16, 3, 6
P_LAG, Q_DIR = 3, 4
NC = P_LAG * Q_DIR                  # 12
TMP_MULT = 1e-3                     # 临时核 multiplier（写死不选）
GATE = 0.05
SUBBANDS = {"0.5-9Hz": (0.5, 9.0), "0.5-3Hz": (0.5, 3.0), "3-9Hz": (3.0, 9.0)}
EDGE = 20


# ── 基 ────────────────────────────────────────────────────────────────
def lag_basis():
    """前 3 个 DCT-II 基，**覆盖整个 16 帧滞后**（不是短滞后集中），单位 L2 范数。"""
    t = np.arange(NLAG)
    L = np.stack([np.cos(np.pi * p * (t + 0.5) / NLAG) for p in range(P_LAG)])
    return L / np.linalg.norm(L, axis=1, keepdims=True)


def dir_basis(H):
    """H̃ (16,3,6) → 展开 **16×18** → SVD → 前 4 个**右奇异向量** → 还原 3×6，归一化。"""
    U, S, Vt = np.linalg.svd(H.reshape(NLAG, NAX * NJ), full_matrices=False)
    M = Vt[:Q_DIR].reshape(Q_DIR, NAX, NJ)
    return M / np.linalg.norm(M.reshape(Q_DIR, -1), axis=1)[:, None, None]


# ── 设计矩阵 ──────────────────────────────────────────────────────────
def design_H(da, S1):
    """固定核设计：(T, 96)。每列独立映射到 3 个输出轴（与 `fit_H` 同形）。"""
    return S1.causal_design(da, NLAG)                       # (T, 16*6)


def design_D(da, gs, L, M):
    """修正设计：(T, 3, 12)。`Z[t,a,pq] = Σ_τ g(s[t−τ]) L_p[τ] (M_q δa[t−τ])[a]`。"""
    T = len(da)
    gd = gs[:, None] * da                                   # g(s[t])·δa[t]  (T,6)
    Z = np.zeros((T, NAX, NC))
    for p in range(P_LAG):
        # Σ_τ L_p[τ] · gd[t−τ]   —— 因果，零填充历史
        acc = np.zeros((T, NJ))
        for tau in range(NLAG):
            if tau == 0:
                acc += L[p, 0] * gd
            elif tau < T:
                acc[tau:] += L[p, tau] * gd[:-tau]
        for q in range(Q_DIR):
            Z[:, :, p * Q_DIR + q] = acc @ M[q].T           # (T,6)@(6,3) → (T,3)
    return Z


def ridge_scale(X, ncol):
    return max(float(np.square(X).sum()) / max(ncol, 1), np.finfo(float).tiny)


# ── 拟合 ──────────────────────────────────────────────────────────────
def fit_joint(eps, m_H, m_D, L, Ms, c_zero):
    """联合 ridge。`c_zero=True` ⇒ **显式**只解固定核（不是大 λ 近似）。

    ⚠ 两个尺度**分别**算（预注册 §2bis）。
    """
    XH, XD, Y = [], [], []
    for e in eps:
        xh = design_H(e.da, e.S1)
        XH.append(xh - xh.mean(0, keepdims=True))
        Y.append(np.asarray(e.gap, float) - np.mean(e.gap, 0, keepdims=True))
        if not c_zero:
            xd = design_D(e.da, e.gs, L, Ms)
            XD.append(xd - xd.mean(0, keepdims=True))
    XH = np.concatenate(XH); Y = np.concatenate(Y)
    nH = XH.shape[1]                                        # 96
    # ⚠ 固定核的尺度按**展平后 288 参数**算 ⇒ 与旧实现 sum(X²)/96 数值相同
    lam_H = m_H * ridge_scale(XH, nH)
    if c_zero:
        G = XH.T @ XH + lam_H * np.eye(nH)
        coef = np.linalg.solve(G, XH.T @ Y)                 # (96,3)
        return coef.reshape(NLAG, NJ, NAX).transpose(0, 2, 1), np.zeros(NC), lam_H, 0.0
    XD = np.concatenate(XD)                                 # (N,3,12)
    N = len(XH)
    # 展平到 (N*3) 行：H 块 288 列（按轴分块），修正块 12 列
    A = np.zeros((N * NAX, nH * NAX + NC))
    b = Y.reshape(-1)
    for a in range(NAX):
        A[a::NAX, a * nH:(a + 1) * nH] = XH
        A[a::NAX, nH * NAX:] = XD[:, a, :]
    lam_D = m_D * ridge_scale(A[:, nH * NAX:], NC)          # **只用 12 列算**
    pen = np.concatenate([np.full(nH * NAX, lam_H), np.full(NC, lam_D)])
    th = np.linalg.solve(A.T @ A + np.diag(pen), A.T @ b)
    H0 = np.stack([th[a * nH:(a + 1) * nH] for a in range(NAX)], 1)   # (96,3)
    return H0.reshape(NLAG, NJ, NAX).transpose(0, 2, 1), th[nH * NAX:], lam_H, lam_D


def predict(e, H0, c, L, Ms):
    p = e.S1.causal_conv(H0, e.da)
    if np.any(c):
        p = p + design_D(e.da, e.gs, L, Ms) @ c
    return p


def energy(e, H0, c, L, Ms, band):
    S1 = e.S1
    r = np.asarray(e.gap, float) - predict(e, H0, c, L, Ms)
    r = r - r.mean(0, keepdims=True)
    old = S1.METRIC_BAND
    try:
        S1.METRIC_BAND = band
        r = S1.bandpass41(r)
    finally:
        S1.METRIC_BAND = old
    v = r[EDGE:-EDGE]
    return float((v ** 2).sum() / (NAX * len(v)))


# ── 折内三步 ──────────────────────────────────────────────────────────
def fold_fit(tr, m_H, m_D, c_zero):
    """① 临时核 → ② 冻结基底 → ③ 联合拟合。**基底此后不再更新。**

    ⚠ 临时核正则、基底归一化、g 的 s̄/σ_s **全部只用 `tr`（该折训练种子）**。
    """
    L = lag_basis()
    if c_zero:
        H0, c, lh, ld = fit_joint(tr, m_H, 0.0, L, None, True)
        return H0, c, L, None, lh, ld
    Ht, _, _, _ = fit_joint(tr, TMP_MULT, 0.0, L, None, True)   # ① 临时核
    Ms = dir_basis(Ht)                                          # ② 冻结基底
    H0, c, lh, ld = fit_joint(tr, m_H, m_D, L, Ms, False)       # ③ 联合
    return H0, c, L, Ms, lh, ld


def set_g(eps, tr):
    """`g(s)=(s−s̄)/σ_s`；**`s̄`,`σ_s` 只由训练折算**。"""
    sm = np.concatenate([e.s for e in tr])
    mu, sd = float(sm.mean()), float(sm.std())
    for e in eps:
        e.gs = (e.s - mu) / (sd if sd > 0 else 1.0)
    return mu, sd


def cv_select(p1, grid, only_czero):
    """P1 leave-one-seed-out。返回 (最优组合, 全部行)。**确定性全序**见预注册 §2ter。"""
    seeds = sorted({e.seed for e in p1})
    combos = ([(m, None) for m in grid] if only_czero
              else [(m, None) for m in grid] + list(itertools.product(grid, grid)))
    rows = []
    for m_H, m_D in combos:
        cz = m_D is None
        tot = []
        for sd_ in seeds:
            tr = [e for e in p1 if e.seed != sd_]
            te = [e for e in p1 if e.seed == sd_]
            set_g(p1, tr)
            H0, c, L, Ms, _, _ = fold_fit(tr, m_H, 0.0 if cz else m_D, cz)
            tot.append(np.mean([energy(e, H0, c, L, Ms, SUBBANDS["0.5-9Hz"]) for e in te]))
        rows.append({"m_H": float(m_H), "m_D": (None if cz else float(m_D)),
                     "c_zero": cz, "cv": float(np.mean(tot))})
    # 全序: CV 升序 → c≡0 优先 → m_D 大 → m_H 大
    key = lambda r: (r["cv"], 0 if r["c_zero"] else 1,
                     -(r["m_D"] or np.inf), -r["m_H"])
    return sorted(rows, key=key)[0], rows


def self_test(eps, S1, grid):
    """自测。核心：**同一 multiplier 下 c≡0 与独立固定核实现数值一致**。"""
    print("\n" + "=" * 76 + "\n自测\n" + "=" * 76)
    ok = {}
    def chk(k, c, d=""):
        ok[k] = bool(c); print(f"  {'✓' if c else '**✗**'} {k}{('  ' + d) if d else ''}")

    p1 = eps["P1"]; set_g([e for v in eps.values() for e in v], p1)
    L = lag_basis()

    # ① c≡0 与 fit_step1_candidate.fit_H 逐位一致
    worst_H = worst_p = worst_e = 0.0
    for m in grid:
        H_ref, lam_ref = S1.fit_H(p1, m)
        H_new, c_new, lam_new, _ = fit_joint(p1, m, 0.0, L, None, True)
        worst_H = max(worst_H, float(np.abs(H_ref - H_new).max()))
        worst_p = max(worst_p, abs(lam_ref - lam_new))
        for e in p1[:3]:
            pr = S1.causal_conv(H_ref, e.da)
            worst_e = max(worst_e, float(np.abs(pr - predict(e, H_new, c_new, L, None)).max()))
        assert not np.any(c_new)
    chk("1 **c≡0 的核与 fit_H 逐位一致**（5 个 multiplier）", worst_H < 1e-12, f"max |ΔH| {worst_H:.2e}")
    chk("1b **λ_H 与旧实现一致**（两个尺度分别算的前提下）", worst_p < 1e-18, f"max |Δλ| {worst_p:.2e}")
    chk("1c **逐帧预测一致**", worst_e < 1e-12, f"max |Δpred| {worst_e:.2e}")

    # CV 得分一致
    b_cz, rows_cz = cv_select(p1, grid, only_czero=True)
    ref = {}
    for m in grid:
        tot = []
        for sd_ in sorted({e.seed for e in p1}):
            tr = [e for e in p1 if e.seed != sd_]; te = [e for e in p1 if e.seed == sd_]
            H_ref, _ = S1.fit_H(tr, m)
            tot.append(np.mean([energy(e, H_ref, np.zeros(NC), L, None,
                                       SUBBANDS["0.5-9Hz"]) for e in te]))
        ref[m] = float(np.mean(tot))
    dmax = max(abs(ref[r["m_H"]] - r["cv"]) for r in rows_cz)
    chk("1d **CV 得分与独立固定核实现一致**", dmax < 1e-12, f"max |ΔCV| {dmax:.2e}")

    # ② 尺度分别算：加修正列**不改变** λ_H
    _,_,lh_cz,_ = fit_joint(p1, 1.0, 0.0, L, None, True)
    Ht,_,_,_ = fit_joint(p1, TMP_MULT, 0.0, L, None, True); Ms = dir_basis(Ht)
    _,_,lh_j,ld_j = fit_joint(p1, 1.0, 1.0, L, Ms, False)
    chk("2 **加入 12 个修正列不改变 λ_H**（两个尺度分别算）",
        abs(lh_cz - lh_j) < 1e-18, f"{lh_cz:.6g} vs {lh_j:.6g}")
    chk("2b λ_Δ 由**仅 12 列**的设计算出且非零", ld_j > 0, f"λ_Δ={ld_j:.4g}")

    # ③ 基底与基
    chk("3 L_p: 3 个 DCT-II, 单位范数, **覆盖全 16 帧**",
        L.shape == (P_LAG, NLAG) and np.allclose(np.linalg.norm(L,axis=1), 1)
        and np.all(np.abs(L[0]) > 0))
    chk("3b M_q: 由 H̃ 的 **16×18** SVD 前 4 个右奇异向量得, 形状 (4,3,6), 单位范数",
        Ms.shape == (Q_DIR, NAX, NJ)
        and np.allclose(np.linalg.norm(Ms.reshape(Q_DIR,-1),axis=1), 1))

    # ④ 凸性/线性：ΔK 部分对 c 线性
    e0 = p1[0]; Z = design_D(e0.da, e0.gs, L, Ms)
    c1_, c2_ = np.random.default_rng(0).normal(size=NC), np.random.default_rng(1).normal(size=NC)
    lin = np.abs((Z @ (2*c1_ - 3*c2_)) - (2*(Z@c1_) - 3*(Z@c2_))).max()
    chk("4 修正项对 c **严格线性** ⇒ 联合问题凸", lin < 1e-10, f"{lin:.2e}")

    # ⑤ 因果性：修正设计只用过去
    da2 = e0.da.copy(); da2[100:] = 0
    Z2 = design_D(da2, e0.gs, L, Ms)
    chk("5 修正设计**因果**（把 t≥100 的 δa 清零不影响 t<100）",
        np.abs(Z[:100] - Z2[:100]).max() < 1e-12)

    # ⑥ 平局全序确定性
    fake=[{"m_H":1.0,"m_D":None,"c_zero":True,"cv":1.0},
          {"m_H":1.0,"m_D":10.0,"c_zero":False,"cv":1.0},
          {"m_H":10.0,"m_D":1.0,"c_zero":False,"cv":1.0}]
    key = lambda r: (r["cv"], 0 if r["c_zero"] else 1, -(r["m_D"] or np.inf), -r["m_H"])
    chk("6 平局时 **c≡0 分支胜出**", sorted(fake,key=key)[0]["c_zero"])
    chk("6b 同为非零时 **m_Δ 大者优先**",
        sorted([f for f in fake if not f["c_zero"]],key=key)[0]["m_D"]==10.0)

    # ⑦ g 的统计量只用训练折
    tr=[e for e in p1 if e.seed!=p1[0].seed]
    mu,sd = set_g(p1, tr); sm=np.concatenate([e.s for e in tr])
    chk("7 **g 的 s̄/σ_s 只由训练折算**", abs(mu-sm.mean())<1e-12 and abs(sd-sm.std())<1e-12)

    print(f"\n  ⇒ 自测 {sum(ok.values())}/{len(ok)} 通过")
    return 0 if all(ok.values()) else 1


def main() -> int:
    import gc
    gc.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    import fit_step1_candidate as S1
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    grid = list(S1.RIDGE_GRID)
    calib = json.loads((ROOT / "outputs/step1_H/step1_H_v1.json").read_text())["calibration"]
    R = np.asarray(calib["R"], np.float64); b = np.asarray(calib.get("b", [0, 0, 0]), np.float64)
    rend = IdealChunkRenderer()
    logs = S1.discover_logs(ROOT / "outputs/ab4", ("P1", "P2", "P3"))
    eps = {t: [S1.load_episode(p, t, R, b, rend, 550) for p in ps] for t, ps in logs.items()}
    S1.validate_protocol(eps)
    for t, ps in logs.items():
        for e, p in zip(eps[t], ps):
            pay = json.loads(p.read_text())
            cmd, _, _ = S1._render_logged_chunks(
                sorted(pay["chunks"], key=lambda c: int(c["start"])), rend, len(pay["frames"]))
            e.s = cmd[:len(e.da), 1] + cmd[:len(e.da), 2]
            e.S1 = S1
    print(f"载入 " + " · ".join(f"{t} {len(v)} 集" for t, v in sorted(eps.items()))
          + f"   网格 {grid} · ΔK {NC} 系数 · L_p=DCT-II×{P_LAG}（**覆盖全 16 帧**）")

    if a.self_test:
        return self_test(eps, S1, grid)

    print("\n" + "=" * 76 + "\n选参（P1 leave-one-seed-out；**① 与 ③ 各自独立选**）\n" + "=" * 76)
    best1, rows1 = cv_select(eps["P1"], grid, only_czero=True)
    best3, rows3 = cv_select(eps["P1"], grid, only_czero=False)
    print(f"  ① 固定核（5 个 c≡0 分支）  ⇒ m_H = {best1['m_H']:g}   CV {best1['cv']:.7g}")
    print(f"  ③ 本候选（全部 30 组合）  ⇒ m_H = {best3['m_H']:g} · "
          f"m_Δ = {best3['m_D']}   c≡0 = {best3['c_zero']}   CV {best3['cv']:.7g}")
    if best3["c_zero"]:
        print("  ⚠ **③ 的 CV 直接选中 `c≡0` 分支**")

    set_g([e for v in eps.values() for e in v], eps["P1"])
    H1, c1, L1, M1, _, _ = fold_fit(eps["P1"], best1["m_H"], 0.0, True)
    H3, c3, L3, M3, lh3, ld3 = fold_fit(eps["P1"], best3["m_H"],
                                        best3["m_D"] or 0.0, best3["c_zero"])
    print(f"\n  ③ 的 c = {np.round(c3, 6).tolist()}")
    print(f"     ‖c‖ = {np.linalg.norm(c3):.6g} · λ_H = {lh3:.4g} · λ_Δ = {ld3:.4g}")

    print("\n" + "=" * 76 + "\n评价\n" + "=" * 76)
    res = {"selection": {"fixed": best1, "kphi": best3},
           "cv_rows_fixed": rows1, "cv_rows_kphi": rows3,
           "c": c3.tolist(), "tiers": {}}
    verdict = {}
    for tier in ("P2", "P3"):
        role = "**主比较**" if tier == "P2" else "**次要：跨幅值压力验证**"
        print(f"\n  {tier} —— {role}")
        tv = {}
        for bn, band in SUBBANDS.items():
            acc = {}
            for e in eps[tier]:
                acc.setdefault(e.seed, []).append(
                    (energy(e, H1, c1, L1, M1, band), energy(e, H3, c3, L3, M3, band)))
            ds = {s: (np.mean([x[0] for x in v]) - np.mean([x[1] for x in v]))
                     / np.mean([x[0] for x in v]) for s, v in acc.items()}
            dbar = float(np.mean(list(ds.values()))); allpos = all(v > 0 for v in ds.values())
            ok = allpos and dbar >= GATE
            print(f"    [{bn:>8}] d̄ = {dbar*100:+6.2f}%   " +
                  " ".join(f"{s}:{v*100:+6.2f}%" for s, v in sorted(ds.items()))
                  + ("  ⇒ **通过**" if ok else ""))
            tv[bn] = {"d_bar": dbar, "d_s": {str(k): v for k, v in ds.items()},
                      "all_positive": allpos, "pass": bool(ok)}
        verdict[tier] = tv["0.5-9Hz"]["pass"]
        res["tiers"][tier] = tv
    res["verdict"] = {"P2_pass": verdict["P2"], "P3_pass": verdict["P3"],
                      "c_is_zero": bool(best3["c_zero"])}
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n→ {OUT}")
    print("\n" + "=" * 76 + "\n判读（预注册 §5，事前写死）\n" + "=" * 76)
    if best3["c_zero"]:
        print("  **CV 选中 `c ≡ 0`** ⇒ 只能报：")
        print("  **「本轮没有证据支持增加位形修正」**")
        print("  ⚠ **不临时扩模型、不改基底、不换调度量、不重跑。**")
    else:
        print(f"  P2 主判据（三个 d_s 全正 且 d̄ ≥ 5%）："
              f"{'**通过**' if verdict['P2'] else '**不通过**'}")
        print(f"  P3 压力验证：{'**通过**' if verdict['P3'] else '**不通过**'}")
    print("\n  ⚠ **阴性结果只针对这个预定基空间**（DCT-II 前 3 基 × 4 个 SVD 方向），"
          "不推广到其它滞后结构或调度量。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
