#!/usr/bin/env python3
"""C 的功效分析（2026-07-28，用户规格定稿）。

**四条前提（用户 2026-07-28 钉住）**
  1. D1/D2/混合 必须按**相同总峰值、总能量、机时**比较。三者都是 16 位姿 ×
     同样猝发时长 × 2 mrad 总峰值 ⇒ **机时相同**, 差别只在能量怎么分布。
     否则 D1 频点更多, 局部秩优势可能只是拿到了更多激励能量。
  2. 噪声协方差是**小样本**估计。仿真按**整次运行聚类重采样**并收缩,
     **不把频点当独立样本**。
  3. τ 的 0.0070 ms 只是"分支/模型形状/时钟均已知"下的**局部** CRLB,
     它低于 t_send_span ≈ 1.4 ms, **不能作为绝对时延精度** —— 报为条件信息量指标。
  4. D1 的局部秩优势**不能**写成"实验设计更优", 要等全局分支选择率与 LOPO 预测。

**流程**
  · 全局扫描 τ, **每个候选 τ 内重新优化对应层级的参数**(不先定分支再拟合)
  · 注入 +102 / −52 两个分支 + 平滑复数 shape error(幅值/相位斜率/三轴方向/位姿依赖,
    均按 B′ 实测量级)
  · 噪声用整次 A 差分块重采样, 保留频率间与三轴间相关
  · 16 训练条件内按**物理位姿分组**做 LOPO; 5 个 holdout 完全封存, 只用于最终一次验收
  · 报: 分支选对率 / 错误分支优势 / 层级选对率 / 过度升级率 / holdout 白化预测误差+CI

**放行门槛（用户定）**
  两个分支下选对率均 ≥95%; 错误层级升级率 ≤10%; holdout 改善的 CI 不跨零。
  若 D1 只提高局部秩、不提高这些指标 ⇒ 选混合设计。

**局限（必须随结果一起报）**
  本分析在**局部线性化**下做(参数进入 FRF 按 Jacobian 线性)。真实拟合是非线性的,
  故这里的选对率是**乐观上界**。

用法: python sim/power_analysis_C.py [--trials 200]
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

import analyze_probe_b2 as A
import train_pi_e2 as TP
from g_model import Rigid, load_ensemble
import select_poses_C as S

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/probe_c"
TAU_TRUE = {"+102ms": 0.102, "-52ms": -0.052}
TAU_GRID = np.arange(-0.250, 0.2501, 0.005)
PEAK = 0.002                      # 总峰值 2 mrad —— 三种设计相同
# B′ 实测量级的 shape error
SHAPE = {"log_mag_sd": np.log(3.0),        # |z| 实测跨 0.7–9.5 ⇒ 约 3× 散布
         "phase_slope_sd": 0.020,          # 附加时延散布 20 ms
         "dir_deg_sd": 15.0,               # 方向残差 0.12–0.55 ⇒ 7–33°
         "pose_dep": 1.0}                  # 位形依赖强度(1 = B′ 量级)


def designs():
    """三种设计: 频点集合。**总峰值固定** ⇒ 每音调幅值 ∝ 1/n_tones(等能量)。"""
    F_wide = np.arange(2.0, 8.01, 0.5)          # 13 音调
    F_narrow = np.array([6.0, 6.5, 7.0, 7.5])   # 4 音调
    return {"D1 全宽带": {"anchor": F_wide, "rest": F_wide, "n_anchor": 16},
            "D2 全窄带": {"anchor": F_narrow, "rest": F_narrow, "n_anchor": 16},
            "混合(4锚点宽带)": {"anchor": F_wide, "rest": F_narrow, "n_anchor": 4}}


def noise_blocks(freqs):
    """整次 A 差分块: 3 个不重叠对 × 窗 ⇒ 每块保留频率间与三轴间相关。"""
    BARE = {"A1": "replay_ep0_A1", "A2": "replay_ep0_A2", "A3": "replay_ep0_r2_A3",
            "A4": "replay_ep0_r2_A4", "A5": "replay_ep0_r3_A5", "A6": "replay_ep0_r3_A6"}
    seq = np.load(A.PB / "seq_ep0_j2_4mrad_p.npz", allow_pickle=True)
    Rc = {k: json.loads((A.PB / f"{v}.json").read_text()) for k, v in BARE.items()}
    W = {k: A.arm_windows(Rc[k], seq, 0) for k in Rc}
    blocks = []
    for w in [int(x) for x in seq["wins"]]:
        if not all(w in W[k] for k in W):
            continue
        LI = lambda k, f: A.lockin_at(W[k][w]["y"], W[k][w]["t"], f, W[k][w]["dt"])
        for a_, b_ in [("A1", "A2"), ("A3", "A4"), ("A5", "A6")]:
            v = np.array([(LI(b_, f) - LI(a_, f)) / 2 for f in freqs])   # (nf,3)
            blocks.append(v)
    return np.array(blocks)                     # (nblk, nf, 3) complex


def shape_error(H, F, phi, rng, scale=1.0):
    """平滑复数 shape error: 幅值 / 相位斜率 / 三轴方向 / 位姿依赖。"""
    # 位姿依赖: 用 φ 的一个投影驱动幅值偏差(B′ 的失配就是位形相关的)
    u = float(np.dot(phi[1:6], rng.normal(size=5)))
    lm = rng.normal(0, SHAPE["log_mag_sd"] * scale) + SHAPE["pose_dep"] * 0.3 * u
    mag = np.exp(lm)
    tau_x = rng.normal(0, SHAPE["phase_slope_sd"] * scale)
    ph = np.exp(-2j * np.pi * F * tau_x)[:, None]
    ang = np.radians(rng.normal(0, SHAPE["dir_deg_sd"] * scale))
    c, s = np.cos(ang), np.sin(ang)
    Hr = np.stack([c * H[:, 0] - s * H[:, 1], s * H[:, 0] + c * H[:, 1], H[:, 2]], 1)
    return mag * ph * Hr


def wilson(k, n, z=1.96):
    """选对率的 Wilson 置信区间(小样本比例的正确 CI)。"""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fairness_check(F_sets):
    """用**实际发送序列**核对公平性: max|u| / Σu²Δt / crest / 逐音调能量 / ZOH 衰减。"""
    from make_probe_b2 import schroeder, NEXC
    print(f"=== 公平性实测校验(实际发送序列) ===")
    print(f"  {'设计':>14}{'音调':>5}{'max|u| mrad':>13}{'Σu²Δt':>12}{'crest':>7}"
          f"{'每音调能量':>11}{'ZOH均衰减':>10}")
    ref = None
    for nm, F in F_sets.items():
        w, _, cf = schroeder(np.asarray(F, float), NEXC)
        u = w * PEAK
        E = float(np.sum(u ** 2) / 20.0)
        if ref is None:
            ref = E
        zoh = float(np.mean(np.sinc(np.asarray(F, float) / 20.0)))   # sinc(πf/fs)/(πf/fs)
        print(f"  {nm:>14}{len(F):>5}{np.abs(u).max()*1e3:>13.4f}{E:>12.4e}{cf:>7.2f}"
              f"{E/len(F):>11.3e}{zoh:>10.4f}")
    print(f"  Σu²Δt 相对差异 = "
          f"{max(abs(float(np.sum((schroeder(np.asarray(F,float),NEXC)[0]*PEAK)**2)/20.0)/ref-1) for F in F_sets.values())*100:.3f}%"
          f"  (要求 <1%)")
    print()


def seq_energy(F, peak=PEAK):
    """该频点集在给定峰值下的实际总能量 Σu²Δt(20Hz 采样)。"""
    from make_probe_b2 import schroeder, NEXC
    w, _, _ = schroeder(np.asarray(F, float), NEXC)
    return float(np.sum((w * peak) ** 2) / 20.0)


def amp_for(F, mode, E_star):
    """按归一模式给出该频点集的峰值。

    'energy': 等能量 —— 峰值 = PEAK·√(E*/E(PEAK)); 2 mrad 是**安全上限**不是额度
    'peak'  : 等峰值 —— 峰值恒为 PEAK, 能量可不同(真实安全约束下的设计效率)
    """
    if mode == "peak":
        return PEAK
    return PEAK * np.sqrt(E_star / seq_energy(F, PEAK))


def model_and_U(g, poses, R, F_of, amp_of_pose):
    """H_model(f) 与**实际序列**的锁相系数 U(f)。

    用户 2026-07-28: 不再按音调数手动缩放噪声。直接用 Y=H·U_actual+N ⇒
    每频点幅值、crest、能量归一、ZOH 衰减都自然进入, 不需要 √(n/4) 近似。
    **ZOH 差异不应消掉** —— 它是频率配置经真实 20Hz 接口后的固有结果。
    """
    Hm, Um = {}, {}
    tgc = {}
    for pi, p in enumerate(poses):
        F = F_of[pi]
        amp = amp_of_pose[pi]
        for j in (1, 2):
            cmd7 = S.synth_cmd(p)
            da, w0 = S.probe_da(len(cmd7), j, amp=amp, seed=pi * 10 + j, freqs=F)
            # H = Y_model/U (frf_of 已除 U); U 单独算, 走同一带限求值+锁相
            Hm[(pi, j)] = S.frf_of(g, cmd7, da, w0, R, F)
            from make_probe_b2 import NEXC
            tg = np.arange(0, (NEXC + 30) / S.FPS, 0.01)
            uu = A._bl_eval(da[:, j], w0 / S.FPS + tg)
            Um[(pi, j)] = np.array([A.lockin_at(uu, tg, f) for f in F])
    return Hm, Um


def anchor_search(poses, grp, g, R, nb_cache, trials_sel=120, trials_val=120):
    """锚点选择: 从 **7 个物理位姿组**里枚举 C(7,4)=35 个四锚点集合。

    · 四个锚点必须来自**四个不同物理位姿**(避免 grip/速度配对重复占位)
    · 排序指标用**最坏情况**: S(A) = min_{τ,ε} Q_5%[ΔSSE(A,τ,ε)]
    · 用一批 seed 选择, 另一批**独立** seed 验证 ⇒ 避免锚点选择偏差
    """
    F_wide = np.arange(2.0, 8.01, 0.5)
    F_narrow = np.array([6.0, 6.5, 7.0, 7.5])
    E_star = min(seq_energy(F_wide), seq_energy(F_narrow))
    gkeys = list(grp)
    print(f"\n=== 锚点选择: C({len(gkeys)},4) = "
          f"{len(list(itertools.combinations(range(len(gkeys)), 4)))} 个集合 ===")
    print(f"  四锚点来自**四个不同物理位姿组**; 指标 S(A)=min_{{τ,ε}} Q5%[ΔSSE]")
    print(f"  选择 {trials_sel} seed / 验证 {trials_val} **独立** seed")

    def evaluate(anchor_groups, seed0, ntr):
        anchors = {grp[gkeys[gi]][0] for gi in anchor_groups}   # 每组取首条件
        F_of = {pi: (F_wide if pi in anchors else F_narrow) for pi in range(len(poses))}
        amp_of = {pi: amp_for(F_of[pi], "energy", E_star) for pi in range(len(poses))}
        Hm, Um = model_and_U(g, poses, R, F_of, amp_of)
        keys = sorted(Hm)
        worst = np.inf
        for br, tt in TAU_TRUE.items():
            ds = []
            for t_ in range(ntr):
                rr = np.random.default_rng(seed0 + t_)
                obs, base = {}, {}
                for (pi, j) in keys:
                    Fp = F_of[pi]
                    H0 = Hm[(pi, j)]
                    Ht = shape_error(H0, Fp, S.phi_of(poses[pi]), rr)
                    Ht = Ht * np.exp(-2j * np.pi * Fp * tt)[:, None]
                    nb = nb_cache[tuple(Fp)]
                    U = Um[(pi, j)]
                    obs[(pi, j)] = Ht + nb[rr.integers(len(nb))] / np.where(
                        np.abs(U) < 1e-15, 1e-15, U)[:, None]
                    base[(pi, j)] = H0
                curve = []
                for tau in TAU_GRID:
                    r2 = 0.0
                    for k in keys:
                        Fp = F_of[k[0]]
                        Hb = base[k] * np.exp(-2j * np.pi * Fp * tau)[:, None]
                        gf = np.vdot(Hb.ravel(), obs[k].ravel()) / max(
                            np.vdot(Hb.ravel(), Hb.ravel()).real, 1e-30)
                        r2 += np.linalg.norm(obs[k] - gf * Hb) ** 2
                    curve.append(r2)
                curve = np.array(curve)
                other = TAU_TRUE["-52ms" if br == "+102ms" else "+102ms"]
                io = int(np.abs(TAU_GRID - other).argmin())
                lo, hi_ = max(0, io - 6), min(len(curve), io + 7)
                ds.append(curve[lo:hi_].min() - curve.min())
            worst = min(worst, float(np.percentile(ds, 5)))       # Q5% 后取两分支最小
        return worst

    combos = list(itertools.combinations(range(len(gkeys)), 4))
    sel = [(c, evaluate(c, 5000, trials_sel)) for c in combos]
    sel.sort(key=lambda x: -x[1])
    print(f"\n  {'排名':>4}{'锚点组':>16}{'S(A) 选择批':>14}{'S(A) 验证批':>14}{'覆盖':>22}")
    top = []
    for r, (c, sv) in enumerate(sel[:3] + [sel[-1]]):
        vv = evaluate(c, 90000, trials_val)                       # **独立** seed 验证
        cov = "/".join(sorted({S.stratified_holdout.__doc__ and "" or ""} or {""}))
        qs = [f"({np.degrees(gkeys[gi][0]):.0f},{np.degrees(gkeys[gi][1]):.0f})"
              for gi in c]
        lab = "最差" if r == 3 else f"#{r+1}"
        print(f"  {lab:>4}{str(c):>16}{sv:>14.3e}{vv:>14.3e}   {' '.join(qs)}")
        if r < 3:
            top.append((c, sv, vv))
    return top, gkeys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    OUT.mkdir(parents=True, exist_ok=True)

    cands, klo, khi = S.candidate_poses()
    Ph_all = np.array([S.phi_of(c) for c in cands])
    tr = S.greedy_select(cands, 16, Ph_all)
    ho, cls = S.stratified_holdout(cands, tr, Ph_all, 5)
    poses = [cands[i] for i in tr]
    hposes = [cands[i] for i in ho]
    # 物理位姿分组(LOPO 按 **(q2,q3)** 分组, 不按条件)
    grp = {}
    for k, i in enumerate(tr):
        grp.setdefault(S.key_qq(cands[i]), []).append(k)
    print(f"训练 16 条件 → **{len(grp)} 个物理位姿组** (LOPO 按组留出, 非按条件)")
    print(f"  组大小 {[len(v) for v in grp.values()]}")
    print(f"holdout 5 条件, **完全封存**, 只用于最终一次验收\n")

    rig = Rigid(); models = load_ensemble(rigid=rig)
    man = json.loads((TP.LROOT / "pp2-deploy" / "manifest.json").read_text())
    R = np.array(man["R"], np.float32)
    g = models[len(models) // 2]

    print(f"=== 等峰值/等能量/等机时 前提核对 ===")
    for nm, d in designs().items():
        n_t = len(d["anchor"]) * d["n_anchor"] + len(d["rest"]) * (16 - d["n_anchor"])
        print(f"  {nm:>16}: 锚点 {d['n_anchor']} 位姿 × {len(d['anchor'])} 音调 + "
              f"其余 {16-d['n_anchor']} × {len(d['rest'])} 音调"
              f"   总峰值 {PEAK*1e3:.0f} mrad(相同)  猝发时长相同 ⇒ **机时相同**")
    print(f"  ⇒ 差别只在**能量如何分布**, 不在总量。\n")
    fairness_check({"D1 宽带": np.arange(2.0, 8.01, 0.5),
                    "D2 窄带": np.array([6.0, 6.5, 7.0, 7.5])})

    E_STAR = [min(seq_energy(np.arange(2.0, 8.01, 0.5)),
                  seq_energy(np.array([6.0, 6.5, 7.0, 7.5])))]
    MODE = ["energy"]
    print(f"  E* = min(E_D1, E_D2) = {E_STAR[0]:.4e}")
    for nm_, F_ in [("D1 宽带", np.arange(2.0, 8.01, 0.5)),
                    ("D2 窄带", np.array([6.0, 6.5, 7.0, 7.5]))]:
        print(f"    {nm_}: 等能量峰值 = {amp_for(F_, 'energy', E_STAR[0])*1e3:.3f} mrad"
              f"   (等峰值模式恒为 {PEAK*1e3:.2f})")
    print()

    ALL = {}
    for MODE[0] in ("energy", "peak"):
        lbl_mode = ("**主结果**: 等能量·等机时·峰值≤2mrad ⇒ 隔离频率分配效果"
                    if MODE[0] == "energy" else
                    "**次结果**: 等峰值2mrad·等机时·能量可不同 ⇒ 真实安全约束下的设计效率")
        print(f"\n{'='*72}\n{lbl_mode}\n{'='*72}")
        res = ALL[MODE[0]] = {}
        for nm, d in designs().items():
            # **逐位姿的频段**: 前 n_anchor 个位姿用 anchor 频段, 其余用 rest 频段。
            # (先前 bug: 三种设计都用了 anchor ⇒ 混合与 D1 数字完全相同, 是假的)
            F_of = {pi: (d["anchor"] if pi < d["n_anchor"] else d["rest"])
                    for pi in range(len(poses))}
            NB_F = np.array([6.0, 6.5, 7.0, 7.5])
            nb = noise_blocks(NB_F)
            amp_of_pose = {pi: amp_for(F_of[pi], MODE[0], E_STAR[0])
                           for pi in range(len(poses))}
            Hm, Um = model_and_U(g, poses, R, F_of, amp_of_pose)
            keys = sorted(Hm)
            rows = {}
            for br, tt in TAU_TRUE.items():
                hit, dsse = 0, []
                for t_ in range(a.trials):
                    rr = np.random.default_rng(1000 + t_)
                    obs, base = {}, {}
                    for (pi, j) in keys:
                        Fp = F_of[pi]
                        H0 = Hm[(pi, j)]
                        Ht = shape_error(H0, Fp, S.phi_of(poses[pi]), rr)
                        Ht = Ht * np.exp(-2j * np.pi * Fp * tt)[:, None]
                        # **Y = H·U_actual + N** ⇒ H_obs = H + N/U。幅值/crest/能量归一/
                        # ZOH 全部自然进入, 不再手动按音调数缩放噪声。
                        blk = nb[rr.integers(len(nb))]                 # **整块**重采样
                        N = blk[rr.integers(len(blk), size=len(Fp))]
                        U = Um[(pi, j)]
                        obs[(pi, j)] = Ht + N / np.where(np.abs(U) < 1e-15, 1e-15, U)[:, None]
                        base[(pi, j)] = H0
                    # 全局 τ 扫描, 每个 τ 内重优化(此处 L1 线性化: 每位姿一个复增益)
                    curve = []
                    for tau in TAU_GRID:
                        r2 = 0.0
                        for k in keys:
                            Fp = F_of[k[0]]
                            Hb = base[k] * np.exp(-2j * np.pi * Fp * tau)[:, None]
                            num = np.vdot(Hb.ravel(), obs[k].ravel())
                            den = np.vdot(Hb.ravel(), Hb.ravel()).real
                            gfit = num / max(den, 1e-30)
                            r2 += np.linalg.norm(obs[k] - gfit * Hb) ** 2
                        curve.append(r2)
                    curve = np.array(curve)
                    i_best = int(curve.argmin())
                    tb = TAU_GRID[i_best]
                    ok = abs(tb - tt) < 0.030
                    hit += ok
                    # **白化 ΔSSE**(绝对量, 不做相对化) —— "劣势%" 在正确残差趋零时会爆
                    other = TAU_TRUE["-52ms" if br == "+102ms" else "+102ms"]
                    io = int(np.abs(TAU_GRID - other).argmin())
                    lo, hi_ = max(0, io - 6), min(len(curve), io + 7)
                    dsse.append(curve[lo:hi_].min() - curve.min())
                lo_ci, hi_ci = wilson(hit, a.trials)
                rows[br] = {"branch_hit": hit / a.trials,
                            "wilson_lo": lo_ci, "wilson_hi": hi_ci,
                            "dsse_median": float(np.median(dsse)),
                            "dsse_p05": float(np.percentile(dsse, 5))}
            res[nm] = rows
            for b, v in rows.items():
                print(f"  {nm:>16} {b:>7}: 选对率 {v['branch_hit']*100:5.1f}% "
                      f"[Wilson {v['wilson_lo']*100:.1f}, {v['wilson_hi']*100:.1f}]"
                      f"   ΔSSE 中位 {v['dsse_median']:.3e}  5%分位 {v['dsse_p05']:.3e}")

    print(f"\n=== 放行门槛核对: **两分支选对率的 95%CI 下界 ≥95%** ===")
    for nm, rows in res.items():
        lb = min(v["wilson_lo"] for v in rows.values())
        print(f"  {nm:>16}: {'✓ 过' if lb >= 0.95 else '✗ 未过'}"
              f"  (两分支 CI 下界的最小值 {lb*100:.1f}%)")
    print(f"\n⚠ 局限(必须随结果一起报):")
    print(f"  · τ 扫描内的重优化目前是**每位姿一个复增益**的**替代模型**, 不是完整 L1/L2/L3。")
    print(f"    它**不必然是乐观上界** —— 可能帮错误分支, 也可能掩盖正确分支下的模型失配。")
    print(f"  · 层级选对率 / 过度升级率 / group-LOPO / holdout 预测 **尚未纳入**。")
    print(f"  · shape error 目前是**随机**扰动; 尚未做**受 B′ 幅值约束的最不利搜索**。")
    print(f"  · 混合的 4 个锚点尚未按 φ/时序信息量选取, 也未对多个锚点集合复测。")
    print(f"  ⇒ 现阶段**不能**据此宣布任何设计更优(用户第4条)。")
    (OUT / "power_C.json").write_text(json.dumps(
        {"trials": a.trials, "results": res, "n_pose_groups": len(grp),
         "shape_error": SHAPE, "peak_mrad": PEAK * 1e3}, indent=1, ensure_ascii=False))
    print(f"\n→ {(OUT / 'power_C.json').relative_to(ROOT.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
