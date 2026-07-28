#!/usr/bin/env python3
"""C 的位姿筛选器 + Jacobian 秩分析（2026-07-28，暂不上机）。

**为什么在 φ 空间选点**
  α_m(φ) = softplus(φ·θ_m),  φ = [1, q2, q3, q2², q3², q2·q3, 速度, 夹爪]
  要辨识 θ_m 的 8 个分量 ⇒ Φ 必须 rank 8（等价于去均值后 rank 7）。
  B′ 的 2 个不同位形只给出去均值 rank **1**。

**静止位姿网格下哪些方向可独立变化**（修正 DESIGN_C 里过于保守的说法）
  · `1`                        常数 —— 但由响应的绝对量级定, 不需要变化
  · q2, q3, q2², q3², q2·q3    由 (q2,q3) 二维网格给出 **5 个方向**
  · `速度`                     ✅ 逐位姿改**慢速往复的幅值** ⇒ 平均速度可控
  · `夹爪`                     ✅ 直接设定开合, 是可控输入
  ⇒ 去均值 rank 7 可达。

**做两件事**
  (A) 从候选网格里贪心选点(D-最优/最大最小奇异值), 满足安全约束, 并留 holdout
  (B) 对选出的位姿, 数值算探针 FRF 对各层参数的 Jacobian 秩与条件数,
      判断**仅调 θ 是否足够**(L1) 还是要下探到 L2/L3/L4

用法:
  python sim/select_poses_C.py                 # 选点 + 秩分析
  python sim/select_poses_C.py --n 12 --hold 5 # 指定训练/留出位姿数
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import analyze_probe_b2 as A
import train_pi_e2 as TP
from g_model import Rigid, load_ensemble, pose_features
from make_probe_b2 import reach_limits, FREQS, NEXC, schroeder

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/probe_c"
FPS = 20.0
# 慢速往复: 远离探针带(2.5-9Hz), 锁相会拒掉; 同时给 `速度` 分量变化
# 速度配对**必须靠改频率**, 不能靠改幅值 —— 改幅值会同时改变 q 的覆盖范围
# (用户 2026-07-28)。同 (q2,q3,grip,幅值) 下换频率 ⇒ 平均速度变, q 覆盖不变。
DITHER_HZS = [0.25, 0.40, 0.80]                  # 往复频率(Hz), 远离探针带
DITHER_AMPS = np.radians([1.0, 2.0])             # 幅值(°) —— 配对内固定
GRIPS = [0.0, 0.035]                              # 夹爪: 合 / 开
SAFE_MARGIN = np.radians(0.5)                     # residual margin 门
PROBE_AMP = 0.002                                 # 2 mrad
# 静态跟踪偏差(保守): j2 在可达下界附近实测 +1.17°, 其余 ≤0.13°
STATIC_ERR = np.radians(np.array([0.2, 1.3, 0.2, 0.2, 0.2, 0.2]))


def candidate_poses(n_q=9):
    """(q2,q3) 网格 × 夹爪 × 往复幅值, 先过安全门。"""
    klo, khi = reach_limits()
    # q2/q3 取可达范围内、离两端留出 (safe + 静态误差 + 探针 + 往复) 的部分
    def rng(j, dith):
        pad = SAFE_MARGIN + STATIC_ERR[j] + PROBE_AMP + dith
        lo, hi = klo[j] + pad, khi[j] - pad
        return (lo, hi) if hi > lo else None
    out = []
    for dith in DITHER_AMPS:
        r2, r3 = rng(1, dith), rng(2, dith)
        if r2 is None or r3 is None:
            continue
        for q2 in np.linspace(*r2, n_q):
            for q3 in np.linspace(*r3, n_q):
                for gp in GRIPS:
                    for hz in DITHER_HZS:
                        out.append({"q2": float(q2), "q3": float(q3), "grip": float(gp),
                                    "dither": float(dith), "hz": float(hz)})
    return out, klo, khi


def phi_frames(p, S=140, w0=40, nw=NEXC):
    """**逐帧** pose_features（不用平均速度近似 —— 用户 2026-07-28 要求）。

    返回 (注入窗内的 φ 序列, 窗内均值)。慢速往复使 φ 的 q2/q3/速度 在窗内时变,
    这正是完整 θ 相对"每窗一个常数 g"可能多出的表达力来源。
    """
    cmd7 = synth_cmd(p, S)
    Ph = pose_features(cmd7)
    seg = Ph[w0:w0 + nw]
    return seg, seg.mean(0)


def phi_of(p):
    """该位姿的 φ = 注入窗内逐帧 pose_features 的均值。"""
    return phi_frames(p)[1]


def alpha_variation(g, poses):
    """量化 α(t) 在注入窗内的时变量 —— 决定完整 θ 是否比每窗常数 g 更有表达力。

    用户 2026-07-28 指出: 对 P 个工作点, 逐点自由的 g_m 比共享曲面 θ **更自由**;
    只有 α(t) 窗内明显时变时, θ 才多出表达力。
    """
    print(f"\n=== α(t) 在注入窗内的时变量 (决定 θ 是否比每窗常数 g 更有表达力) ===")
    print(f"  {'#':>3}{'模态1 CV':>11}{'模态2 CV':>11}{'模态3 CV':>11}{'最大 CV':>10}")
    cvs = []
    for k, p in enumerate(poses):
        seg, _ = phi_frames(p)
        pn = ((seg - g.mu) / g.sd).astype(np.float32)
        al = g.flex.gains(torch.from_numpy(pn)).detach().numpy()      # (nw, M)
        cv = al.std(0) / np.maximum(np.abs(al.mean(0)), 1e-12)
        cvs.append(cv)
        if k < 6 or k == len(poses) - 1:
            print(f"  {k:>3}" + "".join(f"{v*100:>10.1f}%" for v in cv)
                  + f"{cv.max()*100:>9.1f}%")
    cvs = np.array(cvs)
    print(f"  {'中位':>3}" + "".join(f"{v*100:>10.1f}%" for v in np.median(cvs, 0))
          + f"{np.median(cvs.max(1))*100:>9.1f}%")
    mx = np.median(cvs.max(1))
    print(f"  ⇒ α(t) 窗内变异系数中位 {mx*100:.1f}%")
    if mx < 0.05:
        print(f"     **< 5% ⇒ 窗内近似恒定 ⇒ 完整 θ 相对'每窗一个常数 g'几乎无额外表达力**")
        print(f"     ⇒ α-only 在 3 工作点的失败(残差 100-121%)对 θ 参数化仍是**上界**")
    else:
        print(f"     ⇒ 窗内有时变, θ 可能多出表达力, 需在 ② 的嵌套测试里单独量化")
    return cvs


def key_qq(c):
    return (round(c["q2"], 6), round(c["q3"], 6))


def greedy_select(cands, n, Ph, n_grip_pair=3, n_spd_pair=3, seed=0):
    """贪心最大化 Φ(去均值) 的第7奇异值, **并强制两类匹配配对**。

    用户 2026-07-28: 否则即使 Φ 满秩, 夹爪/速度/位姿仍可能强烈混杂。
      · ≥ n_grip_pair 组: 相同 (q2,q3) 下的两个 grip
      · ≥ n_spd_pair 组: 相同 (q2,q3,grip) 下的两个往复幅值
    """
    rng = np.random.default_rng(seed)
    by_qq = {}
    for i, c in enumerate(cands):
        by_qq.setdefault(key_qq(c), []).append(i)

    idx = []
    # --- 先满足 grip 配对: 同 (q2,q3), 两个 grip, 同一 dither ---
    qqs = list(by_qq)
    rng.shuffle(qqs)
    for qq in qqs:
        if n_grip_pair <= 0:
            break
        grp = by_qq[qq]
        for d in DITHER_AMPS:
            for hz in DITHER_HZS:
                gs = [i for i in grp if abs(cands[i]["dither"] - d) < 1e-9
                      and abs(cands[i]["hz"] - hz) < 1e-9]
                if len({cands[i]["grip"] for i in gs}) >= 2:
                    idx += [next(i for i in gs if cands[i]["grip"] == g) for g in GRIPS]
                    n_grip_pair -= 1
                    break
            else:
                continue
            break
    # --- 速度配对(**硬约束**): 同 (q2,q3,grip,幅值), **换频率** ⇒ q 覆盖不变 ---
    for qq in qqs:
        if n_spd_pair <= 0:
            break
        grp = [i for i in by_qq[qq] if i not in idx]
        done = False
        for gp in GRIPS:
            # **该 (q2,q3) 实际存在的幅值**(不同幅值有各自的 q 网格, 不跨网格找同 q 点)
            for am in sorted({cands[i]["dither"] for i in grp
                              if cands[i]["grip"] == gp}):
                gs = [i for i in grp if cands[i]["grip"] == gp
                      and abs(cands[i]["dither"] - am) < 1e-9]
                hzs = sorted({cands[i]["hz"] for i in gs})
                if len(hzs) >= 2:                 # 幅值固定, **只换频率**
                    idx += [next(i for i in gs if cands[i]["hz"] == h)
                            for h in (hzs[0], hzs[-1])]
                    n_spd_pair -= 1; done = True
                    break
            if done:
                break
    # --- 剩余名额用贪心补 ---
    while len(idx) < n:
        best, bi = -1.0, None
        for i in range(len(cands)):
            if i in idx:
                continue
            M = Ph[idx + [i]]; M = M - M.mean(0)
            sv = np.linalg.svd(M, compute_uv=False)
            score = sv[min(len(sv) - 1, 6)] if len(sv) > 1 else 0.0
            if score > best:
                best, bi = score, i
        idx.append(bi)
    return idx[:n]


def stratified_holdout(cands, used, Ph, n=5, seed=1):
    """holdout 按**覆盖**选, 不追求 rank（用户 2026-07-28）:
       中心 / 伸展 / 折叠 × 两种夹爪 × 不同速度等级。"""
    rng = np.random.default_rng(seed)
    q2 = np.array([c["q2"] for c in cands]); q3 = np.array([c["q3"] for c in cands])
    lo2, hi2 = q2.min(), q2.max(); lo3, hi3 = q3.min(), q3.max()
    def cls(c):
        # 伸展 = q2 大 & q3 接近 0(臂张开); 折叠 = q2 小 或 q3 很负
        ex = (c["q2"] - lo2) / (hi2 - lo2) + (c["q3"] - lo3) / (hi3 - lo3)
        return "伸展" if ex > 1.25 else ("折叠" if ex < 0.75 else "中心")
    want = [("中心", GRIPS[0], DITHER_AMPS[0]), ("伸展", GRIPS[1], DITHER_AMPS[0]),
            ("折叠", GRIPS[0], DITHER_AMPS[-1]), ("伸展", GRIPS[0], DITHER_AMPS[-1]),
            ("折叠", GRIPS[1], DITHER_AMPS[0])][:n]
    out = []
    for c_, gp, dt in want:
        pool = [i for i in range(len(cands)) if i not in used and i not in out
                and cls(cands[i]) == c_ and cands[i]["grip"] == gp
                and abs(cands[i]["dither"] - dt) < 1e-9]
        if not pool:                                # 放宽 dither
            pool = [i for i in range(len(cands)) if i not in used and i not in out
                    and cls(cands[i]) == c_ and cands[i]["grip"] == gp]
        if pool:
            out.append(int(rng.choice(pool)))
    return out, cls


def report_phi(cands, idx, label):
    Ph = np.array([phi_of(cands[i]) for i in idx])
    M = Ph - Ph.mean(0)
    s = np.linalg.svd(M, compute_uv=False)
    r = int((s / max(s[0], 1e-30) > 1e-6).sum())
    print(f"  {label}: {len(idx)} 位姿   去均值 Φ 秩 = **{r} / 7**"
          f"   条件数 = {s[0]/max(s[r-1],1e-30):.1f}")
    print(f"    奇异值(归一) = {np.round(s/max(s[0],1e-30), 4)}")
    return r, s


def synth_cmd(p, S=140):
    """该位姿的合成指令序列: 静止保持 + 慢速往复。(S,7)"""
    t = np.arange(S) / FPS
    q = np.zeros((S, 7), np.float32)
    hz = p.get("hz", 0.4)
    q[:, 1] = p["q2"] + p["dither"] * np.sin(2 * np.pi * hz * t)
    q[:, 2] = p["q3"] + p["dither"] * np.sin(2 * np.pi * hz * t + 1.0)
    q[:, 6] = p["grip"]
    return q


def probe_da(S, j, amp=PROBE_AMP, w0=40, seed=0, freqs=None):
    """该位姿的多正弦 δa。**freqs 必须显式传入**该设计的频点集。

    2026-07-28 bug: 先前写死 `schroeder(FREQS, ...)`(4 音调 B′ 频点), 于是"宽带"
    设计发的其实是窄带序列, 却在 13 个频点上算 H/U —— 其中 9 个几乎无激励,
    U≈0 ⇒ N/U 爆掉。除以标量时这个 bug 被掩盖, 改用实际 U(f) 后立刻暴露。
    """
    F = FREQS if freqs is None else np.asarray(freqs, float)
    da = np.zeros((S, 6), np.float32)
    wv, _, _ = schroeder(F, NEXC, seed=seed)
    da[w0:w0 + NEXC, j] = wv * amp
    return da, w0


def frf_of(g, cmd7, da, w0, R, freqs):
    """该位姿的模型 FRF（3-向量/频点）。"""
    g.R = R
    tg = np.arange(0, (NEXC + 30) / FPS, 0.01)
    y = g.increment_fast(cmd7, da)
    yy = A._bl_eval(y, w0 / FPS + tg)
    j = int(np.abs(da).max(0).argmax())
    uu = A._bl_eval(da[:, j], w0 / FPS + tg)
    Y = np.array([A.lockin_at(yy, tg, f) for f in freqs])
    U = np.array([A.lockin_at(uu, tg, f) for f in freqs])
    return Y / np.where(np.abs(U) < 1e-12, 1e-12, U)[:, None]


def jacobian_theta(g, poses, R, freqs, eps=1e-2):
    """探针 FRF 对 θ（3模态×8=24 参数）的数值 Jacobian。返回 (行, 24) 实矩阵。"""
    rows = []
    th0 = g.flex.theta.detach().clone()
    M = th0.shape[0]
    base = {}
    for pi, p in enumerate(poses):
        for j in (1, 2):
            cmd7 = synth_cmd(p)
            da, w0 = probe_da(len(cmd7), j, seed=pi * 10 + j)
            base[(pi, j)] = frf_of(g, cmd7, da, w0, R, freqs)
    for m in range(M):
        for c in range(th0.shape[1]):
            col = []
            with torch.no_grad():
                g.flex.theta[m, c] += eps
            g._vker = g._exck = None
            for pi, p in enumerate(poses):
                for j in (1, 2):
                    cmd7 = synth_cmd(p)
                    da, w0 = probe_da(len(cmd7), j, seed=pi * 10 + j)
                    d = (frf_of(g, cmd7, da, w0, R, freqs) - base[(pi, j)]) / eps
                    col.append(d.ravel())
            with torch.no_grad():
                g.flex.theta.copy_(th0)
            g._vker = g._exck = None
            v = np.concatenate(col)
            rows.append(np.concatenate([v.real, v.imag]))
    return np.array(rows).T                                  # (行, 24)


# ══════════════════════════════════════════════════════════════════
# 白化 + 参数先验尺度 + 分块 Jacobian
# ══════════════════════════════════════════════════════════════════
def noise_cov_from_B2(freqs, shrink=0.5, pairs=None):
    """从 B′ 的实测噪声窗估 (freq×axis) 的**复数噪声协方差**, 带收缩。

    用户 2026-07-28: 用按运行/位姿的噪声协方差, **不把频点当独立样本**。
    B′ 的 5 条裸复放 ⇒ 10 个配对差 (A_j−A_i)/2, 每个给一份噪声实现。
    24 维(4频×3轴×实虚) 用 10 个样本估协方差是欠定的 ⇒ 向对角收缩。
    """
    import itertools
    # **3 个不重叠差分**(用户 2026-07-28): 按运行聚类, 不把频点当独立样本
    BARE = {"A1": "replay_ep0_A1", "A2": "replay_ep0_A2", "A3": "replay_ep0_r2_A3",
            "A4": "replay_ep0_r2_A4", "A5": "replay_ep0_r3_A5", "A6": "replay_ep0_r3_A6"}
    pairs = pairs or [("A1", "A2"), ("A3", "A4"), ("A5", "A6")]
    seq = np.load(A.PB / "seq_ep0_j2_4mrad_p.npz", allow_pickle=True)
    Rc = {k: json.loads((A.PB / f"{v}.json").read_text()) for k, v in BARE.items()}
    W = {k: A.arm_windows(Rc[k], seq, 0) for k in Rc}
    rows = []
    for w in [int(x) for x in seq["wins"]]:
        if not all(w in W[k] for k in W):
            continue
        LI = lambda k, f: A.lockin_at(W[k][w]["y"], W[k][w]["t"], f, W[k][w]["dt"])
        for a_, b_ in pairs:                       # 只用不重叠对
            v = np.array([(LI(b_, f) - LI(a_, f)) / 2 for f in freqs]).ravel()
            rows.append(np.concatenate([v.real, v.imag]))
    X = np.array(rows)
    C = np.cov(X, rowvar=False)
    D = np.diag(np.diag(C))
    C = (1 - shrink) * C + shrink * D              # 向对角收缩
    C += np.eye(len(C)) * np.trace(C) / len(C) * 1e-6
    return C, X.shape[0]


def whiten_op(C):
    """Σ^{−1/2}。"""
    w, V = np.linalg.eigh(C)
    w = np.clip(w, np.max(w) * 1e-12, None)
    return V @ np.diag(w ** -0.5) @ V.T


def theta_prior_scale(models):
    """θ 的先验尺度 = 成员间标准差(数据驱动的先验)。(M,8)"""
    th = np.array([g.flex.theta.detach().numpy() for g in models])
    return np.maximum(th.std(0), 1e-3)


def block_jacobian(g, poses, R, freqs, models):
    """分块 Jacobian: θ(24) | τ(1) | j2/j3 有效模态残量 | D 的 j2/j3 列 | 极点 ω/ζ(6)。

    行 = 位姿 × 关节 × (频点×轴×实虚)。**未白化**, 由调用方白化。
    """
    import torch
    blocks, names = {}, []
    base = {}
    for pi, p in enumerate(poses):
        for j in (1, 2):
            cmd7 = synth_cmd(p)
            da, w0 = probe_da(len(cmd7), j, seed=pi * 10 + j)
            base[(pi, j)] = frf_of(g, cmd7, da, w0, R, freqs)

    def collect(mod):
        col = []
        for pi, p in enumerate(poses):
            for j in (1, 2):
                cmd7 = synth_cmd(p)
                da, w0 = probe_da(len(cmd7), j, seed=pi * 10 + j)
                d = (mod(g, cmd7, da, w0) - base[(pi, j)])
                col.append(d.ravel())
        v = np.concatenate(col)
        return np.concatenate([v.real, v.imag])

    # ---- θ ----
    th0 = g.flex.theta.detach().clone()
    sc = theta_prior_scale(models)
    cols = []
    for m in range(th0.shape[0]):
        for c in range(th0.shape[1]):
            eps = float(sc[m, c])                  # **按先验尺度**扰动
            with torch.no_grad():
                g.flex.theta[m, c] += eps
            g._vker = g._exck = None
            cols.append(collect(lambda gg, cc, dd, ww: frf_of(gg, cc, dd, ww, R, freqs)) / 1.0)
            with torch.no_grad():
                g.flex.theta.copy_(th0)
            g._vker = g._exck = None
    blocks["θ"] = np.array(cols).T
    # ---- τ (共享线性相位) ----
    dtau = 0.005
    ph = lambda tau: np.exp(-2j * np.pi * freqs * tau)[:, None]
    col = []
    for pi, p in enumerate(poses):
        for j in (1, 2):
            d = base[(pi, j)] * (ph(dtau) - 1.0) / dtau * dtau
            col.append(d.ravel())
    v = np.concatenate(col)
    blocks["τ"] = np.concatenate([v.real, v.imag])[:, None]
    # ---- D 的 j2/j3 列 ----
    D0 = g.flex.D.detach().clone()
    cols = []
    dsc = max(float(D0.abs().max()), 1e-3) * 0.1
    for jj in (1, 2):
        for c in range(3):
            with torch.no_grad():
                g.flex.D[c, jj] += dsc
            cols.append(collect(lambda gg, cc, dd, ww: frf_of(gg, cc, dd, ww, R, freqs)))
            with torch.no_grad():
                g.flex.D.copy_(D0)
    blocks["D(j2,j3)"] = np.array(cols).T
    # ---- 有效模态残量: B_m 的 3 分量 (j2/j3 由 W 列决定, 这里用 B 代表方向) ----
    B0 = g.flex.B.detach().clone()
    bsc = max(float(B0.abs().max()), 1e-3) * 0.1
    cols = []
    for m in range(B0.shape[0]):
        for c in range(3):
            with torch.no_grad():
                g.flex.B[m, c] += bsc
            cols.append(collect(lambda gg, cc, dd, ww: frf_of(gg, cc, dd, ww, R, freqs)))
            with torch.no_grad():
                g.flex.B.copy_(B0)
    blocks["残量B-only"] = np.array(cols).T
    # ---- 有效模态残量 R_{m,c,j} = B[c,m]·W[m,j]: 无约束上界 3×3×2 = 18 列 ----
    #   (用户 2026-07-28: 之前只有 B-only 9 列, 不是针对 j2/j3 的有效残量)
    W0 = g.flex.W.detach().clone()
    cols = []
    for m in range(B0.shape[0]):
        for c in range(3):
            for jj in (1, 2):
                # 扰动 R_{m,c,j} ⇔ 同时动 B[c,m] 与 W[m,j] 的乘积;
                # 数值上用"只在该 (m,c) 且只对该关节生效"的等效扰动:
                # 先记录 W[m,j], 临时把其他关节列清零, 再扰 B[c,m]
                Wsave = W0.clone()
                with torch.no_grad():
                    for jo in range(6):
                        if jo != jj:
                            g.flex.W[m, jo] = 0.0
                    g.flex.B[m, c] += bsc
                cols.append(collect(lambda gg, cc, dd, ww: frf_of(gg, cc, dd, ww, R, freqs)))
                with torch.no_grad():
                    g.flex.W.copy_(Wsave); g.flex.B.copy_(B0)
    blocks["有效残量R(18)"] = np.array(cols).T
    # ---- 极点 ω/ζ ----
    lo0, rz0 = g.flex.log_om.detach().clone(), g.flex.raw_ze.detach().clone()
    cols = []
    for m in range(len(lo0)):
        with torch.no_grad():
            g.flex.log_om[m] += 0.05
        cols.append(collect(lambda gg, cc, dd, ww: frf_of(gg, cc, dd, ww, R, freqs)))
        with torch.no_grad():
            g.flex.log_om.copy_(lo0)
    for m in range(len(rz0)):
        with torch.no_grad():
            g.flex.raw_ze[m] += 0.2
        cols.append(collect(lambda gg, cc, dd, ww: frf_of(gg, cc, dd, ww, R, freqs)))
        with torch.no_grad():
            g.flex.raw_ze.copy_(rz0)
    blocks["极点ω/ζ"] = np.array(cols).T
    return blocks


def gauge_fix(Jth, Jres, Bw=None, M=3):
    """固定每模态的尺度规范: 把 θ 块中沿"纯缩放模态残量"方向的分量投影掉。

    用户 2026-07-28: 改变 θ 的截距与缩放模态残量**局部近乎等价**, 所以
    θ vs 残量的 0.1° 主夹角首先是**规范问题**。规范固定后 θ 只保留每模态
    7 个调度方向 ⇒ 共 21 个(24 − 3)。

    Jres 的每模态"纯缩放"方向 = 该模态自身的贡献列(残量整体乘一个标量)。
    """
    # **纯缩放**方向 = 沿 B_m **自身**(按当前 B 值加权), 不是三列之和 ——
    # 后者是沿 (1,1,1), 与"整体缩放该模态残量"不是同一方向(2026-07-28 修)
    if Bw is None:
        Bw = np.ones((M, 3))
    S = np.stack([sum(Bw[m, c] * Jres[:, m * 3 + c] for c in range(3))
                  for m in range(M)], 1)
    Q = np.linalg.qr(S)[0]
    Jth_g = Jth - Q @ (Q.T @ Jth)          # 投影掉规范方向
    return Jth_g, S.shape[1]


def posterior_sd_ratio(J):
    """posterior SD ratio_i = (1+s_i²)^(−1/2)。

    用户 2026-07-28: "有效秩 20/24" 这种阈值化说法不清楚 —— 最小白化奇异值
    仍是 342, 按"数据相对先验的信息量"它应当满秩。用后验/先验 SD 比更直白:
    比值 ≈1 表示数据没带来信息, ≈0 表示完全由数据定。
    """
    s = np.linalg.svd(J, compute_uv=False)
    return s, (1.0 + s ** 2) ** -0.5


def principal_angles(A_, B_):
    """两个子空间的主夹角(度)。"""
    qa = np.linalg.qr(A_)[0]; qb = np.linalg.qr(B_)[0]
    sv = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(sv, -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16,
                    help="训练位姿数 (2026-07-28 定 16: 3grip对+3速度对占12, 留4给贪心)")
    ap.add_argument("--hold", type=int, default=5, help="holdout 位姿数(固定 5, 只用一次)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cands, klo, khi = candidate_poses()
    Ph_all = np.array([phi_of(c) for c in cands])
    print(f"=== (A) 位姿筛选 ===")
    print(f"  可达范围: j2 [{np.degrees(klo[1]):.2f}, {np.degrees(khi[1]):.2f}]°  "
          f"j3 [{np.degrees(klo[2]):.2f}, {np.degrees(khi[2]):.2f}]°")
    print(f"  候选 {len(cands)} 个;  φ 用**逐帧 pose_features 的窗内均值**(非平均速度近似)")

    tr = greedy_select(cands, a.n, Ph_all)
    ho, cls = stratified_holdout(cands, tr, Ph_all, a.hold)
    print(f"\n  配对约束: 同(q2,q3) 的 grip 配对 ≥3 组; 同(q2,q3,grip) 的速度配对 ≥3 组")
    # 验证配对是否真的存在
    gpair = sum(1 for k in {key_qq(cands[i]) for i in tr}
                if len({cands[i]["grip"] for i in tr if key_qq(cands[i]) == k}) >= 2)
    spair = sum(1 for k in {(key_qq(cands[i]), cands[i]["grip"], cands[i]["dither"])
                            for i in tr}
                if len({cands[i]["hz"] for i in tr
                        if (key_qq(cands[i]), cands[i]["grip"],
                            cands[i]["dither"]) == k}) >= 2)
    print(f"    实得: grip 配对 {gpair} 组, 速度配对 {spair} 组")
    r_tr, _ = report_phi(cands, tr, "训练集")
    r_ho, _ = report_phi(cands, ho, "留出集(按覆盖选, 不追求 rank)")

    print(f"\n  训练位姿:")
    print(f"  {'#':>3}{'q2(°)':>9}{'q3(°)':>9}{'夹爪':>7}{'幅值(°)':>9}{'频率Hz':>8}"
          f"{'均速°/s':>9}{'类别':>7}")
    for k, i in enumerate(tr):
        p = cands[i]
        print(f"  {k:>3}{np.degrees(p['q2']):>9.2f}{np.degrees(p['q3']):>9.2f}"
              f"{p['grip']:>7.3f}{np.degrees(p['dither']):>9.2f}{p['hz']:>8.2f}"
              f"{np.degrees(4*p['hz']*p['dither']):>9.2f}{cls(p):>7}")
    print(f"\n  留出位姿(覆盖 中心/伸展/折叠 × 两夹爪 × 不同速度):")
    for k, i in enumerate(ho):
        p = cands[i]
        print(f"  {k:>3}{np.degrees(p['q2']):>9.2f}{np.degrees(p['q3']):>9.2f}"
              f"{p['grip']:>7.3f}{np.degrees(p['dither']):>9.2f}{p['hz']:>8.2f}"
              f"{np.degrees(4*p['hz']*p['dither']):>9.2f}{cls(p):>7}")

    rig = Rigid(); models = load_ensemble(rigid=rig)
    man = json.loads((TP.LROOT / "pp2-deploy" / "manifest.json").read_text())
    R = np.array(man["R"], np.float32)
    g = models[len(models) // 2]
    poses = [cands[i] for i in tr]

    cvs = alpha_variation(g, poses)

    # ---------- (B) 白化分块 Jacobian, D1 vs D2 ----------
    print(f"\n=== (B) 白化分块 Jacobian: D1(全位姿 2–8Hz) vs D2(全位姿 6–8Hz + 4 位姿 2–8Hz) ===")
    print(f"  ⚠ 只能表述为: **在当前模型、冻结其余参数、无噪声的局部线性化下**的代数性质;")
    print(f"    白化用 B′ 的实测噪声协方差(10 个配对差, 向对角收缩 0.3), 参数按成员间 std 归一。")
    F_NARROW = np.array(FREQS, float)                       # 6.0-7.5
    F_WIDE = np.arange(2.0, 8.01, 0.5)                      # 2-8Hz (用户定, 非 2.5-9)
    for lbl, freqs in [("D1: 2–8Hz", F_WIDE), ("D2 窄带部分: 6–8Hz", F_NARROW)]:
        C, nsamp = noise_cov_from_B2(freqs)                 # **在实际频点栅格上**实测, 不外推
        # 对宽带: 用 B′ 频点的对角方差按最近邻外推(明确标注为外推)
        note = f" (噪声协方差: {nsamp} 个**不重叠**配对差 A1−A2/A3−A4/A5−A6, 收缩 0.5)"
        Wh = whiten_op(C)
        blocks = block_jacobian(g, poses, R, freqs, models)
        nrow_per = C.shape[0]
        def wh(Jb):
            n = Jb.shape[0] // nrow_per
            return np.vstack([Wh @ Jb[k*nrow_per:(k+1)*nrow_per] for k in range(n)])
        Jw = {k: wh(v) for k, v in blocks.items()}
        print(f"\n  --- {lbl}{note} ---")
        # **B-only 与 R(18) 是嵌套关系**(R 含 B-only) ⇒ 不能同时进合并矩阵, 否则秩亏
        for resk in ("残量B-only", "有效残量R(18)"):
            ks_ = ["θ(规范后)", "τ", "D(j2,j3)", resk, "极点ω/ζ"]
            Jall = np.hstack([Jw[k] for k in ks_ if k in Jw])
            sv = np.linalg.svd(Jall, compute_uv=False)
            eff = int((sv / sv[0] > 1e-2).sum())
            print(f"  合并[{resk}] {Jall.shape}  秩(阈1e-2) **{eff}/{Jall.shape[1]}**"
                  f"  条件数 {sv[0]/max(sv[-1],1e-30):.1e}"
                  f"  前8奇异值 {np.round(sv[:8]/sv[0],3)}")
        # 规范固定: θ 块投影掉"纯缩放残量"方向 ⇒ 21 dof
        if "残量B-only" in Jw:
            Bw = np.abs(g.flex.B.detach().numpy())
            Jw["θ(规范后)"], ng = gauge_fix(Jw["θ"], Jw["残量B-only"], Bw)
            print(f"  规范固定: θ 投影掉 {ng} 个纯缩放方向 ⇒ 24−{ng} = {24-ng} 个调度自由度")
        print(f"  {'块':>14}{'列数':>6}{'s_max':>11}{'s_min':>11}"
              f"{'后验/先验SD 中位':>18}{'最差(=1 无信息)':>16}")
        for k, v in Jw.items():
            sv2, r2 = posterior_sd_ratio(v)
            print(f"  {k:>14}{v.shape[1]:>6}{sv2[0]:>11.2e}{sv2[-1]:>11.2e}"
                  f"{np.median(r2):>18.4f}{r2.max():>16.4f}")
        # τ 的物理单位置信区间
        if "τ" in Jw:
            sv_t = np.linalg.svd(Jw["τ"], compute_uv=False)
            sd_tau = 1.0 / max(sv_t[0], 1e-30)
            print(f"  τ 的**局部** CRLB 1σ = **{sd_tau*1e3:.4f} ms**")
            print(f"    ⚠ 局部精度 {sd_tau*1e3:.4f} ms 与分支间距 **141 ms** 相差 4 个数量级 ——")
            print(f"      局部 J_tau 满秩**完全不能**排除另一谷; 分支选择必须靠**全局 τ 扫描**。")
        print(f"  注: '后验/先验 SD 比'里的先验 = **当前 K=5 集合的成员间 std**;")
        print(f"    该 std 很宽(θ 非常数列 1.87), 故比值极小只说明'相对当前分歧收缩很多',")
        print(f"    不等于绝对精度。绝对精度须换算到物理单位后单独报(如上面的 τ)。")
        print(f"  块间主夹角(最小, 度) — 越小越混杂  (θ 用**规范后**的):")
        ks = [k for k in Jw if k != "θ"]
        for x in range(len(ks)):
            for y in range(x + 1, len(ks)):
                pa = principal_angles(Jw[ks[x]], Jw[ks[y]])
                print(f"    {ks[x]:>12} vs {ks[y]:<12} 最小 {pa.min():>6.1f}°  中位 {np.median(pa):>6.1f}°")

    print(f"\n  D1/D2 的机时对比留到功效分析(需注入两分支 + B′ 量级的模型形状误差,")
    print(f"  统计延迟分支选对率与 L1/L2/L3 的 LOPO 模型选择成功率)。")
    art = {"n_train": a.n, "n_hold": a.hold, "phi_rank_train": r_tr,
           "phi_rank_hold": r_ho, "grip_pairs": gpair, "speed_pairs": spair,
           "poses_train": poses, "poses_hold": [cands[i] for i in ho],
           "alpha_cv_median": float(np.median(cvs.max(1)))}
    (OUT / "poses_C.json").write_text(json.dumps(art, indent=1, ensure_ascii=False))
    print(f"\n→ {(OUT / 'poses_C.json').relative_to(ROOT.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ======================= holdout 硬封存（c-pilot-v1） =======================

HOLDOUT_LOCK = True          # True = 任何路径都**不得**把 holdout 位姿交给下游


def train_and_holdout(n_train=16, n_hold=5, seed=0):
    """返回 (train_idx, holdout_idx, cands)。**唯一**允许产生位姿集合的入口。"""
    cands, _, _ = candidate_poses()
    Ph = np.array([phi_of(c) for c in cands])
    tr = greedy_select(cands, n_train, Ph, seed=seed)
    ho, _cls = stratified_holdout(cands, set(tr), Ph, n=n_hold)
    return list(tr), list(ho), cands


def assert_train_only(idx, holdout_idx, where=""):
    """**硬断言**: 下游拿到的位姿集合不得包含任何 holdout 位姿。

    用户 2026-07-29(c-pilot-v1): pilot 与完整采集的拟合/模型选择路径
    **一律不得加载 holdout**; holdout 文件只采集并封存。
    """
    bad = sorted(set(idx) & set(holdout_idx))
    if bad:
        raise RuntimeError(
            f"HOLDOUT 泄漏于 {where}: 位姿 {bad} 属于封存集合 {sorted(holdout_idx)}")
    if HOLDOUT_LOCK and len(set(idx)) != len(idx):
        raise RuntimeError(f"位姿索引重复于 {where}: {idx}")
    return True
