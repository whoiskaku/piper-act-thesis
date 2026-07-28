#!/usr/bin/env python3
"""3.2 真实层级 L0/L1/L2/L3 + group-LOPO（2026-07-28，用户规格）。

**用户钉住的规则（逐条落实位置）**
  R1 每个层级、每个 LOPO fold 都**重新做全局 τ 扫描**；只能用训练组选分支    → `fit_scan`
  R2 16 条件按 **7 个物理位姿组**留出，同组 grip/速度全部一起移除；
     汇总时**七组等权**，不按条件数加权                                    → `group_masks`, `lopo`
  R3 正则强度**只在训练组内部**选择 —— 用**嵌套**内层 LOPO，杜绝复用外折  → `lopo` 内层
  R4 层级必须**真正嵌套**，打印每层开放的参数块、有效自由度和规范；
     **B-only 与 R18 不能同时**作为独立块                                  → `LEVELS`, `describe`
  R5 层级用 LOPO 预测误差 + **one-standard-error rule**                    → `pick_level`
  R6 5 个真实 holdout **始终封存**，层级确定后才用一次                     → `HOLDOUT_SEALED`

**为什么付得起**
  各层级写成围绕 Ĝ 标称值的局部线性化 `H = e^{-2πjfτ}(H₀ + A·η)`。
  则 `A^H A` 中 τ 的相位因子是逐行单位模标量, **两两相消 ⇒ Gram 与 τ 无关**;
  只有右端项随 τ 变, 且恰为一个 NUDFT。⇒ Cholesky 可跨 trial 缓存,
  τ 扫描退化为一次矩阵乘。嵌套 λ 选择(7×6 折)才付得起。

用法:
  python sim/levels_C.py --stage verify  --trials 100    # 验证优化与无泄漏
  python sim/levels_C.py --stage screen  --trials 300    # 筛层级 + 最坏方向
  python sim/levels_C.py --stage final   --trials 1500   # 冻结后终跑
"""
from __future__ import annotations
import argparse
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

# ---- R4: 真正嵌套的层级。B-only(9) **不作为独立块**, 只保留有效残量 R18 ----
LEVELS = {
    "L0": ["gain"],
    "L1": ["gain", "θ"],
    "L2": ["gain", "θ", "R18", "D6"],
    "L3": ["gain", "θ", "R18", "D6", "poles6"],
}
# τ 网格步长 1 ms: P.TAU_GRID 是 5 ms, 真值 +102/−52 ms **不在格点上**,
# 会造成 2 ms 量化失配并泄漏进 η(2026-07-28 验证阶段发现)
TAU_GRID = np.arange(-0.250, 0.2501, 0.001)
LAM_GRID = np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0])
LAM_TINY = 1e-10                 # 仅用于"优化正确性"检查, 不进入选层级
HOLDOUT_SEALED = True          # R6: 本模块**任何路径都不读** holdout 行


def prior_scale(models, attr):
    """**统一的**先验尺度 = K 个 bootstrap 成员之间的标准差(数据驱动)。

    2026-07-28: 原先 B/D 用「10% of max」、log_om 用 0.05、raw_ze 用 0.2 都是拍的,
    结果各块的 η=1 含义不同 —— L3 的 poles6 在同尺度下几乎不可见(注入 SNR
    1.48 vs L2 的 1.50), 使「L2/L3 被错误降级」沦为扰动幅度的产物。
    改为与 θ 一致: **η=1 ⇔ 该参数的一个先验标准差**。
    """
    v = np.array([getattr(g.flex, attr).detach().numpy() for g in models])
    return np.maximum(v.std(0), np.abs(v).mean() * 1e-3 + 1e-9)


# ======================= Jacobian（逐位姿频点集） =======================

def level_jacobian(g, poses, R, F_of, amp_of, models):
    """围绕标称 Ĝ 的局部线性化列。行 = 位姿×关节×频点×轴（复）。

    与 `select_poses_C.block_jacobian` 的差别: **逐位姿使用各自的频点集**
    （混合设计里锚点是宽带、其余是窄带），且不含 B-only 块。
    """
    import torch
    keys = [(pi, j) for pi in range(len(poses)) for j in (1, 2)]
    cache = {}
    for pi, p in enumerate(poses):
        cache[pi] = S.synth_cmd(p)

    def evaluate():
        out = []
        for (pi, j) in keys:
            cmd7 = cache[pi]
            da, w0 = S.probe_da(len(cmd7), j, amp=amp_of[pi],
                                seed=pi * 10 + j, freqs=F_of[pi])
            out.append(S.frf_of(g, cmd7, da, w0, R, F_of[pi]).ravel())
        return np.concatenate(out)

    H0 = evaluate()

    def column(mut, restore):
        mut()
        g._vker = g._exck = None
        v = evaluate() - H0
        restore()
        g._vker = g._exck = None
        return v

    blocks = {}
    # gain: (1+η_re+jη_im)·H0  ⇒ 两列 H0 与 j·H0
    blocks["gain"] = np.stack([H0, 1j * H0], 1)

    th0 = g.flex.theta.detach().clone()
    sc = S.theta_prior_scale(models)
    cols = []
    for m in range(th0.shape[0]):
        for c in range(th0.shape[1]):
            e = float(sc[m, c])
            cols.append(column(
                lambda m=m, c=c, e=e: g.flex.theta.data.__setitem__((m, c),
                                                                    g.flex.theta.data[m, c] + e),
                lambda: g.flex.theta.data.copy_(th0)))
    blocks["θ"] = np.stack(cols, 1)

    B0 = g.flex.B.detach().clone(); W0 = g.flex.W.detach().clone()
    bsc_m = prior_scale(models, "B")            # **先验尺度**, 逐元素
    cols = []
    for m in range(B0.shape[0]):
        for c in range(3):
            for jj in (1, 2):
                def mut(m=m, c=c, jj=jj):
                    with torch.no_grad():
                        for jo in range(6):
                            if jo != jj:
                                g.flex.W[m, jo] = 0.0
                        g.flex.B[m, c] += float(bsc_m[m, c])
                cols.append(column(mut, lambda: (g.flex.W.data.copy_(W0),
                                                 g.flex.B.data.copy_(B0))))
    blocks["R18"] = np.stack(cols, 1)

    D0 = g.flex.D.detach().clone()
    dsc_m = prior_scale(models, "D")
    cols = []
    for jj in (1, 2):
        for c in range(3):
            cols.append(column(
                lambda c=c, jj=jj: g.flex.D.data.__setitem__((c, jj),
                                                             g.flex.D.data[c, jj]
                                                             + float(dsc_m[c, jj])),
                lambda: g.flex.D.data.copy_(D0)))
    blocks["D6"] = np.stack(cols, 1)

    lo0 = g.flex.log_om.detach().clone(); rz0 = g.flex.raw_ze.detach().clone()
    lsc = prior_scale(models, "log_om"); zsc = prior_scale(models, "raw_ze")
    cols = []
    for m in range(len(lo0)):
        cols.append(column(
            lambda m=m: g.flex.log_om.data.__setitem__(
                m, g.flex.log_om.data[m] + float(lsc[m])),
            lambda: g.flex.log_om.data.copy_(lo0)))
    for m in range(len(rz0)):
        cols.append(column(
            lambda m=m: g.flex.raw_ze.data.__setitem__(
                m, g.flex.raw_ze.data[m] + float(zsc[m])),
            lambda: g.flex.raw_ze.data.copy_(rz0)))
    blocks["poles6"] = np.stack(cols, 1)

    # ---- 规范: 把 θ 中沿"纯缩放模态残量"的方向投影掉(21 = 24 − 3) ----
    Bw = np.abs(B0.detach().numpy())
    # 「整体缩放模态 m 的残量」= 按当前 B 值加权、**两个关节列一起**求和
    #   （R18 是按 (m,c,joint) 展开的; 只取一个关节会给出错误的规范方向）
    Sdir = np.stack([sum(Bw[m, c] * (blocks["R18"][:, (m * 3 + c) * 2]
                                     + blocks["R18"][:, (m * 3 + c) * 2 + 1])
                         for c in range(3))
                     for m in range(B0.shape[0])], 1)
    Q = np.linalg.qr(np.concatenate([Sdir.real, Sdir.imag], 0))[0]
    Th = np.concatenate([blocks["θ"].real, blocks["θ"].imag], 0)
    Th = Th - Q @ (Q.T @ Th)
    n = len(H0)
    blocks["θ"] = Th[:n] + 1j * Th[n:]

    # 行元信息
    frow, krow, prow = [], [], []
    for ki, (pi, j) in enumerate(keys):
        for f in F_of[pi]:
            for _ in range(3):
                frow.append(f); krow.append(ki); prow.append(pi)
    return (H0, blocks, np.array(frow), np.array(krow), np.array(prow), keys)


# ======================= 噪声与白化 =======================

def noise_rows(keys, F_of, Um, nb_cache):
    """每行的噪声样本池(FRF 空间 = N/U), 与白化权重。"""
    pool, sd = [], []
    for (pi, j) in keys:
        F = F_of[pi]
        nb = nb_cache[tuple(F)]                    # (nblk, nf, 3)
        ui = 1.0 / np.where(np.abs(Um[(pi, j)]) < 1e-15, 1e-15, Um[(pi, j)])
        v = nb * ui[None, :, None]                 # (nblk, nf, 3)
        pool.append(v.reshape(len(nb), -1))
        sd.append(np.sqrt(np.maximum((np.abs(v) ** 2).mean(0), 1e-30)).ravel())
    return np.concatenate(pool, 1), np.concatenate(sd)


# ======================= 拟合器（Gram 与 τ 无关 ⇒ 跨 trial 缓存） =======================

class Fitter:
    def __init__(self, H0, blocks, frow, w):
        self.frow, self.w = frow, w
        self.H0 = H0 * w
        self.cols = {}
        off = 0
        self.span = {}
        mats = []
        for nm in ("gain", "θ", "R18", "D6", "poles6"):
            Bm = blocks[nm] * w[:, None]
            self.span[nm] = (off, off + Bm.shape[1]); off += Bm.shape[1]
            mats.append(Bm)
        self.Aall = np.concatenate(mats, 1)
        self.E = np.exp(2j * np.pi * np.outer(TAU_GRID, frow))     # (nτ, n)
        self._chol = {}
        self._subc = {}
        self._rhs = {}

    def idx(self, level):
        """块顺序 = gain,θ,R18,D6,poles6, 而各层级恰是其**连续前缀**
        ⇒ 可以只算一次全列右端项, 各层级切片即可(见 `rhs`)。"""
        return np.arange(0, max(self.span[b][1] for b in LEVELS[level]))

    def rhs(self, obs, mask, mk):
        """右端项 Bt(τ) 与 ‖y(τ)‖²。**与 λ、与层级都无关**（层级只是切片）。

        Bt(τ) = Re(A^H(obs·e^{+2πjfτ})) − Re(A^H H0)  —— 对频率的 NUDFT。
        每个 mask 每个 trial 只算一次 ⇒ 相比逐(层级,λ)重算快约 17×。
        """
        if mk in self._rhs:
            return self._rhs[mk]
        Asub, Em = self._sub(mask, mk)
        y0 = obs[mask]; H0m = self.H0[mask]
        Bt = (Em @ (Asub.conj() * y0[:, None])).real - (Asub.conj().T @ H0m).real[None, :]
        yy = ((np.abs(y0) ** 2).sum() + (np.abs(H0m) ** 2).sum()
              - 2.0 * (Em @ (H0m.conj() * y0)).real)
        self._rhs[mk] = (Bt, yy)
        return Bt, yy

    def _sub(self, mask, mk):
        if mk not in self._subc:
            self._subc[mk] = (self.Aall[mask], self.E[:, mask])
        return self._subc[mk]

    def new_trial(self):
        self._rhs = {}

    def gram(self, mask, level, lam):
        """Re(A^H A) + λI 的 Cholesky。**与 τ 无关 ⇒ 可缓存**。"""
        key = (mask.tobytes(), level, float(lam))
        if key not in self._chol:
            Asub = self.Aall[mask][:, self.idx(level)]
            M = (Asub.conj().T @ Asub).real
            M = M + lam * np.trace(M) / max(M.shape[0], 1) * np.eye(M.shape[0])
            self._chol[key] = (np.linalg.cholesky(M), M)
        return self._chol[key]

    def fit_scan(self, obs, mask, level, lam, mk=None):
        """R1: 在 mask（**只含训练组**）上做**全局 τ 扫描**, 返回 (τ̂, η, 曲线)。

        τ 网格覆盖 [−250, +250] ms 全程 —— **不先定分支再拟合**。
        """
        mk = mk if mk is not None else mask.tobytes()
        Bt_all, yy = self.rhs(obs, mask, mk)
        p = len(self.idx(level))
        Bt = Bt_all[:, :p]                       # 层级 = 连续前缀
        L, M = self.gram(mask, level, lam)
        Et = np.linalg.solve(L.T, np.linalg.solve(L, Bt.T)).T          # (nτ, p)
        sse = yy - 2.0 * (Bt * Et).sum(1) + ((Et @ M) * Et).sum(1)
        t = int(np.argmin(sse))
        return TAU_GRID[t], Et[t], sse

    def resid(self, obs, mask, level, tau, eta):
        """在 mask 上的白化 SSE（用于留出组的预测误差）。"""
        ci = self.idx(level)
        pred = (self.H0[mask] + self.Aall[mask][:, ci] @ eta.astype(complex))
        pred = pred * np.exp(-2j * np.pi * self.frow[mask] * tau)
        return float((np.abs(obs[mask] - pred) ** 2).sum()), int(mask.sum())


# ======================= group-LOPO（R2/R3/R5） =======================

def lopo(fit, obs, gmasks):
    """嵌套 group-LOPO。外层留一组做预测误差; **内层**只在训练组里选 λ(R3)。

    返回 {level: (七组等权平均误差, 七组误差数组, 各外折 τ̂)}。
    """
    ng = len(gmasks)
    fit.new_trial()                       # 清 rhs 缓存(obs 变了), Gram/子矩阵仍复用
    trm = [(~gmasks[o], f"o{o}".encode()) for o in range(ng)]
    inm = {(o, i): ((~gmasks[o]) & ~gmasks[i], f"o{o}i{i}".encode())
           for o in range(ng) for i in range(ng) if i != o}
    out = {}
    for lv in LEVELS:
        errs, taus = [], []
        for o in range(ng):
            tr, trk = trm[o]
            # ---- 内层: 在剩余 6 组里选 λ ----
            best, blam = np.inf, LAM_GRID[0]
            for lam in LAM_GRID:
                acc = []
                for i in range(ng):
                    if i == o:
                        continue
                    itr, itk = inm[(o, i)]
                    tau_i, eta_i, _ = fit.fit_scan(obs, itr, lv, lam, itk)
                    s, n = fit.resid(obs, gmasks[i], lv, tau_i, eta_i)
                    acc.append(s / max(n, 1))          # R2: 组内先归一, 组间等权
                m = float(np.mean(acc))
                if m < best:
                    best, blam = m, lam
            # ---- 外层: 用 λ* 在 6 组上重拟合(含**全局 τ 扫描**), 预测留出组 ----
            tau_o, eta_o, _ = fit.fit_scan(obs, tr, lv, blam, trk)
            s, n = fit.resid(obs, gmasks[o], lv, tau_o, eta_o)
            errs.append(s / max(n, 1)); taus.append(tau_o)
        errs = np.array(errs)
        out[lv] = (float(errs.mean()), errs, np.array(taus))   # **七组等权**
    return out


def pick_level(res):
    """R5: one-standard-error rule —— 统计上相当时选**更简单**的一层。"""
    order = list(LEVELS)
    means = {lv: res[lv][0] for lv in order}
    best = min(means, key=means.get)
    e = res[best][1]
    se = float(e.std(ddof=1) / np.sqrt(len(e)))
    thr = means[best] + se
    for lv in order:                       # order 已由简到繁
        if means[lv] <= thr:
            return lv, best, se, thr
    return best, best, se, thr


def describe(fit):
    print("  【层级定义（R4: 真正嵌套）】")
    print(f"  {'层级':>6}{'开放参数块':>34}{'列数':>7}{'累计dof':>9}")
    for lv, bl in LEVELS.items():
        p = sum(fit.span[b][1] - fit.span[b][0] for b in bl)
        print(f"  {lv:>6}{'+'.join(bl):>34}{p:>7}{p:>9}")
    print("  规范: θ 已投影掉 3 个「纯缩放模态残量」方向 ⇒ 24 → **21**")
    print("  **B-only(9) 不作为独立块**（与 R18 嵌套, 会造成秩亏）")
    n = fit.Aall.shape[0]
    print(f"\n  各块的白化列 RMS（η=1 ⇔ 一个先验标准差; 白化后噪声 RMS≈1）")
    print(f"  {'块':>8}{'列数':>6}{'列 RMS 中位':>13}{'块整体 SNR':>12}")
    for b, (lo, hi) in fit.span.items():
        col = fit.Aall[:, lo:hi]
        rms = np.sqrt((np.abs(col) ** 2).mean(0))
        print(f"  {b:>8}{hi-lo:>6}{np.median(rms):>13.3f}"
              f"{np.sqrt((rms ** 2).sum()):>12.2f}")

    # 每一层**新增**块相对前面所有块张成空间的正交残量 —— 决定该层是否真能被辨识
    print(f"\n  逐层**新增**信息量（新块对前面各层张成空间的正交残量）")
    print(f"  {'层级':>6}{'新增块':>14}{'正交残量 SNR':>14}{'被前层解释':>12}")
    prev = np.zeros((fit.Aall.shape[0], 0), complex)
    for lv, bl in LEVELS.items():
        new = [b for b in bl if b not in
               (LEVELS[list(LEVELS)[list(LEVELS).index(lv) - 1]] if lv != "L0" else [])]
        if not new:
            continue
        Nw = np.concatenate([fit.Aall[:, slice(*fit.span[b])] for b in new], 1)
        if prev.shape[1]:
            X = np.concatenate([prev.real, prev.imag], 0)
            Y = np.concatenate([Nw.real, Nw.imag], 0)
            Q = np.linalg.qr(X)[0]
            Res = Y - Q @ (Q.T @ Y)
            frac = 1.0 - (Res ** 2).sum() / max((Y ** 2).sum(), 1e-30)
            snr = np.sqrt((Res ** 2).sum() / (2 * n))
        else:
            frac, snr = 0.0, np.sqrt((np.abs(Nw) ** 2).sum() / n)
        print(f"  {lv:>6}{'+'.join(new):>14}{snr:>14.2f}{frac:>11.0%}")
        prev = np.concatenate([prev, Nw], 1)
    print()


# ======================= trial 生成 =======================

def make_obs(fit, npool, rng, level_true, tau_true, eta_sd, noise_on=True):
    """按真值层级生成一次观测（白化空间）。"""
    eta = np.zeros(fit.Aall.shape[1])
    for b in LEVELS[level_true]:
        lo, hi = fit.span[b]
        sd = 0.05 if b == "gain" else eta_sd          # 增益只允许小偏差
        eta[lo:hi] = rng.normal(0, sd, hi - lo)
    sig = fit.H0 + fit.Aall @ eta.astype(complex)
    obs = sig * np.exp(-2j * np.pi * fit.frow * tau_true)
    if noise_on:
        obs = obs + npool[rng.integers(len(npool))]
    return obs, eta


def branch_absorb(taus, tau_true):
    """R: 更自由的层级是否**吸收错误 τ 分支**。"""
    other = P.TAU_TRUE["-52ms" if abs(tau_true - P.TAU_TRUE["+102ms"]) < 1e-9
                       else "+102ms"]
    return (float(np.mean(np.abs(taus - tau_true) > 0.030)),
            float(np.mean(np.abs(taus - other) < 0.030)))


# ======================= 各阶段 =======================

def stage_verify(fit, npool, gmasks, rng, ntr, eta_sd):
    print("  【A. 优化正确性（无噪声, 真值层级, λ→0）】")
    print(f"    真值 τ 必须落在网格上: 步长 {1e3*(TAU_GRID[1]-TAU_GRID[0]):.0f} ms, "
          f"+102/−52 ms 在格点上 = "
          f"{all(np.min(np.abs(TAU_GRID - t)) < 1e-9 for t in P.TAU_TRUE.values())}")
    print(f"  {'真值层级':>10}{'τ 误差(ms)':>12}{'η 相对误差':>12}{'残差/信号':>12}"
          f"{'注入 SNR':>11}")
    ok = True
    full = np.ones(len(fit.frow), bool)
    for lv in LEVELS:
        for br, tt in P.TAU_TRUE.items():
            obs, eta = make_obs(fit, npool, np.random.default_rng(7), lv, tt,
                                eta_sd, noise_on=False)
            th, eh, _ = fit.fit_scan(obs, full, lv, LAM_TINY)
            ci = fit.idx(lv)
            rel = np.linalg.norm(eh - eta[ci]) / max(np.linalg.norm(eta[ci]), 1e-12)
            s, n = fit.resid(obs, full, lv, th, eh)
            sg = float((np.abs(obs) ** 2).sum())
            # 注入信号相对噪声的量级（白化空间 ⇒ 噪声 RMS ≈ 1/行）
            snr = np.linalg.norm(fit.Aall[:, ci] @ eta[ci].astype(complex)) / np.sqrt(len(obs))
            bad = abs(th - tt) > 1e-9
            ok &= not bad
            print(f"  {lv+'/'+br:>10}{(th-tt)*1e3:>12.1f}{rel:>12.2e}"
                  f"{np.sqrt(s/sg):>12.2e}{snr:>11.2f}")
    print(f"  ⇒ τ 恢复{'**全部精确**' if ok else '**有偏差**'}\n")

    print("  【B. 无泄漏检查】")
    print(f"    · Jacobian 只含 16 个训练条件, **holdout 位姿未进入任何矩阵**"
          f"  (HOLDOUT_SEALED={HOLDOUT_SEALED})")
    obs, _ = make_obs(fit, npool, rng, "L2", P.TAU_TRUE["+102ms"], eta_sd)
    same = True
    for o in range(len(gmasks)):
        tr = ~gmasks[o]
        t1, e1, _ = fit.fit_scan(obs, tr, "L2", 1e-2)
        obs2 = obs.copy()
        obs2[gmasks[o]] = rng.normal(size=int(gmasks[o].sum())) * 1e3   # 破坏留出组
        t2, e2, _ = fit.fit_scan(obs2, tr, "L2", 1e-2)
        same &= (t1 == t2) and np.allclose(e1, e2)
    print(f"    · 破坏留出组后训练侧 (τ̂, η) {'**逐位不变**' if same else '**改变了 ⇒ 泄漏**'}")
    ovl = max(int((gmasks[i] & gmasks[j]).sum())
              for i in range(len(gmasks)) for j in range(i + 1, len(gmasks)))
    print(f"    · 七个物理位姿组两两行重叠 = {ovl}（应为 0；同组 grip/速度整组移除）\n")

    print(f"  【C. {ntr} trial 层级选择（含噪声）】")
    return run_trials(fit, npool, gmasks, ntr, eta_sd, rng)


def run_trials(fit, npool, gmasks, ntr, eta_sd, rng):
    lv_list = list(LEVELS)
    conf = np.zeros((4, 4), int)
    absorb = {lv: [] for lv in lv_list}
    wrongbr = {lv: [] for lv in lv_list}
    gerr = {lv: [] for lv in lv_list}
    for t in range(ntr):
        lt = lv_list[t % 4]
        tt = P.TAU_TRUE["+102ms" if (t // 4) % 2 == 0 else "-52ms"]
        obs, _ = make_obs(fit, npool, np.random.default_rng(90000 + t), lt, tt, eta_sd)
        res = lopo(fit, obs, gmasks)
        sel, _, _, _ = pick_level(res)
        conf[lv_list.index(lt), lv_list.index(sel)] += 1
        for lv in lv_list:
            a, b = branch_absorb(res[lv][2], tt)
            absorb[lv].append(a); wrongbr[lv].append(b)
            gerr[lv].append(res[lv][1])
    return conf, absorb, wrongbr, gerr, lv_list


def report(conf, absorb, wrongbr, gerr, lv_list):
    n = conf.sum()
    print(f"  混淆矩阵（行=真值, 列=所选; n={n}）")
    hdr = "真/选"
    print(f"  {hdr:>8}" + "".join(f"{c:>8}" for c in lv_list) + f"{'选对率':>9}")
    for i, r in enumerate(lv_list):
        tot = max(conf[i].sum(), 1)
        print(f"  {r:>8}" + "".join(f"{conf[i, j]:>8}" for j in range(4))
              + f"{conf[i, i]/tot:>8.1%}")
    up = (conf[0, 1:].sum() + conf[1, 2:].sum()) / max(conf[:2].sum(), 1)
    dn = (conf[2, :2].sum() + conf[3, :3].sum()) / max(conf[2:].sum(), 1)
    print(f"\n  **L0/L1 被错误升级** = {up:.1%}   (门槛 ≤10%)")
    print(f"  **L2/L3 被错误降级** = {dn:.1%}")
    print(f"\n  {'层级':>6}{'τ 偏离真值>30ms':>17}{'落到错误分支':>14}"
          f"{'组间误差 CV':>13}")
    for lv in lv_list:
        ge = np.array(gerr[lv])
        cv = float(np.mean(ge.std(1) / np.maximum(ge.mean(1), 1e-30)))
        print(f"  {lv:>6}{np.mean(absorb[lv]):>16.1%}{np.mean(wrongbr[lv]):>14.1%}"
              f"{cv:>13.2f}")
    print(f"\n  每组白化预测误差（七组等权; 不只报总体均值）")
    print(f"  {'层级':>6}" + "".join(f"{'g'+str(i):>11}" for i in range(7)))
    for lv in lv_list:
        ge = np.array(gerr[lv]).mean(0)
        print(f"  {lv:>6}" + "".join(f"{v:>11.3e}" for v in ge))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="verify", choices=["verify", "screen", "final"])
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--eta", type=float, default=0.3)
    ap.add_argument("--design", default="hybrid1", choices=["hybrid1", "D1", "D2"])
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

    if a.design == "D1":
        F_of = {pi: F_WIDE for pi in range(16)}
    elif a.design == "D2":
        F_of = {pi: F_NARROW for pi in range(16)}
    else:
        anc = {grp[gkeys[gi]][0] for gi in (1, 2, 5, 6)}
        F_of = {pi: (F_WIDE if pi in anc else F_NARROW) for pi in range(16)}
    E_star = min(P.seq_energy(F_WIDE), P.seq_energy(F_NARROW))
    amp_of = {pi: P.amp_for(F_of[pi], "energy", E_star) for pi in range(16)}

    print(f"=== 3.2 真实层级 + group-LOPO ({a.design}, stage={a.stage}) ===\n")
    H0, blocks, frow, krow, prow, keys = level_jacobian(g, poses, R, F_of, amp_of, models)
    _, Um = P.model_and_U(g, poses, R, F_of, amp_of)
    nb_cache = {tuple(F_WIDE): P.noise_blocks(F_WIDE),
                tuple(F_NARROW): P.noise_blocks(F_NARROW)}
    npool, sd = noise_rows(keys, F_of, Um, nb_cache)
    w = 1.0 / sd
    npool = npool * w[None, :]
    fit = Fitter(H0, blocks, frow, w)
    describe(fit)

    gmasks = [np.isin(prow, grp[gk]) for gk in gkeys]
    print(f"  七个物理位姿组的行数: {[int(m.sum()) for m in gmasks]}")
    print(f"  条件数/组: {[len(grp[gk]) for gk in gkeys]}  (整组留出, 组间等权)\n")

    rng = np.random.default_rng(3)
    if a.stage == "verify":
        conf, ab, wb, ge, lvl = stage_verify(fit, npool, gmasks, rng, a.trials, a.eta)
    else:
        conf, ab, wb, ge, lvl = run_trials(fit, npool, gmasks, a.trials, a.eta, rng)
    report(conf, ab, wb, ge, lvl)
    np.savez(OUT / f"levels_C_{a.design}_{a.stage}.npz", conf=conf,
             absorb=np.array([ab[l] for l in lvl]),
             wrongbr=np.array([wb[l] for l in lvl]),
             gerr=np.array([ge[l] for l in lvl]))
    print(f"\n→ outputs/probe_c/levels_C_{a.design}_{a.stage}.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
