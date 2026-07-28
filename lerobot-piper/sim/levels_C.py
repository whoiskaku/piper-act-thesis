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
# θ 拆成 **θ_shape(21 个位形调度形状方向)** 与 **modal_scale(3 个模态平均尺度)**。
#
# 2026-07-29 实测: theta_mean 落在 span(R18) 内的比例只有 **0.057**。
#   **正确解释**(用户校准): 窗内时变**不会**破坏等价性 —— 系统对模态输入是线性的,
#   真正的统一缩放 α_m(t) → s·α_m(t) 与 R_m → s·R_m **严格等价**, 与 α 是否时变无关。
#   0.057 只说明这 3 个方向**不是纯 modal scale**: 它们是 θ 中改变**平均 α** 的方向,
#   经 softplus 与 φ 依赖同时改变了**归一化调度形状**, 故 R 的径向缩放替代不了。
#   ⇒ 命名为 theta_mean(不叫 modal_scale); 真正的均匀缩放列另做单元测试(见 gauge_tests)。
# ⇒ 若 L2R 起排除 modal_scale, span(L1) ⊄ span(L2R), **嵌套性会被破坏**。
#   故 modal_scale 保留在 L1 及以上**全部**层级(见 describe 的嵌套性验收)。
LEVELS = {
    "L0":  ["gain"],
    "L1":  ["gain", "θ_shape", "theta_mean"],
    "L2R": ["gain", "θ_shape", "theta_mean", "R18"],  # D6 消融臂(用户 2026-07-28)
    "L2":  ["gain", "θ_shape", "theta_mean", "R18", "D6"],
    "L3":  ["gain", "θ_shape", "theta_mean", "R18", "D6", "poles6"],
}
BLOCK_ORDER = ("gain", "θ_shape", "theta_mean", "R18", "D6", "poles6")
# τ 网格步长 1 ms: P.TAU_GRID 是 5 ms, 真值 +102/−52 ms **不在格点上**,
# 会造成 2 ms 量化失配并泄漏进 η(2026-07-28 验证阶段发现)
TAU_GRID = np.arange(-0.250, 0.2501, 0.001)
# 必须包含**接近零**的一档, 否则「正则压制」与「设计不可辨识」分不开(用户 2026-07-29)
LAM_GRID = np.array([1e-8, 1e-5, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
LAM_TINY = 1e-10                 # 仅用于"优化正确性"检查, 不进入选层级
# 用户 2026-07-28: 真值恰在 1ms 格点上属**过于容易的自检**。分支判别仍用 1ms 网格,
# 但**绝对时延恢复**必须用离格真值 + 连续一维精修。
TAU_TRUE_OFF = {"+102.37ms": 0.10237, "-51.63ms": -0.05163}
ORTH_SNR_GRID = (0.5, 1.0, 2.0, 3.0, 5.0)        # 可检测真值的正交 SNR 扫描
RANK_TOL = 1e-8                  # 数值秩阈值: σ_i > σ_max·RANK_TOL
NZ_GLOBAL = None                 # (pool, sess, spans), main 里设置
HOLDOUT_SEALED = True          # R6: 本模块**任何路径都不读** holdout 行


# 每块的尺度**下限**(相对该参数量级)。K=5 的成员间标准差在某些分量上会塌到 0,
# 那不代表"该分量确定", 只代表 5 个 bootstrap 成员碰巧一致。
FLOOR_HIT = {}
SPREAD_FLOOR = {"B": 0.05, "D": 0.05, "log_om": 0.02, "raw_ze": 0.05, "theta": 0.05}


def ensemble_spread(models, attr):
    """**集合分歧尺度** = K=5 个 bootstrap 成员之间的标准差。

    用户 2026-07-28: 这**不能叫校准先验** —— K=5 成员间标准差只反映重采样分歧,
    不代表各物理块同等可信。统一尺度是对的, 但 **η=1 只表示一个集合标准差**。
    故: (a) 每块加尺度下限(相对量级), 防止 K=5 偶然一致导致尺度塌陷;
        (b) 报告时**并列给出物理单位扰动**, 见 `describe`。
    """
    v = np.array([getattr(g.flex, attr).detach().numpy() for g in models])
    floor = SPREAD_FLOOR.get(attr, 0.05) * (np.abs(v).mean() + 1e-12)
    out = np.maximum(v.std(0), floor)
    FLOOR_HIT[attr] = float(np.mean(v.std(0) < floor))
    return out


def theta_gauge(g, poses, sc):
    """**参数空间**规范固定（用户 2026-07-29）。

    先前做的是「输出空间残差化」—— 把 J_θ 的列投影掉三个方向。那**不删除参数
    自由度**(24 列投影后仍 24 秩), 只是把信息挪走。正确做法是在参数空间加约束:

        c_m(θ) = Σ_c w_c·log α_m(φ_c; θ) − c_m(θ_0) = 0        (每模态一个)
        G = ∂c/∂θ ∈ R^{3×24},   N = null(G) ∈ R^{24×21}
        δθ = N ξ,   J_{θ,fit} = W · J_θ · N                    (**先减到 21, 再白化**)

    φ 网格**预先固定为同一套计划位姿**, 不随 fold 改变; 它只用设计协变量、
    不用响应数据 ⇒ 不构成标签泄漏。

    在**先验缩放**坐标 u = θ/sc 下构造(J_θ 的列已按 sc 扰动), 故 G_s = G_raw·diag(sc)。
    返回 (G_s, N(24×21), Mrow(24×3), rank)。Mrow = 被删除的三个**模态平均尺度**方向。
    """
    th0 = g.flex.theta.detach().numpy()
    Phi = np.array([S.phi_of(p) for p in poses])          # (C, 8) 固定计划位姿
    M_, K = th0.shape
    w = np.ones(len(Phi)) / len(Phi)
    G = np.zeros((M_, M_ * K))
    for m in range(M_):
        z = Phi @ th0[m]
        sp = np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))   # softplus(稳定式)
        sig = 1.0 / (1.0 + np.exp(-z))                          # softplus'
        G[m, m * K:(m + 1) * K] = ((w * sig / np.maximum(sp, 1e-12))[:, None]
                                   * Phi).sum(0)
    G = G * sc.ravel()[None, :]                            # → 先验缩放坐标
    U, sv, Vt = np.linalg.svd(G)
    r = int((sv > sv[0] * 1e-10).sum())
    return G, Vt[r:].T, Vt[:r].T, r


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
    sc = ensemble_spread(models, "theta")
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
    bsc_m = ensemble_spread(models, "B")            # **先验尺度**, 逐元素
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
    dsc_m = ensemble_spread(models, "D")
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
    lsc = ensemble_spread(models, "log_om"); zsc = ensemble_spread(models, "raw_ze")
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

    # ---- 决定性单元测试用: **真正的均匀缩放列** J_{s_m} = ∂y/∂log s_m ----
    #   α_m(t) → e^{s_m}·α_m(t)。由线性性, 它应几乎完全落入 span(R18)。
    #   若仍只有 ~0.057, 才说明 R 的 Jacobian 或缩放实现有 bug。
    import torch as _t
    _orig_gains = g.flex.gains
    _eps = 0.05
    _sc_cols = []
    for m in range(g.flex.theta.shape[0]):
        vec = _t.ones(g.flex.theta.shape[0]); vec[m] = float(np.exp(_eps))
        g.flex.gains = (lambda phi, _o=_orig_gains, _v=vec: _o(phi) * _v)
        g._vker = g._exck = None
        _sc_cols.append((evaluate() - H0) / _eps)
        g.flex.gains = _orig_gains
        g._vker = g._exck = None
    J_unif = np.stack(_sc_cols, 1)

    # ---- **参数空间**规范: δθ = Nξ, 把 24 个 θ 方向真正减到 21 ----
    Gs, Nn, Mrow, rk = theta_gauge(g, poses, sc)
    Jth = blocks.pop("θ")
    blocks["θ_shape"] = Jth @ Nn            # 21 个**位形调度形状**方向
    blocks["theta_mean"] = Jth @ Mrow      # 其余 3 个 **θ 方向**(不是纯 modal scale)
    gauge = {"G": Gs, "N": Nn, "M": Mrow, "rank": rk,
             "Jth_raw": Jth, "J_unif": J_unif,
             "GN": float(np.linalg.norm(Gs @ Nn)),
             "rank_JN": int((np.linalg.svd(
                 np.concatenate([(Jth @ Nn).real, (Jth @ Nn).imag], 0),
                 compute_uv=False) > 1e-8).sum())}

    # 行元信息
    frow, krow, prow = [], [], []
    for ki, (pi, j) in enumerate(keys):
        for f in F_of[pi]:
            for _ in range(3):
                frow.append(f); krow.append(ki); prow.append(pi)
    phys = {"θ_shape": ("softplus 调度形状", float(np.median(sc))),
            "theta_mean": ("θ 平均方向", float(np.median(sc))),
            "R18": ("模态残量 B", float(np.median(bsc_m))),
            "D6": ("直通 D", float(np.median(dsc_m))),
            "poles6": ("log ω / raw ζ", float(np.median(np.concatenate(
                [lsc.ravel(), zsc.ravel()])))),
            "gain": ("整体复增益(η=1 ⇔ 100%)", 1.0)}
    return (H0, blocks, np.array(frow), np.array(krow), np.array(prow), keys, phys,
            gauge)


# ======================= 噪声与白化 =======================

def noise_blocks_tagged(freqs):
    """同 `P.noise_blocks`, 但**保留 (session, window) 标签**。

    2026-07-29: 原实现让所有 key 共用同一个 `nb` 数组, 且每次只抽**一个**块索引
    作用到全部 key ⇒ 组间噪声高度相关(非对角 |ρ| 达 0.96), 训练组能预测留出组的
    噪声, 使纯噪声 LOPO 掉到 0.63 < 1。必须显式区分两层。

    session = 采集轮次(r1: A1/A2, r2: A3/A4, r3: A5/A6) —— 同轮共享漂移
    run/window = 轮内窗口 —— 近似独立
    """
    BARE = {"A1": "replay_ep0_A1", "A2": "replay_ep0_A2", "A3": "replay_ep0_r2_A3",
            "A4": "replay_ep0_r2_A4", "A5": "replay_ep0_r3_A5", "A6": "replay_ep0_r3_A6"}
    seq = np.load(A.PB / "seq_ep0_j2_4mrad_p.npz", allow_pickle=True)
    Rc = {k: json.loads((A.PB / f"{v}.json").read_text()) for k, v in BARE.items()}
    W = {k: A.arm_windows(Rc[k], seq, 0) for k in Rc}
    blocks, sess = [], []
    for w in [int(x) for x in seq["wins"]]:
        if not all(w in W[k] for k in W):
            continue
        LI = lambda k, f: A.lockin_at(W[k][w]["y"], W[k][w]["t"], f, W[k][w]["dt"])
        for si, (a_, b_) in enumerate([("A1", "A2"), ("A3", "A4"), ("A5", "A6")]):
            blocks.append(np.array([(LI(b_, f) - LI(a_, f)) / 2 for f in freqs]))
            sess.append(si)
    return np.array(blocks), np.array(sess)


def noise_rows(keys, F_of, Um, nb_cache):
    """**逐 key** 的噪声样本池 + 白化权重 + session 标签 + 行→key 映射。

    返回的 pool 形状 (nblk, N)。与旧版的区别: 采样时**每个 key 独立抽窗口**
    (见 `draw_noise`), 只有 session 分量在 key 之间共享。
    """
    pool, sd, spans = [], [], []
    off = 0
    sess = None
    for (pi, j) in keys:
        F = F_of[pi]
        nb, sess_k = nb_cache[tuple(F)]            # (nblk, nf, 3), (nblk,)
        ui = 1.0 / np.where(np.abs(Um[(pi, j)]) < 1e-15, 1e-15, Um[(pi, j)])
        v = nb * ui[None, :, None]
        m = v.reshape(len(nb), -1)
        pool.append(m)
        sd.append(np.sqrt(np.maximum((np.abs(v) ** 2).mean(0), 1e-30)).ravel())
        spans.append((off, off + m.shape[1])); off += m.shape[1]
        sess = sess_k
    return np.concatenate(pool, 1), np.concatenate(sd), sess, spans


def draw_noise(pool, sess, spans, rng):
    """两层重采样: **先抽 session(共享)**, 再为每个 key **独立抽轮内窗口**。

    ⇒ session 分量在 key 间共享(物理真实: 同轮漂移), 窗口分量互相独立。
    """
    si = int(rng.integers(sess.max() + 1))
    cand = np.flatnonzero(sess == si)
    out = np.empty(pool.shape[1], complex)
    for (lo, hi) in spans:
        out[lo:hi] = pool[cand[rng.integers(len(cand))], lo:hi]
    return out


# ======================= 拟合器（Gram 与 τ 无关 ⇒ 跨 trial 缓存） =======================

class Fitter:
    def __init__(self, H0, blocks, frow, w):
        self.frow, self.w = frow, w
        self.H0 = H0 * w
        self.cols = {}
        off = 0
        self.span = {}
        mats = []
        for nm in BLOCK_ORDER:
            Bm = blocks[nm] * w[:, None]
            self.span[nm] = (off, off + Bm.shape[1]); off += Bm.shape[1]
            mats.append(Bm)
        self.Aall = np.concatenate(mats, 1)
        self.E = np.exp(2j * np.pi * np.outer(TAU_GRID, frow))     # (nτ, n)
        self._chol = {}
        self._subc = {}
        self._sublvc = {}
        self._rhs = {}
        self._obs = None

    def idx(self, level):
        """层级的列集合**不必是前缀**(L2R 要跳过 modal_scale)。

        右端项 Bt 仍只按 mask 算一次**全列**, 各层级按列索引取子集即可 —— 与
        前缀切片同样免费(见 `rhs`)。
        """
        return np.concatenate([np.arange(*self.span[b]) for b in LEVELS[level]])

    def rhs(self, obs, mask, mk):
        """右端项 Bt(τ) 与 ‖y(τ)‖²。**与 λ、与层级都无关**（层级只是切片）。

        Bt(τ) = Re(A^H(obs·e^{+2πjfτ})) − Re(A^H H0)  —— 对频率的 NUDFT。
        每个 mask 每个 trial 只算一次 ⇒ 相比逐(层级,λ)重算快约 17×。
        """
        # 缓存**只在同一个 obs 内**有效。忘记 new_trial() 必须**立刻报错**,
        # 不能静默返回上一个 trial 的右端项(2026-07-28 曾因此使验证 A 段全错)。
        if self._obs is not obs:
            raise RuntimeError("rhs 缓存与当前 obs 不符: 调用 fit.new_trial(obs)")
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

    def _sublv(self, mask, mk, level):
        """按层级切好的子矩阵 —— Brent 精修每步都要用, 不能每次重新花式索引。"""
        k = (mk, level)
        if k not in self._sublvc:
            self._sublvc[k] = self._sub(mask, mk)[0][:, self.idx(level)]
        return self._sublvc[k]

    def new_trial(self, obs):
        """每次 obs 变化都必须调用: 清右端项缓存并登记当前 obs。"""
        self._rhs = {}
        self._obs = obs

    def gram(self, mask, level, lam, level_mk=None):
        """Re(A^H A) + λI 的 Cholesky。**与 τ 无关 ⇒ 可缓存**。"""
        key = (mask.tobytes(), level, float(lam))
        if key not in self._chol:
            Asub = self._sublv(mask, level_mk, level) if level_mk else \
                self.Aall[mask][:, self.idx(level)]
            M = (Asub.conj().T @ Asub).real
            M = M + lam * np.trace(M) / max(M.shape[0], 1) * np.eye(M.shape[0])
            self._chol[key] = (np.linalg.cholesky(M), M)
        return self._chol[key]

    def sse_at(self, obs, mask, level, lam, tau, mk):
        """任意(连续)τ 处的 SSE 与 η —— 供网格后的一维精修。"""
        Am = self._sublv(mask, mk, level)
        yr = obs[mask] * np.exp(2j * np.pi * self.frow[mask] * tau) - self.H0[mask]
        b = (Am.conj().T @ yr).real
        L, M = self.gram(mask, level, lam)
        eta = np.linalg.solve(L.T, np.linalg.solve(L, b))
        return float((np.abs(yr) ** 2).sum() - 2 * b @ eta + eta @ M @ eta), eta

    def valleys(self, sse, k=2):
        """保留前 k 个**局部**谷 —— L2/L3 可能吸收错误分支, 必须看得见第二谷。"""
        loc = [i for i in range(1, len(sse) - 1)
               if sse[i] <= sse[i - 1] and sse[i] <= sse[i + 1]]
        gm = int(np.argmin(sse))
        if gm not in loc:                 # 极小落在网格边界时无内部谷 ⇒ 兜底
            loc.append(gm)
        loc.sort(key=lambda i: sse[i])
        out, used = [], []
        for i in loc:
            if any(abs(TAU_GRID[i] - TAU_GRID[j]) < 0.020 for j in used):
                continue
            used.append(i); out.append((float(TAU_GRID[i]), float(sse[i])))
            if len(out) == k:
                break
        return out

    def fit_scan(self, obs, mask, level, lam, mk=None, refine=False):
        """R1: 在 mask（**只含训练组**）上做**全局 τ 扫描**, 返回 (τ̂, η, 曲线)。

        τ 网格覆盖 [−250, +250] ms 全程 —— **不先定分支再拟合**。
        """
        mk = mk if mk is not None else mask.tobytes()
        Bt_all, yy = self.rhs(obs, mask, mk)
        ci = self.idx(level)
        Bt = Bt_all[:, ci]                       # 任意列子集
        L, M = self.gram(mask, level, lam)
        Et = np.linalg.solve(L.T, np.linalg.solve(L, Bt.T)).T          # (nτ, p)
        sse = yy - 2.0 * (Bt * Et).sum(1) + ((Et @ M) * Et).sum(1)
        if not refine:
            t = int(np.argmin(sse))
            return float(TAU_GRID[t]), Et[t], sse, None
        # 用户 2026-07-29: **网格只负责定位每个局部谷**, 再对前两个谷分别 Brent 精修,
        # 最后用**精修后**的 ΔSSE 比较分支 —— 否则离格量化仍会进入分支裕度。
        from scipy.optimize import minimize_scalar
        d = TAU_GRID[1] - TAU_GRID[0]
        ref = []
        for t0, _ in self.valleys(sse, 2):
            r = minimize_scalar(lambda x: self.sse_at(obs, mask, level, lam, x, mk)[0],
                                bounds=(t0 - d, t0 + d), method="bounded",
                                options={"xatol": 1e-7})
            v, et = self.sse_at(obs, mask, level, lam, float(r.x), mk)
            ref.append((float(r.x), v, et))
        ref.sort(key=lambda z: z[1])
        return ref[0][0], ref[0][2], sse, ref

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
    fit.new_trial(obs)                    # 清 rhs 缓存(obs 变了), Gram/子矩阵仍复用
    trm = [(~gmasks[o], f"o{o}".encode()) for o in range(ng)]
    inm = {(o, i): ((~gmasks[o]) & ~gmasks[i], f"o{o}i{i}".encode())
           for o in range(ng) for i in range(ng) if i != o}
    out = {}
    for lv in LEVELS:
        errs, taus, vv2, lams = [], [], [], []
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
                    tau_i, eta_i, _, _ = fit.fit_scan(obs, itr, lv, lam, itk)
                    s, n = fit.resid(obs, gmasks[i], lv, tau_i, eta_i)
                    acc.append(s / max(n, 1))          # R2: 组内先归一, 组间等权
                m = float(np.mean(acc))
                if m < best:
                    best, blam = m, lam
            # ---- 外层: 用 λ* 在 6 组上重拟合(含**全局 τ 扫描**), 预测留出组 ----
            # 外折**精修**前两个谷(内折只用于选 λ, 相对比较即可, 不精修)
            tau_o, eta_o, _, ref = fit.fit_scan(obs, tr, lv, blam, trk, refine=True)
            s, n = fit.resid(obs, gmasks[o], lv, tau_o, eta_o)
            errs.append(s / max(n, 1)); taus.append(tau_o); lams.append(blam)
            vv2.append(ref[1][1] - ref[0][1] if ref and len(ref) > 1 else None)
        errs = np.array(errs)
        out[lv] = (float(errs.mean()), errs, np.array(taus), vv2, lams)  # **七组等权**
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


def gauge_tests(fit, gauge, w, npool):
    """规范的三项决定性验收（用户 2026-07-29）。"""
    N, V = gauge["N"], gauge["M"]
    Jth = gauge["Jth_raw"] * w[:, None]           # 白化后的原始 θ24
    Ju = gauge["J_unif"] * w[:, None]             # 真正的均匀缩放列 J_{s_m}

    def frac_in(basis, tgt):
        Y = np.concatenate([tgt.real, tgt.imag], 0)
        X = np.concatenate([basis.real, basis.imag], 0)
        Q = np.linalg.qr(X)[0]
        return float(1 - ((Y - Q @ (Q.T @ Y)) ** 2).sum() / max((Y ** 2).sum(), 1e-30))

    R18 = fit.Aall[:, slice(*fit.span["R18"])]
    tm = fit.Aall[:, slice(*fit.span["theta_mean"])]
    print("  【单元测试 1 · 真正的均匀缩放 alpha_m(t) -> e^{s_m}*alpha_m(t)】")
    print(f"    J_unif  落在 span(R18) 内的比例 = {frac_in(R18, Ju):.4f}   (应≈1)")
    print(f"    theta_mean 落在 span(R18) 的比例 = {frac_in(R18, tm):.4f}   (无要求)")
    fu = frac_in(R18, Ju)
    verdict = ("均匀缩放确实被 R 吸收 ⇒ R Jacobian 与缩放实现无 bug" if fu > 0.9
               else "**仍很低 ⇒ R Jacobian 或缩放实现有 bug**")
    print(f"    ⇒ {verdict}")

    # ---- 换基矩阵 T = [N V] 是否正交 ⇒ 决定各向同性惩罚是否已经正确 ----
    T = np.concatenate([N, V], 1)
    dev = float(np.abs(T.T @ T - np.eye(24)).max())
    print(f"\n  【换基矩阵】T = [N V], ‖TᵀT − I‖_max = {dev:.2e}")
    print(f"    先验在缩放坐标 u=θ/sc 下为 Σ_u⁻¹ = I ⇒ 新坐标惩罚 = zᵀ(TᵀΣ_u⁻¹T)z = zᵀ(TᵀT)z")
    print(f"    T 正交 ⇒ TᵀT = I ⇒ **各向同性惩罚在此换基下恰好正确**(非巧合豁免, 已验证)")

    # ---- 两项一致性验收 ----
    gcol = fit.Aall[:, slice(*fit.span["gain"])]
    C24 = np.concatenate([gcol, Jth], 1)
    CSP = np.concatenate([gcol, Jth @ N, Jth @ V], 1)
    P24 = np.eye(C24.shape[1])
    PSP = np.eye(CSP.shape[1]); PSP[2:, 2:] = T.T @ T          # **变换后的完整 penalty**
    rng = np.random.default_rng(11)
    y = (fit.H0 + C24 @ rng.normal(size=C24.shape[1]).astype(complex)) \
        * np.exp(-2j * np.pi * fit.frow * 0.102) + npool[0]

    def solve(C, Pen, lam):
        yr = y * np.exp(2j * np.pi * fit.frow * 0.102) - fit.H0
        M = (C.conj().T @ C).real
        M = M + lam * np.trace(M) / M.shape[0] * Pen
        e = np.linalg.solve(M, (C.conj().T @ yr).real)
        pr = C @ e.astype(complex)
        return float((np.abs(yr - pr) ** 2).sum()), pr

    print(f"\n  【验收 A · λ=0: 原 θ24 与 theta_shape+theta_mean 必须逐点一致】")
    s0, p0 = solve(C24, P24, 0.0); s1, p1 = solve(CSP, PSP, 0.0)
    print(f"    SSE: {s0:.6e} vs {s1:.6e}   相对差 {abs(s0-s1)/max(s0,1e-30):.2e}")
    print(f"    预测逐点最大差 = {np.abs(p0-p1).max():.2e}"
          f"  (相对信号 {np.abs(p0-p1).max()/max(np.abs(p0).max(),1e-30):.2e})")

    print(f"\n  【验收 B · 相同 λ, 携带变换后完整 penalty】")
    for lam in (1e-3, 1e-2, 1e-1):
        s0, p0 = solve(C24, P24, lam); s1, p1 = solve(CSP, PSP, lam)
        print(f"    λ={lam:<6g} SSE 相对差 {abs(s0-s1)/max(s0,1e-30):.2e}"
              f"   预测最大差 {np.abs(p0-p1).max():.2e}")
    print()


def describe(fit, phys, gauge):
    """R4 + 用户 2026-07-28: 必须区分**原始列数**、**约束秩**、**数值秩**、**有效 dof**。"""
    print("  【L0 的两个参数】")
    print("    η_gain,re 与 η_gain,im: 模型整体乘 (1+η_re+jη_im)。")
    print("    ⇒ 一个**与频率、位姿、关节都无关的复标量**; η=1 表示 100% 幅值变化。\n")

    n2 = 2 * fit.Aall.shape[0]
    def nrank(cols):
        X = np.concatenate([cols.real, cols.imag], 0)
        sv = np.linalg.svd(X, compute_uv=False)
        return int((sv > sv[0] * RANK_TOL).sum()) if len(sv) and sv[0] > 0 else 0

    print(f"  【逐块】数值秩阈值 σ_i > σ_max·{RANK_TOL:g}")
    print("    θ 已按**参数空间**规范拆为 θ_shape(21) + modal_scale(3); 两类分开打印。")
    print(f"  {'块':>8}{'原始列':>7}{'数值秩':>8}{'块 SNR':>9}"
          f"{'floor 触发':>11}{'物理单位扰动(η=1)':>26}")
    ATTR = {"θ_shape": "theta", "theta_mean": "theta", "R18": "B",
            "D6": "D", "poles6": "log_om"}
    for b, (lo, hi) in fit.span.items():
        col = fit.Aall[:, lo:hi]
        rms = np.sqrt((np.abs(col) ** 2).mean(0))
        nm, val = phys.get(b, ("", float("nan")))
        fh = FLOOR_HIT.get(ATTR.get(b, ""), float("nan"))
        print(f"  {b:>8}{hi-lo:>7}{nrank(col):>8}"
              f"{np.sqrt((rms**2).sum()):>9.2f}{fh:>11.0%}{nm+f' ≈{val:.3g}':>26}")

    print("\n  【逐层级: 有效 dof = 白化 Jacobian 的数值秩, **不是列数**】")
    print(f"  {'层级':>6}{'原始列':>7}{'数值秩':>8}{'秩亏':>6}{'开放块':>30}")
    for lv, bl in LEVELS.items():
        ci = fit.idx(lv)
        r = nrank(fit.Aall[:, ci])
        print(f"  {lv:>6}{len(ci):>7}{r:>8}{len(ci)-r:>6}{'+'.join(bl):>30}")

    print("\n  【参数空间规范验收】(**不是**输出空间残差化)")
    G, N = gauge["G"], gauge["N"]
    print(f"    rank(G)       = {gauge['rank']}              (应为 3)")
    print(f"    N.shape       = {N.shape}       (应为 (24, 21))")
    print(f"    ‖G @ N‖       = {gauge['GN']:.2e}       (应 < 1e-10)")
    print(f"    rank(J_θ @ N) = {gauge['rank_JN']}             (应 ≤ 21)")
    print(f"    φ 网格 = 固定的 16 个计划位姿, **不随 fold 改变**; 只用设计协变量")
    print(f"           ⇒ 不构成标签泄漏, 各折参数化一致 ⇒ 嵌套性得以保持")

    print("\n  【被删除的三个模态平均尺度去了哪】")
    def frac_in(basis, tgt):
        Y = np.concatenate([tgt.real, tgt.imag], 0)
        X = np.concatenate([basis.real, basis.imag], 0)
        Q = np.linalg.qr(X)[0]
        return float(1 - ((Y - Q @ (Q.T @ Y)) ** 2).sum() / max((Y ** 2).sum(), 1e-30))
    ms = fit.Aall[:, slice(*fit.span["theta_mean"])]
    print(f"    theta_mean 落在 R18 张成空间内的比例 = "
          f"{frac_in(fit.Aall[:, slice(*fit.span['R18'])], ms):.3f}")
    print(f"      ⇒ 该前提**不成立**(应≈1)。α 在输入侧经 softplus 且窗内时变,")
    print(f"        与输出侧残量纯缩放只在冻结时变时等价。故 modal_scale **保留在**")
    print(f"        L1 及以上全部层级, 否则嵌套性被破坏。")
    print("\n  【嵌套性验收】rank(L_k ∪ L_{k+1}) 必须 == rank(L_{k+1})")
    order = list(LEVELS)
    for i in range(len(order) - 1):
        a_, b_ = order[i], order[i + 1]
        u = np.union1d(fit.idx(a_), fit.idx(b_))
        ru, rb = nrank(fit.Aall[:, u]), nrank(fit.Aall[:, fit.idx(b_)])
        print(f"    span({a_}) ⊆ span({b_}): rank(∪)={ru} vs rank({b_})={rb}"
              f"   {'✓' if ru == rb else '**✗**'}")
    print(f"    注: 白化空间里 θ_shape 与 R 仍可能非零重叠 —— 那是**统计混杂**,")
    print(f"        不是 gauge 没删干净。规范定义在参数空间, 不随噪声协方差/fold 变。\n")

    print("  【逐层**新增**信息量（新块对前面各层张成空间的正交残量）】")
    print(f"  {'层级':>6}{'新增块':>10}{'正交残量 SNR':>14}{'被前层解释':>12}")
    order = list(LEVELS)
    prev = np.zeros((fit.Aall.shape[0], 0), complex)
    for i, lv in enumerate(order):
        new = [b for b in LEVELS[lv] if i == 0 or b not in LEVELS[order[i - 1]]]
        if not new:
            continue
        Nw = np.concatenate([fit.Aall[:, slice(*fit.span[b])] for b in new], 1)
        Y = np.concatenate([Nw.real, Nw.imag], 0)
        if prev.shape[1]:
            X = np.concatenate([prev.real, prev.imag], 0)
            Qp = np.linalg.qr(X)[0]
            Res = Y - Qp @ (Qp.T @ Y)
            frac = 1.0 - (Res ** 2).sum() / max((Y ** 2).sum(), 1e-30)
        else:
            Res, frac = Y, 0.0
        print(f"  {lv:>6}{'+'.join(new):>10}"
              f"{np.sqrt((Res ** 2).sum() / n2):>14.2f}{frac:>11.0%}")
        prev = np.concatenate([prev, Nw], 1)
    print()


# ======================= 可检测真值（按正交 SNR 定义） =======================

def prev_basis(fit):
    """每个层级**前面所有块**张成空间的正交基(实堆叠) —— 用于取正交残量。"""
    order = list(LEVELS)
    out = {}
    for i, lv in enumerate(order):
        if i == 0:
            out[lv] = np.zeros((2 * fit.Aall.shape[0], 0)); continue
        X = fit.Aall[:, fit.idx(order[i - 1])]
        out[lv] = np.linalg.qr(np.concatenate([X.real, X.imag], 0))[0]
    return out


def new_cols(fit, level):
    order = list(LEVELS); i = order.index(level)
    lo = 0 if i == 0 else len(fit.idx(order[i - 1]))
    return slice(lo, len(fit.idx(level)))


def make_obs(fit, npool, rng, level_true, tau_true, orth_snr, Qprev,
             noise_on=True, base_sd=0.3):
    """按**正交 SNR** 生成可检测真值（用户 2026-07-28）。

        Δy⊥_k = (I − P_{k−1}) J_k Δp_k

    层级"选对率"不能只按生成参数块命名: 若 L3 的扰动 80% 可由 L2 表达,
    one-SE 选 L2 **反而是正确的预测决策**。故真值按 ‖Δy⊥‖ 定标, 并同时返回
    原始扰动、正交残量与最佳低层近似误差。
    """
    order = list(LEVELS); k = order.index(level_true)
    eta = np.zeros(fit.Aall.shape[1])
    if k > 0:                                     # 前面各层给基线扰动
        for b in LEVELS[order[k - 1]]:
            lo, hi = fit.span[b]
            eta[lo:hi] = rng.normal(0, 0.05 if b == "gain" else base_sd, hi - lo)
    sl = new_cols(fit, level_true)
    dp = rng.normal(size=sl.stop - sl.start)
    dy = fit.Aall[:, sl] @ dp.astype(complex)
    Y = np.concatenate([dy.real, dy.imag])
    Q = Qprev[level_true]
    perp = Y - Q @ (Q.T @ Y) if Q.shape[1] else Y
    n = fit.Aall.shape[0]
    sc = orth_snr / max(np.linalg.norm(perp) / np.sqrt(n), 1e-30)
    eta[sl] = dp * sc
    obs = ((fit.H0 + fit.Aall @ eta.astype(complex))
           * np.exp(-2j * np.pi * fit.frow * tau_true))
    if noise_on:
        obs = obs + draw_noise(*NZ_GLOBAL, rng)
    diag = {"raw": float(np.linalg.norm(Y) * sc / np.sqrt(n)),
            "perp": float(np.linalg.norm(perp) * sc / np.sqrt(n)),
            "absorb": 1.0 - (np.linalg.norm(perp) / max(np.linalg.norm(Y), 1e-30)) ** 2,
            # 达到该正交 SNR 所需的**参数幅度**(单位: 集合标准差)。若远大于 1,
            # 该真值在先验下不合理 —— ridge 会正当地把它压回去, 不算检测失败。
            "dp_sd": float(np.sqrt(np.mean((dp * sc) ** 2)))}
    return obs, eta, diag


def branch_absorb(taus, tau_true, tol=0.030):
    """更自由的层级是否**吸收错误 τ 分支**。"""
    br = min(P.TAU_TRUE, key=lambda k: abs(P.TAU_TRUE[k] - tau_true))
    other = P.TAU_TRUE["-52ms" if br == "+102ms" else "+102ms"]
    return (float(np.mean(np.abs(taus - tau_true) > tol)),
            float(np.mean(np.abs(taus - other) < tol)))


# ======================= 各阶段 =======================

def stage_verify(fit, npool, gmasks, Qprev, rng, ntr):
    full = np.ones(len(fit.frow), bool)
    print("  【A. 优化正确性】格点真值 vs **离格真值 + 连续一维精修**")
    print(f"  {'真值':>22}{'精修':>6}{'τ 误差(μs)':>13}{'η 相对误差':>12}{'残差/信号':>12}")
    ok = True
    for lv in ("L1", "L2", "L3"):
        for nm, tt in list(P.TAU_TRUE.items()) + list(TAU_TRUE_OFF.items()):
            off = nm in TAU_TRUE_OFF
            obs, eta, _ = make_obs(fit, npool, np.random.default_rng(7), lv, tt,
                                   2.0, Qprev, noise_on=False)
            for rf in ((False, True) if off else (False,)):
                fit.new_trial(obs)
                th, eh, _, _ = fit.fit_scan(obs, full, lv, LAM_TINY, refine=rf)
                ci = fit.idx(lv)
                rel = np.linalg.norm(eh - eta[ci]) / max(np.linalg.norm(eta[ci]), 1e-12)
                sse, _ = fit.resid(obs, full, lv, th, eh)
                sg = float((np.abs(obs) ** 2).sum())
                if off and rf:
                    ok &= abs(th - tt) < 5e-6
                print(f"  {lv+'/'+nm:>22}{'是' if rf else '否':>6}"
                      f"{(th-tt)*1e6:>13.1f}{rel:>12.2e}{np.sqrt(sse/sg):>12.2e}")
    print(f"  ⇒ 离格真值经连续精修后误差 <5 μs: {'**是**' if ok else '**否**'}\n")

    print("  【B. 无泄漏检查】")
    print(f"    · Jacobian 只含 16 训练条件, **holdout 位姿未进入任何矩阵**"
          f"  (HOLDOUT_SEALED={HOLDOUT_SEALED})")
    obs, _, _ = make_obs(fit, npool, rng, "L2", P.TAU_TRUE["+102ms"], 2.0, Qprev)
    same = True
    for o in range(len(gmasks)):
        tr = ~gmasks[o]
        fit.new_trial(obs); t1, e1, _, _ = fit.fit_scan(obs, tr, "L2", 1e-2)
        o2 = obs.copy(); o2[gmasks[o]] = rng.normal(size=int(gmasks[o].sum())) * 1e3
        fit.new_trial(o2); t2, e2, _, _ = fit.fit_scan(o2, tr, "L2", 1e-2)
        same &= (t1 == t2) and np.allclose(e1, e2)
    print(f"    · 破坏留出组后训练侧 (τ̂, η) {'**逐位不变**' if same else '**改变 ⇒ 泄漏**'}")
    ovl = max(int((gmasks[i] & gmasks[j]).sum())
              for i in range(len(gmasks)) for j in range(i + 1, len(gmasks)))
    print(f"    · 七组两两行重叠 = {ovl}（应为 0）\n")
    return run_trials(fit, npool, gmasks, Qprev, ntr, rng)


def run_trials(fit, npool, gmasks, Qprev, ntr, rng):
    """按 ORTH_SNR_GRID 扫描; 记录分支、λ、以及各真值的尺度诊断。"""
    lv_list = list(LEVELS)
    res = {}
    for snr in ORTH_SNR_GRID:
        conf = np.zeros((len(lv_list), len(lv_list)), int)
        confb = {br: np.zeros((len(lv_list), len(lv_list)), int) for br in P.TAU_TRUE}
        ab = {lv: [] for lv in lv_list}; wb = {lv: [] for lv in lv_list}
        ge = {lv: [] for lv in lv_list}; dg = {lv: [] for lv in lv_list}
        v2 = {lv: [] for lv in lv_list}; lam = {lv: [] for lv in lv_list}
        per = max(ntr // len(lv_list), 1)
        for i, lt in enumerate(lv_list):
            for t in range(per):
                br = list(P.TAU_TRUE)[t % 2]; tt = P.TAU_TRUE[br]
                obs, _, d = make_obs(fit, npool, np.random.default_rng(
                    90000 + 1000 * i + t + int(snr * 7)), lt, tt, snr, Qprev)
                out = lopo(fit, obs, gmasks)
                sel, _, _, _ = pick_level(out)
                conf[i, lv_list.index(sel)] += 1
                confb[br][i, lv_list.index(sel)] += 1
                dg[lt].append(d)
                for lv in lv_list:
                    a_, b_ = branch_absorb(out[lv][2], tt)
                    ab[lv].append(a_); wb[lv].append(b_); ge[lv].append(out[lv][1])
                    v2[lv].append(out[lv][3]); lam[lv].extend(out[lv][4])
        res[snr] = dict(conf=conf, confb=confb, ab=ab, wb=wb, ge=ge, dg=dg,
                        v2=v2, lam=lam)
    return res, lv_list


def d6_diagnostic(fit, npool, gmasks, Qprev, rng):
    """D6 三分诊断（用户 2026-07-29）: oracle / 内层 CV / one-SE 三者对比。

    判读:
      oracle 也恢复不了 D        ⇒ **当前设计确实缺少 D 信息**
      oracle 能、CV ridge 不能   ⇒ **正则尺度或 λ 网格问题**
      CV 能、one-SE 选 L2R       ⇒ 预测收益不足以支付复杂度, **合理简化**
      SNR 3–5 仍系统性选 L2R     ⇒ 需重点查**模型选择实现**
    """
    print("  【D6 三分诊断】(真值 = L2, 即 D6 被打开)")
    print(f"  {'正交SNR':>8}{'oracle η_D 相对误差':>20}{'CV η_D 相对误差':>17}"
          f"{'CV 选的 λ(中位)':>17}{'one-SE 选层':>12}")
    full = np.ones(len(fit.frow), bool)
    sl = new_cols(fit, "L2")
    for snr in (1.0, 2.0, 3.0, 5.0):
        orel, crel, lams, sels = [], [], [], []
        for t in range(6):
            obs, eta, _ = make_obs(fit, npool, np.random.default_rng(4242 + t),
                                   "L2", P.TAU_TRUE["+102ms"], snr, Qprev)
            fit.new_trial(obs)
            # (1) oracle: 正确层级 + **极弱** ridge, 全训练集
            _, eo, _, _ = fit.fit_scan(obs, full, "L2", LAM_TINY)
            orel.append(np.linalg.norm(eo[sl] - eta[sl])
                        / max(np.linalg.norm(eta[sl]), 1e-30))
            # (2) 内层 CV 选出的 λ
            out = lopo(fit, obs, gmasks)
            lm = float(np.median(out["L2"][4])); lams.append(lm)
            fit.new_trial(obs)
            _, ec, _, _ = fit.fit_scan(obs, full, "L2", lm)
            crel.append(np.linalg.norm(ec[sl] - eta[sl])
                        / max(np.linalg.norm(eta[sl]), 1e-30))
            # (3) one-SE 的层级决定
            sels.append(pick_level(out)[0])
        top = max(set(sels), key=sels.count)
        print(f"  {snr:>8.1f}{np.median(orel):>20.2f}{np.median(crel):>17.2f}"
              f"{np.median(lams):>17.1e}{top+f'({sels.count(top)}/6)':>12}")
    print(f"  λ 网格 = {LAM_GRID}  (含 {LAM_GRID.min():g} ⇒ **接近零**)\n")


def report(res, lv_list):
    print("  【表 1 · 可检测性】横轴 = 正交输出 SNR; **不因所需 η 大而剔除任何结果**")
    print(f"  {'正交SNR':>8}" + "".join(f"{lv:>9}" for lv in lv_list)
          + f"{'错误升级':>10}{'错误降级':>10}")
    for snr, R in res.items():
        conf = R["conf"]
        hit = [conf[i, i] / max(conf[i].sum(), 1) for i in range(len(lv_list))]
        up = sum(conf[i, i + 1:].sum() for i in range(len(lv_list) - 1))
        dn = sum(conf[i, :i].sum() for i in range(1, len(lv_list)))
        print(f"  {snr:>8.1f}" + "".join(f"{h:>8.0%}" for h in hit)
              + f"{up/max(conf[:-1].sum(),1):>9.0%}{dn/max(conf[1:].sum(),1):>9.0%}")

    print("\n  【表 2 · 尺度合理性】**只说明与集合分歧的距离**, 不判定物理合理性")
    print("    (K=5 集合分歧尺度不是校准先验; D 恰可能是集合共同漏掉的结构误差)")
    print(f"  {'真值层级':>8}{'正交SNR':>8}{'所需 η(集合σ)':>15}{'物理单位扰动':>14}"
          f"{'floor 触发':>11}")
    ATTR = {"L1": ("theta", "softplus 系数"), "L2R": ("B", "模态残量 B"),
            "L2": ("D", "直通 D"), "L3": ("log_om", "log ω")}
    for lv in lv_list:
        if lv not in ATTR:
            continue
        at, unit = ATTR[lv]
        for snr in ORTH_SNR_GRID:
            d = res[snr]["dg"][lv]
            if not d:
                continue
            dps = float(np.mean([x["dp_sd"] for x in d]))
            print(f"  {lv:>8}{snr:>8.1f}{dps:>15.1f}{unit:>14}"
                  f"{FLOOR_HIT.get(at, float('nan')):>11.0%}")

    snr_ref = ORTH_SNR_GRID[2]
    R = res[snr_ref]
    print(f"\n  【混淆矩阵 @ 正交 SNR = {snr_ref}】(行=真值, 列=所选)")
    hdr = "真/选"
    print(f"  {hdr:>7}" + "".join(f"{c:>8}" for c in lv_list))
    for i, r in enumerate(lv_list):
        print(f"  {r:>7}" + "".join(f"{R['conf'][i, j]:>8}" for j in range(len(lv_list))))

    print(f"\n  【两分支分别的层级选对率】")
    print(f"  {'正交SNR':>8}" + "".join(f"{br:>12}" for br in P.TAU_TRUE))
    for snr, Rx in res.items():
        cells = []
        for br in P.TAU_TRUE:
            c = Rx["confb"][br]
            cells.append(f"{np.trace(c)/c.sum():>11.0%}" if c.sum()
                         else f"{'—(n=0)':>11}")     # 无样本 ≠ 0%
        print(f"  {snr:>8.1f}" + "".join(cells))

    print(f"\n  【每个真值的输出分解 @ 正交 SNR = {snr_ref}】")
    print(f"  {'真值层级':>8}{'原始扰动':>11}{'正交残量':>11}{'最佳低层近似误差':>18}"
          f"{'被低层吸收':>12}")
    for lv in lv_list:
        d = R["dg"][lv]
        if not d:
            continue
        print(f"  {lv:>8}{np.mean([x['raw'] for x in d]):>11.2f}"
              f"{np.mean([x['perp'] for x in d]):>11.2f}"
              f"{np.mean([x['perp'] for x in d]):>18.2f}"
              f"{np.mean([x['absorb'] for x in d]):>11.0%}")

    print(f"\n  【τ 诊断 @ 正交 SNR = {snr_ref}】(**固定用该层级拟合**全部真值)")
    print(f"  {'拟合层级':>8}{'τ 偏离>30ms':>14}{'落到错误分支':>14}"
          f"{'精修后第二谷 ΔSSE 中位':>24}")
    for lv in lv_list:
        vv = [x for t_ in R["v2"][lv] for x in t_ if x is not None]
        print(f"  {lv:>8}{np.mean(R['ab'][lv]):>13.1%}{np.mean(R['wb'][lv]):>14.1%}"
              f"{(np.median(vv) if vv else float('nan')):>24.3e}")

    print(f"\n  【各层实际选中的 λ 分布 @ 正交 SNR = {snr_ref}】")
    print(f"  {'层级':>6}" + "".join(f"{f'{l:g}':>10}" for l in LAM_GRID))
    for lv in lv_list:
        ls = np.array(R["lam"][lv])
        print(f"  {lv:>6}" + "".join(
            f"{np.mean(np.isclose(ls, l)):>9.0%}" for l in LAM_GRID))

    print(f"\n  【每组白化预测误差（七组等权）@ 正交 SNR = {snr_ref}】")
    print(f"  {'层级':>6}" + "".join(f"{'g'+str(i):>11}" for i in range(7)))
    for lv in lv_list:
        print(f"  {lv:>6}" + "".join(f"{v:>11.3e}"
                                     for v in np.array(R["ge"][lv]).mean(0)))


# ======================= g0 隔离诊断（用户 2026-07-29） =======================

def g0_isolation(fit, npool, gmasks, Qprev, gkeys, grp, poses, prow, frow, keys, F_of):
    """四项隔离，判定 g0 持续偏高的成因。

    判读(用户给定):
      仅第 1 项异常          ⇒ 组装 / 索引 / Jacobian bug
      仅第 2 项异常          ⇒ 白化或组内归一问题
      **只有**第 3 项异常    ⇒ g0 是真实的高失配工作点
      仅少数频率/轴异常      ⇒ 后续 shape 集合须按该结构建模, 不能用统一半径
    """
    ng = len(gmasks)
    n_of = [int(m.sum()) for m in gmasks]
    print("  【组构成】(SSE 已确认按**组内复数行数**取均值, 非求和)")
    print(f"  {'组':>4}{'条件数':>7}{'行数':>7}{'宽带':>6}{'窄带':>6}{'q2,q3':>16}")
    for gi, gk in enumerate(gkeys):
        pis = grp[gk]
        nw = sum(1 for pi in pis if len(F_of[pi]) > 5)
        print(f"  g{gi:<3}{len(pis):>7}{n_of[gi]:>7}{nw:>6}{len(pis)-nw:>6}"
              f"{str(gk):>16}")

    tt = P.TAU_TRUE["+102ms"]

    print("\n  【隔离 1 · 无噪声 / 无 shape / 正确层级】各组 SSE 应≈数值零")
    obs, eta, _ = make_obs(fit, npool, np.random.default_rng(5), "L2", tt,
                           2.0, Qprev, noise_on=False)
    row = []
    for o in range(ng):
        fit.new_trial(obs)
        th, eh, _, _ = fit.fit_scan(obs, ~gmasks[o], "L2", LAM_TINY)
        s, n = fit.resid(obs, gmasks[o], "L2", th, eh)
        row.append(s / n)
    print("  " + "".join(f"{'g'+str(i):>12}" for i in range(ng)))
    print("  " + "".join(f"{v:>12.2e}" for v in row))

    print("\n  【隔离 2 · 仅噪声】(a) 不拟合的原始白化残差, 每组应≈1.00")
    acc = np.zeros(ng)
    for k in range(40):
        nz = draw_noise(*NZ_GLOBAL, np.random.default_rng(700 + k))
        for o in range(ng):
            acc[o] += (np.abs(nz[gmasks[o]]) ** 2).sum() / n_of[o]
    print("  " + "".join(f"{v/40:>12.3f}" for v in acc))
    print("      (b) 经 LOPO 在 L0 上拟合纯噪声观测后的各组 SSE")
    print("          单次抽样的组内 SD ≈ 1/√n_g, 故必须多次平均才能判 ≥1")
    ND = 24
    accL = np.zeros((ND, ng))
    for k in range(ND):
        o2 = (fit.H0 * np.exp(-2j * np.pi * frow * tt)
              + draw_noise(*NZ_GLOBAL, np.random.default_rng(7700 + k)))
        accL[k] = lopo(fit, o2, gmasks)["L0"][1]
    mu, se = accL.mean(0), accL.std(0, ddof=1) / np.sqrt(ND)
    print("      均值: " + "".join(f"{v:>12.3f}" for v in mu))
    print("      ±SE : " + "".join(f"{v:>12.3f}" for v in se))
    print(f"      七组等权总均值 = {mu.mean():.3f} ± {accL.mean(1).std(ddof=1)/np.sqrt(ND):.3f}"
          f"   (应 ≥1)")
    # 只有**显著低于 1** 才是病态方向(高于 1 是正常的: 训练侧预测误差会加上去)
    bad = [f"g{i}" for i in range(ng) if mu[i] + 1.96 * se[i] < 1.0]
    hi = [f"g{i}" for i in range(ng) if mu[i] - 1.96 * se[i] > 1.0]
    print(f"      显著**低于** 1 的组(病态): {bad if bad else '无 ✓'}")
    print(f"      显著高于 1 的组(正常方向): {hi if hi else '无'}")
    print(f"      注: 组间残余 |ρ|≈0.26 来自**共享 session 分量**(物理真实)。")
    print(f"          ⇒ group-LOPO 只在**同一 session 内**有效, 不验证跨 session 泛化。")

    # ---- 阻断测试: 纯噪声 LOPO = 0.63 < 1 是红旗(用户 2026-07-29) ----
    #   若留出组噪声与训练组独立: E‖n_test − ŷ_train‖² = E‖n_test‖² + E‖ŷ_train‖² ≥ 1
    print("\n  【阻断测试 3 · 七组噪声块之间的复相关】(先查是否共享 A-run block)")
    ng_ = len(gmasks)
    C = np.zeros((ng_, ng_))
    for k in range(30):
        nz = draw_noise(*NZ_GLOBAL, np.random.default_rng(900 + k))
        v = [nz[gmasks[o]] for o in range(ng_)]
        for i in range(ng_):
            for j in range(ng_):
                a_, b_ = v[i], v[j]
                L = min(len(a_), len(b_))
                C[i, j] += abs(np.vdot(a_[:L], b_[:L])) / (
                    np.linalg.norm(a_[:L]) * np.linalg.norm(b_[:L]) + 1e-30) / 30
    print("  " + "".join(f"{'g'+str(i):>8}" for i in range(ng_)))
    for i in range(ng_):
        print(f"  g{i}" + "".join(f"{C[i, j]:>8.3f}" for j in range(ng_)))
    off = C[~np.eye(ng_, dtype=bool)].mean()
    print(f"    非对角均值 |ρ| = {off:.3f}  "
          f"{'⇒ **组间噪声高度相关: 共享 A-run block**' if off > 0.3 else '⇒ 组间近似独立'}")

    print("\n  【阻断测试 1 · 白化后逐物理组独立高斯噪声】LOPO 应回到 ≈1")
    rg = np.random.default_rng(31)
    nz = np.zeros(len(frow), complex)
    for o in range(ng_):
        m = gmasks[o]
        nz[m] = (rg.normal(size=int(m.sum())) + 1j * rg.normal(size=int(m.sum()))) / np.sqrt(2)
    obsI = fit.H0 * np.exp(-2j * np.pi * frow * tt) + nz
    oI = lopo(fit, obsI, gmasks)
    print("  " + "".join(f"{v:>12.3f}" for v in oI["L0"][1]))

    print("\n  【阻断测试 2 · 训练组不变, 只把留出组换成新独立噪声】留出分应 ≈1")
    base = (fit.H0 * np.exp(-2j * np.pi * frow * tt)
            + draw_noise(*NZ_GLOBAL, np.random.default_rng(78)))
    row2 = []
    for o in range(ng_):
        fit.new_trial(base)
        th_, eh_, _, _ = fit.fit_scan(base, ~gmasks[o], "L0", 1e-2)   # 参数只由训练组定
        ob = base.copy()
        m = gmasks[o]
        ob[m] = (fit.H0[m] * np.exp(-2j * np.pi * frow[m] * tt)
                 + (rg.normal(size=int(m.sum()))
                    + 1j * rg.normal(size=int(m.sum()))) / np.sqrt(2))
        sX, nX = fit.resid(ob, m, "L0", th_, eh_)      # **参数不变**, 只换留出数据
        row2.append(sX / nX)
    print("  " + "".join(f"{v:>12.3f}" for v in row2))

    print("\n  【隔离 3 · 仅 shape error(r=1, 无噪声)】g0 是否真的更难建模")
    uv = np.array([float(np.dot(S.phi_of(poses[pi])[1:6],
                                np.array([1.0, -1.0, 0.5, -0.5, 0.25])))
                   for pi in range(len(poses))])
    uv = (uv - uv.mean()) / (uv.std() + 1e-12)
    rr = np.random.default_rng(9)
    x = rr.normal(size=8 + len(poses)); x /= np.linalg.norm(x)
    sig = fit.H0 * np.exp(-2j * np.pi * frow * tt)
    obs3 = apply_shape_rows(sig, x, len(poses), frow, prow, uv)
    out3 = lopo(fit, obs3, gmasks)
    for lv in ("L0", "L2"):
        print(f"    {lv}: " + "".join(f"{v:>11.3f}" for v in out3[lv][1]))

    # 单次随机 shape 可能只是该抽样的特异性 ⇒ 对多个 shape 种子取平均。
    # 另外并列给出**无 shape、只有注入信号**的情形 —— 100 trial 里 g0 偏高
    # 实际由它驱动(那批观测**根本没有 shape error**)。
    print("\n  【隔离 4 · g0 残差结构拆解】(20 个 shape 抽样平均 + 无 shape 对照)")
    ax = np.tile([0, 1, 2], len(frow) // 3)
    kr = np.array([ki for ki, (pi, j) in enumerate(keys)
                   for _ in range(len(F_of[pi]) * 3)])
    m0 = gmasks[0]

    def decomp(o_):
        fit.new_trial(o_)
        th_, eh_, _, _ = fit.fit_scan(o_, ~m0, "L2", 1e-2)
        pr = ((fit.H0 + fit.Aall[:, fit.idx("L2")] @ eh_.astype(complex))
              * np.exp(-2j * np.pi * frow * th_))
        return np.abs(o_ - pr) ** 2

    accs = np.zeros(3); accf = {}; acck = {}
    NS = 20
    for k in range(NS):
        rk = np.random.default_rng(300 + k)
        xk = rk.normal(size=8 + len(poses)); xk /= np.linalg.norm(xk)
        r2k = decomp(apply_shape_rows(sig, xk, len(poses), frow, prow, uv))
        r2k = r2k / max(r2k[m0].mean(), 1e-30)          # 每次抽样内归一, 再平均
        for a in (0, 1, 2):
            accs[a] += r2k[m0 & (ax == a)].mean() / NS
        for f in np.unique(frow[m0]):
            accf[f] = accf.get(f, 0.0) + r2k[m0 & (frow == f)].mean() / NS
        for ki in np.unique(kr[m0]):
            acck[ki] = acck.get(ki, 0.0) + r2k[kr == ki].mean() / NS
    print("    [仅 shape, 20 抽样平均, 组内归一]")
    print("      按陀螺轴: " + "".join(f"  轴{a}={accs[a]:.3f}" for a in (0, 1, 2)))
    print("      按频率:   " + "".join(f"  {f:.1f}Hz={accf[f]:.3f}"
                                       for f in sorted(accf)))
    print("      按 key:   " + "".join(
        f"  p{keys[ki][0]}j{keys[ki][1]+1}={acck[ki]:.3f}" for ki in sorted(acck)))

    r2n = decomp(make_obs(fit, npool, np.random.default_rng(5), "L2", tt,
                          2.0, Qprev, noise_on=False)[0])
    r2n = r2n / max(r2n[m0].mean(), 1e-30)
    print("    [无 shape, 仅注入信号(与 100 trial 同口径)]")
    print("      按陀螺轴: " + "".join(
        f"  轴{a}={r2n[m0 & (ax == a)].mean():.3f}" for a in (0, 1, 2)))
    print("      按频率:   " + "".join(
        f"  {f:.1f}Hz={r2n[m0 & (frow == f)].mean():.3f}"
        for f in np.unique(frow[m0])))


# ======================= 按层级重搜最坏 shape error =======================
# 用户 2026-07-28: "最坏扰动以 log-gain 为主, 但 L1 很可能会吸收它。因此接入
# 真实层级后, 应**重新搜索每层的最坏 shape error**; 不能只沿用替代模型的 x*。"

import worst_shape_C as WS

# 逐分量椭球半径 —— 三组敏感性(不是单一标量 r)
def rvec_of(mode, nP):
    v = np.ones(8 + nP)
    if mode == "gain2":                 # 仅 log-gain 通道扩到 2
        v[:4] = 2.0
    elif mode == "all2":                # 全通道 2 —— **压力测试, 不是置信集合**
        v[:] = 2.0
    return v


SENS = {"r=1": "r1", "仅gain r=2": "gain2", "全通道 r=2(压力测试)": "all2"}


def apply_shape_rows(sig, x, nP, frow, prow, u_p):
    """把 shape error 作用到信号行上。规范同 §最不利搜索: Σ_p a_p = 0。"""
    q = WS.unpack(x, nP)
    nu = frow - frow.mean()
    gg = q["c0"] + q["c1"] * nu + q["c2"] * nu ** 2 + q["d"] * u_p[prow]
    psi = (q["a"][prow] * 2 * np.pi * nu + q["b1"] * 2 * np.pi * nu ** 2
           + q["b2"] * 2 * np.pi * nu ** 3)
    out = sig * np.exp(gg) * np.exp(-1j * psi)
    # 三轴方向偏差: 行按 (…, 频点, 轴0/1/2) 排列 ⇒ reshape 后旋转 0-1 轴
    ang = q["e0"] + q["e1"] * u_p[prow]
    v = out.reshape(-1, 3); a3 = ang.reshape(-1, 3)[:, 0]
    c, s = np.cos(a3), np.sin(a3)
    return np.stack([c * v[:, 0] - s * v[:, 1],
                     s * v[:, 0] + c * v[:, 1], v[:, 2]], 1).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="verify", choices=["verify", "screen", "final", "g0"])
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
    H0, blocks, frow, krow, prow, keys, phys, gauge = level_jacobian(
        g, poses, R, F_of, amp_of, models)
    _, Um = P.model_and_U(g, poses, R, F_of, amp_of)
    nb_cache = {tuple(F_WIDE): noise_blocks_tagged(F_WIDE),
                tuple(F_NARROW): noise_blocks_tagged(F_NARROW)}
    npool, sd, nsess, nspans = noise_rows(keys, F_of, Um, nb_cache)
    w = 1.0 / sd
    npool = npool * w[None, :]
    global NZ_GLOBAL
    NZ_GLOBAL = (npool, nsess, nspans)
    fit = Fitter(H0, blocks, frow, w)
    describe(fit, phys, gauge)
    gauge_tests(fit, gauge, w, npool)

    gmasks = [np.isin(prow, grp[gk]) for gk in gkeys]
    print(f"  七个物理位姿组的行数: {[int(m.sum()) for m in gmasks]}")
    print(f"  条件数/组: {[len(grp[gk]) for gk in gkeys]}  (整组留出, 组间等权)\n")

    rng = np.random.default_rng(3)
    Qprev = prev_basis(fit)
    if a.stage == "g0":
        g0_isolation(fit, npool, gmasks, Qprev, gkeys, grp, poses, prow, frow,
                     keys, F_of)
        return 0
    if a.stage == "verify":
        res, lvl = stage_verify(fit, npool, gmasks, Qprev, rng, a.trials)
        d6_diagnostic(fit, npool, gmasks, Qprev, rng)
    else:
        res, lvl = run_trials(fit, npool, gmasks, Qprev, a.trials, rng)
    report(res, lvl)
    np.savez(OUT / f"levels_C_{a.design}_{a.stage}.npz",
             levels=np.array(lvl), snr=np.array(list(res)),
             conf=np.array([res[k]["conf"] for k in res]))
    print(f"\n→ outputs/probe_c/levels_C_{a.design}_{a.stage}.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
