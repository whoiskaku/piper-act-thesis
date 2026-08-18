#!/usr/bin/env python3
"""Step 1 预注册的设计分析 —— 判据功效、方差情景、H2 样本量公式（2026-08-19）。

支撑 `reports/PREREG_STEP1.md`。**在采集任何数据之前**跑完并冻结。

三件事：
  ① 旧 A/D 数据能提供什么、**不能**提供什么（见下方"能传递性"一节）
  ② H1 **四项统计条件**的工作特性 —— **Monte Carlo 联合模拟**（不是各项功效取 min）
     ⇒ 这是**完整五项门禁的上界**；条件⑤(成功率/安全性)无法模拟
  ③ H2 后续 confirmatory 的样本量公式（此刻冻结，数值待本批实测 SD 代入）

⚠ 关于旧 A/D 数据（`outputs/ab2/AD_result.json`）的**能传递性**：
  · A 臂 CV —— 仅作**历史规划锚点**（同任务、同裸 ACT 基线），用于事前排 n；
    **不宣称确定可传递** —— C 臂的 CV 未知
  · 臂内 lag-1 自相关 —— **弱可传递**，仅用于说明"episode 尺度的漂移确实存在"
  · A↔D 的指标配对相关 —— **不可传递**。那次是**块跑**（非交替），
    按 index 配对的 episode 在时间上相隔很远，测的根本不是交替设计下的 ρ。
    它既不是 ρ 的估计，**也不是** ρ 的下界 —— 是另一个量。
  · D 臂 CV —— **不可传递**。D 是失败的 π，与新候选结构无关。

用法: python sim/step1_design_analysis.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
# `outputs/` 被 .gitignore 忽略 ⇒ 用版本库内的冻结副本, 干净 checkout 也能复现
# (注意不能放 reports/data/ —— .gitignore 里裸 `data` 会匹配任意层级的 data 目录)
AD = ROOT / "reports/step1_AD_frozen.json"
AD_SHA256 = "63dc56362ef7c72598229efde712114d2d5297fe7951d24ca7517a3cb313632c"
AD_N = 10

# ---- 冻结的判据参数（改动须同步 PREREG_STEP1.md 并留改动记录）----
BAR = 0.15              # H1 实用显著性下限：median(r_CA) ≥ 15%
ALPHA = 0.05            # 双侧
SIGN_K, SIGN_N = 8, 10  # H1 符号条件：≥8/10 为正
N_BLOCK = 10            # H1 事前固定 10 个 matched triplet
N_SESSION = 2           # 分 2 个会话，每会话 5 个 block
POWER = 0.80            # H2 样本量的目标功效
DELTA_H2 = 0.035        # H2 最小有意义效应 3.5 pp（causal_ceiling 的边际估计）
NMAX_H2 = 60            # H2 可接受的最大 n（对）；超过则判定本平台不可行
NSIM = 40000
RNG = np.random.default_rng(20260819)


# ══════════════════════════════════════════════════════════════════════
def load_ad():
    """读冻结输入并校验长度与 sha256 —— 输入变了必须显式改冻结值。"""
    d = json.loads(AD.read_text())
    A, D = [float(x) for x in d["A"]], [float(x) for x in d["D"]]
    payload = json.dumps({"A": A, "D": D}, sort_keys=True, separators=(",", ":")).encode()
    got = hashlib.sha256(payload).hexdigest()
    assert len(A) == len(D) == AD_N, f"长度校验失败: {len(A)}/{len(D)} ≠ {AD_N}"
    assert got == AD_SHA256, f"sha256 校验失败:\n  期望 {AD_SHA256}\n  实得 {got}"
    print(f"  输入校验 ✓  n={AD_N}  sha256={got[:16]}…")
    return np.array(A), np.array(D)


def n_from_t(sd, delta, alpha, power, cap):
    """非中心 t 迭代求 n（单样本双侧）。z 近似会低估 n, 故用精确解。"""
    for n in range(3, cap * 4):
        df, ncp = n - 1, delta / sd * np.sqrt(n)
        crit = stats.t.ppf(1 - alpha / 2, df)
        pw = (1 - stats.nct.cdf(crit, df, ncp)) + stats.nct.cdf(-crit, df, ncp)
        if pw >= power:
            return n
    return None


def cv(x):
    return x.std(ddof=1) / x.mean()


def sd_r_from(cv_a, cv_c, rho):
    """delta 法: r = 1 − L_C/L_A, 小效应下 SD(r) ≈ √(CV_A² + CV_C² − 2ρ·CV_A·CV_C)。"""
    return float(np.sqrt(cv_a ** 2 + cv_c ** 2 - 2 * rho * cv_a * cv_c))


def section_1_old_data():
    A, D = load_ad()
    print("=" * 78)
    print("① 旧 A/D 数据（源 outputs/ab2/AD_result.json, 2026-07-29, **块跑**）")
    print("=" * 78)

    def mad_out(x):
        """离群判定: |x−中位数| > 3 × **标定 MAD**(1.4826×MAD, 正态下 ≈σ) ⇒ 即 3σ 等价。
        与 PREREG_STEP1.md 的措辞必须一致 —— 是**标定 MAD**, 不是裸 MAD。"""
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        return np.abs(x - med) > 3 * 1.4826 * mad

    outA, outD = mad_out(A), mad_out(D)
    print(f"  离群判定(3×标定MAD, 即 3σ 等价): A {outA.sum()} 个, D {outD.sum()} 个"
          f"{' → D ep' + ','.join(map(str, np.where(outD)[0])) if outD.any() else ''}")

    print(f"\n  {'':22}{'n':>4}{'mean':>10}{'SD':>10}{'CV':>9}{'SEM/mean':>10}")
    for nm, x in (("A 裸ACT 全部", A), ("A 裸ACT 剔离群", A[~outA]),
                  ("D  π   全部", D), ("D  π   剔离群", D[~outD])):
        print(f"  {nm:22}{len(x):>4}{x.mean():>10.5f}{x.std(ddof=1):>10.5f}"
              f"{cv(x)*100:>8.2f}%{cv(x)/np.sqrt(len(x))*100:>9.2f}%")
    print("\n  ⇒ **A 臂 CV 仅作「历史规划锚点」** —— 用于事前排 n, 不宣称确定可传递到本设计。")
    print("     D 是失败的 π, 其 CV 与新候选无关。两套(保留/剔除离群)都在上表, 不默认删除。")

    print("\n  A↔D 指标配对相关（**块跑**, 与交替设计的 ρ 不是同一个量, 不可传递）:")
    for nm, keep in (("全部 10 对", np.ones(10, bool)), ("剔 D 离群", ~outD)):
        a, b = A[keep], D[keep]
        r, p = stats.pearsonr(a, b)
        print(f"    {nm:14} ρ={r:+.3f} (p={p:.3f}, n={len(a)})")

    print("\n  臂内 lag-1 自相关（仅用于说明 episode 尺度漂移**存在**）:")
    for nm, x in (("A", A), ("D(剔离群)", D[~outD])):
        r, p = stats.pearsonr(x[:-1], x[1:])
        z, se = np.arctanh(r), 1 / np.sqrt(len(x) - 1 - 3)
        print(f"    {nm:12} ρ={r:+.3f} (p={p:.3f}, n={len(x)-1}), "
              f"95%CI=[{np.tanh(z-1.96*se):+.2f}, {np.tanh(z+1.96*se):+.2f}]  ← CI 极宽")
    return cv(A)


# ══════════════════════════════════════════════════════════════════════
def h1_criterion(r):
    """H1 的**四项统计条件**（①②③④）。r: (nsim, N_BLOCK) 的 r_CA。

    ⚠ 这不是完整的五项门禁 —— 条件⑤（成功率/安全性）取决于真实任务结果，
    **无法模拟**。故本函数给出的是**完整五项门禁的功效上界**：真实功效 ≤ 此值。
    """
    med = np.median(r, axis=1) >= BAR                       # ① 中位数 ≥ 15%
    t = stats.ttest_1samp(r, 0.0, axis=1)                   # ② 单样本 t 检验 on r_i
    sig = (t.pvalue < ALPHA) & (r.mean(axis=1) > 0)
    sign = (r > 0).sum(axis=1) >= SIGN_K                    # ③ ≥8/10 为正
    per = r.reshape(len(r), N_SESSION, N_BLOCK // N_SESSION)
    sess = (np.median(per, axis=2) > 0).all(axis=1)         # ④ 每会话中位数 > 0
    return med & sig & sign & sess


def section_2_h1(cv_a):
    print("\n" + "=" * 78)
    print(f"② H1 工作特性 —— {N_BLOCK} 个 matched triplet")
    print("   **Monte Carlo 联合模拟四项统计条件 ⇒ 完整五项门禁的功效上界**")
    print("=" * 78)
    print(f"  ①median(r_CA)≥{BAR:.0%}  ②单样本t(r_i) p<{ALPHA}  "
          f"③≥{SIGN_K}/{SIGN_N}为正  ④每会话中位数>0")
    print(f"  ⑤成功率/安全性 —— **无法模拟**, 故下表是上界, 真实功效 ≤ 此值")
    print(f"\n  情景模型: **保守恒定 SD** —— r_i ~ N(真实效应, SD), SD 与效应无关。")
    print(f"    (精确的 delta 法含均值比因子 (L_C/L_A)²<1, 会使 SD 略小 ⇒ 本模型偏保守)")
    print(f"  CV_A = CV_C = {cv_a*100:.2f}%（A 臂**历史规划锚点**; C 臂 CV 未知）")

    rhos = [0.0, 0.35, 0.683]
    print(f"\n  {'真实效应':>8} |" + "".join(
        f"  ρ={r:.2f} (SD={sd_r_from(cv_a,cv_a,r)*100:4.1f}%)" for r in rhos))
    print("  " + "-" * 74)
    for true in (0.00, 0.10, 0.15, 0.20, 0.26, 0.33):
        row = f"  {true:8.0%} |"
        for rho in rhos:
            sd = sd_r_from(cv_a, cv_a, rho)
            r = RNG.normal(true, sd, (NSIM, N_BLOCK))
            row += f"{h1_criterion(r).mean():>21.1%}"
        print(row)

    print(f"\n  ⇒ n={N_BLOCK} **事前固定**, 不依赖旧 A/D 的 ρ 点估计。")
    print(f"     但它**不是**对所有 ρ 都有足够功效 —— 上表即为预注册的情景表:")
    for rho in rhos:
        sd = sd_r_from(cv_a, cv_a, rho)
        p20 = h1_criterion(RNG.normal(0.20, sd, (NSIM, N_BLOCK))).mean()
        p15 = h1_criterion(RNG.normal(0.15, sd, (NSIM, N_BLOCK))).mean()
        print(f"       ρ={rho:.2f}: 真实20% → {p20:.0%}   真实15% → {p15:.0%}")
    print(f"     真实效应恰在门槛 {BAR:.0%} 时功效≈50% 是**门槛=效应**的必然结果, 设计已知并接受。")

    print(f"\n  各条件单独的通过率（真实效应 20%, ρ=0.35）—— 看哪一条最卡:")
    sd = sd_r_from(cv_a, cv_a, 0.35)
    r = RNG.normal(0.20, sd, (NSIM, N_BLOCK))
    t = stats.ttest_1samp(r, 0.0, axis=1)
    per = r.reshape(NSIM, N_SESSION, N_BLOCK // N_SESSION)
    rates = []
    for nm, c in (("① median ≥ 15%", np.median(r, 1) >= BAR),
                  ("② t 检验 p<0.05", (t.pvalue < ALPHA) & (r.mean(1) > 0)),
                  (f"③ ≥{SIGN_K}/{SIGN_N} 为正", (r > 0).sum(1) >= SIGN_K),
                  ("④ 每会话中位数>0", (np.median(per, 2) > 0).all(1))):
        rates.append(c.mean())
        print(f"    {nm:20} {c.mean():6.1%}")
    joint = h1_criterion(r).mean()
    print(f"    {'──四项联合(=五项上界)':20} {joint:6.1%}")
    print(f"    对照: 各项取 min = {min(rates):.1%} ≠ 联合 {joint:.1%} "
          f"（各条件不独立, 取 min 只是上界）⇒ 必须联合模拟")


# ══════════════════════════════════════════════════════════════════════
def section_3_h2(cv_a):
    print("\n" + "=" * 78)
    print("③ H2 后续 confirmatory 的样本量 —— **公式此刻冻结, 数值待本批实测 SD 代入**")
    print("=" * 78)
    z = (stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(POWER)) ** 2
    print(f"  求解: 使**单样本双侧 t 检验**达到 power 的最小 n（非中心 t 迭代）")
    print(f"    α={ALPHA} (双侧)   power={POWER:.0%}   "
          f"δ_min={DELTA_H2:.3f} (3.5pp, causal_ceiling 的测量侧边际估计)")
    print(f"    Nmax={NMAX_H2} 对")
    print(f"    (z 近似 n=⌈{z:.3f}·SD²/δ²⌉ 会**低估** —— 一并列出作对照, 判定以 t 为准)")
    print(f"\n  {'SD(r_CB)':>10}{'z 近似':>9}{'非中心 t':>10}   判定")
    print("  " + "-" * 62)
    for sd in (0.06, 0.08, 0.092, 0.117, 0.132, 0.164):
        nz = int(np.ceil(z * sd ** 2 / DELTA_H2 ** 2))
        nt = n_from_t(sd, DELTA_H2, ALPHA, POWER, NMAX_H2)
        ok = nt is not None and nt <= NMAX_H2
        note = f"采用 n={nt}" if ok else f"> Nmax={NMAX_H2} ⇒ **判 confirmatory 不可行**"
        print(f"  {sd*100:9.1f}%{nz:9d}{(nt if nt else -1):10d}   {note}")

    print(f"\n  ⚠ 「不可行」的准确含义（**不得**简写成「本平台 H2 不可行」）:")
    print(f"     「在当前 δ_min={DELTA_H2:.3f}、目标功效 {POWER:.0%} 与 Nmax={NMAX_H2} 的机时预算下,")
    print(f"       H2 的 confirmatory 不可行」 —— 这**不等于**证明 IMU 没有效果。")

    print(f"\n  ⚠ 本批 {N_BLOCK} 个 block 的 H2 **仅为探索性**, 且**完全不并入**后续检验:")
    print("     · 本批完整报告 H2 效应量、配对 CI、方向、跨会话一致性")
    print("     · SD(r_CB)/ρ_BC 标注「可行性/方差估计」, **不作通过门槛**")
    print(f"       (n={N_BLOCK} 时 ρ 的 CI 很宽, 见 ① 中 lag-1 的 CI 宽度)")
    print("     · 后续 H2 confirmatory **只用全新独立数据**, n 由上式代入本批实测 SD")
    print("     · **不设** blinded-SSR 并入分支 —— 「本批既报告又并入」自相矛盾, 已删除")


def main():
    print("Step 1 预注册设计分析  —— 支撑 reports/PREREG_STEP1.md")
    print(f"数据: {AD.relative_to(ROOT)}   随机种子: 20260819   nsim={NSIM}\n")
    cv_a = section_1_old_data()
    section_2_h1(cv_a)
    section_3_h2(cv_a)
    print("\n" + "=" * 78)
    print("冻结声明: 本文件的参数(BAR/ALPHA/SIGN_K/N_BLOCK/POWER/DELTA_H2/NMAX_H2)")
    print("即预注册的判据。改动须同步 PREREG_STEP1.md 并留改动记录与理由。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
