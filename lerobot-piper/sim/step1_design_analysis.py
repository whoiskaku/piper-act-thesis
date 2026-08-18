#!/usr/bin/env python3
"""Step 1 预注册的设计分析 —— 判据功效、方差情景、H2 样本量公式（2026-08-19）。

支撑 `reports/PREREG_STEP1.md`。**在采集任何数据之前**跑完并冻结。

三件事：
  ① 旧 A/D 数据能提供什么、**不能**提供什么（见下方"能传递性"一节）
  ② H1 五项合取判据的工作特性 —— **Monte Carlo 联合模拟**，不是各项功效取 min
  ③ H2 后续 confirmatory 的样本量公式（此刻冻结，数值待本批实测 SD 代入）

⚠ 关于旧 A/D 数据（`outputs/ab2/AD_result.json`）的**能传递性**：
  · A 臂 CV —— **可传递**（同任务、同裸 ACT 基线，新实验的 A 臂还是它）
  · 臂内 lag-1 自相关 —— **弱可传递**，仅用于说明"episode 尺度的漂移确实存在"
  · A↔D 的指标配对相关 —— **不可传递**。那次是**块跑**（非交替），
    按 index 配对的 episode 在时间上相隔很远，测的根本不是交替设计下的 ρ。
    它既不是 ρ 的估计，**也不是** ρ 的下界 —— 是另一个量。
  · D 臂 CV —— **不可传递**。D 是失败的 π，与新候选结构无关。

用法: python sim/step1_design_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
AD = ROOT / "outputs/ab2/AD_result.json"

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
    d = json.loads(AD.read_text())
    return np.array(d["A"]), np.array(d["D"])


def cv(x):
    return x.std(ddof=1) / x.mean()


def sd_r_from(cv_a, cv_c, rho):
    """delta 法: r = 1 − L_C/L_A, 小效应下 SD(r) ≈ √(CV_A² + CV_C² − 2ρ·CV_A·CV_C)。"""
    return float(np.sqrt(cv_a ** 2 + cv_c ** 2 - 2 * rho * cv_a * cv_c))


def section_1_old_data():
    A, D = load_ad()
    print("=" * 78)
    print("① 旧 A/D 数据（outputs/ab2/AD_result.json, 2026-07-29, **块跑**）")
    print("=" * 78)

    # 离群点判定规则与预注册一致: 超出该臂中位数 3×MAD
    def mad_out(x):
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        return np.abs(x - med) > 3 * 1.4826 * mad

    outA, outD = mad_out(A), mad_out(D)
    print(f"  离群判定(中位数 3×MAD): A {outA.sum()} 个, D {outD.sum()} 个"
          f"{' → D ep' + ','.join(map(str, np.where(outD)[0])) if outD.any() else ''}")

    print(f"\n  {'':22}{'n':>4}{'mean':>10}{'SD':>10}{'CV':>9}{'SEM/mean':>10}")
    for nm, x in (("A 裸ACT 全部", A), ("A 裸ACT 剔离群", A[~outA]),
                  ("D  π   全部", D), ("D  π   剔离群", D[~outD])):
        print(f"  {nm:22}{len(x):>4}{x.mean():>10.5f}{x.std(ddof=1):>10.5f}"
              f"{cv(x)*100:>8.2f}%{cv(x)/np.sqrt(len(x))*100:>9.2f}%")
    print("\n  ⇒ **可传递的只有 A 臂 CV**。D 是失败的 π, 其 CV 与新候选无关。")
    print("     两套（保留/剔除离群）都在上表, 不默认删除。")

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
    """H1 五项合取判据。r: (nsim, N_BLOCK) 的 r_CA。

    条件 5（成功率/安全性不下降）**无法模拟**（取决于真实任务结果），
    故本函数给出的是**上界** —— 真实功效 ≤ 此值。
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
    print(f"② H1 工作特性 —— {N_BLOCK} 个 matched triplet, **Monte Carlo 联合模拟**五项合取")
    print("=" * 78)
    print(f"  判据: median(r_CA)≥{BAR:.0%} 且 单样本t(r_i) p<{ALPHA} 且 "
          f"≥{SIGN_K}/{SIGN_N}为正 且 每会话中位数>0")
    print(f"  （条件⑤成功率/安全性无法模拟 ⇒ 下表为**上界**）")
    print(f"\n  假设 CV_A = CV_C = {cv_a*100:.2f}%（旧 A 臂实测; C 臂 CV 未知）")

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
    print(f"    {'──五项合取':20} {joint:6.1%}")
    print(f"    对照: 各项取 min = {min(rates):.1%} ≠ 联合 {joint:.1%} "
          f"（各条件不独立, 取 min 只是上界）⇒ 必须联合模拟")


# ══════════════════════════════════════════════════════════════════════
def section_3_h2(cv_a):
    print("\n" + "=" * 78)
    print("③ H2 后续 confirmatory 的样本量 —— **公式此刻冻结, 数值待本批实测 SD 代入**")
    print("=" * 78)
    z = (stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(POWER)) ** 2
    print(f"  n = ceil( (z_{{1−α/2}} + z_{{1−β}})² · SD(r_CB)² / δ_min² ),  上取整, 封顶 Nmax")
    print(f"    α={ALPHA} (双侧)   power={POWER:.0%}   系数={z:.3f}")
    print(f"    δ_min={DELTA_H2:.3f} (3.5 pp, causal_ceiling 的测量侧边际估计)")
    print(f"    Nmax={NMAX_H2} 对 —— 超过即判定「本平台参数下 H2 不可行」并如实报告")
    print(f"\n  {'SD(r_CB)':>10}{'公式给出 n':>12}   判定")
    print("  " + "-" * 52)
    for sd in (0.06, 0.08, 0.092, 0.117, 0.132, 0.164):
        n = int(np.ceil(z * sd ** 2 / DELTA_H2 ** 2))
        note = f"采用 n={n}" if n <= NMAX_H2 else f"{n} > Nmax={NMAX_H2} ⇒ **判不可行**"
        print(f"  {sd*100:9.1f}%{n:12d}   {note}")
    print(f"\n  ⚠ 本批 {N_BLOCK} 个 block 中 H2 **仅为探索性**:")
    print("     交付 SD(r_CB)、ρ_BC、效应量与 CI —— 标注为「可行性/方差估计」, **不作通过门槛**")
    print(f"     (n={N_BLOCK} 时 ρ 的 CI 很宽, 见 ① 中 lag-1 的 CI 宽度)")
    print("     后续 H2 confirmatory 用**独立**数据, n 由上式代入本批实测 SD 计算")
    print("\n  若要把本批并入未来 H2 检验, 必须走**一次性 blinded SSR**（否则属 optional stopping）:")
    print(f"     · 检查时点: 本批 {N_BLOCK} 个 block 全部完成后, 仅此一次")
    print("     · 只读取 SD(r_CB); **不读**均值、方向、p 值")
    print("     · 公式/α/power/δ_min/Nmax/取整规则 = 上述, 不得改动")


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
