#!/usr/bin/env python3
"""**缺的那一格**：完美预测（真实 e₀）+ 严格因果 + 部署约束，δa 通道到底能交付多少？

为什么必须算这个
----------------
`sim/oracle_causal_feedback.py` 测到：严格因果 + chunk 钳位 + 接触门控 + ε 下，
**指令预测** 3.88% → **指令+实时 IMU 反馈** 4.27%（部署合法档）。
但那时**瓶颈没有分离**：到底是

  (a) **预测器不行**（`W` 从指令预测 gap 的可预测性只有 31.6%，反馈 R²=0.018），还是
  (b) **约束卡死**（chunk 末端钳位 / 幅值上界 / 接触窗归零）？

若是 (a)，一个更强的 π 有发挥空间；若是 (b)，**再好的 π 也到不了门槛，训了白训**。

本脚本把预测器换成**完美的**（直接用真实 e₀ 的未来值），**只保留部署约束**，
并把约束**逐项剥开**，得到归因表。

⚠ 这是**诊断上界，不是可部署策略**
------------------------------------
用到了 t+k 的真实 e₀，**不可**用来生成训练标签
（用户 2026-09-02：「不能用整集未来的 e₀ 解一个非因果最优解，再让因果 π 模仿」）。
它只回答「约束留了多少空间」。`STEP1_RESULT_v1.md` §2.2 的 99.8%
是**非因果且无约束**的容量上界；本表补的是中间那一格。

⚠ 被控对象用**已验证的 H**（跨 seed β 0.9546 / 幅值线性 1.0134），
与 4.27% 那次同一口径，可直接比较。它同时是**任何**因果 δa 策略在该信息集下的上界 ——
用 `g2-incr-v1` 训出的 π 只能逼近、不能超过。

用法:
    python -u sim/oracle_constraint_ceiling.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "sim/step4_gvnext"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import oracle_causal_feedback as O                                   # noqa: E402
import fit_step1_candidate as S1                                     # noqa: E402


def roll(e, H, eps, k_mult, clamp=True, contact=True, oracle=True, W=None):
    """严格因果滚动解。`oracle=True` ⇒ horizon 用**真实 e₀**（诊断上界）。"""
    T = len(e["e0"])
    M1 = S1.N_LAGS
    hs_full = H.reshape(M1 * 3, 6)
    lam = k_mult * max(float(np.trace(hs_full.T @ hs_full)) / 6.0, np.finfo(float).tiny)
    da = np.zeros((T, 6))
    resp = np.zeros((T + M1, 3))
    e0 = e["e0"]
    ff = e["x"] @ W.T if W is not None else None
    nsat = 0
    for t in range(T):
        K = int(min(S1.M, e["cap"][t] - t)) if clamp else S1.M
        K = max(K, 0)
        tgt = np.empty((K + 1, 3))
        for k in range(K + 1):
            j = min(t + k, T - 1)
            tgt[k] = (e0[j] if oracle else ff[j]) + resp[t + k]
        hs = H[:K + 1].reshape((K + 1) * 3, 6)
        d = -np.linalg.solve(hs.T @ hs + lam * np.eye(6), hs.T @ tgt.reshape(-1))
        if eps is not None:
            nsat += int(np.any(np.abs(d) > eps))
            d = np.clip(d, -eps, eps)
        if contact and e["contact"][t]:
            d[:] = 0.0
        da[t] = d
        for k in range(M1):
            resp[t + k] += H[k] @ d
    return da, nsat / max(T, 1)


def score(E, H, das):
    num = den = 0.0
    rms, sat = [], []
    for e, (da, sp) in zip(E, das):
        base = S1.bandpass41(e["e0"])
        new = S1.bandpass41(e["e0"] + S1.causal_conv(H, da))
        num += float(np.square(new).sum()); den += float(np.square(base).sum())
        rms.append(float(np.sqrt(np.mean(np.square(da)))) * 1e3)
        sat.append(sp * 100)
    return 1 - num / max(den, 1e-30), float(np.mean(rms)), float(np.mean(sat))


def main() -> int:
    import gc
    gc.disable()
    H, E = O.load_all()
    print(f"ab4 {len(E)} 集（P0 5 + P1 8）· 被控对象 = 已验证的 H · "
          f"评价带 {S1.METRIC_BAND[0]}–{S1.METRIC_BAND[1]} Hz\n")
    print("  **完美预测**（horizon 用真实 e₀）+ 约束逐项剥开 —— 诊断上界，非可部署策略\n")
    print(f"  {'#':>3}  {'配置':<44}{'降幅':>9}{'δa RMS':>10}{'触界%':>8}")
    rows = []
    cfgs = [
        ("因果滚动 · 无 chunk 钳位 · 无接触 · 无限幅", dict(clamp=False, contact=False, eps=None)),
        ("因果滚动 · **chunk 钳位** · 无接触 · 无限幅", dict(clamp=True, contact=False, eps=None)),
        ("因果 · chunk 钳位 · **接触门控** · 无限幅", dict(clamp=True, contact=True, eps=None)),
        ("因果 · chunk 钳位 · 接触 · **±3 mrad**", dict(clamp=True, contact=True, eps=0.003)),
        ("因果 · chunk 钳位 · 接触 · ±2 mrad（Step-1 内部限幅）", dict(clamp=True, contact=True, eps=0.002)),
        ("因果 · chunk 钳位 · 接触 · **±10 mrad（认证 ε，部署档）**",
         dict(clamp=True, contact=True, eps=0.010)),
    ]
    # ⚠ **之前的错误**：把 `fit_step1_candidate.py` 的 `B_LIMIT=2mrad`/`C_LIMIT=3mrad`
    # 当成部署约束。那是 **Step-1 冻结候选自己的内部限幅**，不是部署的认证 ε=10 mrad。
    # 用错的 ε 直接导致了「4.27% ⇒ 立即停止」那个结论。±2/±3 mrad 的触界率
    # 分别是 29.6% / 12.4%，**本来就违反 sat_pct ≤ 1%**，不是可部署档。
    from train_rq import KFOLD
    fits = []
    for i_ in range(len(E)):
        tr = [E[j] for j in range(len(E)) if j != i_]
        W_ = O.fit_w(tr); fits.append((W_, O.fit_fb(tr, W_)))
    for i, (name, kw) in enumerate(cfgs, 1):
        das = [roll(e, H, kw["eps"], 1.0, clamp=kw["clamp"], contact=kw["contact"]) for e in E]
        r, rm, sa = score(E, H, das)
        rows.append({"cfg": name, **kw, "reduction": r, "da_rms_mrad": rm, "sat_pct": sa})
        print(f"  {i:>3}  {name:<44}{r*100:>8.2f}%{rm:>10.2f}{sa:>8.1f}")

    print(f"\n  {'':>3}  {'对照（已测，预测器受限）':<44}{'降幅':>9}")
    print(f"  {'':>3}  {'指令预测 · 同部署约束 ±3mrad':<44}{3.88:>8.2f}%")
    print(f"  {'':>3}  {'指令+实时 IMU 反馈 · 同约束 ±3mrad':<44}{4.27:>8.2f}%")
    print(f"  {'':>3}  {'非因果 · 无约束（STEP1 §2.2）':<44}{99.8:>8.2f}%")

    dep = rows[5]["reduction"] * 100          # **±10mrad 认证 ε**，完美预测
    print(f"\n  ── 归因 ──")
    das = [O.roll(e, H, W_, Cs_, 0.010, 1.0) for e, (W_, Cs_) in zip(E, fits)]
    r_pred = score(E, H, das)[0] * 100
    print(f"  完美预测 + 全部部署约束（**±10mrad 认证 ε**）= **{dep:.2f}%**（触界 "
          f"{rows[5]['sat_pct'] if 'sat_pct' in rows[5] else 0:.1f}%）")
    print(f"  同约束下指令+反馈预测（逐集留一折外）        = **{r_pred:.2f}%**")
    print(f"  ⇒ **预测器还能贡献的空间 = {dep - r_pred:+.2f} pp**")
    print(f"\n  判读（用户 2026-09-02 定：离线需 ≈20% 才能在真机拿到 ≈15%）：")
    if dep < 6:
        print(f"  **约束卡死** —— 完美预测也只有 {dep:.2f}%，远低于 20%。")
        print(f"  ⇒ 无论 π 多强都到不了门槛；**训练 π 不会改变结论**。")
    elif dep < 20:
        print(f"  完美预测 {dep:.2f}% 仍低于 20% ⇒ 门槛结构上不可达，但预测器确有 "
              f"{dep-4.27:.2f} pp 空间。")
    else:
        print(f"  **瓶颈是预测器** —— 完美预测有 {dep:.2f}% ≥ 20% ⇒ 更强的 π 有发挥空间。")
    OUT = ROOT / "outputs/step1_H/constraint_ceiling.json"
    OUT.write_text(json.dumps(
        {"rows": rows,
         "reference": {"cmd_only_3mrad": 3.88, "cmd_plus_feedback_3mrad": 4.27,
                       "noncausal_unconstrained_step1": 99.8},
         "deployment_cell_3mrad_perfect_prediction": dep,
         "caliber": ("被控对象=已验证 H; 完美预测=horizon 用真实 e0(**诊断上界, 非可部署**, "
                     "不得用于生成训练标签); 评价带 0.5-9Hz; ab4 13 集; "
                     "H 同时是任何因果 δa 策略在该信息集下的上界"),
         }, indent=1, ensure_ascii=False))
    print(f"\n  → {OUT}")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
