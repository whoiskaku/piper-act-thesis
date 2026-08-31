#!/usr/bin/env python3
"""**约束式 Pareto 的正式门禁** —— 量化「明显下降」，避免结果出来后临场解释。

## 一、选择（**只用训练集**，不看任何留出 / pp2）

```
候选池 = 3 个 seed 的全部已完成 trial（3026 个）
合格   = L_q^train ≤ 0.03222
选择   = 合格者中**训练 IMU 最小**者
```

**不重跑 Optuna。** 只有现有近优 trial 中候选不足时，才考虑**一次固定预算**的
新 Pareto 搜索；**不得因为结果接近而继续追加**。

## 二、评价（选择完成后**一次性**，四条**全部**满足才算通过）

| # | 条件 |
|---|---|
| A | 留出 A、留出 B 的**关节误差**均**不比 G-q-v1 差超过 5%** |
| B | 留出 A、留出 B 的 **IMU** 均**至少改善 5%** |
| C | pp2 的 IMU **逐集至少 9/17 改善**，**且逐集中位改善为正** |
| D | pp2 只作**域外开发评价**，**不称最终密封 holdout** |

**全局位移**：关节固定 `L=0`、IMU 固定 `L=1`，**两个模型用同一位移**
（两者各自的最优位移本就一致，见 `RESULT_OPTUNA_IDENT_ENC.md` §17.2）
⇒ 逐集计数有明确定义，且不偏向任一方。

**口径**：IMU 同时报**等权**（逐条平均）与**能量加权**两种，
门槛在两种口径下都判，不通过就是不通过（不得事后择优）。

## 三、判定

| 结果 | 处理 |
|---|---|
| 四条全过 | 冻结为新的关节—IMU 折中候选 **G-qimu-v1** |
| 任一不过 | **停止 23 参数调优**，记录「**常系数伺服参数层已到顶**」，进入新增模型结构比较 |

结构阶段（`+ 传动/连杆柔性` vs `+ IMU 支架二阶模态`）**在本门禁出结论之后**才启动，
且都以**固定的 G-q-v1** 作底座、只拟合新增参数
⇒ 每次只回答一个问题，参数取值与模型结构不混在同一个结论里。

用法:
    python sim/step2_g/pareto_gate.py --run
"""
from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g"):
    sys.path.insert(0, str(_p))

THRESH = 0.03222          # L_q^train 的合格线（预先定死）
L_JOINT, L_IMU = 0, 1     # 全局位移，两个模型相同
OUT = ROOT / "outputs/step2_g/pareto"


def select():
    """→ (θ*, θ_pareto, 诊断)。**只用训练侧信息。**"""
    from pareto_screen import pool
    P = pool()
    star = P[0]
    cand = [r for r in P if r[0] <= THRESH]
    sel = min(cand, key=lambda r: r[1])
    return star, sel, {"n_pool": len(P), "n_qualified": len(cand)}


def per_ep(theta, data, R, pp2: bool):
    """→ (逐集关节, 逐集IMU等权, 总IMU能量加权)。位移固定，不逐集搜索。"""
    from sim.step1_v2_loss import bandpass41
    from g_q_v1 import predict_with_theta
    pj, pi, num, den = [], [], 0.0, 0.0
    for nm, act, enc, real, tv, tg in data:
        q, g = predict_with_theta(act, theta)
        k = min(len(q), len(enc) - L_JOINT)
        if pp2:
            pj.append(float(np.sqrt(np.mean((q[:k] - enc[L_JOINT:L_JOINT + k]) ** 2))))
        else:
            pj.append(float(np.mean([np.sqrt(np.mean(
                (q[:k, j] - enc[L_JOINT:L_JOINT + k, j]) ** 2)) / max(tv[j], 1e-9)
                for j in tg])))
        k2 = min(len(g), len(real) - L_IMU)
        a_ = bandpass41(real[L_IMU:L_IMU + k2])
        b_ = bandpass41(g[:k2] @ R.T)
        e, t = float(np.square(a_ - b_).sum()), float(np.square(a_).sum())
        pi.append(e / max(t, 1e-12))
        num += e
        den += t
    return np.array(pj), np.array(pi), num / max(den, 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    gc.disable()
    if not a.run:
        print(__doc__.split("用法:")[0])
        return 0

    from sim.step1_v2_loss import load_calib
    from sim.step2_fidelity import CALIB
    from optuna_ident_enc import split
    from pareto_screen import _load_suite, _load_pp2
    R = load_calib(CALIB)[0]
    star, sel, dg = select()

    print("**约束式 Pareto 正式门禁**\n")
    print(f"  候选池 {dg['n_pool']} 个 trial · 合格线 L_q^train ≤ {THRESH:g}"
          f" ⇒ 合格 {dg['n_qualified']} 个")
    print(f"  θ* (G-q-v1)  : L_q {star[0]:.6f}  训练IMU {star[1]:.4f}"
          f"   seed {str(star[3])[:4]} #{star[4]}")
    print(f"  θ_pareto     : L_q {sel[0]:.6f}  训练IMU {sel[1]:.4f}"
          f"   seed {str(sel[3])[:4]} #{sel[4]}")
    print(f"  **选择只用训练集，未看留出 A/B 与 pp2**")
    print(f"  全局位移固定：关节 L={L_JOINT}、IMU L={L_IMU}（两模型相同）\n")

    tr, ha, hb = split()
    SETS = [("留出A", _load_suite(ha), False), ("留出B", _load_suite(hb), False),
            ("pp2", _load_pp2(), True)]
    res = {}
    print(f"  {'数据集':7s}{'关节 θ*':>11s}{'关节 θp':>11s}{'Δ关节':>9s}"
          f"{'IMU等权 θ*':>12s}{'IMU等权 θp':>12s}{'Δ等权':>9s}"
          f"{'IMU能量 θ*':>12s}{'IMU能量 θp':>12s}{'Δ能量':>9s}")
    for sn, data, is_pp2 in SETS:
        j0, i0, e0 = per_ep(star[2], data, R, is_pp2)
        j1, i1, e1 = per_ep(sel[2], data, R, is_pp2)
        res[sn] = (j0, i0, e0, j1, i1, e1)
        print(f"  {sn:7s}{j0.mean():>11.4f}{j1.mean():>11.4f}"
              f"{(j1.mean()/j0.mean()-1)*100:>8.1f}%"
              f"{i0.mean():>12.4f}{i1.mean():>12.4f}{(i1.mean()/i0.mean()-1)*100:>8.1f}%"
              f"{e0:>12.4f}{e1:>12.4f}{(e1/e0-1)*100:>8.1f}%", flush=True)

    print(f"\n{'='*84}\n**门禁判定**（四条全部满足才通过）\n")
    ok = {}
    # A: 留出 A/B 关节不差超过 5%
    detA = []
    for sn in ("留出A", "留出B"):
        j0, _, _, j1, _, _ = res[sn]
        d = j1.mean() / j0.mean() - 1
        detA.append((sn, d, d <= 0.05))
    ok["A"] = all(x[2] for x in detA)
    print(f"  A 留出 A/B 关节误差不差超过 5%: " +
          " · ".join(f"{s} {d*100:+.1f}% {'✓' if g else '**✗**'}" for s, d, g in detA)
          + f"   ⇒ {'✓' if ok['A'] else '**✗**'}")
    # B: 留出 A/B IMU 至少改善 5%（两种口径都判）
    detB = []
    for sn in ("留出A", "留出B"):
        _, i0, e0, _, i1, e1 = res[sn]
        de, dn = i1.mean() / i0.mean() - 1, e1 / e0 - 1
        detB.append((sn, de, dn, de <= -0.05 and dn <= -0.05))
    ok["B"] = all(x[3] for x in detB)
    print(f"  B 留出 A/B IMU 至少改善 5%（等权与能量加权都要）: " +
          " · ".join(f"{s} 等权 {de*100:+.1f}% / 能量 {dn*100:+.1f}% {'✓' if g else '**✗**'}"
                     for s, de, dn, g in detB)
          + f"   ⇒ {'✓' if ok['B'] else '**✗**'}")
    # C: pp2 逐集 ≥9/17 改善且中位改善为正
    _, i0, _, _, i1, _ = res["pp2"]
    d = i0 - i1                                     # >0 = 改善
    nwin, n = int((d > 0).sum()), len(d)
    med = float(np.median(d / i0)) * 100
    ok["C"] = nwin >= 9 and med > 0
    print(f"  C pp2 IMU 逐集 ≥9/{n} 改善 且 逐集中位改善为正: "
          f"**{nwin}/{n}**，中位 **{med:+.1f}%**   ⇒ {'✓' if ok['C'] else '**✗**'}")
    print(f"  D pp2 只作**域外开发评价**，不称最终密封 holdout   ⇒ ✓（措辞约束，已遵守）")
    ok["D"] = True

    passed = all(ok.values())
    print(f"\n{'='*84}")
    if passed:
        print("⇒ **四条全过 ⇒ 冻结为新的关节—IMU 折中候选 G-qimu-v1**")
    else:
        bad = [k for k, v in ok.items() if not v]
        print(f"⇒ **第 {', '.join(bad)} 条不通过 ⇒ 停止 23 参数调优**")
        print("   记录：**常系数伺服参数层已到顶**；进入新增模型结构比较")
        print("   （+ 传动/连杆柔性  vs  + IMU 支架二阶模态，均以固定 G-q-v1 为底座）")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate.json").write_text(json.dumps({
        "threshold": THRESH, "L_joint": L_JOINT, "L_imu": L_IMU,
        "n_pool": dg["n_pool"], "n_qualified": dg["n_qualified"],
        "theta_star": {"L": star[0], "imu_train": star[1], "seed": star[3], "trial": star[4]},
        "theta_pareto": {"L": sel[0], "imu_train": sel[1], "seed": sel[3],
                         "trial": sel[4], "theta": sel[2].tolist()},
        "sets": {k: {"joint_star": v[0].tolist(), "imu_star": v[1].tolist(),
                     "imu_energy_star": v[2], "joint_pareto": v[3].tolist(),
                     "imu_pareto": v[4].tolist(), "imu_energy_pareto": v[5]}
                 for k, v in res.items()},
        "gate": ok, "passed": bool(passed),
        "pp2_wins": nwin, "pp2_n": n, "pp2_median_pct": med,
    }, ensure_ascii=False, indent=2))
    print(f"\n  → {OUT/'gate.json'}")
    return 0


if __name__ == "__main__":
    _c = main()
    sys.stdout.flush()
    import os
    os._exit(_c)
