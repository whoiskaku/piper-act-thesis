#!/usr/bin/env python3
"""**约束式 Pareto 筛选** —— 23 个参数还剩多少余量？（不新搜索，只用已有 trial）

## 筛选规则（**只看训练侧**）

```
候选池 = 3 个 seed 的全部已完成 trial
筛选   = 训练损失 L_q ≤ 1.05 × L_q*        (L_q* = 0.030684 ⇒ 阈值 0.032218)
选择   = 其中**训练集 IMU** 最小者
```

**选择过程不看留出 A / 留出 B / pp2。** 训练集 IMU 在搜索时已逐 trial 记录为
`user_attr["imu"]`，本身不参与过选参（目标函数只是 `L_q`）。

## 评估

选中的 θ 与冻结的 θ* **走逐字相同的代码路径**（`g_q_v1.predict_with_theta`
—— 同一限幅器、同一事件网格、同一读数惯例，已验证喂 `THETA_STAR` 时与
`predict_qpos` 逐位相同）。对同一组全局位移 `L ∈ 0..5` 都算、各取自己最优。

> 这是为了排除「用某个模型的评估壳去量另一个模型」的混淆 ——
> 已实际踩过一次，见 `G_Q_V1_FREEZE.md` §9。

## 判读（预先定死）

| 结果 | 处理 |
|---|---|
| IMU 明显下降 **且** 关节性能保持 | 冻结为下一版候选 |
| IMU 只改善几个百分点 | **停止继续调整这 23 个参数** |

> **15–19% 不是已知上限** —— 那只是 phys-v1 的 θ 与 θ* 这**两个具体点**之间的
> 差异。真正的约束 Pareto 上限只有筛完才知道。

用法:
    python sim/step2_g/pareto_screen.py --screen      # 只做训练侧筛选(秒级)
    python sim/step2_g/pareto_screen.py --run         # 筛选 + 三个集上评估
"""
from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g"):
    sys.path.insert(0, str(_p))

TOL = 1.05
COL = ROOT / "outputs/step2_g/collect"
OUT = ROOT / "outputs/step2_g/pareto"


def pool():
    """→ 排序后的 (训练L, 训练IMU, θ, seed, trial号) 列表。"""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from optuna_ident_enc import SEEDS, study_name, storage, theta_of
    out = []
    for sd in SEEDS:
        st = optuna.load_study(study_name=study_name(sd), storage=storage(sd))
        for t in st.trials:
            if t.value is None:
                continue
            im = t.user_attrs.get("imu")
            if im is None:
                continue
            out.append((float(t.value), float(im), theta_of(t), sd, t.number))
    out.sort(key=lambda r: r[0])
    return out


def frontier(P):
    """不同松弛量下，训练集 IMU 的最小值 —— 展示前沿形状（**不用于选参**）。"""
    Lstar = P[0][0]
    rows = []
    for tol in (1.00, 1.01, 1.02, 1.05, 1.10, 1.20, 1.50):
        cand = [r for r in P if r[0] <= Lstar * tol]
        if not cand:
            continue
        b = min(cand, key=lambda r: r[1])
        rows.append((tol, Lstar * tol, len(cand), b[1], b[0], b[3], b[4]))
    return Lstar, rows


def cmd_screen(_a):
    P = pool()
    Lstar, rows = frontier(P)
    thr = Lstar * TOL
    print(f"**约束式 Pareto 筛选（只看训练侧）**\n")
    print(f"  候选池 {len(P)} 个 trial（3 seed 合并）")
    print(f"  L_q* = {Lstar:.6f}   阈值 = {TOL:g} × L_q* = **{thr:.6f}**\n")
    print(f"  前沿形状（**仅供参考，不用于选参**）")
    print(f"  {'松弛':>7s}{'阈值':>11s}{'候选数':>8s}{'最小训练IMU':>13s}{'其训练L':>10s}"
          f"{'来源':>22s}")
    for tol, th, n, imu, lq, sd, tn in rows:
        mark = "  ← 本次选用" if abs(tol - TOL) < 1e-9 else ""
        print(f"  {tol:>7.2f}{th:>11.6f}{n:>8d}{imu:>13.4f}{lq:>10.6f}"
              f"{f'seed {str(sd)[:4]} #{tn}':>22s}{mark}")
    cand = [r for r in P if r[0] <= thr]
    star = P[0]
    sel = min(cand, key=lambda r: r[1])
    print(f"\n  θ*（训练损失最优）: L_q {star[0]:.6f}  训练IMU {star[1]:.4f}"
          f"   seed {str(star[3])[:4]} #{star[4]}")
    print(f"  **选中**（松弛内训练IMU最小）: L_q {sel[0]:.6f}  训练IMU {sel[1]:.4f}"
          f"   seed {str(sel[3])[:4]} #{sel[4]}")
    print(f"  ⇒ 训练侧: 关节 {(sel[0]/star[0]-1)*100:+.1f}%   "
          f"IMU **{(sel[1]/star[1]-1)*100:+.1f}%**")
    if sel[4] == star[4] and sel[3] == star[3]:
        print(f"  ⚠ 选中的就是 θ* 本身 ⇒ **松弛内没有更好的 IMU 可换**")
    return P, star, sel


def _load_suite(names):
    man = json.loads((ROOT / "outputs/step2_g/manifest.json").read_text())
    out = []
    for nm in names:
        d = json.loads((COL / f"replay_{nm}.json").read_text())
        cmd = np.degrees(np.asarray(d["cmd"], float)[:, :6])
        out.append((nm, np.asarray(d["cmd"], float),
                    np.asarray(d["state"], float)[:, :6] / 1000.0,
                    np.asarray(d["imu"], float)[:, 3:6],
                    cmd.max(0) - cmd.min(0), man[nm]["target"]))
    return out


def _load_pp2():
    from sim.step2_gap_baseline import bare_logs
    out = []
    for p in bare_logs():
        fr = json.loads(p.read_text())["frames"]
        out.append((p.stem, np.asarray([f["a"] for f in fr], float),
                    np.asarray([f["state"] for f in fr], float)[:, :6] / 1000.0,
                    np.asarray([f["imu"] for f in fr], float)[:, 3:6], None, None))
    return out


def evaluate_set(theta, data, R, pp2: bool):
    """→ (关节最优值, 关节最优L, IMU最优值, IMU最优L)。全局位移各取自己最优。"""
    from sim.step1_v2_loss import bandpass41
    from g_q_v1 import predict_with_theta
    qs, gs = [], []
    for _, act, _, _, _, _ in data:
        q, g = predict_with_theta(act, theta)
        qs.append(q); gs.append(g)
    bj = (9e9, -1); bi = (9e9, -1)
    for L in range(6):
        pj, num, den = [], 0.0, 0.0
        for (nm, act, enc, real, tv, tg), q, g in zip(data, qs, gs):
            k = min(len(q), len(enc) - L)
            if pp2:
                pj.append(float(np.sqrt(np.mean((q[:k] - enc[L:L + k]) ** 2))))
            else:
                pj.append(float(np.mean([np.sqrt(np.mean((q[:k, j] - enc[L:L + k, j]) ** 2))
                                         / max(tv[j], 1e-9) for j in tg])))
            k2 = min(len(g), len(real) - L)
            a_ = bandpass41(real[L:L + k2])
            num += float(np.square(a_ - bandpass41(g[:k2] @ R.T)).sum())
            den += float(np.square(a_).sum())
        vj, vi = float(np.mean(pj)), num / max(den, 1e-12)
        if vj < bj[0]:
            bj = (vj, L)
        if vi < bi[0]:
            bi = (vi, L)
    return bj + bi


def cmd_run(a):
    gc.disable()
    P, star, sel = cmd_screen(a)
    if sel[4] == star[4] and sel[3] == star[3]:
        print(f"\n{'='*74}\n⇒ **松弛内的最优 IMU 就是 θ* 本身 ⇒ 无余量可换，"
              f"停止调整这 23 个参数。**")
        return 0
    from sim.step1_v2_loss import load_calib
    from sim.step2_fidelity import CALIB
    from optuna_ident_enc import split
    R = load_calib(CALIB)[0]
    tr, ha, hb = split()
    SETS = [("留出A", _load_suite(ha), False), ("留出B", _load_suite(hb), False),
            ("pp2", _load_pp2(), True)]
    print(f"\n{'='*74}\n**评估（同一冻结代码路径，扫 L∈0..5 各取最优）**\n")
    print(f"  {'数据集':8s}{'θ (冻结)':>26s}{'θ_pareto':>26s}")
    print(f"  {'':8s}{'关节':>12s}{'IMU':>14s}{'关节':>12s}{'IMU':>14s}")
    res = {}
    for sn, data, is_pp2 in SETS:
        r0 = evaluate_set(star[2], data, R, is_pp2)
        r1 = evaluate_set(sel[2], data, R, is_pp2)
        res[sn] = (r0, r1)
        print(f"  {sn:8s}{r0[0]:>9.4f}(L{r0[1]}){r0[2]:>11.4f}(L{r0[3]})"
              f"{r1[0]:>9.4f}(L{r1[1]}){r1[2]:>11.4f}(L{r1[3]})", flush=True)
    print(f"\n  {'数据集':8s}{'关节变化':>12s}{'IMU变化':>12s}")
    for sn, (r0, r1) in res.items():
        print(f"  {sn:8s}{(r1[0]/r0[0]-1)*100:>11.1f}%{(r1[2]/r0[2]-1)*100:>11.1f}%")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "screen.json").write_text(json.dumps({
        "L_star": star[0], "tol": TOL, "threshold": star[0] * TOL,
        "theta_star": {"L": star[0], "imu": star[1], "seed": star[3], "trial": star[4]},
        "theta_pareto": {"L": sel[0], "imu": sel[1], "seed": sel[3], "trial": sel[4],
                         "theta": sel[2].tolist()},
        "eval": {k: {"star": list(v[0]), "pareto": list(v[1])} for k, v in res.items()},
    }, ensure_ascii=False, indent=2))
    print(f"\n  → {OUT/'screen.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    gc.disable()
    if a.screen:
        cmd_screen(a); return 0
    if a.run:
        return cmd_run(a)
    print(__doc__.split("用法:")[0])
    return 0


if __name__ == "__main__":
    _c = main()
    sys.stdout.flush()
    import os
    os._exit(_c)
