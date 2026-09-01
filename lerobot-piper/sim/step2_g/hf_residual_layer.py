#!/usr/bin/env python3
"""**有效动态残差层**（固定预算，不用神经网络）。

```
ω̂ = ω̂_Gq + (H_real − H_Gq) * u_hf
```

`u_hf` = 指令中被 G-q-v1 的固件限幅器**平滑掉**的高频增量 = `cmd − ref`。

## 为什么这样做

- **每关节自己的幅值与相位**（ΔH 是 (16 lag, 3 轴, 6 关节) 的 FIR），
  不再要求跨关节共享标量 —— 那正是 B′ 判否的原因（跨关节 4.9×）。
- **`H_real` 已经过跨 seed（β 中位 0.9546）与幅值线性（比值 1.0134）验证**
  （`STEP1_RESULT_v1.md`，`outputs/ab4/` 23 集）。
- **能表达振铃**，不需要继续猜弹簧 / 支架 / 接触参数
  （那三条都已量化收口：§21、`PREREG_STEP2_STRUCTURE` §8）。
- **不改变 G-q-v1 的关节轨迹** —— 只加在陀螺输出上。

## 口径约束（**必须遵守**）

**这是「有效传感器动态补充」，不声称是真实物理模态。**
`ΔH` 是「真机比 G-q-v1 多出来的那部分线性响应」的经验 FIR，
**不对应任何已辨识的机械结构**。

## 已知风险（写在跑之前）

`H` 的幅值线性只在 **1.5–3.0 mrad** 上验证过（ab4 的 P3）。
若 `u_hf` 的量级远超此范围，本层就是在**外推**。
运行时会打印 `u_hf` 的实测量级与该范围的比值 —— **结果必须带这个比值一起读。**

## 对齐（推导，不搜索）

- `H` 的索引惯例：`fit_H` 用 `gap = real[t+ALIGN_L] − ideal[t]`（`ALIGN_L=3`）
  ⇒ `causal_conv(H, u)[t]` 预测的是 **`real[t+3]`** 的贡献。
- pp2 评估惯例：`g[t]` 对 `real[t+L]`（`L=1`）。
- ⇒ 相加时 `ΔH` 的输出要移 **`ALIGN_L − L = 2`** 帧：`g[t] + conv[t−2]`。

**该移位由惯例推出，不作搜索。** 会同时打印「conv 与实际残差的相关 vs 移位」
作为**验证**：若峰值不在推导值上，如实报告，不据此改值。

## 门禁（预注册，跑前定死；**不追加搜索预算**）

| # | 条件 | 阈值 |
|---|---|---|
| 1 | pp2 总体 r/m 改善 | **≥ 5%** |
| 2 | pp2 逐集改善数 | **≥ 9/17** |
| 3 | pp2 逐集中位改善 | **> 0** |
| 4 | 套件留出 A、留出 B 不恶化 | **≤ +5%** |

**若本层仍失败** ⇒ 说明现有数据不足以构造高保真 pp2 的 G；
届时才进入「多样化任务数据 + 学习型高频残差」，**不再试物理小补丁**。

改完 G 必出 pp2 波形对比图（常规）。

用法:
    python sim/step2_g/hf_residual_layer.py --run
"""
from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g"):
    sys.path.insert(0, str(_p))

FPS = 20.0
RIDGE = 1.0                 # 与 fig_gq_vs_H 相同；H_real / H_Gq 用同一值
L_EVAL = 1                  # pp2/套件评估的全局位移
OUT = ROOT / "outputs/step2_g/hf_layer"
CALIB = ROOT / "outputs/p2_labels/pp2a-deploy/manifest.json"


def fit_dH():
    """从 ab4 的 P1（8 集）拟合 H_real 与 H_Gq，返回 ΔH 与两者。"""
    import copy
    from fit_step1_candidate import load_episode, fit_H
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    import g_q_v1 as GQ
    cal = json.loads(CALIB.read_text())
    R = np.asarray(cal["R"], float)
    b = np.asarray(cal.get("b", [0.0, 0.0, 0.0]), float)
    rend = IdealChunkRenderer()
    paths = sorted((ROOT / "outputs/ab4").glob("P1_ep*.json"))
    p1 = [load_episode(p, "P1", R, b, rend, 400) for p in paths]
    H_real, lam = fit_H(p1, RIDGE)
    p1g = []
    for p, e in zip(paths, p1):
        fr = json.loads(p.read_text())["frames"]
        a = np.asarray([x["a"] for x in fr], float)
        da = np.asarray([x["da"] for x in fr], float)
        n = min(len(a), len(e.da))
        a0 = a[:n].copy(); a0[:, :6] -= da[:n]
        g1 = GQ.predict_gyro(a[:n]); g0 = GQ.predict_gyro(a0)
        k = min(len(g1), len(g0), len(e.gap))
        gq = np.zeros_like(e.gap); gq[:k] = (g1[:k] - g0[:k]) @ R.T
        eg = copy.copy(e); eg.gap = gq
        p1g.append(eg)
    H_gq, _ = fit_H(p1g, RIDGE)
    return H_real - H_gq, H_real, H_gq, R, lam


def u_hf(act):
    """指令中被限幅器平滑掉的高频增量（弧度，(T,6)）。"""
    from sim.step2_fidelity import A_MAX
    from optuna_ident_enc import accel_limited_dt
    n = len(act)
    cmd = np.degrees(act[:, :6])
    ref = np.empty((n, 6))
    for j in range(6):
        ref[:, j] = accel_limited_dt(cmd[:, j], float(A_MAX[j]), np.full(n, 1 / FPS))
    return np.radians(cmd - ref)


def predict(act, dH, R, shift: int):
    """ω̂ = ω̂_Gq + ΔH * u_hf（IMU 系, 20 Hz）。**不改关节轨迹。**"""
    from fit_step1_candidate import causal_conv
    import g_q_v1 as GQ
    g = GQ.predict_gyro(act) @ R.T                       # 先旋到 IMU 系
    c = causal_conv(dH, u_hf(act))                       # (T,3)，索引对 real[t+ALIGN_L]
    n = min(len(g), len(c))
    out = g[:n].copy()
    if shift > 0:
        out[shift:n] += c[:n - shift]                    # conv[t-shift]
    elif shift < 0:
        out[:n + shift] += c[-shift:n]
    else:
        out += c[:n]
    return out


def sets():
    from sim.step2_gap_baseline import bare_logs
    from optuna_ident_enc import split
    man = json.loads((ROOT / "outputs/step2_g/manifest.json").read_text())
    COL = ROOT / "outputs/step2_g/collect"
    _, ha, hb = split()
    out = {}
    out["pp2"] = [{"name": p.stem,
                   "act": np.asarray([f["a"] for f in json.loads(p.read_text())["frames"]], float),
                   "real": np.asarray([f["imu"] for f in json.loads(p.read_text())["frames"]],
                                      float)[:, 3:6]} for p in bare_logs()]
    for k, names in (("留出A", ha), ("留出B", hb)):
        out[k] = []
        for nm in names:
            d = json.loads((COL / f"replay_{nm}.json").read_text())
            out[k].append({"name": nm, "act": np.asarray(d["cmd"], float),
                           "real": np.asarray(d["imu"], float)[:, 3:6]})
    return out


def score(data, dH, R, shift):
    from sim.step1_v2_loss import bandpass41
    import g_q_v1 as GQ
    r0, r1 = [], []
    for e in data:
        g0 = GQ.predict_gyro(e["act"]) @ R.T
        g1 = predict(e["act"], dH, R, shift)
        k = min(len(g0), len(g1), len(e["real"]) - L_EVAL)
        a_ = bandpass41(e["real"][L_EVAL:L_EVAL + k])
        t_ = max(float(np.square(a_).sum()), 1e-30)
        r0.append(float(np.square(a_ - bandpass41(g0[:k])).sum()) / t_)
        r1.append(float(np.square(a_ - bandpass41(g1[:k])).sum()) / t_)
    return np.array(r0), np.array(r1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    gc.disable()
    if not a.run:
        print(__doc__.split("用法:")[0]); return 0
    from sim.step1_v2_loss import ALIGN_L, bandpass41
    from fit_step1_candidate import causal_conv
    import g_q_v1 as GQ

    print("**有效动态残差层** —— 固定预算，不用神经网络\n")
    dH, H_real, H_gq, R, lam = fit_dH()
    print(f"  ΔH = H_real − H_Gq  {dH.shape}  (ab4 P1 8 集, ridge={RIDGE}, λ={lam:.4g})")
    S = sets()

    # ── 风险量化: u_hf 的幅值 vs H 验证过的 1.5–3.0 mrad ──
    mags = []
    for e in S["pp2"]:
        u = u_hf(e["act"])
        mags.append([float(np.sqrt(np.mean(u[:, j] ** 2))) * 1000 for j in range(6)])
    mg = np.array(mags)
    print(f"\n  **风险量化**: u_hf 的 RMS（mrad，pp2 17 集均值）")
    print("    " + "  ".join(f"j{j+1} {mg[:, j].mean():.2f}" for j in range(6)))
    print(f"    H 的幅值线性验证范围 = **1.5–3.0 mrad**"
          f"   ⇒ 外推倍数 中位 **{np.median(mg.mean(0))/3.0:.1f}×**（相对 3.0）")

    # ── 对齐验证（不搜索, 只报告）──
    shift = ALIGN_L - L_EVAL
    print(f"\n  对齐: ALIGN_L({ALIGN_L}) − L_eval({L_EVAL}) = **{shift}** 帧（推导值）")
    e0 = S["pp2"][0]
    g0 = GQ.predict_gyro(e0["act"]) @ R.T
    c = causal_conv(dH, u_hf(e0["act"]))
    k = min(len(g0), len(c), len(e0["real"]) - L_EVAL)
    resid = bandpass41(e0["real"][L_EVAL:L_EVAL + k]) - bandpass41(g0[:k])
    cs = []
    for s in range(0, 6):
        cc = np.zeros((k, 3))
        if s == 0: cc = c[:k]
        else: cc[s:] = c[:k - s]
        cb = bandpass41(cc)
        cs.append(float(np.corrcoef(resid.ravel(), cb.ravel())[0, 1]))
    print("    验证 conv 与实际残差的相关 vs 移位: " +
          "  ".join(f"{s}:{v:+.3f}" for s, v in enumerate(cs)) +
          f"   峰值在 {int(np.argmax(cs))}"
          f" {'✓ 与推导一致' if int(np.argmax(cs)) == shift else '**⚠ 与推导不一致（如实报告，不据此改值）**'}")

    # ── 评估 ──
    print(f"\n  {'数据集':8s}{'r/m 无残差层':>13s}{'r/m 有残差层':>13s}{'变化':>9s}"
          f"{'逐集改善':>10s}{'逐集中位改善':>14s}")
    res = {}
    for k_ in ("pp2", "留出A", "留出B"):
        r0, r1 = score(S[k_], dH, R, shift)
        d = (r0 - r1) / r0
        res[k_] = (float(r0.mean()), float(r1.mean()), int((d > 0).sum()), len(d),
                   float(np.median(d)))
        print(f"  {k_:8s}{r0.mean():>13.4f}{r1.mean():>13.4f}"
              f"{(r1.mean()/r0.mean()-1)*100:>8.1f}%{res[k_][2]:>7d}/{res[k_][3]:<3d}"
              f"{np.median(d)*100:>13.1f}%", flush=True)

    p0, p1, pw, pn, pm = res["pp2"]
    ok1 = p1 <= p0 * 0.95
    ok2 = pw >= 9
    ok3 = pm > 0
    ok4 = all(res[k_][1] <= res[k_][0] * 1.05 for k_ in ("留出A", "留出B"))
    print(f"\n{'='*76}\n**门禁**（预注册）")
    print(f"  1 pp2 总体改善 ≥5%      {(1-p1/p0)*100:+.1f}%   {'✓' if ok1 else '**✗**'}")
    print(f"  2 pp2 逐集改善 ≥9/17    {pw}/{pn}      {'✓' if ok2 else '**✗**'}")
    print(f"  3 pp2 逐集中位改善 >0   {pm*100:+.1f}%   {'✓' if ok3 else '**✗**'}")
    print(f"  4 留出 A/B 不恶化 >5%   "
          f"A {(res['留出A'][1]/res['留出A'][0]-1)*100:+.1f}% · "
          f"B {(res['留出B'][1]/res['留出B'][0]-1)*100:+.1f}%   {'✓' if ok4 else '**✗**'}")
    good = ok1 and ok2 and ok3 and ok4
    print(f"\n⇒ " + ("**通过** —— 冻结为高频残差层候选" if good else
          "**未通过** ⇒ 现有数据不足以构造高保真 pp2 的 G；"
          "\n   按预注册进入「多样化任务数据 + 学习型高频残差」，**不再试物理小补丁**。"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate.json").write_text(json.dumps(
        {"shift": shift, "corr_vs_shift": cs, "u_hf_rms_mrad": mg.mean(0).tolist(),
         "res": {k_: list(v) for k_, v in res.items()}, "passed": bool(good)},
        ensure_ascii=False, indent=2))
    print(f"\n  → {OUT/'gate.json'}")
    return 0


if __name__ == "__main__":
    _c = main()
    sys.stdout.flush()
    import os
    os._exit(_c)
