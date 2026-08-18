#!/usr/bin/env python3
"""Step 1 的门：**存不存在**一个 δa 能缩小 gap？（2026-08-04，按指导教員的 Step 1）

指導教員 2026-08-04 面談的白板：

    Step 1  G無し
    実世界の動きとの差を消す
      実世界の実測＝IMU値を元に、Simulation と IMU 値の差が無くなるように LR などを追加する

他设的门是「**先证明补正确实能缩小 gap**」。我们已经上机失败两次，
第三次之前应先离线回答更基本的问题：**存不存在这样的 δa？**
连"存在"都证不出来的话，任何学习方法都找不到它。

---

**做法**：不建被控对象模型 —— 用 D/E/F 三条臂**实测**的响应 H
（`measure_da_response.py` 的岭回归冲激响应，纯数据，无物理假设）。

  ① 在训练集上拟合 H
  ② 在**留出集**上还原"无补正"的基线：base = gap − H⊛δa_applied
  ③ 求最优 δa：min ‖base + H⊛δa‖² + λ‖δa‖²，频域逐频率闭式解
  ④ 扫 λ ⇒ (所需 ‖δa‖, 可达降幅) 的 Pareto 曲线

**这是上界，不是可达值**，三条都要记住：
  · **非因果** —— 频域解用到全序列（含未来），真机上拿不到
  · H 的泛化性差（跨版本 D→E 仅 7.6%），所以留出评估是必须的
  · 循环卷积近似边界

**怎么读**：
  · 若在 ε=10 mrad 内降幅就很可观 ⇒ 动作空间够用，问题在"怎么求解 δa"
  · 若要很大的 ‖δa‖ 才有降幅 ⇒ 动作空间（ε）是瓶颈，须放开界
  · 若无论多大 ‖δa‖ 都降不下来 ⇒ **误差不在 δa 的可达子空间里，Step 1 此路不通**

用法: python sim/step1_gate.py
"""
from __future__ import annotations
import glob
import json
import re
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parents[1]
TP = 15                      # H 的正滞后范围（帧）
RIDGE = 1e-6
EPS_B = 0.010                # 现有硬界 10 mrad
ARMS = [("D", "outputs/ab2"), ("E", "outputs/ab3"), ("F", "outputs/ab3")]


def episodes(root, arm, mf=500):
    fs = sorted(glob.glob(f"{root}/{arm}_ep*.json"),
                key=lambda p: int(re.search(r"ep(\d+)", p).group(1)))
    return [f for f in fs if json.load(open(f))["summary"]["frames"] > mf][:10]


def load(f, R, rend, bp):
    d = json.load(open(f))
    imu = np.array([x["imu"] for x in d["frames"] if x["imu"] is not None], np.float32)
    da = np.array([x["da"] for x in d["frames"]], np.float32)
    outs, prev = [], None
    for c in d["chunks"]:
        raw = np.array(c["raw"], np.float32)
        outs.append(rend.render(raw, prev)); prev = raw
    ide = np.concatenate(outs)[:len(imu)]
    n = min(len(ide), len(da))
    gap = bp(imu[:n, 3:6] - ide[:n, 3:6] @ R.T)
    return da[:n].astype(np.float64), gap.astype(np.float64)


def design(da, lags):
    n = len(da)
    X = np.zeros((n, len(lags) * 6))
    for i, t in enumerate(lags):
        X[t:, i * 6:(i + 1) * 6] = da[:n - t] if t else da
    return X


def fit_H(DA, GAP, lags):
    A = np.concatenate([design(d, lags) for d in DA])
    Y = np.concatenate(GAP)
    A = A - A.mean(0); Y = Y - Y.mean(0)
    G = A.T @ A
    G += RIDGE * np.trace(G) / len(G) * np.eye(len(G))
    W = np.linalg.solve(G, A.T @ Y)                     # (L*6, 3)
    return W.reshape(len(lags), 6, 3).transpose(0, 2, 1)   # (L,3,6)


def conv(H, da):
    """H(L,3,6) ⊛ da(n,6) → (n,3)，因果（滞后 0..L−1）。"""
    n = len(da)
    out = np.zeros((n, 3))
    for t in range(H.shape[0]):
        out[t:] += da[:n - t] @ H[t].T
    return out


def optimal_da(H, base, lam):
    """频域闭式解：min ‖base + H⊛δa‖² + λ‖δa‖²。**非因果**（用到全序列）。"""
    n = len(base)
    Hf = np.fft.rfft(H, n=n, axis=0)                    # (nf,3,6)
    Bf = np.fft.rfft(base, n=n, axis=0)                 # (nf,3)
    Df = np.zeros((Hf.shape[0], 6), complex)
    for k in range(Hf.shape[0]):
        Hk = Hf[k]
        M = Hk.conj().T @ Hk + lam * np.eye(6)
        Df[k] = -np.linalg.solve(M, Hk.conj().T @ Bf[k])
    da = np.fft.irfft(Df, n=n, axis=0)
    return da


def main():
    from analyze_ab import _bp
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    bp = _bp()
    R = np.array(json.load(open(ROOT / "outputs/p2_labels/pp2-deploy/manifest.json"
                                ))["R"], np.float32)
    rend = IdealChunkRenderer()
    lags = list(range(0, TP + 1))

    print("Step 1 的门：δa 通道**有没有能力**抵消 gap")
    print("（H 由真机实测，非建模；留出评估；非因果上界）\n")

    for arm, root in ARMS:
        fs = episodes(str(ROOT / root), arm)
        if len(fs) < 6:
            print(f"  {arm}: 集数不足，跳过"); continue
        DA, GAP = [], []
        for f in fs:
            d, g = load(f, R, rend, bp)
            DA.append(d); GAP.append(g)
        k = max(len(fs) - 3, 4)                          # 前 k 集训，其余留出
        H = fit_H(DA[:k], GAP[:k], lags)

        print(f"  === {arm} ===  训练 {k} 集 / 留出 {len(fs)-k} 集")
        print(f"  {'λ':>10}{'‖δa‖ RMS':>12}{'‖δa‖ p99':>11}"
              f"{'留出降幅':>10}{'ε 内?':>8}")
        rows = []
        for lam in (1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3):
            red, rms, p99 = [], [], []
            for d, g in zip(DA[k:], GAP[k:]):
                base = g - conv(H, d)                    # 还原"无补正"基线
                dnew = optimal_da(H, base, lam)
                res = base + conv(H, dnew)               # **因果卷积**施加，口径一致
                red.append(1 - (res ** 2).sum() / (base ** 2).sum())
                rms.append(np.sqrt((dnew ** 2).mean()))
                p99.append(np.percentile(np.abs(dnew), 99))
            r, m, q = np.mean(red), np.mean(rms), np.mean(p99)
            rows.append((lam, m, q, r))
            print(f"  {lam:>10.0e}{m*1e3:>11.2f}m{q*1e3:>10.2f}m{r*100:>9.1f}%"
                  f"{('✓' if q <= EPS_B else '✗'):>8}")
        # 在 ε 约束内能拿到多少
        ok = [r for r in rows if r[2] <= EPS_B]
        best = max(ok, key=lambda r: r[3]) if ok else None
        if best:
            print(f"  ⇒ ε=10 mrad 内最好: 降幅 **{best[3]*100:.1f}%**"
                  f"（‖δa‖ RMS {best[1]*1e3:.2f} mrad）")
        else:
            print(f"  ⇒ **没有任何 λ 能在 ε=10 mrad 内实现**")
        print()

    # ---------- 决定性检查：换一个 H 还成立吗 ----------
    print("=" * 70)
    print("决定性检查：用**另一条臂**的 H 求解 δa，再用本臂的 H 评估其真实效果")
    print("（E 与 F 同一天、同一台机器、同一会话，只差 FF 开关）\n")
    def prep(arm, root, ntr):
        fs = episodes(str(ROOT / root), arm)
        DA, GAP = [], []
        for f in fs:
            d, g = load(f, R, rend, bp)
            DA.append(d); GAP.append(g)
        return DA, GAP, fit_H(DA[:ntr], GAP[:ntr], lags)
    DAe, GAPe, He = prep("E", "outputs/ab3", 5)
    DAf, GAPf, Hf = prep("F", "outputs/ab3", 7)
    self_, cross = [], []
    for d, g in zip(DAf[7:], GAPf[7:]):
        base = g - conv(Hf, d)
        for H_solve, acc in ((Hf, self_), (He, cross)):
            dn = optimal_da(H_solve, base, 1e1)
            acc.append(1 - ((base + conv(Hf, dn)) ** 2).sum() / (base ** 2).sum())
    print(f"  用 F 自己的 H 求解、F 的 H 评估: {np.mean(self_)*100:>7.1f}%   ← 自洽")
    print(f"  用 E 的   H 求解、F 的 H 评估: {np.mean(cross)*100:>7.1f}%   ← **跨臂**")
    print("\n  ⇒ 换一个同日同机的 H，「最优」δa 从大幅降低变成大幅恶化。")
    print("     87–93% 是**自洽**而非真实能力；δa 的逆问题**极度病态**。\n")

    print("=" * 70)
    print("判读：")
    print("  · ε 内降幅可观      ⇒ 动作空间够用，问题在如何求解 δa")
    print("  · 要很大 ‖δa‖ 才降  ⇒ ε 是瓶颈，须放开界")
    print("  · 怎么都降不下来    ⇒ 误差不在 δa 可达子空间，Step 1 此路不通")
    print("\n  注：这是**非因果上界**（频域解用到未来）。因果天花板另有实测 ≈30%。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
