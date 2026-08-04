#!/usr/bin/env python3
"""直接**测量**手臂对 δa 的真实响应（2026-08-04）—— 不建模，只量。

**为什么**：两版 π 都失败，而它们用的驱动模型完全不同（v1 拟合动态 Ĝ，v2 解析单帧）。
到底是模型不准，还是这个动作空间本身就没用？这不需要再造第三个模型 ——
D / E / F 三条臂**都施加过 δa 并同步录了 IMU**，真实响应可以直接量。

**做法**：把 δa 当输入、带通 gap 当输出，用岭回归估经验冲激响应

    gap(t) ≈ Σ_τ H(τ)·δa(t−τ)        τ = −TN … +TP，H(τ) ∈ R^{3×6}

**闭环偏置的读法**（关键，不能忽略）：δa(t) 是 gap 历史（到 t−L）的函数，
所以这是闭环数据。标准读法是按 τ 的符号分开：

  · τ ≥ 0 的部分 ≈ **被控对象响应**（δa 之后发生的事）  ← 我们要的
  · τ < 0 的部分 ≈ **控制器**（π 如何依 gap 产生 δa）    ← 反馈路径，不是响应

τ<0 若出现显著幅度，那是反馈的印记，**不可当作响应读**。

**零假设**：把 δa 环移 200 帧后重估，得到纯噪声下的 H 幅度 —— 这是判显著的地板。

**最后的判决**：用测出的 H 算 δa 的贡献 `Ĥ⊛δa`，看它与实际 gap 是**同向还是反向**。
同向 ⇒ 施加的修正在**增大** gap（动作空间被用反了或根本没用）。

用法: python sim/measure_da_response.py
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
TN, TP = 5, 15                     # 负/正滞后范围（帧）
RIDGE = 1e-6
NULL_SHIFT = 200


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
    gap = bp(imu[:n, 3:6] - ide[:n, 3:6] @ R.T)          # (n,3) 带通
    return da[:n], gap


def design(da, lags):
    """δa(n,6) → 设计阵 (n, len(lags)*6)，第 τ 块是 δa(t−τ)。"""
    n = len(da)
    X = np.zeros((n, len(lags) * 6), np.float64)
    for i, tau in enumerate(lags):
        if tau >= 0:
            X[tau:, i * 6:(i + 1) * 6] = da[:n - tau] if tau else da
        else:
            X[:n + tau, i * 6:(i + 1) * 6] = da[-tau:]
    return X


def fit(DA, GAP, lags):
    """→ H (len(lags),3,6), 解释方差比。"""
    Xs = [design(d, lags) for d in DA]
    X = np.concatenate(Xs); Y = np.concatenate(GAP).astype(np.float64)
    X = X - X.mean(0); Y = Y - Y.mean(0)
    G = X.T @ X
    G += RIDGE * np.trace(G) / len(G) * np.eye(len(G))
    W = np.linalg.solve(G, X.T @ Y)                       # (L*6, 3)
    pred = X @ W
    ev = 1 - ((Y - pred) ** 2).sum() / (Y ** 2).sum()
    return W.reshape(len(lags), 6, 3).transpose(0, 2, 1), ev, X, Y, pred


def main():
    from analyze_ab import _bp
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    bp = _bp()
    R = np.array(json.load(open(ROOT / "outputs/p2_labels/pp2-deploy/manifest.json"
                                ))["R"], np.float32)
    rend = IdealChunkRenderer()
    lags = list(range(-TN, TP + 1))

    print("经验冲激响应 δa → 带通 gap（岭回归，闭环数据）")
    print(f"  滞后范围 τ = {-TN}…{TP} 帧；τ<0 是**反馈路径**不是响应\n")
    OUT = {}
    for arm, root in (("D", "outputs/ab2"), ("E", "outputs/ab3"), ("F", "outputs/ab3")):
        fs = episodes(str(ROOT / root), arm)
        DA, GAP = [], []
        for f in fs:
            d, g = load(f, R, rend, bp)
            DA.append(d); GAP.append(g)
        H, ev, X, Y, pred = fit(DA, GAP, lags)
        # 零假设：环移 δa
        DAn = [np.roll(d, NULL_SHIFT, axis=0) for d in DA]
        Hn, evn, *_ = fit(DAn, GAP, lags)
        amp = np.linalg.norm(H, axis=(1, 2))              # 每个滞后的响应幅度
        ampn = np.linalg.norm(Hn, axis=(1, 2))
        # 判决：测出的响应把 gap 推向哪
        contrib = pred                                     # Ĥ⊛δa
        cos = float((contrib * Y).sum()
                    / np.sqrt((contrib ** 2).sum() * (Y ** 2).sum()))
        share = float((contrib ** 2).sum() / (Y ** 2).sum())
        OUT[arm] = dict(H=H, amp=amp, ampn=ampn, ev=ev, evn=evn, cos=cos, share=share)
        print(f"  === {arm} ===  {len(fs)} 集，{len(Y)} 帧")
        print(f"    解释方差  实测 {ev*100:>5.2f}%   零假设(环移) {evn*100:>5.2f}%")
        print(f"    {'τ(帧)':>7}" + "".join(f"{t:>7}" for t in lags[::2]))
        print(f"    {'|H|':>7}" + "".join(f"{amp[i]:>7.3f}" for i in range(0, len(lags), 2)))
        print(f"    {'零假设':>7}" + "".join(f"{ampn[i]:>7.3f}" for i in range(0, len(lags), 2)))
        neg = amp[:TN].max(); pos = amp[TN:].max()
        print(f"    峰值: τ<0 {neg:.3f}（反馈印记）  τ≥0 {pos:.3f}（响应）  "
              f"零假设上限 {ampn.max():.3f}")
        print(f"    **Ĥ⊛δa 与实际 gap 的方向相关 = {cos:+.3f}**"
              f"（占 gap 能量 {share*100:.1f}%）")
        print(f"    ⇒ {'同向：施加的 δa 在**增大** gap' if cos > 0 else '反向：δa 在减小 gap'}\n")

    # 与 Ĝ 的预测对比
    print("=" * 70)
    print("与 Ĝ 的冲激响应对比（Ĝ 说 δa 应该产生什么）")
    try:
        import torch
        import train_pi_e2 as TP_
        G = TP_.GEnsemble()
        N, T0 = 96, 40
        imp = np.zeros((1, N, 6), np.float32); imp[0, T0, :] = 1.0
        a_arm = np.tile(np.array([0.0, 0.5, -0.5, 0.0, 0.5, 0.0], np.float32), (N, 1))
        epj, al, RB, RD = G.ep_context(a_arm, np.zeros(N, np.float32), R)
        de = TP_.apply_G(torch.from_numpy(imp).to(TP_.DEV), epj.unsqueeze(0),
                         al.unsqueeze(0), RB.unsqueeze(0), RD.unsqueeze(0), G)
        gk = de[0].mean(0).detach().cpu().numpy()          # (N,3)
        gam = np.linalg.norm(gk, axis=1)
        print(f"  {'τ(帧)':>7}" + "".join(f"{t:>7}" for t in lags[::2]))
        print(f"  {'Ĝ |h|':>7}" + "".join(
            f"{gam[T0+t] if 0 <= T0+t < N else 0:>7.3f}" for t in lags[::2]))
        print(f"  Ĝ 的峰值滞后 τ = {int(np.argmax(gam)) - T0} 帧，峰值 {gam.max():.3f}")
    except Exception as e:
        print(f"  (Ĝ 载入失败: {type(e).__name__}: {e})")

    print("\n" + "=" * 70)
    print("判读：")
    print("  · τ≥0 的 |H| 若不显著高于零假设 ⇒ δa 对 gap 没有可测的影响 ⇒ 动作空间无效")
    print("  · Ĥ⊛δa 与 gap **同向** ⇒ 实际施加的修正在增大 gap（用反了）")
    print("  · Ĝ 的峰值滞后/幅度若与实测差很远 ⇒ Ĝ 不准（已知它符号也反）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
