#!/usr/bin/env python3
"""π 的直接监督回归 —— 去掉 Ĝ（2026-08-04，用户定的主线）。

**为什么**：已部署的 π（`train_pi_e2.py`）训练损失是
    l1 = ‖bandpass(e + Ĝ[δa])‖²
**完全经由 Ĝ**。而 Ĝ 的增量通道被真机证伪：Ĝ 说 δa 减小 gap 6.5%，真机测得**增大 38%**，
最差 4–6 Hz 偏差 +81%。⇒ π 学得越好，真机越差（实测 +32.7%，1/10 集改善）。

但教师标签 δa* 是用**解析雅可比**算的，**不经 Ĝ**：
    δa* = −dt·(SᵀS+λI)⁻¹Sᵀ·e_gyro_band,   S = R_cal·R_siteᵀ·Jr_arm(q_cmd)

实测（`pi_v1_s0.pt` 在验证集 3136 运动帧上）：Ĝ 训练出的 π 与 δa* **方向相关 −0.088**、
余弦>0 仅 44–49%（随机 50%）、幅值比 0.57、**j4 输出恒为 0**。
⇒ **δa\* 这条路从未被真机检验过**。本脚本就是去检验它。

**本脚本做的事**：同一骨干（TCN，11334 参数）、同一特征（51 维）、同一划分，
损失换成对 δa* 的加权 MSE。**没有 Ĝ，没有闭环仿真，没有集成。**
输出仍经 `tanh × ε`（ε=10 mrad）硬界，保留部署时的安全性质。

已知风险（一并记录）：δa* 建立在"δa 在一帧内保持 → 附加关节速度 δq̇ ≈ δa/dt"的
单帧模型上，`build_dagger_labels` 注释自承"有偏（伺服≠单帧速度模型）"。
直接回归会继承这个偏差 —— 这正是它可能失败的地方，也是必须上真机才能判定的原因。

用法:
  python sim/train_pi_direct.py --steps 4000            # 训练 + 留出评估
  python sim/train_pi_direct.py --compare               # 只对比 Ĝ训练的 π vs 标签
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))

from lerobot.common.policies.act.b_runtime import build_offline

ROOT = Path(__file__).resolve().parents[1]
LROOT = ROOT / "outputs/p2_labels"
OUT = ROOT / "outputs/p2_pi"
TAGS = ["pp2-deploy", "ins-deploy"]
TRAIN_EPS, VAL_EPS = list(range(8)), [8, 9]
LSHIFT, FEAT, EPS_B = 3, 51, 0.010
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEG, BATCH = 96, 32


def load_split(eps_list):
    """→ [{X, lab, w, motion, tag, ep}]  —— 只用标签，不碰 Ĝ。"""
    out = []
    for tag in TAGS:
        for ei in eps_list:
            z = np.load(LROOT / tag / f"ep{ei:03d}.npz")
            X, _, _ = build_offline(z, L=LSHIFT)
            n = min(len(X), len(z["label"]))
            gate = X[:n, -1]
            w = np.clip(gate / 0.15, 0.2, 1.0).astype(np.float32)
            w[z["mask_contact"][:n]] = 0.0          # 接触帧不训（与原口径一致）
            w[z["mask_still"][:n]] = 0.0            # still 段标签本就置零
            out.append({
                "X": torch.from_numpy(X[:n]).to(DEV),
                "lab": torch.from_numpy(z["label"][:n].astype(np.float32)).to(DEV),
                "w": torch.from_numpy(w).to(DEV),
                "motion": torch.from_numpy(~z["mask_still"][:n]).to(DEV),
                "tag": tag, "ep": ei})
    return out


def batches(eps, steps, rng):
    for _ in range(steps):
        bi = rng.integers(0, len(eps), BATCH)
        X, Y, W = [], [], []
        for i in bi:
            e = eps[i]
            T = len(e["X"])
            o = int(rng.integers(0, max(T - SEG, 1)))
            X.append(e["X"][o:o + SEG]); Y.append(e["lab"][o:o + SEG])
            W.append(e["w"][o:o + SEG])
        yield torch.stack(X), torch.stack(Y), torch.stack(W)


def fit_quality(model, eps):
    """留出集上对 δa* 的拟合质量。只在运动帧上评。"""
    model.eval()
    P, L = [], []
    with torch.no_grad():
        for e in eps:
            d = model(e["X"].unsqueeze(0))[0]
            m = e["motion"]
            P.append(d[m].cpu().numpy()); L.append(e["lab"][m].cpu().numpy())
    P, L = np.concatenate(P), np.concatenate(L)
    dir_corr = float((P * L).sum() / np.sqrt((P ** 2).sum() * (L ** 2).sum()))
    r2 = float(1 - ((P - L) ** 2).sum() / (L ** 2).sum())
    cos = (P * L).sum(1) / np.maximum(
        np.linalg.norm(P, axis=1) * np.linalg.norm(L, axis=1), 1e-12)
    per_j = [float(np.corrcoef(P[:, j], L[:, j])[0, 1]) for j in range(6)]
    return {"dir_corr": dir_corr, "r2": r2, "cos_pos": float((cos > 0).mean()),
            "rms_pred_mrad": float(np.sqrt((P ** 2).mean()) * 1e3),
            "rms_lab_mrad": float(np.sqrt((L ** 2).mean()) * 1e3),
            "per_joint": per_j, "n": len(P)}


def show(name, q):
    print(f"  {name:<22} 方向相关 {q['dir_corr']:+.3f}   R² {q['r2']:+.3f}   "
          f"余弦>0 {q['cos_pos']*100:.1f}%   ‖pred‖ {q['rms_pred_mrad']:.2f} / "
          f"‖lab‖ {q['rms_lab_mrad']:.2f} mrad")
    print(f"  {'':22} 逐关节 " + " ".join(f"j{j+1}{q['per_joint'][j]:+.2f}"
                                          for j in range(6)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", action="store_true", help="只对比已部署 π 与标签")
    a_ = ap.parse_args()

    from bakeoff_backbones import TCN
    tr, va = load_split(TRAIN_EPS), load_split(VAL_EPS)
    print(f"device={DEV}  train {sum(len(e['X']) for e in tr)} / "
          f"val {sum(len(e['X']) for e in va)} 帧")

    print("\n【基准】已部署的 Ĝ 训练版 π 对 δa* 的拟合（留出集）")
    old = TCN(d=FEAT).to(DEV)
    old.load_state_dict(torch.load(OUT / "pi_v1_s0.pt", map_location=DEV))
    q_old = fit_quality(old, va)
    show("pi_v1_s0 (Ĝ训练)", q_old)
    if a_.compare:
        return 0

    print(f"\n【训练】直接回归 δa*，{a_.steps} 步")
    torch.manual_seed(a_.seed)
    rng = np.random.default_rng(a_.seed)
    model = TCN(d=FEAT).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a_.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a_.steps)
    # **按留出 R² 早停**。R² 不是随手选的判据: 若标签 δa* 是所需修正, 则施加 δa 后
    # 的残差 = ‖δa*−δa‖²/‖δa*‖² = 1−R² ⇒ R² 直接等于"交付了所需修正的百分之多少"。
    # 实测 4000 步会过拟合(R² 500步 +0.160 → 4000步 −0.015), 必须早停。
    t0 = time.time()
    best = {"r2": -1e9}
    every = max(a_.steps // 40, 25)
    for i, (X, Y, W) in enumerate(batches(tr, a_.steps, rng)):
        model.train()
        pred = model(X)
        loss = (W.unsqueeze(-1) * (pred - Y) ** 2).sum() / W.sum().clamp(min=1) / 6
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        if (i + 1) % every == 0:
            q = fit_quality(model, va)
            if q["r2"] > best["r2"]:
                best = dict(q, step=i + 1)
                best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if (i + 1) % (every * 5) == 0:
                print(f"    {i+1:>5} 步  loss {loss.item():.3e}   "
                      f"留出方向相关 {q['dir_corr']:+.3f}  R² {q['r2']:+.3f}")
    print(f"  ({time.time()-t0:.0f}s)   最佳在第 {best['step']} 步")
    model.load_state_dict(best_sd)

    q_new = fit_quality(model, va)
    print("\n【结果】留出集上对 δa* 的拟合")
    show("直接回归 (本次)", q_new)
    show("pi_v1_s0 (Ĝ训练)", q_old)

    ck = OUT / f"pi_direct_s{a_.seed}.pt"
    torch.save(model.state_dict(), ck)
    (OUT / f"pi_direct_s{a_.seed}.json").write_text(json.dumps(
        {"direct": q_new, "g_trained_baseline": q_old,
         "steps": a_.steps, "lr": a_.lr, "seed": a_.seed}, indent=1))
    print(f"\n→ {ck.relative_to(ROOT.parent)}")
    print("\n判读：留出方向相关若 ≫ 0（比如 >0.7），说明 δa* 是**可从因果特征预测的**，")
    print("      得到一个与已部署 π 截然不同（方向相关 −0.088）的候选，值得上真机。")
    print("      若拟合不起来，说明标签本身不可预测，应先改标签而不是上机。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
