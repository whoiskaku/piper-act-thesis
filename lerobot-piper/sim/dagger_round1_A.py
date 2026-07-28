#!/usr/bin/env python3
"""DAgger Round 1 主实验 —— **只加 A 臂**（2026-07-29，用户最终规格）。

为什么只用 A 臂（用户裁决）：
  · 复现 WORKLOG §10.4，与历史 13.9% → 4.7% **同口径**
  · A 臂是裸 ACT ⇒ `da_applied ≡ 0`，标签**没有总量/残量歧义**
  · C 臂实测 `a_sent − raw` RMS = 13.71 mrad，远超 ε=10 —— 它含 **FF 逆算前馈**
    与 δa_π 两部分（FF 占主导，π 只有 ~0.9 mrad）。不拆开不能用。
    ⇒ C 臂另作 on-policy 辅助实验，**不与 A 混池**。

三臂设计（每个 seed k ∈ 0..4，**同步数、同 β、同采样权重、同验证规则、同随机种子**）：
    P0 = 冻结 pi_v1_s{k}
    P1 = 从 P0 继续训练，**仅**原 4 个数据集
    P2 = 从 P0 继续训练，原 4 个数据集 + A0–5
  ⇒ P2−P1 = **真正的数据聚合收益**
    P2−P0 = 实际升级收益
    P1−P0 = **额外训练本身造成的漂移**（不减掉它就会把漂移算成数据收益）

A 臂划分：A0–5 训练 / **A6 验证** / **A7–9 最终封存测试**。
P1、P2 用**同一个**验证集，且把 **step 0 的 πv1 纳入候选** —— 否则早停仍可能
被迫在"都比 πv1 差"的检查点里选一个。

用法: python sim/dagger_round1_A.py --smoke | --steps 3000 --seeds 5
"""
from __future__ import annotations
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

import train_pi_e2 as TP
from lerobot.common.policies.act.b_runtime import TCN as RTCN, load_tcn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/p2_dagger"
OLD_TAGS = ["pp2-deploy", "ins-deploy", "pp2f-deploy", "pp2f60-deploy"]
OLD_N = {"pp2-deploy": 10, "ins-deploy": 10, "pp2f-deploy": 7, "pp2f60-deploy": 3}
A_TAG = "abA-deploy"
A_TRAIN, A_VAL, A_TEST = [0, 1, 2, 3, 4, 5], [6], [7, 8, 9]


def load(tag, eps, G):
    TP.TAGS = [tag]
    out = TP.load_eps(eps, G)
    return out


def val_score(model, eps_va, G):
    """验证规则：A6 上的模型内消减（越大越好）。P1/P2 共用。"""
    return TP.evaluate(model, eps_va, G)["red"]


def train_from(init, steps, eps_tr, eps_va, G, beta, seed, lr=3e-3, ckpt_every=250):
    """从 `init` 继续训练；**step 0（即 init 本身）也是候选**，按验证集选最好的。"""
    torch.manual_seed(seed); np.random.seed(seed)
    m = copy.deepcopy(init).to(TP.DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    best = (val_score(m.eval(), eps_va, G), 0, copy.deepcopy(m.state_dict()))
    m.train()
    for i, (bi, off) in enumerate(TP.batch_iter(eps_tr, steps)):
        X, e, w, epJ, al, RB, RD = TP.gather(eps_tr, bi, off)
        l1, l2, l3, _ = TP.losses(m(X), e, w, epJ, al, RB, RD, G, beta, 0.0)
        opt.zero_grad(); (l1 + l2 + l3).backward(); opt.step()
        if (i + 1) % ckpt_every == 0:
            m.eval(); v = val_score(m, eps_va, G); m.train()
            if v > best[0]:
                best = (v, i + 1, copy.deepcopy(m.state_dict()))
    m.load_state_dict(best[2]); m.eval()
    return m, best[0], best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--beta", type=float, default=300.0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.seeds = 300, 1
    OUT.mkdir(parents=True, exist_ok=True)
    TP.BP = TP.make_bp()
    G = TP.GEnsemble()

    old_tr = []
    for t in OLD_TAGS:
        old_tr += load(t, list(range(OLD_N[t])), G)
    a_tr = load(A_TAG, A_TRAIN, G)
    a_va = load(A_TAG, A_VAL, G)
    a_te = load(A_TAG, A_TEST, G)
    print(f"=== DAgger Round 1 主实验 · 只加 A 臂 (β={a.beta:g}, {a.steps} 步 × "
          f"{a.seeds} 配对种子) ===")
    print(f"  原 4 集: {sum(len(e['e']) for e in old_tr)} 帧 ({len(old_tr)} 集)")
    print(f"  A 训练 A{A_TRAIN}: {sum(len(e['e']) for e in a_tr)} 帧")
    print(f"  A 验证 A{A_VAL}: {sum(len(e['e']) for e in a_va)} 帧  "
          f"(P1/P2 共用, step0=πv1 纳入候选)")
    print(f"  A **封存测试** A{A_TEST}: {sum(len(e['e']) for e in a_te)} 帧\n")

    # ---- 先在封存的 A7–9 上重算 πv1 锚点 ----
    print("  【锚点】πv1 在**封存** A7–9 上（与历史 4.7% 同口径）")
    anch = []
    for k in range(a.seeds):
        r = TP.evaluate(load_tcn(f"outputs/p2_pi/pi_v1_s{k}.pt").to(TP.DEV), a_te, G)
        anch.append(r)
        print(f"    s{k}: 消减 {r['red']*100:5.1f}%  最坏成员 {r['worst']*100:5.1f}%  "
              f"‖δa‖={r['da_rms_mrad']:.2f}mrad")
    print(f"    中位 **{np.median([r['red'] for r in anch])*100:+.1f}%**"
          f"   (历史记载 4.7%)\n")

    res = {"P0": anch, "P1": [], "P2": []}
    for k in range(a.seeds):
        p0 = load_tcn(f"outputs/p2_pi/pi_v1_s{k}.pt").to(TP.DEV)
        for arm, tr in (("P1", old_tr), ("P2", old_tr + a_tr)):
            t0 = time.time()
            m, v, st = train_from(p0, a.steps, tr, a_va, G, a.beta, seed=k)
            r = TP.evaluate(m, a_te, G); r["val"] = v; r["best_step"] = st
            res[arm].append(r)
            torch.save(m.state_dict(), OUT / f"A_{arm}_s{k}.pt")
            print(f"  s{k} {arm}: val(A6)={v*100:5.1f}% @step{st:<5d} "
                  f"→ 测试 A7-9 消减 {r['red']*100:5.1f}%  "
                  f"最坏 {r['worst']*100:5.1f}%  ‖δa‖={r['da_rms_mrad']:.2f}  "
                  f"({time.time()-t0:.0f}s)")

    print(f"\n  {'臂':>5}{'处理':>34}{'测试消减 中位':>15}{'最坏成员':>11}{'‖δa‖':>8}")
    lab = {"P0": "冻结 pi_v1", "P1": "继续训练 · 仅原 4 集",
           "P2": "继续训练 · 原 4 集 + A0-5"}
    md = {}
    for arm in ("P0", "P1", "P2"):
        rs = res[arm]; md[arm] = float(np.median([r["red"] for r in rs]))
        print(f"  {arm:>5}{lab[arm]:>34}{md[arm]*100:>14.1f}%"
              f"{np.median([r['worst'] for r in rs])*100:>10.1f}%"
              f"{np.median([r['da_rms_mrad'] for r in rs]):>8.2f}")
    # 配对差（同 seed 配对，不是中位数相减）
    def paired(x, y):
        d = [res[x][k]["red"] - res[y][k]["red"] for k in range(a.seeds)]
        return float(np.median(d)), float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    for x, y, nm in (("P2", "P1", "**数据聚合收益**"), ("P2", "P0", "实际升级收益"),
                     ("P1", "P0", "**额外训练的漂移**")):
        m_, se = paired(x, y)
        print(f"  {x}−{y}  {nm:>22}: {m_*100:+.2f} pp  (配对 SE {se*100:.2f})")
    (OUT / "round1_A.json").write_text(json.dumps(
        {"a_split": {"train": A_TRAIN, "val": A_VAL, "test": A_TEST},
         "old_tags": OLD_TAGS, "beta": a.beta, "steps": a.steps,
         "seeds": a.seeds, "res": res}, indent=1, ensure_ascii=False))
    print(f"\n→ outputs/p2_dagger/round1_A.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
