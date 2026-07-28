#!/usr/bin/env python3
"""DAgger Round 1 —— **只测数据聚合本身**（2026-07-29，用户规格）。

问题（离线只答这一个）:
    聚合后 π 在**未参与训练的新真机 rollout** 上，是否明显优于原来的 4.7%？

设计（两臂**环境完全相同**，只有训练数据不同 ⇒ 归因干净）:
    baseline  训练 = pp2-deploy(0-9) + ins-deploy(0-9)
    aggregated 训练 = 上述 + **onpolicy-r1(0-6)**
    两臂共同的留出 = **onpolicy-r1 {7,8,9}**（完整的新 rollout，不是随机切帧）

π 架构与特征**保持不变**（不同时加 δa 历史、滤波等），否则无法归因。
Ĝ 当**不确定环境**：K=5 集成，报名义成员与**最坏成员**；β=300 保守。

用法:
  python sim/dagger_round1.py --smoke              # 200 步计时
  python sim/dagger_round1.py --steps 4000 --seeds 3
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import train_pi_e2 as TP

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/p2_dagger"
HOLDOUT = [7, 8, 9]                      # 完整的新 rollout 留出（用户 2026-07-29 定 3 条）
NEW_TRAIN = [0, 1, 2, 3, 4, 5, 6]
OLD_TAGS = ["pp2-deploy", "ins-deploy"]
NEW_TAG = "onpolicy-r1"


def load_arm(arm, G):
    """按臂加载训练集；留出集两臂共用。"""
    tr = []
    TP.TAGS = OLD_TAGS
    tr += TP.load_eps(list(range(10)), G)          # 原数据全用（0-9）
    if arm == "aggregated":
        TP.TAGS = [NEW_TAG]
        tr += TP.load_eps(NEW_TRAIN, G)            # + 新 on-policy(0-6)
    TP.TAGS = [NEW_TAG]
    va = TP.load_eps(HOLDOUT, G)                   # 留出 = 新 rollout {7,8,9}
    TP.TAGS = OLD_TAGS
    return tr, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--beta", type=float, default=300.0)   # 用户: 保守 β=300
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.seeds = 200, 1
    OUT.mkdir(parents=True, exist_ok=True)
    TP.BP = TP.make_bp()
    from bakeoff_backbones import TCN
    G = TP.GEnsemble()
    print(f"=== DAgger Round 1 · 只测数据聚合 (β={a.beta:g}, {a.steps} 步 × "
          f"{a.seeds} seed) ===")
    print(f"  留出 = {NEW_TAG} {HOLDOUT}（完整新 rollout，两臂共用）\n")

    res = {}
    for arm in ("baseline", "aggregated"):
        tr, va = load_arm(arm, G)
        ntr = sum(len(e["e"]) for e in tr)
        print(f"  [{arm}] 训练 {len(tr)} 集 / {ntr} 帧   留出 {len(va)} 集 / "
              f"{sum(len(e['e']) for e in va)} 帧")
        rs = []
        for s in range(a.seeds):
            t0 = time.time()
            m = TP.train_one(TCN, 3e-3, s, a.steps, tr, G, a.beta, a.gamma,
                             log_every=10 ** 9)
            r = TP.evaluate(m, va, G)
            r["seed"] = s
            rs.append(r)
            print(f"    seed{s}: 消减 {r['red']*100:5.1f}%  (运动段 "
                  f"{r['red_motion']*100:5.1f}%, **最坏成员 {r['worst']*100:5.1f}%**)  "
                  f"‖δa‖={r['da_rms_mrad']:.2f}mrad sat={r['sat_pct']:.0f}% "
                  f"粗糙={r['rough_mrad']:.2f}  ({time.time()-t0:.0f}s)")
            torch.save(m.state_dict(), OUT / f"{arm}_s{s}.pt")
        res[arm] = rs

    # δa≡0 对照（消减应≈0）
    class Zero(torch.nn.Module):
        def forward(self, x):
            return x.new_zeros(x.shape[0], x.shape[1], 6)
    _, va = load_arm("baseline", G)
    z = TP.evaluate(Zero().to(TP.DEV), va, G)
    print(f"\n  [δa≡0 对照] 消减 {z['red']*100:.3f}% (应≈0)")

    print(f"\n  {'臂':>12}{'消减 中位':>12}{'运动段':>10}{'最坏成员':>11}"
          f"{'‖δa‖mrad':>11}{'sat%':>7}")
    for arm in ("baseline", "aggregated"):
        rs = res[arm]
        md = lambda k: float(np.median([r[k] for r in rs]))
        print(f"  {arm:>12}{md('red')*100:>11.1f}%{md('red_motion')*100:>9.1f}%"
              f"{md('worst')*100:>10.1f}%{md('da_rms_mrad'):>11.2f}{md('sat_pct'):>7.0f}")
    b = float(np.median([r["red"] for r in res["baseline"]]))
    g = float(np.median([r["red"] for r in res["aggregated"]]))
    print(f"\n  **聚合增益 = {(g-b)*100:+.1f} pp**  ({b*100:.1f}% → {g*100:.1f}%)")
    print(f"  参照: 原报告的新 rollout 效果 4.7%")
    (OUT / "round1.json").write_text(json.dumps(
        {"holdout": HOLDOUT, "new_train": NEW_TRAIN, "beta": a.beta,
         "steps": a.steps, "seeds": a.seeds, "zero_ctrl": z, "res": res},
        indent=1, ensure_ascii=False))
    print(f"\n→ outputs/p2_dagger/round1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
