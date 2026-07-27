#!/usr/bin/env python3
"""A+B (2026-07-22 用户批准): pp2 数据扩容 + β 全程重扫, 因子化一次跑清。

A: 建标签 pp2f-deploy(eval_pp2_fork_100k, 7集) + pp2f60-deploy(eval_pp2_fork_60k, 3集)
   — 20Hz+IMU 的 pick-place 家族; 逐数据集独立标定(R/lag)。
   划分: fork100k 0-5 训 / 6 验; fork60k 0-2 全训。pp2 训练集 8 → 17。
B: β∈{300,1000} × 数据∈{旧16, 扩25} × 3种子 (TCN-deep, 6000步, γ=0);
   外加 β=100×扩×1种子 探索位。
输出: outputs/p2_pi/ab_expand.json
"""
from __future__ import annotations
import json
import sys
import time

import numpy as np
import torch

NEW_SETS = [("enolab-guo/eval_pp2_fork_100k", "pp2f-deploy", 7),
            ("enolab-guo/eval_pp2_fork_60k", "pp2f60-deploy", 3)]


def build_labels():
    import build_dagger_labels as B
    from pathlib import Path
    for repo, tag, n in NEW_SETS:
        out = B.OUT_ROOT / tag / "manifest.json"
        if out.exists():
            print(f"[labels] {tag} 已存在, 跳过", flush=True)
            continue
        # 前置检查: state_mdeg 必须在(qw 特征依赖编码器)
        from compare_ideal_methods import load_episode
        d = load_episode(repo, 0)
        assert "state_mdeg" in d and d["state_mdeg"] is not None, f"{repo} 缺编码器!"
        B.DATASETS = [(repo, tag)]
        sys.argv = ["build_dagger_labels", "--eps", str(n)]
        print(f"[labels] 构建 {tag} ({n} 集) ...", flush=True)
        B.main()


def main():
    build_labels()

    import train_pi_e2 as TP
    from rerace_dropout import TCNDeep_D
    TP.BP = TP.make_bp()
    G = TP.GEnsemble()

    eps_old = TP.load_eps(TP.TRAIN_EPS, G)                 # pp2×8 + ins×8
    va = TP.load_eps(TP.VAL_EPS, G)
    va_pp = [e for e in va if e["tag"] == "pp2-deploy"]
    va_ins = [e for e in va if e["tag"] == "ins-deploy"]
    tags0 = TP.TAGS
    TP.TAGS = ["pp2f-deploy"]
    ppf_tr, ppf_va = TP.load_eps(list(range(6)), G), TP.load_eps([6], G)
    TP.TAGS = ["pp2f60-deploy"]
    ppf60_tr = TP.load_eps([0, 1, 2], G)
    TP.TAGS = tags0
    eps_new = eps_old + ppf_tr + ppf60_tr                  # 25 集

    res = {}
    def cell(data, dname, beta, seeds):
        recs = []
        for sd in seeds:
            t0 = time.time()
            m = TP.train_one(TCNDeep_D, 1e-3, sd, 6000, data, G, beta, 0.0,
                             log_every=10**9)
            r = {grp: TP.evaluate(m, ev, G)["red"] * 100
                 for grp, ev in [("pp2", va_pp), ("ins", va_ins), ("ppf", ppf_va)]}
            r["da"] = TP.evaluate(m, va_pp, G)["da_rms_mrad"]
            recs.append(r)
            print(f"  [{dname} β={beta:g} s{sd}] pp2={r['pp2']:5.1f}% ins={r['ins']:5.1f}% "
                  f"ppf={r['ppf']:5.1f}% ‖δa‖={r['da']:.2f} ({time.time()-t0:.0f}s)", flush=True)
        key = f"{dname}_b{beta:g}"
        res[key] = {g: [float(np.mean([x[g] for x in recs])),
                        float(np.std([x[g] for x in recs]))] for g in recs[0]}
        (TP.OUT / "ab_expand.json").write_text(json.dumps(res, indent=1))

    for beta in (1e3, 3e2):
        for data, dname in [(eps_old, "old16"), (eps_new, "new25")]:
            cell(data, dname, beta, (0, 1, 2))
    cell(eps_new, "new25", 1e2, (0,))                      # 探索位
    print("\n== 汇总 ==")
    for k, v in res.items():
        print(f"  {k:>16}: pp2={v['pp2'][0]:5.1f}±{v['pp2'][1]:.1f}%  "
              f"ins={v['ins'][0]:5.1f}±{v['ins'][1]:.1f}%  ppf={v['ppf'][0]:5.1f}%")


if __name__ == "__main__":
    main()
