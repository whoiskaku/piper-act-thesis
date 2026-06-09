#!/usr/bin/env python3
"""#63(a) — S0 / S0+Hattn / S0+Hattn+zero 跨 seed 汇总。

每变体 glob 所有 seed 目录, best-by-rule(hf_ratio↓ & L_BC≤1.1×baseline)取每 seed 最优,
报告 mean±range + per-joint mean。回答:
  S0+Hattn 是否稳定低于 S0(seed 稳健性)?
  S0+Hattn+zero(同架构无真实 image)≈ S0 → 增益来自 image 信息; ≈ S0+Hattn → 来自参数容量。

用法: python sim/compare_s0h_seeds.py
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np

TRAIN = Path(__file__).resolve().parents[1] / "outputs/train/refine_net"
BC_TOL = 1.1


def best_by_rule(history):
    best = None
    for h in history:
        v = h["val"]
        if v["l_bc"] > BC_TOL * v["l_bc_act"]:
            continue
        if best is None or v["hf_ratio"] < best["hf_ratio"]:
            best = v
    return best or history[-1]["val"]


def collect(pat):
    hfs, jrs, nps = [], [], None
    for d in sorted(glob.glob(str(TRAIN / pat))):
        try:
            hist = json.loads((Path(d) / "history.json").read_text())
            cfg = json.loads((Path(d) / "config.json").read_text())
        except FileNotFoundError:
            continue
        v = best_by_rule(hist)
        hfs.append(v["hf_ratio"])
        jrs.append([r / (a + 1e-9) for r, a in zip(v["hf_refined_joints"], v["hf_act_joints"])])
        nps = cfg["n_params"]
    return hfs, np.array(jrs), nps


def main():
    variants = [
        ("S0",            "phase1_S0_in-act_output_hf0.1_dl0.1*"),
        ("S0+Hattn",      "phase1_S0+Hattn_in-act_output_hf0.1_dl0.1*"),
        ("S0+Hattn+zero", "phase1_S0+Hattn+zero_in-act_output_hf0.1_dl0.1*"),
    ]
    print(f"\n{'='*92}")
    print("#63(a) S0 / S0+Hattn / S0+Hattn+zero  跨 seed (best-by-rule hf_ratio)")
    print(f"{'='*92}")
    print(f"{'变体':>15} {'参数':>10} {'n_seed':>6} {'hf_ratio mean[min,max]':>26} {'j1/j2/j3/grip(mean)':>22}")
    print("-" * 92)
    res = {}
    for name, pat in variants:
        hfs, jrs, nps = collect(pat)
        if not hfs:
            print(f"{name:>15}  (无结果)"); continue
        res[name] = np.array(hfs)
        jm = jrs.mean(axis=0)
        print(f"{name:>15} {nps:>10,} {len(hfs):>6} "
              f"{np.mean(hfs):>10.3f}[{min(hfs):.3f},{max(hfs):.3f}]      "
              + " ".join(f"{x:.2f}" for x in jm))
    print("-" * 92)
    if all(k in res for k in ["S0", "S0+Hattn", "S0+Hattn+zero"]):
        s0, sa, sz = res["S0"].mean(), res["S0+Hattn"].mean(), res["S0+Hattn+zero"].mean()
        print(f"[效应] S0+Hattn − S0 = {sa-s0:+.3f}  ;  S0+Hattn+zero − S0 = {sz-s0:+.3f}")
        print("  判读:")
        print(f"   · S0+Hattn 稳定低于 S0(range 不重叠)→ image 注意力读出确有用")
        print(f"   · zero≈S0(≈{s0:.2f}) 而 attn<<S0 → 增益来自 **image 信息**(非参数容量)✓")
        print(f"   · zero≈attn(≈{sa:.2f}) → 增益来自 **参数容量**,需重新解读")


if __name__ == "__main__":
    main()
