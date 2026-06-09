#!/usr/bin/env python3
"""#63 — S0 vs S0+H 对照: ACT encoder 摘要是否破 hf 地板(尤其 j3)。

读 phase1_S0_in-act_output_* 与 phase1_S0+H_in-act_output_* 的 history.json,
best-by-rule(hf_ratio↓ 且 L_BC≤1.1×baseline), 比 hf_ratio / 逐关节 / L_BC / |ΔA| / 参数。

用法: python sim/compare_s0h.py
"""
from __future__ import annotations
import json, glob
from pathlib import Path

TRAIN = Path(__file__).resolve().parents[1] / "outputs/train/refine_net"
JOINTS = ["j1", "j2", "j3", "grip"]
BC_TOL = 1.1


def best_by_rule(history):
    best = None
    for h in history:
        v = h["val"]
        if v["l_bc"] > BC_TOL * v["l_bc_act"]:
            continue
        if best is None or v["hf_ratio"] < best[1]["hf_ratio"]:
            best = (h["epoch"], v)
    return best or (history[-1]["epoch"], history[-1]["val"])


def main():
    runs = [("S0", "phase1_S0_in-act_output_hf0.1_dl0.1_*"),
            ("S0+H", "phase1_S0+H_in-act_output_hf0.1_dl0.1_*"),
            ("S0+Hattn", "phase1_S0+Hattn_in-act_output_hf0.1_dl0.1_*")]
    print(f"\n{'='*86}")
    print("#63  S0 vs S0+H (ACT encoder 摘要 token)  best-by-rule(hf_ratio↓ & L_BC≤1.1×baseline)")
    print(f"{'='*86}")
    print(f"{'变体':>6} {'参数':>10} {'ep':>4} {'hf_ratio':>9} {'L_BC':>7} {'|ΔA|':>7}  {'j1/j2/j3/grip hf_ratio':>26}")
    print("-" * 86)
    rows = {}
    for name, pat in runs:
        ds = sorted(glob.glob(str(TRAIN / pat)))
        if not ds:
            print(f"{name:>6}  (无结果: {pat})"); continue
        cfg = json.loads((Path(ds[-1]) / "config.json").read_text())
        hist = json.loads((Path(ds[-1]) / "history.json").read_text())
        ep, v = best_by_rule(hist)
        jr = [r / (a + 1e-9) for r, a in zip(v["hf_refined_joints"], v["hf_act_joints"])]
        rows[name] = {"hf": v["hf_ratio"], "jr": jr, "n": cfg["n_params"]}
        print(f"{name:>6} {cfg['n_params']:>10,} {ep:>4} {v['hf_ratio']:>9.3f} {v['l_bc']:>7.4f} "
              f"{v['l_delta']:>7.4f}  " + " ".join(f"{x:.2f}" for x in jr))
    print("-" * 86)
    if "S0" in rows and "S0+H" in rows:
        d = rows["S0+H"]["hf"] - rows["S0"]["hf"]
        dj3 = rows["S0+H"]["jr"][2] - rows["S0"]["jr"][2]
        print(f"[效应] 加 ACT encoder 摘要:hf_ratio {d:+.3f}  ;  j3 {dj3:+.3f}")
        print("  判读: 明显负(更低) → 信息瓶颈, ACT embedding 有用(论文正向);"
              " ≈0 → 地板非 info, 强化控制器/容量机制结论。")


if __name__ == "__main__":
    main()
