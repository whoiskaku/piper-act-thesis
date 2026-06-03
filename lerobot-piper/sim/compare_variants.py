#!/usr/bin/env python3
"""Stage 3b 输入消融 S0–S3 对比汇总。

读取 phase1_S{0,1,2,3}_hf0.1_dl0.1_* 各变体 history.json，按规则选 ckpt:
  primary  : mean hf_ratio (refined/ACT) 越低越好
  guardrail: L_BC ≤ 1.1 × ACT_baseline
输出: variant | n_enc_tok | best_ep | hf_ratio | L_BC | |ΔA| | per-joint hf_ratio
回答: history 各分支(state / hist_action / hist_imu)各值多少 hf_ratio 改善?

用法: python sim/compare_variants.py
"""
from __future__ import annotations
import json, glob
from pathlib import Path

TRAIN = Path(__file__).resolve().parents[1] / "outputs/train/refine_net"
JOINTS = ["j1", "j2", "j3", "grip"]
BC_TOL = 1.1
DESC = {"S0": "仅ACT", "S1": "+state", "S2": "+state+hist_action", "S3": "全部(=full)"}


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
    print(f"\n{'='*92}")
    print("Stage 3b 输入消融 S0–S3  (λ_hf=0.10, λ_delta=0.1, 200ep; best-by-rule: hf_ratio↓ & L_BC≤1.1×baseline)")
    print(f"{'='*92}")
    print(f"{'变体':>4} {'说明':>18} {'ep':>4} {'hf_ratio':>9} {'L_BC':>7} {'|ΔA|':>7}  {'j1/j2/j3/grip hf_ratio':>28}")
    print("-" * 92)
    rows = {}
    base = None
    for v in ["S0", "S1", "S2", "S3"]:
        ds = sorted(glob.glob(str(TRAIN / f"phase1_{v}_hf0.1_dl0.1_*")))
        if not ds:
            print(f"{v:>4}  (无结果)")
            continue
        cfg = json.loads((Path(ds[-1]) / "config.json").read_text())
        hist = json.loads((Path(ds[-1]) / "history.json").read_text())
        ep, val = best_by_rule(hist)
        jr = [r / (a + 1e-9) for r, a in zip(val["hf_refined_joints"], val["hf_act_joints"])]
        rows[v] = {"ep": ep, "hf_ratio": val["hf_ratio"], "l_bc": val["l_bc"],
                   "l_bc_act": val["l_bc_act"], "delta": val["l_delta"],
                   "jr": jr, "n_params": cfg["n_params"]}
        if base is None:
            base = val["l_bc_act"]
        print(f"{v:>4} {DESC[v]:>18} {ep:>4} {val['hf_ratio']:>9.3f} {val['l_bc']:>7.4f} "
              f"{val['l_delta']:>7.4f}  " + " ".join(f"{x:.2f}" for x in jr))
    print("-" * 92)
    print(f"ACT baseline L_BC = {base:.4f}" if base else "")
    # 增量贡献
    if all(k in rows for k in ["S0", "S1", "S2", "S3"]):
        print("\n[增量] 每加一路输入,hf_ratio 变化(越负越好):")
        print(f"  +state      (S0→S1): {rows['S1']['hf_ratio']-rows['S0']['hf_ratio']:+.3f}")
        print(f"  +hist_action(S1→S2): {rows['S2']['hf_ratio']-rows['S1']['hf_ratio']:+.3f}")
        print(f"  +hist_imu   (S2→S3): {rows['S3']['hf_ratio']-rows['S2']['hf_ratio']:+.3f}")
        print(f"  参数量 S0={rows['S0']['n_params']:,} → S3={rows['S3']['n_params']:,}")
        print("  判读: 若 S0/S1 已接近 S3 → history 可砍(更小网络); 若某路带来明显下降 → 该路值得留。")


if __name__ == "__main__":
    main()
