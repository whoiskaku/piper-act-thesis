#!/usr/bin/env python3
"""Stage 3b — λ_hf Pareto sweep 汇总。

读取 phase1_runB_hf*_dl0.1_* 各 run 的 history.json，按拍板的选择规则挑 ckpt：
  primary  : mean hf_ratio (refined/act) 越低越好
  guardrail: ① L_BC_refined ≤ 1.1 × ACT_L_BC_baseline
             ② mean |ΔA| 不爆（< DELTA_CAP）
             ③ per-joint 公平：任一 active joint 的 hf_ratio 不得 ≥ 1.0
                （即不许牺牲某个关节来换平均值）

输出 Pareto 表：λ_hf | sel_ep | hf_ratio | L_BC | mean|ΔA| | j1/j2/j3/grip hf_ratio

用法:  python sim/sweep_pareto.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_GLOB = "phase1_runB_hf*_dl0.1_lr0.0003_bs64_ep200"
TRAIN_DIR = REPO_ROOT / "outputs/train/refine_net"

DELTA_CAP = 0.02       # mean |ΔA| 上限 (rad)；远大于 Run B 的 0.003
BC_TOL = 1.1           # L_BC ≤ 1.1 × baseline
JOINT_RATIO_CAP = 1.0  # 任一 active joint hf_ratio 须 < 1.0
JOINT_NAMES = ["j1", "j2", "j3", "grip"]


def select_ckpt(history):
    """Return (epoch, val_stats) of best-by-rule epoch, or (None, reason)."""
    best = None
    n_bc_fail = n_delta_fail = n_joint_fail = 0
    for h in history:
        v = h["val"]
        if v["l_bc"] > BC_TOL * v["l_bc_act"]:
            n_bc_fail += 1; continue
        if v["l_delta"] > DELTA_CAP:
            n_delta_fail += 1; continue
        jratios = [r / (a + 1e-12)
                   for r, a in zip(v["hf_refined_joints"], v["hf_act_joints"])]
        if max(jratios) >= JOINT_RATIO_CAP:
            n_joint_fail += 1; continue
        if best is None or v["hf_ratio"] < best[1]["hf_ratio"]:
            best = (h["epoch"], v, jratios)
    if best is None:
        return None, f"no epoch passed (bc_fail={n_bc_fail} delta_fail={n_delta_fail} joint_fail={n_joint_fail})"
    return best, None


def main():
    run_dirs = sorted(TRAIN_DIR.glob(RUN_GLOB))
    if not run_dirs:
        print(f"[err] no runs match {RUN_GLOB} under {TRAIN_DIR}")
        sys.exit(1)

    rows = []
    for d in run_dirs:
        cfg = json.loads((d / "config.json").read_text())
        lam_hf = cfg["lambda_hf"]
        history = json.loads((d / "history.json").read_text())
        sel, reason = select_ckpt(history)
        final = history[-1]["val"]
        if sel is None:
            rows.append((lam_hf, None, final, None, reason))
        else:
            ep, v, jr = sel
            rows.append((lam_hf, ep, v, jr, None))

    rows.sort(key=lambda r: r[0])
    print(f"\n{'='*92}")
    print("Stage 3b — λ_hf Pareto sweep  (λ_delta=0.1, λ_acc=0, 200ep)")
    print(f"  ACT baseline: L_BC={rows[0][2]['l_bc_act']:.4f}, hf_act(mean)={rows[0][2]['hf_act']:.4f}")
    print(f"  规则: hf_ratio↓ | L_BC≤1.1×baseline | mean|ΔA|<{DELTA_CAP} | 每个active joint ratio<{JOINT_RATIO_CAP}")
    print(f"{'='*92}")
    hdr = f"{'λ_hf':>5} {'sel_ep':>6} {'hf_ratio':>9} {'L_BC':>7} {'mean|ΔA|':>9}  " + \
          " ".join(f"{n:>6}" for n in JOINT_NAMES)
    print(hdr)
    print("-" * 92)
    for lam_hf, ep, v, jr, reason in rows:
        if ep is None:
            print(f"{lam_hf:>5} {'--':>6} {'GUARDRAIL FAIL':>9}  {reason}")
            # 仍打印 final 供参考
            jr_f = [r/(a+1e-12) for r, a in zip(v['hf_refined_joints'], v['hf_act_joints'])]
            print(f"{'':>5} {'(fin)':>6} {v['hf_ratio']:>9.3f} {v['l_bc']:>7.4f} {v['l_delta']:>9.4f}  "
                  + " ".join(f"{x:>6.3f}" for x in jr_f))
        else:
            print(f"{lam_hf:>5} {ep:>6} {v['hf_ratio']:>9.3f} {v['l_bc']:>7.4f} {v['l_delta']:>9.4f}  "
                  + " ".join(f"{x:>6.3f}" for x in jr))
    print("-" * 92)
    # 推荐：通过 guardrail 且 hf_ratio 最低的稳点
    ok = [(lh, ep, v, jr) for lh, ep, v, jr, rs in rows if ep is not None]
    if ok:
        best = min(ok, key=lambda r: r[2]["hf_ratio"])
        print(f"[best-by-rule] λ_hf={best[0]}  ep{best[1]}  "
              f"hf_ratio={best[2]['hf_ratio']:.3f}  L_BC={best[2]['l_bc']:.4f}  "
              f"max-joint-ratio={max(best[3]):.3f}")


if __name__ == "__main__":
    main()
