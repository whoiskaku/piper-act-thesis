#!/usr/bin/env python3
"""j4 扫频的分级 pilot —— **开发/安全数据，不进标定集，也不进检验集。**

为什么扫频要单独 pilot：主套件里 j4 的扫频实测 **53.7 °/s / 1700 °/s²**，是被
判否的 P05（683 °/s²）的 **2.5 倍**、阶跃 pilot 已验证上限（342）的 **5 倍**。
**目视审查看不出加速度**，这条本会直接上机。

为什么不直接套阶跃的界：扫频是**连续激励**，阶跃是**单次瞬态**，失效机理未必
相同（阶跃的失败表现为滞后从 2 帧变 3 帧 + 加减速段残差占 97%）。而简单封到
342 °/s² 后幅值只剩 0.7°、激励掉一个数量级，等于放弃 2–6 Hz 的主力激励。
故单独逐级验证。

分级规则（沿用阶跃 pilot 的原则：**同一级不同时增加幅度与频率**）：
```
阶段 A  固定 f1 = 3 Hz，幅值 0.15 → 0.30 → 0.60 → 1.20    (加速 38→306 °/s²)
阶段 B  幅值锁在阶段 A 证过的值，f1 3 → 4 → 6 Hz
```
每级重复 2 次；按顺序逐条跑，`pilot_j4.py --check` 过关才准跑下一条。

门禁与阶跃 pilot **完全相同**（固定 2 帧滞后 p99 ≤ 0.5° / 稳定位置带 /
最终到位 / 限位 / 速度），一个字不改。

用法:
    python sim/step2_g/pilot_j4_chirp.py --list
    python sim/step2_g/pilot_j4_chirp.py --save
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sim.step2_g.sequences import (                                    # noqa: E402
    HOME_DEG, Seq, check_limits, chirp, place,
)
from sim.step2_g.pilot_j4 import GATE, J, nearest_home_pose            # noqa: E402

MODEL = ROOT / "sim/piper_model.xml"
OUT = ROOT / "outputs/step2_g/pilot_j4_chirp"
REPEATS = 2
PURPOSE = "development/safety pilot (chirp) — NOT for calibration, NOT for validation"
# 阶段 A：固定 f1，只涨幅值。阶段 B：幅值锁在 A 的顶档，只涨 f1。
PHASE_A = [(0.15, 3.0), (0.30, 3.0), (0.60, 3.0), (1.20, 3.0)]
PHASE_B_F1 = [4.0, 6.0]
PHASE_B_AMP = 0.60          # ≤ 阶段 A 已验证；在 f1=6 Hz 下仍 ≤342 °/s²


def ladder(m) -> list[Seq]:
    pose = nearest_home_pose(m)
    out: list[Seq] = []
    proven = {"amp": 0.0}

    def add(tag, phase, amp, f1):
        cmd = place(m, chirp(J, pose, amp, f1=f1), J)
        if phase.startswith("A"):
            proven["amp"] = max(proven["amp"], amp)
        else:
            assert amp <= proven["amp"] + 1e-9, (
                f"{tag}: 幅值 {amp} 超过阶段 A 已验证的 {proven['amp']} "
                f"⇒ 会同时增加幅度与频率，违反分级规则")
        for r in range(REPEATS):
            out.append(Seq(f"{tag}_r{r+1}", phase, (J,), cmd, pose,
                           {"phase": phase, "amp_param": amp, "f1_hz": f1,
                            "rep": r + 1, "proven_amp": round(proven["amp"], 3)}))

    for i, (a, f1) in enumerate(PHASE_A, 1):
        add(f"CH{i:02d}_A_f{f1:.0f}_a{a:.2f}", "A_幅值阶梯", a, f1)
    for i, f1 in enumerate(PHASE_B_F1, len(PHASE_A) + 1):
        add(f"CH{i:02d}_B_f{f1:.0f}_a{PHASE_B_AMP:.2f}", "B_频率阶梯", PHASE_B_AMP, f1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.parse_args()
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    seqs = ladder(m)
    print(f"j4 扫频分级 pilot —— {PURPOSE}")
    print(f"已验证干净区间(来自阶跃 pilot): ≤30.2 °/s / ≤342 °/s²  "
          f"—— 但扫频是连续激励, 该界仅作参考\n")
    print(f"{'#':<3}{'序列':<24}{'阶段':<12}{'f1':>5}{'幅值':>8}{'峰速':>8}"
          f"{'峰加速':>9}{'≤已验证':>9}{'hash':>13}")
    for i, s in enumerate(seqs, 1):
        c = check_limits(m, s.cmd)
        q = np.rad2deg(s.cmd[:, J])
        mark = "✓" if c["v_max_deg"][J] <= 30.2 and c["a_max_deg"][J] <= 342 else "**超**"
        print(f"{i:<3}{s.name:<24}{s.meta['phase']:<12}{s.meta['f1_hz']:>5.0f}"
              f"{q.ptp():>7.3f}°{c['v_max_deg'][J]:>8.1f}{c['a_max_deg'][J]:>9.0f}"
              f"{mark:>9}{s.sha:>13}")
    print(f"\n执行: 按 # 顺序**逐条**跑; 每条跑完用 "
          f"`pilot_j4.py --check` 评估, 未过关**不得**跑下一条。")
    print(f"同一级**不同时**增加幅度与频率 —— 阶段 A 只涨幅值, 阶段 B 才涨 f1。")
    if len(sys.argv) > 1 and "--save" in sys.argv:
        OUT.mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(seqs, 1):
            np.savez(OUT / f"{s.name}.npz", cmd7=s.cmd.astype(np.float32),
                     da=np.zeros((len(s.cmd), 6), np.float32), tag=s.name,
                     order=i, purpose=PURPOSE, sha=s.sha)
        (OUT / "manifest.json").write_text(json.dumps(
            {"purpose": PURPOSE, "joint": "j4", "primitive": "chirp",
             "why_separate_pilot": "主套件 j4 扫频实测 53.7°/s / 1700°/s², "
                                   "为被判否 P05(683) 的 2.5 倍、已验证上限(342)的 5 倍",
             "gate": GATE,
             "order": [{"n": i, "name": s.name, "sha": s.sha, **s.meta}
                       for i, s in enumerate(seqs, 1)]}, ensure_ascii=False, indent=1))
        print(f"\n→ {OUT.relative_to(ROOT)}/  ({len(seqs)} 个 npz + manifest.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
