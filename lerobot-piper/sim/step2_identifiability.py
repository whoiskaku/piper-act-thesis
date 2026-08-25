#!/usr/bin/env python3
"""Step 2 可辨识性分析 —— 上机器**之前**先问「这些参数分得开吗」。

动机: 现有 17 集 pp2 的灵敏度分析里, 30 个参数有 10 个几乎不可辨识
(灵敏度 <0.5%), 且 b↔kv 六个关节全部共线到 |cos|>0.995。加更多**同类**数据
不会改善 —— 那是激励不足, 不是样本量不足。

本脚本回答两个结构性问题(纯计算, 零机时):

  ① **哪些关节的转子惯量 Ia 能靠位形变化来检验?**
     绕 j 轴的等效惯量 M[j,j] 只取决于**比 j 更远端**的关节姿态。
     故远端关节越少, I_link(q) 的可变范围越小; j6 没有远端关节 ⇒ I_link 恒定
     ⇒ Ia6 与 I_link6 **原理上不可分离**(除非外加已知负载)。

  ② **现有 probe_c 的三个位形够用吗?**
     不够 —— 实测三者的 I_link@j3 **完全相同**。因为它们只改了 j2/j3(近端),
     而 j4/j5/j6 全为 0。这不是「位形不够多」, 是「改错了关节」。

用法: python sim/step2_identifiability.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "sim/piper_model.xml"
PROBE_C = ROOT / "outputs/probe_c"
N_SAMPLE = 4000
SEED = 0


def link_inertia(m, d, M, q6: np.ndarray) -> np.ndarray:
    """给定位形下各关节的等效连杆惯量 diag(M)[:6]（armature=0 时不含 Ia）。"""
    d.qpos[:6] = q6
    d.qpos[6] = d.qpos[7] = 0.0
    mujoco.mj_forward(m, d)
    mujoco.mj_fullM(m, M, d.qM)
    return np.diag(M)[:6].copy()


def main() -> int:
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    assert np.allclose(m.dof_armature, 0), "模型的 armature 非零 —— diag(M) 会含 Ia"
    d = mujoco.MjData(m)
    M = np.zeros((m.nv, m.nv))

    # ── ② 现有三位形够不够 ────────────────────────────────────────
    print("① 现有 probe_c 的三个位形对 j3 的惯量")
    tags = ("p289", "p56", "p753")
    vals = []
    for t in tags:
        f = PROBE_C / f"replay_c_{t}_j3_A1_c{t[1:]}_3.json"
        if not f.exists():
            print(f"   （缺 {f.name}，跳过）")
            continue
        c = np.asarray(json.loads(f.read_text())["cmd"], float)
        q = c[len(c) // 2, :6]
        I = link_inertia(m, d, M, q)
        vals.append(I[2])
        print(f"   {t:<6} j1..j6 = {np.round(np.rad2deg(q), 1)}   I_link@j3 = {I[2]:.5f}")
    if vals:
        v = np.array(vals)
        print(f"   极差/均值 = **{(v.max()-v.min())/v.mean()*100:.1f}%**"
              f"  ⇒ {'❌ 三位形对 j3 完全等价, 分不出 Ia3' if v.ptp()/v.mean() < 0.01 else '有变化'}")
        print("   原因: 三者只改 j2/j3（**近端**），而 j4/j5/j6 全为 0。\n")

    # ── ① 每个关节的可达惯量范围 ──────────────────────────────────
    rng = np.random.default_rng(SEED)
    Q = rng.uniform(m.jnt_range[:6, 0], m.jnt_range[:6, 1], size=(N_SAMPLE, 6))
    I = np.array([link_inertia(m, d, M, q) for q in Q])
    print(f"② 各关节 I_link 在全工作空间（{N_SAMPLE} 次随机位形）的可达范围")
    print(f"{'关节':<6}{'最小':>10}{'最大':>10}{'极差/均值':>11}{'倍数':>8}{'远端':>6}   结论")
    verdict = {}
    for j in range(6):
        r = (I[:, j].max() - I[:, j].min()) / I[:, j].mean()
        ratio = I[:, j].max() / max(I[:, j].min(), 1e-12)
        ok = "✅ 可检验" if r > 0.3 else ("⚠ 勉强" if r > 0.05 else "❌ 不可分离")
        verdict[j] = r
        print(f"j{j+1:<5}{I[:, j].min():>10.5f}{I[:, j].max():>10.5f}{r*100:>10.1f}%"
              f"{ratio:>7.1f}×{5-j:>6}   {ok}")

    print("\n③ 对参数集的直接后果")
    good = [f"j{j+1}" for j in range(6) if verdict[j] > 0.3]
    bad = [f"j{j+1}" for j in range(6) if verdict[j] <= 0.05]
    print(f"   · {', '.join(good)}: Ia 可与 I_link 分离检验 ⇒ 可独立拟合")
    print(f"   · {', '.join(bad)}: I_link 几乎恒定 ⇒ Ia 与 I_link **结构性混淆**")
    print(f"     ⇒ 不要假装分开。对这些关节应拟合**可辨识的组合** I_eff = I_link + Ia，")
    print(f"       并明确声明未分离（或用已知外加负载专门打破简并）。")
    print("\n④ 激励设计的三条硬约束（由本分析导出）")
    print("   · 激励关节 j 时必须变的是**远端** j+1..6，改近端无效")
    print("   · j4/j5/j6 在 pp2 与 probe_c 里**从未被激励** ⇒ 其全部参数不可辨识")
    print("   · 位形变化**不能**打破 b↔kv 共线（实测跨三位形仍 |cos|=1.000）；")
    print("     打破它要靠**速度剖面**：恒速段 q̇≈q̇_des ⇒ kv≈0、b 主导；")
    print("     快瞬态 q̇≠q̇_des ⇒ kv 起作用。两者都要有。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
