#!/usr/bin/env python3
"""**FF 之后的墙**：G-2 对墙的预测能力，在 C 臂真正面对的工况下还成立吗？（零机时）

为什么必须先查这个
------------------
G-2 预测墙的 52.1% 是在 **ab4 裸 ACT** 上测的，但 **C 臂的 π 实际修的是
「FF 之后」的墙** —— 它比裸墙小约 33%（FF 实测 −33.0%）且**形状不同**。
墙变小 ⇒ 同样的绝对误差占比更高 ⇒ 预测**相对**更难。

三臂对照（把「FF 的影响」与「换会话的影响」分开）
------------------------------------------------
| 臂 | 会话 | 指令 | 用途 |
|---|---|---|---|
| `ab4 P0` | ab4 | 裸 ACT | 已测 **52.1%**（背景） |
| **`A`** | ab2 | 裸 ACT | 隔离**换会话**的影响 |
| **`B3b13`** | ab2 | **FF** | 隔离 **FF** 的影响 ← C 臂的真实工况 |

口径（写死）
------------
- `ideal` = `calibrate(render(chunks[].raw))` —— 从**原始 ACT 指令**（FF/δa 之前）
  逐 chunk 渲染，与 `FF_TUNING_FINAL` 同。
- **G-2 的输入 = 实际下发的指令**（帧里的 `a`，**FF 之后**）——
  它预测的是机器人实际会怎么动。实测 `a` 与 `chunks[].raw` 最大差 0.131 rad ⇒ 确是 FF 后。
- 标定 `R,b` 用 **`outputs/p2_labels/pp2-deploy/manifest.json`** ——
  这是当初 FF 分析（`analyze_AD.py`）**既有的**选择，非事后另挑。
  （ab4/step1 用的是 `pp2a-deploy`，不同会话各有标定。）
- `ALIGN_L=3`（与 G-2 预测墙的工作同约定；`FF_TUNING_FINAL` 用的是未对齐口径，
  故其 −33.0% **不可与本表并列**）。
- 剔除不完整集（帧数 < 550）。

⚠ 本脚本只回答**预测能力**是否在 FF 之后仍成立；
交付量与部署冻结是下一步。

用法:
    python -u sim/g2_wall_after_ff.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "sim/step4_gvnext"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fit_step1_candidate as S1                                    # noqa: E402
import g2_on_ab4 as G                                               # noqa: E402
import gain_G_vs_H as GH                                            # noqa: E402

ALIGN_L = S1.ALIGN_L
BANDS = ((0.5, 3.0), (3.0, 9.0), (0.5, 9.0))
MINF = 550
CAL = ROOT / "outputs/p2_labels/pp2-deploy/manifest.json"
OUT = ROOT / "outputs/step1_H/g2_wall_after_ff.json"


def run_arm(paths, R32, b32, rend, g2, tag):
    acc = {bd: [0.0, 0.0, 0.0, 0.0] for bd in BANDS}
    wall = 0.0
    nep = 0
    for p in paths:
        pay = json.loads(p.read_text())
        fr = pay.get("frames", [])
        if len(fr) < MINF:
            print(f"    ⚠ 剔除不完整集 {p.stem}（{len(fr)} 帧 < {MINF}）")
            continue
        a = np.asarray([f["a"] for f in fr], float)
        if a.ndim == 3:
            a = a[:, 0]
        real = np.asarray([f["imu"] for f in fr], float)[:, 3:6]
        _, ideal_sim, _ = S1._render_logged_chunks(pay.get("chunks", []), rend, len(fr))
        ideal = S1.calibrate_ideal_imu(ideal_sim.astype(np.float32), R32, b32)[:, 3:6]
        # G-2 吃**实际下发**的指令（FF 之后）
        cmd = np.zeros((len(fr), 7)); cmd[:, :min(7, a.shape[1])] = a[:, :min(7, a.shape[1])]
        t = np.arange(len(fr)) / 20.0
        dd = {"name": p.stem, "cmd": cmd, "t_send": t, "t_read": t,
              "tg_i": np.arange(len(fr) * 5) / 100.0,
              "gyro100": np.repeat(real, 5, axis=0)}
        w = g2.omega(dd)
        w6 = np.zeros((len(w), 6), np.float32); w6[:, 3:] = w
        ws = S1.calibrate_ideal_imu(w6, R32, b32)[:, 3:]
        n = min(len(real) - ALIGN_L, len(ideal), len(ws) - ALIGN_L)
        e0 = real[ALIGN_L:ALIGN_L + n] - ideal[:n]
        e0h = ws[ALIGN_L:ALIGN_L + n].astype(float) - ideal[:n].astype(float)
        for bd in BANDS:
            x, y = GH.bp(e0, *bd), GH.bp(e0h, *bd)
            acc[bd][0] += float(np.square(x - y).sum())
            acc[bd][1] += float(np.square(x).sum())
            acc[bd][2] += float((x * y).sum())
            acc[bd][3] += float(np.square(y).sum())
        wall += float(np.square(GH.bp(e0, 0.5, 9.0)).sum())
        nep += 1
    out = {}
    for bd in BANDS:
        e_, d_, dot, nb = acc[bd]
        out[f"{bd[0]}-{bd[1]}"] = [1 - e_ / max(d_, 1e-30),
                                   dot ** 2 / max(d_ * nb, 1e-30)]
    return out, nep, wall


def main() -> int:
    import gc
    gc.disable()
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    man = json.loads(CAL.read_text())
    R32 = np.asarray(man["R"], np.float32)
    b32 = np.asarray(man.get("b", [0, 0, 0]), np.float32)
    rend = IdealChunkRenderer()
    g2 = G.G2()
    ab2 = ROOT / "outputs/ab2"
    arms = {"A（ab2·裸 ACT）": sorted(ab2.glob("A_ep*.json")),
            "**B3b13（ab2·FF）**": sorted(ab2.glob("B3b13_ep*.json"))}
    print(f"标定 = {CAL.relative_to(ROOT)}（当初 FF 分析既有的选择）· ALIGN_L={ALIGN_L}")
    print(f"G-2 输入 = **实际下发指令**（FF 之后）· ideal = 从 chunks[].raw 渲染")
    print(f"背景：ab4 P0（裸 ACT，另一会话）已测 **52.1%**\n")

    res, walls = {}, {}
    for tag, paths in arms.items():
        print(f"  {tag}")
        r, nep, wall = run_arm(paths, R32, b32, rend, g2, tag)
        res[tag.replace("*", "")] = r
        walls[tag.replace("*", "")] = wall / max(nep, 1)
        for bd, v in r.items():
            print(f"    {bd:<10}原始解释率 {v[0]*100:>7.1f}%   最优标量后 ρ² {v[1]*100:>7.1f}%")
        print(f"    （{nep} 集）\n")

    k = list(res)
    a_, f_ = res[k[0]]["0.5-9.0"], res[k[1]]["0.5-9.0"]
    wa, wf = walls[k[0]], walls[k[1]]
    print(f"  ── 判读 ──")
    print(f"  墙的能量：裸 ACT {wa:.4g} → FF 后 {wf:.4g}"
          f"（{(wf/max(wa,1e-30)-1)*100:+.0f}%）")
    print(f"  G-2 解释率：裸 ACT(ab2) {a_[0]*100:.1f}%  →  **FF 后 {f_[0]*100:.1f}%**"
          f"（{(f_[0]-a_[0])*100:+.1f} pp）")
    print(f"  换会话的影响：ab4 裸 52.1% → ab2 裸 {a_[0]*100:.1f}%"
          f"（{(a_[0]-0.521)*100:+.1f} pp）")
    print(f"\n  ⚠ `FF_TUNING_FINAL` 的 −33.0% 是**未对齐口径**，与本表（ALIGN_L=3）**不可并列**。")
    OUT.write_text(json.dumps(
        {"arms": res, "wall_energy_mean": walls, "align_L": ALIGN_L,
         "calib": str(CAL), "ab4_bare_reference_percent": 52.1,
         "metrics_order": ["raw_explained_fraction", "rho2_after_optimal_scalar"],
         "caliber": ("ideal 从 chunks[].raw(FF/δa 之前)渲染; G-2 输入 = **实际下发指令**"
                     "(FF 之后); 标定用 pp2-deploy manifest(当初 FF 分析既有选择, 非事后另挑); "
                     "ALIGN_L=3(FF_TUNING_FINAL 的 -33.0% 是未对齐口径, **不可并列**); "
                     "剔除帧数<550 的不完整集; ab2 无时间戳⇒假设均匀 20Hz"),
         }, indent=1, ensure_ascii=False))
    print(f"\n  → {OUT}")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
