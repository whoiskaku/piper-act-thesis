#!/usr/bin/env python3
"""**G-2 的绝对预测能力能否作用于 π 真正的瓶颈：预测墙 e₀？**（零机时）

为什么问这个
------------
约束天花板实验把瓶颈定死了（同一求解器 + 同一部署约束）：

    实际预测器（指令+IMU+编码器，线性）   5.08%
    **完美预测 e₀**                    **22.07%**

⇒ 瓶颈是「**预测墙**」，不是「算修正」。而预测 e₀ 是一个**绝对**问题：

    ê₀[t] = calib(ω̂_G2)[t+ALIGN_L] − ideal[t]

`ideal` 从指令精确渲染、已知；`ω̂_G2` 是 G-2 的**绝对**输出 —— 正是它的强项
（0.5–3 Hz ρ²=0.836 / 3–9 Hz 0.520）。**而 G-2 在 π 管线里从未承担过预测角色。**

⚠ 一个化简（决定了这个检验的本质）
----------------------------------
    ê₀ − e₀ = calib(ω̂_G2)[t+3] − real[t+3]

即**预测 e₀ 的误差 == G-2 预测 `real` 的绝对误差**。
⇒ 本检验实质是问：**G-2 的绝对误差，比墙 `e₀` 本身小还是大？**
分母是 ‖e₀‖²（墙），不是 ‖real‖² —— 墙远小于 real，所以解释率**可以是很负的**。

对照
----
`STEP1_RESULT_v1` §2.1：线性 `W`（36 维指令特征 × 16 帧窗）在留出集上
**能预测 gap 能量的 31.60%**。同一批数据、同一频带 ⇒ 可直接比较。

口径（写死）
------------
- 数据 = **ab4 P0**（无探针 ⇒ `e₀ = gap`），逐集报告 + 汇总
- `ALIGN_L=3` 与标定 `(R,b)` **事前固定**，**不扫位移、不重拟旋转**
- G-2 输出用**与 ideal 同一个** `calibrate_ideal_imu` 转入真机传感器系（不自己另写旋转）
- 频带 0.5–3 / 3–9 / 0.5–9 Hz 分别报
- 两个数：**原始解释率** `1−‖ê₀−e₀‖²/‖e₀‖²` 与**最优标量后**的解释率（= ρ²）
- ⚠ ab4 无时间戳 ⇒ G 前向在**假设均匀 20 Hz** 下跑

判读
----
- 明显高于 31.6% ⇒ G-2 的绝对保真度**直接打在瓶颈上**，值得建这条路
- 不比 31.6% 好 ⇒ 绝对保真度对预测墙也没帮助 —— 那才是主线真正要面对的坏消息

用法:
    python -u sim/g2_predicts_wall.py
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
W_BASELINE = 31.6                     # STEP1 §2.1，同数据同频带
OUT = ROOT / "outputs/step1_H/g2_predicts_wall.json"


def main() -> int:
    import gc
    gc.disable()
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    calib = json.loads(
        (ROOT / "outputs/step1_H/step1_H_v1.json").read_text())["calibration"]
    R = np.asarray(calib["R"], np.float32)
    b = np.asarray(calib.get("b", [0, 0, 0]), np.float32)
    rend = IdealChunkRenderer()
    g2 = G.G2()
    logs = S1.discover_logs(ROOT / "outputs/ab4", ("P0",))
    print(f"ab4 P0 {len(logs['P0'])} 集（无探针 ⇒ e₀ = gap）· "
          f"ALIGN_L={ALIGN_L} 与 (R,b) 固定，不扫位移/不重拟旋转")
    print(f"⚠ ab4 无时间戳 ⇒ G 前向假设均匀 20 Hz")
    print(f"对照：线性 W 在留出集上能预测 gap 能量的 **{W_BASELINE}%**"
          f"（STEP1 §2.1）\n")

    acc = {bd: [0.0, 0.0, 0.0, 0.0] for bd in BANDS}    # Σerr², Σe0², Σ<e0,ê0>, Σê0²
    rows = {}
    for p in logs["P0"]:
        ep = S1.load_episode(p, "P0", np.asarray(R, np.float64),
                             np.asarray(b, np.float64), rend, 550)
        pay = json.loads(p.read_text())
        frames = pay["frames"]
        _, ideal_sim, _ = S1._render_logged_chunks(pay.get("chunks", []), rend, len(frames))
        ideal = S1.calibrate_ideal_imu(ideal_sim.astype(np.float32), R, b)
        dd = G.ab4_to_dd(p, p.stem)
        w = g2.omega(dd)                                  # 机体系，(T,3)
        # 用**与 ideal 同一个**标定函数转入真机传感器系（补 3 个加速度通道占位）
        w6 = np.zeros((len(w), 6), np.float32); w6[:, 3:] = w
        w_sensor = S1.calibrate_ideal_imu(w6, R, b)[:, 3:]
        n = len(ep.gap)
        m = min(n, len(w_sensor) - ALIGN_L, len(ideal))
        e0 = np.asarray(ep.gap, float)[:m]
        e0h = w_sensor[ALIGN_L:ALIGN_L + m].astype(float) - ideal[:m, 3:6].astype(float)
        per = {}
        for bd in BANDS:
            a_, b_ = GH.bp(e0, *bd), GH.bp(e0h, *bd)
            err = float(np.square(a_ - b_).sum()); den = float(np.square(a_).sum())
            dot = float((a_ * b_).sum()); nb = float(np.square(b_).sum())
            acc[bd][0] += err; acc[bd][1] += den; acc[bd][2] += dot; acc[bd][3] += nb
            per[f"{bd[0]}-{bd[1]}"] = [1 - err / max(den, 1e-30),
                                       dot ** 2 / max(den * nb, 1e-30)]
        rows[p.stem] = per
    print(f"  {'集':<9}{'带':<10}{'原始解释率':>12}{'最优标量后(ρ²)':>16}")
    for k, per in rows.items():
        for i, (bd, v) in enumerate(per.items()):
            print(f"  {k if i == 0 else '':<9}{bd:<10}{v[0]*100:>11.1f}%{v[1]*100:>15.1f}%")
    print()
    agg = {}
    for bd in BANDS:
        e_, d_, dot, nb = acc[bd]
        agg[f"{bd[0]}-{bd[1]}"] = [1 - e_ / max(d_, 1e-30),
                                   dot ** 2 / max(d_ * nb, 1e-30)]
    print(f"  {'**汇总**':<9}")
    for bd, v in agg.items():
        print(f"  {'':<9}{bd:<10}{v[0]*100:>11.1f}%{v[1]*100:>15.1f}%")

    key = "0.5-9.0"
    raw, rho = agg[key][0] * 100, agg[key][1] * 100
    print(f"\n  ── 判读 ──")
    print(f"  汇总 0.5–9 Hz：原始解释率 **{raw:.1f}%** · 最优标量后 ρ² **{rho:.1f}%**")
    print(f"  对照 线性 W：**{W_BASELINE}%**")
    if raw > W_BASELINE:
        print(f"  ⇒ **G-2 的绝对保真度直接打在瓶颈上**（{raw:.1f}% > {W_BASELINE}%），值得建这条路")
    else:
        print(f"  ⇒ **不比 W 好**（{raw:.1f}% vs {W_BASELINE}%）—— "
              f"绝对保真度对预测墙也没帮助。")
        print(f"     注：原始解释率为负表示 **G-2 的绝对误差比墙 e₀ 本身还大**；"
              f"\n     最优标量后的 ρ²={rho:.1f}% 才是「方向上有多少信息」的读数。")
    OUT.write_text(json.dumps(
        {"per_episode": rows, "aggregate": agg, "w_baseline_percent": W_BASELINE,
         "metrics_order": ["raw_explained_fraction", "rho2_after_optimal_scalar"],
         "caliber": ("ab4 P0(无探针⇒e0=gap); ALIGN_L=3 与 (R,b) **事前固定**, "
                     "不扫位移/不重拟旋转; G-2 输出用**与 ideal 同一个** "
                     "calibrate_ideal_imu 转入真机传感器系; "
                     "**ê0−e0 == G-2 对 real 的绝对误差**, 分母是 ‖e0‖²(墙)而非 ‖real‖² "
                     "⇒ 解释率可以为负; ab4 无时间戳⇒假设均匀 20Hz"),
         }, indent=1, ensure_ascii=False))
    print(f"\n  → {OUT}")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
