#!/usr/bin/env python3
"""接缝超出是真的，还是渲染伪影？（2026-07-29）

`ff_transfer_offline.py` ② 显示 chunk 开头 10 帧的 gap² 是中段的 5.9×(pp2)/2.2×(insert)。
但 `IdealChunkRenderer` **逐 chunk** 渲染，每段末尾的二阶导只能用自然边界条件
(见 build_dagger_labels.kin_ideal_chunked 的注释) —— 这本身就会在接缝处抬高 ideal 的
误差，与真机跟不跟得上无关。**必须把这两者分开**，否则 blend 的收益空间被高估。

判别法：同一批数据，ideal 换成**整集样条**(kin_ideal_20，无接缝)再算同一张剖面。
  · 整集渲染下接缝超出**仍在** ⇒ ACT 指令在接缝处真的跳，被控对象跟不上 —— blend 有靶
  · 整集渲染下接缝超出**消失** ⇒ 之前看到的是渲染伪影 —— blend 在打影子

注意整集样条**用到了未来数据**，只能用于本诊断，不能用于任何训练/部署口径。

用法: python sim/ff_seam_confound.py [--eps 6]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parents[1]
CHUNK, SMAX = 100, 6
TAGS = [("pp2-deploy", "pp2"), ("ins-deploy", "insert")]


def profile(ideal, real, bp, s):
    n = len(ideal) - SMAX
    e = (bp(real[s:s + n] - ideal[:n]) ** 2).sum(1)
    pos = np.arange(n) % CHUNK
    return np.array([e[pos == p].mean() for p in range(CHUNK)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, default=6)
    a_ = ap.parse_args()
    from analyze_ab import _bp
    from build_dagger_labels import kin_ideal_chunked
    from compare_ideal_methods import kin_ideal_20
    bp = _bp()

    print("chunk 内位置剖面：逐chunk渲染(部署口径) vs 整集样条(仅诊断)")
    print(f"  {'任务':>8}{'渲染':>12}{'开头10帧':>11}{'中段30-90':>12}{'倍率':>8}"
          f"{'末尾3帧':>10}{'倍率':>8}")
    for tag, name in TAGS:
        man = json.loads((ROOT / "outputs/p2_labels" / tag / "manifest.json").read_text())
        R = np.array(man["R"], np.float32)
        acc = {"逐chunk": [], "整集样条": []}
        for ep in range(a_.eps):
            kc, d, _ = kin_ideal_chunked(man["repo"], ep)
            kf, _d2, _ = kin_ideal_20(man["repo"], ep)
            T = min(len(d["imu_real"]), len(kc), len(kf))
            real = d["imu_real"][:T, 3:].astype(np.float32)
            for key, k in (("逐chunk", kc), ("整集样条", kf)):
                acc[key].append(profile((k[:T, 3:] @ R.T).astype(np.float32), real, bp, 3))
        for key in acc:
            p = np.mean(acc[key], 0)
            h, m, t = p[:10].mean(), p[30:90].mean(), p[-3:].mean()
            print(f"  {name:>8}{key:>12}{h:>11.5f}{m:>12.5f}{h/m:>8.2f}×"
                  f"{t:>10.5f}{t/m:>8.2f}×")
    print("\n  判读：整集样条下倍率若仍 ≫1 ⇒ 接缝超出是真的(指令跳/跟不上)，blend 有靶；"
          "\n        若塌到 ~1 ⇒ 之前看到的主要是逐chunk渲染的自然边界伪影。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
