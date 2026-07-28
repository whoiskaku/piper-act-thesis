#!/usr/bin/env python3
"""A vs D 配对分析 —— **论文主张的那个对比**（2026-07-29）。

    A = 裸 ACT (mode=off)          D = ACT + π (mode=pi_only, **FF 关闭**)
    D − A = π 的净效果

为什么重做：WORKLOG §10 的 C 臂是 **FF+π**，实测 FF : δa_π = 9.8 : 1，
π 只占施加修正能量的 **1.0%**；且 `full` 模式下 ideal 从 raw+FF 渲染，
与 π 的训练口径（从 raw 渲染）**失配 13.6 mrad**。C−A 因此无法归因到 π。
`pi_only` 两者都从 raw 渲染，与训练一致。

块跑（非交替）的代价：A 与 D 间隔约 10 分钟，温漂/磨损整体落在 D 上。
故**同时报臂内前后半差**作为漂移尺度 —— D−A 若小于它就不能宣称效果。

用法: python sim/analyze_AD.py [--root outputs/ab2]
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

from analyze_ab import load_run

ROOT = Path(__file__).resolve().parents[1]


def full_eps(root, arm, min_frames=500):
    """只取完整集（中断重跑的短集剔除），按 ep 序号排序。"""
    out = []
    for f in sorted(root.glob(f"{arm}_ep*.json"),
                    key=lambda p: int(re.search(r"ep(\d+)", p.name).group(1))):
        s = json.loads(f.read_text())["summary"]
        if s["frames"] > min_frames:
            out.append((int(re.search(r"ep(\d+)", f.name).group(1)), f, s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/ab2")
    ap.add_argument("--calib", default="outputs/p2_labels/pp2-deploy/manifest.json")
    a = ap.parse_args()
    root = ROOT / a.root
    R = np.array(json.loads((ROOT / a.calib).read_text())["R"], np.float32)

    arms = {}
    for arm in ("A", "D"):
        eps = full_eps(root, arm)
        arms[arm] = eps
        s0 = eps[0][2]
        print(f"  {arm}: {len(eps)} 完整集 {[e for e, _, _ in eps]}   "
              f"mode={s0['mode']} pi={s0['use_pi']} ff={s0['use_ff']} "
              f"ckpt={Path(str(s0['pi_ckpt'])).name}")
    assert not arms["D"][0][2]["use_ff"], "✗ D 臂 ff=on —— 跑成 FF 臂了"
    n = min(len(arms["A"]), len(arms["D"]))
    print(f"\n  按采集次序配对 {n} 对（初始位姿非显式种子控制，次序是唯一可用对应）\n")

    E, DET = {}, {}
    for arm in ("A", "D"):
        vals, dets = [], []
        for _, f, s in arms[arm][:n]:
            r = load_run(f, R)
            # **每帧** gap 能量 —— 集长 560-596 不等, 整集求和会被长度带偏
            # (§10.1 报的也是每帧: A 0.0555 / C 0.0583)
            vals.append(r["energy"] / r["n"]); dets.append(r)
        E[arm] = np.array(vals); DET[arm] = dets
        print(f"  {arm} 逐集 gap 能量: {np.round(E[arm], 4).tolist()}")

    print(f"\n  {'臂':>4}{'gap 能量/帧 均值':>18}{'中位':>12}{'相对 A':>10}")
    for arm in ("A", "D"):
        print(f"  {arm:>4}{E[arm].mean():>18.4g}{np.median(E[arm]):>12.4g}"
              f"{E[arm].mean()/E['A'].mean()*100-100:>9.1f}%")

    d = E["A"] - E["D"]                       # 正 = D 更好（gap 更小）
    t, p = stats.ttest_rel(E["A"], E["D"])
    rel = d.mean() / E["A"].mean() * 100
    ci = stats.t.interval(0.95, n - 1, loc=d.mean(),
                          scale=stats.sem(d)) if n > 1 else (np.nan, np.nan)
    print(f"\n  **D−A（π 的净效果）= {rel:+.1f}%**   t={t:+.2f}  p={p:.4f}  "
          f"{'✓ 显著' if p < 0.05 else '✗ 不显著'}")
    print(f"    配对差 95%CI = [{ci[0]:.4g}, {ci[1]:.4g}]  "
          f"（相对 {ci[0]/E['A'].mean()*100:+.1f}% … {ci[1]/E['A'].mean()*100:+.1f}%）")
    print(f"    {int((d > 0).sum())}/{n} 对 D 更好")

    print(f"\n  【块跑的漂移尺度】臂内前后半差（前 {n//2} vs 后 {n-n//2} 集）")
    for arm in ("A", "D"):
        h1, h2 = E[arm][:n // 2].mean(), E[arm][n // 2:].mean()
        print(f"    {arm}: {h1:.4g} → {h2:.4g}   ({(h2-h1)/h1*100:+.1f}%)")
    drift = max(abs((E[arm][n//2:].mean() - E[arm][:n//2].mean())
                    / E[arm][:n//2].mean() * 100) for arm in ("A", "D"))
    print(f"    最大臂内漂移 = {drift:.1f}%   vs  D−A = {abs(rel):.1f}%")
    print(f"    ⇒ {'**D−A 超过漂移尺度**' if abs(rel) > drift else '**D−A 未超过漂移尺度, 不能宣称效果**'}")

    (ROOT / "outputs/ab2/AD_result.json").write_text(json.dumps(
        {"n_pairs": n, "A": E["A"].tolist(), "D": E["D"].tolist(),
         "rel_pct": rel, "t": float(t), "p": float(p),
         "ci": [float(ci[0]), float(ci[1])], "drift_pct": float(drift)},
        indent=1, ensure_ascii=False))
    print(f"\n  【副判据】")
    for arm in ("A", "D"):
        d_ = DET[arm]
        print(f"    {arm}: gap_rms {np.mean([x['gap_rms'] for x in d_]):.4f}  "
              f"‖δa‖ {np.mean([x['da_rms'] for x in d_]):.2f} mrad  "
              f"峰值 {max(x['da_peak'] for x in d_):.2f} mrad  "
              f"帧数 {np.mean([x['n'] for x in d_]):.0f}")
    print(f"\n→ outputs/ab2/AD_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
