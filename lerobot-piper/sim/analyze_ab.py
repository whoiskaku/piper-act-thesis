#!/usr/bin/env python3
"""③b 三臂 A/B 分析（2026-07-22）— 配对检验 + 第③幕图。

三臂: A 裸ACT / B 逆算前馈(FF1/2, 指导教員の案) / C 本方法(FF+π)
    B−A = 逆算案的效果     C−B = **反馈的净增值**

关键口径（闭环 A/B 的必然约束）:
  三臂的指令流各不相同(不同观测→不同 chunk), **没有公共参照**, 曲线不能逐点叠。
  所以每臂对**自己的原始 ACT 指令**渲染 ideal, 各算各的 gap, 只比统计量。
  "自己的原始 ACT 指令" = wrapper 日志里的 chunks[].raw（FF 与 δa 之前）。

主判据: 带通(0.5-9Hz) gap 能量, 剔接触帧, **分任务** —— 配对 t 检验(按初始位姿配对)
副判据: 任务成功率 / ‖δa‖ / 饱和率 / gate 触发率 / stale·NaN 计数
预注册淘汰条款: **若 C−B 不显著, 我们自己砍掉 π, 保留 FF 层毕业。**

用法:
  python sim/analyze_ab.py --root outputs/ab            # 读真机日志
  python sim/analyze_ab.py --self-test                  # 无机时自检(合成三臂)
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports/figures"
FPS, CHUNK = 20.0, 100
ARMS = ("A", "B", "C")            # 原三臂设计的默认值; 可用 --arms 覆盖


def arm_episodes(root: Path, arm: str, min_frames: int, n: int | None):
    """按**采集次序**取该臂的有效集。

    两处必须这样做, 否则配对会错位(2026-08-04 修):
      · 排序用 ep 后的整数, 不能用字典序 —— ep10 会排到 ep2 前面, ≥10 集必错
      · 过滤掉中断集(frames 过少), 与 plot_arm_compare/gcheck 的口径一致
    """
    import re
    fs = sorted(root.glob(f"{arm}_ep*.json"),
                key=lambda p: int(re.search(r"ep(\d+)", p.name).group(1)))
    fs = [f for f in fs
          if json.loads(f.read_text()).get("summary", {}).get("frames", 0) > min_frames]
    return fs[:n] if n else fs


def _bp():
    import train_pi_e2 as TP
    if TP.BP is None:
        TP.BP = TP.make_bp()
    return lambda x: TP.BP(torch.from_numpy(np.ascontiguousarray(x, np.float32))
                           .view(1, 1, -1, 3).to(TP.DEV))[0, 0].cpu().numpy()


def load_run(path, R, renderer=None):
    """一次 rollout → (gap 能量, 副判据)。ideal 从**原始 ACT 指令**逐 chunk 渲染。"""
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    d = json.loads(Path(path).read_text())
    fr, ch = d["frames"], d.get("chunks", [])
    imu = np.array([f["imu"] for f in fr if f["imu"] is not None], np.float32)
    da = np.array([f["da"] for f in fr], np.float32)
    assert len(ch), f"{path} 缺 chunks(原始ACT指令) — 需用补过日志的 wrapper 重跑"
    rend = renderer or IdealChunkRenderer()
    outs, prev = [], None
    for c in ch:
        raw = np.array(c["raw"], np.float32)
        outs.append(rend.render(raw, prev))
        prev = raw
    ideal_sim = np.concatenate(outs)[:len(imu)]
    ideal = ideal_sim[:, 3:6] @ R.T                            # → 真机系
    real = imu[:len(ideal), 3:6]
    bp = _bp()
    gap = bp(real - ideal)
    return {"energy": float((gap ** 2).sum()), "n": len(gap),
            "gap_rms": float(np.sqrt((gap ** 2).mean())),
            "da_rms": float(np.sqrt((da ** 2).mean()) * 1e3),
            "da_peak": float(np.abs(da).max() * 1e3),
            "summary": d.get("summary", {})}


def paired(x, y):
    """配对 t 检验: 返回 (均值差, t, 双侧 p, 95%CI)。

    2026-08-04 修: 原来用正态近似 `erfc(|t|/√2)`, n=10 时**系统性低估 p**。
    实例: insert 的 B3−A, t=−1.97 → 正态近似 0.0493(判"显著"), t(9) 真值 0.0809(不显著)。
    结论会被这一步翻转, 所以改用 t 分布。用 pp2 旧数据校验: 新算法给出
    B3−A p=0.0015 / B3b13−A p=0.0001, 与历史已发布值逐位一致。
    """
    d = np.asarray(y, float) - np.asarray(x, float)
    n = len(d)
    m, s = d.mean(), d.std(ddof=1)
    se = s / np.sqrt(n)
    t = m / max(se, 1e-12)
    from scipy import stats
    p = float(2 * stats.t.sf(abs(t), df=n - 1))
    tcrit = float(stats.t.ppf(0.975, df=n - 1))          # n=10 时 2.262, 不是 1.96
    return m, t, p, (m - tcrit * se, m + tcrit * se)


def synth_runs(n=10, seed=0):
    """自检: 合成三臂结果（注入已知效应: B 比 A 好 20%, C 比 B 再好 15%）。"""
    rng = np.random.default_rng(seed)
    base = rng.lognormal(0.0, 0.35, n)                          # 集间差异(配对结构)
    noise = lambda: rng.normal(1.0, 0.08, n)
    return {"A": base * noise(), "B": base * 0.80 * noise(), "C": base * 0.68 * noise()}


def report(E, task, arms=ARMS):
    n = min(len(E[a]) for a in arms)
    print(f"\n═══ {task} ═══   (每臂 {n} 集, 按采集次序配对)")
    print(f"{'臂':>8}{'gap 能量(均值)':>16}{'相对 A':>10}{'改善集数':>10}")
    for a in arms:
        imp = int(np.sum(np.array(E[a][:n]) < np.array(E["A"][:n]))) if a != "A" else 0
        print(f"{a:>8}{np.mean(E[a]):>16.4g}"
              f"{np.mean(E[a])/np.mean(E['A'])*100-100:>9.1f}%"
              f"{(f'{imp}/{n}' if a != 'A' else '—'):>10}")

    # 比较对: 每个非 A 臂 vs A, 再加相邻臂之间(看逐步增益)
    pairs = [("A", a, f"{a}−A") for a in arms if a != "A"]
    rest = [a for a in arms if a != "A"]
    pairs += [(rest[i], rest[i + 1], f"{rest[i+1]}−{rest[i]}  (逐步增益)")
              for i in range(len(rest) - 1)]

    res = {}
    for lo, hi, name in pairs:
        m, t, p, ci = paired(E[lo][:n], E[hi][:n])
        rel = m / np.mean(E[lo][:n]) * 100
        sig = "✓ 显著" if p < 0.05 else "✗ 不显著"
        print(f"  {name:<28} {rel:+6.1f}%   t={t:+.2f}  p={p:.4f}  {sig}")
        res[f"{hi}-{lo}"] = {"rel_pct": rel, "t": t, "p": p, "ci": list(ci)}

    # 淘汰条款只对**原三臂设计**成立, 不能套到任意臂集上
    if tuple(arms) == ("A", "B", "C") and res.get("C-B", {}).get("p", 1) >= 0.05:
        print("\n  ⚠ **预注册淘汰条款触发**: C−B 不显著 ⇒ π 无净增值,")
        print("     按事前约定砍掉 π, 保留 FF 层毕业。**是否执行由用户决定。**")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/ab")
    ap.add_argument("--task", default="ins", help="pp2 / ins —— 决定用哪个 manifest 的 R")
    ap.add_argument("--arms", default="A,B,C", help="逗号分隔, 第一个必须是基线 A")
    ap.add_argument("--n", type=int, default=10, help="每臂取前 n 集(按采集次序)")
    ap.add_argument("--min-frames", type=int, default=500, help="低于此帧数视为中断集, 剔除")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    arms = tuple(s.strip() for s in a.arms.split(",") if s.strip())
    assert arms[0] == "A", f"第一个臂必须是基线 A, 收到 {arms}"

    if a.self_test:
        print("=== 自检: 合成三臂 (注入 B比A好20%, C比B再好15%) ===")
        E = synth_runs(10)
        res = report(E, "self-test")
        ok = (res["B-A"]["p"] < 0.05 and res["C-B"]["p"] < 0.05
              and -25 < res["B-A"]["rel_pct"] < -15 and -22 < res["C-B"]["rel_pct"] < -8)
        print(f"\n自检: {'✅ 检验流程正确(效应量与显著性都回收)' if ok else '❌ 流程有问题'}")
        return 0 if ok else 1

    root = ROOT / a.root
    man = json.loads((ROOT / f"outputs/p2_labels/{a.task}-deploy/manifest.json").read_text())
    R = np.array(man["R"], np.float32)
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    rend = IdealChunkRenderer()
    E, S = {}, {}
    for arm in arms:
        fs = arm_episodes(root, arm, a.min_frames, a.n)
        assert fs, f"未找到 {root}/{arm}_ep*.json (或全部短于 {a.min_frames} 帧)"
        rs = [load_run(f, R, rend) for f in fs]
        E[arm] = [r["energy"] / r["n"] for r in rs]            # 每帧能量(消除长度差)
        S[arm] = rs
        sm = rs[0]["summary"]
        print(f"{arm}: {len(fs)} 集  ‖δa‖={np.mean([r['da_rms'] for r in rs]):.2f}mrad "
              f"峰={max(r['da_peak'] for r in rs):.1f}mrad  "
              f"ff_lag={sm.get('ff_lag','?')} ff_blend={sm.get('ff_blend','?')}"
              f"(有效{sm.get('ff_blend_effective','?')}) mode={sm.get('mode','?')}")
    if len({len(E[x]) for x in arms}) > 1:
        print(f"  ⚠ 各臂集数不等 {[len(E[x]) for x in arms]} ⇒ 配对截断到最短")
    res = report(E, a.task, arms)
    (root / "ab_result.json").write_text(json.dumps(
        {"arms": list(arms),
         "per_arm_energy": {k: list(map(float, v)) for k, v in E.items()},
         "per_arm_summary": {k: [r["summary"] for r in S[k]] for k in arms},
         "tests": res}, indent=1))
    print(f"\n→ {root/'ab_result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
