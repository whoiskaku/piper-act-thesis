"""B4 的统计与判定（方案 `reports/PLAN_b4.md` §6，约束 3）。

- 回放：组 = 轨迹、单位 = 轮；闭环：组 = 位置、单位 = 配对。两级自助法：先有放回地抽组，再在每个抽到的组里有放回地抽单位。
- 判定只看关节偏差（相对原始策略指令）的 95% 区间上限 < 0，加运动护栏；末端角速度等只报告。
  不沿用 `ilc_route1` 的 compare / verdict（两个主指标的判据 b）和 hier（只抽轮次、序列不抽）。
"""
from __future__ import annotations

from math import comb

import numpy as np


def rel_change(x: float, y: float) -> float:
    """x 相对 y 的变化（%）。"""
    return (x - y) / y * 100.0


def hier2(rel: dict[str, list[float]], n_boot: int = 10000, seed: int = 0) -> dict:
    """rel = {组: 各单位的相对变化（%）}。效应 = 各组中位数的平均；95% 区间 = 两级自助（先抽组、再抽组内单位）。"""
    groups = list(rel)
    vals = [np.asarray(rel[g], float) for g in groups]
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, len(vals), len(vals))
        boot[b] = np.mean([np.median(vals[i][rng.integers(0, len(vals[i]), len(vals[i]))]) for i in picked])
    return {"effect": float(np.mean([np.median(v) for v in vals])),
            "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
            "per_group": {g: {"median": float(np.median(v)), "n": len(v)} for g, v in zip(groups, vals)},
            "n_groups": len(vals)}


def judge(r: dict) -> str:
    """r = {"joint_raw", "path", "speed", …}（各为 hier2 的结果）。护栏：路径长度、关节速度的效应在 ±5% 内；
    有效：关节偏差的 95% 区间上限 < 0（不设最小降幅）。其余键只报告。"""
    bad = [name for key, name in (("path", "路径长度"), ("speed", "关节速度")) if abs(r[key]["effect"]) > 5]
    if bad:
        return f"护栏未过（{'、'.join(bad)}的变化超过 ±5%）"
    return "有效" if r["joint_raw"]["ci95"][1] < 0 else "关节偏差没有显著下降"


def strict_count(pairs: list[tuple[bool, bool]]) -> tuple[int, int, bool]:
    """pairs = [(Shift 成功, Shift＋网络成功)] → (成功→失败数, 失败→成功数, 前者不多于后者)。"""
    n_sf = sum(1 for s, n in pairs if s and not n)
    n_fs = sum(1 for s, n in pairs if n and not s)
    return n_sf, n_fs, n_sf <= n_fs


def judge_closed(r: dict, pairs: list[tuple[bool, bool]], n_required: int = 20) -> str:
    """闭环：对数必须正好 n_required；在 judge 之外，任务时长增加 ≤ 5%、策略指令自身速度下降 ≤ 5%、成功率严格计数不下降。"""
    if len(pairs) != n_required:
        return f"只报告、未完成（完整对数 {len(pairs)}/{n_required}）"
    bad = []
    if r["duration"]["effect"] > 5:
        bad.append("任务时长增加超过 5%")
    if r["cmd_speed"]["effect"] < -5:
        bad.append("策略指令自身速度下降超过 5%")
    n_sf, n_fs, ok = strict_count(pairs)
    if not ok:
        bad.append(f"成功率下降（成功→失败 {n_sf} 对，失败→成功 {n_fs} 对）")
    return f"护栏未过（{'；'.join(bad)}）" if bad else judge(r)


def chance_fail_prob(p: float, n: int) -> float:
    """两组真实失败率都是 p、各对独立时，n 对里「成功→失败」严格多于「失败→成功」的精确概率（三项分布求和）。"""
    q = p * (1 - p)                    # 一对是「成功→失败」（或「失败→成功」）的概率
    rest = 1 - 2 * q
    return float(sum(comb(n, x) * comb(n - x, y) * q ** (x + y) * rest ** (n - x - y)
                     for x in range(n + 1) for y in range(min(x, n - x + 1))))
