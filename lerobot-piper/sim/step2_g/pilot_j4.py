#!/usr/bin/env python3
"""j4 分级开发 pilot —— **开发/安全数据，不进入标定集，也不进入检验集。**

为什么 j4 需要单独 pilot: 它在 pp2 的全部 17 集里指令范围是 `0.0…0.0` ——
**从未动过**。因此本项目对 j4 没有任何已验证的运动包络，其速度/加速度上限
（现取 850 °/s²，借自 j5）**只是保守的操作上限，不是经过认证的硬件极限**；
也**不**直接跳到 j6 的 2976 °/s²。

分级规则（逐级放行，每级过关才升级）:
  · 起点 = **最接近 HOME 的远端位形** + 慢速 + 小幅
  · 幅值 25% → 50% → 100%，**每级重复 2 次**
  · **同一级不同时增加幅度与频率** —— 幅值阶梯跑完，才动上升时间
  · 快阶跃 → chirp → 多关节多正弦 放在最后
  · 每级之间由操作者依据 `--check` 的结果放行

用法:
    python sim/step2_g/pilot_j4.py --list
    python sim/step2_g/pilot_j4.py --save
    python sim/step2_g/pilot_j4.py --check outputs/step2_g/pilot_j4/replay_P01_A.json
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
    AMAX_DEG, FPS, HOME_DEG, LIMIT_MARGIN_DEG, Seq, VMAX_DEG,
    check_limits, chirp, max_amp, multisine, pick_distal_poses, place,
    slow_sweep, smooth_step,
)

MODEL = ROOT / "sim/piper_model.xml"
OUT = ROOT / "outputs/step2_g/pilot_j4"
J = 3                                   # j4 (0-based)
REPEATS = 2
PURPOSE = "development/safety pilot — NOT for calibration, NOT for validation"

# 放行门槛（每级过关才升级）。都是从录到的日志能直接算的量。
# ── 跟踪判据（2026-08-24 修订并冻结；**失败后修订**，理由见下）────────────
# 起因: P03 的 p99 = 2.705° > 2.0° 判否。实测三条:
#     P01 幅值 2.061°  p99 1.443°  最优滞后 2 帧  补后 0.121°  (滞后占 92%)
#     P02 幅值 2.061°  p99 1.212°  最优滞后 2 帧  补后 0.314°  (滞后占 74%)
#     P03 幅值 4.122°  p99 2.705°  最优滞后 2 帧  补后 0.239°  (滞后占 91%)
# 幅值 ×2.00 ⇒ p99 ×1.87, 滞后恒为 2 帧, 补后残差始终 <0.32°
# ⇒ 原判据用**绝对阈值**去量一个正比于**速度**的量(滞后 × 速度), 在幅值阶梯上
#    必然在某一级撞墙, 且撞的是量纲错误而非物理极限。滞后本身是 G 要建模的
#    对象, 不是故障。
# 修订: 固定补偿 **2 帧(100 ms)**, 判 p99(|e_t|) ≤ 0.5°, 其中
#     e_t = q_state[t + LAG] − q_cmd[t]
# **不逐条搜索最优滞后** —— 那会把异常延迟一起拟合掉, 使门禁失效。
# 若真实滞后变化, 固定 2 帧后的残差会增大并触发门禁, 不会被自动隐藏。
LAG_FRAMES = 2                         # 冻结; 实测 P01/P02/P03 的最优滞后均为 2
GATE = {
    "min_frames": 100,
    "max_stale": 0, "max_nan": 0,
    "track_err_p99_deg": 0.5,          # p99(|q_state[t+2] − q_cmd[t]|)
    "lag_frames": LAG_FRAMES,
    "limit_margin_deg": LIMIT_MARGIN_DEG,
}


def nearest_home_pose(m) -> np.ndarray:
    """j4 的三个远端位形里**最接近 HOME** 的那个（远端 = j5, j6）。"""
    home = np.deg2rad(HOME_DEG)
    poses = pick_distal_poses(m, J, n=3)
    return min(poses, key=lambda p: float(np.abs(p[J + 1:6] - home[J + 1:6]).sum()))


def ladder(m) -> list[Seq]:
    """严格有序的分级阶梯。索引即执行顺序，不得跳级。"""
    pose = nearest_home_pose(m)
    out: list[Seq] = []
    # **已验证幅值上限**：阶段 A 逐级证到哪，后续阶段就不得超过哪。
    # 这是「同一级不同时增加幅度与频率」的可执行形式 —— 阶段 B/C/D 只许提高
    # 速度/频率，幅值必须落在阶段 A 已经跑过的区间内。
    proven = {"amp": 0.0}

    def add(tag, phase, level, cmd, meta):
        amp = float(meta.get("amp_deg", 0.0))
        if "revision" in meta:
            # 修订项不参与「已验证幅值」断言（它降的是**动态强度**，幅值不变），
            # 但仍要带上 proven_amp_deg 以便与其余条目同表打印。
            cmd = place(m, cmd, J)
            # 修订项的两次重复按用户规格命名为 P05b / P06b（不是 _r1/_r2）
            _names = {"P05b": ("P05b_step_rise10", "P06b_step_rise10"),
                      "P07c": ("P07c_step_rise7", "P08c_step_rise7")}[tag]
            for r, nm in enumerate(_names):
                out.append(Seq(nm, phase, (J,), cmd, pose,
                               {"level": level, "phase": phase, "rep": r + 1,
                                "proven_amp_deg": round(proven["amp"], 3), **meta}))
            return
        if phase.startswith("A"):
            proven["amp"] = max(proven["amp"], amp)
        else:
            assert amp <= proven["amp"] + 1e-9, (
                f"{tag}: 幅值 {amp:.3f}° 超过阶段 A 已验证的 {proven['amp']:.3f}° "
                f"⇒ 会同时增加幅度与频率, 违反分级规则")
        meta = {**meta, "proven_amp_deg": round(proven["amp"], 3)}
        cmd = place(m, cmd, J)
        for r in range(REPEATS):
            out.append(Seq(f"P{len(out)+1:02d}_{tag}_r{r+1}", phase, (J,), cmd, pose,
                           {"level": level, "phase": phase, "rep": r + 1, **meta}))

    # ── 阶段 A: 慢速(rise=5) + 慢扫，只涨幅值 ──
    a_slow = max_amp(m, J, lambda a: smooth_step(J, pose, a, 5))
    for frac in (0.25, 0.50, 1.00):
        add(f"A_step_slow_f{frac:.2f}", "A_幅值阶梯", frac,
            smooth_step(J, pose, a_slow * frac, 5),
            {"amp_deg": round(a_slow * frac, 3), "rise_frames": 5})
    # 慢扫受**行程**而非速度约束, 可达幅值远超阶跃阶梯顶(实测 19.1° vs 8.2°)
    # ⇒ 必须**封顶到阶段 A 已验证的幅值**, 否则等于跳级。
    a_sw = min(max_amp(m, J, lambda a: slow_sweep(J, pose, a, 2.0)), a_slow)
    add("A_sweep_v2", "A_幅值阶梯", 1.00, slow_sweep(J, pose, a_sw, 2.0),
        {"amp_deg": round(a_sw, 3), "speed_dps": 2.0})

    # ── 阶段 B: 幅值锁在 100%，只涨速度(rise 5→2) ──
    a_fast = min(max_amp(m, J, lambda a: smooth_step(J, pose, a, 2)), a_slow)
    add("B_step_fast", "B_频率阶梯", 1.00, smooth_step(J, pose, a_fast, 2),
        {"amp_deg": round(a_fast, 3), "rise_frames": 2})
    a_sw8 = min(max_amp(m, J, lambda a: slow_sweep(J, pose, a, 8.0)), a_slow)
    add("B_sweep_v8", "B_频率阶梯", 1.00, slow_sweep(J, pose, a_sw8, 8.0),
        {"amp_deg": round(a_sw8, 3), "speed_dps": 8.0})

    # ── 阶段 C: chirp ──
    a_c = min(max_amp(m, J, lambda a: chirp(J, pose, a)), a_slow)
    add("C_chirp", "C_扫频", 1.00, chirp(J, pose, a_c), {"amp_deg": round(a_c, 3)})

    # ── 设计修订 P05b/P06b（2026-08-24）: 顶档改为**降低动态强度**而非降幅值 ──
    # 起因: P05(8.245°, rise=5, 峰速 58.3°/s, 峰加速 683°/s²) 被门禁判否。
    # 诊断(零机时, 现有数据):
    #   · 亚帧滞后扫描: 最优滞后随速度变 98 ms(≤29°/s) → 133 ms(52°/s),
    #     但补偿后残差仍 0.891° > 0.5° ⇒ **延迟解释不完**
    #   · 速度峰值无平台(前5峰离散度 4.60 vs P03 的 0.56) ⇒ **不是速率饱和**
    #     ⇒ j4 上限**不该**取 29°/s
    #   · 残差 **97% 的能量集中在高加速度段**(a>p80), 高速段 82%, 低速平稳段 0.625°
    # ⇒ 异常**主要集中于加减速瞬态**。
    # **措辞界限**: 尚不能严格证明「只有加速度起作用」—— P05b 同时降低了速度与
    # 加速度, 两者未解耦。但这不妨碍工程决策: 目标不是精确定位极限, 而是找到
    # **正式辨识可用的干净激励**。40°/s 边界试探留待「发现激励强度不足」时再做。
    #
    # 修订: **保留 8.245° 大幅值**(可重复性 94.2% 靠它), 上升时间 5 → 10 帧
    # ⇒ 峰速与峰加速度退回已验证范围。原 P05/P06 及其失败记录**保留不覆盖**。
    add("P05b", "A_幅值阶梯(修订)", 1.00,
        smooth_step(J, pose, a_slow, 10),
        {"amp_deg": round(a_slow, 3), "rise_frames": 10,
         "revision": "replaces P05/P06; rise 5→10 to cut dynamic intensity"})

    # ── 设计试探 P07c/P08c（2026-08-26）: rise=7，补 2–6 Hz 的激励缺口 ──
    # 触发条件是 P05b/P06b 定稿时预设的「若后续发现激励强度不足」，现已满足:
    #   逐频段**信号能量**(陀螺, 绝对值)实测 ——
    #     0.5–2 Hz : 2.061°rise5 0.94 | 4.122°rise5 3.82 | 8.245°rise10 **11.52**
    #     2–3.5 Hz : 0.67 | **2.17** | 0.80(×0.37)
    #     3.5–6 Hz : 0.35 | **0.84** | **0.24(×0.28)**
    #   rise 5→10 把 2–6 Hz 的激励压掉 63–72%，可重复性 81.4% → 54.1%。
    #   8.245°rise10 的总分 96.6% 高，是**低频能量 ×3 把加权拉上去**，
    #   不是处处更好 —— 而 **2–6 Hz 正是转子惯量与伺服带宽的信息所在**。
    # rise=7 落在**已验证干净的 28.6°/s** 与**失败的 52.4°/s** 之间。
    add("P07c", "试探(rise7)", 1.00, smooth_step(J, pose, a_slow, 7),
        {"amp_deg": round(a_slow, 3), "rise_frames": 7,
         "revision": "probe: fill 2–6 Hz gap left by rise10"})

    # ── 阶段 D: 多关节多正弦（最后） ──
    amps = {j: 2.0 for j in range(6)}
    cmd = multisine(tuple(range(6)), np.deg2rad(HOME_DEG), amps, seed=0)
    for j in range(6):
        cmd = place(m, cmd, j)
    add("D_multisine", "D_多关节", 1.00, cmd, {"amp_deg": 2.0})
    return out


def check(log: Path) -> int:
    """按放行门槛评估一条已录的 pilot 集。→ 0 = 可升级。"""
    d = json.loads(log.read_text())
    cmd = np.asarray(d["cmd"], float)[:, :6]
    st = np.asarray(d["state"], float)[:, :6] / 1000.0        # 毫度 → 度
    n = min(len(cmd), len(st))
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    # **固定** LAG_FRAMES 帧补偿（不逐条搜索最优滞后）
    L = GATE["lag_frames"]
    k = n - L
    err = np.abs(np.rad2deg(cmd[:k]) - st[L:L + k])
    err_raw = np.abs(np.rad2deg(cmd[:n]) - st[:n])
    # 最优滞后仅**报告**, 不参与门禁 —— 它变了正是要看见的信号
    _best = min(((np.percentile(np.abs(np.rad2deg(cmd[:n - q]) - st[q:n]), 99), q)
                 for q in range(9)))
    lo = np.rad2deg(m.jnt_range[:6, 0]) + GATE["limit_margin_deg"]
    hi = np.rad2deg(m.jnt_range[:6, 1]) - GATE["limit_margin_deg"]
    v = np.abs(np.diff(st[:n], axis=0)) * FPS
    rows = [
        ("帧数", n, f"≥ {GATE['min_frames']}", n >= GATE["min_frames"]),
        ("IMU 样本(100Hz)", len(d.get("imu100", [])), "> 0", len(d.get("imu100", [])) > 0),
        (f"跟踪误差 p99 (度, 固定补 {L} 帧)",
         round(float(np.percentile(err[:, J], 99)), 3),
         f"≤ {GATE['track_err_p99_deg']}",
         float(np.percentile(err[:, J], 99)) <= GATE["track_err_p99_deg"]),
        ("实测位置越限", int(((st[:n] < lo) | (st[:n] > hi)).sum()), "= 0",
         not ((st[:n] < lo) | (st[:n] > hi)).any()),
        ("实测 max|v| j4 (度/s)", round(float(v[:, J].max()), 1),
         f"≤ {VMAX_DEG[J]:.0f}", v[:, J].max() <= VMAX_DEG[J] * 1.15),
        ("提前终止", d.get("stopped") or "无", "无", d.get("stopped") is None),
    ]
    print(f"{log.name}\n{'项':<24}{'实测':>16}{'门槛':>12}   ")
    print(f"{'  [报告] 未补偿 p99 (度)':<24}"
          f"{round(float(np.percentile(err_raw[:, J], 99)), 3):>16}{'—':>12}   ")
    print(f"{'  [报告] 本集最优滞后 (帧)':<24}{_best[1]:>16}"
          f"{f'冻结 {L}':>12}   {'⚠ 与冻结值不符' if _best[1] != L else ''}")
    ok = True
    for nm, val, thr, good in rows:
        ok &= bool(good)
        print(f"{nm:<24}{str(val):>16}{thr:>12}   {'✓' if good else '✗'}")
    print(f"\n⇒ {'✅ 可升到下一级' if ok else '❌ **不得升级** —— 先查明原因'}")
    print(f"（本集为 {PURPOSE}）")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--check", type=Path)
    a = ap.parse_args()
    if a.check:
        return check(a.check)
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    seqs = ladder(m)
    pose = seqs[0].pose_distal
    print(f"j4 分级开发 pilot —— {PURPOSE}\n")
    print(f"位形 = 最接近 HOME 的远端配置: j5={np.rad2deg(pose[4]):.1f}° "
          f"j6={np.rad2deg(pose[5]):.1f}°  (HOME: {HOME_DEG[4]:.1f}° / {HOME_DEG[5]:.1f}°)")
    print(f"j4 操作上限(**保守值, 非硬件认证极限**): "
          f"v ≤ {VMAX_DEG[J]:.0f}°/s, a ≤ {AMAX_DEG[J]:.0f}°/s²  —— 借自 j5\n")
    print(f"{'#':<4}{'序列':<26}{'阶段':<14}{'幅值(度)':>10}{'帧':>6}"
          f"{'max|v|':>9}{'max|a|':>9}{'≤已验证':>9}{'hash':>13}")
    for i, s in enumerate(seqs, 1):
        c = check_limits(m, s.cmd)
        print(f"{i:<4}{s.name:<26}{s.meta['phase']:<14}"
              f"{s.meta.get('amp_deg', 0):>10.3f}{len(s.cmd):>6}"
              f"{c['v_max_deg'][J]:>9.1f}{c['a_max_deg'][J]:>9.0f}"
              f"{s.meta['proven_amp_deg']:>9.2f}{s.sha:>13}"
              + ("" if c["limit_ok"] and c["vel_ok"] and c["acc_ok"] else "  ✗超限"))
    print(f"\n执行规则: 按 # 顺序**逐条**跑, 每条跑完用 `--check` 评估;"
          f" 未过关**不得**跑下一条。")
    print(f"同一级**不同时**增加幅度与频率 —— 阶段 A 只涨幅值, 阶段 B 才涨速度。")
    if a.save:
        OUT.mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(seqs, 1):
            np.savez(OUT / f"{s.name}.npz", cmd7=s.cmd.astype(np.float32),
                     da=np.zeros((len(s.cmd), 6), np.float32), tag=s.name,
                     order=i, purpose=PURPOSE, sha=s.sha)
        (OUT / "manifest.json").write_text(json.dumps(
            {"purpose": PURPOSE, "joint": "j4",
             "operational_limits_note": "v/a 上限为保守操作值, 非硬件认证极限; 借自 j5",
             "gate": GATE,
             "order": [{"n": i, "name": s.name, "sha": s.sha, **s.meta}
                       for i, s in enumerate(seqs, 1)]}, ensure_ascii=False, indent=1))
        print(f"\n→ {OUT.relative_to(ROOT)}/  ({len(seqs)} 个 npz + manifest.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
