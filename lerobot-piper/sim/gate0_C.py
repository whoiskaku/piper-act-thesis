#!/usr/bin/env python3
"""C pilot 的**零机时 ⓪ 预算**与阈值确认（c-pilot-v1，2026-07-29）。

计算对象是**预移动 + 慢速往复 + 探针叠加后的完整指令**，不是探针单独。

    residual_margin = 命令余量 − 慢速跟踪误差 p99 − 安全预留 > 0

**跟踪误差口径**：`make_probe_b2.tracking_err` 需反查源 rollout，对合成的 C 位姿
用不了。C 是静止 + 慢动工况，用整段任务轨迹的 p99 过于保守 ⇒ 改为按**命令速度**
筛出与 C 同量级的慢速帧再取 p99。该值仍只是**零机时占位**，
pilot 必须用本次 A1 的实测数据重算（见 RUNBOOK_C_pilot_v1.md 门禁 ⓪）。

用法: python sim/gate0_C.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

import select_poses_C as S
import make_probe_b2 as MB

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/probe_c"
FPS = 20.0
PREMOVE_S, SETTLE_S = 3.0, 1.5
SAFE_RESERVE = np.radians(0.3)        # 安全预留(冻结值), 与 B′ 同
F_WIDE = np.arange(2.0, 8.01, 0.5)
F_NARROW = np.array([6.0, 6.5, 7.0, 7.5])


def tracking_p99_sweep(thresholds):
    """|编码器 − 指令| p99 随**命令速度筛选阈值**的变化（逐关节，rad）。

    2026-07-29 修: 初版取 3×dither = 30°/s, 结果保留了 87% 的帧 —— 等于没筛,
    p99 仍是任务轨迹快速瞬态的 3.85°。C 的注入窗是**准静止保持**, 必须扫阈值
    并看曲线在哪里趋平, 不能拍一个。
    """
    from compare_ideal_methods import load_episode
    E, Aa = [], []
    for repo, n in [("enolab-guo/eval_pickplace2_100k", 10), ("enolab-guo/eval_ab_A", 10)]:
        for ep in range(n):
            d = load_episode(repo, ep)
            q = np.asarray(d["state_mdeg"], float)[:, :6] * np.pi / 180 / 1000
            a = np.asarray(d["action"], float)[:, :6]
            m = min(len(q), len(a))
            E.append(q[:m]); Aa.append(a[:m])
    q = np.concatenate(E); a = np.concatenate(Aa)
    err = np.abs(q - a)
    v = np.abs(np.diff(a, axis=0, prepend=a[:1])) * FPS        # 指令速度 rad/s
    out = []
    for th in thresholds:
        m = v.max(1) < th
        out.append((th, np.percentile(err[m], 99, axis=0) if m.sum() > 200 else None,
                    int(m.sum()), len(err)))
    return out


def p99_by_saturation(speed_hi, lo, hi):
    """准静止 p99, 按**命令是否越出可达范围**分开（2026-07-29）。

    两次迭代:
      v1 混池 ⇒ j2 得 1.63°(含饱和区 +1.17° 静态偏差), 用它扣非饱和点得假性 −1.86°
      v2 按"距边界 2° 内"划分 ⇒ 失败: j2 可达下界 −0.07°、j3 上界 +0.09° 恰在休息位
         附近, 4550/4605 帧都被判"接近", 远离侧只剩 55 帧
      v3(本版) 按**命令越界与否**划分 —— 静态偏差正是命令越出可达范围时产生的
    """
    from compare_ideal_methods import load_episode
    E, Aa = [], []
    for repo, n in [("enolab-guo/eval_pickplace2_100k", 10), ("enolab-guo/eval_ab_A", 10)]:
        for ep in range(n):
            d = load_episode(repo, ep)
            q = np.asarray(d["state_mdeg"], float)[:, :6] * np.pi / 180 / 1000
            a = np.asarray(d["action"], float)[:, :6]
            m = min(len(q), len(a))
            E.append(q[:m]); Aa.append(a[:m])
    q = np.concatenate(E); a = np.concatenate(Aa)
    err = np.abs(q - a)
    v = np.abs(np.diff(a, axis=0, prepend=a[:1])) * FPS
    slow = v.max(1) < speed_hi
    far, near = np.zeros(6), np.zeros(6)
    nf, nn = np.zeros(6, int), np.zeros(6, int)
    for k in range(6):
        inside = (a[:, k] >= lo[k]) & (a[:, k] <= hi[k])   # 命令在可达范围内
        m1 = slow & inside; m2 = slow & ~inside            # m2 = 命令越界
        far[k] = np.percentile(err[m1, k], 99) if m1.sum() > 200 else np.nan
        near[k] = np.percentile(err[m2, k], 99) if m2.sum() > 200 else np.nan
        nf[k], nn[k] = int(m1.sum()), int(m2.sum())
    return far, near, nf, nn


def envelopes():
    """20 集的关节速度 / 增量加速度包络（逐关节 p99.9）。"""
    from compare_ideal_methods import load_episode
    V, Acc = [], []
    for repo, n in [("enolab-guo/eval_pickplace2_100k", 10), ("enolab-guo/eval_ab_A", 10)]:
        for ep in range(n):
            q = (np.asarray(load_episode(repo, ep)["state_mdeg"], float)[:, :6]
                 * np.pi / 180 / 1000)
            v = np.diff(q, axis=0) * FPS
            V.append(np.abs(v)); Acc.append(np.abs(np.diff(v, axis=0) * FPS))
    return (np.percentile(np.concatenate(V), 99.9, 0),
            np.percentile(np.concatenate(Acc), 99.9, 0))


def full_command(p, j, freqs, amp, start):
    """**完整指令** = 预移动(余弦缓入缓出) + 稳定 + (慢速往复 + 探针)。

    返回 (cmd, i_hold, w0): `i_hold` = 保持段起始帧。⓪ 对**注入窗**与**预移动段**
    分开评估 —— 预移动是一次受监控的慢速斜坡, 其起点余量由停放位姿决定,
    与位姿本身无关; 混在一起会让所有位姿得到同一个(错误的)数字。
    """
    body = S.synth_cmd(p)                                # (S,7) 静止保持+慢速往复
    da, w0 = S.probe_da(len(body), j, amp=amp, freqs=freqs)
    body = body.copy()
    body[:, :6] += da                                    # 探针叠加
    npre, nset = int(PREMOVE_S * FPS), int(SETTLE_S * FPS)
    t = np.arange(npre) / max(npre - 1, 1)
    ease = 0.5 * (1 - np.cos(np.pi * t))                 # 余弦缓入缓出
    pre = start[None, :] + ease[:, None] * (body[0] - start)[None, :]
    settle = np.repeat(body[0][None, :], nset, 0)
    return np.concatenate([pre, settle, body], 0), npre + nset, w0


# 机械臂停放在 safe_disable_position (j5 = 0.52 rad = 29.8°), 不是全零。
# 初版误用 np.zeros ⇒ 预移动起点余量恒为 j2 可达下界的 0.07°, 使所有位姿
# 得到同一个 −4.08°(2026-07-29 修)。
START = np.zeros(7, np.float32); START[4] = 0.52


def main():
    tr, ho, cands = S.train_and_holdout(16, 5)
    S.assert_train_only(tr, ho, where="gate0_C")         # **硬断言**
    poses = [cands[i] for i in tr]
    grp = {}
    for k, i in enumerate(tr):
        grp.setdefault(S.key_qq(cands[i]), []).append(k)
    gkeys = list(grp)
    anchors = sorted({grp[gkeys[gi]][0] for gi in (1, 2, 5, 6)})

    lo, hi = MB.reach_limits()
    vmax, amax = envelopes()
    nm = ["j1", "j2", "j3", "j4", "j5", "j6"]

    print("=== C pilot 零机时 ⓪ 预算 (c-pilot-v1) ===\n")
    print(f"  holdout 封存 = {sorted(ho)}；本预算只用 train 位姿\n")

    ths = np.radians([0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 1e4])
    sw = tracking_p99_sweep(ths)
    print("  【跟踪误差 p99 随命令速度筛选阈值的变化】(j2 / j3, 单位 °)")
    print(f"    {'阈值(°/s)':>11}{'保留帧':>12}{'j2 p99':>9}{'j3 p99':>9}")
    for th, v, nk, nt in sw:
        lab = "全部" if th > 1e3 else f"{np.degrees(th):.1f}"
        if v is None:
            print(f"    {lab:>11}{nk:>7}/{nt:<5} {'样本不足':>16}"); continue
        print(f"    {lab:>11}{nk:>7}/{nt:<5}"
              f"{np.degrees(v[1]):>9.3f}{np.degrees(v[2]):>9.3f}")
    use = [x for x in sw if x[1] is not None and x[0] <= np.radians(2.0)]
    p99_pool = use[-1][1] if use else sw[-1][1]
    p99 = p99_pool
    print(f"    ⇒ 取**准静止**口径(≤2°/s, 混池): j2 {np.degrees(p99[1]):.3f}°, "
          f"j3 {np.degrees(p99[2]):.3f}°")
    print(f"      (整段口径 j2 {np.degrees(sw[-1][1][1]):.3f}° ⇒ 直接沿用会多扣 "
          f"{np.degrees(sw[-1][1][1]-p99[1]):.2f}°)")
    far, near, nf, nn = p99_by_saturation(np.radians(2.0), lo, hi)
    print(f"\n  【准静止 p99 按**命令是否越出可达范围**分开】")
    print(f"    {'关节':>5}{'命令在范围内 p99(°)':>21}{'帧数':>8}"
          f"{'命令越界 p99(°)':>18}{'帧数':>8}")
    for k in (1, 2):
        print(f"    {nm[k]:>5}{np.degrees(far[k]):>21.3f}{nf[k]:>8}"
              f"{np.degrees(near[k]):>18.3f}{nn[k]:>8}")
    print(f"    ⇒ 混池会把**命令越界时的静态偏差**算到不越界的工作点上。")
    print(f"      **注入窗**用远离饱和的值; 预移动另按 B′ 已验证的判据"
          f"(稳定 + 可达性感知容差), 不用 residual margin。")
    print(f"      **仍只是零机时占位**, pilot 必须用本次 A1 实测重算。\n")

    print("  【可达范围与包络】(20 集实测)")
    for k in (1, 2):
        print(f"    {nm[k]}: 可达 [{np.degrees(lo[k]):+.2f}, {np.degrees(hi[k]):+.2f}]°  "
              f"v_p99.9={np.degrees(vmax[k]):.1f}°/s  a_p99.9={np.degrees(amax[k]):.0f}°/s²")
    print()

    p99 = np.where(np.isnan(far), p99_pool, far)     # 注入窗: 远离饱和口径
    rows = []
    for pi in range(len(poses)):
        p = poses[pi]
        wide = pi in anchors
        F = F_WIDE if wide else F_NARROW
        E_star = min(_seq_energy(F_WIDE), _seq_energy(F_NARROW))
        amp = 0.002 * np.sqrt(E_star / _seq_energy(F))
        w_inj, w_pre = np.inf, np.inf
        vmx = np.zeros(6); amx = np.zeros(6)
        for j in (1, 2):
            cmd, i_hold, w0 = full_command(p, j, F, amp, START)
            q = cmd[:, :6]
            seg = {"pre": q[:i_hold], "inj": q[i_hold:]}
            for tag, z in seg.items():
                v = np.abs(np.diff(z, axis=0) * FPS)
                a_ = np.abs(np.diff(np.diff(z, axis=0) * FPS, axis=0) * FPS)
                if tag == "inj":
                    vmx = np.maximum(vmx, v.max(0)); amx = np.maximum(amx, a_.max(0))
                for jj in (1, 2):
                    m = min((z[:, jj] - lo[jj]).min(), (hi[jj] - z[:, jj]).min())
                    rm = m - p99[jj] - SAFE_RESERVE
                    if tag == "inj":
                        w_inj = min(w_inj, rm)
                    else:
                        # 预移动**不用** residual margin(见上); 只记录裸命令余量
                        w_pre = min(w_pre, m)
        rows.append((pi, wide, amp, w_inj, w_pre, vmx, amx))

    print("  【逐位姿 residual margin】命令余量 − 准静止跟踪 p99 − 安全预留"
          f"({np.degrees(SAFE_RESERVE):.1f}°)")
    print("    注: **注入窗**与**预移动段**分开评估 —— 预移动是受监控的慢速斜坡,")
    print("        其起点余量由停放位姿决定, 与位姿无关; 混算会让所有位姿同值。")
    print(f"  {'位姿':>5}{'带宽':>6}{'峰值mrad':>10}{'注入窗 rm(°)':>14}"
          f"{'预移动裸余量(°)':>17}{'越速':>6}{'越加速':>8}{'判定':>8}")
    for pi, wide, amp, wi, wp, v, a_ in rows:
        ov = bool((v[[1, 2]] > vmax[[1, 2]]).any())
        oa = bool((a_[[1, 2]] > amax[[1, 2]]).any())
        ok = wi > 0 and not ov and not oa
        print(f"  {pi:>5}{'宽带' if wide else '窄带':>6}{amp*1e3:>10.2f}"
              f"{np.degrees(wi):>14.3f}{np.degrees(wp):>17.3f}"
              f"{'是' if ov else '否':>6}{'是' if oa else '否':>8}"
              f"{'通过' if ok else '**不过**':>8}")

    wa = [r for r in rows if r[1]]
    q2c = np.median([poses[r[0]]["q2"] for r in wa])
    q3c = np.median([poses[r[0]]["q3"] for r in wa])
    center = min(wa, key=lambda r: (poses[r[0]]["q2"] - q2c) ** 2
                 + (poses[r[0]]["q3"] - q3c) ** 2)
    cand2 = [r for r in wa if r[0] != center[0]]
    minrm = min(cand2 if cand2 else wa, key=lambda r: r[3])
    narrow = min([r for r in rows if not r[1]], key=lambda r: r[3])
    print("\n  【pilot 三位姿】")
    for tag, r in [("P1 中心宽带锚点", center), ("P2 最小 rm 宽带锚点", minrm),
                   ("P3 窄带非锚点", narrow)]:
        p = poses[r[0]]
        print(f"    {tag:>18}: 位姿 {r[0]}  q2={np.degrees(p['q2']):+.1f}° "
              f"q3={np.degrees(p['q3']):+.1f}°  grip={p['grip']:.3f}  "
              f"dither={np.degrees(p['dither']):.1f}°  注入窗 rm = {np.degrees(r[3]):+.3f}°")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate0_C_budget.json").write_text(json.dumps({
        "holdout_sealed": sorted(ho),
        "p99_quasistatic_deg": np.degrees(p99).tolist(),
        "p99_alltask_deg": np.degrees(sw[-1][1]).tolist(),
        "reach_lo_deg": np.degrees(lo).tolist(), "reach_hi_deg": np.degrees(hi).tolist(),
        "v_env_deg_s": np.degrees(vmax).tolist(),
        "a_env_deg_s2": np.degrees(amax).tolist(),
        "safe_reserve_deg": float(np.degrees(SAFE_RESERVE)),
        "pilot": {"P1": int(center[0]), "P2": int(minrm[0]), "P3": int(narrow[0])},
        "rm_inj_deg": {int(r[0]): float(np.degrees(r[3])) for r in rows},
        "premove_raw_margin_deg": {int(r[0]): float(np.degrees(r[4])) for r in rows},
    }, indent=1, ensure_ascii=False))
    print("\n→ outputs/probe_c/gate0_C_budget.json")
    return 0


def _seq_energy(F):
    from make_probe_b2 import schroeder, NEXC
    w, _, _ = schroeder(np.asarray(F, float), NEXC)
    return float(np.sum((w * 0.002) ** 2) / 20.0)


if __name__ == "__main__":
    raise SystemExit(main())
