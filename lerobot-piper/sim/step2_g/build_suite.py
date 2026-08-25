#!/usr/bin/env python3
"""Step 2 激励套件 —— 生成序列、报度量、跑仿真可辨识性筛选。**零机时。**

流程: 设计 → 在仿真里量灵敏度与条件数 → 不满足就改设计 → 满足后才上机器。

报告每条序列: 关节范围 / 速度 / 加速度 / 惯量跨度 / 灵敏度 / hash;
以及全套件合并后的**条件数**与逐参数灵敏度。

安全口径: 只报「**关节限位通过**」。`piper_model.xml` 碰撞全关
(contype/conaffinity 非零 0/17) ⇒ **不得**声称「碰撞安全通过」。
序列会走出 pp2 已验证包络(尤其 j4/j5), 上机前须由人在 MuJoCo 里目视确认。

用法:
    python sim/step2_g/build_suite.py --list        # 只列序列与度量
    python sim/step2_g/build_suite.py --analyze     # 加跑灵敏度/条件数
    python sim/step2_g/build_suite.py --save        # 落盘 npz 供 replay_probe 用
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

from sim.step2_g import params as PA                                  # noqa: E402
from sim.step2_g.sequences import (                                   # noqa: E402
    FPS, Seq, check_limits, chirp, inertia_span, multisine,
    HOME_DEG, MIN_USABLE_AMP_DEG, check_limits, fit_limits, fit_workspace, max_amp,
    pick_distal_poses, place, safe_center, slow_sweep, smooth_step,
)

MODEL = ROOT / "sim/piper_model.xml"
# j4 的阶跃包络: 由真机分级 pilot 实测确定, 不走通用的 (rise 2/5 × frac) 规则。
# A 补 2–6 Hz, B 补 0.5–2 Hz; 详见 reports/STEP2_PILOT_GATE_REVISION.md C 节。
J4 = 3
J4_ENVELOPES = ((4.122, 5), (8.245, 10))
J4_AMP_MAX = 8.245        # pilot 测过的最大幅值; 慢扫封到此(动态本就合规)
J4_CHIRP_PENDING = True   # j4 扫频**移出主套件**, 待其单独 pilot 通过后再加回
OUT = ROOT / "outputs/step2_g"


def build(m) -> list[Seq]:
    """生成套件。**调用前 `m` 必须已应用 `PA.baseline`** —— 见下。

    工作空间检查是**动力学**判据, 其结果取决于模型参数。生成端若用 XML 原值
    (`kp1=500`)、审查端用 `PA.baseline`(`kp1=5000`), 两边过冲不同 ⇒ 余量判定不同,
    实测有 3 条生成端过、审查端报「余量不足」。故此处强制对齐。
    """
    assert m.actuator_gainprm[0, 0] == PA.baseline(m)[0], (
        "build() 需要已应用 PA.baseline 的模型 —— 否则生成与审查的动力学不一致")
    seqs: list[Seq] = []
    dropped: list[tuple[str, float]] = []

    def add(name, kind, j, cmd, pose, meta):
        # 居中到限位内, 再按**工作空间**(台面上方, 动力学)二分缩幅
        cmd, ws = fit_workspace(m, cmd, j)
        amp = float(np.rad2deg(cmd[:, j].max() - cmd[:, j].min()))
        if amp < MIN_USABLE_AMP_DEG:
            dropped.append((name, amp))            # 幅值不够用 ⇒ 丢弃并报告
            return
        seqs.append(Seq(name, kind, (j,), cmd, pose,
                        {**meta, "ws_scale": round(ws, 3), "amp_final_deg": round(amp, 3)}))
    poses = {j: pick_distal_poses(m, j, n=3 if j < 4 else 1) for j in range(6)}
    # 幅值按各关节的量程与 pp2 包络取保守值
    amp = {0: 8.0, 1: 8.0, 2: 8.0, 3: 12.0, 4: 10.0, 5: 12.0}
    for j in range(6):
        for pi, pose in enumerate(poses[j]):
            # ── 平滑阶跃: 两个上升时间(快/慢) × 两个幅值 ──
            # ── j4 的阶跃包络由真机 pilot 定死（2026-08-26）──────────────
            # 见 reports/STEP2_PILOT_GATE_REVISION.md C 节。j4 在 pp2 里从未动过,
            # 故其包络不能沿用通用规则, 必须用 pilot 实测的**已验证干净区间**:
            #   ✓ ≤30.2°/s / ≤342°/s²    ✗ ≥46.0°/s / 383°/s²
            # 且**单包络覆盖不了全带** —— rise=10 在 3.5–6 Hz 的激励只有 rise=5 的
            # 0.28 倍(该段正是转子惯量与伺服带宽的信息所在), 而 rise=5 在 0.5–2 Hz
            # 只有 rise=10 的 1/3。故用双包络, 每个频段都不弱于任一单包络。
            # rise=7 试探被否: 两次重复跨在阈值两侧(0.492 过 / 0.608 否) ⇒ 系统性贴线。
            if j == J4:
                for a_, rise in J4_ENVELOPES:
                    add(f"S_j4_p{pi}_rise{rise}_a{a_:.1f}", "step", j,
                        smooth_step(j, pose, a_, rise), pose,
                        {"amp_deg": a_, "rise_frames": rise,
                         "envelope": "pilot-validated (STEP2_PILOT_GATE_REVISION C)"})
            else:
                # **不能用 `continue`** —— 它会跳过本位形剩下的慢扫与扫频生成,
                # 实测那样会连带丢掉 j4 的 6 条 W_ 和 3 条 C_(套件 96→84)。
                # 只替换**阶跃**分支, 其余基元照常生成。
                for rise in (2, 5):
                    amax = max_amp(m, j, lambda a, r=rise: smooth_step(j, pose, a, r))
                    for frac in (0.35, 1.0):      # 阶梯取**可达最大值的比例**
                        a_ = amax * frac
                        add(f"S_j{j+1}_p{pi}_r{rise}_f{frac:.2f}", "step", j,
                            smooth_step(j, pose, a_, rise), pose,
                            {"amp_deg": round(a_, 3), "rise_frames": rise})
            # ── 慢速双向扫描: 两档低速（为 fc） ──
            for sp in (2.0, 8.0):
                aw = max_amp(m, j, lambda a, v=sp: slow_sweep(j, pose, a, v))
                if j == J4:
                    # 动态强度本就合规(2°/s / 40°/s²), 但 19.125° 是 pilot 测过的
                    # 最大幅值 8.245° 的 2.3 倍 —— **没测过就是没测过**, 封顶。
                    aw = min(aw, J4_AMP_MAX)
                add(f"W_j{j+1}_p{pi}_v{sp:.0f}", "sweep", j,
                    slow_sweep(j, pose, aw, sp), pose,
                    {"speed_dps": sp, "amp_deg": round(aw, 3)})
            # ── 扫频 ──
            if j == J4 and J4_CHIRP_PENDING:
                # j4 扫频实测 **53.7°/s / 1700°/s²** —— 是被判否的 P05(683)的 2.5 倍、
                # 已验证上限(342)的 5 倍。**目视审查看不出加速度**, 这条本会直接上机。
                # 扫频是**连续激励**而非单次瞬态, 失效机理未必与阶跃相同, 故不套用
                # 阶跃的界, 也不简单封顶(封到 342 后幅值只剩 0.7°, 激励掉一个数量级),
                # 而是移出主套件、单独跑 pilot(sim/step2_g/pilot_j4_chirp.py)。
                continue
            ac = max_amp(m, j, lambda a: chirp(j, pose, a))
            add(f"C_j{j+1}_p{pi}", "chirp", j, chirp(j, pose, ac), pose,
                {"amp_deg": round(ac, 3)})
    # ── 多关节多正弦（惯量耦合） ──
    ma = {j: amp[j] * 0.4 for j in range(6)}
    base = np.deg2rad([float(np.clip(HOME_DEG[j], *sorted((
        np.rad2deg(m.jnt_range[j, 0]) + 5.0 + ma[j],
        np.rad2deg(m.jnt_range[j, 1]) - 5.0 - ma[j])))) for j in range(6)])
    for s in range(3):
        cmd = multisine(tuple(range(6)), base, ma, seed=s)
        for j in range(6):                        # 逐关节缩到速度/加速度限内
            cmd, _ = fit_limits(m, cmd, j)
        seqs.append(Seq(f"M_all_s{s}", "multisine", tuple(range(6)), cmd, base,
                        {"seed": s}))

    # **丢弃必须报出来。** 早先版本只往 `dropped` 里 append 却没有任何打印, 序列被
    # **静默丢弃** —— 实测 101 条设计只剩 97 条而日志毫无痕迹。
    if dropped:
        print(f"⚠ 因工作空间余量不足**丢弃 {len(dropped)} 条**"
              f"（缩幅后仍 < {MIN_USABLE_AMP_DEG}°）:")
        for nm, a_ in dropped:
            print(f"    {nm}  最终幅值 {a_:.3f}°")
    # 去重: 限位/工作空间缩放可能把不同幅值压成同一条序列
    seen, uniq = set(), []
    for s_ in seqs:
        if s_.sha in seen:
            continue
        seen.add(s_.sha)
        uniq.append(s_)
    if len(uniq) < len(seqs):
        print(f"（去重 {len(seqs)} → {len(uniq)} 条）")
    return uniq


_MCACHE: list = []


def simulate(m0, cmd: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """20 Hz 指令 → 100 Hz 陀螺（与真机 imu100 同率）。

    模型只加载一次并复用：`PA.apply` 会覆写全部被标定的字段，故无状态残留；
    每次从 XML 重载会让 2700 次仿真的开销由加载主导。
    """
    if not _MCACHE:
        _MCACHE.append(mujoco.MjModel.from_xml_path(str(MODEL)))
    m = _MCACHE[0]
    PA.apply(m, theta)
    dt = m.opt.timestep
    sub = int(round(1 / 100.0 / dt))
    d = mujoco.MjData(m)
    d.qpos[:6] = cmd[0, :6]
    mujoco.mj_forward(m, d)
    d.ctrl[:6] = cmd[0, :6]
    for _ in range(int(0.5 / dt)):
        mujoco.mj_step(m, d)
    out = np.zeros((len(cmd) * 5, 3))
    k = 0
    for t in range(len(cmd)):
        d.ctrl[:6] = cmd[t, :6]
        for _ in range(5):
            for _ in range(sub):
                mujoco.mj_step(m, d)
            out[k] = d.sensordata[3:6]
            k += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--max", type=int, help="灵敏度分析只跑前 N 条(调试用)")
    a = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(str(MODEL))
    PA.apply(m, PA.baseline(m))                    # 与 visual_check 对齐
    seqs = build(m)
    tot_s = sum(len(s.cmd) for s in seqs) / FPS
    print(f"套件 {len(seqs)} 条序列   合计 {tot_s/60:.1f} 分钟指令时长"
          f"（不含复位/启动开销）\n")

    bad = []
    print(f"{'序列':<22}{'类型':<10}{'帧':>5}{'关节范围(度)':>18}"
          f"{'max|v|':>8}{'max|a|':>8}{'惯量跨度':>18}{'hash':>14}")
    for s in seqs:
        j = s.target[0]
        chk = check_limits(m, s.cmd)
        lo, hi = inertia_span(m, s.cmd, j)
        ok = chk["limit_ok"] and chk["vel_ok"] and chk["acc_ok"]
        if not ok:
            bad.append(s.name)
        print(f"{s.name:<22}{s.kind:<10}{len(s.cmd):>5}"
              f"{f'{chk[chr(113)+chr(95)+chr(108)+chr(111)+chr(95)+chr(100)+chr(101)+chr(103)][j]:6.1f}…{chk[chr(113)+chr(95)+chr(104)+chr(105)+chr(95)+chr(100)+chr(101)+chr(103)][j]:6.1f}':>18}"
              f"{chk['v_max_deg'][j]:>8.1f}{chk['a_max_deg'][j]:>8.0f}"
              f"{f'{lo:.5f}…{hi:.5f}':>18}{s.sha:>14}"
              + ("" if ok else "  ✗超限"))
    print(f"\n各关节的**跨位形**惯量跨度（激励 j 时靠远端配置拉开；序列**内**恒定）:")
    for j in range(6):
        vs = sorted({inertia_span(m, s.cmd, j)[0] for s in seqs if s.target[0] == j})
        if vs:
            r = vs[-1] / max(vs[0], 1e-12)
            tag = "✅" if r > 1.3 else "❌ 不可分离 Ia"
            print(f"  j{j+1}: " + " / ".join(f"{v:.5f}" for v in vs)
                  + f"   {r:.1f}×   {tag}")
    print(f"\n**关节限位 + 速度/加速度**: {len(seqs)-len(bad)}/{len(seqs)} 通过"
          + (f"   ✗ {bad}" if bad else ""))
    print("⚠ 碰撞在模型里是**关闭**的 ⇒ 未做碰撞判定；上机前须人工目视确认。")

    if a.save:
        OUT.mkdir(parents=True, exist_ok=True)
        # **manifest 先写**: npz 先写会在 manifest 失败时留下「90 个 npz + 无
        # manifest」的半成品 —— 文件都在, 看起来像成功了。实测这么坏过一次
        # (meta 里的字符串 "envelope" 被 float() 硬转而抛错)。
        man = {s.name: {"kind": s.kind, "target": list(s.target), "sha": s.sha,
                        "frames": len(s.cmd),
                        # meta 值可能是字符串(如 envelope 的出处), **不要硬转 float**
                        **{k: (float(v) if isinstance(v, (int, float)) else str(v))
                           for k, v in s.meta.items()}}
               for s in seqs}
        tmp = OUT / "manifest.json.tmp"
        tmp.write_text(json.dumps(man, ensure_ascii=False, indent=1))
        for s in seqs:
            np.savez(OUT / f"{s.name}.npz", cmd7=s.cmd.astype(np.float32),
                     da=np.zeros((len(s.cmd), 6), np.float32),
                     tag=s.name, kind=s.kind, target=np.array(s.target),
                     pose=s.pose_distal, sha=s.sha)
        tmp.replace(OUT / "manifest.json")          # 全部 npz 落地后才生效
        # 自验: 每条 manifest 都要有对应 npz, 且 npz 里的 sha 与 manifest 一致
        bad = []
        for nm, v in man.items():
            f = OUT / f"{nm}.npz"
            if not f.exists():
                bad.append(f"{nm}: npz 缺失")
                continue
            z = np.load(f, allow_pickle=True)
            if str(z["sha"]) != v["sha"]:
                bad.append(f"{nm}: npz sha {z['sha']} ≠ manifest {v['sha']}")
        if bad:
            raise SystemExit(f"落盘自验失败:\n  " + "\n  ".join(bad))
        print(f"\n→ {OUT.relative_to(ROOT)}/  ({len(seqs)} 个 npz + manifest.json)"
              f"   自验 ✅ 逐条 sha 一致")

    if not a.analyze:
        return 0

    th0 = PA.baseline(m)
    step = PA.perturb_scale(th0)
    use = seqs[: a.max] if a.max else seqs
    print(f"\n仿真灵敏度（{len(use)} 条序列, {PA.N_P} 参数, 100 Hz 陀螺）…")
    S = [[] for _ in range(PA.N_P)]
    for n_, s in enumerate(use):
        b0 = simulate(m, s.cmd, th0)
        for i in range(PA.N_P):
            th = th0.copy()
            th[i] += step[i]
            S[i].append((simulate(m, s.cmd, th) - b0).ravel())
        if (n_ + 1) % 10 == 0:
            print(f"  {n_+1}/{len(use)}")
    S = np.array([np.concatenate(r) for r in S])
    nrm = np.linalg.norm(S, axis=1)
    scale = np.linalg.norm(np.concatenate([simulate(m, s.cmd, th0).ravel() for s in use]))
    print(f"\n{'参数':<7}{'灵敏度':>10}   {'参数':<7}{'灵敏度':>10}   "
          f"{'参数':<7}{'灵敏度':>10}   {'参数':<7}{'灵敏度':>10}")
    for j in range(6):
        cells = [f"{PA.NAMES[b*6+j]:<7}{nrm[b*6+j]/scale*100:>9.2f}%" for b in range(4)]
        print("   ".join(cells))
    dead = [PA.NAMES[i] for i in range(PA.N_P) if nrm[i] / scale < 0.005]
    print(f"\n灵敏度 <0.5% 的参数: {dead if dead else '无'}")

    U = S / np.maximum(nrm[:, None], 1e-30)
    w = np.linalg.svd(U, compute_uv=False)
    print(f"条件数 = {w[0]/w[-1]:.2e}   奇异值(归一化) {np.round(w/w[0], 3)[:8]} …")
    C = U @ U.T
    pr = sorted(((abs(C[i, k]), PA.NAMES[i], PA.NAMES[k])
                 for i in range(PA.N_P) for k in range(i + 1, PA.N_P)), reverse=True)
    print("最共线的 6 对:")
    for c, x, y in pr[:6]:
        print(f"  {x:<6}↔ {y:<6} |cos| = {c:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
