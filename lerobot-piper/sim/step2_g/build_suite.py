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
    HOME_DEG, check_limits, fit_limits, max_amp, pick_distal_poses, place,
    safe_center, slow_sweep, smooth_step,
)

MODEL = ROOT / "sim/piper_model.xml"
OUT = ROOT / "outputs/step2_g"


def build(m) -> list[Seq]:
    seqs: list[Seq] = []
    def add(name, kind, j, cmd, pose, meta):
        cmd = place(m, cmd, j)                     # 按实际偏移范围居中到限位内
        seqs.append(Seq(name, kind, (j,), cmd, pose, meta))
    poses = {j: pick_distal_poses(m, j, n=3 if j < 4 else 1) for j in range(6)}
    # 幅值按各关节的量程与 pp2 包络取保守值
    amp = {0: 8.0, 1: 8.0, 2: 8.0, 3: 12.0, 4: 10.0, 5: 12.0}
    for j in range(6):
        for pi, pose in enumerate(poses[j]):
            # ── 平滑阶跃: 两个上升时间(快/慢) × 两个幅值 ──
            for rise in (2, 5):
                amax = max_amp(m, j, lambda a, r=rise: smooth_step(j, pose, a, r))
                for frac in (0.35, 1.0):          # 阶梯取**可达最大值的比例**
                    a_ = amax * frac
                    add(f"S_j{j+1}_p{pi}_r{rise}_f{frac:.2f}", "step", j,
                        smooth_step(j, pose, a_, rise), pose,
                        {"amp_deg": round(a_, 3), "rise_frames": rise})
            # ── 慢速双向扫描: 两档低速（为 fc） ──
            for sp in (2.0, 8.0):
                aw = max_amp(m, j, lambda a, v=sp: slow_sweep(j, pose, a, v))
                add(f"W_j{j+1}_p{pi}_v{sp:.0f}", "sweep", j,
                    slow_sweep(j, pose, aw, sp), pose,
                    {"speed_dps": sp, "amp_deg": round(aw, 3)})
            # ── 扫频 ──
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
    return seqs


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
        for s in seqs:
            np.savez(OUT / f"{s.name}.npz", cmd7=s.cmd.astype(np.float32),
                     da=np.zeros((len(s.cmd), 6), np.float32),
                     tag=s.name, kind=s.kind, target=np.array(s.target),
                     pose=s.pose_distal, sha=s.sha)
        (OUT / "manifest.json").write_text(json.dumps(
            {s.name: {"kind": s.kind, "target": list(s.target), "sha": s.sha,
                      "frames": len(s.cmd), **{k: float(v) for k, v in s.meta.items()}}
             for s in seqs}, ensure_ascii=False, indent=1))
        print(f"\n→ {OUT.relative_to(ROOT)}/  ({len(seqs)} 个 npz + manifest.json)")

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
