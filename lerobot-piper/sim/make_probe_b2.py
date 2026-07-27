#!/usr/bin/env python3
"""B′ 探针生成器（2026-07-27, 用户规格定稿）— 6-8Hz 窄带多正弦。

动机: 现有 5Hz 小波探针在 6-8Hz **不可辨识**(相干度 0.08-0.46, CI ±15~46%),
      而 π 的效力主要来自该频段(δa 限到 <7Hz 时效果 4.1%→2.5%)。

规格(用户钉死):
  · 频点 6.0 / 6.5 / 7.0 / 7.5 Hz —— 避开精确 8/10 Hz(20Hz ZOH 下可信上限约 8Hz)
  · **4/8 mrad = 合成多正弦的总峰值**, 非每频点各 4/8; 合成后归一化, 绝不超 8 mrad
  · **低峰值因子相位方案**(Schroeder), 非任意随机; 保存 seed 与实际发送序列
  · 每发 2 秒(40帧)激励 + **≥1.2-1.5s 衰减区**; 接触窗不注入
  · 按关节分别激励(j2 / j3), 每种配置独立一条复放(不混合, 避免位形与余振污染)

复放结构(每个基准 episode 6 条):
  D方案(正负成对, 用户 2026-07-27 定): (B+−B−)/2 提取线性增量, 噪声方差 1.5σ²→0.5σ²(3×SNR)
  (B+ + B−)/2 − (A1+A2)/2 = 偶次残差, 应≈0; 不为零即暴露非线性/漂移/复放不一致
  阶段1 ep0(10条): A1 · B±j2-4 · B±j3-4 · B±j2-8 · B±j3-8 · A2
  阶段2 ep1/2(各6条): A1 · B±j2-8 · B±j3-8 · A2      合计 22 条(必要时补4mrad→30条)
  四条 B 的**注入窗口相同**(可比), 顺序在 episode 间随机化; A1/A2 由四条 B 共用。
  ep0/ep1 拟合, **ep2 完全留出验收**。共 6×3 = 18 条。

输出: outputs/probe_b2/seq_ep{E}_{cfg}.npz  +  windows_ep{E}.json(共用注入窗)
用法: python sim/make_probe_b2.py [--eps 0 1 2]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

import train_pi_e2 as TP
from g_model import Rigid, load_ensemble
from wall_portrait import smooth

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/probe_b2"
TAG = "pp2-deploy"
FREQS = np.array([6.0, 6.5, 7.0, 7.5])     # Hz
NEXC = 40                                   # 2.0 s 激励
NDEC = 30                                   # 1.5 s 衰减(≥1.2-1.5s)
PERIOD = NEXC + NDEC                        # 3.5 s 一发
AMPS = {"4mrad": 0.004, "8mrad": 0.008}     # 合成**总峰值**(不顶 ε=10; 见 D 方案)
SIGNS = {"p": +1.0, "m": -1.0}              # D 方案: 正负成对(相位相反)
JOINTS = {"j2": 1, "j3": 2}
GATE_TH = 0.04
FPS = 20.0


def schroeder(freqs, n, fps=FPS, seed=0):
    """Schroeder 低峰值因子多正弦; 峰值归一化到 1。返回 (n,), 以及相位表。"""
    rng = np.random.default_rng(seed)
    k = np.arange(len(freqs))
    # Schroeder 相位: φ_k = −π k(k−1)/N  (等幅谱), 再叠一个 seed 相关的整体旋转
    ph = -np.pi * k * (k - 1) / len(freqs) + rng.uniform(0, 2 * np.pi)
    t = np.arange(n) / fps
    x = sum(np.sin(2 * np.pi * f * t + p) for f, p in zip(freqs, ph))
    cf = np.abs(x).max()
    return (x / cf).astype(np.float32), ph, float(cf / np.sqrt(np.mean(x ** 2)) * 1)


def robot_limits(margin=0.035):
    """真机安全限位 = 20 集实测编码器范围 ± margin(默认 2°)。

    2026-07-27: MuJoCo 的 jnt_range 是**保守软限**(j2 下界写 0°, 但真机实测到 −0.1°、
    指令到 −1.9° 且正常执行 20 集)。用它当约束会把 j2 近半数帧判为"越界"、
    使双向多正弦无处可注入。故改用真机实测范围 + 余量。
    """
    from compare_ideal_methods import load_episode
    E = []
    for repo, n in [("enolab-guo/eval_pickplace2_100k", 10), ("enolab-guo/eval_ab_A", 10)]:
        for ep in range(n):
            E.append(load_episode(repo, ep)["state_mdeg"][:, :6] * np.pi / 180 / 1000)
    E = np.concatenate(E)
    return E.min(0) - margin, E.max(0) + margin


def tracking_err(a_arm):
    """本条基准轨迹自身的 |编码器 − 指令| 逐帧误差。反查源 rollout 取真机 state。

    2026-07-28: 安全余量不能只看指令。编码器会滞后/超调, 真机曾出现 4-7° 的跟踪误差,
    远大于探针的 0.23-0.46°。故窗口选择必须扣掉它(residual margin)。
    """
    from compare_ideal_methods import load_episode
    best = None
    for repo, n in [("enolab-guo/eval_pickplace2_100k", 10), ("enolab-guo/eval_ab_A", 10)]:
        for ep in range(n):
            d = load_episode(repo, ep)
            a = np.asarray(d["action"], float)[:, :6]
            if len(a) < len(a_arm):
                continue
            e = np.abs(a[:len(a_arm)] - a_arm).max()
            if best is None or e < best[0]:
                best = (e, repo, ep, d)
    if best is None or best[0] > 1e-6:
        raise RuntimeError(f"未能反查到基准轨迹的源 rollout (最小差 {best and best[0]})")
    d = best[3]
    n = len(a_arm)
    q = np.asarray(d["state_mdeg"], float)[:n, :6] * np.pi / 180 / 1000
    return np.abs(q - a_arm[:n]), f"{best[1]} ep{best[2]}"


def pick_windows(z, T, mj_lo, mj_hi, a_arm, amp_max, err, j, need=0.5):
    """选 **该关节 j** 的注入窗: 运动段、非接触、且

        residual margin = min(B± 到限位的余量) − p99(|编码器−指令|)  ≥ need(默认 0.5°)

    按关节独立选 —— 同一窗对 j2 安全不代表对 j3 安全(实测 w280 对 j3 为 −2.7°)。
    """
    gate = smooth(np.linalg.norm(z["real"][:, 3:6], axis=1))
    moving = gate >= GATE_TH
    contact = z["mask_contact"]
    need = np.radians(need)
    wins = []
    t = PERIOD
    while t + PERIOD < T:
        seg = slice(t, t + NEXC)
        if not moving[seg].all() or contact[seg].any():
            t += 10
            continue
        v = a_arm[seg, j]
        m = min((v - amp_max - mj_lo[j]).min(), (mj_hi[j] - v - amp_max).min())
        if m - np.percentile(err[seg, j], 99) < need:
            t += 10
            continue
        wins.append(int(t))
        t += PERIOD
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import mujoco
    from ideal_imu_kinematic import MODEL_PATH
    mj = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    mlo, mhi = mj.jnt_range[:6, 0], mj.jnt_range[:6, 1]
    rlo, rhi = robot_limits()                       # 真机实测范围 ±2°
    lo = np.maximum(mlo - 0.035, rlo)               # 兜底: 模型限位也放宽同样余量
    hi = np.minimum(mhi + 0.035, rhi)
    print("安全限位(真机实测±2° ∩ 模型±2°):")
    for j in (1, 2):
        print(f"  j{j+1}: [{np.degrees(lo[j]):.1f}°, {np.degrees(hi[j]):.1f}°]")
    rig = Rigid()
    models = load_ensemble(rigid=rig)
    man = json.loads((TP.LROOT / TAG / "manifest.json").read_text())
    R = np.array(man["R"], np.float32)

    for EP in a.eps:
        z = np.load(TP.LROOT / TAG / f"ep{EP:03d}.npz")
        a_arm = z["a_arm"].astype(np.float32)
        grip = z["grip"].astype(np.float32)
        cmd7 = np.concatenate([a_arm, grip[:, None]], 1)
        T = len(a_arm)
        err, src = tracking_err(a_arm)
        WJ = {jn: pick_windows(z, T, lo, hi, a_arm, AMPS["8mrad"], err, j)
              for jn, j in JOINTS.items()}
        (OUT / f"windows_ep{EP}.json").write_text(json.dumps(
            {"windows_by_joint": WJ, "n_exc": NEXC, "n_dec": NDEC,
             "freqs": FREQS.tolist(), "src_rollout": src,
             "track_p99_deg": {jn: float(np.degrees(np.percentile(err[:, j], 99)))
                               for jn, j in JOINTS.items()}}))
        print(f"\n=== ep{EP}: {T}帧={T/FPS:.0f}s · 跟踪误差源 {src} ===")
        for jn, j in JOINTS.items():
            wins = WJ[jn]
            print(f"  {jn}: 注入窗 {len(wins)} 个 @ {wins}   "
                  f"(全局跟踪 p99 {np.degrees(np.percentile(err[:, j], 99)):.2f}°)")
            for w in wins:
                sg = slice(w, w + NEXC)
                v = a_arm[sg, j]
                m = min((v - AMPS["8mrad"] - lo[j]).min(), (hi[j] - v - AMPS["8mrad"]).min())
                e9 = np.percentile(err[sg, j], 99)
                print(f"      窗{w}: min(B±)余量 {np.degrees(m):6.3f}° − 窗内跟踪p99 "
                      f"{np.degrees(e9):6.3f}° = residual {np.degrees(m-e9):+6.3f}°")
        for jn, j in JOINTS.items():
            wins = WJ[jn]
            for an, amp in AMPS.items():
              for sn, sg in SIGNS.items():
                da = np.zeros((T, 6), np.float32)
                phases = []
                for wi, t in enumerate(wins):
                    seed = EP * 100 + j * 10 + wi          # 可复现; ± 用**同一** seed
                    w, ph, cf = schroeder(FREQS, NEXC, seed=seed)
                    da[t:t + NEXC, j] = w * amp * sg       # B− 是 B+ 的严格取反
                    phases.append({"win": int(t), "seed": int(seed), "sign": sg,
                                   "phases": ph.tolist(), "crest": float(cf)})
                pk = np.abs(da).max()
                assert pk <= amp + 1e-9, f"峰值 {pk*1e3:.2f} 超设定 {amp*1e3}"
                assert pk <= TP.EPS_B + 1e-9, "超 ε 硬界"
                pred = np.stack([(setattr(g, "R", R), g.increment_fast(cmd7, da))[1]
                                 for g in models])
                cfg = f"{jn}_{an}_{sn}"
                np.savez(OUT / f"seq_ep{EP}_{cfg}.npz", da=da, pred=pred, R=R, cmd7=cmd7,
                         wins=np.array(wins), freqs=FREQS, amp=amp, joint=j,
                         phases=json.dumps(phases), ep=EP, cfg=cfg)
                print(f"  {cfg:>10}: 峰值 {pk*1e3:.1f}mrad  crest {phases[0]['crest']:.2f}  "
                      f"Ĝ预测峰 {np.abs(pred).max(axis=(1,2)).mean()*1e3:.0f} mrad/s")
    print(f"\n→ {OUT}   注入窗按关节独立(residual margin ≥ 0.5° 约束)")
    print("先跑 ep0 的 6 条 4mrad pilot; 门禁六项过了再谈 8mrad")


if __name__ == "__main__":
    main()
