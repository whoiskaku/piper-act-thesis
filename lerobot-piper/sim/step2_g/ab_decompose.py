#!/usr/bin/env python3
"""A/B 分解 —— 为什么激励套件上稳健的 3.4–7.3 倍**没有**转化到 pp2 任务域。

    ACT 指令 ──限幅器──▶ 预测关节 q ──刚体/传感器──▶ 预测 IMU
                  A                        B

**A**：限幅后的关节轨迹是否比原 G 更接近**实测编码器**。
      观测量 = 关节角（度）。参与比较的三条：
        · 指令本身（零参数运动学基线，假设完美跟踪）
        · G-v1（MuJoCo 线性伺服，j1 kp 手调 5000，无限幅器）
        · G-v2（限幅器 ⊕ MuJoCo 刚性伺服，XML 原值）

**B**：把**实测编码器**直接喂进刚体/传感器段，能否重现**实测 IMU**。
      运动学渲染：设 `qpos` = 编码器、`qvel` = 中心差分，`mj_forward` 读 `ee_gyro`。
      只需 qpos+qvel（陀螺不依赖 qacc），故该段与执行段完全解耦。
      同时给出「指令的运动学渲染」（即四区域图里的黄线）作对照。

**判读规则（执行前已定，不得事后改）**

| 结果 | 结论与动作 |
|---|---|
| A 就失败 | 限幅器不适用于连续 20 Hz MOVE J 重规划；查状态是否跨指令保持、真实时间戳、多关节并发 |
| A 改善、B 失败 | 限幅器正确，缺口在结构振动 / IMU 支架 / 传感器；**停止改物理参数，转小型神经残差** |
| A、B 都改善但总指标仍差 | 查对齐、坐标变换、评分聚合 |
| 无明确归因 | 接受「仅在单关节激励域有效」，**不继续扩物理模型** |

**S1s07 保留在主分析**；去除高能量集只作**敏感性附注**。

用法:
    python sim/step2_g/ab_decompose.py
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "sim/step2_g"))

from sim.step1_v2_loss import ALIGN_L, BAND, bandpass41, load_calib, render_ideal  # noqa: E402
from sim.step2_gap_baseline import MODEL, aa_down, bare_logs                        # noqa: E402
from sim.step2_fidelity import CALIB, apply_limiter                                 # noqa: E402

FPS = 20.0


def sim_qpos(act: np.ndarray, limiter: str, jkp: dict) -> np.ndarray:
    """跑 MuJoCo, 返回**关节角轨迹**(度, 20 Hz)。A 段用。"""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    for j, kp in jkp.items():
        m.actuator_gainprm[j, 0] = kp
        m.actuator_biasprm[j, 1] = -kp
    a = apply_limiter(act, limiter)
    dt = m.opt.timestep
    nsub = int(round(1 / FPS / dt))
    d = mujoco.MjData(m)
    d.qpos[:6] = a[0, :6]
    mujoco.mj_forward(m, d)
    d.ctrl[:6] = a[0, :6]
    for _ in range(int(0.3 / dt)):
        mujoco.mj_step(m, d)
    out = np.zeros((len(a), 6))
    for t in range(len(a)):
        d.ctrl[:6] = a[t, :6]
        for _ in range(nsub):
            mujoco.mj_step(m, d)
        out[t] = np.degrees(d.qpos[:6])
    return out


def kin_gyro(q_deg: np.ndarray) -> np.ndarray:
    """**运动学渲染**: 关节角轨迹 → 陀螺三轴。只用 qpos + qvel（陀螺不需 qacc）。

    这一段**完全不含执行器/伺服** —— 它就是「刚体 + 传感器」本身，
    所以能把 B 与 A 干净地分开。
    """
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    q = np.radians(np.asarray(q_deg, float))
    T = len(q)
    v = np.zeros_like(q)
    v[1:-1] = (q[2:] - q[:-2]) * FPS / 2.0            # 中心差分
    v[0] = (q[1] - q[0]) * FPS
    v[-1] = (q[-1] - q[-2]) * FPS
    out = np.zeros((T, 3))
    for t in range(T):
        d.qpos[:6] = q[t]
        d.qvel[:6] = v[t]
        mujoco.mj_forward(m, d)
        out[t] = d.sensordata[3:6]
    return out


def load_ep(p: Path):
    d = json.loads(p.read_text())
    fr = d["frames"]
    act = np.asarray([f["a"] for f in fr], float)
    imu = np.asarray([f["imu"] for f in fr], float)[:, 3:6]
    enc = np.asarray([f["state"] for f in fr], float)[:, :6] / 1000.0   # 毫度→度
    return d, fr, act, imu, enc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    a_ = ap.parse_args()
    R, _ = load_calib(CALIB)
    logs = bare_logs()[: a_.n]
    print(f"A/B 分解 —— pp2 开发基准集 {len(logs)} 集（S1s07 保留在主分析）")
    print(f"口径: 带通 {BAND[0]}–{BAND[1]} Hz（B 段）/ 关节角 RMS 度（A 段）\n")

    A, B = [], []
    for p in logs:
        d, fr, act, imu, enc = load_ep(p)
        n = len(fr) - ALIGN_L

        # ── A: 关节轨迹 vs 实测编码器 ──
        q_cmd = np.degrees(act[:, :6])
        q_v1 = sim_qpos(act, "none", {0: 5000.0})
        q_v2 = sim_qpos(act, "a", {})
        k = min(n, len(q_v1), len(q_v2), len(enc) - ALIGN_L)
        e = enc[ALIGN_L:ALIGN_L + k]
        rms = lambda x: float(np.sqrt(np.mean((x[:k] - e) ** 2)))
        A.append((p.stem, rms(q_cmd), rms(q_v1), rms(q_v2)))

        # ── B: 实测编码器 → 刚体/传感器 → 陀螺 ──
        g_enc = kin_gyro(enc[ALIGN_L:ALIGN_L + k])       # 用实测关节
        g_cmd = kin_gyro(q_cmd[:k])                      # 用指令(= 黄线的等价物)
        real = bandpass41(imu[ALIGN_L:ALIGN_L + k])
        be = bandpass41(g_enc @ R.T)
        bc = bandpass41(g_cmd @ R.T)
        t_ = float(np.square(real).sum())
        B.append((p.stem, float(np.square(real - be).sum()),
                  float(np.square(real - bc).sum()), t_))
        print(f"  {p.stem:22s} A(度): 指令 {A[-1][1]:5.2f} · G-v1 {A[-1][2]:5.2f} · "
              f"G-v2 {A[-1][3]:5.2f}   |  B 残差/实测: 编码器 "
              f"{B[-1][1]/t_:5.3f} · 指令 {B[-1][2]/t_:5.3f}", flush=True)

    # ── A 汇总 ──
    Aa = np.array([[x[1], x[2], x[3]] for x in A])
    print(f"\n{'═'*70}\n【A】关节轨迹 vs 实测编码器（RMS 度，越小越好）")
    print(f"  指令(零参数) {Aa[:,0].mean():.3f}   G-v1 {Aa[:,1].mean():.3f}   "
          f"**G-v2 {Aa[:,2].mean():.3f}**")
    print(f"  逐集 G-v2 优于 G-v1: **{int((Aa[:,2] < Aa[:,1]).sum())}/{len(Aa)}**"
          f"   中位改善 {np.median((Aa[:,1]-Aa[:,2])/Aa[:,1])*100:+.1f}%")
    print(f"  逐集 G-v2 优于「指令」: {int((Aa[:,2] < Aa[:,0]).sum())}/{len(Aa)}")

    # ── B 汇总 ──
    Bb = np.array([[x[1], x[2], x[3]] for x in B])
    r_enc = Bb[:, 0].sum() / Bb[:, 2].sum()
    r_cmd = Bb[:, 1].sum() / Bb[:, 2].sum()
    print(f"\n【B】刚体/传感器段：喂**实测编码器**能否重现实测 IMU（残差/实测）")
    print(f"  **喂实测编码器 {r_enc:.4f}**   喂指令 {r_cmd:.4f}")
    print(f"  逐集喂编码器 <1.0（即优于「预测全零」）: "
          f"**{int((Bb[:,0] < Bb[:,2]).sum())}/{len(Bb)}**")
    print(f"  逐集中位 {np.median(Bb[:,0]/Bb[:,2]):.3f}")

    # ── 敏感性（附注，不改主分析）──
    o = np.argsort(-Bb[:, 2])
    rest = o[3:]
    print(f"\n【敏感性附注】去掉 IMU 能量前 3 集（S1s07 仍在主分析内）:")
    print(f"  A: G-v2 优于 G-v1 {int((Aa[rest,2] < Aa[rest,1]).sum())}/{len(rest)}")
    print(f"  B: 喂编码器 {Bb[rest,0].sum()/Bb[rest,2].sum():.4f}")

    # ── A 段的**对照**: 同一检验放在激励套件上 ──
    # 这一步是把「限幅器本身错」与「限幅器扛不住连续 20 Hz 重规划」分开的关键:
    # 套件是**孤立阶跃/扫频**(每次跳变之间有 1.6–2.0 s 静止), pp2 是**每 50 ms
    # 重规划一次**。若 A 在套件上改善、在 pp2 上恶化, 归因就落在重规划上。
    from calibrate import load_suite as _ls
    sd = _ls(True)[:12]
    print(f"\n【A 的对照】同一检验放在**激励套件**上（{len(sd)} 条）")
    sa = []
    for nm, cmd, _real in sd:
        d0 = json.loads((ROOT / f"outputs/step2_g/collect/replay_{nm}.json").read_text())
        enc_s = np.asarray(d0["state"], float)[:, :6] / 1000.0
        q1 = sim_qpos(cmd, "none", {0: 5000.0})
        q2 = sim_qpos(cmd, "a", {})
        kk = min(len(q1), len(q2), len(enc_s))
        f = lambda x: float(np.sqrt(np.mean((x[:kk] - enc_s[:kk]) ** 2)))
        sa.append((f(np.degrees(cmd[:, :6])), f(q1), f(q2)))
    S = np.array(sa)
    print(f"  指令(零参数) {S[:,0].mean():.3f}   G-v1 {S[:,1].mean():.3f}   "
          f"**G-v2 {S[:,2].mean():.3f}**  (RMS 度)")
    print(f"  逐条 G-v2 优于 G-v1: **{int((S[:,2] < S[:,1]).sum())}/{len(S)}**")
    suite_ok = (S[:, 2] < S[:, 1]).sum() > len(S) / 2

    # ── 判读 ──
    a_ok = (Aa[:, 2] < Aa[:, 1]).sum() > len(Aa) / 2
    b_ok = r_enc < 0.5
    print(f"\n{'═'*70}\n【判读】（规则执行前已定）")
    print(f"  A {'改善' if a_ok else '**失败**'} · B {'成立' if b_ok else '**失败**'}")
    if not a_ok:
        print("  ⇒ **A 就失败**。")
        if suite_ok:
            print("     且**套件上 A 是改善的** ⇒ 归因落在「连续 20 Hz 重规划」，"
                  "而非限幅器本身。")
            print("     下一步查：状态是否跨指令保持、真实时间戳、多关节并发。")
        else:
            print("     且**套件上 A 也不改善** ⇒ 限幅器的关节域预测本身就不成立，"
                  "此前的收益只体现在陀螺口径上。")
    elif not b_ok:
        print("  ⇒ **A 改善、B 失败**：限幅器正确，缺口在结构振动 / IMU 支架 / 传感器。")
        print("     **停止改物理参数，转小型神经残差。**")
    else:
        print("  ⇒ **A、B 都改善但总指标仍差**：查对齐、坐标变换、评分聚合。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
