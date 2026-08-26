#!/usr/bin/env python3
"""**G-v3 —— 已判否、非默认。代码与产物保留供审计，不物理删除。**

╔══════════════════════════════════════════════════════════════════════════╗
║ 判否依据（公平矩阵，2026-08-26，报告 §7quinquies）                        ║
║   规划器 3 档 × kp1 2 档、其余条件逐字相同。G-v3(流式同步) vs 无限幅：      ║
║     kp1=500 : 主指标改善 **7/17**，逐集中位 **−12.7 pp**                  ║
║     kp1=5000: 主指标改善 **6/17**，逐集中位 **−13.5 pp**                  ║
║   ⇒ **两个 kp 下都不满足「多数集改善 + 正中位」** ⇒ 删除 G-v3，不再改规划器。║
║   关节角上确有微弱优势(2.587 vs 2.600 / 0.916 vs 0.924, 0.3–0.9%)，       ║
║   但不构成收益，且在 IMU 上为负。                                          ║
║                                                                          ║
║ **本文件下方原有的立项理由（「缺口在规划层」）已被推翻** ——                 ║
║ 那条归因把 kp1 的效应（+178%）错记在了规划器头上（实际 +4~14%）。          ║
║ 见报告的全篇订正横幅。B 段结论不受影响。                                    ║
║                                                                          ║
║ 现行物理底座：**g_pi_phys_v1.py**（不启用任何限幅器，kp1=5000）。          ║
╚══════════════════════════════════════════════════════════════════════════╝

（以下为立项时的原始说明，保留供审计。）

**G-v3 候选** —— 修一个已定位的结构错误：规划层。**不覆盖已判定的 G-v2。**

```
20 Hz 连续目标
    ↓
速度连续、跨帧保留状态的重规划器          ← 不再「每个目标都停下」
    ↓
多关节共用时间尺度，同步到最慢关节          ← 不再六个独立 bang-bang
    ↓
MuJoCo + 传感器模型（G-v2 原样，不动）
```

## 为什么是这两条（A/B 分解的结论，报告 §7quater）

· **B 段成立**：喂**实测编码器**进现有刚体/传感器链，残差/实测 **0.3245**、
  17/17 一致 ⇒ 只要关节轨迹正确，现链就能解释约 **68%** 的带通陀螺能量。
  **缺的是确定性的固件规划层，不该先让神经网络顶替它。**
· **A 段在 pp2 失败（0/17）但在套件上改善（9/12）** ⇒ 归因是**连续 20 Hz 重规划**。
  两个具体结构错误：
    ① 现跟踪器是时间最优 bang-bang，**每个目标都要求停下**。套件目标保持
       1.6–2.0 s（对），pp2 每 50 ms 换目标 ⇒ 反复刹车，**近零速帧占 31–52%**；
    ② **逐关节独立**，而真实 MOVE J 通常**同步各关节到最慢的那个**。
       pp2 单集 5 个关节同时动，套件每条只 1 个 ⇒ 该错误在套件上结构性不可见。

## 固定规则（不得在拟合中调整）

1. 保留参考位置**与速度**状态，跨帧连续；
2. 新目标只更新**未来路径**，不强制在当前目标处速度归零；
3. **只有目标持续不变时才减速停下**（`v_ref → 0` 自然发生）；
4. 多关节使用**共同进度**：每帧算各关节的最短用时 `T_j`，取 `λ = max_j T_j`，
   令 `a_j,eff = a_max_j · (T_j/λ)²`（点到点下时间 ∝ 1/√a，故如此缩放使各关节
   同时到达）；
5. **沿用已标定的加速度上限**，不调 `a_max` / `kp` / `damping`；
6. 仍假设**固定 20 Hz** —— pp2 日志无逐帧时间戳，**这是已知限制**。

## 一次性停止条件（执行前已定）

| # | 条件 |
|---|---|
| 1 | 激励套件**不能失去**原有点到点收益 |
| 2 | pp2 的 **A 段**必须**多数集改善且中位改善为正**，最好优于「直接用指令」的 0.633° |
| 3 | pp2 **最终指标**也必须**多数集改善且中位为正** |

**任一关键条件不满足 ⇒ 不再尝试第三种规划器，直接转小型神经残差。**

用法:
    python sim/step2_g/g_v3.py --gate     # 跑三条停止条件
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "sim/step2_g"))

from sim.step1_v2_loss import ALIGN_L, bandpass41, load_calib      # noqa: E402
from sim.step2_gap_baseline import MODEL, aa_down, bare_logs        # noqa: E402
from sim.step2_fidelity import CALIB                                # noqa: E402
from g_v2 import PARAMS as GV2                                      # noqa: E402

FPS = 20.0
A_MAX = np.array(GV2["a_max_deg_s2"], float)      # 沿用已标定值, 规则 5
SUB = 20                                          # 每帧子步数
COL = ROOT / "outputs/step2_g/collect"
SUITE = ROOT / "outputs/step2_g"


def _min_time(x: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
    """双积分器把误差 (x, v) 打到 (0,0) 的最短用时（|u| ≤ a）。

    σ = x + v|v|/(2a)：σ>0 先减速、σ<0 先加速。
    自检：x = −A, v = 0 ⇒ T = 2√(A/a)，与点到点三角廓线一致。
    """
    sig = x + v * np.abs(v) / (2 * a)
    r_pos = np.sqrt(np.maximum(v * v / 2 + a * x, 0.0))
    r_neg = np.sqrt(np.maximum(v * v / 2 - a * x, 0.0))
    T = np.where(sig > 0, (v + 2 * r_pos) / a, (-v + 2 * r_neg) / a)
    return np.maximum(T, 0.0)


def plan(cmd7: np.ndarray, a_max: np.ndarray = A_MAX,
         sub: int = SUB, sync: bool = True) -> np.ndarray:
    """G-v3 规划器：速度连续 + 跨帧保状态 + 多关节共同进度。

    `sync=False` 退化为「逐关节独立的流式限幅」，用来把规则 4 的贡献单独拆出来。
    """
    c = np.degrees(np.asarray(cmd7, float)[:, :6])
    n = len(c)
    dt = 1.0 / (FPS * sub)
    q = c[0].copy()
    v = np.zeros(6)
    out = np.empty((n, 6))
    for t in range(n):
        q0 = c[t - 1] if t > 0 else c[0]
        q1 = c[t]
        vref = (q1 - q0) * FPS                       # 规则 2/3: 目标带速度
        # 规则 4: 共同进度 —— 每帧按最慢关节缩放各自的加速度上限
        if sync:
            T = _min_time(q - q1, v - vref, a_max)
            lam = float(T.max())
            aeff = (a_max * np.clip((T / lam) ** 2, 1e-3, 1.0)
                    if lam > 1e-9 else a_max.copy())
        else:
            aeff = a_max
        for k in range(sub):
            qref = q0 + (q1 - q0) * (k + 1) / sub    # 帧内线性插值参考位置
            x = q - qref
            ev = v - vref
            sig = x + ev * np.abs(ev) / (2 * aeff)
            done = (np.abs(sig) < aeff * dt * dt) & (np.abs(ev) < aeff * dt)
            u = np.where(sig > 0, -aeff, aeff)
            v = np.where(done, vref, v + u * dt)
            q = np.where(done, qref, q + v * dt)
        out[t] = q
    full = np.array(cmd7, float, copy=True)
    full[:, :6] = np.radians(out)
    return full


# G-v1 的定义: 无限幅器 + **j1 的 kp 手调为 5000**(dyn_replay_cmd 的历史默认)。
# 漏掉这个覆盖会把 G-v1 的关节 RMS 从 0.92° 抬到 2.85°, **制造假通过** ——
# 首次写门槛时就犯了这个错, 记此备戒: **基线必须与 A/B 分解里的定义逐字一致**。
GV1_JKP = {0: 5000.0}


def _mj(act: np.ndarray, want: str, nframes: int | None = None,
        jkp: dict | None = None):
    """跑 MuJoCo（G-v2 的刚体/伺服段，原样不动）。want ∈ {'qpos','gyro'}。"""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    for j, kp in (jkp or {}).items():
        m.actuator_gainprm[j, 0] = kp
        m.actuator_biasprm[j, 1] = -kp
    dt = m.opt.timestep
    nsub = int(round(1 / FPS / dt))
    d = mujoco.MjData(m)
    d.qpos[:6] = act[0, :6]
    mujoco.mj_forward(m, d)
    d.ctrl[:6] = act[0, :6]
    for _ in range(int(0.3 / dt)):
        mujoco.mj_step(m, d)
    if want == "qpos":
        o = np.zeros((len(act), 6))
        for t in range(len(act)):
            d.ctrl[:6] = act[t, :6]
            for _ in range(nsub):
                mujoco.mj_step(m, d)
            o[t] = np.degrees(d.qpos[:6])
        return o
    hi = np.zeros((len(act) * nsub, 6))
    for t in range(len(act)):
        d.ctrl[:6] = act[t, :6]
        for k in range(nsub):
            mujoco.mj_step(m, d)
            hi[t * nsub + k] = d.sensordata[:6]
    return aa_down(hi, 1 / dt, nframes or len(act))[:, 3:6]


def predict_qpos(cmd7):
    return _mj(plan(cmd7), "qpos")


def predict_gyro(cmd7, nframes=None):
    return _mj(plan(cmd7), "gyro", nframes)


# ────────────────────── 三条停止条件（执行前已定）──────────────────────
def _traces(p, R, plan_fn, jkp=None):
    """组装与 step2_fidelity.episode_traces 同口径的 trace（避免循环 import）。"""
    from sim.step1_v2_loss import render_ideal
    d = json.loads(p.read_text())
    fr = d["frames"]
    act = np.asarray([f["a"] for f in fr], float)
    real = np.asarray([f["imu"] for f in fr], float)[:, 3:6]
    kin = render_ideal(d["chunks"], len(fr))
    n = min(len(fr) - ALIGN_L, len(kin))
    dyn = _mj(plan_fn(act), "gyro", len(fr), jkp)
    n = min(n, len(dyn))
    return {"ideal": bandpass41(kin[:n, 3:6] @ R.T),
            "real": bandpass41(real[ALIGN_L:ALIGN_L + n]),
            "dyn": bandpass41(dyn[:n]),
            "seams": [c["start"] - ALIGN_L for c in d["chunks"][1:]
                      if 0 <= c["start"] - ALIGN_L < n],
            "n": n, "name": p.stem}


def gate(n_ep=None, n_suite=12):
    from sim.step2_fidelity import regions, fidelity, SEAM_W, apply_limiter
    from calibrate import load_suite
    R, _ = load_calib(CALIB)
    v2plan = lambda a: apply_limiter(a, "a")           # G-v2 的规划段
    hard = f"接缝后{SEAM_W}帧"

    # ── 门槛 1: 激励套件不能失去原有点到点收益 ──
    print("【门槛 1】激励套件：G-v3 不能失去 G-v2 的点到点收益")
    sd = load_suite(True)[:n_suite]
    n1 = n2 = n3 = t3 = 0.0
    aw = 0
    for nm, cmd, real in sd:
        d0 = json.loads((COL / f"replay_{nm}.json").read_text())
        enc = np.asarray(d0["state"], float)[:, :6] / 1000.0
        g2 = _mj(v2plan(cmd), "gyro", len(real))
        g3 = _mj(plan(cmd), "gyro", len(real))
        k = min(len(g2), len(g3), len(real))
        a_ = bandpass41(real[:k])
        n1 += float(np.square(a_ - bandpass41(g2[:k] @ R.T)).sum())
        n2 += float(np.square(a_ - bandpass41(g3[:k] @ R.T)).sum())
        t3 += float(np.square(a_).sum())
        q2 = _mj(v2plan(cmd), "qpos"); q3 = _mj(plan(cmd), "qpos")
        kk = min(len(q2), len(q3), len(enc))
        r2 = np.sqrt(np.mean((q2[:kk] - enc[:kk]) ** 2))
        r3 = np.sqrt(np.mean((q3[:kk] - enc[:kk]) ** 2))
        aw += r3 <= r2 * 1.10                          # 允许 10% 退化
    print(f"  陀螺 残差/实测: G-v2 {n1/t3:.4f} → **G-v3 {n2/t3:.4f}**")
    print(f"  关节 RMS 不退化超 10% 的条数: **{aw}/{len(sd)}**")
    g1_ok = (n2 <= n1 * 1.10) and aw > len(sd) / 2
    print(f"  ⇒ {'✓ 通过' if g1_ok else '**✗ 未通过**'}\n")

    # ── 门槛 2: pp2 的 A 段 ──
    print("【门槛 2】pp2 A 段：多数集改善 且 中位改善为正（最好优于指令 0.633°）")
    logs = bare_logs()[:n_ep]
    rows = []
    for p in logs:
        d = json.loads(p.read_text()); fr = d["frames"]
        act = np.asarray([f["a"] for f in fr], float)
        enc = np.asarray([f["state"] for f in fr], float)[:, :6] / 1000.0
        q1 = _mj(act, "qpos", jkp=GV1_JKP)             # **G-v1**(含 j1 kp=5000)
        q2 = _mj(v2plan(act), "qpos")
        q3 = _mj(plan(act), "qpos")
        k = min(len(q1), len(q2), len(q3), len(enc) - ALIGN_L)
        e = enc[ALIGN_L:ALIGN_L + k]
        f = lambda x: float(np.sqrt(np.mean((x[:k] - e) ** 2)))
        rows.append((p.stem, f(np.degrees(act[:, :6])), f(q1), f(q2), f(q3)))
        print(f"  {p.stem:22s} 指令 {rows[-1][1]:5.2f} · G-v1 {rows[-1][2]:5.2f} · "
              f"G-v2 {rows[-1][3]:5.2f} · **G-v3 {rows[-1][4]:5.2f}**", flush=True)
    Ar = np.array([r[1:] for r in rows])
    win = int((Ar[:, 3] < Ar[:, 1]).sum())
    med = float(np.median((Ar[:, 1] - Ar[:, 3]) / Ar[:, 1]) * 100)
    print(f"  均值: 指令 {Ar[:,0].mean():.3f} · G-v1 {Ar[:,1].mean():.3f} · "
          f"G-v2 {Ar[:,2].mean():.3f} · **G-v3 {Ar[:,3].mean():.3f}**")
    print(f"  G-v3 优于 G-v1: **{win}/{len(Ar)}**   中位改善 **{med:+.1f}%**"
          f"   优于「指令」: {int((Ar[:,3] < Ar[:,0]).sum())}/{len(Ar)}")
    g2_ok = win > len(Ar) / 2 and med > 0
    print(f"  ⇒ {'✓ 通过' if g2_ok else '**✗ 未通过**'}\n")

    # ── 门槛 3: pp2 最终指标 ──
    print("【门槛 3】pp2 最终指标（接缝后 10 帧）：多数集改善 且 中位为正")
    dd = []
    for p in logs:
        t1 = _traces(p, R, lambda a: a, GV1_JKP)       # **G-v1**(含 j1 kp=5000)
        t3 = _traces(p, R, plan)                       # G-v3
        f1 = fidelity(t1, regions(t1)[hard])[2]
        f3 = fidelity(t3, regions(t3)[hard])[2]
        dd.append((f3 - f1) * 100)
    dd = np.array(dd)
    print(f"  逐集配对差 (G-v3 − G-v1): 正向 **{int((dd>0).sum())}/{len(dd)}**  "
          f"中位 **{np.median(dd):+.1f}pp**  均值 {dd.mean():+.1f}pp")
    g3_ok = (dd > 0).sum() > len(dd) / 2 and np.median(dd) > 0
    print(f"  ⇒ {'✓ 通过' if g3_ok else '**✗ 未通过**'}\n")

    print("═" * 66)
    ok = g1_ok and g2_ok and g3_ok
    print(f"三条停止条件: 1 {'✓' if g1_ok else '✗'} · 2 {'✓' if g2_ok else '✗'} · "
          f"3 {'✓' if g3_ok else '✗'}")
    print("⇒ " + ("**全部通过 —— G-v3 成立，可考虑采 holdout**" if ok else
                  "**任一关键条件不满足 ⇒ 不再尝试第三种规划器，转小型神经残差**"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--nsuite", type=int, default=12)
    a = ap.parse_args()
    if not a.gate:
        print(__doc__.split("用法:")[0])
        return 0
    return gate(a.n, a.nsuite)


if __name__ == "__main__":
    raise SystemExit(main())
