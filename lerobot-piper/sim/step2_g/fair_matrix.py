#!/usr/bin/env python3
"""**公平矩阵** —— 把「规划器」与「j1 kp」拆成两个独立因子，其余条件完全相同。

## 为什么必须重做

此前所有「G-v1 → G-v2」的比较都**同时改变了两个东西**：

| | 规划器 | j1 kp |
|---|---|---|
| G-v1 | 无限幅 | **5000**（历史手调） |
| G-v2 | 点到点限幅 | **500**（XML 原值） |

⇒ 因果比较无效。2×2 快查已显示 pp2 的 A 段里
**kp 5000→500 贡献 +178%，而限幅器只贡献 +4~14%** ——
此前把账全记在规划层上是**错误归因**。

本模块跑完整矩阵：**规划器 3 档 × kp1 2 档 = 6 格**，其余条件逐字相同。

> `kp = 5000` **只是控制变量**，不代表重新接受这个参数
> （它是当初为凑 2 帧滞后手调的，标定集网格已判 `×1` 最优）。

## 报告项（每格都给）

关节 RMS（度）· IMU 四区域指标 · **逐集改善数** · **逐集中位改善**。

## 随后一次性的 kp 分解（不再展开更广的机理诊断）

| 结果 | 处理 |
|---|---|
| 两数据集的**关节角**都偏好 5000，但**标定集 IMU** 偏好 500 | kp=500 在替传感器/结构误差背锅；物理跟踪层用高刚度，误差交给传感器或神经残差 |
| 标定集与 pp2 的**关节角本身**就偏好不同 kp | 单一固定伺服模型不成立，停止调 kp，转条件化神经残差 |
| G-v3 在相同 kp 下**没有**多数集 + 正中位收益 | 删除 G-v3，不再改规划器 |
| G-v3 在**两个** kp 下都稳定改善 | 接受流式同步结构，但仍**不能用 pp2 选最终 kp** |

用法:
    python sim/step2_g/fair_matrix.py            # 全部
    python sim/step2_g/fair_matrix.py --n 5      # 只用前 5 集(试跑)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "sim/step2_g"))

from sim.step1_v2_loss import ALIGN_L, bandpass41, load_calib, render_ideal  # noqa: E402
from sim.step2_gap_baseline import MODEL, aa_down, bare_logs                  # noqa: E402
from sim.step2_fidelity import CALIB, SEAM_W, apply_limiter, regions, fidelity  # noqa: E402
from g_v3 import plan as v3plan                                               # noqa: E402

FPS = 20.0
COL = ROOT / "outputs/step2_g/collect"
PLANNERS = {                       # 规划器 3 档
    "无限幅":     lambda a: a,
    "点到点限幅": lambda a: apply_limiter(a, "a"),
    "流式同步":   v3plan,
}
KPS = {"kp1=500": {}, "kp1=5000": {0: 5000.0}}        # kp1 2 档（5000 是控制变量）


def run_mj(act, jkp, nframes):
    """**一次** MuJoCo 同时取 qpos(度) 与 20 Hz 陀螺 —— 省一半机时。"""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    for j, kp in jkp.items():
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
    q = np.zeros((len(act), 6))
    hi = np.zeros((len(act) * nsub, 6))
    for t in range(len(act)):
        d.ctrl[:6] = act[t, :6]
        for k in range(nsub):
            mujoco.mj_step(m, d)
            hi[t * nsub + k] = d.sensordata[:6]
        q[t] = np.degrees(d.qpos[:6])
    return q, aa_down(hi, 1 / dt, nframes)[:, 3:6]


def cell_pp2(logs, R, planner, jkp):
    """→ (逐集关节RMS, 逐集主指标相对变化, 逐集难处相对变化)。"""
    rms, seam, top = [], [], []
    for p in logs:
        d = json.loads(p.read_text())
        fr = d["frames"]
        act = np.asarray([f["a"] for f in fr], float)
        real = np.asarray([f["imu"] for f in fr], float)[:, 3:6]
        enc = np.asarray([f["state"] for f in fr], float)[:, :6] / 1000.0
        kin = render_ideal(d["chunks"], len(fr))
        q, g = run_mj(planner(act), jkp, len(fr))
        n = min(len(fr) - ALIGN_L, len(kin), len(g))
        k = min(n, len(enc) - ALIGN_L)
        rms.append(float(np.sqrt(np.mean(
            (q[:k] - enc[ALIGN_L:ALIGN_L + k]) ** 2))))
        tr = {"ideal": bandpass41(kin[:n, 3:6] @ R.T),
              "real": bandpass41(real[ALIGN_L:ALIGN_L + n]),
              "dyn": bandpass41(g[:n]),
              "seams": [c["start"] - ALIGN_L for c in d["chunks"][1:]
                        if 0 <= c["start"] - ALIGN_L < n],
              "n": n, "name": p.stem}
        rg = regions(tr)
        seam.append(fidelity(tr, rg[f"接缝后{SEAM_W}帧"])[2] * 100)
        top.append(fidelity(tr, rg["gap能量前10%"])[2] * 100)
    return np.array(rms), np.array(seam), np.array(top)


def cell_suite(data, R, planner, jkp):
    """→ (逐条关节RMS, 加权 IMU 残差/实测)。"""
    rms, ne, te = [], 0.0, 0.0
    for nm, cmd, real in data:
        d0 = json.loads((COL / f"replay_{nm}.json").read_text())
        enc = np.asarray(d0["state"], float)[:, :6] / 1000.0
        q, g = run_mj(planner(cmd), jkp, len(real))
        k = min(len(q), len(enc))
        rms.append(float(np.sqrt(np.mean((q[:k] - enc[:k]) ** 2))))
        kk = min(len(g), len(real))
        a_ = bandpass41(real[:kk])
        ne += float(np.square(a_ - bandpass41(g[:kk] @ R.T)).sum())
        te += float(np.square(a_).sum())
    return np.array(rms), ne / te


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--nsuite", type=int, default=20)
    a_ = ap.parse_args()
    R, _ = load_calib(CALIB)
    logs = bare_logs()[: a_.n]
    from calibrate import load_suite
    sd = load_suite(True)[: a_.nsuite]
    print(f"公平矩阵：规划器 3 档 × kp1 2 档，其余条件逐字相同")
    print(f"  pp2 开发基准集 {len(logs)} 集 · 激励套件 {len(sd)} 条")
    print(f"  ⚠ kp=5000 **只是控制变量**，不代表重新接受该参数\n")

    P = {}
    for kn, jkp in KPS.items():
        for pn, fn in PLANNERS.items():
            P[(pn, kn)] = cell_pp2(logs, R, fn, jkp)
            print(f"  · pp2 {pn}/{kn} 完成", flush=True)
    S = {}
    for kn, jkp in KPS.items():
        for pn, fn in PLANNERS.items():
            S[(pn, kn)] = cell_suite(sd, R, fn, jkp)
            print(f"  · 套件 {pn}/{kn} 完成", flush=True)

    print(f"\n{'═'*94}\n【pp2】{'':22s}{'关节RMS°':>10s}{'接缝后10帧':>12s}"
          f"{'难处前10%':>11s}{'规划器 vs 无限幅(同 kp)':>26s}")
    print(f"  （四区域两列均为**逐集中位**，不加权，避免被单集主导）")
    for kn in KPS:
        base = P[("无限幅", kn)]
        for pn in PLANNERS:
            r, s_, t_ = P[(pn, kn)]
            if pn == "无限幅":
                w = ""
            else:
                d_ = s_ - base[1]
                w = (f"改善 {int((d_>0).sum())}/{len(d_)}  中位 {np.median(d_):+.1f}pp")
            print(f"  {pn:12s}{kn:12s}{r.mean():>10.3f}"
                  f"{np.median(s_):>11.1f}%{np.median(t_):>10.1f}%{w:>26s}")
    print(f"\n【激励套件】{'':16s}{'关节RMS°':>10s}{'IMU 残差/实测':>14s}")
    for kn in KPS:
        for pn in PLANNERS:
            r, ratio = S[(pn, kn)]
            print(f"  {pn:12s}{kn:12s}{r.mean():>10.3f}{ratio:>14.4f}")

    # ── kp 因子（固定规划器）──
    print(f"\n{'═'*94}\n【kp 分解】固定规划器，只看 kp1 500 → 5000")
    print(f"  {'规划器':12s}{'pp2 关节RMS':>22s}{'套件 关节RMS':>22s}"
          f"{'套件 IMU 残差/实测':>24s}")
    for pn in PLANNERS:
        p5, p50 = P[(pn, "kp1=500")][0].mean(), P[(pn, "kp1=5000")][0].mean()
        s5, s50 = S[(pn, "kp1=500")][0].mean(), S[(pn, "kp1=5000")][0].mean()
        i5, i50 = S[(pn, "kp1=500")][1], S[(pn, "kp1=5000")][1]
        print(f"  {pn:12s}{p5:>10.3f} → {p50:<9.3f}{s5:>10.3f} → {s50:<9.3f}"
              f"{i5:>11.4f} → {i50:<11.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
