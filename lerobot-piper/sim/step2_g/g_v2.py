#!/usr/bin/env python3
"""**G-v2 —— 已冻结的 G**（2026-08-26 冻结，冻结记录见 `reports/G_V2_FREEZE.md`）。

    指令 → [加速度限幅 a_j]（运动学，位形无关） → [MuJoCo 刚性伺服，XML 原值] → 关节

这是一个**具名、哈希钉死**的产物，不是一组命令行开关。凡是要引用「G 的表现」，
都应经由本模块，而不是 `step2_fidelity.py --limiter ... --kpmul ...` 的临时组合。

## 冻结了什么

| 项 | 值 | 来源 |
|---|---|---|
| 加速度上限 `a_j` (°/s²) | 319 / 319 / 307 / 346 / 391 / 307 | 90 条套件的**大幅值真阶跃**二参数拟合 |
| 纯传输延迟 | **不加** | 与 MuJoCo 伺服自身滞后重复计入，标定集 0.5198→0.8140 |
| 伺服 `kp` | XML 原值 500/200/200/100/50/50（**不用手调的 j1=5000**） | 标定集网格 `kp×1` 最优 |
| `damping` | XML 原值 500/5/20/5/5/5 | 同上，`damp×1` 最优 |
| 柔性自由度 | **不加** | G-flex 最小版判否（报告 §7ter） |
| 模型文件 | `sim/piper_model.xml` sha256[:16] = `7f15d5a97d629896` | — |
| 冻结时 git | `5efb69d` | — |

## 冻结时的数字（**口径必须一并引用**）

· **标定集**（90 条套件中的 79 条，`calibration`）：带通 0.5–9 Hz 陀螺
  残差/实测 = **0.5198**；无限幅器的同一模型是 **2.5718**（>1 ⇒ 比输出全零还差）。
· **pp2 开发基准集**（17 集裸 ACT，**不是 holdout**）：接缝后 10 帧 **−17.1%**、
  gap 能量前 10% **+4.5%**（当前 G 为 −30.9% / +2.5%）。

> 那 17 集**已被用于选择**限幅器、kp 和下一步结构，**不能作为留出性能引用**。
> 真正的 holdout 在本结构冻结**之后**另采 —— 这正是冻结的目的。

## 明确未冻结 / 未声称的

· `a_j` 中 **j4 弱可辨识**（±60% 只让阶跃 RMS 变 1.3×，其余关节 3.7–18.6×）。
· **未**声称测出传动刚度与电机惯量；G-v2 里根本没有柔性自由度。
· 观测到的 ~3 Hz 腕部余振 **G-v2 不建模**，其成因未定
  （IMU 支架模态 / 整臂弯曲模 / 重力补偿均未排除）。

用法:
    from g_v2 import predict_gyro, PARAMS
    gyro20 = predict_gyro(cmd7, nframes)      # (N,3) 20 Hz 陀螺, 机体系
    python sim/step2_g/g_v2.py --selftest     # 复现冻结时的标定集数字
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "sim/step2_g"))

from sim.step2_gap_baseline import MODEL, aa_down                # noqa: E402
from accel_limit import accel_limited                            # noqa: E402

FPS = 20.0

PARAMS = {
    "name": "G-v2",
    "frozen": "2026-08-26",
    "git": "5efb69d",
    "model_xml": str(MODEL),
    "model_sha16": "7f15d5a97d629896",
    "a_max_deg_s2": [319.0, 319.0, 307.0, 346.0, 391.0, 307.0],
    "pure_delay_s": None,          # 明确不加
    "kp_mul": 1.0,                 # 用 XML 原值, 不用手调的 j1=5000
    "damping_mul": 1.0,
    "flex_dof": None,              # 明确不加
    "calib_residual_ratio": 0.5198,
    "calib_residual_ratio_no_limiter": 2.5718,
    "devbench_seam10_pct": -17.1,
    "devbench_top10_pct": +4.5,
}


def model_sha16() -> str:
    return hashlib.sha256(Path(MODEL).read_bytes()).hexdigest()[:16]


def assert_frozen() -> None:
    """模型文件被改动过就**拒绝运行** —— 冻结必须是可执行的，不能只是一句话。"""
    got = model_sha16()
    if got != PARAMS["model_sha16"]:
        raise SystemExit(
            f"**G-v2 冻结校验失败**: {MODEL} 的 sha16 = {got}，"
            f"冻结记录是 {PARAMS['model_sha16']}。\n"
            f"模型被改过 ⇒ 这已不是 G-v2。要么恢复文件，要么另立 G-v3 并重新冻结。")


def limit(cmd7: np.ndarray) -> np.ndarray:
    """指令 → 固件 MOVE J 的加速度限幅。**G-v2 的第一段。**"""
    out = np.array(cmd7, float, copy=True)
    for j in range(6):
        out[:, j] = np.radians(
            accel_limited(np.degrees(np.asarray(cmd7, float)[:, j]),
                          PARAMS["a_max_deg_s2"][j]))
    return out


def predict_gyro(cmd7: np.ndarray, nframes: int | None = None) -> np.ndarray:
    """G-v2 的完整前向：限幅 → 刚性 MuJoCo → 20 Hz 陀螺三轴（机体系，未旋到 IMU）。"""
    import mujoco
    assert_frozen()
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    act = limit(cmd7)
    n = nframes if nframes is not None else len(act)
    dt = m.opt.timestep
    nsub = int(round(1 / FPS / dt))
    d = mujoco.MjData(m)
    d.qpos[:6] = act[0, :6]
    mujoco.mj_forward(m, d)
    d.ctrl[:6] = act[0, :6]
    for _ in range(int(0.3 / dt)):
        mujoco.mj_step(m, d)
    hi = np.zeros((len(act) * nsub, 6))
    for t in range(len(act)):
        d.ctrl[:6] = act[t, :6]
        for k in range(nsub):
            mujoco.mj_step(m, d)
            hi[t * nsub + k] = d.sensordata[:6]
    return aa_down(hi, 1 / dt, n)[:, 3:6]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="在标定集上复现冻结时的残差比")
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    assert_frozen()
    print(f"**{PARAMS['name']}** 冻结于 {PARAMS['frozen']}  git {PARAMS['git']}")
    print(f"  模型 sha16 {model_sha16()} ✓")
    print(f"  a_max = {PARAMS['a_max_deg_s2']} °/s²")
    print(f"  纯延迟 {PARAMS['pure_delay_s']} · kp×{PARAMS['kp_mul']} · "
          f"damping×{PARAMS['damping_mul']} · 柔性 {PARAMS['flex_dof']}")
    if not a.selftest:
        return 0
    from sim.step1_v2_loss import bandpass41, load_calib
    from calibrate import load_suite, CALIB
    R, _ = load_calib(CALIB)
    data = load_suite(True)
    if a.n:
        data = data[: a.n]
    num = den = 0.0
    for nm, cmd, real in data:
        sim = predict_gyro(cmd, len(real))
        k = min(len(sim), len(real))
        x = bandpass41(real[:k]); y = bandpass41(sim[:k] @ R.T)
        num += float(np.square(x - y).sum()); den += float(np.square(x).sum())
    got = num / den
    exp = PARAMS["calib_residual_ratio"]
    print(f"\n自检（标定集 {len(data)} 条）: 残差/实测 = {got:.4f}"
          f"   冻结记录 {exp:.4f}   差 {abs(got-exp):.4f}")
    print("  " + ("✓ 复现" if abs(got - exp) < (0.02 if not a.n else 0.20)
                  else "**✗ 不复现 —— 冻结产物与代码已不一致**"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
