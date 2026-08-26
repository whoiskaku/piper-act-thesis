#!/usr/bin/env python3
"""**G-π-phys-v1** —— π 的连续 ACT 模式所用的物理底座（2026-08-26 冻结）。

```
ACT 连续指令
   → 不启用点到点限幅器，也不启用 G-v3
   → MuJoCo: kp1 = 5000，j2–j6 保持 XML 值
   → 刚体 IMU / 传感器模型
```

## 为什么是这个配置（公平矩阵，报告 §7quinquies）

规划器 3 档 × `kp1` 2 档、其余条件逐字相同：

· **两个数据集的关节角误差都偏好 `kp1=5000`**，三种规划器一致
  （pp2 2.600→0.924 / 套件 1.924→1.844，其余同向）。
· **两个限幅器在 pp2 上都不通过**「多数集改善 + 正中位」：
  点到点 6/17 中位 −13.6 pp（kp=500）/ 6/17 −13.8 pp（kp=5000）；
  流式同步 7/17 −12.7 pp / 6/17 −13.5 pp。
· 故 π 的连续 ACT 模式**两个都不启用**。

## 措辞约束（必须遵守）

- `kp1 = 5000` 只称 **「有效跟踪增益」**，**不解释为真实机械刚度**。
  它是当初为凑 2 帧滞后手调的值，本次是按**公平矩阵的关节角证据**重新采纳，
  与原先的手调理由无关。
- **点到点限幅器不是被否定的**：它在激励套件上有效
  （IMU 残差/实测 2.7388 → 0.5551，kp1=500；2.7706 → 0.4758，kp1=5000），
  **保留为探针 / 长保持模式**，只是 π 的连续 ACT 模式不用它。
- **G-v3（流式同步）已判否、非默认**，代码与产物保留供审计，不物理删除。

## 已知未解

- **IMU 侧的剩余误差不是 kp 能解决的**：pp2 主指标在所有 6 格里都是
  −17.8% ~ −26.1%（逐集中位）。⇒ 下一步是**命令→关节轨迹层的神经残差**，
  不是继续调物理参数。
- pp2 日志**无逐帧时间戳**，全程假设固定 20 Hz。
- ~3 Hz 腕部余振不建模；成因未定。

用法:
    from g_pi_phys_v1 import predict_gyro, PARAMS
    python sim/step2_g/g_pi_phys_v1.py --selftest
"""
from __future__ import annotations
import argparse, hashlib, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "sim/step2_g"))

from sim.step2_gap_baseline import MODEL, aa_down                # noqa: E402

FPS = 20.0

PARAMS = {
    "name": "G-π-phys-v1",
    "frozen": "2026-08-26",
    "supersedes": "G-v2（因规划器与 kp1 混淆而失效，见 G_V2_FREEZE.md 的 superseded 记录）",
    "planner": None,                       # 不启用限幅器 / G-v3
    "kp1_effective_tracking_gain": 5000.0,  # **不是真实机械刚度**
    "kp_j2_j6": "XML 原值 200/200/100/50/50",
    "damping": "XML 原值",
    "sensor": "刚体 IMU / 传感器模型",
    "model_xml": str(MODEL),
    "model_sha16": "7f15d5a97d629896",
}


def model_sha16() -> str:
    return hashlib.sha256(Path(MODEL).read_bytes()).hexdigest()[:16]


def assert_frozen() -> None:
    got = model_sha16()
    if got != PARAMS["model_sha16"]:
        raise SystemExit(
            f"**G-π-phys-v1 冻结校验失败**: {MODEL} 的 sha16 = {got}，"
            f"记录是 {PARAMS['model_sha16']}。模型被改过 ⇒ 这已不是本版本。")


def _sim(act: np.ndarray, want: str, nframes: int | None = None):
    import mujoco
    assert_frozen()
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    kp = PARAMS["kp1_effective_tracking_gain"]          # 只改 j1
    m.actuator_gainprm[0, 0] = kp
    m.actuator_biasprm[0, 1] = -kp
    dt = m.opt.timestep
    nsub = int(round(1 / FPS / dt))
    d = mujoco.MjData(m)
    a = np.asarray(act, float)                          # **不过任何限幅器**
    d.qpos[:6] = a[0, :6]
    mujoco.mj_forward(m, d)
    d.ctrl[:6] = a[0, :6]
    for _ in range(int(0.3 / dt)):
        mujoco.mj_step(m, d)
    q = np.zeros((len(a), 6))
    hi = np.zeros((len(a) * nsub, 6))
    for t in range(len(a)):
        d.ctrl[:6] = a[t, :6]
        for k in range(nsub):
            mujoco.mj_step(m, d)
            hi[t * nsub + k] = d.sensordata[:6]
        q[t] = np.degrees(d.qpos[:6])
    return q if want == "qpos" else aa_down(hi, 1 / dt,
                                            nframes or len(a))[:, 3:6]


def predict_qpos(cmd7) -> np.ndarray:
    """ACT 指令 → 关节角轨迹（度，20 Hz）。**A 段的物理底座。**"""
    return _sim(cmd7, "qpos")


def predict_gyro(cmd7, nframes=None) -> np.ndarray:
    """ACT 指令 → 20 Hz 陀螺三轴（机体系，未旋到 IMU）。"""
    return _sim(cmd7, "gyro", nframes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    assert_frozen()
    print(f"**{PARAMS['name']}** 冻结于 {PARAMS['frozen']}")
    print(f"  模型 sha16 {model_sha16()} ✓")
    print(f"  规划器: {PARAMS['planner']}（**不启用限幅器 / G-v3**）")
    print(f"  kp1 = {PARAMS['kp1_effective_tracking_gain']:.0f}"
          f"  ——「**有效跟踪增益**」，不解释为真实机械刚度")
    print(f"  j2–j6: {PARAMS['kp_j2_j6']}   传感器: {PARAMS['sensor']}")
    if not a.selftest:
        return 0
    import json
    from sim.step1_v2_loss import ALIGN_L
    from sim.step2_gap_baseline import bare_logs
    rs = []
    for p in bare_logs()[:5]:
        d = json.loads(p.read_text()); fr = d["frames"]
        act = np.asarray([f["a"] for f in fr], float)
        enc = np.asarray([f["state"] for f in fr], float)[:, :6] / 1000.0
        q = predict_qpos(act)
        k = min(len(q), len(enc) - ALIGN_L)
        rs.append(float(np.sqrt(np.mean(
            (q[:k] - enc[ALIGN_L:ALIGN_L + k]) ** 2))))
    print(f"\n自检（pp2 前 5 集，关节 RMS 度）: {np.mean(rs):.3f}"
          f"   公平矩阵全 17 集记录 0.924")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
