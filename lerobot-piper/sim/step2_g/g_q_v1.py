#!/usr/bin/env python3
"""**G-q-v1** —— 命令 → 编码器关节轨迹的**有效预测模型**（2026-08-30 冻结）。

```
ACT の関節指令列 → [固件 MOVE J 加速度限幅(冻结)] → [MuJoCo 位置伺服 θ*] → 関節軌道 + 手首ジャイロ
```

## 名称与口径约束（**必须遵守**）

- 全称只能写 **「命令 → 编码器关节轨迹的**有效预测模型**」**。
- **不得**称为「24 个真实物理参数已辨识的 G」，也不得说「辨识出了真实机械参数」。
  编码器在**电机 / 减速器侧**，看不到 IMU 所见的连杆与支架振动。
- 逐块的可辨识性**完全不同**，报告时不得混为一谈：

| 块 | 跨 3 seed 极差 | 整块置零的代价(训练) | 可否解释数值 |
|---|---|---|---|
| `kp` (j1–j5) | **1.6×** | 主要因 | **已约束**，但仍称「有效跟踪增益」 |
| `c` (damping) | **2.0×** | 主要因 | **已约束**，是 `b+kv` 的合并量 |
| `Ia` (armature) | 217× | **+2.1%** | **不得解释个体值** —— 只以 `(M_link+Ia)/kp` 可辨识 |
| `fc` (frictionloss) | 71× | **−0.2%**（噪声内） | **数值无意义** —— 在本数据里根本没被观测到 |
| `kp6` | — | — | **预先声明的 gauge**，固定 50.0（`params.KP6_GAUGE`） |

- **IMU 侧也更好，但绝对水平仍然差**（订正见下）：

| 数据集 | 指标 | G-π-phys-v1 | **G-q-v1** | |
|---|---|---|---|---|
| 套件留出A | 关节 L | 0.1490 (L=2) | **0.0353** (L=0) | −76% |
| | IMU 等权 | 0.7696 (L=2) | **0.5898** (L=1) | −23% |
| 套件留出B | 关节 L | 0.1272 | **0.0557** | −56% |
| | IMU 等权 | 0.6816 | **0.4962** | −27% |
| **pp2（域外，17 集）** | 关节 RMS° | 0.7644 | **0.4304** | −44% |
| | IMU 残差/实测 | 0.9797 | **0.5643** | −42% |

  两个冻结件**各按自己的定义**评估（phys-v1 不过限幅器 + 帧末读数；
  G-q-v1 过限幅器 + 帧首读数），输入都用名义 20 Hz，对同一组全局位移
  `L ∈ 0..5` 都算、各取自己最优。**phys-v1 在 pp2 的 L=3 得 0.9241，
  与公平矩阵记录的 0.924 完全一致 ⇒ 复现正确。**

> **订正（2026-08-30，同日）**：本 docstring 曾写「IMU 在未知位形上退化 +15.2%」。
> **那是错的。** 那个数字来自把 phys-v1 的 θ 放进 **G-q-v1 的评估壳**里
> （给它加了它本不使用的限幅器、用 G-q-v1 的读数/时基惯例、且未扫位移）。
> 那是个合法的**只比 θ** 的对照，**不是两个冻结件的比较**。
> **同壳只比 θ 时，θ* 的 IMU 确实比 phys-v1 的 θ 差 15.2%（留出A）/ 18.6%（留出B）**
> —— 即编码器拟合出的 θ 对 IMU 略有代价；但**冻结件整体（限幅器 + θ*）在
> IMU 上仍胜出 23–42%**，因为限幅器带来的收益超过 θ 的代价。

- **绝对水平的诚实说法**：IMU 残差/实测最好 **0.564**（pp2）——
  `1.0 = 预测全零`，所以**仍有约 56% 的带内能量未被解释**。
  G-q-v1 **不是**「高保真 IMU 的 G」。
- **无根本权衡**：在编码器拟合出的参数族内部（留出A，28 个横跨全 range 的 trial），
  编码器 L 与 IMU 的相关 **ρ = +0.580** ⇒ **二者同向变好**。
  IMU 只是地板高得多，不是与关节精度冲突。

## 选参规则（防留出污染）

θ* = 3 个独立 seed（各 ~1008 trial）中**训练损失最低**者
（seed 71828182，trial #787，训练 0.030684）。**选参未使用任何留出集。**

## 性能（`RESULT_OPTUNA_IDENT_ENC.md`）

`L` = 逐条等权（激励关节 RMS / 该条指令行程）。

| 数据集 | 指令即输出 | 现 XML | G-π-phys-v1 | **G-q-v1** |
|---|---|---|---|---|
| 训练 p0+p1 (62) | 0.1722 | 0.1530 | 0.0986 | **0.0307** |
| **留出A 位形 p2 (25)** | 0.1738 | 0.1920 | 0.1241 | **0.0356** |
| 留出B 多正弦 (3) | 0.1953 | 0.1217 | 0.1065 | **0.0552** |

留出 A ≈ 训练（0.0356 vs 0.0307）⇒ **位形泛化差距几乎为零**
（对比 IMU 目标那一轮：训练 0.3956 → 留出 A 0.5513，差 39%）。
留出 B 比训练差 77% ⇒ **逐关节同定的参数不完全复合到多关节运动**，此缺口未追。

## 已知未解

- 剩余误差（训练 3.1% / 留出A 3.6% / 留出B 5.5% 行程）**不来自这 4 类参数**。
  已知候选：冻结的 `A_MAX`、未建模柔性，**以及已被无模型证明的被控对象非线性**
  （精确 2.8571× 指令缩放下归一化响应不重合，6/6 关节）——
  **常系数线性位置伺服在原理上覆盖不了后者。**
- **G-π-phys-v1 未被覆盖**，其冻结记录继续有效。G-q-v1 在关节轨迹层严格更好、
  IMU 侧持平；**π 是否改用 G-q-v1 需另行决定。**

用法:
    from g_q_v1 import predict_qpos, predict_gyro, PARAMS
    python sim/step2_g/g_q_v1.py --selftest
"""
from __future__ import annotations
import argparse, gc, hashlib, sys
from pathlib import Path
import numpy as np

# **本环境已知问题**：mujoco+scipy 共存时，`json.loads` 期间触发的垃圾回收会
# SIGSEGV（faulthandler 定位到 `Garbage-collecting` + `json/decoder.py`），
# 并且退出阶段还会抛 `KeyError: 'lmbdas'`。入口处关 GC、收尾用 os._exit 规避。

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g"):
    sys.path.insert(0, str(_p))

FPS = 20.0
READ_OFFSET_STEPS = 1          # 读状态比写指令晚约 1 个 mj 步(实测 t−t_send 中位 1.29 ms)

# θ* —— 24 个数，其中 kp6 是 gauge ⇒ **自由量 23**
THETA_STAR = np.array([
    2070.0093590749184, 1471.961275490957, 2013.98383545991, 228.80179644659222, 777.2482615110063, 50.0,   # kp1-6
    122.00033174786081, 77.00139817338736, 13.744661193214538, 11.282280609887316, 60.5616418069021, 1.4442345835717376,   # c1-6
    1.0939020126092636, 0.0024970024315407215, 0.005817700134589335, 0.0010455210801899497, 0.10604814240516588, 0.479214618243829,   # fc1-6
    0.00011713587493220477, 0.4162666888487766, 0.0008259177313839206, 0.10976079123032775, 0.06243121288999726, 0.0002899116723280812,   # Ia1-6
])

PARAMS = {
    "name": "G-q-v1",
    "full_name": "命令 → 编码器关节轨迹的有效预测模型",
    "frozen": "2026-08-30",
    "selected_by": "3 seed 中**训练损失**最低者；**选参未使用留出集**",
    "seed": 71828182, "trial": 787,
    "L_train": 0.030684, "L_holdoutA": 0.0356, "L_holdoutB": 0.0552,
    # 冻结件对冻结件（各按自己定义、扫全局位移各取最优、名义 20 Hz）
    "imu_holdoutA": 0.5898, "imu_holdoutB": 0.4962, "imu_pp2": 0.5643,
    "imu_holdoutA_phys_v1": 0.7696, "imu_pp2_phys_v1": 0.9797,
    "joint_pp2_deg": 0.4304, "joint_pp2_deg_phys_v1": 0.7644,
    "imu_note": ("冻结件对冻结件：IMU 也更好 23–42%（留出A/B/pp2）。但最好只到 0.564，"
                 "**1.0 = 预测全零 ⇒ 仍有约 56% 带内能量未解释**，不是高保真 IMU 的 G。"
                 "同壳只比 θ 时 θ* 对 IMU 有 15–19% 的代价，被限幅器的收益盖过。"),
    "scope": "主输出 = 关节轨迹；IMU 亦优于 phys-v1，但绝对保真度仍低",
    "n_free": 23, "kp6_gauge": 50.0,
    "limiter": "apply_limiter(·,'a') —— 冻结的固件 MOVE J 加速度限幅",
    "theta_sha16": "ad590ed4ba0b5d33",
    "model_sha16": "7f15d5a97d629896",
    "supersedes": None,      # **不覆盖 G-π-phys-v1**
}


def theta_sha16() -> str:
    return hashlib.sha256(THETA_STAR.tobytes()).hexdigest()[:16]


def model_sha16() -> str:
    from sim.step2_gap_baseline import MODEL
    return hashlib.sha256(Path(MODEL).read_bytes()).hexdigest()[:16]


def assert_frozen() -> None:
    for got, want, what in ((theta_sha16(), PARAMS["theta_sha16"], "θ*"),
                            (model_sha16(), PARAMS["model_sha16"], "piper_model.xml")):
        if got != want:
            raise SystemExit(f"**G-q-v1 冻结校验失败**：{what} 的 sha16 = {got}，"
                             f"记录是 {want}。已被改动 ⇒ 这已不是本版本。")


def _schedule(dts: np.ndarray, mj_dt: float):
    """帧时长 → (写指令的步号, 读状态的步号)。累计换算，不逐帧取整以免漂移。"""
    ts = np.concatenate([[0.0], np.cumsum(np.asarray(dts, float))[:-1]])
    send = np.rint(ts / mj_dt).astype(int)
    send = np.maximum.accumulate(np.maximum(send, 0))
    return send, send + READ_OFFSET_STEPS


def _sim(cmd7, dts=None, t_send=None, t_read=None):
    """→ (关节轨迹 度, 20 Hz 陀螺三轴)。

    `dts`      每帧指令保持时长(秒)。**缺省为名义 1/20 s** —— 部署口径，
               因为 pp2 等日志**没有逐帧时间戳**。
    `t_send/t_read` 若给出实际时间戳则优先，还原**辨识时的口径**。
    """
    import mujoco
    from scipy.signal import butter, filtfilt
    from sim.step2_fidelity import A_MAX
    from optuna_ident_enc import accel_limited_dt
    from sim.step2_gap_baseline import MODEL
    import params as P
    assert_frozen()

    a = np.asarray(cmd7, float)
    n = len(a)
    if t_send is not None and t_read is not None:
        ts, tr = np.asarray(t_send, float)[:n], np.asarray(t_read, float)[:n]
        d_ = np.append(np.diff(ts), np.median(np.diff(ts)))
    else:
        d_ = np.full(n, 1.0 / FPS) if dts is None else np.asarray(dts, float)[:n]
        ts, tr = None, None

    cmd = np.degrees(a[:, :6])
    ref = np.empty((n, 6))
    for j in range(6):
        ref[:, j] = accel_limited_dt(cmd[:, j], float(A_MAX[j]), d_)
    ref = np.radians(ref)

    m = mujoco.MjModel.from_xml_path(str(MODEL))
    P.apply(m, THETA_STAR)                       # **kp6 由 params 压回 gauge**
    mj_dt = m.opt.timestep
    if ts is not None:
        t0 = ts[0]
        send = np.maximum.accumulate(np.maximum(np.rint((ts - t0) / mj_dt).astype(int), 0))
        read = np.maximum.accumulate(np.maximum(np.rint((tr - t0) / mj_dt).astype(int), send))
    else:
        send, read = _schedule(d_, mj_dt)

    d = mujoco.MjData(m)
    d.qpos[:6] = a[0, :6]
    g0 = float(np.clip(a[0, 6], 0, 0.035)) if a.shape[1] > 6 else 0.0
    if m.nq > 7:
        d.qpos[6], d.qpos[7] = g0, -g0
    mujoco.mj_forward(m, d)
    d.ctrl[:6] = ref[0]
    if m.nu > 7:
        d.ctrl[6], d.ctrl[7] = g0, -g0
    for _ in range(int(round(0.3 / mj_dt))):
        mujoco.mj_step(m, d)

    ntot = int(max(send[-1], read[-1])) + 1
    hi = np.empty((ntot, 3))
    q = np.empty((n, 6))
    si = ri = 0
    for s in range(ntot):
        while si < n and send[si] <= s:
            d.ctrl[:6] = ref[si]; si += 1
        while ri < n and read[ri] <= s:
            q[ri] = np.degrees(d.qpos[:6]); ri += 1
        hi[s] = d.sensordata[3:6]
        mujoco.mj_step(m, d)
    while ri < n:
        q[ri] = np.degrees(d.qpos[:6]); ri += 1
    ba = butter(4, 0.45 * FPS / ((1.0 / mj_dt) / 2.0))
    return q, filtfilt(*ba, hi, axis=0)[np.clip(read, 0, ntot - 1)]


def predict_qpos(cmd7, dts=None, t_send=None, t_read=None) -> np.ndarray:
    """ACT 指令 → 关节角轨迹（度）。**这是 G-q-v1 的主输出。**"""
    return _sim(cmd7, dts, t_send, t_read)[0]


def predict_gyro(cmd7, dts=None, t_send=None, t_read=None) -> np.ndarray:
    """ACT 指令 → 20 Hz 陀螺三轴（机体系，未旋到 IMU）。

    比 G-π-phys-v1 好 23–42%（留出A/B/pp2，冻结件对冻结件）。
    ⚠ 但**绝对保真度仍低**：最好只到 0.564 残差/实测，`1.0 = 预测全零`
    ⇒ 仍有约 56% 带内能量未解释。**不得称为「高保真 IMU 的 G」。**
    """
    return _sim(cmd7, dts, t_send, t_read)[1]


def main() -> int:
    gc.disable()                       # 见文件头的环境问题注释
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    assert_frozen()
    print(f"**{PARAMS['name']}** —— {PARAMS['full_name']}（冻结 {PARAMS['frozen']}）")
    print(f"  θ* sha16 {theta_sha16()} ✓   模型 sha16 {model_sha16()} ✓")
    print(f"  自由量 {PARAMS['n_free']}（kp6 = {PARAMS['kp6_gauge']:.0f} 为 gauge）")
    print(f"  选参: {PARAMS['selected_by']}")
    print(f"  ⚠ **不得**称为「24 个真实物理参数已辨识的 G」")
    print(f"  ⚠ 适用范围: {PARAMS['scope']}")
    print(f"  IMU 留出A {PARAMS['imu_holdoutA']:.4f}(phys-v1 {PARAMS['imu_holdoutA_phys_v1']:.4f})"
          f" / 留出B {PARAMS['imu_holdoutB']:.4f}"
          f" / pp2 {PARAMS['imu_pp2']:.4f}(phys-v1 {PARAMS['imu_pp2_phys_v1']:.4f})")
    print(f"  pp2 关节 RMS {PARAMS['joint_pp2_deg']:.4f}° "
          f"(phys-v1 {PARAMS['joint_pp2_deg_phys_v1']:.4f}°)")
    print(f"     {PARAMS['imu_note']}")
    if not a.selftest:
        return 0

    import mujoco
    from optuna_ident_enc import split, prep, evaluate
    from sim.step1_v2_loss import load_calib
    from sim.step2_fidelity import CALIB
    from sim.step2_gap_baseline import MODEL
    R = load_calib(CALIB)[0]
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    tr, ha, hb = split()
    print(f"\n  ── 复现冻结数字（辨识口径：实际时间戳）──")
    print(f"  {'数据集':12s}{'L':>9s}{'记录':>9s}{'差':>9s}{'IMU':>9s}")
    for nm, names, rec in (("训练", tr, PARAMS["L_train"]),
                           ("留出A", ha, PARAMS["L_holdoutA"]),
                           ("留出B", hb, PARAMS["L_holdoutB"])):
        r = evaluate(m, prep(names, m.opt.timestep), THETA_STAR, R)
        print(f"  {nm:12s}{r['loss']:>9.4f}{rec:>9.4f}{r['loss']-rec:>+9.4f}{r['imu']:>9.4f}",
              flush=True)

    print(f"\n  ── **部署口径**：名义 20 Hz（pp2 等日志无逐帧时间戳）──")
    import json
    COL = ROOT / "outputs/step2_g/collect"
    per_a, per_n = [], []
    for nm in tr[:20]:
        dd = json.loads((COL / f"replay_{nm}.json").read_text())
        c7 = np.asarray(dd["cmd"], float)
        enc = np.asarray(dd["state"], float)[:, :6] / 1000.0
        ts, trd = np.asarray(dd["t_send"], float), np.asarray(dd["t"], float)
        k = min(len(c7), len(enc), len(ts), len(trd))
        cm = np.degrees(c7[:k, :6]); tv = cm.max(0) - cm.min(0)
        import json as _j
        tg = _j.loads((ROOT / "outputs/step2_g/manifest.json").read_text())[nm]["target"]
        for lab, kw, acc in (("actual", dict(t_send=ts[:k], t_read=trd[:k]), per_a),
                             ("nominal", {}, per_n)):
            q = predict_qpos(c7[:k], **kw)
            acc.append(float(np.mean([np.sqrt(np.mean((q[:k, j] - enc[:k, j]) ** 2))
                                      / max(tv[j], 1e-9) for j in tg])))
    pa, pn = float(np.mean(per_a)), float(np.mean(per_n))
    print(f"  训练集的 20 条子集（**绝对值与全集 0.0307 不可比**，只看相对差）")
    print(f"    实际时间戳 {pa:.4f}   名义 20 Hz {pn:.4f}"
          f"   **相对代价 {(pn/pa-1)*100:+.1f}%** ⇒ 无逐帧时间戳的日志可用")
    return 0


if __name__ == "__main__":
    sys.stdout.flush()
    _c = main()
    sys.stdout.flush()
    import os
    os._exit(_c)
