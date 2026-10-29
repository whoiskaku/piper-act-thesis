#!/usr/bin/env python3
"""**命令 → 编码器关节轨迹层的有效参数辨识**（Optuna，23 个自由量）。

## 名称约束（必须遵守）

本实验只能称 **「命令 → 编码器关节轨迹层的**有效**参数辨识」**。

**即使成功，也不得说「辨识出了真实机械参数」**：编码器在电机 / 减速器侧，
看不到 IMU 所见的连杆、支架振动；这里拟合到的是**能复现电机侧轨迹的有效值**。

## 为什么做这一步

IMU 目标的 2890 trial 已判否，且主发现是**参数不可辨识**
（近优 10% 集内 `armature` 跨 648–8638×，见 `PREREG_OPTUNA_IDENT.md` §7.3）。
但那个结论有一个未排除的解释：**我们一直用腕部 3 轴 IMU 去间接拟合伺服参数**，
而伺服参数对关节跟踪是直接作用。本实验就是要把两种解释分开：

| 观察 | 判定 |
|---|---|
| 编码器留出改善 **且** 近优参数明显收窄 **且** 跨 seed 稳定 | 原 IMU 目标确实条件较差 |
| 编码器预测改善 **但** 参数仍跨几十倍 | 只得到**有效预测模型**，得不到唯一物理参数 |
| 编码器**仍不能**约束参数 | **停止继续堆 trial**，转电流观测 |
| 编码器改善 **但** IMU 退化 | 只改善了电机侧轨迹，**不能升级为完整 G** |

> **不预设 `armature` 一定会收窄**：MOVE J 规划 + 20 Hz 编码器仍可能让它
> 只以 `(M + Ia)/kp` 这样的组合出现，个体不可分。

约束式 Pareto（「编码器误差不劣于最优 5%，其中选 IMU 最小」）**只有在编码器
已经约束住参数之后**才考虑；现在不上双目标。

## 参数化：24 个数，**23 个自由量**（遵循 `params.py`）

| 块 | MuJoCo 字段 | 自由? | 范围（log 均匀） |
|---|---|---|---|
| `kp1–kp5` | `gainprm[j,0]`，`biasprm[j,1]=−kp` | ✓ | XML × [0.2, 20] |
| **`kp6`** | 同上 | **✗ 固定 50.0** | `params.KP6_GAUGE`（预先声明的 gauge） |
| `c1–c6` | `dof_damping[j]`，`biasprm[j,2]≡0` | ✓ | XML × [0.05, 20] |
| `fc1–fc6` | `dof_frictionloss[j]` | ✓ | [1e-3, 5.0] N·m |
| `Ia1–Ia6` | `dof_armature[j]` | ✓ | [1e-4, 1.0] kg·m² |

`params.py` 已声明的三处结构性事实，本模块**全部遵守**（直接调用 `params.apply`）：

1. **`b` 与 `kv` 精确简并** ⇒ `biasprm[:,2] ≡ 0`，全部记入 `dof_damping`。
   （已核验 XML 里 `biasprm[:6,2]` 本就全为 0。）
2. **j5/j6 的 `Ia` 与 `I_link` 结构性混淆** ⇒ 只能报 `I_eff`，**声明未分离**。
3. **j6 的 `kp` 与 `c` 只有一个自由度** ⇒ `kp6` 固定为 gauge，自由量是
   `τ6 = c6/kp6`。**上一轮 IMU 搜索放开了 kp6（最优值 134.3 / 439），
   违反了这条已声明的 gauge —— 那部分「不可辨识」是自造的**，本轮修正。

## 预注册（跑第一个 trial 之前定死）

**切分**：与 IMU 轮**完全相同** —— 训练 62（位形 p0+p1）/ 留出 A 25（位形 p2）/
留出 B 3（多正弦）。留出集在优化中一次都不评估。

**时基**：用日志里的实际 `t_send`（指令写总线）与 `t`（读状态），
**不假定严格 20 Hz**。实测 dt 中位 50.112 ms、p99 55.0、max 62.6，
1.63% 的帧偏离 >5 ms；在 2164 帧的慢扫上按名义 50 ms 会累积漂约 5 帧。
仿真按 mj 步网格还原事件：帧 k 在 `t_send[k]` 写 ctrl、在 `t[k]` 读 qpos。

**主损失**（只用关节**位置**轨迹，不做 20 Hz 差分速度/加速度）：

```
e_i = mean_{j ∈ 激励关节}  RMS_t( q_sim[t,j] − q_enc[t,j] ) / 行程_ij
L   = mean_i e_i                                  # 每条序列等权
```

`行程_ij` = 该条序列**指令**在关节 j 上的峰峰值（与参数无关 ⇒ 归一化不随 θ 变）。
激励关节取 `manifest[name]["target"]`；已交叉验证：**0/90** 条存在非 target 关节
行程超过 target 行程 5% 的情况。

**不逐集搜索滞后。** 时基由实际时间戳还原，不再补任何自由滞后。

**稳态误差单独报告，不进损失**：静止段（指令速度≈0）与匀速段（扫描的恒速段）
的平均带符号误差，以及静置后的静态平衡偏差。

**IMU 指标同一次仿真一并算出，但不参与选参**（记为 `user_attr`）。

**预算固定**：3 个独立 seed × 1000 trial。**结束后不因结果接近再追加。**

### 判读阈值（预先定死）

- **留出改善**：留出 A 的 `L` 优于 G-π-phys-v1，**且**优于「指令即输出」。
- **明显收窄**：近优 10% 集内，23 个自由参数的**倍差中位数 ≤ 10×**
  （IMU 轮的同一统计量作对照）。
- **跨 seed 稳定**：3 个 seed 各自最优参数，**逐参数三者极差 ≤ 3×**。
- **IMU 退化**：最优参数的训练集 IMU 等权均值比 G-π-phys-v1 差 **>5%**。

用法:
    python sim/step2_g/optuna_ident_enc.py --selftest
    python sim/step2_g/optuna_ident_enc.py --baseline
    python sim/step2_g/optuna_ident_enc.py --run --trials 1000 --workers 16
    python sim/step2_g/optuna_ident_enc.py --report
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g"):
    sys.path.insert(0, str(_p))

FPS = 20.0
SEEDS = (20260830, 71828182, 31415926)          # 3 个独立 seed，预算固定
N_TRIALS = 1000                                  # 每 seed
PENALTY = 10.0
SETTLE_S = 0.3
COL = ROOT / "outputs/step2_g/collect"
SUITE = ROOT / "outputs/step2_g"
OUT = ROOT / "outputs/step2_g/optuna_enc"

XML_KP = np.array([500.0, 200.0, 200.0, 100.0, 50.0, 50.0])
XML_DAMP = np.array([500.0, 5.0, 20.0, 5.0, 5.0, 5.0])
Z6 = np.zeros(6)
BASE_XML = np.concatenate([XML_KP, XML_DAMP, Z6, Z6])
BASE_PHYS_V1 = np.concatenate([[5000.0, 200, 200, 100, 50, 50], XML_DAMP, Z6, Z6])
# 上一轮 IMU 目标的最优（**违反 kp6 gauge**，仅作交叉参考，kp6 会被 params.apply 压回 50）
IMU_BEST_2890 = np.concatenate([
    [969.3, 1802.0, 286.2, 20.14, 14.67, 134.3],
    [37.63, 67.97, 11.29, 0.5026, 0.7437, 11.01],
    [0.01632, 1.182, 0.00102, 0.0621, 0.01316, 0.002451],
    [0.009656, 0.0005271, 0.0001669, 0.0002067, 0.001009, 0.001525]])

LO = np.concatenate([XML_KP * 0.2, XML_DAMP * 0.05, np.full(6, 1e-3), np.full(6, 1e-4)])
HI = np.concatenate([XML_KP * 20.0, XML_DAMP * 20.0, np.full(6, 5.0), np.full(6, 1.0)])


# ── 限幅器：按实际帧长 ────────────────────────────────────────────────────
def accel_limited_dt(cmd_deg, a_max: float, dts, sub: int = 20):
    """与冻结的 `accel_limit.accel_limited` 同构，但每帧用**实际**时长。

    `dts` 全为 1/FPS 时**逐位还原**冻结版本（`--selftest` 验证）。
    """
    q, v, out = float(cmd_deg[0]), 0.0, []
    for tgt, T in zip(cmd_deg, dts):
        dt = float(T) / sub
        for _ in range(sub):
            e = float(tgt) - q
            d_stop = v * abs(v) / (2.0 * a_max)
            s = e - d_stop
            if abs(s) < a_max * dt * dt and abs(v) < a_max * dt:
                q, v = float(tgt), 0.0
            else:
                v += float(np.clip(a_max * np.sign(s), -a_max, a_max)) * dt
                q += v * dt
        out.append(q)
    return np.asarray(out)


# ── 数据 ────────────────────────────────────────────────────────────────
def split():
    from optuna_ident import split as _s
    return _s()


def prep(names: list[str], mj_dt: float = 0.002) -> list[dict]:
    """全部与 θ 无关的量只算一次：限幅参考、事件步网格、行程、稳态掩码。"""
    from sim.step1_v2_loss import bandpass41
    from sim.step2_fidelity import A_MAX
    man = json.loads((SUITE / "manifest.json").read_text())
    out = []
    for nm in names:
        d = json.loads((COL / f"replay_{nm}.json").read_text())
        cmd7 = np.asarray(d["cmd"], float)
        cmd = np.degrees(cmd7[:, :6])
        enc = np.asarray(d["state"], float)[:, :6] / 1000.0
        real = np.asarray(d["imu"], float)[:, 3:6]
        ts = np.asarray(d["t_send"], float)
        tr = np.asarray(d["t"], float)
        n = min(len(cmd), len(enc), len(real), len(ts), len(tr))
        cmd, enc, real, ts, tr = cmd[:n], enc[:n], real[:n], ts[:n], tr[:n]

        # 帧 k 的指令保持时长 = t_send[k+1] − t_send[k]；末帧用中位
        dts = np.diff(ts)
        dts = np.append(dts, np.median(dts))

        ref = np.empty((n, 6))
        for j in range(6):
            ref[:, j] = accel_limited_dt(cmd[:, j], float(A_MAX[j]), dts)
        t0 = ts[0]
        send_step = np.rint((ts - t0) / mj_dt).astype(int)
        read_step = np.rint((tr - t0) / mj_dt).astype(int)
        send_step = np.maximum.accumulate(np.maximum(send_step, 0))
        read_step = np.maximum.accumulate(np.maximum(read_step, send_step))

        tg = list(man[nm]["target"])
        travel = cmd.max(0) - cmd.min(0)
        dc = np.abs(np.diff(cmd, axis=0, prepend=cmd[:1]))
        still = np.zeros((n, 6), bool)
        const = np.zeros((n, 6), bool)
        for j in tg:
            still[:, j] = _runs(dc[:, j] < 0.005, 10, 5)
            nz = dc[dc[:, j] > 0.005, j] if (dc[:, j] > 0.005).any() else np.array([])
            if nz.size:
                med = float(np.median(nz))
                const[:, j] = _runs((dc[:, j] > 0.5 * med) & (dc[:, j] < 1.5 * med), 20, 10)
        a = bandpass41(real)
        out.append({"name": nm, "n": n, "ref": np.radians(ref), "cmd": cmd,
                    "enc": enc, "enc0": enc[0], "g0": float(np.clip(cmd7[0, 6], 0, 0.035))
                    if cmd7.shape[1] > 6 else 0.0,
                    "send_step": send_step, "read_step": read_step,
                    "target": tg, "travel": travel,
                    "still": still, "const": const,
                    "abp": a, "ten": float(np.square(a).sum())})
    out.sort(key=lambda x: -int(x["read_step"][-1]))     # 长的先跑, 便于并行填谷
    return out


def _runs(mask: np.ndarray, minlen: int, skip: int) -> np.ndarray:
    """保留长度 ≥ minlen 的 True 段，并丢掉每段开头 skip 帧（去瞬态）。"""
    out = np.zeros_like(mask)
    i, n = 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if j - i >= minlen:
            out[i + skip:j] = True
        i = j
    return out


# ── 前向 ────────────────────────────────────────────────────────────────
def roll(m, it) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """事件精确回放 → (读数时刻的 qpos 度, 读数时刻的陀螺, 静置后平衡位形)。

    帧 k：在 `t_send[k]` 写 ctrl、在 `t[k]` 读 qpos —— 两者落在同一 mj 步时
    **先写后读**，故读到的仍是上一条指令下的状态（与真机因果一致）。
    q 与陀螺**都在步进前记录**，即时刻 `s·mj_dt`。
    """
    import mujoco
    from scipy.signal import butter, filtfilt
    dt = m.opt.timestep
    d = mujoco.MjData(m)
    d.qpos[:6] = np.radians(it["enc0"])                  # **实测**初始位形
    if m.nq > 7:
        d.qpos[6], d.qpos[7] = it["g0"], -it["g0"]
    mujoco.mj_forward(m, d)
    d.ctrl[:6] = it["ref"][0]
    if m.nu > 7:
        d.ctrl[6], d.ctrl[7] = it["g0"], -it["g0"]
    for _ in range(int(round(SETTLE_S / dt))):
        mujoco.mj_step(m, d)
    q_eq = np.degrees(d.qpos[:6].copy())                 # 静态平衡诊断

    send, read = it["send_step"], it["read_step"]
    ntot = int(max(send[-1], read[-1])) + 1
    hi = np.empty((ntot, 3))
    q = np.empty((len(read), 6))
    si = ri = 0
    ns, nr = len(send), len(read)
    for s in range(ntot):
        while si < ns and send[si] <= s:
            d.ctrl[:6] = it["ref"][si]
            si += 1
        while ri < nr and read[ri] <= s:
            q[ri] = np.degrees(d.qpos[:6])
            ri += 1
        hi[s] = d.sensordata[3:6]
        mujoco.mj_step(m, d)
    while ri < nr:                                        # 末帧兜底
        q[ri] = np.degrees(d.qpos[:6])
        ri += 1
    ba = butter(4, 0.45 * FPS / ((1.0 / dt) / 2.0))
    g = filtfilt(*ba, hi, axis=0)[np.clip(read, 0, ntot - 1)]
    return q, g, q_eq


def evaluate(m, items, theta, R, want_diag: bool = False) -> dict:
    """→ {loss, per, imu, (diag)}。发散 ⇒ 全 PENALTY。

    ⚠ 参数顺序是 `(m, items, **theta**, **R**)` —— 与 `optuna_ident.evaluate`
    的 `(m, items, R, p)` **相反**。下面的断言就是为了让顺序写错时立刻炸掉，
    而不是把 3×3 的旋转矩阵当 θ 静默算出垃圾（已实际踩过一次）。
    """
    import params as P
    from sim.step1_v2_loss import bandpass41
    theta = np.asarray(theta, float)
    if theta.shape != (P.N_P,):
        raise TypeError(f"theta 形状应为 ({P.N_P},)，收到 {theta.shape} —— "
                        f"很可能把 theta 和 R 的位置写反了")
    if np.shape(R) != (3, 3):
        raise TypeError(f"R 形状应为 (3,3)，收到 {np.shape(R)} —— 同上")
    P.apply(m, theta)                                    # **kp6 由 params 压回 gauge**
    per, imu_per, dg = [], [], []
    for it in items:
        try:
            q, g, q_eq = roll(m, it)
        except Exception:                                # noqa: BLE001
            return {"loss": PENALTY, "per": np.full(len(items), PENALTY),
                    "imu": PENALTY, "diag": []}
        if not (np.isfinite(q).all() and np.isfinite(g).all()):
            return {"loss": PENALTY, "per": np.full(len(items), PENALTY),
                    "imu": PENALTY, "diag": []}
        err = q - it["enc"]
        ej = [float(np.sqrt(np.mean(err[:, j] ** 2))) / max(it["travel"][j], 1e-9)
              for j in it["target"]]
        per.append(min(float(np.mean(ej)), PENALTY))
        b = bandpass41(g @ R.T)
        imu_per.append(min(float(np.square(it["abp"] - b).sum())
                           / max(it["ten"], 1e-12), PENALTY))
        if want_diag:
            dg.append({"name": it["name"], "target": it["target"],
                       "eq": (q_eq - it["enc0"]).tolist(),
                       "still": [float(err[it["still"][:, j], j].mean())
                                 if it["still"][:, j].any() else np.nan
                                 for j in it["target"]],
                       "const": [float(err[it["const"][:, j], j].mean())
                                 if it["const"][:, j].any() else np.nan
                                 for j in it["target"]],
                       "per_joint": ej})
    per = np.array(per)
    return {"loss": float(per.mean()), "per": per,
            "imu": float(np.mean(imu_per)), "diag": dg}


def cmd_predictor(items) -> float:
    """「指令即输出」——完全不用模型的平凡预测器，给 L 一个可读的零点。"""
    per = []
    for it in items:
        err = it["cmd"] - it["enc"]
        per.append(float(np.mean([np.sqrt(np.mean(err[:, j] ** 2))
                                  / max(it["travel"][j], 1e-9) for j in it["target"]])))
    return float(np.mean(per))


# ── Optuna ──────────────────────────────────────────────────────────────
def free_names() -> list[str]:
    """23 个自由量的名字（`params.NAMES` 去掉 gauge）。"""
    import params as P
    return [n for n in P.NAMES if n not in P.FIXED]


def suggest(trial) -> np.ndarray:
    """→ 24 维 θ；`kp6` 不参与采样，直接填 gauge 值。"""
    import params as P
    th = np.empty(P.N_P)
    for i, nm in enumerate(P.NAMES):
        if nm in P.FIXED:
            th[i] = P.KP6_GAUGE
        else:
            th[i] = trial.suggest_float(nm, float(LO[i]), float(HI[i]), log=True)
    return th


def flat(theta: np.ndarray) -> dict:
    import params as P
    return {nm: float(np.clip(theta[i], LO[i], HI[i]))
            for i, nm in enumerate(P.NAMES) if nm not in P.FIXED}


def theta_of(trial) -> np.ndarray:
    import params as P
    return np.array([P.KP6_GAUGE if nm in P.FIXED else trial.params[nm]
                     for nm in P.NAMES])


def storage(seed: int):
    import optuna
    OUT.mkdir(parents=True, exist_ok=True)
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    return JournalStorage(JournalFileBackend(str(OUT / f"journal_s{seed}.log")))


def study_name(seed: int) -> str:
    return f"g_ident_enc_s{seed}"


def make_study(seed: int, create: bool, sampler_seed: int | None = None):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    s = optuna.create_study(
        study_name=study_name(seed), storage=storage(seed), direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=seed if sampler_seed is None else sampler_seed,
            multivariate=True, group=True, n_startup_trials=40))
    if create and not s.trials:
        s.enqueue_trial(flat(BASE_XML))
        s.enqueue_trial(flat(BASE_PHYS_V1))
    return s


def _worker(seed: int, rank: int, n_trials: int) -> None:
    import mujoco
    from sim.step1_v2_loss import load_calib
    from sim.step2_fidelity import CALIB
    from sim.step2_gap_baseline import MODEL
    R = load_calib(CALIB)[0]
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    items = prep(split()[0], m.opt.timestep)
    st = make_study(seed, create=False, sampler_seed=seed + 1009 * rank)
    t0 = time.time()

    def obj(trial):
        r = evaluate(m, items, suggest(trial), R)
        trial.set_user_attr("imu", r["imu"])
        return r["loss"]

    def cb(study, trial):
        if rank == 0:
            b = study.best_trial
            print(f"  seed {seed}  [{len(study.trials):4d}] 本 {trial.value:.4f}"
                  f"   最优 {b.value:.4f} (IMU {b.user_attrs.get('imu', float('nan')):.4f})"
                  f"   {time.time()-t0:.0f}s", flush=True)

    st.optimize(obj, n_trials=n_trials, callbacks=[cb], gc_after_trial=True)
    # **解释器 teardown 段错误规避**：本环境在 mujoco+scipy 共存时退出阶段会
    # 抛 `deletion of interned string failed: KeyError: 'lmbdas'` 并 SIGSEGV。
    # trial 已逐条落进 journal，直接硬退出，避免把干净结束误判成崩溃。
    sys.stdout.flush(); sys.stderr.flush()
    import os; os._exit(0)


# ── 命令 ────────────────────────────────────────────────────────────────
def cmd_selftest(_a) -> int:
    import mujoco
    from sim.step2_fidelity import A_MAX, apply_limiter
    from sim.step2_g.accel_limit import accel_limited
    import params as P
    print("**自检**\n")
    d = json.loads((COL / "replay_S_j1_p0_r5_f1.00.json").read_text())
    cmd = np.degrees(np.asarray(d["cmd"], float)[:, :6])
    ok = True
    for j in range(6):
        a = accel_limited(cmd[:, j], float(A_MAX[j]))
        b = accel_limited_dt(cmd[:, j], float(A_MAX[j]), np.full(len(cmd), 1.0 / FPS))
        e = float(np.abs(a - b).max())
        ok &= e == 0.0
        print(f"  限幅器 j{j+1}  dts≡50ms 时 max|冻结版 − 变步长版| = {e:.3e}"
              f"  {'✓' if e == 0.0 else '**✗**'}")
    m = mujoco.MjModel.from_xml_path(str(ROOT / "sim/piper_model.xml"))
    P.apply(m, BASE_XML)
    print(f"\n  params.apply 后 kp = {m.actuator_gainprm[:6,0]}")
    print(f"    kp6 = {m.actuator_gainprm[5,0]:.1f}  (gauge {P.KP6_GAUGE:.1f})"
          f"  {'✓' if m.actuator_gainprm[5,0] == P.KP6_GAUGE else '**✗**'}")
    print(f"    biasprm[:,2] = {m.actuator_biasprm[:6,2]}  (kv 必须全 0)"
          f"  {'✓' if not m.actuator_biasprm[:6,2].any() else '**✗**'}")
    th = BASE_XML.copy(); th[5] = 999.0
    P.apply(m, th)
    print(f"    喂 kp6=999 后实际生效 {m.actuator_gainprm[5,0]:.1f}"
          f"  {'✓ gauge 生效' if m.actuator_gainprm[5,0] == P.KP6_GAUGE else '**✗**'}")
    print(f"\n  自由量 {len(free_names())} 个 (params.N_FREE = {P.N_FREE})"
          f"  {'✓' if len(free_names()) == P.N_FREE else '**✗**'}")
    it = prep(["S_j1_p0_r5_f1.00"], m.opt.timestep)[0]
    print(f"\n  事件网格 {it['name']}: 帧 {it['n']}  末 send_step {it['send_step'][-1]}"
          f"  末 read_step {it['read_step'][-1]}"
          f"  名义步数(n−1 帧) {int((it['n']-1)*(1/FPS)/m.opt.timestep)}"
          f"  ⇒ 实际比名义 {it['send_step'][-1]-int((it['n']-1)*(1/FPS)/m.opt.timestep):+d} 步")
    print(f"    read−send (步) 中位 {np.median(it['read_step']-it['send_step']):.1f}"
          f"  (1 步 = {m.opt.timestep*1000:.0f} ms)")
    return 0 if ok else 1


def cmd_baseline(_a) -> int:
    import mujoco
    from sim.step1_v2_loss import load_calib
    from sim.step2_fidelity import CALIB
    from sim.step2_gap_baseline import MODEL
    R = load_calib(CALIB)[0]
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    tr, ha, hb = split()
    sets = [("训练 p0+p1", prep(tr, m.opt.timestep)),
            ("留出A p2", prep(ha, m.opt.timestep)),
            ("留出B 多正弦", prep(hb, m.opt.timestep))]
    print("**编码器目标 基线**   L = 逐条等权(激励关节 RMS / 该条行程)\n")
    print(f"  {'配置':22s}{'数据集':14s}{'L':>9s}{'IMU 等权':>10s}")
    for sn, items in sets:
        print(f"  {'指令即输出(无模型)':22s}{sn:14s}{cmd_predictor(items):>9.4f}{'—':>10s}")
    for nm, th in (("现 XML 模型", BASE_XML), ("G-π-phys-v1", BASE_PHYS_V1),
                   ("IMU轮最优(kp6被压回)", IMU_BEST_2890)):
        for sn, items in sets:
            r = evaluate(m, items, th, R)
            print(f"  {nm:22s}{sn:14s}{r['loss']:>9.4f}{r['imu']:>10.4f}", flush=True)
    return 0


def cmd_run(a) -> int:
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    tr, ha, hb = split()
    print(f"**命令 → 编码器关节轨迹层的有效参数辨识**")
    print(f"  自由量 {len(free_names())} 个（kp6 = 50.0 固定为 gauge）")
    print(f"  训练 {len(tr)} · 留出A {len(ha)} · 留出B {len(hb)}   ← 留出集本阶段**不评估**")
    print(f"  预算**固定**: {len(SEEDS)} seed × {a.trials} trial，结束后不追加\n")
    for seed in SEEDS:
        _run_seed(seed, a.trials, a.workers, ctx)
    return cmd_report(a)


def _n_done(seed: int) -> int:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        st = optuna.load_study(study_name=study_name(seed), storage=storage(seed))
    except Exception:                                    # noqa: BLE001
        return 0
    return sum(1 for t in st.trials if t.value is not None)


def _run_seed(seed: int, target: int, workers: int, ctx) -> None:
    """跑到 `target` 个完成 trial 为止，**带崩溃重启**。

    本环境有已知的偶发 SIGSEGV（见 `infra_train_harness_background_segfault`）。
    worker 静默死掉会丢掉它剩下的 trial，所以按「已完成数」补齐而不是按进程数。
    重试**只为达到预注册预算，不超过它**；若一轮零进展则停止（说明不是偶发崩溃）。
    """
    make_study(seed, create=True)
    for att in range(1, 21):
        n = _n_done(seed)
        if n >= target:
            break
        per = int(np.ceil((target - n) / workers))
        ps = [ctx.Process(target=_worker, args=(seed, i, per)) for i in range(workers)]
        for p in ps:
            p.start()
        for p in ps:
            p.join()
        bad = [p.exitcode for p in ps if p.exitcode != 0]
        n2 = _n_done(seed)
        print(f"  ── seed {seed} 第 {att} 轮: {n} → {n2}/{target}"
              f"   非零退出 {len(bad)}/{workers} {sorted(set(bad))[:4]} ──", flush=True)
        if n2 <= n:
            print(f"  **seed {seed} 一轮零进展，停止重试**", flush=True)
            break
    print(f"  ── seed {seed} 收尾: {_n_done(seed)} trial ──\n", flush=True)


def _spans(trials, names, tol=0.10) -> tuple[dict, int]:
    """近优 tol 集内，每个自由参数的倍差。→ ({名: 倍差}, 集大小)"""
    best = min(t.value for t in trials)
    near = [t for t in trials if t.value <= best * (1 + tol)]
    out = {}
    for nm in names:
        v = np.array([t.params[nm] for t in near if nm in t.params])
        out[nm] = float(v.max() / max(v.min(), 1e-15)) if v.size else float("nan")
    return out, len(near)


def _imu_round_spans(tol=0.10) -> tuple[float, float]:
    """IMU 轮（`optuna_ident`）的同一统计量，用于对照。→ (23 项中位, 含 kp6 的中位)"""
    import optuna
    from optuna_ident import STUDY, storage as st_imu
    try:
        s = optuna.load_study(study_name=STUDY, storage=st_imu())
    except Exception:                                    # noqa: BLE001
        return float("nan"), float("nan")
    tri = [t for t in s.trials if t.value is not None]
    if not tri:
        return float("nan"), float("nan")
    # IMU 轮的命名是 kp1..kp6 / damp / arm / fric ⇒ 映射到本轮的块名
    m = {"kp": "kp", "damp": "c", "arm": "Ia", "fric": "fc"}
    best = min(t.value for t in tri)
    near = [t for t in tri if t.value <= best * (1 + tol)]
    allsp, sp23 = [], []
    for c, blk in m.items():
        for j in range(6):
            v = np.array([t.params[f"{c}{j+1}"] for t in near])
            f = float(v.max() / max(v.min(), 1e-15))
            allsp.append(f)
            if not (c == "kp" and j == 5):
                sp23.append(f)
    return float(np.median(sp23)), float(np.median(allsp))


def cmd_report(_a) -> int:
    import mujoco, optuna
    import params as P
    from sim.step1_v2_loss import load_calib
    from sim.step2_fidelity import CALIB
    from sim.step2_gap_baseline import MODEL
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    R = load_calib(CALIB)[0]
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    tr, ha, hb = split()
    S = {sn: prep(nm, m.opt.timestep) for sn, nm in
         (("训练", tr), ("留出A", ha), ("留出B", hb))}
    names = free_names()

    studies, bests = {}, {}
    for seed in SEEDS:
        try:
            st = optuna.load_study(study_name=study_name(seed), storage=storage(seed))
        except Exception:                                # noqa: BLE001
            continue
        tri = [t for t in st.trials if t.value is not None]
        if tri:
            studies[seed] = tri
            bests[seed] = min(tri, key=lambda t: t.value)
    if not studies:
        print("没有可读的 study。先跑 --run。")
        return 1

    print(f"\n{'='*84}\n**命令 → 编码器关节轨迹层的有效参数辨识 —— 结果**")
    print(f"  （**不得**解释为真实机械参数：编码器在电机/减速器侧，"
          f"看不到 IMU 所见的连杆与支架振动）\n")
    for seed, tri in studies.items():
        print(f"  seed {seed}: {len(tri)} trial   最优 {bests[seed].value:.4f}")

    print(f"\n  {'配置':24s}{'训练':>9s}{'留出A':>9s}{'留出B':>9s}{'训练IMU':>10s}")
    rows = {}
    for nm, th in [("指令即输出(无模型)", None), ("现 XML 模型", BASE_XML),
                   ("G-π-phys-v1", BASE_PHYS_V1),
                   ("IMU轮最优(kp6压回50)", IMU_BEST_2890)] + \
                  [(f"**seed {s} 最优**", theta_of(bests[s])) for s in bests]:
        if th is None:
            v = [cmd_predictor(S[k]) for k in ("训练", "留出A", "留出B")]
            print(f"  {nm:24s}{v[0]:>9.4f}{v[1]:>9.4f}{v[2]:>9.4f}{'—':>10s}")
            rows[nm] = v + [float("nan")]
            continue
        r = [evaluate(m, S[k], th, R) for k in ("训练", "留出A", "留出B")]
        v = [x["loss"] for x in r] + [r[0]["imu"]]
        rows[nm] = v
        print(f"  {nm:24s}{v[0]:>9.4f}{v[1]:>9.4f}{v[2]:>9.4f}{v[3]:>10.4f}", flush=True)

    # ── 收窄 ──
    print(f"\n{'─'*84}\n  **近优 10% 集内的参数倍差**（越小＝越被约束）")
    med_by_seed, sp_by_seed = {}, {}
    for seed, tri in studies.items():
        sp, k = _spans(tri, names)
        sp_by_seed[seed] = sp
        med_by_seed[seed] = float(np.median(list(sp.values())))
        blk = {b: float(np.median([sp[n] for n in names if n.startswith(b)]))
               for b in ("kp", "c", "fc", "Ia")}
        print(f"    seed {seed} (近优集 {k:3d} 个): 23 项中位 **{med_by_seed[seed]:.1f}×**"
              f"   kp {blk['kp']:.1f}×  c {blk['c']:.1f}×"
              f"  fc {blk['fc']:.1f}×  Ia {blk['Ia']:.1f}×")
    imu23, imu24 = _imu_round_spans()
    print(f"    ── 对照：IMU 轮 23 项中位 **{imu23:.1f}×**（含 kp6 的 24 项 {imu24:.1f}×）")
    narrowed = float(np.median(list(med_by_seed.values()))) <= 10.0

    # ── 跨 seed 稳定 ──
    print(f"\n  **跨 seed 稳定性**（3 个最优参数的逐参数极差）")
    stab = {}
    if len(bests) >= 2:
        for nm in names:
            v = np.array([bests[s].params[nm] for s in bests])
            stab[nm] = float(v.max() / max(v.min(), 1e-15))
        med_stab = float(np.median(list(stab.values())))
        worst = sorted(stab.items(), key=lambda x: -x[1])[:6]
        print(f"    逐参数极差 中位 **{med_stab:.1f}×**   最差 6 项: " +
              ", ".join(f"{k} {v:.0f}×" for k, v in worst))
        stable = med_stab <= 3.0
    else:
        med_stab, stable = float("nan"), False
        print("    只有 1 个 seed，无法评估")

    # ── 留出改善 / IMU 退化 ──
    bl = rows["G-π-phys-v1"]
    cp = rows["指令即输出(无模型)"]
    bs = min((rows[f"**seed {s} 最优**"] for s in bests), key=lambda v: v[0])
    improved = bs[1] < bl[1] and bs[1] < cp[1]
    imu_deg = bs[3] > bl[3] * 1.05

    print(f"\n{'='*84}\n**预注册判读**")
    print(f"  留出A 改善(优于 phys-v1 {bl[1]:.4f} 且优于指令 {cp[1]:.4f}): "
          f"{bs[1]:.4f}  {'✓' if improved else '**✗**'}")
    print(f"  参数明显收窄(23 项倍差中位 ≤ 10×): "
          f"{np.median(list(med_by_seed.values())):.1f}×  {'✓' if narrowed else '**✗**'}")
    print(f"  跨 seed 稳定(逐参数极差中位 ≤ 3×): {med_stab:.1f}×"
          f"  {'✓' if stable else '**✗**'}")
    print(f"  IMU 未退化(不比 phys-v1 差 5% 以上): {bs[3]:.4f} vs {bl[3]:.4f}"
          f"  {'**✗ 已退化**' if imu_deg else '✓'}")
    if improved and narrowed and stable:
        v = "**原 IMU 目标确实条件较差** —— 编码器目标下参数被约束住了"
    elif improved and not narrowed:
        v = ("**只得到有效预测模型，得不到唯一物理参数** —— 编码器预测改善但参数仍跨几十倍")
    elif not improved:
        v = "**编码器仍不能约束参数 ⇒ 停止继续堆 trial，转电流观测**"
    else:
        v = "**混合结果，逐条按上表读**"
    if imu_deg:
        v += "\n     并且 **IMU 退化 ⇒ 只改善了电机侧轨迹，不能升级为完整 G**"
    print(f"\n⇒ {v}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps({
        "rows": {k: [None if np.isnan(x) else float(x) for x in v] for k, v in rows.items()},
        "span_median_by_seed": med_by_seed, "span_imu_round_23": imu23,
        "cross_seed_span": stab, "cross_seed_median": med_stab,
        "improved": bool(improved), "narrowed": bool(narrowed),
        "stable": bool(stable), "imu_degraded": bool(imu_deg),
        "best_theta": {str(s): theta_of(bests[s]).tolist() for s in bests},
        "param_names": P.NAMES, "fixed": list(P.FIXED),
        "kp6_gauge": P.KP6_GAUGE, "n_free": len(names),
    }, ensure_ascii=False, indent=2))
    print(f"\n  → {OUT/'result.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.selftest:
        return cmd_selftest(a)
    if a.baseline:
        return cmd_baseline(a)
    if a.run:
        return cmd_run(a)
    if a.report:
        return cmd_report(a)
    print(__doc__.split("用法:")[0])
    return 0


def _exit(code: int):
    """绕开 teardown 段错误（见 `_worker` 注释）；输出已 flush。"""
    sys.stdout.flush(); sys.stderr.flush()
    import os
    os._exit(code)


if __name__ == "__main__":
    _exit(main())
