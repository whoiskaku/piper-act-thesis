#!/usr/bin/env python3
"""**第二问：结构扩展的公平比较** —— 固定 G-q-v1，每次只加一个自由度。

底座 = **G-q-v1 的 θ* 逐字固定**（`kp`/`c`/`fc`/`Ia` 全部），限幅器不变，
**只增加 1 个自由度、只拟合新增参数**。拟合集 = 训练 62 条（位形 p0+p1）。

## 候选 F：传动 / 连杆柔性自由度

执行器留电机侧（编码器读电机侧），连杆侧加被动铰。**去掉旧 G-flex 的
`MOTOR_HZ=40` 硬钉**，改用 G-q-v1 冻结的 `armature`。

> ⚠ 该 `armature` **只能称「有效参数」**。它在 IMU 与编码器两个目标下都
> **未被辨识**（近优集内跨 3–4 个数量级、跨 seed 极差 217×、整块置零只 +2.1%）。
> 这里用它**仅作数值底座**，**不声称已物理辨识电机惯量**。

**离散因子 `j ∈ {j2, j4, j5}`**，由**可解析性判据** `dt·ωn ≤ 0.25` 定出
（`ωn = √(kp_j/Ia_j)` 是柔性模型里电机侧的固有频率）：

```
j1 669 Hz (dt·ωn 8.41) ✗   j2  9.5 Hz (0.119) ✓   j3 248 Hz (3.12) ✗
j4 7.3 Hz (0.091)      ✓   j5 17.8 Hz (0.223) ✓   j6  66 Hz (0.831) ✗
```

**被排除的三个关节是 `Ia` 不可辨识的数值后果，不是物理判断** ——
它们的有效 `Ia` 落在搜索下界附近，要解析得把 `dt` 缩小约 34 倍，不划算。
（物理上 j2 = 臂下垂模载体、j5 = 腕部载体，本就是首选。）

## 候选 B：IMU 支架的**传感器侧有效二阶模态**

现状：`ee_imu_site` 是直接挂在 `link6` 上的**无质量 site**（`pos="0 0 0.13"`），
支架完全未建模。但 **0.05 kg 不足以定义一个转动刚体**（还需 COM、惯量、铰点，
而我们没有支架尺寸）。

⇒ 故**不建物理支架**，改建**传感器侧有效二阶模态**：基座激励下，
测得角速度 = 刚体角速度经传递率滤波

```
H(s) = (2ζω·s + ω²) / (s² + 2ζω·s + ω²)        沿选定轴，其余两轴不变
```

DC 增益 1（刚体极限）、`ω` 附近共振放大、高频衰减（传感器与基座解耦）。
**只有 (ω, ζ) 两个参数 —— 不需要质量 / COM / 惯量 / 铰点。**

> ⚠ **不声称辨识了真实支架的物理参数。** 这是一个**有效**二阶模型，
> 只在「传感器侧存在一个二阶动力学」这一层面上有意义。

**离散因子 = 轴 `∈ {x, y, z}`**（IMU site 系）。

滤波在 **500 Hz** 内部速率上做（`ω ≤ 12 Hz` ⇒ ≥41 步/周期），再抗混叠降到 20 Hz。

## 公平性

两候选：各 **1** 个新增自由度、**2** 个连续参数、**3** 个离散选项、同一底座、
同一拟合集、同一网格（`ω` 7 值 log ∈ [1.5, 12] Hz × `ζ` 4 值
{0.05, 0.10, 0.20, 0.40} × 3 = **84 格**）、同一指标。
**预算固定 168 格，不追加。**

## 门槛（预注册，跑前定死）

**选择**：只看**训练集**，每候选取训练 IMU 等权最小、且训练关节 L 不比 G-q-v1
差超过 5% 的格。

**A/B 用来选结构；现有 pp2 用作开发门禁**（冻结后再用**新采** pp2 做最终确认）。

| # | 条件 | 阈值 |
|---|---|---|
| 1 | 留出 A、留出 B 关节误差不比 G-q-v1 差超过 | 5% |
| 2 | 留出 A、留出 B 的 IMU 均至少改善 | 5% |
| 3 | 留出 A 按「目标关节 × 位形」分组，多数分组改善 | >50% |
| 4 | **pp2 IMU 至少改善** | **5%** |
| 5 | **pp2 IMU 逐集改善数** | **≥9/17** |
| 6 | **pp2 IMU 逐集中位改善为正** | >0 |
| 7 | **pp2 关节轨迹不恶化超过** | **5%** |
| 8 | 两候选间：胜者在留出 A 的 IMU 上需比另一候选好至少 5%，否则记「两者都不足以判定」 | 5% |
| 9 | **步长稳定检查**：最优格在 `dt` 减半后指标变化 | <2% |

**判定**：某候选 1–7 + 9 全过 ⇒ 冻结为候选；
**若两候选在 pp2 上都不改善 ⇒ 立即收口**：它们不能解决当前任务的 IMU 缺口，
**不再扩大柔性结构**。

用法:
    python sim/step2_g/structure_compare.py --smoke     # 1 格 × 5 条, 自检
    python sim/step2_g/structure_compare.py --run
"""
from __future__ import annotations
import argparse, gc, json, sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g"):
    sys.path.insert(0, str(_p))

FPS = 20.0
OUT = ROOT / "outputs/step2_g/structure"
FLEX_J = (1, 3, 4)                    # 0-based: j2, j4, j5
AXES = (0, 1, 2)
W_GRID = np.exp(np.linspace(np.log(1.5), np.log(12.0), 7))
Z_GRID = (0.05, 0.10, 0.20, 0.40)
FLEX_RANGE = 0.35                     # rad, 防数值爆炸
REF_POSE = np.radians([-4.9, 51.5, -25.2, 0.0, 9.4, 4.0])   # HOME, 定义 M_eff


# ── θ* 按**名字**写入（柔性模型里 dof 下标会因新自由度而位移）──
def apply_theta_named(m, theta) -> None:
    import mujoco
    import params as P
    th = P.unpack(np.asarray(theta, float))
    for k in range(6):
        jn = f"joint{k+1}"
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        da = int(m.jnt_dofadr[jid])
        m.dof_damping[da] = float(th["c"][k])
        m.dof_armature[da] = float(th["Ia"][k])
        m.dof_frictionloss[da] = float(th["fc"][k])
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, jn)
        kp = P.KP6_GAUGE if k == 5 else float(th["kp"][k])
        m.actuator_gainprm[aid, 0] = kp
        m.actuator_biasprm[aid, 1] = -kp
        m.actuator_biasprm[aid, 2] = 0.0          # kv 已并入 c


def build_flex(j: int, K: float, C: float, path: Path, theta) -> Path:
    """在关节 j（0-based）处插柔性自由度。结构照搬 `gflex.build_flex_xml`，
    但**不再硬钉 MOTOR_HZ**，电机侧 armature 由 `apply_theta_named` 在运行时写入。"""
    import params as P
    from sim.step2_gap_baseline import MODEL
    nm = f"link{j+1}"
    tree = ET.parse(MODEL)
    root = tree.getroot()
    parent = next(b for b in root.iter("body")
                  if any(c.get("name") == nm for c in b.findall("body")))
    link = next(c for c in parent.findall("body") if c.get("name") == nm)
    jt = next(x for x in link.findall("joint") if x.get("name") == f"joint{j+1}")

    rotor = ET.Element("body", {"name": f"{nm}_rotor",
                               "pos": link.get("pos", "0 0 0"),
                               "quat": link.get("quat", "1 0 0 0")})
    ET.SubElement(rotor, "inertial", {"pos": "0 0 0", "mass": "0.001",
                                      "diaginertia": "1e-9 1e-9 1e-9"})
    link.remove(jt)
    rotor.append(jt)                                     # joint{j} 移到电机侧
    link.set("pos", "0 0 0")
    link.attrib.pop("quat", None)
    flex = ET.Element("joint", {
        "name": f"joint{j+1}_flex", "pos": "0 0 0", "axis": jt.get("axis", "0 0 1"),
        "limited": "true", "range": f"{-FLEX_RANGE} {FLEX_RANGE}",
        "stiffness": f"{K:.8g}", "damping": f"{C:.8g}", "springref": "0"})
    link.insert(1, flex)
    idx = list(parent).index(link)
    parent.remove(link)
    rotor.append(link)
    parent.insert(idx, rotor)

    comp = root.find("compiler")
    md = comp.get("meshdir", "")
    comp.set("meshdir", str((Path(MODEL).parent / md).resolve()))
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def m_eff(path: Path, j: int, theta) -> float:
    """参考位形下 `joint{j}_flex` 的质量矩阵对角元（模型给的，不是辨识量）。"""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(path))
    apply_theta_named(m, theta)
    d = mujoco.MjData(m)
    for k in range(6):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{k+1}")
        d.qpos[int(m.jnt_qposadr[jid])] = REF_POSE[k]
    mujoco.mj_forward(m, d)
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, M, d.qM)
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{j+1}_flex")
    a = int(m.jnt_dofadr[fid])
    return float(M[a, a])


def flex_xml(j: int, w_hz: float, zeta: float, theta) -> Path:
    """两遍：先用占位刚度量 M_eff，再按 (ω, ζ) 换算 K/C 重建。"""
    tmp = OUT / f"_probe_j{j+1}.xml"
    build_flex(j, 100.0, 1.0, tmp, theta)
    Me = m_eff(tmp, j, theta)
    w = 2 * np.pi * w_hz
    p = OUT / f"flex_j{j+1}_w{w_hz:.3f}_z{zeta:g}.xml"
    build_flex(j, Me * w * w, 2.0 * zeta * w * Me, p, theta)
    return p


# ── 前向：柔性模型（事件精确回放，与 g_q_v1 同惯例）──
def roll_flex(xml: Path, item, theta, mj_dt_scale: float = 1.0):
    """→ (关节角 度[电机侧], 20 Hz 陀螺)。读数惯例与 `g_q_v1._sim` 一致。"""
    import mujoco
    from scipy.signal import butter, filtfilt
    m = mujoco.MjModel.from_xml_path(str(xml))
    if mj_dt_scale != 1.0:
        m.opt.timestep = m.opt.timestep * mj_dt_scale
    apply_theta_named(m, theta)
    dt = m.opt.timestep
    qa = []
    for k in range(6):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{k+1}")
        qa.append(int(m.jnt_qposadr[jid]))
    d = mujoco.MjData(m)
    # 与冻结的 G-q-v1 使用完全相同的可用输入初始化：只读首帧命令，不能把
    # 实测 enc0 偷渡给候选 F（否则候选比基线多看了一项真机状态）。
    for k in range(6):
        d.qpos[qa[k]] = float(item["act"][0, k])
    g0 = (float(np.clip(item["act"][0, 6], 0, 0.035))
          if item["act"].shape[1] > 6 else 0.0)
    for jn, val in (("joint7", g0), ("joint8", -g0)):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid >= 0:
            d.qpos[int(m.jnt_qposadr[jid])] = val
    mujoco.mj_forward(m, d)
    d.ctrl[:6] = item["ref"][0]
    for an, val in (("joint7", g0), ("joint8", -g0)):
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, an)
        if aid >= 0:
            d.ctrl[aid] = val
    for _ in range(int(round(0.3 / dt))):
        mujoco.mj_step(m, d)

    send = np.rint(np.asarray(item["send_s"], float) / dt).astype(int)
    send = np.maximum.accumulate(np.maximum(send, 0))
    read = send + 1
    n = len(send)
    ntot = int(read[-1]) + 1
    hi = np.empty((ntot, 3))
    q = np.empty((n, 6))
    si = ri = 0
    for s in range(ntot):
        while si < n and send[si] <= s:
            d.ctrl[:6] = item["ref"][si]; si += 1
        while ri < n and read[ri] <= s:
            q[ri] = np.degrees([d.qpos[a] for a in qa]); ri += 1
        hi[s] = d.sensordata[3:6]
        mujoco.mj_step(m, d)
    while ri < n:
        q[ri] = np.degrees([d.qpos[a] for a in qa]); ri += 1
    ba = butter(4, 0.45 * FPS / ((1.0 / dt) / 2.0))
    return q, filtfilt(*ba, hi, axis=0)[np.clip(read, 0, ntot - 1)]


# ── 前向：传感器侧有效二阶模态（后置滤波，刚体解只算一次）──
def bracket_gyro(hi, fs, read, axis: int, w_hz: float, zeta: float):
    """基座激励传递率 H(s) = (2ζω s + ω²)/(s² + 2ζω s + ω²)，只作用于选定轴。"""
    from scipy.signal import bilinear, lfilter, butter, filtfilt
    w = 2 * np.pi * w_hz
    b, a = bilinear([2 * zeta * w, w * w], [1.0, 2 * zeta * w, w * w], fs=fs)
    out = np.array(hi, float, copy=True)
    out[:, axis] = lfilter(b, a, hi[:, axis])
    ba = butter(4, 0.45 * FPS / (fs / 2.0))
    return filtfilt(*ba, out, axis=0)[read]


# ── 数据 ──
def prep(names, pp2: bool):
    from sim.step2_fidelity import A_MAX
    from optuna_ident_enc import accel_limited_dt
    from sim.step2_gap_baseline import bare_logs
    man = json.loads((ROOT / "outputs/step2_g/manifest.json").read_text())
    COL = ROOT / "outputs/step2_g/collect"
    out = []
    src = bare_logs() if pp2 else names
    for it in src:
        if pp2:
            fr = json.loads(Path(it).read_text())["frames"]
            nm = Path(it).stem
            act = np.asarray([f["a"] for f in fr], float)
            enc = np.asarray([f["state"] for f in fr], float)[:, :6] / 1000.0
            real = np.asarray([f["imu"] for f in fr], float)[:, 3:6]
            tv, tg, pose = None, None, None
        else:
            nm = it
            d = json.loads((COL / f"replay_{nm}.json").read_text())
            act = np.asarray(d["cmd"], float)
            enc = np.asarray(d["state"], float)[:, :6] / 1000.0
            real = np.asarray(d["imu"], float)[:, 3:6]
            cm = np.degrees(act[:, :6])
            tv, tg = cm.max(0) - cm.min(0), man[nm]["target"]
            pose = nm.split("_p")[1][0] if "_p" in nm else "x"
        n = min(len(act), len(enc), len(real))
        act, enc, real = act[:n], enc[:n], real[:n]
        cmd = np.degrees(act[:, :6])
        ref = np.empty((n, 6))
        for j in range(6):
            ref[:, j] = accel_limited_dt(cmd[:, j], float(A_MAX[j]), np.full(n, 1 / FPS))
        out.append({"name": nm, "act": act, "enc": enc, "real": real,
                    "enc0": enc[0], "ref": np.radians(ref),
                    "send_s": np.arange(n) / FPS, "travel": tv,
                    "target": tg, "pose": pose, "pp2": pp2})
    return out


def score(qs, gs, data, R):
    """→ (逐条关节, 逐条 IMU 等权)。位移固定 关节 L=0 / IMU L=1。"""
    from sim.step1_v2_loss import bandpass41
    pj, pi = [], []
    for it, q, g in zip(data, qs, gs):
        k = min(len(q), len(it["enc"]))
        if it["pp2"]:
            pj.append(float(np.sqrt(np.mean((q[:k] - it["enc"][:k]) ** 2))))
        else:
            pj.append(float(np.mean([np.sqrt(np.mean((q[:k, j] - it["enc"][:k, j]) ** 2))
                                     / max(it["travel"][j], 1e-9) for j in it["target"]])))
        k2 = min(len(g), len(it["real"]) - 1)
        a_ = bandpass41(it["real"][1:1 + k2])
        b_ = bandpass41(g[:k2] @ R.T)
        pi.append(float(np.square(a_ - b_).sum()) / max(float(np.square(a_).sum()), 1e-12))
    return np.array(pj), np.array(pi)


# ── 候选 B 的刚体解缓存（一次算完，84 格几乎免费）──
def rigid_hi(data, theta):
    import g_q_v1 as GQ
    out = []
    for it in data:
        q, g20, hi, read, fs = GQ._sim(it["act"], theta=theta, return_hi=True)
        out.append((q, g20, hi, read, fs))
    return out


def eval_bracket(cache, data, R, axis, w, z):
    qs = [c[0] for c in cache]
    gs = [bracket_gyro(c[2], c[4], c[3], axis, w, z) for c in cache]
    return score(qs, gs, data, R)


# ── 柔性网格：进程池，一格一任务 ──
_G = {}


def _init(split_name):
    import gc as _gc, mujoco
    _gc.disable()
    from sim.step1_v2_loss import load_calib
    from sim.step2_fidelity import CALIB
    from optuna_ident_enc import split
    from g_q_v1 import THETA_STAR
    tr, ha, hb = split()
    _G["R"] = load_calib(CALIB)[0]
    _G["theta"] = THETA_STAR
    _G["data"] = prep({"train": tr, "hA": ha, "hB": hb}[split_name], False)


def _cell_flex(a):
    j, w, z = a
    try:
        xml = flex_xml(j, w, z, _G["theta"])
        qs, gs = [], []
        for it in _G["data"]:
            q, g = roll_flex(xml, it, _G["theta"])
            if not (np.isfinite(q).all() and np.isfinite(g).all()):
                return (j, w, z, 9e9, 9e9)
            qs.append(q); gs.append(g)
        pj, pi = score(qs, gs, _G["data"], _G["R"])
        return (j, w, z, float(pj.mean()), float(pi.mean()))
    except Exception:                                    # noqa: BLE001
        return (j, w, z, 9e9, 9e9)


def _grid_flex(split_name, cells, workers=14):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init, initargs=(split_name,)) as pool:
        return pool.map(_cell_flex, cells)


def gate_eval(kind, par, theta, R):
    """在留出A/留出B/pp2 上评估选中的格 + G-q-v1 基线。→ dict"""
    import g_q_v1 as GQ
    from optuna_ident_enc import split
    tr, ha, hb = split()
    out = {}
    for sn, names, is_pp2 in (("留出A", ha, False), ("留出B", hb, False),
                              ("pp2", None, True)):
        data = prep(names, is_pp2)
        b_qs = [GQ.predict_qpos(it["act"]) for it in data]
        b_gs = [GQ.predict_gyro(it["act"]) for it in data]
        bj, bi = score(b_qs, b_gs, data, R)
        if kind == "F":
            j, w, z = par
            xml = flex_xml(j, w, z, theta)
            qs, gs = zip(*[roll_flex(xml, it, theta) for it in data])
            cj, ci = score(list(qs), list(gs), data, R)
        else:
            ax, w, z = par
            cache = rigid_hi(data, theta)
            cj, ci = eval_bracket(cache, data, R, ax, w, z)
        out[sn] = {"bj": bj, "bi": bi, "cj": cj, "ci": ci,
                   "groups": [(it["target"][0] if it["target"] else -1, it["pose"])
                              for it in data] if not is_pp2 else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--extend", action="store_true")
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    gc.disable()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.extend:
        return cmd_extend(a)
    if a.gate:
        return cmd_gate(a)
    if not (a.smoke or a.run):
        print(__doc__.split("用法:")[0]); return 0

    from sim.step1_v2_loss import load_calib
    from sim.step2_fidelity import CALIB
    from optuna_ident_enc import split
    from g_q_v1 import THETA_STAR as TH
    R = load_calib(CALIB)[0]
    tr, ha, hb = split()

    if a.smoke:
        data = prep(tr[:5], False)
        print("自检: 5 条训练序列, 各候选 1 格")
        import g_q_v1 as GQ
        bq = [GQ.predict_qpos(it["act"]) for it in data]
        bg = [GQ.predict_gyro(it["act"]) for it in data]
        bj, bi = score(bq, bg, data, R)
        print(f"  G-q-v1 基线      关节 {bj.mean():.4f}  IMU {bi.mean():.4f}")
        xml = flex_xml(1, 3.0, 0.10, TH)
        qs, gs = zip(*[roll_flex(xml, it, TH) for it in data])
        fj, fi = score(list(qs), list(gs), data, R)
        print(f"  F j2 ω3 ζ0.10   关节 {fj.mean():.4f}  IMU {fi.mean():.4f}")
        cache = rigid_hi(data, TH)
        cj, ci = eval_bracket(cache, data, R, 1, 3.0, 0.10)
        print(f"  B 轴y ω3 ζ0.10  关节 {cj.mean():.4f}  IMU {ci.mean():.4f}"
              f"   (关节应与基线**逐位相同**: max|Δ| "
              f"{max(np.abs(c[0]-b).max() for c,b in zip(cache,bq)):.2e})")
        return 0

    print("**第二问：结构扩展的公平比较**")
    print(f"  底座 G-q-v1 θ* 固定 · 拟合集 训练 {len(tr)} 条")
    print(f"  网格 ω 7 值 [1.5,12] Hz × ζ 4 值 × 3 离散 = 84 格/候选，共 168 格（固定）\n")
    cells_f = [(j, float(w), z) for j in FLEX_J for w in W_GRID for z in Z_GRID]
    print(f"  候选 F: {len(cells_f)} 格，关节 {[f'j{j+1}' for j in FLEX_J]}"
          f"（判据 dt·ωn≤0.25；j1/j3/j6 因 Ia 落在下界不可解析而排除）", flush=True)
    rf = _grid_flex("train", cells_f, a.workers)

    data_tr = prep(tr, False)
    cache_tr = rigid_hi(data_tr, TH)
    rb = []
    for ax in AXES:
        for w in W_GRID:
            for z in Z_GRID:
                pj, pi = eval_bracket(cache_tr, data_tr, R, ax, float(w), z)
                rb.append((ax, float(w), z, float(pj.mean()), float(pi.mean())))
    print(f"  候选 B: {len(rb)} 格（后置滤波，刚体解只算一次）", flush=True)

    import g_q_v1 as GQ
    bq = [GQ.predict_qpos(it["act"]) for it in data_tr]
    bg = [GQ.predict_gyro(it["act"]) for it in data_tr]
    b0j, b0i = score(bq, bg, data_tr, R)
    print(f"\n  训练集基线 G-q-v1: 关节 {b0j.mean():.4f}  IMU {b0i.mean():.4f}")

    sel = {}
    for kind, rows, lab in (("F", rf, "关节"), ("B", rb, "轴")):
        good = [r for r in rows if r[3] <= b0j.mean() * 1.05]
        print(f"\n  ── 候选 {kind}: {len(good)}/{len(rows)} 格满足训练关节不劣化 5% ──")
        top = sorted(good, key=lambda r: r[4])[:6] if good else []
        print(f"  {'离散':>6s}{'ω Hz':>8s}{'ζ':>7s}{'训练关节':>10s}{'训练IMU':>10s}"
              f"{'vs 基线 IMU':>12s}")
        for r in top:
            d0 = f"j{r[0]+1}" if kind == "F" else "xyz"[r[0]]
            print(f"  {d0:>6s}{r[1]:>8.2f}{r[2]:>7.2f}{r[3]:>10.4f}{r[4]:>10.4f}"
                  f"{(r[4]/b0i.mean()-1)*100:>11.1f}%")
        sel[kind] = top[0] if top else None
    (OUT / "grid.json").write_text(json.dumps(
        {"flex": rf, "bracket": rb, "base_train": [float(b0j.mean()), float(b0i.mean())]},
        ensure_ascii=False, indent=2))
    print(f"\n  → {OUT/'grid.json'}")
    return 0




# ══ 一次性网格外延（研究者 2026-08-30 放行；**固定预算，此后不再追加**）══
#
# 依据（`outputs/step2_g/structure/grid.json` 实测，非事后解释）:
#   F 沿 ω 固定 ζ=0.40: 6.00:0.4824 → **8.49:0.4726** → 12.00:0.4770  ⇒ **ω 是内点**
#   F 沿 ζ 固定 ω=8.49: 0.05:0.6434 → 0.10:0.5234 → 0.20:0.4772 → **0.40:0.4726**
#                                                        ⇒ **ζ 在边缘, 单调要更大**
#   B 沿 ζ 固定 ω=8.49: 0.4778 / 0.4775 / 0.4782 / 0.4799  ⇒ **两维都是内点**
# ⇒ **只外延 F, 只沿 ζ 向上**（ζ 与 ω 交互 ⇒ 带一个 ω 局部细化）。**B 不外延**,
#   理由是它的最优在两维都是内点 —— 证据判断, 非偏好。
#
# **预注册的解释约束**: 若胜出格的 ζ ≥ 0.7, **不得称「柔性共振模态」** ——
# 大 ζ 下柔性铰趋于纯阻尼器, 加进去的是**电机–连杆之间的有效滞后 / 低通**,
# 那是**不同的物理主张**。
#
# **终止规则**: 这是唯一一次外延。若 F 仍不过条件 4 ⇒ **收口, 不再扩大柔性结构**。
Z_EXT = (0.6, 0.8, 1.0, 1.4, 2.0)
W_EXT = (6.00, 7.13, 8.49, 10.10, 12.00)
W_REFINE = (7.13, 10.10)


def ext_cells():
    c = [(j, float(w), z) for j in FLEX_J for w in W_EXT for z in Z_EXT]      # 75
    c += [(4, float(w), z) for w in W_REFINE for z in Z_GRID]                 # 8 (仅 j5)
    return c


def cmd_extend(a) -> int:
    cells = ext_cells()
    print(f"**一次性网格外延（仅候选 F）** {len(cells)} 格，固定预算，此后不追加")
    print(f"  ζ 向上 {Z_EXT} · ω {W_EXT} · 关节 {[f'j{j+1}' for j in FLEX_J]}")
    print(f"  + j5 的 ω 细化 {W_REFINE} × 原 ζ {Z_GRID}")
    print(f"  **B 不外延** —— 其最优在 ω 与 ζ 两维都是内点\n", flush=True)
    rows = _grid_flex("train", cells, a.workers)
    G = json.loads((OUT / "grid.json").read_text())
    G["flex_ext"] = rows
    (OUT / "grid.json").write_text(json.dumps(G, ensure_ascii=False, indent=2))
    b0j, b0i = G["base_train"]
    good = [r for r in rows if r[3] <= b0j * 1.05]
    print(f"  {len(good)}/{len(rows)} 格满足训练关节不劣化 5%")
    print(f"  {'关节':>6s}{'ω Hz':>8s}{'ζ':>7s}{'训练关节':>10s}{'训练IMU':>10s}{'vs 基线':>10s}")
    for r in sorted(good, key=lambda r: r[4])[:8]:
        print(f"  {'j'+str(r[0]+1):>6s}{r[1]:>8.2f}{r[2]:>7.2f}{r[3]:>10.4f}{r[4]:>10.4f}"
              f"{(r[4]/b0i-1)*100:>9.1f}%")
    allf = G["flex"] + rows
    bf = min([r for r in allf if r[3] <= b0j * 1.05], key=lambda r: r[4])
    print(f"\n  **F 合并后最优**: j{bf[0]+1} ω={bf[1]:.2f} ζ={bf[2]:g}"
          f"  训练 IMU {bf[4]:.4f} ({(bf[4]/b0i-1)*100:+.1f}%)")
    if bf[2] >= 0.7:
        print(f"  ⚠ **ζ = {bf[2]:g} ≥ 0.7 ⇒ 按预注册不得称「柔性共振模态」**，"
              f"只能称「电机–连杆间的有效滞后 / 低通」")
    return 0


def cmd_gate(a) -> int:
    """读 grid.json 的最优格，跑九条门禁 + 步长稳定检查。"""
    import g_q_v1 as GQ
    from sim.step1_v2_loss import load_calib
    from sim.step2_fidelity import CALIB
    from g_q_v1 import THETA_STAR as TH
    R = load_calib(CALIB)[0]
    G = json.loads((OUT / "grid.json").read_text())
    b0j, b0i = G["base_train"]
    best = {}
    for kind, key in (("F", "flex"), ("B", "bracket")):
        rows_all = G[key] + (G.get("flex_ext", []) if key == "flex" else [])
        good = [r for r in rows_all if r[3] <= b0j * 1.05]
        best[kind] = min(good, key=lambda r: r[4])
    print("**九条门禁**（预注册；A/B 选结构，现有 pp2 作开发门禁）\n")
    for kind in ("F", "B"):
        d0, w, z, tj, ti = best[kind]
        lab = f"j{d0+1}" if kind == "F" else "xyz"[d0]
        print(f"  候选 {kind} 最优格: {lab}  ω={w:.2f} Hz  ζ={z:g}"
              f"   训练 IMU {ti:.4f} ({(ti/b0i-1)*100:+.1f}%)", flush=True)
    print()
    res = {}
    for kind in ("F", "B"):
        d0, w, z, _, _ = best[kind]
        res[kind] = gate_eval(kind, (d0, float(w), z), TH, R)
        print(f"  ── 候选 {kind} ──")
        print(f"  {'数据集':8s}{'关节 基线':>11s}{'关节 候选':>11s}{'Δ':>8s}"
              f"{'IMU 基线':>11s}{'IMU 候选':>11s}{'Δ':>8s}")
        for sn, v in res[kind].items():
            print(f"  {sn:8s}{v['bj'].mean():>11.4f}{v['cj'].mean():>11.4f}"
                  f"{(v['cj'].mean()/v['bj'].mean()-1)*100:>7.1f}%"
                  f"{v['bi'].mean():>11.4f}{v['ci'].mean():>11.4f}"
                  f"{(v['ci'].mean()/v['bi'].mean()-1)*100:>7.1f}%", flush=True)
        v = res[kind]["pp2"]
        d = v["bi"] - v["ci"]
        print(f"    pp2 逐集: 改善 **{int((d>0).sum())}/{len(d)}**，"
              f"中位改善 **{np.median(d/v['bi'])*100:+.1f}%**")
        ok = {}
        for sn in ("留出A", "留出B"):
            u = res[kind][sn]
            ok[f"1{sn}"] = u["cj"].mean() / u["bj"].mean() - 1 <= 0.05
            ok[f"2{sn}"] = u["ci"].mean() / u["bi"].mean() - 1 <= -0.05
        u = res[kind]["留出A"]
        gg = {}
        for (t, p), a1, a2 in zip(u["groups"], u["bi"], u["ci"]):
            gg.setdefault((t, p), [0.0, 0.0])
            gg[(t, p)][0] += a1; gg[(t, p)][1] += a2
        nw = sum(1 for k2, vv in gg.items() if vv[1] < vv[0])
        ok["3"] = nw > len(gg) / 2
        ok["4"] = v["ci"].mean() / v["bi"].mean() - 1 <= -0.05
        ok["5"] = int((d > 0).sum()) >= 9
        ok["6"] = float(np.median(d / v["bi"])) > 0
        ok["7"] = v["cj"].mean() / v["bj"].mean() - 1 <= 0.05
        print(f"    条件 1 关节留出 {'✓' if ok['1留出A'] and ok['1留出B'] else '✗'}"
              f" · 2 IMU留出≥5% {'✓' if ok['2留出A'] and ok['2留出B'] else '**✗**'}"
              f" · 3 分组多数 {nw}/{len(gg)} {'✓' if ok['3'] else '**✗**'}"
              f" · 4 pp2 IMU≥5% {'✓' if ok['4'] else '**✗**'}"
              f" · 5 pp2 ≥9/17 {'✓' if ok['5'] else '**✗**'}"
              f" · 6 pp2 中位>0 {'✓' if ok['6'] else '**✗**'}"
              f" · 7 pp2 关节 {'✓' if ok['7'] else '**✗**'}")
        res[kind]["_ok"] = ok
        print()

    # 条件 9: 步长稳定检查
    print("  ── 条件 9 步长稳定检查（最优格）──")
    from optuna_ident_enc import split
    tr = split()[0][:12]
    dsm = prep(tr, False)
    d0, w, z, _, _ = best["F"]
    q1, g1 = zip(*[roll_flex(flex_xml(d0, float(w), z, TH), it, TH) for it in dsm])
    q2, g2 = zip(*[roll_flex(flex_xml(d0, float(w), z, TH), it, TH, 0.5) for it in dsm])
    _, i1 = score(list(q1), list(g1), dsm, R)
    _, i2 = score(list(q2), list(g2), dsm, R)
    dF = abs(i2.mean() / i1.mean() - 1) * 100
    d0b, wb, zb, _, _ = best["B"]
    cache = rigid_hi(dsm, TH)
    _, j1 = eval_bracket(cache, dsm, R, d0b, float(wb), zb)
    j2v = []
    for c, it in zip(cache, dsm):
        n = len(c[2]); t = np.arange(n) / c[4]
        t4 = np.arange(0, t[-1], 1.0 / (4 * c[4]))
        hi4 = np.stack([np.interp(t4, t, c[2][:, k]) for k in range(3)], 1)
        r4 = np.clip((c[3] * 4), 0, len(hi4) - 1)
        j2v.append(bracket_gyro(hi4, 4 * c[4], r4, d0b, float(wb), zb))
    _, i2b = score([c[0] for c in cache], j2v, dsm, R)
    dB = abs(i2b.mean() / j1.mean() - 1) * 100
    print(f"    F: dt 减半 ⇒ IMU 变化 **{dF:.2f}%** {'✓' if dF < 2 else '**✗**'}")
    print(f"    B: 滤波率 ×4 ⇒ IMU 变化 **{dB:.2f}%** {'✓' if dB < 2 else '**✗**'}")

    # 条件 8 只在两候选各自均过 1–7、9 时用于判优；若只有一个候选通过，
    # 不要求它再战胜一个本身不合格的候选。
    baseF = all(res["F"]["_ok"].values()) and dF < 2
    baseB = all(res["B"]["_ok"].values()) and dB < 2
    fA = res["F"]["留出A"]["ci"].mean()
    bA = res["B"]["留出A"]["ci"].mean()
    winF = fA <= 0.95 * bA
    winB = bA <= 0.95 * fA
    if baseF and baseB:
        passF, passB = bool(winF), bool(winB)
    else:
        passF, passB = bool(baseF), bool(baseB)
    print(f"    条件 8 留出A 判优: F {fA:.4f} / B {bA:.4f} ⇒ "
          f"{'F胜' if winF else ('B胜' if winB else '差异不足5%')}"
          f"{'（仅两者基础门均过时生效）' if not (baseF and baseB) else ''}")

    pp2_none = all(res[k]["pp2"]["ci"].mean() >= res[k]["pp2"]["bi"].mean()
                   for k in ("F", "B"))
    print(f"\n{'='*80}")
    if passF or passB:
        print(f"⇒ 通过: {'F' if passF else ''}{'B' if passB else ''}")
    else:
        print("⇒ **两候选均未通过门禁。**")
        if pp2_none:
            print("   **且两者在 pp2 上都不改善 ⇒ 立即收口**：")
            print("   它们不能解决当前任务的 IMU 缺口，**不再扩大柔性结构**。")
    (OUT / "gate.json").write_text(json.dumps({
        "best": {k: list(v) for k, v in best.items()},
        "stability_pct": {"F": dF, "B": dB},
        "condition8": {"F_vs_B_holdoutA": float(fA / bA - 1),
                       "winF": bool(winF), "winB": bool(winB),
                       "applied": bool(baseF and baseB)},
        "sets": {k: {sn: {kk: (vv.tolist() if hasattr(vv, "tolist") else vv)
                          for kk, vv in u.items() if kk != "groups"}
                     for sn, u in res[k].items() if sn != "_ok"} for k in res},
        "passF": bool(passF), "passB": bool(passB), "pp2_none": bool(pp2_none),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    _c = main()
    sys.stdout.flush()
    import os
    os._exit(_c)
