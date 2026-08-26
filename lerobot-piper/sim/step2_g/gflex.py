#!/usr/bin/env python3
"""**G-flex 最小版** —— 只在 j2 加**一个**柔性自由度，检验最小柔性结构能否解释残差。

## 这一步在检验什么（措辞受限）

`§4.3` 测到「真机静态刚度 ≥1.5e5 N·m/rad（无重力垂落）」与「移动后腕部余振
隐含 ~170 N·m/rad」相差 **773–939 倍**。该差距**支持**柔性假设，但**不能唯一
证明它** —— 同样能产生「静态很硬、动态却在 ~3 Hz 振」的还有：固件的重力补偿、
整臂结构弯曲模态、以及 **IMU 支架自身的模态**（有径向伸出，完全未建模）。
旁证也指向不止一个模态：j1 阶跃激起的是「j2 主导的臂下垂模」而非 j1 自己的模态，
且 §4.3 的频率–惯量标度**未能复现**。

⇒ 故本模块的定位是 **「检验最小柔性结构能否解释残差」**，
   **不是**「已确认传动柔性」，更不是「已分别测出传动刚度与电机惯量」。

## 参数化（只拟合可辨识的组合量）

在 `link1` 与 `link2` 之间插入 `link2_rotor`：执行器留在电机侧的 `joint2`，
连杆侧多一个被动的 `joint2_flex`（弹簧 + 阻尼）。编码器读电机侧，IMU 看连杆侧
—— 与真机的测点位置一致。

三个**组合参数**（逐个都不是单独的物理量）：
    ω_f      柔性模态频率 (Hz)        → 弹簧刚度 K = M_eff · (2π ω_f)²
    ζ_f      模态阻尼比               → 阻尼 C = 2 ζ_f (2π ω_f) M_eff
    kp2mul   电机侧伺服刚度倍数        → 决定静态刚度, 并与 ω_f 共同决定模态被
                                       激励的强度（即用户所说的「耦合强度」）
`M_eff` 取自模型在参考位形下 `joint2_flex` 的质量矩阵对角元 —— 它是**模型给的**，
所以 **K 与电机惯量并未被分别辨识**，只有上面三个组合量被拟合。

## 纪律

· 只用 **90 条激励套件**（`calibration`）拟合，**不碰 pp2 那 17 集**
  （那 17 集已用于选择限幅器/kp/结构，现称 **pp2 开发基准集**，不是留出集）。
· 按**位形 / 序列类型**分组验证；**若多数分组没有一致改善就停止柔性扩展**。
· 不覆盖 `piper_model.xml`；G-flex 写到 `outputs/step2_g/gflex/`。

用法:
    python sim/step2_g/gflex.py --grid          # 在 90 条上拟合三参数
    python sim/step2_g/gflex.py --check         # 只建模型并自检结构
"""
from __future__ import annotations
import argparse, copy, json, sys, time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "sim/step2_g"))

from sim.step1_v2_loss import BAND, bandpass41, load_calib      # noqa: E402
from sim.step2_gap_baseline import MODEL, aa_down                # noqa: E402
from sim.step2_fidelity import apply_limiter                     # noqa: E402

OUT = ROOT / "outputs/step2_g/gflex"
COL = ROOT / "outputs/step2_g/collect"
SUITE = ROOT / "outputs/step2_g"
CALIB = ROOT / "outputs/p2_labels/pp2a-deploy/manifest.json"
FPS = 20.0
MOTOR_HZ = 40.0                       # 电机侧带宽(建模选择, 见 build_flex_xml)
FLEX_RANGE = 0.35                     # rad, 柔性自由度的行程上限(防数值爆炸)
REF_POSE = np.radians([-4.9, 51.5, -25.2, 0.0, 9.4, 4.0])   # HOME, 定义 M_eff


def build_flex_xml(K: float, C: float, kp2mul: float, path: Path) -> Path:
    """在 j2 处插入一个柔性自由度, 写出新 XML。**不改原文件。**"""
    tree = ET.parse(MODEL)
    root = tree.getroot()
    parent = next(b for b in root.iter("body")
                  if any(c.get("name") == "link2" for c in b.findall("body")))
    link2 = next(c for c in parent.findall("body") if c.get("name") == "link2")
    j2 = next(j for j in link2.findall("joint") if j.get("name") == "joint2")

    rotor = ET.Element("body", {"name": "link2_rotor",
                                "pos": link2.get("pos", "0 0 0"),
                                "quat": link2.get("quat", "1 0 0 0")})
    # 电机侧转子: 只给很小的惯量占位 —— 它的作用是承载 joint2 与执行器,
    # **不代表已辨识出的电机惯量**(那个量本模块不声称测到)。
    ET.SubElement(rotor, "inertial", {"pos": "0 0 0", "mass": "0.001",
                                      "diaginertia": "1e-6 1e-6 1e-6"})
    link2.remove(j2)
    # **电机侧必须有反射惯量**, 否则 kp2 一大 ωn=√(kp/Ia) 就冲到 1e5 rad/s,
    # 在 dt=2 ms 下必然发散 —— 那是在测积分器, 不是在测物理。
    # 这里把电机侧带宽**钉死**在 MOTOR_HZ, 反解 armature = kp2/(2π·MOTOR_HZ)²。
    # 这是为数值可解而做的**建模选择**, 已记录; **不是**辨识出来的电机惯量。
    kp2 = float(next(a_ for a_ in root.iter("position")
                     if a_.get("joint") == "joint2").get("kp", 200)) * kp2mul
    j2.set("armature", f"{kp2 / (2 * np.pi * MOTOR_HZ) ** 2:.6g}")
    rotor.append(j2)                                    # joint2 移到电机侧
    link2.set("pos", "0 0 0")
    if "quat" in link2.attrib:
        del link2.attrib["quat"]                        # 变换已由 rotor 承担
    flex = ET.Element("joint", {
        "name": "joint2_flex", "pos": "0 0 0", "axis": j2.get("axis", "0 0 1"),
        "limited": "true", "range": f"{-FLEX_RANGE} {FLEX_RANGE}",
        "stiffness": f"{K:.6g}", "damping": f"{C:.6g}", "springref": "0"})
    link2.insert(1, flex)                               # 紧跟 inertial 之后
    idx = list(parent).index(link2)
    parent.remove(link2)
    rotor.append(link2)
    parent.insert(idx, rotor)

    # 新 XML 写在别的目录 ⇒ 网格的相对路径会失效, 把 meshdir 钉成绝对路径
    comp = root.find("compiler")
    if comp is None:
        comp = ET.Element("compiler")
        root.insert(0, comp)
    md = comp.get("meshdir", "")
    comp.set("meshdir", str((Path(MODEL).parent / md).resolve()))

    for act in root.iter("position"):                   # 电机侧伺服刚度
        if act.get("joint") == "joint2":
            act.set("kp", f"{float(act.get('kp', 200)) * kp2mul:.6g}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def m_eff(path: Path) -> float:
    """参考位形下 `joint2_flex` 的质量矩阵对角元 —— 用来把 ω_f 换算成 K。"""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    adr = {}
    for jn in ("joint1", "joint2", "joint2_flex", "joint3", "joint4",
               "joint5", "joint6"):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        adr[jn] = (int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid]))
    for k, jn in enumerate(("joint1", "joint2", "joint3", "joint4",
                            "joint5", "joint6")):
        d.qpos[adr[jn][0]] = REF_POSE[k]
    mujoco.mj_forward(m, d)
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, M, d.qM)
    return float(M[adr["joint2_flex"][1], adr["joint2_flex"][1]]), adr


def make(omega_f: float, zeta_f: float, kp2mul: float) -> tuple[Path, dict]:
    """两遍: 先用占位刚度建模型量 M_eff, 再按 ω_f/ζ_f 换算出 K/C 重建。"""
    tmp = OUT / "_probe.xml"
    build_flex_xml(100.0, 1.0, kp2mul, tmp)
    Me, adr = m_eff(tmp)
    w = 2 * np.pi * omega_f
    K = Me * w * w
    C = 2.0 * zeta_f * w * Me
    path = OUT / f"piper_flex_w{omega_f:g}_z{zeta_f:g}_kp{kp2mul:g}.xml"
    build_flex_xml(K, C, kp2mul, path)
    return path, {"M_eff": Me, "K": K, "C": C, "adr": adr}


def sim_gyro_flex(cmd7, xml: Path, adr: dict, nframes: int,
                  limiter: str = "a") -> np.ndarray:
    """限幅器 → G-flex → 20 Hz 陀螺。关节按**名字**索引, 不硬编码 qpos 下标。"""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(xml))
    act = apply_limiter(np.asarray(cmd7, float), limiter)
    dt = m.opt.timestep
    nsub = int(round(1 / FPS / dt))
    d = mujoco.MjData(m)
    qa = [adr[f"joint{k+1}"][0] for k in range(6)]
    for k in range(6):
        d.qpos[qa[k]] = act[0, k]
    mujoco.mj_forward(m, d)
    d.ctrl[:6] = act[0, :6]
    for _ in range(int(0.3 / dt)):
        mujoco.mj_step(m, d)
    T = len(act)
    hi = np.zeros((T * nsub, 6))
    for t in range(T):
        d.ctrl[:6] = act[t, :6]
        for k in range(nsub):
            mujoco.mj_step(m, d)
            hi[t * nsub + k] = d.sensordata[:6]
    return aa_down(hi, 1 / dt, nframes)[:, 3:6]


# ────────────────────── 数据与评价 ──────────────────────
def load_suite(quick: bool = True, n: int | None = None):
    """→ [(name, kind, target, pose_key, cmd7, real_gyro20)]。与 calibrate.py 同源。"""
    man = json.loads((SUITE / "manifest.json").read_text())
    done = [r["name"] for r in json.loads((COL / "state.json").read_text())["done"]]
    out = []
    for nm in sorted(done):
        if quick and man[nm]["kind"] == "sweep" and man[nm]["frames"] > 500:
            continue
        d = json.loads((COL / f"replay_{nm}.json").read_text())
        pose = np.load(SUITE / f"{nm}.npz")["pose"]
        out.append((nm, man[nm]["kind"], tuple(man[nm]["target"]),
                    tuple(np.round(np.degrees(pose), 0)),
                    np.asarray(d["cmd"], float),
                    np.asarray(d["imu"], float)[:, 3:6]))
    return out[:n] if n else out


def resid(sim, real, R):
    """→ (残差能量, 实测能量)  带通 0.5–9 Hz, 与 step2_fidelity 同口径。"""
    k = min(len(sim), len(real))
    a = bandpass41(real[:k])
    b = bandpass41(sim[:k] @ R.T)
    if not np.isfinite(b).all():
        return np.inf, 1.0
    return float(np.square(a - b).sum()), float(np.square(a).sum())


def eval_flex(data, R, xml, adr):
    """→ {name: (e, t)}。"""
    out = {}
    for nm, kind, tgt, pose, cmd, real in data:
        try:
            sim = sim_gyro_flex(cmd, xml, adr, len(real))
        except Exception:                                     # noqa: BLE001
            out[nm] = (np.inf, 1.0)
            continue
        out[nm] = resid(sim, real, R)
    return out


def eval_rigid(data, R):
    """基线 = **限幅器 + 刚性 MuJoCo**（G-v1 + 限幅，kp×1），完全同口径。"""
    from calibrate import sim_gyro
    out = {}
    for nm, kind, tgt, pose, cmd, real in data:
        try:
            sim = sim_gyro(cmd, 1.0, 1.0, len(real), "a")
        except Exception:                                     # noqa: BLE001
            out[nm] = (np.inf, 1.0)
            continue
        out[nm] = resid(sim, real, R)
    return out


def groups(data):
    """按**序列类型**和**目标关节**分组（位形隐含在关节内：每关节 1–3 个位形）。"""
    g = {}
    for nm, kind, tgt, pose, cmd, real in data:
        g.setdefault(f"kind={kind}", []).append(nm)
        if len(tgt) == 1:
            g.setdefault(f"j{tgt[0]+1}", []).append(nm)
            g.setdefault(f"j{tgt[0]+1} {pose}", []).append(nm)
    return g


def ratio(res, names):
    e = sum(res[n][0] for n in names if n in res)
    t = sum(res[n][1] for n in names if n in res)
    return e / max(t, 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--n", type=int, default=None, help="只用前 N 条(提速)")
    ap.add_argument("--wf", default="2.0,3.0,4.5")
    ap.add_argument("--zf", default="0.10,0.25")
    ap.add_argument("--kp2", default="1,10")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if a.check:
        xml, info = make(3.0, 0.15, 1.0)
        import mujoco
        m = mujoco.MjModel.from_xml_path(str(xml))
        print(f"G-flex 结构自检 → {xml.name}")
        print(f"  nq={m.nq} nv={m.nv} nu={m.nu}  (刚性版 nq=8)")
        for j in range(m.njnt):
            print(f"  joint[{j}] {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j):12s}"
                  f" qposadr={m.jnt_qposadr[j]} stiff={m.jnt_stiffness[j]:.4g}")
        print(f"  M_eff@HOME = {info['M_eff']:.5f} kg·m²  ⇒ K={info['K']:.1f} N·m/rad"
              f"  C={info['C']:.3f}")
        print(f"  **只加了 1 个自由度, 只在 j2** —— 未扩到其余关节。")
        return 0

    R, _ = load_calib(CALIB)
    data = load_suite(True, a.n)
    print(f"标定集: 90 条套件中的 {len(data)} 条（pp2 开发基准集 17 集**未参与**）")
    print("基线 = 限幅器 + 刚性 MuJoCo(kp×1)；候选 = 限幅器 + G-flex(j2 单模态)\n")
    base = eval_rigid(data, R)
    gs = groups(data)
    b_all = ratio(base, [d[0] for d in data])
    print(f"基线 全体残差/实测 = {b_all:.4f}\n")

    print(f"{'ω_f Hz':>8s}{'ζ_f':>7s}{'kp2×':>7s}{'全体':>9s}{'vs基线':>9s}"
          f"{'改善的分组':>12s}{'秒':>6s}")
    best = (np.inf, None)
    rows = []
    for wf in [float(x) for x in a.wf.split(",")]:
        for zf in [float(x) for x in a.zf.split(",")]:
            for kp2 in [float(x) for x in a.kp2.split(",")]:
                t0 = time.time()
                xml, info = make(wf, zf, kp2)
                res = eval_flex(data, R, xml, info["adr"])
                r_all = ratio(res, [d[0] for d in data])
                win = sum(1 for k, v in gs.items() if ratio(res, v) < ratio(base, v))
                rows.append((wf, zf, kp2, r_all, win, len(gs), res))
                print(f"{wf:>8.2f}{zf:>7.2f}{kp2:>7g}{r_all:>9.4f}"
                      f"{(1 - r_all / b_all) * 100:>+8.1f}%{win:>8d}/{len(gs):<3d}"
                      f"{time.time() - t0:>6.0f}", flush=True)
                if r_all < best[0]:
                    best = (r_all, (wf, zf, kp2), res)
    print(f"\n**最优 G-flex: ω_f={best[1][0]:g} Hz  ζ_f={best[1][1]:g}  "
          f"kp2×{best[1][2]:g}  残差/实测={best[0]:.4f}**")
    print(f"  vs 基线 {b_all:.4f}  ⇒ {(1 - best[0] / b_all) * 100:+.1f}%")

    print(f"\n分组一致性（判据: **多数分组无一致改善就停止柔性扩展**）")
    print(f"{'分组':<22s}{'条':>4s}{'基线':>9s}{'G-flex':>9s}{'改善':>9s}")
    nwin = 0
    for k in sorted(gs):
        rb, rf = ratio(base, gs[k]), ratio(best[2], gs[k])
        d_ = (1 - rf / rb) * 100 if np.isfinite(rb) and rb > 0 else np.nan
        nwin += d_ > 0
        print(f"{k:<22s}{len(gs[k]):>4d}{rb:>9.4f}{rf:>9.4f}{d_:>+8.1f}%")
    print(f"\n改善的分组: **{nwin}/{len(gs)}**")
    print("⇒ " + ("**多数分组一致改善 —— 可考虑逐关节扩展 / 长保持实验拆物理参数**"
                  if nwin > len(gs) * 0.6 else
                  "**多数分组无一致改善 —— 按预定判据停止柔性扩展**"))
    print("\n注: 本步只检验「最小柔性结构能否解释残差」。ω_f/ζ_f/kp2 是**组合量**，"
          "\n    **未**声称已分别测出传动刚度与电机惯量。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
