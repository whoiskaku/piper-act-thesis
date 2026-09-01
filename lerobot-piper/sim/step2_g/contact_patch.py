#!/usr/bin/env python3
"""**局部结构补丁：桌面接触**（一次固定预算的验证）。

## 为什么是桌面而不是物体

`reports/PP2_PHASE_FINDING.md`（2026-08-04，由研究者回放录像发现、数据证实）：
**pp2 的大 gap ＝ 机械臂压在桌面上** —— ACT 指令要末端走到桌面以下，
真机被挡住，伺服继续下压（压深约 4.4 mm、持续约 0.20 s），
且**三臂压深相同 ⇒ 是 ACT 指令本身的性质**。

而 `sim/piper_model.xml` 里 `contype/conaffinity=0` ⇒ **手臂直接穿过桌面**。
这与 §20 的实测一致：残差在夹爪开合窗富集 **2.44×**、窗内 r/m 0.778 vs 窗外 0.494。

⇒ 该补的是**桌面接触**，不是物体抓取。（物体侧另有 `build_object_model.py`，
本补丁**不动它** —— pp2 日志没有物体位姿，且证据指向桌面。）

## 预算（固定，一次，不追加）

| 阶段 | 内容 | 次数 |
|---|---|---|
| 标定 | 桌面高度 `z_t` 的 8 点网格，**只用编码器侧**的「停住高度」匹配 | 8 × 17 |
| 评估 | 选定 `z_t` 上算 IMU + 关节 + 出图 | 1 × 17 |

- **不扫接触刚度**：用 `sim/object_sim_config.json` 里已标定的
  `solref [0.2,1.0] · solimp [0.9,0.95,0.001] · condim 4 · friction [1.0,0.005,1e-4]`
  （`calib_contact` 2026-05-30 对真机靶标定过）。
- **只开夹爪↔桌面**（`contype/conaffinity=4`），其余 geom 保持 0
  ⇒ 不会重新引入 2026-07-08 修过的 mesh 自碰撞伪影（那个 bug 曾把 j1 锁死）。
- G-q-v1 的 θ\* **逐字不动**。

## 判读（预先定死）

| # | 条件 | 阈值 |
|---|---|---|
| 1 | **接触窗内** r/m 相对 0.778 下降 | **≥ 20%**（即 ≤ 0.622） |
| 2 | 窗外 r/m 不恶化 | ≤ +5% |
| 3 | pp2 关节 RMS 不恶化 | ≤ +5% |
| — | 总体 r/m | **只报告**。天花板已算出 **12.5%**（§20.3），不作为通过条件 |

**这是局部结构补丁，不预设它能救活整个 pp2 保真度。**

改完 G 必出 pp2 波形对比图（常规）。

用法:
    python sim/step2_g/contact_patch.py --calib     # 标定 z_t（只用编码器）
    python sim/step2_g/contact_patch.py --eval      # 评估 + 出图
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
OUT = ROOT / "outputs/step2_g/contact"
# **配置订正（2026-08-30，标定跑完后）**：
# `object_sim_config.json` 的 `solref timeconst = 0.2 s` 是给「61 g 胶带静置桌面」
# 标定的（`calib_contact` 2026-05-30），**不适用于刚性伺服下压**。实测：
#   timeconst 0.2 → 仿真最低 z −10.1 mm（比实测 −6.39 还低）⇒ 压穿, 干预未实现
#   timeconst 0.02（MuJoCo 默认）→ 被挡住
# ⇒ 改用 MuJoCo 默认 solref。这是修正配置缺陷，不是参数搜索：
#   只取标准默认值一个点，预算与预注册相同（8×17 标定 + 1×17 评估）。
#   其余接触参数（friction/solimp/condim）仍用已标定值。
CFG = {**json.loads((ROOT / "sim/object_sim_config.json").read_text())["contact"],
       "solref": [0.02, 1.0]}
GRIP_GEOMS = ("link6", "gripper_base", "link7", "link8")   # 只这几个参与碰撞
Z_GRID = np.linspace(-0.030, 0.005, 8)                     # 桌面顶面 z (m), 固定 8 点


def build_xml(z_top: float, path: Path) -> Path:
    """加一块**只与夹爪碰撞**的桌面；其余 geom 的 contype/conaffinity 保持 0。"""
    from sim.step2_gap_baseline import MODEL
    tree = ET.parse(MODEL)
    root = tree.getroot()
    wb = root.find("worldbody")
    ET.SubElement(wb, "geom", {
        "name": "table_col", "type": "box",
        "pos": f"0.22 0 {z_top - 0.02:.6f}", "size": "0.42 0.36 0.02",
        "contype": "4", "conaffinity": "4", "rgba": "0.85 0.80 0.70 0.35",
        "friction": " ".join(f"{v:g}" for v in CFG["friction"]),
        "solref": " ".join(f"{v:g}" for v in CFG["solref"]),
        "solimp": " ".join(f"{v:g}" for v in CFG["solimp"]),
        "condim": str(CFG["condim"])})
    for g in root.iter("geom"):
        if g.get("mesh") in GRIP_GEOMS:
            g.set("contype", "4"); g.set("conaffinity", "4")
            g.set("friction", " ".join(f"{v:g}" for v in CFG["friction"]))
            g.set("solref", " ".join(f"{v:g}" for v in CFG["solref"]))
            g.set("solimp", " ".join(f"{v:g}" for v in CFG["solimp"]))
            g.set("condim", str(CFG["condim"]))
    comp = root.find("compiler")
    md = comp.get("meshdir", "")
    comp.set("meshdir", str((Path(MODEL).parent / md).resolve()))
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def roll(xml: Path, act, theta):
    """事件精确回放（与 g_q_v1 同惯例）→ (关节角 度, 20Hz 陀螺, EE site z 序列)。"""
    import mujoco
    from scipy.signal import butter, filtfilt
    from sim.step2_fidelity import A_MAX
    from optuna_ident_enc import accel_limited_dt
    import params as P
    m = mujoco.MjModel.from_xml_path(str(xml))
    P.apply(m, theta)
    dt = m.opt.timestep
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site")
    n = len(act)
    cmd = np.degrees(act[:, :6])
    ref = np.empty((n, 6))
    for j in range(6):
        ref[:, j] = accel_limited_dt(cmd[:, j], float(A_MAX[j]), np.full(n, 1 / FPS))
    ref = np.radians(ref)
    d = mujoco.MjData(m)
    d.qpos[:6] = act[0, :6]
    g0 = float(np.clip(act[0, 6], 0, 0.035)) if act.shape[1] > 6 else 0.0
    if m.nq > 7:
        d.qpos[6], d.qpos[7] = g0, -g0
    mujoco.mj_forward(m, d)
    d.ctrl[:6] = ref[0]
    if m.nu > 7:
        d.ctrl[6], d.ctrl[7] = g0, -g0
    for _ in range(int(round(0.3 / dt))):
        mujoco.mj_step(m, d)
    send = np.rint(np.arange(n) / FPS / dt).astype(int)
    read = send + 1
    ntot = int(read[-1]) + 1
    hi = np.empty((ntot, 3)); q = np.empty((n, 6)); zz = np.empty(n)
    si = ri = 0
    for s in range(ntot):
        while si < n and send[si] <= s:
            d.ctrl[:6] = ref[si]
            if m.nu > 7:
                d.ctrl[6] = float(np.clip(act[si, 6], 0, 0.035))
                d.ctrl[7] = -d.ctrl[6]
            si += 1
        while ri < n and read[ri] <= s:
            q[ri] = np.degrees(d.qpos[:6]); zz[ri] = d.site_xpos[sid][2]; ri += 1
        hi[s] = d.sensordata[3:6]
        mujoco.mj_step(m, d)
    while ri < n:
        q[ri] = np.degrees(d.qpos[:6]); zz[ri] = d.site_xpos[sid][2]; ri += 1
    ba = butter(4, 0.45 * FPS / ((1.0 / dt) / 2.0))
    return q, filtfilt(*ba, hi, axis=0)[np.clip(read, 0, ntot - 1)], zz


def load_pp2():
    from sim.step2_gap_baseline import bare_logs
    out = []
    for p in bare_logs():
        fr = json.loads(p.read_text())["frames"]
        out.append({"name": p.stem,
                    "act": np.asarray([f["a"] for f in fr], float),
                    "enc": np.asarray([f["state"] for f in fr], float)[:, :6] / 1000.0,
                    "real": np.asarray([f["imu"] for f in fr], float)[:, 3:6]})
    return out


def meas_stall_z(eps):
    """实测编码器的 EE site 最低 z（每集一个数）。"""
    import mujoco
    from sim.step2_gap_baseline import MODEL
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site")
    out = []
    for e in eps:
        z = []
        for q in np.radians(e["enc"]):
            d.qpos[:6] = q; mujoco.mj_forward(m, d); z.append(d.site_xpos[sid][2])
        out.append(float(np.min(z)))
    return np.array(out)


def cmd_calib(_a) -> int:
    """只用**编码器侧**的停住高度标定桌面 z_t。不看 IMU。"""
    from g_q_v1 import THETA_STAR as TH
    eps = load_pp2()
    zm = meas_stall_z(eps)
    print(f"**桌面高度标定（只用编码器，不看 IMU）** {len(eps)} 集")
    print(f"  实测 EE site 最低 z: 中位 {np.median(zm)*1000:.2f} mm "
          f"范围 {zm.min()*1000:.2f}–{zm.max()*1000:.2f}\n")
    print(f"  {'z_t (mm)':>10s}{'仿真最低z 中位(mm)':>20s}{'与实测的 RMS 差(mm)':>21s}")
    best = (9e9, None)
    rows = []
    for zt in Z_GRID:
        xml = build_xml(float(zt), OUT / f"_tab_{zt*1000:.0f}.xml")
        zs = []
        for e in eps:
            try:
                _, _, zz = roll(xml, e["act"], TH)
                zs.append(float(np.min(zz)))
            except Exception:                                # noqa: BLE001
                zs.append(np.nan)
        zs = np.array(zs)
        err = float(np.sqrt(np.nanmean((zs - zm) ** 2))) * 1000
        rows.append((float(zt), float(np.nanmedian(zs)), err))
        print(f"  {zt*1000:>10.1f}{np.nanmedian(zs)*1000:>20.2f}{err:>21.3f}", flush=True)
        if err < best[0]:
            best = (err, float(zt))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calib.json").write_text(json.dumps(
        {"z_grid_mm": [r[0] * 1000 for r in rows], "sim_med_mm": [r[1] * 1000 for r in rows],
         "rms_mm": [r[2] for r in rows], "z_best_mm": best[1] * 1000,
         "rms_best_mm": best[0], "meas_med_mm": float(np.median(zm)) * 1000,
         "note": "只用编码器侧停住高度标定; 未看 IMU"}, ensure_ascii=False, indent=2))
    print(f"\n  **选定 z_t = {best[1]*1000:.1f} mm**（RMS 差 {best[0]:.3f} mm）")
    print(f"  → {OUT/'calib.json'}")
    return 0


def cmd_eval(_a) -> int:
    from sim.step1_v2_loss import bandpass41, load_calib
    from sim.step2_fidelity import CALIB
    from g_q_v1 import THETA_STAR as TH
    import g_q_v1 as GQ
    R = load_calib(CALIB)[0]
    zt = json.loads((OUT / "calib.json").read_text())["z_best_mm"] / 1000.0
    xml = build_xml(zt, OUT / "table_best.xml")
    eps = load_pp2()
    L, W = 1, 10
    acc = {k: 0.0 for k in ("ri", "ro", "mi", "mo", "ri0", "ro0")}
    rows = []
    print(f"**桌面接触补丁的评估**  z_t = {zt*1000:.1f} mm\n")
    print(f"  {'集':22s}{'r/m 无接触':>11s}{'r/m 有接触':>11s}"
          f"{'窗内 无→有':>16s}{'关节RMS 无→有':>17s}")
    for e in eps:
        q0, g0, _ = GQ._sim(e["act"])[0], GQ.predict_gyro(e["act"]), None
        q1, g1, _ = roll(xml, e["act"], TH)
        k = min(len(g0), len(g1), len(e["real"]) - L)
        a_ = bandpass41(e["real"][L:L + k])
        b0 = bandpass41(g0[:k] @ R.T); b1 = bandpass41(g1[:k] @ R.T)
        grip = e["act"][:k, 6]
        dd = np.abs(np.diff(grip, prepend=grip[:1]))
        m = np.zeros(k, bool)
        for i in np.where(dd > 0.5 * max(dd.max(), 1e-9))[0]:
            m[max(0, i - W):min(k, i + W + 1)] = True
        r0, r1, mm = (a_ - b0) ** 2, (a_ - b1) ** 2, a_ ** 2
        acc["ri0"] += r0[m].sum(); acc["ro0"] += r0[~m].sum()
        acc["ri"] += r1[m].sum(); acc["ro"] += r1[~m].sum()
        acc["mi"] += mm[m].sum(); acc["mo"] += mm[~m].sum()
        kk = min(k, len(e["enc"]))
        j0 = float(np.sqrt(np.mean((q0[:kk] - e["enc"][:kk]) ** 2)))
        j1 = float(np.sqrt(np.mean((q1[:kk] - e["enc"][:kk]) ** 2)))
        rows.append((e["name"], r0.sum() / mm.sum(), r1.sum() / mm.sum(),
                     r0[m].sum() / max(mm[m].sum(), 1e-30),
                     r1[m].sum() / max(mm[m].sum(), 1e-30), j0, j1))
        print(f"  {rows[-1][0]:22s}{rows[-1][1]:>11.3f}{rows[-1][2]:>11.3f}"
              f"{rows[-1][3]:>8.3f}→{rows[-1][4]:<7.3f}{j0:>9.3f}→{j1:<7.3f}", flush=True)
    ri0, ri, ro0, ro = acc["ri0"], acc["ri"], acc["ro0"], acc["ro"]
    mi, mo = acc["mi"], acc["mo"]
    tot0, tot1 = (ri0 + ro0) / (mi + mo), (ri + ro) / (mi + mo)
    in0, in1, out0, out1 = ri0 / mi, ri / mi, ro0 / mo, ro / mo
    jj0 = float(np.mean([r[5] for r in rows])); jj1 = float(np.mean([r[6] for r in rows]))
    print(f"\n{'='*74}\n**判读**（预注册）")
    print(f"  1 窗内 r/m {in0:.3f} → **{in1:.3f}** ({(in1/in0-1)*100:+.1f}%)"
          f"   需 ≤ −20%  {'✓' if in1 <= in0*0.8 else '**✗**'}")
    print(f"  2 窗外 r/m {out0:.3f} → {out1:.3f} ({(out1/out0-1)*100:+.1f}%)"
          f"   需 ≤ +5%  {'✓' if out1 <= out0*1.05 else '**✗**'}")
    print(f"  3 关节 RMS {jj0:.3f}° → {jj1:.3f}° ({(jj1/jj0-1)*100:+.1f}%)"
          f"   需 ≤ +5%  {'✓' if jj1 <= jj0*1.05 else '**✗**'}")
    print(f"  — 总体 r/m {tot0:.3f} → **{tot1:.3f}** ({(tot1/tot0-1)*100:+.1f}%)"
          f"   （只报告；天花板 12.5%）")
    (OUT / "eval.json").write_text(json.dumps(
        {"z_t_mm": zt * 1000, "per_ep": rows, "in": [in0, in1], "out": [out0, out1],
         "total": [tot0, tot1], "joint_deg": [jj0, jj1]}, ensure_ascii=False, indent=2))
    print(f"\n  → {OUT/'eval.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    gc.disable()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.calib:
        return cmd_calib(a)
    if a.eval:
        return cmd_eval(a)
    print(__doc__.split("用法:")[0])
    return 0


if __name__ == "__main__":
    _c = main()
    sys.stdout.flush()
    import os
    os._exit(_c)
