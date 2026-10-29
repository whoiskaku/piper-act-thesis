#!/usr/bin/env python3
"""在 **90 条激励套件**（`calibration` 角色）上拟合 G 的伺服参数。

角色分离（`schema.py` §数据角色）:
    calibration = 90 条激励套件（本脚本用）
    holdout     = 17 集 pp2 裸 ACT（**本脚本不碰**，留给 `step2_fidelity.py` 评估）

此前的 kp 扫描是在那 17 集上做的 ⇒ 那是**在 holdout 上调参**，数字不能作为留出性能
引用。本脚本把拟合搬回 calibration 集，之后 `step2_fidelity.py` 上的数字才有意义。

模型结构（串联，非二选一）:
    指令 → [加速度限幅 a_j]（运动学，位形无关，已在阶跃上定） → [MuJoCo 伺服 kp/damping] → 关节

目标量 = **带通后的陀螺**（与 `step2_fidelity` 同口径 0.5–9 Hz），因为 G 最终要
预测的就是 IMU；而不是编码器位置——限幅器之后编码器侧残差已接近噪声底。

用法:
    python sim/step2_g/calibrate.py --grid           # 全局 (kp×, damping×) 网格
    python sim/step2_g/calibrate.py --grid --quick   # 只用 chirp+step，跳过长扫描
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "sim/step2_g"))

from sim.step1_v2_loss import BAND, bandpass41, load_calib          # noqa: E402
from sim.step2_gap_baseline import MODEL, aa_down                    # noqa: E402
from sim.step2_fidelity import apply_limiter                         # noqa: E402

COL = ROOT / "outputs/step2_g/collect"
SUITE = ROOT / "outputs/step2_g"
CALIB = ROOT / "outputs/p2_labels/pp2a-deploy/manifest.json"
FPS = 20.0


def sim_gyro(cmd7: np.ndarray, kpmul: float, dmul: float, nframes: int,
             limiter: str = "a") -> np.ndarray:
    """限幅器 → MuJoCo(kp×kpmul, damping×dmul) → 降采样到 20 Hz 的陀螺三轴。"""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    for j in range(6):
        kp = float(m.actuator_gainprm[j, 0]) * kpmul
        m.actuator_gainprm[j, 0] = kp
        m.actuator_biasprm[j, 1] = -kp
        m.dof_damping[j] = float(m.dof_damping[j]) * dmul
    act = apply_limiter(cmd7, limiter)
    dt = m.opt.timestep
    nsub = int(round(1 / FPS / dt))
    d = mujoco.MjData(m)
    g0 = float(np.clip(act[0, 6], 0, 0.035)) if act.shape[1] > 6 else 0.0
    d.qpos[:6] = act[0, :6]
    if m.nq > 7:
        d.qpos[6], d.qpos[7] = g0, -g0
    mujoco.mj_forward(m, d)
    d.ctrl[:6] = act[0, :6]
    if m.nu > 7:
        d.ctrl[6], d.ctrl[7] = g0, -g0
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


def load_suite(quick: bool):
    """→ [(name, cmd7, real_gyro_20Hz)]。`quick` 跳过很长的慢扫描。"""
    man = json.loads((SUITE / "manifest.json").read_text())
    done = [r["name"] for r in json.loads((COL / "state.json").read_text())["done"]]
    out = []
    for nm in done:
        if quick and man[nm]["kind"] == "sweep" and man[nm]["frames"] > 500:
            continue
        d = json.loads((COL / f"replay_{nm}.json").read_text())
        cmd = np.asarray(d["cmd"], float)
        real = np.asarray(d["imu"], float)[:, 3:6]
        out.append((nm, cmd, real))
    return out


def loss(data, R, kpmul, dmul, limiter="a") -> tuple[float, float]:
    """→ (带通残差能量 / 带通实测能量, 逐条中位相对残差)。"""
    num = den = 0.0
    per = []
    for nm, cmd, real in data:
        try:
            sim = sim_gyro(cmd, kpmul, dmul, len(real), limiter)
        except Exception:                                    # noqa: BLE001
            return np.inf, np.inf
        n = min(len(sim), len(real))
        a = bandpass41(real[:n] )
        b = bandpass41(sim[:n] @ R.T)
        if not np.isfinite(b).all():
            return np.inf, np.inf
        e = float(np.square(a - b).sum()); t = float(np.square(a).sum())
        num += e; den += t
        per.append(e / max(t, 1e-12))
    return num / max(den, 1e-12), float(np.median(per))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--kp", default="1,5,10,25,50")
    ap.add_argument("--damp", default="0.2,1,5")
    ap.add_argument("--limiter", choices=("none", "a", "ad"), default="a")
    a = ap.parse_args()
    R, _ = load_calib(CALIB)
    data = load_suite(a.quick)
    print(f"标定集: **90 条激励套件中的 {len(data)} 条**（holdout 的 17 集 pp2 未参与）")
    print(f"目标: 带通 {BAND[0]}–{BAND[1]} Hz 陀螺三轴的残差能量占比（越小越好）\n")
    kps = [float(x) for x in a.kp.split(",")]
    dms = [float(x) for x in a.damp.split(",")]
    print(f"{'kp×':>7s}{'damp×':>8s}{'残差/实测':>11s}{'逐条中位':>10s}{'秒':>7s}")
    best = (np.inf, None, None)
    for kp in kps:
        for dm in dms:
            t0 = time.time()
            r, med = loss(data, R, kp, dm, a.limiter)
            print(f"{kp:>7g}{dm:>8g}{r:>11.4f}{med:>10.4f}{time.time()-t0:>7.0f}",
                  flush=True)
            if r < best[0]:
                best = (r, kp, dm)
    print(f"\n**最优: kp×{best[1]:g}  damping×{best[2]:g}  残差/实测 = {best[0]:.4f}**")
    print("(1.0 = 预测全零; <1 才算解释了东西)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
