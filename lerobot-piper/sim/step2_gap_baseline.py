#!/usr/bin/env python3
"""Step 2 起点测量 —— 当前 MuJoCo 模型对真机 IMU 的保真度 (Gap1 / Gap2)。

指導教員第⑤条: G 必须是**能自动标定真实动力学参数的通用模拟器**, 不是 Δa 响应模型。
故 Step 2 的对象是 `sim/piper_model.xml` 的物理参数, 而不是任何学出来的响应核。

沿用 `sim/dyn_replay_imu.py` 已定的归因分解:

    kin_ideal : 手臂被"规定"贴住指令 (零标定, 是"理想"的定义)
    dyn_ideal : 带 kp/阻尼的伺服模型追指令 (依赖标定)
    Gap1 = kin → dyn   控制器能解释的部分 (滞后 + 低通)
    Gap2 = dyn → real  **未建模的墙本体** ← Step 2 要压的就是这个

口径与 Step 1 的 `L` 完全一致 (0.5–9 Hz, 陀螺三轴, mean‖·‖²), 以便可比。
**对齐**: kin 用 Step 1 冻结的 +3 帧; dyn 的滞后由伺服模型自己产生, 故用
互相关**实测**最优滞后并一并报告 —— 若标定正确, 该值应趋于 0。

用法:
    python sim/step2_gap_baseline.py                 # 全部裸 ACT 集
    python sim/step2_gap_baseline.py --n 5 --jkp 1:5000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))

from sim.step1_v2_loss import ALIGN_L, BAND, bandpass41, load_calib, render_ideal  # noqa: E402

MODEL = ROOT / "sim/piper_model.xml"
FPS = 20.0
CALIB = ROOT / "outputs/p2_labels/pp2a-deploy/manifest.json"


def bare_logs() -> list[Path]:
    """v3 的裸 ACT 集（筛选阶段 8 + 确认阶段 A 臂 9）—— 辨识本体要用无补正的数据。"""
    out = []
    s = json.loads((ROOT / "outputs/step1_v3/state.json").read_text())
    out += [ROOT / f"outputs/step1_v3/{r['tag']}_ep0.json"
            for r in s["done"] if r["arm"] == "bare"]
    c = ROOT / "outputs/step1_v3_confirm/state.json"
    if c.exists():
        d = json.loads(c.read_text())
        out += [ROOT / f"outputs/step1_v3_confirm/{x['A']['ep_file']}"
                if "ep_file" in x["A"] else
                ROOT / f"outputs/step1_v3_confirm/{x['A']['tag']}_ep0.json"
                for x in d["blocks"]]
    return [p for p in out if p.exists()]


def dyn_replay_cmd(act: np.ndarray, jkp: dict[int, float],
                   settle_s: float = 0.5) -> tuple[np.ndarray, float]:
    """指令 → 伺服模型真积分 → 高频 IMU。`dyn_replay_imu.dyn_replay` 的指令版。"""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    for j, kp in jkp.items():
        m.actuator_gainprm[j, 0] = kp
        m.actuator_biasprm[j, 1] = -kp
    dt = m.opt.timestep
    nsub = int(round(1 / FPS / dt))
    data = mujoco.MjData(m)
    g0 = float(np.clip(act[0, 6], 0, 0.035))
    data.qpos[:6], data.qpos[6], data.qpos[7] = act[0, :6], g0, -g0
    mujoco.mj_forward(m, data)
    data.ctrl[:6], data.ctrl[6], data.ctrl[7] = act[0, :6], g0, -g0
    for _ in range(int(settle_s / dt)):
        mujoco.mj_step(m, data)
    T = len(act)
    hi = np.zeros((T * nsub, 6))
    for t in range(T):
        gg = float(np.clip(act[t, 6], 0, 0.035))
        data.ctrl[:6], data.ctrl[6], data.ctrl[7] = act[t, :6], gg, -gg
        for k in range(nsub):
            mujoco.mj_step(m, data)
            hi[t * nsub + k] = data.sensordata[:6]
    return hi, 1 / dt


def aa_down(hi: np.ndarray, fs_hi: float, T: int) -> np.ndarray:
    from scipy.signal import butter, filtfilt
    ba = butter(4, 0.45 * FPS / (fs_hi / 2))
    return filtfilt(*ba, hi, axis=0)[::int(round(fs_hi / FPS))][:T]


def best_lag(a: np.ndarray, b: np.ndarray, maxlag: int = 8) -> tuple[int, float]:
    """使 ‖bp(a[t+L] − b[t])‖² 最小的 L（a 落后 b 则 L>0）。"""
    best = (0, np.inf)
    for L in range(maxlag + 1):
        n = min(len(a) - L, len(b))
        v = float(np.square(bandpass41(a[L:L + n] - b[:n])).sum(1).mean())
        if v < best[1]:
            best = (L, v)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, help="只跑前 N 集")
    ap.add_argument("--jkp", default="1:5000",
                    help="kp 覆盖 'j:kp,...'（默认沿用 dyn_replay_imu 的手调值）")
    a = ap.parse_args()
    jkp = {int(k) - 1: float(v) for k, v in
           (s.split(":") for s in a.jkp.split(",") if s)} if a.jkp else {}
    R, _ = load_calib(CALIB)
    logs = bare_logs()[: a.n]
    print(f"裸 ACT 集 {len(logs)} 个   kp 覆盖 {jkp or '（无，用 XML 原值）'}")
    print(f"口径 {BAND[0]}–{BAND[1]} Hz 陀螺三轴 mean‖·‖²；kin 对齐 +{ALIGN_L} 帧（冻结）\n")
    print(f"{'集':<26}{'L_kin':>9}{'L_dyn':>9}{'dyn滞后':>8}{'Gap1':>9}{'解释率':>8}")
    rows = []
    for p in logs:
        d = json.loads(p.read_text())
        fr = d["frames"]
        act = np.asarray([f["a"] for f in fr], float)
        real = np.asarray([f["imu"] for f in fr], float)[:, 3:6]
        kin = render_ideal(d["chunks"], len(fr))
        n = min(len(fr) - ALIGN_L, len(kin))
        kin_g = kin[:n, 3:6] @ R.T
        L_kin = float(np.square(bandpass41(real[ALIGN_L:ALIGN_L + n] - kin_g)).sum(1).mean())
        hi, fs = dyn_replay_cmd(act, jkp)
        dyn_g = aa_down(hi, fs, len(fr))[:, 3:6] @ R.T
        lag, L_dyn = best_lag(real, dyn_g)
        m = min(n, len(dyn_g))
        gap1 = float(np.square(bandpass41(dyn_g[:m] - kin_g[:m])).sum(1).mean())
        expl = 1 - L_dyn / L_kin
        rows.append((L_kin, L_dyn, lag, gap1, expl))
        print(f"{p.stem:<26}{L_kin:>9.5f}{L_dyn:>9.5f}{lag:>8}{gap1:>9.5f}{expl*100:>+7.1f}%")
    A = np.array([(r[0], r[1], r[2], r[3], r[4]) for r in rows])
    print(f"\n{'中位':<26}{np.median(A[:,0]):>9.5f}{np.median(A[:,1]):>9.5f}"
          f"{np.median(A[:,2]):>8.1f}{np.median(A[:,3]):>9.5f}{np.median(A[:,4])*100:>+7.1f}%")
    print(f"\n解读:")
    print(f"  L_kin  = Step 1 一直在用的 gap（真机 vs 运动学理想）")
    print(f"  L_dyn  = **Gap2**, 未建模部分 ⇒ Step 2 要压的量")
    print(f"  解释率 = 1 − L_dyn/L_kin ⇒ 当前伺服模型解释掉了多少 gap")
    print(f"  dyn 滞后中位 {np.median(A[:,2]):.1f} 帧 —— 标定正确时应趋于 0"
          f"（当前 kp 是**手调**的，第⑤条要求换成自动标定）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
