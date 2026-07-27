#!/usr/bin/env python3
"""P0 分析层 — 动力学回放: 指令喂 ctrl, 伺服模型真积分, 收 dyn_ideal IMU。

与 kin_ideal(运动学字面理想, ideal_imu_kinematic.py) 的关系:
  kin_ideal: 手臂被"规定"贴住指令 — 零标定依赖, 是"理想"的定义;
  dyn_ideal: 带 kp/阻尼的"无毛病伺服机器人"追指令 — 依赖标定, 只作归因层:
             Gap1 = kin→dyn = 控制器解释的部分(滞后+低通);
             Gap2 = dyn→real = 未建模的墙本体(柔性/共振/…)。
注意: 需 piper_model.xml 已关自碰撞(2026-07-08 修复, 否则 j1 被接触锁死)。
legacy kp 对 j1 过软(damping500/kp500→τ=1s), 用 --j-kp 覆盖(默认 1:5000 ≈ 真机2帧滞后)。

用法:
  python sim/dyn_replay_imu.py --repo enolab-guo/pickplace2_train --episodes 0
输出: outputs/sim/ideal_imu/<repo>/ep{i:03d}_dyn20.npy + 逐关节跟踪表(stdout)
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np

from ideal_imu_kinematic import MODEL_PATH, load_episode

MDEG = np.pi / 180 / 1000


def parse_jkp(s: str) -> dict[int, float]:
    """"1:5000,3:400" → {0:5000.0, 2:400.0} (关节号1基→actuator索引0基)。"""
    out = {}
    if s:
        for kv in s.split(","):
            j, kp = kv.split(":")
            out[int(j) - 1] = float(kp)
    return out


def dyn_replay(repo: str, ep: int, j_kp: dict[int, float], settle_s: float = 0.5):
    import mujoco
    d = load_episode(repo, ep)
    act, fps = d["action"], d["fps"]
    m = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    for j, kp in j_kp.items():
        m.actuator_gainprm[j, 0] = kp
        m.actuator_biasprm[j, 1] = -kp
    dt = m.opt.timestep
    nsub = int(round(1 / fps / dt))
    data = mujoco.MjData(m)
    g0 = float(np.clip(act[0, 6], 0, 0.035))
    data.qpos[:6], data.qpos[6], data.qpos[7] = act[0, :6], g0, -g0
    mujoco.mj_forward(m, data)
    data.ctrl[:6], data.ctrl[6], data.ctrl[7] = act[0, :6], g0, -g0
    for _ in range(int(settle_s / dt)):     # 预热: 消初始瞬态
        mujoco.mj_step(m, data)
    T = len(act)
    imu_hi = np.zeros((T * nsub, 6))
    qpos20 = np.zeros((T, 6))
    for t in range(T):
        gg = float(np.clip(act[t, 6], 0, 0.035))
        data.ctrl[:6], data.ctrl[6], data.ctrl[7] = act[t, :6], gg, -gg
        for k in range(nsub):
            mujoco.mj_step(m, data)
            imu_hi[t * nsub + k] = data.sensordata[:6]
        qpos20[t] = data.qpos[:6]
    return d, imu_hi, 1 / dt, qpos20


def aa_downsample(x_hi, fs_hi, fps, T):
    from scipy.signal import butter, filtfilt
    ba = butter(4, 0.45 * fps / (fs_hi / 2))
    stride = int(round(fs_hi / fps))
    return filtfilt(*ba, x_hi, axis=0)[::stride][:T]


def lagcorr(a, b, maxlag=10):
    best = (0, -2.0)
    for L in range(maxlag + 1):
        aa, bb = (a[:-L], b[L:]) if L else (a, b)
        if aa.std() < 1e-9 or bb.std() < 1e-9:
            continue
        c = float(np.corrcoef(aa, bb)[0, 1])
        if c > best[1]:
            best = (L, c)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--j-kp", default="1:5000", help='kp 覆盖 "关节:kp,...", 空串=不覆盖')
    ap.add_argument("--settle", type=float, default=0.5)
    args = ap.parse_args()
    j_kp = parse_jkp(args.j_kp)

    out_root = Path(__file__).resolve().parents[1] / "outputs/sim/ideal_imu" / args.repo.split("/")[-1]
    out_root.mkdir(parents=True, exist_ok=True)
    for ep in args.episodes:
        d, imu_hi, fs_hi, qpos20 = dyn_replay(args.repo, ep, j_kp, args.settle)
        T = len(d["action"])
        dyn20 = aa_downsample(imu_hi, fs_hi, d["fps"], T)
        np.save(out_root / f"ep{ep:03d}_dyn20.npy", dyn20)
        print(f"[ep{ep}] kp覆盖={args.j_kp or '无'}  → ep{ep:03d}_dyn20.npy")
        print("  关节 | 指令幅度 | sim跟踪: 滞后/corr | real跟踪: 滞后/corr")
        has_real = "state_mdeg" in d
        for j in range(6):
            c = d["action"][:, j]
            Ls, cs = lagcorr(c, qpos20[:, j])
            line = f"   j{j+1}  |  {c.max()-c.min():5.3f}  |   {Ls}帧 / {cs:+.4f}"
            if has_real:
                Lr, cr = lagcorr(c, d["state_mdeg"][:, j] * MDEG)
                line += f"    |   {Lr}帧 / {cr:+.4f}"
            print(line)


if __name__ == "__main__":
    main()
