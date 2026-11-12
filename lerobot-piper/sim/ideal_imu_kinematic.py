#!/usr/bin/env python3
"""P0.2 — 指令 → ideal IMU 运动学管线。

理想 = ACT 指令轨迹的运动学实现(平滑插值 → FK → 虚拟IMU), 不含动力学/控制器:
  q_cmd(t) : 20Hz 离散指令 → 三次样条(C², 精确二阶导) → 连续轨迹
  主路径(输出): MuJoCo 原生传感器管线 — 规定 (q,q̇,q̈) →
    mj_fwdPosition/Velocity → qacc:=q̈ → mj_rnePostConstraint → mj_sensor* → sensordata
    逐样本精确(RNE 用样条精确 q̈), 与旧动力学回放同一套 IMU 约定。
  交叉验证: 解析路径 R_siteᵀ(Jr q̇) 与 R_siteᵀ(d/dt[Jp q̇]−g) —
    gyro 应一致到机器精度; acc 差应集中在样条节点 jerk 跳变处(中心差分抹角, O(dt·Δjerk)),
    且随 --hi-fps 提高而收敛 ⇒ 证明两路实现/重力/坐标系均对。
单位: m/s², rad/s — 与真机 TSND151 落盘量纲一致(tsnd151_reader.py 已转 SI)。

用法:
  python sim/ideal_imu_kinematic.py --repo enolab-guo/pickplace2_train --episodes 0 --plot
输出: outputs/sim/ideal_imu/<repo名>/ep{i:03d}.npz [+ .png]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

SIM_DIR = Path(__file__).resolve().parent
MODEL_PATH = SIM_DIR / "piper_model.xml"
LR_CACHE = Path.home() / ".cache/huggingface/lerobot"
GRAV = None  # 从 model 读


def load_episode(repo: str, ep: int):
    root = LR_CACHE / repo
    info = json.loads((root / "meta/info.json").read_text())
    fps = float(info["fps"])
    import pandas as pd
    df = pd.read_parquet(root / f"data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet")
    out = {
        "fps": fps,
        "action": np.stack(list(df["action"].to_numpy())).astype(np.float64),  # (T,7) rad
    }
    if "observation.imu_real" in df.columns:
        out["imu_real"] = np.stack(list(df["observation.imu_real"].to_numpy())).astype(np.float64)
    if "observation.state" in df.columns:
        out["state_mdeg"] = np.stack(list(df["observation.state"].to_numpy())).astype(np.float64)
    return out


def build_spline(t20: np.ndarray, cmd: np.ndarray, kind: str):
    from scipy.interpolate import CubicSpline, PchipInterpolator
    if kind == "cubic":
        sp = CubicSpline(t20, cmd, axis=0)
    elif kind == "pchip":
        sp = PchipInterpolator(t20, cmd, axis=0)
    else:
        raise ValueError(kind)
    return sp


def full_qpos(q7: np.ndarray) -> np.ndarray:
    """7维指令 → 8维 qpos (j1..j6, joint7=+grip, joint8=−grip)。"""
    out = np.empty(8)
    out[:6] = q7[:6]
    g = float(np.clip(q7[6], 0.0, 0.035))
    out[6], out[7] = g, -g
    return out


def full_qvel(qd7: np.ndarray) -> np.ndarray:
    out = np.empty(8)
    out[:6] = qd7[:6]
    out[6], out[7] = qd7[6], -qd7[6]
    return out


def ideal_imu_traj(model, data, sp, t_hi: np.ndarray, sid: int):
    """一遍扫过: 主输出=MuJoCo 原生传感器(精确), 同时存解析量做交叉验证。"""
    import mujoco
    n = len(t_hi)
    q_hi = sp(t_hi)          # (n,7)
    qd_hi = sp(t_hi, 1)
    qdd_hi = sp(t_hi, 2)
    imu = np.zeros((n, 6))                      # 主输出 [acc3, gyro3] site 系
    v_w = np.zeros((n, 3))                      # 解析(交叉验证用)
    R_all = np.zeros((n, 3, 3))
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for i in range(n):
        data.qpos[:] = full_qpos(q_hi[i])
        data.qvel[:] = full_qvel(qd_hi[i])
        mujoco.mj_fwdPosition(model, data)
        mujoco.mj_fwdVelocity(model, data)
        data.qacc[:] = full_qvel(qdd_hi[i])       # 规定加速度(样条精确二阶导)
        mujoco.mj_rnePostConstraint(model, data)  # cacc(含重力基准)
        mujoco.mj_sensorPos(model, data)
        mujoco.mj_sensorVel(model, data)
        mujoco.mj_sensorAcc(model, data)
        imu[i] = data.sensordata[:6]
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        v_w[i] = jacp @ full_qvel(qd_hi[i])
        R_all[i] = data.site_xmat[sid].reshape(3, 3)
    return q_hi, imu, v_w, R_all


def analytic_crosscheck(model, imu, v_w, R_all, t_hi):
    """解析路径复算 acc/gyro, 返回与主输出的最大差(gyro 应≈0, acc 集中在节点拐角)。"""
    dt = t_hi[1] - t_hi[0]
    a_w = np.gradient(v_w, dt, axis=0)
    f_w = a_w - model.opt.gravity               # 比力 = a − g (静止 → +9.81 ẑ)
    acc_site = np.einsum("nij,nj->ni", R_all.transpose(0, 2, 1), f_w)
    e_acc = np.abs(acc_site[2:-2] - imu[2:-2, :3]).max()
    return float(e_acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="如 enolab-guo/pickplace2_train")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--interp", choices=["cubic", "pchip"], default="cubic")
    ap.add_argument("--hi-fps", type=float, default=200.0)
    ap.add_argument("--source", choices=["action", "state"], default="action",
                    help="理想=指令(action, 默认); state 仅作 sanity 备用")
    ap.add_argument("--smooth-cmd", type=float, default=None, metavar="FC_HZ",
                    help="意图版理想: 渲染前对 20Hz 指令做零相位低通(截止FC_HZ), "
                         "剔除遥操作量化台阶等指令伪影; 默认关(=字面版理想)")
    ap.add_argument("--selftest", type=int, default=64, help="互验采样点数(0=跳过)")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import mujoco
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site")
    assert sid >= 0

    out_root = Path(args.out) if args.out else (
        SIM_DIR.parent / "outputs/sim/ideal_imu" / args.repo.split("/")[-1])
    out_root.mkdir(parents=True, exist_ok=True)

    for ep in args.episodes:
        d = load_episode(args.repo, ep)
        fps = d["fps"]
        if args.source == "action":
            cmd = d["action"]
        else:
            cmd = d["state_mdeg"] * (np.pi / 180 / 1000)  # mdeg→rad(夹爪列口径另论)
        T = len(cmd)
        t20 = np.arange(T) / fps
        if args.smooth_cmd:
            from scipy.signal import butter, filtfilt
            cmd = filtfilt(*butter(2, args.smooth_cmd / (fps / 2)), cmd, axis=0)
        sp = build_spline(t20, cmd, args.interp)

        stride = int(round(args.hi_fps / fps))
        hi_fps = stride * fps                     # 保证 20Hz 采样点精确落在 hi 网格
        t_hi = np.arange(int(round(t20[-1] * hi_fps)) + 1) / hi_fps

        q_hi, imu_hi, v_w, R_all = ideal_imu_traj(model, data, sp, t_hi, sid)
        acc_hi, gyr_hi = imu_hi[:, :3], imu_hi[:, 3:]
        # 20Hz 对比序列: 抗混叠低通(零相位)后再抽样 — 点采样会把 hi 率尖峰折叠进
        # 0-10Hz 虚增"振动"(ep0 实测虚增 1.9-4.5×), 点采样版另存 *_point 供检查。
        from scipy.signal import butter, filtfilt
        ba = butter(4, 0.45 * fps / (hi_fps / 2))
        imu_aa = filtfilt(*ba, imu_hi, axis=0)
        acc20, gyr20 = imu_aa[::stride][:T, :3], imu_aa[::stride][:T, 3:]
        acc20_pt, gyr20_pt = acc_hi[::stride][:T], gyr_hi[::stride][:T]

        report = {"repo": args.repo, "ep": ep, "fps": fps, "hi_fps": hi_fps,
                  "interp": args.interp, "source": args.source, "T": T,
                  "smooth_cmd": args.smooth_cmd}
        if args.selftest:
            e_acc = analytic_crosscheck(model, imu_hi, v_w, R_all, t_hi)
            report["crosscheck_acc_err"] = e_acc
            print(f"[ep{ep}] 交叉验证(解析差分 vs 主输出): Δacc≤{e_acc:.2e} m/s² "
                  f"(应随 --hi-fps 提高收敛)")
        print(f"[ep{ep}] T={T}@{fps:g}fps  t0比力模长≈{np.linalg.norm(acc20[0]):.3f} "
              f"(若t0静止应≈9.81)  |q̈|max={np.abs(sp(t_hi,2)).max():.1f} rad/s²")

        save = {"t20": t20, "cmd": cmd, "acc20": acc20, "gyro20": gyr20,
                "acc20_point": acc20_pt, "gyro20_point": gyr20_pt,
                "t_hi": t_hi, "q_hi": q_hi, "acc_hi": acc_hi, "gyro_hi": gyr_hi,
                "meta": json.dumps(report)}
        if "imu_real" in d:
            save["imu_real"] = d["imu_real"]
        np.savez_compressed(out_root / f"ep{ep:03d}.npz", **save)
        print(f"[ep{ep}] → {out_root / f'ep{ep:03d}.npz'}")

        if args.plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            names = ["ax", "ay", "az", "gx", "gy", "gz"]
            fig, axes = plt.subplots(6, 1, figsize=(13, 12), sharex=True)
            ideal = np.hstack([acc20, gyr20])
            for i, ax in enumerate(axes):
                if "imu_real" in d:
                    ax.plot(t20, d["imu_real"][:T, i], color="#1f77b4", lw=0.8,
                            alpha=0.85, label="real (TSND)")
                ax.plot(t20, ideal[:, i], color="#d62728", lw=0.8,
                        alpha=0.85, label="ideal (kinematic)")
                ax.set_ylabel(names[i])
                ax.grid(alpha=0.3)
                if i == 0:
                    ax.legend(loc="upper right", fontsize=9)
                    ax.set_title(f"{args.repo} ep{ep}: ideal (cmd kinematics) vs real "
                                 f"[interp={args.interp}]")
            axes[-1].set_xlabel("time (s)")
            fig.tight_layout()
            png = out_root / f"ep{ep:03d}.png"
            fig.savefig(png, dpi=120)
            plt.close(fig)
            print(f"[ep{ep}] 图 → {png}")


if __name__ == "__main__":
    main()
