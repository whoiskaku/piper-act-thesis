#!/usr/bin/env python3
"""#76 — 真机 单chunk(end-to-end) 动作平滑验证 (老师 steer; 目标=动作非IMU)。

每条 episode 的【第一个 chunk】(frame 0-100) = 一个干净的开环单 chunk:
ACT 起点预测一次→执行 100 步, 无 chunk 边界、无闭环重规划、residual 模式无 blend。
→ 这是 #75 sim per-chunk 结果的真机对照, 隔离闭环边界问题(#69-real 整流失败的元凶)。

度量(active joints j1/j2/j3/grip, 2-4.5Hz 透传/任务相关带):
  · 命令动作 hp (bandpass-RMS): residual 条件应 < 裸 ACT (残差按构造平滑命令)。
  · 执行关节 hp (observation.state): 真机控制器是否把命令平滑透传到关节运动(真问题)。
对照: 同时算【整段 episode】(含边界), 应见 residual 反而更差(=#69-real 边界淹没, 印证机制)。

注: 用现有 #69-real 录制(eval_s5tok_*)时是 1q/Mq/RLT ckpt(非 Hattn cut2.0)、起始位姿未严格配对
   → 仅粗预览。正式验证须新录 REFINE_MODE=off vs residual+Hattn-cut2.0、同起点、--datasets 指过来。

用法: /home/enolab-guo/miniconda3/envs/lerobot/bin/python sim/eval_realchunk_action.py
      [--datasets ACT=eval_s5tok_A refined=<新录的Hattn-cut2.0数据集> ...]
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

LR = Path.home() / ".cache/huggingface/lerobot/enolab-guo"
FS = 30.0
CHUNK = 100
AJ = [0, 1, 2]                     # 仅臂关节 j1/j2/j3 (夹爪 state 是原始编码器单位~68000/命令含开合阶跃, 排除)
MDEG_TO_RAD = np.pi / 180.0 / 1000.0
_BP = butter(4, [2.0 / 15.0, 4.5 / 15.0], "band")     # 2-4.5Hz 透传带
DEFAULT = ["ACT=eval_s5tok_A", "1q=eval_s5tok_1q", "Mq=eval_s5tok_Mq", "RLT=eval_s5tok_RLT"]


def load_eps(ds, skip=0):
    eps = []
    for pq in sorted(glob.glob(str(LR / ds / "data/chunk-*/episode_*.parquet")))[skip:]:
        try:
            df = pd.read_parquet(pq)
            act = np.stack(df["action"].to_numpy()).astype(np.float64)
            st = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
            imu = (np.stack(df["observation.imu_real"].to_numpy()).astype(np.float64)
                   if "observation.imu_real" in df else None)
        except (KeyError, ValueError):
            continue
        st = st.copy(); st[:, :6] *= MDEG_TO_RAD
        eps.append((act, st, imu))
    return eps


def bp_rms(x, sl):
    """2-4.5Hz bandpass 后 active-joint RMS。x (T,7), sl=帧切片。"""
    seg = x[sl]
    if len(seg) < 30:
        return np.nan
    y = filtfilt(*_BP, seg, axis=0)
    return float(np.sqrt((y[:, AJ] ** 2).mean()))


def bp_rms_mag(sig, sl):
    """1-D 信号(如 IMU accel/gyro 幅值)的 2-4.5Hz bandpass RMS。"""
    seg = sig[sl]
    if len(seg) < 30:
        return np.nan
    return float(np.sqrt((filtfilt(*_BP, seg) ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DEFAULT, help="label=dataset 对; 第一个作 ACT 基线")
    ap.add_argument("--skip", type=int, default=0, help="跳过每个数据集前 N 条(warmup/冷启动); 建议 --skip 1")
    cli = ap.parse_args()
    pairs = [d.split("=", 1) for d in cli.datasets]

    def metrics(eps, sl_fn):
        """对一组 ep, 在 sl_fn(T) 切片上算 cmd/joint/imu-accel/imu-gyro 2-4.5Hz RMS 均值。"""
        cmd, jnt, iacc, igyr = [], [], [], []
        for act, st, imu in eps:
            sl = sl_fn(len(act))
            cmd.append(bp_rms(act, sl)); jnt.append(bp_rms(st, sl))
            if imu is not None:
                am = np.linalg.norm(imu[:, :3] - imu[:, :3].mean(0), axis=1)
                gm = np.linalg.norm(imu[:, 3:], axis=1)
                iacc.append(bp_rms_mag(am, sl)); igyr.append(bp_rms_mag(gm, sl))
        return (np.nanmean(cmd), np.nanmean(jnt),
                np.nanmean(iacc) if iacc else np.nan, np.nanmean(igyr) if igyr else np.nan)

    print(f"\n=== FIRST-CHUNK (frame 0-100 = 单chunk, 无边界/无blend/无闭环重规划) — 主验证 ===")
    print(f"{'cond':>8} {'n':>3} | {'cmd':>9} {'':>5} {'joint':>9} {'':>5} {'IMUacc':>9} {'':>5} {'IMUgyr':>9}")
    base = {}
    for label, ds in pairs:
        eps = load_eps(ds, cli.skip)
        if not eps:
            print(f"{label:>8}  (无数据 {ds})"); continue
        c, j, ia, ig = metrics(eps, lambda T: slice(0, CHUNK))
        if not base:
            base = {"c": c, "j": j, "ia": ia, "ig": ig}
        def f(v, b): return f"{v*1e3:>8.3f}m" if v == v else "  n/a"
        def rt(v, b): return f"({v/b:.2f})" if (b == b and v == v and b) else "     "
        print(f"{label:>8} {len(eps):>3} | {f(c,0)} {rt(c,base['c'])} {f(j,0)} {rt(j,base['j'])} "
              f"{f(ia,0)} {rt(ia,base['ia'])} {f(ig,0)} {rt(ig,base['ig'])}")
    print(f"\n=== FULL-EPISODE (含边界, 对照) — residual 应 >1(#69-real 边界淹没机制) ===")
    base2 = {}
    for label, ds in pairs:
        eps = load_eps(ds, cli.skip)
        if not eps:
            continue
        c, j, ia, ig = metrics(eps, lambda T: slice(0, T))
        if not base2:
            base2 = {"c": c, "j": j, "ia": ia, "ig": ig}
        def f(v): return f"{v*1e3:>8.3f}m" if v == v else "  n/a"
        def rt(v, b): return f"({v/b:.2f})" if (b == b and v == v and b) else "     "
        print(f"{label:>8} {len(eps):>3} | {f(c)} {rt(c,base2['c'])} {f(j)} {rt(j,base2['j'])} "
              f"{f(ia)} {rt(ia,base2['ia'])} {f(ig)} {rt(ig,base2['ig'])}")
    print("-" * 70)
    print("判读: 比值<1=比第一条(裸ACT)更平滑。matched 正式录制(同起点/物体 off vs residual+Hattn-cut2.0):")
    print("  first-chunk: cmd 应<1(像sim ~0.7); joint 看控制器透传; IMU 看是否被地板盖住(#74)。")
    print("  full-episode: residual cmd 应>1(闭环边界淹没=#69-real), 印证'需边界处理'。")


if __name__ == "__main__":
    main()
