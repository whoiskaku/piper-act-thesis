#!/usr/bin/env python3
"""③a 复放执行器（2026-07-22）— 在真机上开环复放固定指令, 可叠加 δa 探针, 并记录 IMU。

与 control_robot 的 `replay` 模式的区别（为什么另写）:
  · replay **不记录 IMU**, 而探针实验的全部信息都在 IMU 里
  · 需要可选地叠加 δa 探针（同一段指令的 A/B 对照）
  · 需要记录每帧的墙钟时间戳, 以便三次复放严格对齐

安全（与部署 wrapper 同级）:
  · 出发前断言 δa ≤ ε；同帧仅一关节
  · 每帧断言指令在关节限位内（读 MuJoCo 模型的 jnt_range）
  · 任一帧 IMU 陈旧/异常 → 立即停机并落盘已采数据
  · 夹爪指令原样透传, 永不施加 δa

用法（真机）:
  python sim/replay_probe.py --robot piper --arm A1     # 裸复放
  python sim/replay_probe.py --robot piper --arm B      # 加探针
  python sim/replay_probe.py --robot piper --arm A2     # 再裸复放(噪声底)
用法（无机时自检）:
  python sim/replay_probe.py --dry-run                  # 只跑安全检查与时序模拟
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "outputs/probe"
FPS, DT = 20.0, 0.05
IMU_STALE_S = 0.05


PB2 = ROOT / "outputs/probe_b2"


def load_seq(seqfile="probe_seq.npz", b2=False):
    root = PB2 if b2 else PROBE
    z = np.load(root / seqfile, allow_pickle=True)
    return z["cmd7"].astype(np.float32), z["da"].astype(np.float32), z


def safety_checks(cmd7, da, model_path=None):
    """出发前的全部断言 —— 任何一条不过就不上机。"""
    from lerobot.common.policies.act.b_runtime import EPS_B
    msgs = []
    assert np.isfinite(cmd7).all() and np.isfinite(da).all(), "指令/探针含 NaN"
    assert np.abs(da).max() <= EPS_B + 1e-12, \
        f"探针 {np.abs(da).max()*1e3:.1f}mrad 超过 ε={EPS_B*1e3:.0f}mrad"
    msgs.append(f"✓ δa 峰值 {np.abs(da).max()*1e3:.1f} mrad ≤ ε {EPS_B*1e3:.0f} mrad")
    assert (np.abs(da) > 0).sum(1).max() <= 1, "同帧多关节同时发"
    msgs.append("✓ 同帧仅一关节")
    assert da.shape[1] == 6, "探针不得触及夹爪"
    msgs.append("✓ 夹爪不受探针影响")

    # 关节限位（从 MuJoCo 模型读）
    try:
        import mujoco
        from ideal_imu_kinematic import MODEL_PATH
        m = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        mlo, mhi = m.jnt_range[:6, 0], m.jnt_range[:6, 1]
        # 与 make_probe_b2.robot_limits 统一: MuJoCo jnt_range 是**保守软限**
        # (j2 下界写 0° 但真机实测到 −0.1°、指令到 −1.9° 且正常执行 20 集)。
        # 安全限位 = 真机实测编码器范围 ±2°  ∩  模型限位 ±2°
        from make_probe_b2 import robot_limits
        rlo, rhi = robot_limits()
        lo = np.maximum(mlo - 0.035, rlo)
        hi = np.minimum(mhi + 0.035, rhi)
        base, tot = cmd7[:, :6], cmd7[:, :6] + da
        exc = lambda x: np.maximum(np.maximum(lo - x, x - hi), 0.0)
        e_b, e_t = exc(base), exc(tot)
        worse = float((e_t - e_b).max())
        assert worse <= 1e-9, f"探针使越界加深 {np.degrees(worse):.3f}°"
        if e_b.max() > 0:
            j = int(np.unravel_index(e_b.argmax(), e_b.shape)[1])
            msgs.append(f"⚠ 基准指令本身越界 j{j+1} 最深 {np.degrees(e_b.max()):.2f}° "
                        f"({int((e_b > 0).any(1).sum())} 帧) — 真机已执行过, 非探针所致")
        msgs.append(f"✓ 探针不加深越界 (最坏 {np.degrees(worse):.4f}°)")
        mar = float(np.minimum(tot - lo, hi - tot).min())
        msgs.append(f"✓ 加探针后离限位最小余量 {np.degrees(mar):.2f}°")
    except ImportError:
        msgs.append("⚠ 未装 mujoco, 跳过限位检查")

    # 指令连续性（探针不得制造跳变）
    step = np.abs(np.diff(cmd7[:, :6] + da, axis=0)).max()
    base = np.abs(np.diff(cmd7[:, :6], axis=0)).max()
    assert step < base * 1.5 + 0.006, f"叠加探针后单帧跳变 {step*1e3:.1f}mrad 过大"
    msgs.append(f"✓ 单帧最大跳变 {step*1e3:.1f} mrad (裸 {base*1e3:.1f})")
    return msgs


def run(arm, dry_run=False, seqfile="probe_seq.npz", tag="", b2=False, outdir=None):
    cmd7, da, z = load_seq(seqfile, b2)
    import re
    is_bare = bool(re.fullmatch(r"A\d+", arm))      # A1/A2/A3/A4… 全是裸复放
    use_probe = arm.startswith("B") or (b2 and not is_bare)
    total = cmd7[:, :6] + (da if use_probe else 0.0)
    print(f"=== 复放 {arm} ({'加探针' if use_probe else '裸'}) · {len(cmd7)} 帧 "
          f"= {len(cmd7)/FPS:.0f}s ===")
    for m in safety_checks(cmd7, da if use_probe else np.zeros_like(da)):
        print("  " + m)

    if dry_run:
        print("\n[dry-run] 不连机器人。时序模拟:")
        t0 = time.perf_counter()
        for i in range(min(100, len(cmd7))):
            time.sleep(0)                       # 仅测循环开销
        oh = (time.perf_counter() - t0) / 100 * 1e3
        print(f"  循环开销 {oh:.3f} ms/帧 (预算 {DT*1e3:.0f} ms)  ✓")
        print(f"  预计时长 {len(cmd7)/FPS:.0f}s")
        print("  ✓ 全部安全检查通过 —— 可上机")
        return 0

    # ---- 真机 ----
    import torch
    from lerobot.common.robot_devices.robots.utils import make_robot
    # 探针只需 IMU+关节: inference_time=True(推理模式,无手柄teleop) · enable_imu=True(默认关!)
    # · cameras={}(不读图, 省循环开销且避开已知的 ffmpeg SIGABRT)
    robot = make_robot("piper", inference_time=True, enable_imu=True, cameras={})
    if not robot.is_connected:
        robot.connect()
    # IMU 连接自检: 动之前先确认 IMU 活着, 否则白跑 59s
    obs0 = robot.capture_observation()
    if "observation.imu_real" not in obs0:
        raise RuntimeError("IMU 未启用! 检查 enable_imu / TSND 设备 /dev/ttyACM0")
    imu0 = obs0["observation.imu_real"].numpy()
    assert np.isfinite(imu0).all(), f"IMU 首帧异常: {imu0}"
    print(f"  IMU 自检 ✓  acc={np.round(imu0[:3],2)}  gyro={np.round(imu0[3:],3)}")

    # ---- 预移动: 缓慢移到轨迹首帧位姿, 稳定后才开始记录 ----
    # 2026-07-28: 机械臂停在 safe_disable_position(j5=0.52rad=29.8°), 而轨迹首帧要求
    # j5=0.16° —— 直接开跑会在 0.6s 内扫过 29°, j5 峰速 102°/s = 既有包络的 155%,
    # 且六臂的起始位姿取决于机械臂停在哪里(门禁③ 就只能验"停机位姿可复现")。
    # 故先用 PREMOVE_S 秒线性插值缓慢过去, 再稳定 PRESETTLE_S 秒。
    # 判据(2026-07-28 A1 二跑修正): 不能要求"到达 cmd7[0]" —— 该指令的 j2 = −1.23°
    # **低于 j2 的可达下界 −0.07°**, 真机在稳态上到不了(静态偏差 +1.17°)。
    # 改判两条: (a) 已**稳定**(末段速度 <0.5°/s);
    #          (b) 偏差 ≤ 该关节"指令超出可达范围的量" + 1° 的裕度。
    # 六臂之间的一致性由门禁③(state_init 离散 <0.5°)把关, 那才是真正要保证的。
    PREMOVE_S, PRESETTLE_S = 3.0, 1.5
    PRE_VEL_TOL = np.radians(0.5)                  # °/s
    q_now = (obs0["observation.state"].numpy()[:6] * np.pi / 180 / 1000).astype(np.float32)
    q_tgt = cmd7[0, :6].astype(np.float32)
    n_pre = int(PREMOVE_S * FPS)
    print(f"  预移动 {PREMOVE_S:.0f}s: 最大位移 "
          f"{np.degrees(np.abs(q_tgt - q_now)).max():.2f}° (j{int(np.abs(q_tgt-q_now).argmax())+1}) "
          f"→ 峰速约 {np.degrees(np.abs(q_tgt - q_now)).max()/PREMOVE_S:.1f}°/s")
    for i in range(n_pre + 1):
        w = (1.0 - np.cos(np.pi * i / n_pre)) / 2.0        # 余弦缓入缓出, 零起始/终止速度
        act = np.concatenate([q_now + w * (q_tgt - q_now), cmd7[0, 6:7]]).astype(np.float32)
        robot.send_action(torch.from_numpy(act))
        time.sleep(DT)
    settle = []
    for _ in range(int(PRESETTLE_S * FPS)):               # 稳定, 让余振衰完
        robot.send_action(torch.from_numpy(
            np.concatenate([q_tgt, cmd7[0, 6:7]]).astype(np.float32)))
        settle.append(robot.capture_observation()["observation.state"].numpy()[:6]
                      * np.pi / 180 / 1000)
        time.sleep(DT)
    obs0 = robot.capture_observation()                    # 起始状态改在**预移动后**取
    q_pre = obs0["observation.state"].numpy()[:6] * np.pi / 180 / 1000
    # (a) 稳定: 末段 0.5s 的关节速度
    tail = np.asarray(settle[-int(0.5 * FPS):], float)
    vmax = float(np.abs(np.diff(tail, axis=0) * FPS).max()) if len(tail) > 1 else 0.0
    # (b) 偏差 ≤ 指令超出可达范围的量 + 1°
    from make_probe_b2 import reach_limits
    klo, khi = reach_limits()
    unreach = np.maximum(np.maximum(klo - q_tgt, q_tgt - khi), 0.0)
    tol = unreach + np.radians(1.0)
    dev = np.abs(q_pre - q_tgt)
    j_bad = int(np.argmax(dev - tol))
    print(f"  预移动后: 末段峰速 {np.degrees(vmax):.3f}°/s (阈 {np.degrees(PRE_VEL_TOL):.1f})"
          f"  {'✓稳定' if vmax <= PRE_VEL_TOL else '✗未稳定'}")
    print(f"    逐关节 |偏差| {np.round(np.degrees(dev), 3)}°  容差 {np.round(np.degrees(tol), 3)}°"
          f"  (含指令超出可达范围 {np.round(np.degrees(unreach), 3)}°)")
    assert vmax <= PRE_VEL_TOL, f"预移动后仍在动 ({np.degrees(vmax):.3f}°/s) —— 停机排查"
    assert (dev <= tol).all(), (
        f"预移动未到位: j{j_bad+1} 偏差 {np.degrees(dev[j_bad]):.3f}° > 容差 "
        f"{np.degrees(tol[j_bad]):.3f}° —— 停机排查, 六臂起始位姿必须一致")
    print(f"  ✓ 预移动到位 (最大超容差 {np.degrees((dev - tol).max()):+.3f}°)")

    # t_send: **JointCtrl 真正写总线的时刻**(ZOH 起点), monotonic 与 TSND host_t 同钟。
    # 同时存 t_send_span = post−pre, 用于量化"实际发送时刻"本身的不确定度。
    import hashlib
    cmd_hash = hashlib.sha256(np.ascontiguousarray(cmd7, np.float32).tobytes()).hexdigest()[:16]
    rec = {"arm": arm, "t": [], "t_send": [], "t_send_span": [], "cmd": [],
           "imu": [], "state": [], "imu100": [], "clock": "time.monotonic",
           "cmd_hash": cmd_hash, "seqfile": seqfile, "use_probe": bool(use_probe),
           "premove": {"s": PREMOVE_S, "settle_s": PRESETTLE_S,
                       "q_parked": q_now.tolist(), "q_start": q_pre.tolist(),
                       "dev_deg": np.degrees(dev).tolist(),
                       "tol_deg": np.degrees(tol).tolist(),
                       "settle_vmax_dps": float(np.degrees(vmax))},
           "state_init": obs0["observation.state"].numpy().tolist()}
    print(f"  基准指令 hash {cmd_hash}  起始关节 "
          f"{np.round(np.array(rec['state_init'][:6]) * np.pi / 180 / 1000, 4)}")
    rdr = getattr(robot, "imu_reader", None)
    if rdr is not None and hasattr(rdr, "start_recording"):
        rdr.start_recording()                      # 100Hz 独立队列(不碰控制侧 buffer)
        print("  100Hz 记录已开启")
    stopped = None
    try:
        for i in range(len(cmd7)):
            t_start = time.perf_counter()
            act = np.concatenate([total[i], cmd7[i, 6:7]]).astype(np.float32)
            robot.send_action(torch.from_numpy(act))
            # 从总线层取 JointCtrl 的实际写入时刻(见 motors/piper.py); 取不到则退回本帧起点
            span = getattr(robot.arm, "last_jointctrl_t", None)
            t_send = 0.5 * (span[0] + span[1]) if span else time.monotonic()
            obs = robot.capture_observation()
            imu = obs.get("observation.imu_real")
            if imu is None or not np.isfinite(imu.numpy()).all():
                stopped = f"帧 {i}: IMU 异常"
                break
            rec["t"].append(time.monotonic())
            rec["t_send"].append(t_send)
            rec["t_send_span"].append(float(span[1] - span[0]) if span else float("nan"))
            rec["cmd"].append(act.tolist())
            rec["imu"].append(imu.numpy().tolist())
            rec["state"].append(obs["observation.state"].numpy().tolist())
            if rdr is not None and hasattr(rdr, "drain_recording"):
                for ht, ms, ac, gy in rdr.drain_recording():   # 逐帧排空, 防环形覆盖
                    rec["imu100"].append([float(ht), int(ms),
                                          *map(float, ac), *map(float, gy)])
            dt = time.perf_counter() - t_start
            if dt < DT:
                time.sleep(DT - dt)
    except KeyboardInterrupt:
        stopped = "用户中断"
    finally:
        if rdr is not None and hasattr(rdr, "stop_recording"):
            for ht, ms, ac, gy in rdr.drain_recording():
                rec["imu100"].append([float(ht), int(ms), *map(float, ac), *map(float, gy)])
            rdr.stop_recording()
        p = (outdir or PROBE) / f"replay_{tag}{arm}.json"   # tag 含 ep 与轮次
        rec["stopped"] = stopped
        rec["seqfile"] = seqfile
        p.write_text(json.dumps(rec))
        print(f"\n采集 {len(rec['imu'])} 帧(20Hz) + {len(rec['imu100'])} 样本(100Hz) → {p}" + (f"  ⚠ 提前终止: {stopped}" if stopped else "  ✓"))
    return 0 if stopped is None else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A1",
                    help="A1/A2 或 B′ 配置名如 j3_8mrad_p")
    ap.add_argument("--ep", type=int, default=None,
                    help="pp2 集号(0/1/2); 用 probe_seq_pp2_ep{ep}.npz, 输出 replay_ep{ep}_*.json")
    ap.add_argument("--b2", action="store_true", help="用 B′(6-8Hz 多正弦)序列")
    ap.add_argument("--amp", default="4mrad", help="A 臂借哪个幅值的 cmd7(内容无关)")
    ap.add_argument("--round", type=int, default=1, choices=[1, 2],
                    help="2 = 反向平衡块(输出文件加 _r2_ 前缀, 不覆盖首轮)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.b2:
        import re as _re
        cfg = ("j2_" + a.amp + "_p" if _re.fullmatch(r"A\d+", a.arm)
               else a.arm)                          # A 臂借任一序列的 cmd7(只用 cmd7)
        seqfile = f"seq_ep{a.ep}_{cfg}.npz"
        # 轮次后缀: 第2轮的 B 臂配置名与第1轮相同, 不加后缀会**覆盖首轮数据**
        tag = f"ep{a.ep}_" if a.round == 1 else f"ep{a.ep}_r{a.round}_"
        raise SystemExit(run(a.arm, a.dry_run, seqfile=seqfile,
                             tag=tag, b2=True, outdir=PB2))
    seqfile = f"probe_seq_pp2_ep{a.ep}.npz" if a.ep is not None else "probe_seq.npz"
    tag = f"ep{a.ep}_" if a.ep is not None else ""
    raise SystemExit(run(a.arm, a.dry_run, seqfile=seqfile, tag=tag))
