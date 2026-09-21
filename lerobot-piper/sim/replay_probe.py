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
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "outputs/probe"
FPS, DT = 20.0, 0.05
IMU_STALE_S = 0.05


PB2 = ROOT / "outputs/probe_b2"


def load_seq(seqfile="probe_seq.npz", b2=False):
    """读序列。`seqfile` 可以是**显式路径**，也可以是 PROBE/PB2 下的文件名。

    2026-08-24: 早先只接受「PROBE/PB2 下的文件名」，外部生成的序列
    （如 `outputs/step2_g/pilot_j4/*.npz`）无法喂进来。现在若 `seqfile` 存在
    （绝对或相对当前目录）就直接用，否则回落到旧的根目录拼接。
    """
    p = Path(seqfile)
    path = p if p.exists() else (PB2 if b2 else PROBE) / seqfile
    if not path.exists():
        raise SystemExit(f"序列文件不存在: {seqfile}（也不在 "
                         f"{(PB2 if b2 else PROBE).name}/ 下）")
    z = np.load(path, allow_pickle=True)
    missing = {"cmd7", "da"} - set(z.files)
    if missing:
        raise SystemExit(f"{path.name} 缺字段 {sorted(missing)}；实有 {list(z.files)}")
    return z["cmd7"].astype(np.float32), z["da"].astype(np.float32), z


EPS_B_FALLBACK = 0.010          # 与 lerobot/common/policies/act/b_runtime.py 一致


# ⚠ **两种探针形态，检查不同** —— 新增 ③c 用的多关节模式时**不删旧模式的断言**。
PROBE_KINDS = ("single_joint_pulse", "multi_joint_continuous")


def safety_checks(cmd7, da, model_path=None, probe_kind="single_joint_pulse"):
    """出发前的全部断言 —— 任何一条不过就不上机。

    → (msgs, degraded)。`degraded=True` 表示有依赖缺失、部分检查被跳过或用了
    回落常量 —— 此时的 dry-run **只能用于诊断接口, 不足以放行上机**。
    """
    degraded = []
    blockers = []           # ⚠ **硬阻断**：非空时禁止进入任何运动
    try:
        from lerobot.common.policies.act.b_runtime import EPS_B
    except Exception as e:                                        # noqa: BLE001
        EPS_B = EPS_B_FALLBACK
        degraded.append(f"EPS_B 取回落常量 {EPS_B_FALLBACK}（导入失败: "
                        f"{type(e).__name__}）—— 上机前须与 b_runtime.py 核对")
    msgs = []
    assert np.isfinite(cmd7).all() and np.isfinite(da).all(), "指令/探针含 NaN"
    assert np.abs(da).max() <= EPS_B + 1e-12, \
        f"探针 {np.abs(da).max()*1e3:.1f}mrad 超过 ε={EPS_B*1e3:.0f}mrad"
    msgs.append(f"✓ δa 峰值 {np.abs(da).max()*1e3:.1f} mrad ≤ ε {EPS_B*1e3:.0f} mrad")
    # ── 形态分支 ────────────────────────────────────────────────────
    if probe_kind not in PROBE_KINDS:
        raise ValueError(f"probe_kind 须为 {PROBE_KINDS}，得到 {probe_kind!r}")
    n_joint = int((np.abs(da) > 0).sum(1).max())
    if probe_kind == "single_joint_pulse":
        # **③a/b2 的旧模式，断言原样保留**
        assert n_joint <= 1, "同帧多关节同时发"
        msgs.append("✓ 同帧仅一关节（single_joint_pulse）")
    else:
        # **③c：多关节连续补正** —— 旧断言在此结构性不成立，故走独立分支，
        # 但**不放松其它任何检查**，并额外报告逐关节的绝对差分（见文末）。
        msgs.append(f"· 同帧最多 {n_joint} 个关节（multi_joint_continuous，"
                    f"**已显式选择该模式**）")
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
        exc = lambda x, a, b: np.maximum(np.maximum(a - x, x - b), 0.0)

        # ── 两条限位必须**分开报**(2026-08-26) ────────────────────────────
        # 合并报会得到「✓ 余量 −61.21°」这种挂着对勾的负余量: 真正的模型限位
        # 越界会被淹没。step2_g 激励套件 73/90 条**按设计**走出 pp2 实测包络
        # (j4 在 pp2 里从没动过), 那不是安全问题; 越模型限位才是。
        #   模型限位 ±2° = 硬安全界, 判 ✓/✗;
        #   pp2 实测包络 = 提示性, 只说明「臂没去过这里」。
        # ±2° 松量沿用原注: pp2 的 j2 指令实到 −1.9°, 低于模型写死的 0° 下界。
        mlo2, mhi2 = mlo - 0.035, mhi + 0.035
        e_m = exc(tot, mlo2, mhi2)
        assert e_m.max() <= 1e-9, (
            "**越模型限位**: j"
            f"{int(np.unravel_index(e_m.argmax(), e_m.shape)[1]) + 1} "
            f"最深 {np.degrees(e_m.max()):.3f}°")
        mar_m = float(np.minimum(tot - mlo2, mhi2 - tot).min())
        jm = int(np.unravel_index(np.minimum(tot - mlo2, mhi2 - tot).argmin(),
                                  tot.shape)[1])
        msgs.append(f"✓ 离**模型限位**最小余量 {np.degrees(mar_m):.2f}° (j{jm+1})")

        # 探针不得加深「合成限位」下的越界 —— 原门禁, 语义不变
        e_b, e_t = exc(base, lo, hi), exc(tot, lo, hi)
        worse = float((e_t - e_b).max())
        assert worse <= 1e-9, f"探针使越界加深 {np.degrees(worse):.3f}°"
        msgs.append(f"✓ 探针不加深越界 (最坏 {np.degrees(worse):.4f}°)")
        if e_b.max() > 0:
            j = int(np.unravel_index(e_b.argmax(), e_b.shape)[1])
            msgs.append(f"· 出 pp2 实测包络 j{j+1} 最远 {np.degrees(e_b.max()):.2f}° "
                        f"({int((e_b > 0).any(1).sum())} 帧) — 提示性, "
                        f"非安全判定; 模型限位见上行")
    except ImportError as e:                                      # noqa: BLE001
        msgs.append(f"⚠ 跳过限位检查（{type(e).__name__}: {e}）")
        degraded.append("关节限位检查被跳过")
        blockers.append("关节限位检查被跳过 ⇒ 无法确认位置安全")

    # 指令连续性（探针不得制造跳变）
    # ⚠ 这是**整集最大值之比**，只说明相对峰值；**不能证明每一处跳变都小**。
    step = np.abs(np.diff(cmd7[:, :6] + da, axis=0)).max()
    base = np.abs(np.diff(cmd7[:, :6], axis=0)).max()
    assert step < base * 1.5 + 0.006, f"叠加探针后单帧跳变 {step*1e3:.1f}mrad 过大"
    msgs.append(f"✓ 单帧最大跳变 {step*1e3:.1f} mrad (裸 {base*1e3:.1f}) "
                f"—— **相对峰值检查，非逐时刻**")

    if probe_kind == "multi_joint_continuous":
        # ── ③c 追加：**逐时刻余量** + 逐关节绝对差分 ─────────────────────
        # 约定：np.diff 索引 i ↔ 帧 i→i+1；二阶差分索引 i 涉及帧 i,i+1,i+2。
        d1u = np.abs(np.diff(cmd7[:, :6], axis=0))
        d1t = np.abs(np.diff(cmd7[:, :6] + da, axis=0))
        ex = d1t - d1u
        i1 = np.unravel_index(np.argmax(ex), ex.shape)
        msgs.append(f"· 逐时刻最坏步长增量 {ex[i1]*1e3:+.3f} mrad @帧{i1[0]} "
                    f"j{i1[1]+1}（绝对 {d1t[i1]*1e3:.3f}，基线同帧 {d1u[i1]*1e3:.3f}）")
        d2t = np.abs(np.diff(cmd7[:, :6] + da, 2, axis=0))
        R2D = 180.0 / np.pi
        msgs.append("· 逐关节绝对指令差分 |Δ|max (rad/s) / |Δ²|max (°/s²)："
                    + " · ".join(f"j{j+1} {d1t[:, j].max()/DT:.3f}/"
                                 f"{d2t[:, j].max()/DT**2*R2D:.0f}" for j in range(6)))
        msgs.append("⚠ **这些是指令差分，不是实测关节速度/加速度**；且仓内目前"
                    "**没有有出处又适用于本设备的速度/加速度上限**")
        blockers.append(
            "multi_joint_continuous：速度/加速度**无有出处又适用于本设备的上限**"
            "（G-v2 A_MAX 是阶跃拟合的模型参数，裸轨迹本身就超它；"
            "发送路径无软件限幅）⇒ **安全审查未完成**")
    return msgs, degraded, blockers



# ── 断开前的分段插值归位（2026-08-26 新增；修复未被任何检查覆盖的真机回停）──────
# 起因: `PiperMotorsBus.safe_disconnect()` 是**一次性写、无插值**:
#     safe_disable_position = [0, 0, 0, 0, 0.52, 0, 0]
#     def safe_disconnect(self): self.write(target_joint=self.safe_disable_position)
# 从 S_j4_p2_rise10_a8.2 的末帧出发, 等于让腕部在低头 62° 的姿态下横滚 114°、
# 同时 j2 下扫 51°, 全部并发、速度由固件决定 —— 实测发生碰撞, 臂停在瘫态。
#
# **为什么模型检查看不见**: j6 单独转 30° 时 link6 原点移动 **0.0 mm**
# (它是绕工具轴的末端滚转)。模型里腕体完全不动 ⇒ 任何基于模型几何的碰撞检查
# 对 j6 的滚转天然是瞎的。而 IMU 与支架挂在末端、有径向伸出, 滚 114° 就是让它
# 扫一个圆。IMU/支架未建模 ⇒ 该运动在仿真里不存在。
#
# 故本函数**不假装能算出安全路径**, 它做的是三件可验证的事:
#   ① **一次只动一组**, 慢速余弦插值 —— 若发生接触, 速度与动量都小;
#   ② **j6 最先单独走** —— 它是模型看不见的那个, 且此刻臂正处在刚刚安全待过
#      整条激励的位形上, 是已知最可信的起点;
#   ③ **每段都有到位门禁** —— 不到位就**停在原地并如实报告**, 不盲目进下一段。
#      (这正是上次缺的: 一次性写之后没有任何检查, 撞停了也无人知晓。)
SAFE_DISABLE = np.array([0.0, 0.0, 0.0, 0.0, 0.52, 0.0])      # 与 SDK 一致(弧度)
RETURN_STAGES = (("j6 腕滚转(模型不可见)", [5]),
                 ("j4/j5 腕", [3, 4]),
                 ("j1–j3 近端", [0, 1, 2]))
RET_S, RET_SETTLE_S, RET_TOL = 3.0, 0.5, np.radians(1.5)


def staged_return(robot, torch, grip: float) -> bool:
    """分段插值归位到 safe_disable_position。返回是否全部到位。"""
    def read():
        return robot.capture_observation()["observation.state"].numpy()[:6] \
            * np.pi / 180 / 1000
    q = read()
    print("\n  分段归位（取代固件的一次性无插值写）")
    for name, idx in RETURN_STAGES:
        tgt = q.copy()
        tgt[idx] = SAFE_DISABLE[idx]
        if np.abs(tgt - q).max() < np.radians(0.2):
            print(f"    {name}: 已在位, 跳过")
            continue
        n = int(RET_S * FPS)
        r = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, n)))
        for i in list(range(n)) + [n - 1] * int(RET_SETTLE_S * FPS):
            act = q + (tgt - q) * r[min(i, n - 1)]
            robot.send_action(torch.from_numpy(
                np.concatenate([act, [grip]]).astype(np.float32)))
            time.sleep(1.0 / FPS)
        q_now = read()
        dev = np.abs(q_now - tgt)[idx]
        mv = np.degrees(np.abs(tgt - q)[idx]).max()
        if dev.max() > RET_TOL:
            j = int(np.array(idx)[int(dev.argmax())])
            print(f"    ✗ {name}: 行程 {mv:.1f}°, **j{j+1} 未到位 "
                  f"{np.degrees(dev.max()):.2f}° > {np.degrees(RET_TOL):.1f}°**")
            print("      ⇒ 停在此处, **不进入下一段**。疑似受阻, 请人工检查后再动。")
            return False
        print(f"    ✓ {name}: 行程 {mv:.1f}°, 最大偏差 {np.degrees(dev.max()):.2f}°")
        q = q_now
    return True


def run(arm, dry_run=False, seqfile="probe_seq.npz", tag="", b2=False,
        outdir=None, robot_type="piper", settle_probe=0.0, canhs=True,
        probe_kind="single_joint_pulse"):
    cmd7, da, z = load_seq(seqfile, b2)
    import re
    is_bare = bool(re.fullmatch(r"A\d+", arm))      # A1/A2/A3/A4… 全是裸复放
    use_probe = arm.startswith("B") or (b2 and not is_bare)
    total = cmd7[:, :6] + (da if use_probe else 0.0)
    print(f"=== 复放 {arm} ({'加探针' if use_probe else '裸'}) · {len(cmd7)} 帧 "
          f"= {len(cmd7)/FPS:.0f}s ===")
    msgs, degraded, blockers = safety_checks(
        cmd7, da if use_probe else np.zeros_like(da), probe_kind=probe_kind)
    for m in msgs:
        print("  " + m)
    for d in degraded:
        print(f"  ⚠ 降级: {d}")

    if dry_run:
        print("\n[dry-run] 不连机器人。时序模拟:")
        t0 = time.perf_counter()
        for i in range(min(100, len(cmd7))):
            time.sleep(0)                       # 仅测循环开销
        oh = (time.perf_counter() - t0) / 100 * 1e3
        print(f"  循环开销 {oh:.3f} ms/帧 (预算 {DT*1e3:.0f} ms)  ✓")
        print(f"  预计时长 {len(cmd7)/FPS:.0f}s")
        if degraded:
            print(f"\n  ⚠⚠ **降级 dry-run** —— {len(degraded)} 项依赖缺失/跳过。")
            print("      仅可用于诊断接口, **不足以放行上机**。")
            print("      请在 lerobot 环境下重跑: "
                  "~/miniconda3/envs/lerobot/bin/python sim/replay_probe.py …")
            return 2
        if blockers:
            print(f"\n  ⛔ **{len(blockers)} 项硬阻断**（dry-run 到此为止，"
                  f"真机会在连接设备前退出）：")
            for b in blockers:
                print(f"     ⛔ {b}")
            return 4
        print("  ✓ 全部安全检查通过 —— 可上机")
        return 0

    # ---- ⛔ 运动入口硬阻断（**不是打印警告**）────────────────────────────
    # 退出码 4 = 安全审查未完成。**发生在 make_robot / connect 之前**，
    # 因此设备不会被连接、电机不会被使能、臂不会动。
    # 解除方式**只有一条**：完成 `reports/CHECKLIST_3c_readonly.md` 的 R 阶段，
    # 把有出处且适用的限值填进 `sim/audit_send_chain_3c.py` 的来源表，
    # 使 safety_checks 不再产生 blockers。**不提供 --force 之类的旁路。**
    if blockers:
        print(f"\n  ⛔⛔ **拒绝上机：{len(blockers)} 项安全审查未完成** ⛔⛔")
        for b in blockers:
            print(f"     ⛔ {b}")
        print("     ⇒ **设备未连接、电机未使能、臂未运动。**")
        print("     ⇒ 解除条件见 reports/CHECKLIST_3c_readonly.md（R 阶段）。")
        return 4

    # ---- 真机 ----
    import torch
    from lerobot.common.robot_devices.robots.utils import make_robot
    # 探针只需 IMU+关节: inference_time=True(推理模式,无手柄teleop) · enable_imu=True(默认关!)
    # · cameras={}(不读图, 省循环开销且避开已知的 ffmpeg SIGABRT)
    # ── 退出码 3 = 「机器还没动就失败」(2026-08-26 冻结) ────────────────────
    # 建机器人 / 连接 / IMU 自检**全部发生在任何运动之前**: 失败时机时为 0、
    # 无数据产生, 因此它对「这条序列能不能跑」不提供任何信息。编排器据此记为
    # **环境失败**, **不计入该序列的作废额度**, 并立即停整轮等人修环境 ——
    # 而不是对着一个 CAN 故障连撞 MAX_VOIDS 次(2026-08-26 实际发生过)。
    # 越过这一段之后的任何失败, 判据仍是**帧数**(⑥-5b), 按正常作废处理。
    try:
        robot = make_robot(robot_type, inference_time=True, enable_imu=True,
                           cameras={})
        if not robot.is_connected:
            robot.connect()
        # IMU 连接自检: 动之前先确认 IMU 活着, 否则白跑 59s
        obs0 = robot.capture_observation()
        if "observation.imu_real" not in obs0:
            raise RuntimeError("IMU 未启用! 检查 enable_imu / TSND 设备 /dev/ttyACM0")
        imu0 = obs0["observation.imu_real"].numpy()
        if not np.isfinite(imu0).all():
            raise RuntimeError(f"IMU 首帧异常: {imu0}")
    except Exception as e:                                        # noqa: BLE001
        traceback.print_exc()
        print(f"\n  ✗✗ **环境失败(运动前)**: {type(e).__name__}: {e}")
        print("      机器未动、无数据产生 ⇒ 不计入该序列作废额度。")
        print("      常见: CAN 未起(ip -br link show can0) / 臂未上电 / "
              "TSND 未插(/dev/ttyACM0)")
        return 3
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
    # ── 稳定判据（2026-08-24 修订并冻结；这是**失败之后**的门槛修订）────────
    # 起因: j4 pilot 的 P03 连续两次被旧判据(末段 0.5s 峰速 ≤ 0.5°/s)拦下。
    # 20 秒静置诊断证明: j4/j6 存在**有界、不衰减**的腕部极限环 ——
    #   20s 内 j4 跳 29 次 / j6 跳 11 次, 幅度 0.104–0.152°, 速率约 1.5 次/秒,
    #   分四段计数 10/8/10/11(**不衰减**), j4 位置始终在 0.284° 带内往复,
    #   其余四关节全程精确 0.0000。编码器细分辨率 ≤0.005° ⇒ 不是量化噪声。
    # 旧判据测的是**速度**, 而它的设计意图是「预移动的余振停了没有」。极限环
    # 不是余振: 有界、不衰减、不发散。用速度阈值去测它, 0.5s 窗口撞上跳变的
    # 概率约一半 ⇒ **门变成抛硬币**(P01/P02 通过时 settle_vmax_dps 恰为 0.0)。
    # 修订: 改测**位置带**(运动是否有界在可接受范围内), 不对 j4/j6 做隐藏豁免。
    PREMOVE_S, PRESETTLE_S = 3.0, 5.0              # 稳定时间 1.5 → 5.0 s
    BAND_WINDOW_S = 2.0                            # 判据只看**最后 2 s**
    # ── 位置带阈值（2026-08-26 修订并冻结；又一次**失败之后**的门槛修订）──────
    # 起因: 主套件第 28 条 S_j2_p0_r5_f0.35 被 0.5° 拦下 (j4 R=0.554°)。
    # 证据(27 条已收 + 该次失败 + 同位形 20s 静置探针):
    #   · **j1/j2/j3 在 27 条里全部精确 0.000°** ⇒ 现象纯在腕部, 与近端无关
    #   · **有界**: 20s 探针累积带宽 2s 内饱和, 之后 18s 增量全为 0.0000
    #   · **非位形决定**: 同一位形三次给出 0.119 / 0.241 / 0.554°
    #   · 与会话时长无关(r = −0.12/−0.24/+0.08); 与腕部离桌高度无关
    #     (r = +0.367 且**方向反**: 最大的两个带宽出现在离桌 694 mm 处)
    #     ⇒ 排除「跑久了变热」与「腕上设备撞桌」两种解释
    #   · 机制: 量子化步长 ~0.107°, 偶发 2 倍双跳, 速率 1.15–3.40 次/s
    #   · 另: 打开碰撞用真实 mesh 跑完 90 条全时间线, **零接触**(臂/桌/地)
    # 即门禁拦下的是已定性的**腕部粘滑极限环**, 不是「预移动余振没停」。
    # 预移动峰速 19.9°/s ⇒ 真没停会是**度**级带宽, 不是 0.5°。
    # 28 次尝试的腕部上尾: 0.554(失败) / 0.465 / 0.459 / 0.415 / 0.368 ——
    # 0.5° 落在这个分布**中间**, 已有两条通过贴到阈值的 93%。
    # 修订(用户 2026-08-26 决定): **近端保持 0.5°, 腕部单独定 0.8°**。
    #   近端实测恒为 0.000 ⇒ 任何非零都有意义, **不放宽**;
    #   腕部 0.8° 较实测最大 0.554° 留 44% 余量, 仍远低于真正未稳定的度级带宽。
    # 判据仍是 max−min, 与已收 27 条**可比**; 逐关节带宽继续完整报告与落盘。
    BAND_TOL = np.radians(np.array([0.5, 0.5, 0.5, 0.8, 0.8, 0.8]))
    if settle_probe > 0:
        # **观测模式**: 只做预移动 + 长时间静置采样, 然后退出, **不跑激励**。
        # 用来回答「j4 的粘滑会不会自己停」—— 这是实测题, 不是判断题。
        # 协议默认值(1.5s)与安全门(0.5°/s)都**不改动**。
        PRESETTLE_S = float(settle_probe)
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
    if settle_probe > 0:
        s = np.degrees(np.asarray(settle, float))
        v = np.abs(np.diff(s, axis=0) * FPS)
        jump = np.where(v.max(1) > np.degrees(PRE_VEL_TOL))[0]
        print(f"\n[静置观测 {PRESETTLE_S:.0f}s] {len(s)} 个采样 @ {FPS:.0f} Hz")
        print(f"  逐关节位移范围(度): {np.round(s.max(0) - s.min(0), 4)}")
        print(f"  逐关节峰速(度/s)  : {np.round(v.max(0), 3)}")
        print(f"  超阈事件 {len(jump)} 次（阈 {np.degrees(PRE_VEL_TOL):.1f}°/s）")
        for k in jump:
            j = int(v[k].argmax())
            print(f"    t={k/FPS:5.2f}s  j{j+1}  {v[k, j]:.3f}°/s "
                  f"(位移 {v[k, j]/FPS:+.4f}°)")
        if len(jump) >= 2:
            half = len(s) // 2
            e1 = int((v[:half - 1].max(1) > np.degrees(PRE_VEL_TOL)).sum())
            e2 = int((v[half:].max(1) > np.degrees(PRE_VEL_TOL)).sum())
            print(f"  **前半段 {e1} 次 vs 后半段 {e2} 次** ⇒ "
                  f"{'在衰减(疑似应力释放瞬态)' if e2 < e1 else '未衰减(持续现象)'}")
        # **分段位置带** —— 事件次数与带宽是两回事: 2026-08-26 的
        # S_j2_p0_r5_f0.35 在 j4=59° 处带宽 5s 内 0.117→0.427°(涨 4 倍),
        # 而门禁判的正是带宽。诊断必须直接给出「带宽是收敛到上界还是继续涨」。
        nseg = 5
        seg = [s[i * len(s) // nseg:(i + 1) * len(s) // nseg] for i in range(nseg)]
        bands = np.array([w.max(0) - w.min(0) for w in seg])       # (nseg, 6)
        cum = np.array([s[:(i + 1) * len(s) // nseg].max(0)
                        - s[:(i + 1) * len(s) // nseg].min(0)
                        for i in range(nseg)])                     # 累积带宽
        jw = int(bands.max(0).argmax())                            # 最活跃关节
        print(f"  分 {nseg} 段位置带 j{jw+1}(度): {np.round(bands[:, jw], 4)}")
        print(f"  **累积**位置带 j{jw+1}(度): {np.round(cum[:, jw], 4)}"
              f"  ⇒ {'趋于上界(有界)' if cum[-1, jw] < 1.15 * cum[nseg//2, jw] else '**仍在增长**'}")
        print(f"  全程位置带(度): {np.round(s.max(0) - s.min(0), 4)}"
              f"   门禁阈 {np.round(np.degrees(BAND_TOL), 1)}°")
        _d = (outdir or PROBE) / f"settle_probe_{tag}{arm}_{int(time.time())}.json"
        _d.parent.mkdir(parents=True, exist_ok=True)
        _d.write_text(json.dumps({
            "mode": "settle_probe", "arm": arm, "seqfile": str(seqfile),
            "settle_s": PRESETTLE_S, "fps": FPS,
            "settle_deg": s.tolist(), "vel_deg_s": v.tolist(),
            "jump_idx": jump.tolist(),
            "band_full_deg": (s.max(0) - s.min(0)).tolist(),
            "band_tol_deg": np.degrees(BAND_TOL).tolist(),
        }, ensure_ascii=False, indent=1))
        print(f"  ⤷ {_d.name}")
        print("  （观测模式：**未跑激励序列**，安全门与协议默认值均未改动）")
        staged_return(robot, torch, float(np.clip(cmd7[0, 6], 0, 0.035)))
        robot.disconnect()
        return 0

    obs0 = robot.capture_observation()                    # 起始状态改在**预移动后**取
    q_pre = obs0["observation.state"].numpy()[:6] * np.pi / 180 / 1000
    # (a) 稳定: **位置带**判据 —— 最后 BAND_WINDOW_S 秒内每个关节的 max−min
    win = np.asarray(settle[-int(BAND_WINDOW_S * FPS):], float)
    band = win.max(0) - win.min(0)
    band_bad = band > BAND_TOL
    # 速度/跳变仍**完整报告**, 但不再决定通过与否
    tail = np.asarray(settle[-int(0.5 * FPS):], float)
    vel = np.abs(np.diff(tail, axis=0) * FPS) if len(tail) > 1 else np.zeros((1, 6))
    vmax = float(vel.max())
    # **单次毛刺 vs 持续振荡**必须分得开: 前者 max≫median、只占一两个采样;
    # 后者两者接近。此前只报一个 max, 分不出来也留不下证据(断言在落盘之前触发)。
    v_med = float(np.median(vel))
    # `vel` 形状是 (采样, 关节) —— 此前把返回值赋成 (关节, 采样), 报错信息里
    # 的关节号是错的（实测把 j4 报成了 j2）。
    k_v, j_v = np.unravel_index(int(vel.argmax()), vel.shape)
    # 判「毛刺 vs 持续」必须看**整个稳定段**, 不能只看末段窗口: 实测 P03 全程
    # 有 3 次离散跳变, 只有 1 次落在末段窗口内 ⇒ 按窗口计数会误判成「单次毛刺、
    # 可重试」, 而它其实是可复现的粘滑/齿隙释放。
    vel_all = np.abs(np.diff(np.asarray(settle, float), axis=0) * FPS)
    n_over_all = int((vel_all > PRE_VEL_TOL).sum())
    n_over = int((vel > PRE_VEL_TOL).sum())
    # (b) 到位容差 —— **两套规则, 按序列来源自动选定, 每次运行都打印**
    #
    # 原规则: 容差 = 「指令超出**可达范围**的量」+ 1°。`reach_limits()` 是 20 集
    # pp2 编码器实测 min/max, 本意很窄且正确 —— 处理**静态饱和**的关节
    # (j2 指令到 −2.40° 但编码器只到 −0.07°, 静态偏差 +1.17°)。
    #
    # 2026-08-26 查出误用: step2_g 激励套件**按设计**走出 pp2 包络(j4 在 pp2 里
    # 从没动过), 于是「为饱和开的口子」变成了「为**没去过的地方**开的口子」——
    # 75/90 条被放宽, j3 最多 110.4°、j6 97.6°、j4 81.5°, **到位门禁等于关闭**。
    # 实测它从未真的起作用: 已收 27 条最大偏差 0.151°, 比 1° 低一个数量级。
    # ⇒ 用户决定(2026-08-26): **step2_g 一律用固定 1°**, 不做饱和豁免。
    #   依据: 套件位形全部在模型限位内且余量 ≥5°, 真机实测到位误差 ≤0.151°。
    #   若后续某条确实命中 j2/j3 饱和点, 会**如实挂掉**并停机排查 —— 这正是
    #   门禁该做的事, 而不是被一个 110° 的豁免提前抹掉。
    # 判据按**序列来源**自动选, 不做成可能忘记传的开关。
    _from_suite = "outputs/step2_g" in str(Path(seqfile).resolve()) \
        if seqfile else False
    if _from_suite:
        unreach = np.zeros(6)
        tol = np.full(6, np.radians(1.0))
        _tol_rule = "固定 1° (step2_g 套件; 不做饱和豁免)"
    else:
        from make_probe_b2 import reach_limits
        klo, khi = reach_limits()
        unreach = np.maximum(np.maximum(klo - q_tgt, q_tgt - khi), 0.0)
        tol = unreach + np.radians(1.0)
        _tol_rule = "可达范围豁免 + 1° (pp2 回放)"
    dev = np.abs(q_pre - q_tgt)
    j_bad = int(np.argmax(dev - tol))
    print(f"  预移动后 [判据: 最后 {BAND_WINDOW_S:.0f}s 位置带 R_j ≤ "
          f"{np.round(np.degrees(BAND_TOL), 1)}° (近端/腕部)]")
    print(f"    R_j = {np.round(np.degrees(band), 4)}°"
          f"   {'✓ 全部在带内' if not band_bad.any() else '✗ 超限 j' + str(int(band_bad.argmax()) + 1)}")
    print(f"    —— 以下仅报告, **不决定通过/失败** ——")
    _sd = np.diff(np.asarray(settle, float), axis=0)      # 带符号, 供方向统计
    print(f"    末段峰速 {np.degrees(vmax):.3f}°/s @ j{j_v+1}; "
          f"中位 {np.degrees(v_med):.3f}°/s")
    print(f"    跳变(>{np.degrees(PRE_VEL_TOL):.1f}°/s) 全稳定段 {n_over_all}/{vel_all.size}"
          f"; 逐关节 {(vel_all > PRE_VEL_TOL).sum(0).tolist()}")
    for _j in range(6):
        _m = np.abs(_sd[:, _j]) * FPS > PRE_VEL_TOL
        if _m.any():
            _v = _sd[_m, _j] * FPS
            print(f"      j{_j+1}: {int(_m.sum())} 次, 幅度 "
                  f"{np.degrees(np.abs(_v)).min():.3f}–{np.degrees(np.abs(_v)).max():.3f}°/s, "
                  f"方向 +{int((_v > 0).sum())}/−{int((_v < 0).sum())}, "
                  f"净位移 {np.degrees(_sd[:, _j].sum()):+.4f}°")
    # **稳定段轨迹每次都存**（不只失败时）—— P01/P02 只存了标量 settle_vmax_dps,
    # 门槛修订后无法回算, 这个缺口不该再出现。
    _tagp = "settle_fail_" if band_bad.any() else "settle_"
    if True:
        _d = (outdir or PROBE) / f"{_tagp}{tag}{arm}_{int(time.time())}.json"
        _d.parent.mkdir(parents=True, exist_ok=True)
        _d.write_text(json.dumps({
            "arm": arm, "seqfile": str(seqfile),
            "settle_deg": np.degrees(np.asarray(settle, float)).tolist(),
            "tail_vel_deg_s": np.degrees(vel).tolist(),
            "vmax_deg_s": float(np.degrees(vmax)),
            "v_median_deg_s": float(np.degrees(v_med)),
            "peak_sample_in_tail": int(k_v) + 1,
            "n_over_tail": n_over, "n_samples_tail": int(vel.size),
            "n_over_full_settle": n_over_all, "n_samples_full": int(vel_all.size),
            "peak_joint_1based": int(j_v) + 1,
            "band_deg": np.degrees(band).tolist(),
            "band_tol_deg": np.degrees(BAND_TOL).tolist(),
            "band_window_s": BAND_WINDOW_S, "settle_s": PRESETTLE_S,
            "gate": "position-band (2026-08-24 修订冻结)",
            "passed": bool(not band_bad.any()),
            "verdict": ("疑似单次毛刺 —— 可重试一次"
                        if n_over_all <= 1 and vmax > 3 * max(v_med, 1e-9)
                        else f"**全稳定段有 {n_over_all} 次超阈 ⇒ 可复现现象, "
                             f"不是毛刺; 不要靠重试绕过**"),
        }, ensure_ascii=False, indent=1))
        print(f"    ⤷ 稳定段轨迹: {_d.name}")
    print(f"    到位容差规则: {_tol_rule}")
    print(f"    逐关节 |偏差| {np.round(np.degrees(dev), 3)}°  容差 {np.round(np.degrees(tol), 3)}°"
          + (f"  (含指令超出可达范围 {np.round(np.degrees(unreach), 3)}°)"
             if unreach.max() > 0 else ""))
    # 逐关节阈值下, 「第一个超标的关节」与「带宽最大的关节」可能**不是同一个**
    # (例: j3 0.6°>0.5 超标, j4 0.7°<0.8 合规, 带宽最大的却是 j4)。
    # 必须报真正超标那一个及**它自己的**阈值, 否则错误信息张冠李戴。
    _jb = int(np.argmax(np.where(band_bad, band - BAND_TOL, -np.inf)))
    assert not band_bad.any(), (
        f"稳定段位置带超限: j{_jb+1} R={np.degrees(band[_jb]):.3f}° "
        f"> {np.degrees(BAND_TOL[_jb]):.1f}° —— **立即停机, 不自动重试**")
    assert (dev <= tol).all(), (
        f"预移动未到位: j{j_bad+1} 偏差 {np.degrees(dev[j_bad]):.3f}° > 容差 "
        f"{np.degrees(tol[j_bad]):.3f}° —— 停机排查, 六臂起始位姿必须一致")
    print(f"  ✓ 预移动到位 (最大超容差 {np.degrees((dev - tol).max()):+.3f}°)")

    # t_send: **JointCtrl 真正写总线的时刻**(ZOH 起点), monotonic 与 TSND host_t 同钟。
    # 同时存 t_send_span = post−pre, 用于量化"实际发送时刻"本身的不确定度。
    import hashlib
    cmd_hash = hashlib.sha256(np.ascontiguousarray(cmd7, np.float32).tobytes()).hexdigest()[:16]
    rec = {"arm": arm, "t": [], "t_send": [], "t_send_span": [], "cmd": [],
           "imu": [], "state": [], "imu100": [], "can_hs": [], "can_q200": [],
           "clock": "time.monotonic",
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
    # 六关节高速 CAN 反馈(0x251–0x256)。**被动只读**, 独立 socket, 不碰控制回路。
    # 轮询 GetArmHighSpdInfoMsgs() 会对六路异步流做无抗混叠抽取 —— 与 TSND 同一个坑。
    can_rec, can_raw = None, []
    if canhs:
        sys.path.insert(0, str(ROOT / "sim/step3_canobs"))
        from can_hs_recorder import CanHighSpdRecorder
        can_rec = CanHighSpdRecorder()
        # wait_first 不能省: 接口 DOWN 时 bind 照样成功, 会静默录到零帧。
        if can_rec.start_recording() and can_rec.wait_first():
            print("  高速 CAN 反馈记录已开启 (0x251–0x256, 被动只读)")
        else:
            print(f"  ⚠ 高速 CAN 反馈未开启: {can_rec.err} —— **本条不能用于 CAN 可观测性分析**")
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
            if can_rec is not None:
                can_raw.extend(can_rec.drain_recording())   # 只搬原始字节, 不解码
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
        if can_rec is not None:
            can_raw.extend(can_rec.drain_recording())
            can_rec.stop_recording()
            # 解码放在停机之后 —— 在回路里解码会拖慢 20 Hz 主环(2026-09-01 教训)
            from can_hs_recorder import decode as _candec
            rec["can_hs"], rec["can_q200"] = _candec(can_raw)
            # 固件/SDK 版本要与数据同文件落盘: effort 标度规则依赖固件版本
            rec["can_hs_meta"] = can_rec.meta(getattr(robot.arm, "piper", None))
        p = (outdir or PROBE) / f"replay_{tag}{arm}.json"   # tag 含 ep 与轮次
        if p.exists():
            # **移开, 不覆盖**。判据修订后重跑同一条序列会同名 —— 原始数据仍然
            # 有效(激励段本身没问题, 只是稳定段判据变了), 覆盖掉就再也回不去了。
            _old = p.with_name(f"{p.stem}.superseded_{int(time.time())}.json")
            p.rename(_old)
            print(f"  ⚠ 已存在同名结果, 移开保留为 {_old.name}")
        rec["stopped"] = stopped
        rec["seqfile"] = seqfile
        p.write_text(json.dumps(rec))
        print(f"\n采集 {len(rec['imu'])} 帧(20Hz) + {len(rec['imu100'])} 样本(100Hz)"
              f" + {len(rec['can_hs'])} 帧(电机200Hz)"
              f" + {len(rec['can_q200'])} 帧(关节200Hz) → {p}"
              + (f"  ⚠ 提前终止: {stopped}" if stopped else "  ✓"))
        # **先落盘, 再动机器** —— 归位若受阻, 数据已经安全。
        # 归位放在这里而不是靠 __del__: 固件的一次性无插值写会从**激励末帧**
        # 直扫 safe_disable, 那正是 2026-08-26 碰撞的成因(见 staged_return 注释)。
        try:
            ok = staged_return(robot, torch, float(np.clip(cmd7[0, 6], 0, 0.035)))
            if not ok:
                rec["return_blocked"] = True
                p.write_text(json.dumps(rec))
                print("  ⚠ **归位受阻已记入结果文件**(return_blocked)。"
                      "机器停在中途, 请人工检查后再运行下一条。")
        except NameError:
            pass                                   # 未走到真机路径
        except Exception as e:                     # noqa: BLE001
            print(f"  ⚠ 归位过程异常: {type(e).__name__}: {e}")
    return 0 if stopped is None else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A1",
                    help="A1/A2 或 B′ 配置名如 j3_8mrad_p")
    ap.add_argument("--ep", type=int, default=None,
                    help="pp2 集号(0/1/2); 用 probe_seq_pp2_ep{ep}.npz, 输出 replay_ep{ep}_*.json")
    ap.add_argument("--b2", action="store_true", help="用 B′(6-8Hz 多正弦)序列")
    ap.add_argument("--amp", default="4mrad", help="A 臂借哪个幅值的 cmd7(内容无关)")
    ap.add_argument("--round", type=int, default=1, choices=[1, 2, 3],
                    help="2=反向平衡块(4mrad) / 3=幅值对比(2mrad, 顺序同第2轮); "
                         "输出文件加 _rN_ 前缀, 不覆盖前轮")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--robot", default="piper",
                    help="机器人类型, 传给 make_robot（此前 docstring 写了它但 CLI 没定义）")
    ap.add_argument("--seqfile", help="**显式**序列 npz 路径; 给了就不再从 --ep/--b2 推导")
    ap.add_argument("--outdir", help="落盘目录; 默认 outputs/probe（--b2 时 outputs/probe_b2）")
    ap.add_argument("--settle-probe", type=float, default=0.0, metavar="SEC",
                    help="**观测模式**: 预移动后静置 SEC 秒并记录, 然后退出, "
                         "不跑激励序列。用于测「粘滑会不会自己停」。")
    ap.add_argument("--probe-kind", default="single_joint_pulse", choices=PROBE_KINDS,
                    help="探针形态。**multi_joint_continuous 必须显式给出**"
                         "（③c 的多关节连续补正；旧模式的「同帧仅一关节」断言不删）")
    ap.add_argument("--no-canhs", action="store_true",
                    help="关闭六关节高速 CAN 反馈记录(默认开; 被动只读, 不影响控制)")
    a = ap.parse_args()
    if a.seqfile:                                  # 显式路径优先, 直接跑
        raise SystemExit(run(a.arm, a.dry_run, seqfile=a.seqfile, tag="",
                             b2=a.b2, outdir=Path(a.outdir) if a.outdir else None,
                             robot_type=a.robot, settle_probe=a.settle_probe,
                             canhs=not a.no_canhs, probe_kind=a.probe_kind))
    if a.b2:
        import re as _re
        cfg = ("j2_" + a.amp + "_p" if _re.fullmatch(r"A\d+", a.arm)
               else a.arm)                          # A 臂借任一序列的 cmd7(只用 cmd7)
        seqfile = f"seq_ep{a.ep}_{cfg}.npz"
        # 轮次后缀: 第2轮的 B 臂配置名与第1轮相同, 不加后缀会**覆盖首轮数据**
        tag = f"ep{a.ep}_" if a.round == 1 else f"ep{a.ep}_r{a.round}_"
        raise SystemExit(run(a.arm, a.dry_run, seqfile=seqfile,
                             tag=tag, b2=True, outdir=PB2, canhs=not a.no_canhs, probe_kind=a.probe_kind))
    seqfile = f"probe_seq_pp2_ep{a.ep}.npz" if a.ep is not None else "probe_seq.npz"
    tag = f"ep{a.ep}_" if a.ep is not None else ""
    raise SystemExit(run(a.arm, a.dry_run, seqfile=seqfile, tag=tag,
                         canhs=not a.no_canhs, probe_kind=a.probe_kind))
