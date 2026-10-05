#!/usr/bin/env python3
"""重复精度 25 次的单次复放执行器 —— 只由 `run_repeat_precision25.py --run-next` 逐条调用，不单独运行。

照搬 pilot 已实测 14 次的执行路径（`replay_probe.py` 的 pilot 分支）：
预移动、位置带与到位门禁、采集循环、清理时不额外发归位或禁能。与 pilot 的差别（用户 2026-10-05 决定）：
  1. 放行核验换成本次入口的 `validate_replay_context`（`replay_probe` 的钩子写死为 pilot 入口）；
  2. 分段归位受会话截止时间约束：每次下发动作之前检查，到期即不再下发任何新运动指令，记 `return_deadline`
     并判失败；分段、速度、到位容差与原函数相同，不为赶时间加速；
  3. 打开 100 Hz 记录后，先以 20 Hz 发 4 帧（0.2 s）与稳定段相同的保持目标（目标不变），再开始第一个正式帧。
     前置保持单独记在 `prehold`，不混入正式的 `cmd` / `t_send`；所有条件（含 A0）统一执行。
     它只增加覆盖余量、不保证覆盖；覆盖由入口按实际时间戳核验。
只从 `replay_probe` 导入不发运动指令的部分：`safety_checks`、`pilot_check_imu_freshness` 与常量。
退出不代表禁能：清理不发归位或禁能命令，电机可能仍使能。
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from replay_probe import (DT, FPS, MISSING_RATE_LIMIT_BLOCKER, PILOT_RETURN_RESERVE_S,  # noqa: E402
                          PILOT_START_RESERVE_S, RET_S, RET_SETTLE_S, RET_TOL, RETURN_STAGES, SAFE_DISABLE,
                          pilot_check_imu_freshness, safety_checks)

PREHOLD_FRAMES = 4
# 以下与 replay_probe._run_impl 内的局部常量相同（2026-08-24 / 08-26 修订并冻结的稳定判据）
PREMOVE_S, PRESETTLE_S, BAND_WINDOW_S = 3.0, 5.0, 2.0
BAND_TOL = np.radians(np.array([0.5, 0.5, 0.5, 0.8, 0.8, 0.8]))   # 近端 0.5°、腕部 0.8°（max−min）
PRE_VEL_TOL = np.radians(0.5)


def _load(seqfile):
    with np.load(seqfile, allow_pickle=False) as z:
        return z["cmd7"].astype(np.float32), z["da"].astype(np.float32)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def staged_return_guarded(robot, torch, grip: float, deadline: float) -> str:
    """分段插值归位到 safe_disable_position（分段、速度、容差同 replay_probe.staged_return），
    每次下发动作之前检查截止时间 → "ok" | "blocked" | "deadline"。到期即停止下发，不加速。"""
    def read():
        return robot.capture_observation()["observation.state"].numpy()[:6] * np.pi / 180 / 1000
    q = read()
    print("\n  分段归位（受截止时间约束）")
    for name, idx in RETURN_STAGES:
        tgt = q.copy()
        tgt[idx] = SAFE_DISABLE[idx]
        if np.abs(tgt - q).max() < np.radians(0.2):
            print(f"    {name}: 已在位, 跳过")
            continue
        n = int(RET_S * FPS)
        r = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, n)))
        for i in list(range(n)) + [n - 1] * int(RET_SETTLE_S * FPS):
            if time.monotonic() >= deadline:
                print(f"    ⛔ {name}: 到达会话截止时间 —— 不再下发任何运动指令（不加速）。"
                      "臂停在中途，按设备程序处置。")
                return "deadline"
            act = q + (tgt - q) * r[min(i, n - 1)]
            robot.send_action(torch.from_numpy(np.concatenate([act, [grip]]).astype(np.float32)))
            time.sleep(1.0 / FPS)
        q_now = read()
        dev = np.abs(q_now - tgt)[idx]
        if dev.max() > RET_TOL:
            j = int(np.array(idx)[int(dev.argmax())])
            print(f"    ✗ {name}: j{j+1} 未到位 {np.degrees(dev.max()):.2f}° —— 停在此处，不进入下一段")
            return "blocked"
        print(f"    ✓ {name}: 最大偏差 {np.degrees(dev.max()):.2f}°")
        q = q_now
    return "ok"


def _guard(scope, robot=None):
    if time.monotonic() >= scope["deadline_monotonic"] - PILOT_RETURN_RESERVE_S:
        raise RuntimeError("现场预算到达停止采集点；保留归位时间")
    if robot is not None:
        pilot_check_imu_freshness(robot)


def _finish_return(robot, torch, cmd7, scope, rec) -> None:
    """受截止约束的归位；结果写进记录（受阻 / 到期 / 异常都判失败）。"""
    try:
        st = staged_return_guarded(robot, torch, float(np.clip(cmd7[0, 6], 0, 0.035)), scope["deadline_monotonic"])
        if st == "blocked":
            rec["return_blocked"] = True
            rec["stopped"] = rec.get("stopped") or "归位受阻"
        elif st == "deadline":
            rec["return_deadline"] = True
            rec["stopped"] = rec.get("stopped") or "归位途中到达会话截止时间：已停止下发运动指令"
    except Exception as e:                                              # noqa: BLE001
        print(f"  ⚠ 归位过程异常: {type(e).__name__}: {e}")
        rec["return_error"] = f"{type(e).__name__}: {e}"
        rec["stopped"] = rec.get("stopped") or "归位异常"


def _abort_before_record(robot, torch, cmd7, scope, started_at_utc, outdir, run, reason) -> int:
    """预采集阶段停止：先落盘，再走受截止约束的归位。"""
    p = outdir / f"replay_{run['run_id']}.json"
    rec = {"run_id": run["run_id"], "scope": scope, "started_at_utc": started_at_utc, "clock": "time.monotonic",
           "t": [], "t_send": [], "cmd": [], "imu": [], "state": [], "imu100": [], "can_hs": [], "can_q200": [],
           "stopped": reason}
    p.write_text(json.dumps(rec))
    _finish_return(robot, torch, cmd7, scope, rec)
    p.write_text(json.dumps(rec))
    print(f"  ✗ 预采集阶段停止，已保留记录: {reason}")
    return 1


def run(run, seqfile, outdir, context, robot_type="piper", canhs=True) -> int:
    """执行入口预留的那一条；本函数拥有清理（不发归位或禁能）。"""
    outdir = Path(outdir)
    devices = []
    code = 5
    try:
        code = _run_impl(run, seqfile, outdir, context, robot_type, canhs, devices)
    finally:
        if devices:
            robot = devices[0]
            # PiperRobot.__del__ 会 disconnect()：一次性无插值写 home 再禁能。绝不能让它覆盖已停止 / 受阻 /
            # 到期的分段归位。这个标志只作用于包装器：它**不**禁能电机，也不建立任何安全状态。
            robot.is_connected = False
            cleanup = {"wrapper_destructor_disarmed": True, "motion_or_disable_sent_by_cleanup": False, "errors": []}
            for name, target, method in (
                    ("imu_stop", getattr(robot, "imu_reader", None), "stop"),
                    ("can_close", getattr(getattr(robot, "arm", None), "piper", None), "DisconnectPort")):
                try:
                    if target is not None:
                        getattr(target, method)()
                    cleanup[name] = target is not None
                except Exception as e:                                  # noqa: BLE001
                    cleanup["errors"].append(f"{name}: {type(e).__name__}: {e}")
            p = outdir / f"replay_{run['run_id']}.json"
            rec = json.loads(p.read_text()) if p.exists() else {
                "run_id": run["run_id"], "imu": [], "stopped": "复放在写出记录之前失败"}
            rec["cleanup"] = cleanup
            if cleanup["errors"]:
                rec["stopped"] = rec.get("stopped") or "清理失败"
                code = 1
            p.write_text(json.dumps(rec))
            print("  清理：不额外发送归位或禁能命令；电机可能仍使能。")
    return code


def _run_impl(run, seqfile, outdir, context, robot_type, canhs, devices) -> int:
    started_at_utc = _utc()
    rid, use_probe = run["run_id"], bool(run["use_probe"])
    cmd7, da = _load(seqfile)
    total = cmd7[:, :6] + (da if use_probe else 0.0)
    print(f"=== 复放 {rid}（{'加探针' if use_probe else '裸'}）· {len(cmd7)} 帧 + 前置保持 {PREHOLD_FRAMES} 帧 ===")
    msgs, degraded, blockers = safety_checks(cmd7, da if use_probe else np.zeros_like(da),
                                             probe_kind="multi_joint_continuous")
    for m in msgs:
        print("  " + m)
    # 没有命令行绕过：放行只来自入口对磁盘 release、manifest、序列哈希与当前预留的核验
    if not context:
        print("  ⛔ 缺少入口预留的上下文（设备未连接）")
        return 4
    try:
        from run_repeat_precision25 import validate_replay_context
        scope = validate_replay_context(context, seqfile, rid, cmd7, da, use_probe)
        deadline = float(scope["deadline_monotonic"])
        needed = max(PILOT_START_RESERVE_S, len(cmd7) / FPS + PREHOLD_FRAMES / FPS + 28.0)
        if not np.isfinite(deadline) or deadline - time.monotonic() < needed:
            raise ValueError("现场预算余量不足，不连接设备")
        scope = dict(scope, deadline_monotonic=deadline)
    except Exception as e:                                              # noqa: BLE001
        print(f"  ⛔ 授权 / 预算核验失败（设备未连接）: {type(e).__name__}: {e}")
        return 4
    if degraded:
        print("  ⛔ 依赖降级，不允许连接设备")
        return 2
    others = [b for b in blockers if b != MISSING_RATE_LIMIT_BLOCKER]
    if others:
        print(f"  ⛔ 还有 {len(others)} 项未豁免的安全阻断（设备未连接）：")
        for b in others:
            print(f"     ⛔ {b}")
        return 4
    print("  · 速率上限来源未知这一项按已记录的工程授权处理；不是安全证明")

    import torch
    from lerobot.common.robot_devices.robots.utils import make_robot
    if scope["deadline_monotonic"] - time.monotonic() < needed:         # 导入可能耗时数秒（同 pilot）
        print("  ⛔ 现场预算余量不足，不开始连接（设备未连接）")
        return 4
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"replay_{rid}.json"
    # 退出码 3 = 初始化失败；make_robot / connect 可能已有使能或归位副作用，不能据此声称机器没动。
    try:
        robot = make_robot(robot_type, inference_time=True, enable_imu=True, cameras={})
        devices.append(robot)
        if not robot.is_connected:
            robot.connect()
        obs0 = robot.capture_observation()
        if "observation.imu_real" not in obs0:
            raise RuntimeError("IMU 未启用! 检查 enable_imu / TSND 设备 /dev/ttyACM0")
        imu0 = obs0["observation.imu_real"].numpy()
        if not np.isfinite(imu0).all():
            raise RuntimeError(f"IMU 首帧异常: {imu0}")
    except Exception as e:                                              # noqa: BLE001
        traceback.print_exc()
        p.write_text(json.dumps({"run_id": rid, "scope": scope, "started_at_utc": started_at_utc, "imu": [],
                                 "stopped": f"初始化失败: {type(e).__name__}: {e}", "traceback": traceback.format_exc()}))
        print("  ✗✗ 环境初始化失败；make_robot/connect 可能已有使能或归位副作用，不能据此声称机器未动。")
        return 3
    print(f"  IMU 自检 ✓  acc={np.round(imu0[:3], 2)}  gyro={np.round(imu0[3:], 3)}")

    # ---- 预移动与稳定（照搬 pilot）----
    q_now = (obs0["observation.state"].numpy()[:6] * np.pi / 180 / 1000).astype(np.float32)
    q_tgt = cmd7[0, :6].astype(np.float32)
    hold = np.concatenate([q_tgt, cmd7[0, 6:7]]).astype(np.float32)     # 稳定段与前置保持的同一目标
    n_pre = int(PREMOVE_S * FPS)
    settle = []
    try:
        for i in range(n_pre + 1):
            _guard(scope, robot)
            w = (1.0 - np.cos(np.pi * i / n_pre)) / 2.0
            act = np.concatenate([q_now + w * (q_tgt - q_now), cmd7[0, 6:7]]).astype(np.float32)
            robot.send_action(torch.from_numpy(act))
            time.sleep(DT)
        for _ in range(int(PRESETTLE_S * FPS)):
            _guard(scope, robot)
            robot.send_action(torch.from_numpy(hold.copy()))
            settle.append(robot.capture_observation()["observation.state"].numpy()[:6] * np.pi / 180 / 1000)
            time.sleep(DT)
        _guard(scope, robot)
        obs0 = robot.capture_observation()                              # 起始状态在预移动之后取
        _guard(scope, robot)
    except BaseException as e:                                          # noqa: BLE001
        return _abort_before_record(robot, torch, cmd7, scope, started_at_utc, outdir, run,
                                    f"{type(e).__name__}: {e}")
    q_pre = obs0["observation.state"].numpy()[:6] * np.pi / 180 / 1000
    win = np.asarray(settle[-int(BAND_WINDOW_S * FPS):], float)
    band = win.max(0) - win.min(0)
    band_bad = band > BAND_TOL
    vel_all = np.abs(np.diff(np.asarray(settle, float), axis=0) * FPS)
    from make_probe_b2 import reach_limits                              # pp2 回放：可达范围豁免 + 1°（同 pilot）
    klo, khi = reach_limits()
    unreach = np.maximum(np.maximum(klo - q_tgt, q_tgt - khi), 0.0)
    tol = unreach + np.radians(1.0)
    dev = np.abs(q_pre - q_tgt)
    _d = outdir / f"{'settle_fail_' if band_bad.any() else 'settle_'}{rid}_{int(time.time())}.json"
    _d.write_text(json.dumps({"run_id": rid, "settle_deg": np.degrees(np.asarray(settle, float)).tolist(),
                              "band_deg": np.degrees(band).tolist(), "band_tol_deg": np.degrees(BAND_TOL).tolist(),
                              "band_window_s": BAND_WINDOW_S, "settle_s": PRESETTLE_S,
                              "n_over_full_settle": int((vel_all > PRE_VEL_TOL).sum()),
                              "gate": "position-band (2026-08-24 修订冻结)", "passed": bool(not band_bad.any()),
                              "dev_deg": np.degrees(dev).tolist(), "tol_deg": np.degrees(tol).tolist()},
                             ensure_ascii=False, indent=1))
    print(f"  稳定段位置带 R_j = {np.round(np.degrees(band), 4)}°  逐关节 |偏差| {np.round(np.degrees(dev), 3)}°")
    try:
        _jb = int(np.argmax(np.where(band_bad, band - BAND_TOL, -np.inf)))
        assert not band_bad.any(), (f"稳定段位置带超限: j{_jb+1} R={np.degrees(band[_jb]):.3f}° "
                                    f"> {np.degrees(BAND_TOL[_jb]):.1f}° —— 立即停机, 不自动重试")
        j_bad = int(np.argmax(dev - tol))
        assert (dev <= tol).all(), (f"预移动未到位: j{j_bad+1} 偏差 {np.degrees(dev[j_bad]):.3f}° > 容差 "
                                    f"{np.degrees(tol[j_bad]):.3f}°")
    except AssertionError as e:
        return _abort_before_record(robot, torch, cmd7, scope, started_at_utc, outdir, run,
                                    f"{type(e).__name__}: {e}")
    print("  ✓ 预移动到位")

    # ---- 记录 ----
    rec = {"run_id": rid, "scope": scope, "started_at_utc": started_at_utc, "clock": "time.monotonic",
           "use_probe": use_probe,
           "cmd_hash": hashlib.sha256(np.ascontiguousarray(cmd7, np.float32).tobytes()).hexdigest()[:16],
           "t": [], "t_send": [], "t_send_span": [], "cmd": [], "imu": [], "state": [], "imu100": [],
           "can_hs": [], "can_q200": [],
           "prehold": {"s": PREHOLD_FRAMES / FPS, "frames": PREHOLD_FRAMES, "target": hold.tolist(),
                       "t": [], "t_send": [], "t_send_span": [], "cmd": []},
           "premove": {"s": PREMOVE_S, "settle_s": PRESETTLE_S, "q_parked": q_now.tolist(), "q_start": q_pre.tolist(),
                       "dev_deg": np.degrees(dev).tolist(), "tol_deg": np.degrees(tol).tolist()},
           "state_init": obs0["observation.state"].numpy().tolist()}
    rdr = getattr(robot, "imu_reader", None)
    if rdr is not None and hasattr(rdr, "start_recording"):
        rdr.start_recording()                                           # 100 Hz 独立队列（不碰控制侧 buffer）
        print("  100Hz 记录已开启")
    can_rec, can_raw = None, []
    if canhs:
        sys.path.insert(0, str(ROOT / "sim/step3_canobs"))
        from can_hs_recorder import CanHighSpdRecorder
        can_rec = CanHighSpdRecorder()
        if can_rec.start_recording() and can_rec.wait_first():          # 接口 DOWN 时 bind 也会成功，须等首帧
            print("  高速 CAN 反馈记录已开启 (0x251–0x256, 被动只读)")
        else:
            print(f"  ⚠ 高速 CAN 反馈未开启: {can_rec.err} —— 本条不能用于 CAN 可观测性分析")

    def drain():
        if rdr is not None and hasattr(rdr, "drain_recording"):
            for ht, ms, ac, gy in rdr.drain_recording():                # 逐帧排空
                rec["imu100"].append([float(ht), int(ms), *map(float, ac), *map(float, gy)])
        if can_rec is not None:
            can_raw.extend(can_rec.drain_recording())                   # 只搬原始字节，不解码

    def step(act, sink, label):
        """下发一帧并读数；返回 None 表示正常，否则返回停止原因。"""
        t_start = time.perf_counter()
        robot.send_action(torch.from_numpy(act))
        span = getattr(robot.arm, "last_jointctrl_t", None)             # JointCtrl 真正写总线的时刻
        t_send = 0.5 * (span[0] + span[1]) if span else time.monotonic()
        obs = robot.capture_observation()
        _guard(scope, robot)
        imu = obs.get("observation.imu_real")
        if imu is None or not np.isfinite(imu.numpy()).all():
            return f"{label}: IMU 异常"
        sink["t"].append(time.monotonic())
        sink["t_send"].append(t_send)
        sink["t_send_span"].append(float(span[1] - span[0]) if span else float("nan"))
        sink["cmd"].append(act.tolist())
        if sink is rec:
            rec["imu"].append(imu.numpy().tolist())
            rec["state"].append(obs["observation.state"].numpy().tolist())
        drain()
        dt = time.perf_counter() - t_start
        if dt < DT:
            time.sleep(DT - dt)
        return None

    stopped = None
    try:
        for k in range(PREHOLD_FRAMES):                                 # 0.2 s 前置保持：目标不变，单独记录
            _guard(scope, robot)
            stopped = step(hold.copy(), rec["prehold"], f"前置保持第 {k} 帧")
            if stopped:
                break
        if stopped is None:
            for i in range(len(cmd7)):
                _guard(scope, robot)
                stopped = step(np.concatenate([total[i], cmd7[i, 6:7]]).astype(np.float32), rec, f"帧 {i}")
                if stopped:
                    break
    except KeyboardInterrupt:
        stopped = "用户中断"
    except Exception as e:                                              # noqa: BLE001
        stopped = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()
    finally:
        try:
            if rdr is not None and hasattr(rdr, "stop_recording"):
                try:
                    drain()
                finally:
                    rdr.stop_recording()
            if can_rec is not None:
                try:
                    can_raw.extend(can_rec.drain_recording())
                finally:
                    can_rec.stop_recording()
                from can_hs_recorder import decode as _candec              # 停止采集后再解码；失败不跳过归位
                rec["can_hs"], rec["can_q200"] = _candec(can_raw)
                rec["can_hs_meta"] = can_rec.meta(getattr(robot.arm, "piper", None))
        except Exception as e:                                          # noqa: BLE001
            stopped = stopped or f"capture finalization: {type(e).__name__}: {e}"
            rec["finalization_error"] = traceback.format_exc()
        rec["stopped"] = stopped
        if p.exists():                                                  # 移开，不覆盖
            p.rename(p.with_name(f"{p.stem}.superseded_{int(time.time())}.json"))
        p.write_text(json.dumps(rec))                                   # 先落盘，再动机器
        print(f"\n采集 {len(rec['imu'])} 帧(20Hz) + {len(rec['imu100'])} 样本(100Hz) → {p.name}"
              + (f"  ⚠ 提前终止: {stopped}" if stopped else "  ✓"))
        _finish_return(robot, torch, cmd7, scope, rec)
        p.write_text(json.dumps(rec))
    return 0 if rec["stopped"] is None else 1
