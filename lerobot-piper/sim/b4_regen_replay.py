"""重新生成网络 ② 的在线回放（设计 `reports/PLAN_b4_regen.md` §4.5–4.6；实现计划 `reports/PLAN_IMPL_b4_regen.md` 任务 6）。
每帧用当时已到达的实测和冻结的网络生成这一帧的指令；shadow 模式照样计算、记录网络的输出，但发 Shift 指令。
Shift 指 shift84（提前 84 ms，用户 11-12 的新基线）：影子模式发它，疑似接触段（窗为 0）和夹爪也用它。

  python -u sim/b4_regen_replay.py dry --tid p44 --source outputs/b4/teacher/sessions/20261104f/replay_p44_b01_r1.json \\
      --net outputs/b4/regen/frozen.json      # 不连机械臂：在线算出的与离线逐位核对，报告每帧计算耗时
  python -u sim/b4_regen_replay.py run --session <会话目录> --tid p44 --rid p44_tshadow1 --mode shadow \\
      --net outputs/b4/regen/frozen.json      # 真机（用户在场）
  dry、run 加 --fb：在网络的指令上加 B 的写死的比例反馈（设计 `reports/PLAN_b3.md` §9，常量 FB_GAIN、FB_FC、FB_N）。

每帧：t0 = 本帧开始，截止 = t0 − CUT_S → 截止以前到达的实测，实测中断就停（两种模式）→ 输入单元、网络、regen_command、
step_check → regen 模式检查不过：这一帧不发，停 → 发出（regen：网络切换后的指令；shadow：s7[k]）→ 记录本帧。
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "step3_canobs"))
import b4_common as C  # noqa: E402
import b4_regen as R  # noqa: E402
import ideal_motion as IM  # noqa: E402
import b4_teacher as B  # noqa: E402
import can_hs_recorder  # noqa: E402
import repeat_reversal as RR  # noqa: E402

import lerobot.common.policies.act.b4_online as O  # noqa: E402

PREMOVE_S, PRESETTLE_S, BAND_WINDOW_S = 3.0, 5.0, 2.0              # 预移动判据，同 replay_probe._run_impl
BAND_TOL = np.radians([0.5, 0.5, 0.5, 0.8, 0.8, 0.8])
PRE_REC_S = 0.5                    # 记录开起来以后在 s7[0] 再停这么久：第 0 帧的实测窗口（0.4 s）要填满
DRY_P99_MS = 10.0                  # dry 的每帧计算耗时 p99 达到它就不许上机
FB_GAIN, FB_FC, FB_N = 0.7, 1.0, 10   # B 的写死的比例反馈（设计 PLAN_b3.md §9）：增益、一阶低通截止频率（Hz）、取截止前多少个编码器采样（50 ms）


class Recorded:
    """一遍回放记录当实测源（dry、测试）：rows(cutoff) → 到达时刻早于截止时刻的 can_q200、imu100 行。"""

    def __init__(self, h: dict):
        self.q, self.imu = np.asarray(h["can_q200"], float), np.asarray(h["imu100"], float)
        self.tq = self.q[:, 0] - h["can_hs_meta"]["clock"]["offset_realtime_minus_monotonic_s"]

    def rows(self, cutoff: float):
        return self.q[:np.searchsorted(self.tq, cutoff)], self.imu[:np.searchsorted(self.imu[:, 0], cutoff)]


class Live:
    """真机实测源：rows(cutoff) 先取出新到的 IMU 100 Hz 样本和 CAN 原始帧（CAN 当场 decode），再返回到达时刻早于截止时刻、
    且不早于截止时刻 − KEEP_S 的行（窗口函数只用 [截止 − 0.45 s, 截止)，结果与传全部行逐位相同）。全部样本另存，回放后写进记录。"""
    KEEP_S = 1.0

    def __init__(self, robot, can_rec):
        self.imu_reader, self.can, self.off = robot.imu_reader, can_rec, can_rec.off0
        self.imu100, self.can_raw, self.q = [], [], []
        self.i0 = self.q0 = 0

    def rows(self, cutoff: float):
        self.imu100 += [[float(ht), int(ms), *map(float, ac), *map(float, gy)] for ht, ms, ac, gy in self.imu_reader.drain_recording()]
        raw = self.can.drain_recording()
        self.can_raw += raw
        self.q += can_hs_recorder.decode(raw)[1]
        old = cutoff - self.KEEP_S
        while self.i0 < len(self.imu100) and self.imu100[self.i0][0] < old:
            self.i0 += 1
        while self.q0 < len(self.q) and self.q[self.q0][0] - self.off < old:
            self.q0 += 1
        q = np.asarray(self.q[self.q0:], float).reshape(-1, 4)
        imu = np.asarray(self.imu100[self.i0:], float).reshape(-1, 8)
        return q[q[:, 0] - self.off < cutoff], imu[imu[:, 0] < cutoff]


def loop(n, base, log, fn, calib, offset, lo, hi, source, mode, send, clock, fb=None) -> dict:
    """逐帧生成并发出指令（见文件头）。mode = "regen" | "shadow"；send(u7) → t_send；clock() → 本帧开始时刻（真机里它负责等到这一帧）。
    已发出的指令用自己前几帧最终发出的；第 0 帧之前机械臂停在 Shift 第一帧。Ctrl-C 算停机。
    fb = 理想关节角的网格 (N, 6)（`ideal_motion.ideal`）时加 B 的写死的比例反馈：截止前 FB_N 个编码器采样的「理想 − 实测」（乘窗）取平均，
    过 FB_FC 的一阶低通，乘 FB_GAIN，加到会动关节窗 > 0 的指令上；regen 模式才真正加，shadow 模式只记录。
    返回 {"cmd", "t_send", "t_frame", "cutoff", "net_first", "net_u", "fb_c", "check", "compute_ms", "stage_ms", "stopped"}；
    net_u 是加反馈之前的指令，fb_c 是五个会动关节的补正（没有反馈时为 0）；stage_ms 是每帧四段的耗时：取实测、算窗口、输入单元与网络、切换与检查。"""
    a7, s7, w = (np.asarray(base[x], float) for x in ("a7", "s7", "w"))
    a_pi, (_, ens) = np.asarray(log["a_pi"], float), O.te_states(log["chunk"])
    sent = s7[:, :6].copy()
    rho, lp = np.exp(-2 * np.pi * FB_FC * O.DT), np.zeros(5)
    rec = {key: [] for key in ("cmd", "t_send", "t_frame", "cutoff", "net_first", "net_u", "fb_c", "check", "compute_ms", "stage_ms")}
    rec["stopped"] = None
    try:
        for k in range(n):
            t0 = clock()
            c0, cut = time.perf_counter(), t0 - O.CUT_S
            q200, imu = source.rows(cut)
            c1 = time.perf_counter()
            enc, age_q = O.enc_window(q200, offset, cut)
            imw, age_i = O.imu_window(imu, calib, cut)
            c2 = time.perf_counter()
            if not (age_q <= O.STALE_ENC_S and age_i <= O.STALE_IMU_S):       # 缺一组时 age 是 inf，也算中断
                rec["stopped"] = f"实测中断：第 {k} 帧，编码器最新采样早于截止 {age_q * 1e3:.0f} ms、IMU {age_i * 1e3:.0f} ms"
                break
            act_past, act_seg = O.regen_act(a7, a_pi, ens, k, RR.HOLD)
            first = np.asarray(fn(*O.regen_tokens(act_past, sent[np.clip(np.arange(k - O.REGEN_P, k), 0, None)], enc, imw, act_seg))[0])
            c3 = time.perf_counter()
            c = np.zeros(5)
            if fb is not None:
                e = O.ideal_err(enc, cut, np.asarray(rec["t_send"]), fb, w)
                lp = rho * lp + (1 - rho) * e[-FB_N:, list(O.ACTIVE)].mean(0)
                c = np.where(w[k] > 0, FB_GAIN * lp, 0.0)
            net_u = O.regen_command(first, s7[k], w[k])
            u7 = net_u.copy()
            u7[list(O.ACTIVE)] += c.astype(np.float32)
            why = O.step_check(u7[:6], sent[k - 1] if k else s7[0, :6], a7[k, :6], lo, hi)
            c4 = time.perf_counter()
            ms = (c4 - c0) * 1e3
            if why and mode == "regen":
                rec["stopped"] = f"第 {k} 帧：{why}"
                break
            out = u7 if mode == "regen" else s7[k].astype(np.float32)
            t_send = send(out)
            sent[k] = out[:6]
            for key, v in (("cmd", out.tolist()), ("t_send", t_send), ("t_frame", t0), ("cutoff", cut), ("net_first", first.tolist()),
                           ("net_u", net_u.tolist()), ("fb_c", c.tolist()), ("check", why), ("compute_ms", ms),
                           ("stage_ms", [round((b - a) * 1e3, 3) for a, b in ((c0, c1), (c1, c2), (c2, c3), (c3, c4))])):
                rec[key].append(v)
    except KeyboardInterrupt:
        rec["stopped"] = "用户中断"
    return rec


def write_outputs(sess: Path, rid: str, rec: dict, mode: str, net_sha: str) -> tuple[Path, Path]:
    """回放记录 replay_{rid}_r1.json（rec 的全部字段，另加 mode、net_sha、cmd_hash、seqfile）和回放序列 seq_{rid}.npz（cmd7 = 实际发出的，
    da = 0，source 记 rid、label、cmd_hash、net_sha、mode）。两边的 cmd_hash 一致，B.replay_result 可以直接读。"""
    sess = Path(sess)
    sess.mkdir(parents=True, exist_ok=True)
    cmd7 = np.asarray(rec["cmd"], np.float32).reshape(-1, 7)
    cmd_hash = hashlib.sha256(np.ascontiguousarray(cmd7).tobytes()).hexdigest()[:16]
    p_rec, p_seq = sess / f"replay_{rid}_r1.json", sess / f"seq_{rid}.npz"
    src = {"rid": rid, "label": f"② {mode}", "cmd_hash": cmd_hash, "net_sha": net_sha, "mode": mode}
    np.savez(p_seq, cmd7=cmd7, da=np.zeros((len(cmd7), 6), np.float32), source=json.dumps(src, ensure_ascii=False))
    p_rec.write_text(json.dumps({**rec, "mode": mode, "net_sha": net_sha, "cmd_hash": cmd_hash, "seqfile": str(p_seq)}))
    return p_rec, p_seq


def dry(tid: str, source_json: Path, net_path: str, fb: bool = False) -> dict:
    """不连机械臂：用录好的一遍回放当实测源，shadow 模式跑 loop；网络每帧的输出要和离线的 R.first_frames（同一个推理函数）逐位一致，
    并报告每帧计算耗时（p99 ≥ DRY_P99_MS 就 ⛔，不许上机）。按真机节奏跑：每帧至少 DT，帧间 CPU 和真机上一样会空闲、降频
    （连续算会低估真机耗时：第一次影子运行 p50 17 ms，当时的连续 dry 只有 2.4 ms）。推理同真机，用 GPU + CUDA graph。"""
    import torch
    torch.set_num_threads(1)
    h = json.loads(Path(source_json).read_text())
    base, log = {**B.load_seq(tid), "s7": R.shift_cmd(tid, IM.LEAD_S)}, R._log(tid)       # Shift = shift84
    fn, sha = O.load_regen(net_path, device="cuda")
    calib = R._calib(Path(source_json).parent)
    bd = json.loads((R.REGEN / "bounds.json").read_text())
    ts, k, nxt = h["t_send"], [-1], [0.0]

    def clock():                                          # 节奏用真实时间，实测窗口用记录里的发令时刻
        time.sleep(max(0.0, nxt[0] - time.monotonic()))
        nxt[0] = time.monotonic() + O.DT
        k[0] += 1
        return ts[k[0]]

    res = loop(len(base["a7"]), base, log, fn, calib, h["can_hs_meta"]["clock"]["offset_realtime_minus_monotonic_s"],
               np.asarray(bd["lo"]), np.asarray(bd["hi"]), Recorded(h), "shadow", lambda u7: ts[k[0]], clock,
               fb=IM.ideal(base["a7"]) if fb else None)
    off = R.first_frames(fn, h, base, log, calib)
    same = res["stopped"] is None and np.array_equal(np.asarray(res["net_first"]), off)
    p50, p99 = np.percentile(res["compute_ms"], [50, 99]) if res["compute_ms"] else (np.nan, np.nan)
    mx = max(res["compute_ms"], default=np.nan)
    fast = p99 < DRY_P99_MS
    print(f"网络 {sha}；{len(res['net_first'])}/{len(off)} 帧" + (f"，停在：{res['stopped']}" if res["stopped"] else ""))
    print("✓ 在线算出的网络输出与离线逐位一致" if same else "⛔ 在线与离线不一致")
    print(f"每帧计算耗时 p50 {p50:.2f} / p99 {p99:.2f} / 最大 {mx:.2f} ms" + ("" if fast else f"  ⛔ p99 ≥ {DRY_P99_MS:.0f} ms，不许上机"))
    st = np.asarray(res["stage_ms"]).reshape(-1, 4)
    print("四段耗时 p50 / p99（ms）：" + "，".join(f"{nm} {np.percentile(st[:, i], 50):.2f} / {np.percentile(st[:, i], 99):.2f}"
                                          for i, nm in enumerate(("取实测", "算窗口", "输入与网络", "切换与检查"))))
    print(f"影子模式下逐帧检查会触发的帧：{sum(c is not None for c in res['check'])}")
    if fb:
        print(f"反馈补正最大 {1e3 * np.abs(res['fb_c']).max():.2f} mrad（开环：录好的机械臂不响应补正，只核对接线）")
    return {"same": bool(same), "p50": float(p50), "p99": float(p99), "max": float(mx), "ok": bool(same and fast)}


def premove(robot, torch, cmd0: np.ndarray, obs0) -> tuple[dict, object]:
    """预移动到 cmd0（判据照抄 replay_probe._run_impl）：3 s 余弦插值，5 s 静置；最后 2 s 每个关节的位置带 ≤ BAND_TOL，
    偏差 ≤ 可达范围豁免 + 1°。不满足就抛 RuntimeError（停机，不自动重试）。返回 (记进回放记录的信息, 预移动后的观测)。"""
    from make_probe_b2 import reach_limits
    read = lambda obs: obs["observation.state"].numpy()[:6] * np.pi / 180 / 1000   # noqa: E731  0.001° → rad
    q0, tgt, grip = read(obs0).astype(np.float32), np.asarray(cmd0[:6], np.float32), np.asarray(cmd0[6:7], np.float32)
    n = int(PREMOVE_S / O.DT)
    print(f"  预移动 {PREMOVE_S:.0f} s：最大位移 {np.degrees(np.abs(tgt - q0)).max():.2f}°")
    for i in range(n + 1):
        w = (1.0 - np.cos(np.pi * i / n)) / 2.0
        robot.send_action(torch.from_numpy(np.concatenate([q0 + w * (tgt - q0), grip]).astype(np.float32)))
        time.sleep(O.DT)
    settle = []
    for _ in range(int(PRESETTLE_S / O.DT)):
        robot.send_action(torch.from_numpy(np.concatenate([tgt, grip])))
        settle.append(read(robot.capture_observation()))
        time.sleep(O.DT)
    obs = robot.capture_observation()
    win = np.asarray(settle[-int(BAND_WINDOW_S / O.DT):], float)
    band, dev = win.max(0) - win.min(0), np.abs(read(obs) - tgt)
    klo, khi = reach_limits()
    tol = np.maximum(np.maximum(klo - tgt, tgt - khi), 0.0) + np.radians(1.0)
    info = {"s": PREMOVE_S, "settle_s": PRESETTLE_S, "q_parked": q0.tolist(), "q_start": read(obs).tolist(),
            "band_deg": np.degrees(band).tolist(), "dev_deg": np.degrees(dev).tolist(), "tol_deg": np.degrees(tol).tolist()}
    print(f"  位置带 {np.round(np.degrees(band), 3)}°（≤ {np.round(np.degrees(BAND_TOL), 1)}°）；偏差 {np.round(np.degrees(dev), 3)}°（≤ {np.round(np.degrees(tol), 3)}°）")
    if (band > BAND_TOL).any():
        raise RuntimeError(f"稳定段位置带超限 {np.round(np.degrees(band), 3)}° —— 停机，不自动重试")
    if (dev > tol).any():
        raise RuntimeError(f"预移动未到位：偏差 {np.round(np.degrees(dev), 3)}° —— 停机排查")
    return info, obs


def run(sess: Path, tid: str, rid: str, mode: str, net_path: str, fb: bool = False) -> int:
    """真机跑一遍（设计 §4.6；用户在场）：IMU 自检 → 预移动到 s7[0] → 开 IMU 100 Hz 与高速 CAN 记录（CAN 要过 wait_first，否则不跑）
    并在原位再停 PRE_REC_S 秒 → loop → 写回放记录与回放序列 → 分段归位。异常照 shift_replay.guarded_run：还没有回放记录就落一份
    stopped 记录并归位；已有就只补原因；最后把 is_connected 置 False（解除析构时的无插值归位）。返回 0 = 跑完、没有停。"""
    import torch
    import replay_probe as RP
    from lerobot.common.robot_devices.robots.utils import make_robot
    sess, p = Path(sess), Path(sess) / f"replay_{rid}_r1.json"
    if p.exists():
        raise SystemExit(f"⛔ {p} 已存在：换一个编号，不覆盖已有的回放")
    B.check_session(sess, C.CALIB)
    sess.mkdir(parents=True, exist_ok=True)
    if not (sess / "session.json").exists():
        (sess / "session.json").write_text(json.dumps({"calib_sha": B._sha(C.CALIB), "rule": B.RULE, "qc_max": B.QC_MAX}))
    base, log = {**B.load_seq(tid), "s7": R.shift_cmd(tid, IM.LEAD_S)}, R._log(tid)       # Shift = shift84
    fbx = IM.ideal(base["a7"]) if fb else None                       # 反馈用的理想：连机械臂之前算好（每条约 3 s）
    fn, net_sha = O.load_regen(net_path, device="cuda")              # 没有 GPU 就在这里报错，机械臂还没连
    calib = B.session_ctx(sess).calib
    bd = json.loads((R.REGEN / "bounds.json").read_text())
    s7 = np.asarray(base["s7"], float)
    grip = float(np.clip(s7[0, 6], 0, 0.035))
    torch.set_num_threads(1)
    sys.setswitchinterval(1e-4)       # 机械臂 SDK、IMU、CAN 的后台线程和推理抢 GIL：默认 5 ms 时，影子运行每帧计算 p50 约 17 ms（dry 2.4 ms）
    robot = None
    try:
        robot = make_robot("piper", inference_time=True, enable_imu=True, cameras={})
        if not robot.is_connected:
            robot.connect()
        home = np.asarray(robot.arm.init_joint_position[:6])            # connect() 末尾 run_calibration 已无插值地发出「到零位」：
        for _ in range(int(5.0 / O.DT)):                                 # 等它到了再读起点，否则预移动从停机位插值、先把臂拉回去
            obs0 = robot.capture_observation()
            if np.abs(obs0["observation.state"].numpy()[:6] * np.pi / 180 / 1000 - home).max() < np.radians(1.0):
                break
            time.sleep(O.DT)
        else:
            raise RuntimeError("connect 之后 5 s 内没有到达零位 —— 停机检查")
        imu0 = obs0.get("observation.imu_real")
        if imu0 is None or not np.isfinite(imu0.numpy()).all():
            raise RuntimeError(f"IMU 自检失败：{imu0}")
        pre, obs = premove(robot, torch, s7[0], obs0)
        rdr, can = robot.imu_reader, can_hs_recorder.CanHighSpdRecorder()
        rdr.start_recording()
        try:
            if not (can.start_recording() and can.wait_first()):
                raise RuntimeError(f"高速 CAN 记录没开起来（{can.err}），没有开始回放")
            live, extra, nxt = Live(robot, can), {"t": [], "t_send_span": [], "imu": [], "state": []}, [0.0]
            for _ in range(int(PRE_REC_S / O.DT)):
                robot.send_action(torch.from_numpy(s7[0].astype(np.float32)))
                time.sleep(O.DT)

            def send(u7):
                robot.send_action(torch.from_numpy(u7))
                span = getattr(robot.arm, "last_jointctrl_t", None)      # JointCtrl 写总线前后的 monotonic，取中点（同 replay_probe）
                o = robot.capture_observation()
                extra["t"].append(time.monotonic())
                extra["t_send_span"].append(float(span[1] - span[0]) if span else float("nan"))
                extra["imu"].append(o["observation.imu_real"].numpy().tolist())
                extra["state"].append(o["observation.state"].numpy().tolist())
                return 0.5 * (span[0] + span[1]) if span else time.monotonic()

            def clock():                                                  # 等到这一帧开始（每帧至少 DT），返回开始时刻
                time.sleep(max(0.0, nxt[0] - time.monotonic()))
                t0 = time.monotonic()
                nxt[0] = t0 + O.DT
                return t0

            print(f"  {rid}：{mode} 模式，{len(s7)} 帧，网络 {net_sha}" + (f"，反馈：开（增益 {FB_GAIN}、{FB_FC} Hz）" if fb else ""))
            res = loop(len(s7), base, log, fn, calib, live.off, np.asarray(bd["lo"]), np.asarray(bd["hi"]), live, mode, send, clock, fb=fbx)
            live.rows(np.inf)
        finally:
            rdr.stop_recording()
            can.stop_recording()
        live.can_raw += can.drain_recording()
        hs, q200 = can_hs_recorder.decode(live.can_raw)
        rec = {"tid": tid, "rid": rid, "clock": "time.monotonic", **res, **extra, "imu100": live.imu100, "can_hs": hs, "can_q200": q200,
               "fb": {"gain": FB_GAIN, "fc": FB_FC, "n": FB_N} if fb else None,
               "can_hs_meta": can.meta(getattr(robot.arm, "piper", None)), "premove": pre,
               "state_init": obs["observation.state"].numpy().tolist()}
        write_outputs(sess, rid, rec, mode, net_sha)
        print(f"  {len(res['cmd'])}/{len(s7)} 帧 → {p}" + (f"  ⚠ 停机：{res['stopped']}" if res["stopped"] else "  ✓"))
        if not RP.staged_return(robot, torch, grip):                    # 先落盘，再归位
            rec["return_blocked"] = True
            p.write_text(json.dumps(json.loads(p.read_text()) | {"return_blocked": True}))
            print("  ⚠ 归位受阻已记入回放记录；请人工检查后再跑下一遍。")
        return 0 if res["stopped"] is None else 1
    except BaseException as e:
        reason = f"{type(e).__name__}: {e}"
        print(f"\n  ✗ 这一遍异常退出：{reason}")
        if p.exists():                                                   # 落盘之后（归位时）出的事：只补原因
            h = json.loads(p.read_text())
            p.write_text(json.dumps(h | {"stopped": h.get("stopped") or reason}))
            return 1
        h = {"tid": tid, "rid": rid, "mode": mode, "net_sha": net_sha, "stopped": reason, "traceback": traceback.format_exc()}
        p.write_text(json.dumps(h))
        if robot is not None and robot.is_connected:
            try:
                if not RP.staged_return(robot, torch, grip):
                    h["return_blocked"] = True
            except BaseException as e2:                                  # 归位中再按 Ctrl-C 也走到这里
                h["return_error"] = f"{type(e2).__name__}: {e2}"
            p.write_text(json.dumps(h))
        return 1
    finally:
        if robot is not None:
            robot.is_connected = False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dry", help="不连机械臂：与离线逐位核对，报告每帧计算耗时")
    r = sub.add_parser("run", help="真机跑一遍（用户在场）")
    for name in ("--tid", "--net"):
        d.add_argument(name, required=True)
        r.add_argument(name, required=True)
    for p in (d, r):
        p.add_argument("--fb", action="store_true", help="加 B 的写死的比例反馈（设计 PLAN_b3.md §9）")
    d.add_argument("--source", required=True, help="当实测源的那一遍回放记录 replay_*_r1.json")
    r.add_argument("--session", required=True)
    r.add_argument("--rid", required=True, help="回放编号，例如 p44_tshadow1")
    r.add_argument("--mode", required=True, choices=("shadow", "regen"))
    a = ap.parse_args(argv)
    if a.cmd == "dry":
        return 0 if dry(a.tid, Path(a.source), a.net, a.fb)["ok"] else 1
    return run(Path(a.session), a.tid, a.rid, a.mode, a.net, a.fb)


if __name__ == "__main__":
    gc.disable()                      # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）
    code = main()
    sys.stdout.flush()
    os._exit(code)
