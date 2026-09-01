#!/usr/bin/env python3
"""**第 1 步：静止 10 秒，确认高速反馈的实际 Hz、缺样率、时间戳稳定性。**

这一步**不改 G、不动机械臂、不加硬件**，也**不使能**机械臂 —— 纯被动监听。

为什么用被动监听 CAN 而不是轮询 SDK
--------------------------------------
`GetArmHighSpdInfoMsgs()` 的六个电机是**六条独立 CAN 帧**(0x251–0x256)，但
SDK 把它们写进**同一个结构体**，`time_stamp` 被最后到达的那一帧覆盖
(`piper_interface_v2.py:1836` 起)。所以按固定频率轮询这个接口，等于对六路
异步流做一次抽取 —— **正是我们在 IMU 上踩过的混叠坑**
(`step2_ceiling.py`: TSND 100 Hz 被 20 Hz 无抗混叠间引)。

被动监听拿到的是每一帧自己的内核收包时刻，没有抽取，也不与控制回路争用。

时钟口径（**重要**）
--------------------
CAN 帧时间戳 = `rx.timestamp` = 内核 `SO_TIMESTAMPNS` = **CLOCK_REALTIME**。
而整条管线(`replay_probe.py` 的 `t`/`t_send`、TSND 的 `host_t`)用的是
**CLOCK_MONOTONIC**。两个时钟不同，本脚本记录二者偏移及其在 10 秒内的漂移，
后续对齐必须用它。

effort 口径（**重要**）
-----------------------
`cal_effort()` 对 J1–J3 乘 1.18125、J4–J6 乘 0.95844，**与固件版本无关**。
但旧固件(1.8-2 及以前) J1–J3 另有倍率规则。因此本阶段：
    **保留原始 current，effort 只当激励代理，不称真实关节力矩。**

用法（机械臂需上电且 can0 已 up；本脚本自己不使能、不发指令）:
    python -u sim/step3_canobs/probe_highspd_rate.py
    python -u sim/step3_canobs/probe_highspd_rate.py --seconds 10 --can can0
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/step3_canobs"
HS_IDS = {0x251: 1, 0x252: 2, 0x253: 3, 0x254: 4, 0x255: 5, 0x256: 6}
JOINT_IDS = {0x2A5: "joint_12", 0x2A6: "joint_34", 0x2A7: "joint_56"}   # 低速关节角反馈


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--can", default="can0")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--tag", default="static")
    a = ap.parse_args()

    import can
    from piper_sdk.protocol.protocol_v2.piper_protocol_v2 import C_PiperParserV2
    from piper_sdk.piper_msgs.msg_v2.arm_messages import PiperMessage

    # SocketCAN 允许 bind 到 state DOWN 的接口而不报错, 然后静默收零帧。
    # 先看一眼 operstate, 免得白等 10 秒才发现接口没起来。
    st = Path(f"/sys/class/net/{a.can}/operstate")
    if st.exists() and st.read_text().strip() in ("down", "unknown"):
        try:                       # 接口 down 时读 carrier 会抛 EINVAL, 不是"没有载波"
            up = Path(f"/sys/class/net/{a.can}/carrier").read_text().strip() == "1"
        except OSError:
            up = False
        if not up:
            print(f"**{a.can} 未启用** (operstate={st.read_text().strip()})。")
            print(f"  请先: sudo ip link set {a.can} up type can bitrate 1000000")
            return 2

    # ---- 时钟偏移: REALTIME(CAN) − MONOTONIC(管线) ----------------------
    def clkpair():
        m0 = time.monotonic(); r = time.time(); m1 = time.monotonic()
        return r - 0.5 * (m0 + m1), m1 - m0
    off0, unc0 = clkpair()

    print(f"被动监听 {a.can}，{a.seconds:.0f} 秒。**不使能、不发指令。**")
    print(f"时钟偏移 REALTIME−MONOTONIC = {off0:.6f} s (读取不确定度 {unc0*1e6:.0f} µs)\n")

    parser = C_PiperParserV2()
    bus = can.interface.Bus(channel=a.can, interface="socketcan")
    rows, nother = [], 0
    t_end = time.monotonic() + a.seconds
    try:
        while time.monotonic() < t_end:
            rx = bus.recv(timeout=0.5)
            if rx is None:
                continue
            cid = rx.arbitration_id
            if cid in HS_IDS:
                msg = PiperMessage()
                ok = parser.DecodeMessage(rx, msg)
                fb = getattr(msg, f"arm_high_spd_feedback_{HS_IDS[cid]}", None)
                rows.append((float(rx.timestamp), HS_IDS[cid],
                             int(getattr(fb, "motor_speed", 0)) if ok else 0,
                             int(getattr(fb, "current", 0)) if ok else 0,
                             int(getattr(fb, "pos", 0)) if ok else 0))
            else:
                nother += 1
    finally:
        bus.shutdown()
    off1, _ = clkpair()

    if not rows:
        print("**没有收到任何 0x251–0x256 高速帧。**")
        print(f"  同期其它 CAN 帧 {nother} 条 —— "
              + ("总线有流量，说明高速反馈默认不发，需要确认固件/使能状态。"
                 if nother else "总线完全静默，检查机械臂是否上电、can0 是否 up。"))
        return 2

    R = np.array(rows, float)
    T = R[:, 0] - R[0, 0]
    span = float(T[-1] - T[0]) if len(T) > 1 else 0.0
    print(f"收到高速帧 {len(R)} 条 / {span:.2f} s   其它帧 {nother} 条")
    print(f"时钟漂移(10s 内) {(off1 - off0)*1e6:+.0f} µs\n")

    print(f"  {'关节':<6}{'帧数':>7}{'实际 Hz':>10}{'间隔中位':>10}{'p99':>9}"
          f"{'最大':>9}{'疑似缺样':>10}{'speed 静止 σ':>14}{'current 静止 σ':>15}")
    summary = {}
    for j in range(1, 7):
        m = R[:, 1] == j
        n = int(m.sum())
        if n < 3:
            print(f"  j{j:<5}{n:>7}      —— 帧太少")
            summary[f"j{j}"] = {"n": n}
            continue
        tj = R[m, 0]
        d = np.diff(tj)
        med = float(np.median(d))
        hz = 1.0 / med if med > 0 else float("nan")
        # 缺样 = 间隔 ≥ 1.5×中位（按中位间隔折算成"少收了几帧"）
        miss = int(np.round(np.clip(d / med - 1.0, 0, None)).sum())
        sp_sd = float(np.std(R[m, 2])) * 1e-3       # 0.001 rad/s → rad/s
        cu_sd = float(np.std(R[m, 3])) * 1e-3       # 0.001 A → A
        print(f"  j{j:<5}{n:>7}{hz:>10.1f}{med*1e3:>9.2f}m{np.percentile(d,99)*1e3:>8.2f}m"
              f"{d.max()*1e3:>8.2f}m{miss:>10d}{sp_sd:>14.5f}{cu_sd:>15.5f}")
        summary[f"j{j}"] = {"n": n, "hz": hz, "dt_med_ms": med * 1e3,
                            "dt_p99_ms": float(np.percentile(d, 99)) * 1e3,
                            "dt_max_ms": float(d.max()) * 1e3, "miss_est": miss,
                            "speed_sd_rad_s": sp_sd, "current_sd_A": cu_sd}

    hzs = [v["hz"] for v in summary.values() if "hz" in v]
    print(f"\n  单关节实际 Hz: 中位 {np.median(hzs):.1f}   "
          f"六路合计 {len(R)/max(span,1e-9):.1f} 帧/s")
    ny = np.median(hzs) / 2
    print(f"  ⇒ 单关节 Nyquist = {ny:.1f} Hz。"
          + (f"**覆盖 3–9 Hz 评价带**（20 Hz 编码器的 Nyquist 只有 10 Hz 且无抗混叠）。"
             if ny >= 9 else f"**不足以覆盖 9 Hz** —— 这一路观测量本身就受限。"))

    # 六路之间的帧错位: 同一"轮"里 j1..j6 到达时刻的散布
    if all("hz" in summary[f"j{j}"] for j in range(1, 7)):
        base = R[R[:, 1] == 1, 0]
        skew = []
        for j in range(2, 7):
            tj = R[R[:, 1] == j, 0]
            idx = np.searchsorted(tj, base)
            idx = np.clip(idx, 0, len(tj) - 1)
            skew.append(np.median(np.abs(tj[idx] - base)))
        print(f"  六路帧错位(相对 j1 的中位绝对差): "
              f"{', '.join(f'j{j}={s*1e3:.2f}ms' for j, s in zip(range(2,7), skew))}")
        print(f"    ⇒ 对齐到统一时基时必须逐关节处理这个错位，不能当同时采样。")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"highspd_rate_{a.tag}.json"
    p.write_text(json.dumps({
        "tag": a.tag, "seconds": a.seconds, "can": a.can,
        "n_highspd": len(R), "n_other": nother, "span_s": span,
        "clock": {"can": "CLOCK_REALTIME (SO_TIMESTAMPNS)",
                  "pipeline": "CLOCK_MONOTONIC",
                  "offset_realtime_minus_monotonic_s": off0,
                  "offset_read_uncertainty_s": unc0,
                  "drift_over_window_s": off1 - off0},
        "effort_caliber": "保留原始 current；effort 仅作激励代理，不称真实关节力矩",
        "per_joint": summary,
        "frames": R.tolist(),
    }))
    print(f"\n  → {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
