#!/usr/bin/env python3
"""**单点探针：j5 后加柔性铰**（ω=8.485 Hz, ζ=0.4，取自旧网格的 j5 最优点）。

**这是零机时的结构筛查，不用于选择或冻结任何参数。** 铰的 (ω, ζ) **事前固定**，
不搜索、不调。

为什么做底座对底座
------------------
`r_q` / `r_s` 是对着 **G-q-v1 底座**拟合的；换底座后重新拟合就等于**选参**。
故本探针只比 **底座 vs 底座**（同一事件网格、同一 θ*、两侧都不加残差层），
G-2 的位置只作背景参照。

四项判读（用户 2026-09-02 指定）
--------------------------------
1. 电机侧 j5 编码器 3–9 Hz 误差
2. 柔性铰的相对运动量
3. 腕部 IMU 3–9 Hz ρ²
4. 0.5–3 Hz 是否退化

  · 编码器与 IMU **都改善** ⇒ 值得开完整网格
  · 编码器不变但 IMU **明显改善** ⇒ 也值得继续（它补的是编码器下游运动）
  · IMU 仅改善约 2–3%，或**低频恶化** ⇒ 立即停止
  · 编码器改善但 IMU 不改善 ⇒ 停止（研究目标是腕部 IMU 保真度）

用法:
    python -u sim/step4_gvnext/probe_flex_j5.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g", ROOT / "sim/step3_canobs", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data import load, bp, q_at, best_fit, valid_files, FPS      # noqa: E402

W_HZ, ZETA, JFLEX = 8.485, 0.4, 4          # 事前固定：旧网格 j5 的 IMU 最优点
BANDS = ((0.5, 3.0), (3.0, 9.0))


def make_item(dd):
    """构造 `roll_flex` 需要的 item：act / ref(限幅器输出) / send_s。"""
    from optuna_ident_enc import accel_limited_dt
    from sim.step2_fidelity import A_MAX
    act = np.asarray(dd["cmd"], float)
    ts = np.asarray(dd["t_send"], float)
    dts = np.append(np.diff(ts), np.median(np.diff(ts)))
    cmd = np.degrees(act[:, :6])
    ref = np.empty_like(cmd)
    for j in range(6):
        ref[:, j] = accel_limited_dt(cmd[:, j], float(A_MAX[j]), dts)
    return {"act": act, "ref": np.radians(ref), "send_s": ts - ts[0]}


def roll_both(xml, item, theta, flex_joint: str | None):
    """`roll_flex` 的变体：额外返回柔性铰的 qpos（无铰则为 None）。"""
    import mujoco
    from scipy.signal import butter, filtfilt
    from structure_compare import apply_theta_named
    m = mujoco.MjModel.from_xml_path(str(xml))
    apply_theta_named(m, theta)
    dt = m.opt.timestep
    qa = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{k+1}")])
          for k in range(6)]
    fa = None
    if flex_joint:
        fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, flex_joint)
        fa = int(m.jnt_qposadr[fid]) if fid >= 0 else None
    d = mujoco.MjData(m)
    for k in range(6):
        d.qpos[qa[k]] = float(item["act"][0, k])
    g0 = float(np.clip(item["act"][0, 6], 0, 0.035)) if item["act"].shape[1] > 6 else 0.0
    for jn, val in (("joint7", g0), ("joint8", -g0)):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid >= 0:
            d.qpos[int(m.jnt_qposadr[jid])] = val
    mujoco.mj_forward(m, d)
    d.ctrl[:6] = item["ref"][0]
    for an, val in (("joint7", g0), ("joint8", -g0)):
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, an)
        if aid >= 0:
            d.ctrl[aid] = val
    for _ in range(int(round(0.3 / dt))):
        mujoco.mj_step(m, d)
    send = np.maximum.accumulate(np.maximum(
        np.rint(np.asarray(item["send_s"], float) / dt).astype(int), 0))
    read = send + 1
    n = len(send); ntot = int(read[-1]) + 1
    hi = np.empty((ntot, 3)); q = np.empty((n, 6))
    fl = np.zeros(n) if fa is not None else None
    si = ri = 0
    for s in range(ntot):
        while si < n and send[si] <= s:
            d.ctrl[:6] = item["ref"][si]; si += 1
        while ri < n and read[ri] <= s:
            q[ri] = [d.qpos[a] for a in qa]
            if fa is not None:
                fl[ri] = d.qpos[fa]
            ri += 1
        hi[s] = d.sensordata[3:6]
        mujoco.mj_step(m, d)
    while ri < n:
        q[ri] = [d.qpos[a] for a in qa]
        if fa is not None:
            fl[ri] = d.qpos[fa]
        ri += 1
    ba = butter(4, 0.45 * FPS / ((1.0 / dt) / 2.0))
    return q, filtfilt(*ba, hi, axis=0)[np.clip(read, 0, ntot - 1)], fl


def main() -> int:
    import gc
    gc.disable()
    from g_q_v1 import THETA_STAR
    from sim.step2_gap_baseline import MODEL
    from structure_compare import flex_xml
    from train_rs import sensor_target

    xml = flex_xml(JFLEX, W_HZ, ZETA, THETA_STAR)
    print(f"柔性铰 @ j{JFLEX+1}   ω={W_HZ} Hz  ζ={ZETA}   （事前固定，不搜索）")
    print(f"  模型 {xml.name}\n")

    for dn, pat, lbl in (("s1", "replay_*.json", "训练族 17 条"),
                         ("s2", "replay_pp2f*.json", "留出 8 条")):
        enc = {"base": [0.0, 0.0], "flex": [0.0, 0.0]}     # [误差能量, 实测能量]
        imu = {b: {"base": [], "flex": []} for b in BANDS}
        fl_rms, sig_rms = [], []
        for f in valid_files(ROOT / f"outputs/step4_gvnext/{dn}", pat):
            dd = load(f)
            it = make_item(dd)
            t20 = dd["tg_i"][::5]
            qb, gb, _ = roll_both(MODEL, it, THETA_STAR, None)
            qf, gf, fl = roll_both(xml, it, THETA_STAR, f"joint{JFLEX+1}_flex")
            qt = q_at(dd, t20)
            tr = np.asarray(dd["t_read"], float)
            Ab, _, _ = sensor_target(dd, t20)
            # ① 电机侧 j5 编码器 3–9 Hz 误差（重采到统一 20 Hz 网格）
            vt = np.gradient(qt[:, JFLEX], 1 / FPS)
            for k, q_ in (("base", qb), ("flex", qf)):
                v = np.interp(t20, tr, np.gradient(q_[:, JFLEX], 1 / FPS))
                a = bp(vt[:, None], 3.0, 9.0)[:, 0]; b = bp(v[:, None], 3.0, 9.0)[:, 0]
                enc[k][0] += float(((a - b) ** 2).sum()); enc[k][1] += float((a ** 2).sum())
            # ② 柔性铰的相对运动量（与 j5 自身运动比）
            flr = np.interp(t20, tr, fl)
            fl_rms.append(float(np.sqrt(np.mean(bp(flr[:, None], 3.0, 9.0)[:, 0] ** 2))))
            sig_rms.append(float(np.sqrt(np.mean(bp(qt[:, JFLEX:JFLEX+1], 3.0, 9.0)[:, 0] ** 2))))
            # ③④ 腕部 IMU（底座自己的陀螺，不加残差层）
            for k, g_ in (("base", gb), ("flex", gf)):
                S = np.stack([np.interp(t20, tr, g_[:, c]) for c in range(3)], 1)
                m_ = min(len(Ab), len(S))
                for b in BANDS:
                    imu[b][k].append(best_fit(Ab[:m_], S[:m_], *b)[1])
        print(f"=== {lbl} ===")
        eb = np.sqrt(enc["base"][0] / enc["base"][1]); ef = np.sqrt(enc["flex"][0] / enc["flex"][1])
        print(f"  ① j5 编码器 3–9 Hz 误差/信号   底座 {eb:.3f} → 柔性 {ef:.3f}"
              f"   ({(ef-eb)/eb*100:+.1f}%)")
        print(f"  ② 柔性铰相对运动 3–9 Hz RMS    {np.degrees(np.mean(fl_rms)):.4f}°"
              f"   （j5 自身带内 RMS {np.degrees(np.mean(sig_rms)):.4f}°，"
              f"占 {np.mean(fl_rms)/max(np.mean(sig_rms),1e-12)*100:.1f}%）")
        for b, tag in ((BANDS[1], "③ 腕部 IMU 3–9 Hz  ρ²"), (BANDS[0], "④ 腕部 IMU 0.5–3 Hz ρ²")):
            mb, mf = np.median(imu[b]["base"]), np.median(imu[b]["flex"])
            print(f"  {tag}       底座 {mb:.3f} → 柔性 {mf:.3f}"
                  f"   ({(mf-mb)/max(mb,1e-12)*100:+.1f}%)")
        print()
    print("  参照：G-2（含 r_q+r_s）在留出上 3–9 Hz 0.520 / 0.5–3 Hz 0.836；"
          "本表是**底座对底座**，不含残差层。")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
