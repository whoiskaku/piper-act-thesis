#!/usr/bin/env python3
"""C pilot 逐位姿分析（c-pilot-v2，2026-07-29）。

臂序 A1 → B+ → B− → A2。**每个位姿跑完立刻分析**，不连续跑完再检查。

pilot 只允许三类结论：
  1. 100 Hz 记录与统一时钟链路是否可靠
  2. 2 mrad 响应是否可见、是否奇对称
  3. 是否存在削顶、明显非线性或安全余量问题
**不得**据此宣称相位 / 时延 / 幅值线性已验证。

用法: python sim/analyze_pilot_C.py --pose 289 --joint 2
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

import analyze_probe_b2 as A
import make_probe_b2 as MB

ROOT = Path(__file__).resolve().parents[1]
PC = ROOT / "outputs/probe_c"
FPS = 20.0
NOM_MS = 10.0


def load(pose, joint):
    out = {}
    for arm in ("A1", "Bp1", "Bm1", "A2"):
        p = PC / f"replay_c_p{pose}_j{joint+1}_{arm}_c{pose}_{joint+1}.json"
        if not p.exists():
            raise SystemExit(f"✗ 缺 {p.name}")
        out[arm] = json.loads(p.read_text())
    return out


def clock(rec):
    z = np.array(rec["imu100"], float)
    t, info = A.fit_clock(z[:, 0], z[:, 1], "")
    return z, t, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", type=int, required=True)
    ap.add_argument("--joint", type=int, default=2)
    a = ap.parse_args()
    R = load(a.pose, a.joint)
    meta = R["A1"]["seq_meta"]
    w0, nexc = meta["w0"], meta["nexc"]
    F = np.array(json.loads(json.dumps(R["Bp1"].get("seq_freqs", []))) or [], float)
    if not len(F):
        z = np.load(ROOT / "outputs/probe" / f"c_p{a.pose}_j{a.joint+1}_Bp.npz")
        F = np.asarray(z["freqs"], float)

    print(f"=== C pilot 分析: 位姿 {a.pose}, 注入 j{a.joint+1} ===")
    print(f"  窗 w0={w0} nexc={nexc} ({nexc/FPS:.1f}s)  频点 {len(F)} 个 "
          f"{F.min():.1f}–{F.max():.1f} Hz\n")

    # ---------- 1. ⓪ 重算：本工作点的准静止跟踪误差 ----------
    print("  【① ⓪ 重算 —— 用本次 A1 实测填上 j2 的缺口】")
    lo, hi = MB.reach_limits()
    for nm in ("A1", "A2"):
        st = np.array(R[nm]["state"], float)[:, :6] * np.pi / 180 / 1000
        cm = np.array(R[nm]["cmd"], float)[:, :6]
        seg = slice(meta["n_appr"], len(st))          # 进场之后 = 保持+注入段
        err = np.abs(st[seg] - cm[seg])
        p99 = np.percentile(err, 99, axis=0)
        print(f"    {nm} 保持段 |编码器−指令| p99: "
              f"j2 {np.degrees(p99[1]):.4f}°  j3 {np.degrees(p99[2]):.4f}°")
        if nm == "A1":
            p99_meas = p99
    print(f"    预算里用的保守上界: j2 1.632°(越界区外推) / j3 0.122°")
    print(f"    ⇒ **实测 j2 {np.degrees(p99_meas[1]):.4f}° 比保守上界小 "
          f"{1.632/max(np.degrees(p99_meas[1]),1e-9):.0f}×** —— 缺口已用实测填上")
    SAFE = np.radians(0.3)
    for nm in ("A1", "Bp1", "Bm1", "A2"):
        q = np.array(R[nm]["cmd"], float)[:, :6]
        rm = {f"j{j+1}": np.degrees(min((q[:, j] - lo[j]).min(), (hi[j] - q[:, j]).min())
                                    - p99_meas[j] - SAFE) for j in (1, 2)}
        print(f"    {nm} residual margin(实测口径): "
              + "  ".join(f"{k}={v:+.3f}°" for k, v in rm.items()))

    # ---------- 2. 门禁 B ----------
    print("\n  【② B 记录链路】")
    print(f"    {'臂':>5}{'100Hz':>8}{'缺样':>8}{'回退':>6}{'重复':>6}{'最大间隔':>10}"
          f"{'ppm':>8}{'延迟p95':>9}{'ok':>4}")
    okB = True
    for nm in ("A1", "Bp1", "Bm1", "A2"):
        z, t, info = clock(R[nm])
        d = np.diff(z[:, 1])
        miss = 1 - len(z) / max((z[-1, 1] - z[0, 1]) / NOM_MS + 1, 1)
        lat = float(np.percentile(z[:, 0] - t, 95) * 1e3)
        good = (miss < 0.02 and (d < 0).sum() == 0 and (d == 0).sum() == 0
                and d.max() <= 20 and abs(info["rate_ppm"]) <= 500
                and lat <= 4.1 and info["ok"])
        okB &= good
        print(f"    {nm:>5}{len(z):>8}{miss:>8.2%}{int((d<0).sum()):>6}"
              f"{int((d==0).sum()):>6}{d.max():>10.1f}{info['rate_ppm']:>8.1f}"
              f"{lat:>9.2f}{'✓' if good else '✗':>4}")
    print(f"    ⇒ 记录链路{'**可靠**' if okB else '**不通过**'}")

    # ---------- 3. 门禁 C ----------
    print("\n  【③ C 起始状态 / C2 相位 / C3 进场方向】")
    s0 = np.array([R[n]["premove"]["q_start"] for n in ("A1", "Bp1", "Bm1", "A2")])
    print(f"    四臂起点 spread: " + " ".join(
        f"j{j+1}={np.degrees(np.ptp(s0[:, j])):.4f}°" for j in range(6)))
    print(f"    C4 门槛 0.20° ⇒ "
          f"{'✓' if np.degrees(np.ptp(s0, 0)).max() <= 0.20 else '**✗**'}"
          f"  (最大 {np.degrees(np.ptp(s0, 0)).max():.4f}°)")
    for nm in ("A1", "Bp1", "Bm1", "A2"):
        ad = R[nm]["approach_dir"]
        pl = ad.get("planned_travel") or {k: -v for k, v in ad["planned"].items()}
        ms = ad.get("measured_travel_from_encoder") or ad.get("measured_from_encoder", {})
        ok = all(pl.get(k) == v for k, v in ms.items()) if ms else False
        print(f"    {nm:>5} C3 行进方向 计划={pl} 实测={ms}"
              f"  {'✓一致' if ok else '**✗不一致**'}")
    # C2: 慢动相位 —— 由编码器拟合, 报四臂离散
    ph = []
    for nm in ("A1", "Bp1", "Bm1", "A2"):
        st = np.array(R[nm]["state"], float)[:, :6] * np.pi / 180 / 1000
        seg = slice(meta["n_appr"], w0)
        y = st[seg, 2] - st[seg, 2].mean()
        tt = np.arange(len(y)) / FPS
        c, s = np.cos(2*np.pi*meta["hz"]*tt), np.sin(2*np.pi*meta["hz"]*tt)
        ph.append(np.arctan2(np.dot(y, s), np.dot(y, c)))
    ph = np.unwrap(np.array(ph))
    print(f"    C2 慢动相位(编码器拟合, j3): {np.round(np.degrees(ph), 2).tolist()}°"
          f"  spread {np.degrees(np.ptp(ph)):.2f}°")

    # ---------- 4. 门禁 D / E ----------
    print("\n  【④ D 奇分量 / E 偶分量】(注入窗内 100 Hz 陀螺锁相)")
    G = {}
    for nm in ("A1", "Bp1", "Bm1", "A2"):
        z, t, _ = clock(R[nm])
        ts = np.array(R[nm]["t_send"], float)
        t0, t1 = ts[w0], ts[min(w0 + nexc, len(ts) - 1)]
        m = (t >= t0) & (t < t1)
        tw, gy = t[m] - t0, z[m, 5:8]
        dt = np.diff(tw, prepend=tw[0] - 0.01)
        G[nm] = np.array([[A.lockin_at(gy[:, ax], tw, f, dt) for ax in range(3)]
                          for f in F])          # (nf, 3) complex
        if nm == "A1":
            print(f"    窗内 100Hz 样本 {int(m.sum())} 个 ({tw[-1]-tw[0]:.2f}s)")
    odd = (G["Bp1"] - G["Bm1"]) / 2
    even = (G["Bp1"] + G["Bm1"]) / 2 - (G["A1"] + G["A2"]) / 2
    noise = (G["A2"] - G["A1"]) / 2
    snr = np.maximum(np.abs(odd) ** 2 - np.abs(noise) ** 2, 0) / np.maximum(
        np.abs(noise) ** 2, 1e-30)
    print(f"    {'f(Hz)':>7}{'|odd|轴0':>11}{'轴1':>10}{'轴2':>10}"
          f"{'SNR_p 轴0':>11}{'轴1':>9}{'轴2':>9}{'|even|/|odd|':>14}")
    for i, f in enumerate(F):
        eo = np.abs(even[i]).max() / max(np.abs(odd[i]).max(), 1e-30)
        print(f"    {f:>7.1f}" + "".join(f"{np.abs(odd[i, k]):>11.4f}" if k == 0
                                         else f"{np.abs(odd[i, k]):>10.4f}" for k in range(3))
              + "".join(f"{snr[i, k]:>11.1f}" if k == 0 else f"{snr[i, k]:>9.1f}"
                        for k in range(3)) + f"{eo:>14.2f}")
    best = snr.max(1)
    n3 = int((best >= 3).sum()); n5 = int((best >= 5).sum())
    print(f"\n    链路门 SNR_p≥3: **{n3}/{len(F)}** 频点   校准门 ≥5: **{n5}/{len(F)}**")
    eo = np.median([np.abs(even[i]).max()/max(np.abs(odd[i]).max(), 1e-30)
                    for i in range(len(F))])
    en = np.median([np.abs(even[i]).max()/max(np.abs(noise[i]).max(), 1e-30)
                    for i in range(len(F))])
    # 零分布: 单窗口下 even 与 noise 各只有**一个复样本** ⇒
    #   |even|²/|noise|² ~ F(2,2), CDF = F/(1+F) ⇒ 95% 分位 F=19 ⇒ 比值 **4.36**
    #   故"比值>3"在零假设下就有 10% 频点会出现 —— 用 <2 当判据是拍的(2026-07-29 修)
    THR = np.sqrt(19.0)
    rat = np.array([np.abs(even[i]).max() / max(np.abs(noise[i]).max(), 1e-30)
                    for i in range(len(F))])
    nex = int((rat > THR).sum()); exp = 0.05 * len(F)
    print(f"    median |even|/|odd| = {eo:.2f}    median |even|/|noise| = {en:.2f}")
    print(f"    零分布 |even|²/|noise|² ~ F(2,2) ⇒ 95% 分位比值 = {THR:.2f}")
    print(f"    超阈频点 **{nex}/{len(F)}**  (零假设期望 {exp:.2f})  "
          f"⇒ 偶分量{'与噪声**相容**' if nex <= max(1, 2*exp) else '**显著高于噪声**'}")
    print(f"    (|even|/|odd| 在单窗口下不可判; 中位数也会被频点覆盖带偏 —— P3 只有")
    print(f"     6-7.5Hz 四个点, 恰是 P1/P2 也偏高的那一段)")
    print(f"    ⚠ 单窗口 ⇒ **相干度 γ² 无法计算**(需 ≥2 窗); 相位/时延本轮不下结论。")

    # ---------- 5. 门禁 F + 削顶 ----------
    print("\n  【⑤ F 停止门 / 削顶】")
    gate = np.array(json.loads((PC / "gateBCF_C.json").read_text())["F_gyro_inc_gate"])
    zA = np.array(R["A1"]["imu100"], float)[:, 5:8]
    for nm in ("Bp1", "Bm1"):
        zB = np.array(R[nm]["imu100"], float)[:, 5:8]
        n = min(len(zA), len(zB))
        q = np.percentile(np.abs(zB[:n] - zA[:n]), 99.9, 0)
        print(f"    {nm} vs A1 gyro 增量 p99.9 = {np.round(q, 4).tolist()}"
              f"  门槛 {np.round(gate, 3).tolist()}  "
              f"{'✓' if (q <= gate).all() else '**✗ 触发停止**'}")
    for nm in ("A1", "Bp1", "Bm1", "A2"):
        st = np.array(R[nm]["state"], float)[:, :6] * np.pi / 180 / 1000
        cm = np.array(R[nm]["cmd"], float)[:, :6]
        ve = np.abs(np.diff(st, axis=0)) * FPS
        vc = np.abs(np.diff(cm, axis=0)) * FPS
        clip = 0
        for j in (1, 2):
            atb = (st[1:, j] <= lo[j] + 1e-9) | (st[1:, j] >= hi[j] - 1e-9)
            clip += int(((ve[:, j] < np.radians(0.5)) & (vc[:, j] > np.radians(1.0))
                         & atb).sum())
        print(f"    {nm} 削顶帧数 = {clip}  {'✓' if clip == 0 else '**✗**'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
