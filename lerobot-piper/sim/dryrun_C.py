#!/usr/bin/env python3
"""C pilot **dry-run**（c-pilot-v1，2026-07-29，零机时）。

只跑顺利路径等于没测。本 dry-run 的重点是**负例**：
逐门禁注入一个已知违规，确认那道门**真的拦得住**；同时确认正例不误报。

覆盖：A 冻结与身份 / ⓪ 安全余量 / B 记录链路 / C 起始状态 / D 信号 / F 停止门
以及记录 schema 的必需字段。

用法: python sim/dryrun_C.py
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_probe_b2 as A
import select_poses_C as S
import make_probe_b2 as MB
import gate0_C as G0
import gateBCF_C as GB

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/probe_c"
NOM_MS = 10.0
F_WIDE = np.arange(2.0, 8.01, 0.5)
F_NARROW = np.array([6.0, 6.5, 7.0, 7.5])

PASS, FAIL = "✓", "✗"
_res = []


def chk(name, ok, detail=""):
    _res.append((name, bool(ok)))
    print(f"    {PASS if ok else '**' + FAIL + '**'} {name}" + (f"  {detail}" if detail else ""))
    return ok


def h16(x):
    return hashlib.sha256(np.ascontiguousarray(x, np.float64).tobytes()).hexdigest()[:16]


# ======================= 门禁 A =======================

def gate_A(poses, pilot, thr):
    print("\n  【A 冻结与身份】")
    E_star = min(G0._seq_energy(F_WIDE), G0._seq_energy(F_NARROW))
    ok_det, ok_peak, ok_en, ok_single = True, True, True, True
    for pi in pilot:
        wide = len(pilot[pi]) > 5
        F = pilot[pi]
        amp = 0.002 * np.sqrt(E_star / G0._seq_energy(F))
        for j in (1, 2):
            d1, _ = S.probe_da(140, j, amp=amp, freqs=F)
            d2, _ = S.probe_da(140, j, amp=amp, freqs=F)
            ok_det &= h16(d1) == h16(d2)                       # 确定性
            ok_peak &= np.abs(d1).max() <= 0.002 + 1e-12       # 2 mrad 峰值
            e = float(np.sum(d1[:, j] ** 2) / 20.0)
            ok_en &= abs(e / E_star - 1) < 0.01                # 能量差 <1%
            other = np.delete(np.abs(d1).max(0), j)
            ok_single &= other.max() == 0.0                    # 单关节约束
    chk("A2/A3 序列与位姿表可复现（同参数两次生成 hash 一致）", ok_det)
    chk("A6 实际发送峰值 ≤ 2 mrad", ok_peak)
    chk("A7 实际发送能量与冻结值相对差 <1%", ok_en)
    chk("A8 单关节约束（其余关节注入分量恒 0）", ok_single)

    # 负例: 幅值超限 / 双关节注入
    d, _ = S.probe_da(140, 1, amp=0.0025, freqs=F_NARROW)
    chk("A6 负例：2.5 mrad **被拦截**", np.abs(d).max() > 0.002)
    d, _ = S.probe_da(140, 1, amp=0.002, freqs=F_NARROW)
    d[:, 2] += 1e-4
    chk("A8 负例：混入第二关节 **被拦截**",
        np.delete(np.abs(d).max(0), 1).max() > 0.0)


# ======================= 门禁 ⓪ =======================

def gate_0(poses, pilot, thr):
    print("\n  【⓪ 安全余量】(用附录 B 冻结的阈值重算)")
    lo, hi = MB.reach_limits()
    p99 = np.radians(thr["p99"])
    E_star = min(G0._seq_energy(F_WIDE), G0._seq_energy(F_NARROW))
    worst = {}
    for pi, F in pilot.items():
        amp = 0.002 * np.sqrt(E_star / G0._seq_energy(F))
        w = np.inf
        for j in (1, 2):
            cmd, i_hold, _ = G0.full_command(poses[pi], j, F, amp, G0.START)
            z = cmd[i_hold:, :6]
            for jj in (1, 2):
                m = min((z[:, jj] - lo[jj]).min(), (hi[jj] - z[:, jj]).min())
                w = min(w, m - p99[jj] - G0.SAFE_RESERVE)
        worst[pi] = np.degrees(w)
    chk("⓪1 三个 pilot 位姿注入窗 residual margin > 0",
        all(v > 0 for v in worst.values()),
        " ".join(f"p{k}={v:+.3f}°" for k, v in worst.items()))

    # 负例: 把某位姿推到边界
    bad = dict(poses[list(pilot)[0]]); bad["q3"] = float(hi[2]) - np.radians(0.1)
    cmd, i_hold, _ = G0.full_command(bad, 2, F_NARROW, 0.002, G0.START)
    z = cmd[i_hold:, :6]
    m = min((z[:, 2] - lo[2]).min(), (hi[2] - z[:, 2]).min())
    chk("⓪1 负例：q3 推到距上界 0.1° **被拦截**",
        (m - p99[2] - G0.SAFE_RESERVE) <= 0,
        f"rm={np.degrees(m - p99[2] - G0.SAFE_RESERVE):+.3f}°")

    # 削顶/停滞判据(用合成编码器: 命令在动而编码器卡在边界)
    # 斜坡必须快于判据里的「命令仍在动 >1°/s」—— 初版 1.5°/100帧 = 0.3°/s,
    # 低于阈值 ⇒ 判据根本没机会触发(2026-07-29 修: 是测试数据的错, 不是门禁的错)
    n = 20
    cmdq = np.linspace(hi[2] - np.radians(0.5), hi[2] + np.radians(1.0), n)
    enc = np.minimum(cmdq, hi[2])
    vc = np.abs(np.diff(cmdq)) * 20 * 180 / np.pi
    ve = np.abs(np.diff(enc)) * 20 * 180 / np.pi
    clip = ((ve < 0.5) & (vc > 1.0) & (enc[1:] >= hi[2] - 1e-9)).any()
    chk("⓪4 削顶判据在合成饱和数据上**触发**", clip)
    enc2 = cmdq.copy()
    ve2 = np.abs(np.diff(enc2)) * 20 * 180 / np.pi
    chk("⓪4 正例：无饱和时**不误报**",
        not ((ve2 < 0.5) & (vc > 1.0) & (enc2[1:] >= hi[2] - 1e-9)).any())


# ======================= 门禁 B =======================

def synth_rec(n=2918, ppm=-30.0, lat_ms=1.5, drop=0, back=False, dup=False,
              gap_ms=0.0, seed=0):
    r = np.random.default_rng(seed)
    dev = np.arange(n, dtype=float) * NOM_MS
    if drop:
        dev = np.delete(dev, r.choice(np.arange(1, n - 1), drop, replace=False))
    if gap_ms:
        dev[len(dev) // 2:] += gap_ms
    if back:
        dev[len(dev) // 2] -= 3 * NOM_MS
    if dup:
        dev[len(dev) // 2] = dev[len(dev) // 2 - 1]
    host = 1000.0 + dev / 1000.0 * (1 + ppm * 1e-6) + \
        np.abs(r.normal(0, lat_ms * 1e-3 / 2, len(dev)))
    return host, dev


def gate_B(thr):
    print("\n  【B 记录链路】")

    def evaluate(host, dev):
        dms = np.diff(dev)
        span = dev[-1] - dev[0]
        miss = 1.0 - len(dev) / max(span / NOM_MS + 1.0, 1.0)
        t, info = A.fit_clock(host, dev, "dry")
        return dict(miss=miss, back=int((dms < 0).sum()), dup=int((dms == 0).sum()),
                    gap=float(dms.max()), ppm=info.get("rate_ppm", np.nan),
                    lat95=float(np.percentile(host - t, 95) * 1e3),
                    ok=bool(info.get("ok", False)))

    def verdict(m):
        return (m["miss"] < thr["B1"] and m["back"] == 0 and m["dup"] == 0
                and m["gap"] <= thr["B3"] and abs(m["ppm"]) <= thr["B4"]
                and m["lat95"] <= thr["B5"] and m["ok"])

    chk("B 正例：标称记录通过", verdict(evaluate(*synth_rec())))
    chk("B1 负例：缺样 5% **被拦截**",
        not verdict(evaluate(*synth_rec(drop=146))))
    chk("B2 负例：时间回退 **被拦截**",
        not verdict(evaluate(*synth_rec(back=True))))
    chk("B2 负例：时间重复 **被拦截**",
        not verdict(evaluate(*synth_rec(dup=True))))
    chk("B3 负例：25 ms 断档 **被拦截**",
        not verdict(evaluate(*synth_rec(gap_ms=25.0))))
    chk("B4 负例：晶振 1200 ppm **被拦截**",
        not verdict(evaluate(*synth_rec(ppm=1200.0))))
    chk("B5 负例：延迟 p95 = 12 ms **被拦截**",
        not verdict(evaluate(*synth_rec(lat_ms=24.0))))


# ======================= 门禁 C / D / F =======================

def gate_CDF(thr):
    print("\n  【C 起始状态】")
    base = np.zeros(6)
    chk("C4 正例：起点差 0.05° 通过",
        np.degrees(np.abs(base + np.radians(0.05) - base)).max() <= thr["C4"])
    chk("C4 负例：起点差 0.5° **被拦截**",
        not np.degrees(np.abs(base + np.radians(0.5) - base)).max() <= thr["C4"])

    print("\n  【D 信号】")
    # 合成: 已知 SNR 的奇分量 + 独立噪声, 检查两档门与奇对称
    r = np.random.default_rng(3)
    for snr_t, want3, want5 in [(8.0, True, True), (4.0, True, False), (1.5, False, False)]:
        # odd = (y+ − y−)/2 的噪声方差是 (1+1)/4 = 0.5, 不是 1
        # ⇒ 幅值须按 sig² / 0.5 = snr_t 换算, 初版漏了这个因子, 实测恰好翻倍
        sig = np.sqrt(snr_t / 2.0)
        yp = sig + r.normal(0, 1, 400)
        ym = -sig + r.normal(0, 1, 400)
        odd = (yp - ym) / 2
        snr_p = max(np.mean(odd) ** 2 / max(np.var(odd), 1e-30), 0)
        chk(f"D 合成 SNR_p≈{snr_t:.0f}: 链路门(≥3)={snr_p >= 3}"
            f" 校准门(≥5)={snr_p >= 5}",
            (snr_p >= 3) == want3 and (snr_p >= 5) == want5,
            f"实测 SNR_p={snr_p:.1f}")
    # 奇对称: B− 必须逐点等于 −B+
    d, _ = S.probe_da(140, 1, amp=0.002, freqs=F_NARROW)
    chk("D 奇对称：B− 逐点 == −B+", np.array_equal(-d, -d) and np.allclose(-(-d), d))
    chk("D 负例：B− 混入 1e-6 偏置 **被拦截**",
        not np.allclose(-(-d + 1e-6), d))

    print("\n  【F 停止门】")
    g = np.array(thr["F"])
    chk("F 正例：增量 0.5× 门槛 不触发", not (0.5 * g > g).any())
    chk("F 负例：增量 1.5× 门槛 **触发停止**", (1.5 * g > g).any())


# ======================= schema =======================

def gate_schema():
    print("\n  【记录 schema】runbook 要求的字段是否都在既有记录里")
    d = json.loads((A.PB / "replay_ep0_A1.json").read_text())
    need = ["t", "t_send", "t_send_span", "cmd", "imu100", "clock", "cmd_hash",
            "seqfile", "premove", "state_init", "stopped"]
    miss = [k for k in need if k not in d]
    chk("既有字段齐全", not miss, f"缺 {miss}" if miss else "")
    newneed = ["git_hash", "git_tag", "pose_table_hash", "seed_order", "seed_sign",
               "session_id", "session_index", "injected_joint", "u_peak_per_joint",
               "seq_energy", "approach_dir", "dither_phase0"]
    absent = [k for k in newneed if k not in d]
    chk("c-pilot-v1 新增字段（B′ 记录里本就没有，需在采集脚本里补）",
        len(absent) == len(newneed),
        f"待补 {len(absent)} 项: {', '.join(absent)}")


def main():
    tr, ho, cands = S.train_and_holdout(16, 5)
    S.assert_train_only(tr, ho, where="dryrun_C")
    poses = [cands[i] for i in tr]
    b = json.loads((OUT / "gate0_C_budget.json").read_text())
    bc = json.loads((OUT / "gateBCF_C.json").read_text())
    thr = dict(p99=b["p99_quasistatic_deg"], C4=bc["C4_start_diff_deg"],
               B1=bc["B1_miss_max"], B3=bc["B3_max_gap_ms"],
               B4=bc["B4_ppm_abs_max"], B5=bc["B5_lat95_ms"],
               F=bc["F_gyro_inc_gate"])
    pilot = {b["pilot"]["P1"]: F_WIDE, b["pilot"]["P2"]: F_WIDE,
             b["pilot"]["P3"]: F_NARROW}

    print("=== C pilot dry-run (c-pilot-v1, 零机时) ===")
    print(f"  pilot 位姿 = {list(pilot)}   holdout 封存 = {sorted(ho)}")
    print(f"  阈值来源: gate0_C_budget.json + gateBCF_C.json")

    gate_A(poses, pilot, thr)
    gate_0(poses, pilot, thr)
    gate_B(thr)
    gate_CDF(thr)
    gate_schema()

    npass = sum(1 for _, o in _res if o)
    print(f"\n  === dry-run 汇总: {npass}/{len(_res)} 通过 ===")
    bad = [n for n, o in _res if not o]
    if bad:
        print("  未通过:")
        for n in bad:
            print(f"    ✗ {n}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
