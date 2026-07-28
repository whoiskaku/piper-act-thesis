#!/usr/bin/env python3
"""B′ pilot 六项门禁检查（用户 2026-07-28 定）。

逐条复放后立即跑，六条齐了跑全量。**任一项不过就停机**，不进入 8 mrad。

  ① 六臂记录质量全部通过；设备时间无回退/重复/明显缺样
  ② 实际发送的 B− 与 −B+ 逐点一致
  ③ 各次复放起始关节状态与原始 command hash 一致
  ④ 奇分量明显高于 A1/A2 噪声
  ⑤ 偶分量与噪声统计相容（无明显偶次非线性/漂移）
  ⑥ 4 mrad 实测响应未逼近既有运动/IMU 安全包络

注意：4 mrad **不能**单独验证幅值线性度 —— 那必须等 4/8 mrad 对比。
相位/增益/时延本阶段只输出观察，不据此下结论。

用法:
  python sim/probe_b2_gate.py --ep 0                 # 全部六条
  python sim/probe_b2_gate.py --ep 0 --only A1       # 刚跑完一条, 只查记录质量
  python sim/probe_b2_gate.py --ep 0 --envelope      # 只重算安全包络并打印
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

import analyze_probe_b2 as A

ROOT = Path(__file__).resolve().parents[1]
PB = ROOT / "outputs/probe_b2"
ENV_CACHE = PB / "safety_envelope.json"
FPS = 20.0
MD = np.pi / 180 / 1000                       # 0.001° → rad

# ep0 顺序（用户定，正负成对做平衡）: A1 → j2+ → j2− → j3− → j3+ → A2
ORDER_EVEN = ["A1", "j2_4mrad_p", "j2_4mrad_m", "j3_4mrad_m", "j3_4mrad_p", "A2"]
ORDER_ODD = ["A1", "j3_4mrad_p", "j3_4mrad_m", "j2_4mrad_m", "j2_4mrad_p", "A2"]


def order_for(ep, amp="4mrad"):
    o = ORDER_EVEN if ep % 2 == 0 else ORDER_ODD
    return [c.replace("4mrad", amp) for c in o]


# ---------------------------------------------------------------- 安全包络
def safety_envelope(rebuild=False):
    """既有 20 集实测的运动/IMU 包络。探针响应逼近它 ⇒ 停。"""
    if ENV_CACHE.exists() and not rebuild:
        return json.loads(ENV_CACHE.read_text())
    from compare_ideal_methods import load_episode
    gy, vel, acc = [], [], []
    for repo, n in [("enolab-guo/eval_pickplace2_100k", 10), ("enolab-guo/eval_ab_A", 10)]:
        for ep in range(n):
            d = load_episode(repo, ep)
            gy.append(np.abs(d["imu_real"][:, 3:6]).max(0))
            q = d["state_mdeg"][:, :6] * MD
            vel.append(np.abs(np.diff(q, axis=0) * FPS).max(0))
            acc.append(np.abs(np.diff(q, 2, axis=0) * FPS ** 2).max(0))
    env = {"gyro_p99": np.percentile(np.array(gy), 99, axis=0).tolist(),
           "gyro_max": np.array(gy).max(0).tolist(),
           "vel_max": np.array(vel).max(0).tolist(),
           "acc_max": np.array(acc).max(0).tolist(), "n_ep": 20}
    ENV_CACHE.write_text(json.dumps(env, indent=1))
    return env


# ---------------------------------------------------------------- 逐项
def limits():
    """返回 (可达下界, 可达上界)。

    2026-07-28: 原来用 robot_limits()(实测范围±2°)。但那对**会饱和的关节**给出比可达
    范围宽 2° 的界 —— j2 指令到过 −2.40° 而编码器只到 −0.07°; j3 到过 +0.94° 而编码器
    只到 +0.09°。用它算余量会高估 2°, 且削顶检测永远触发不了(编码器饱和在 0° 时
    q ≤ lo=−2° 不成立)。余量与削顶都必须对**可达范围**判。
    """
    from make_probe_b2 import reach_limits
    return reach_limits()


def gate0_margin(ep, amp, recA=None):
    """⓪ 逐配置逐窗的余量门禁（**任一 B 配置上机前必须过**）。

        residual margin = min(B+, B−) 到限位的余量 − A 裸复放跟踪误差 p99

    必须为正。历史轨迹曾安全执行**不能**替代这一检查 —— 编码器跟踪误差实测 4-7°,
    远大于探针本身的 0.23-0.46°。
    recA: A1(和 A2) 的实测记录; 给了就用**本次实测**的跟踪误差, 否则用源 rollout 的
    历史值作上机前预估(预估仅供决定要不要开跑, 实测值才是判据)。
    """
    from make_probe_b2 import NEXC, JOINTS, tracking_err
    D = np.degrees
    lo, hi = limits()
    seqs = {jn: np.load(PB / f"seq_ep{ep}_{jn}_{amp}_p.npz", allow_pickle=True)
            for jn in JOINTS}
    a_arm = np.asarray(seqs["j2"]["cmd7"], float)[:, :6]
    if recA:
        errs, srcs = [], []
        for d in recA:
            q = np.asarray(d["state"], float)[:, :6] * MD
            c = np.asarray(d["cmd"], float)[:, :6]
            n = min(len(q), len(c))
            errs.append(np.abs(q[:n] - c[:n]))
            srcs.append(d.get("arm", "?"))
        n = min(len(e) for e in errs)
        err = np.max([e[:n] for e in errs], 0)          # 多条 A 取逐帧最坏
        src = "实测 " + "/".join(srcs)
    else:
        err, src = tracking_err(a_arm)
        src = f"历史预估 {src}"
    print(f"\n⓪ 逐窗余量门禁  ({amp}, 跟踪误差来源: {src})")
    print(f"{'关节':>5}{'窗':>6}{'nominal':>10}{'B+ 余量':>10}{'B− 余量':>10}"
          f"{'跟踪p99':>10}{'residual':>11}")
    ok = True
    for jn, j in JOINTS.items():
        z = seqs[jn]
        A_ = float(z["amp"])
        for w in [int(x) for x in z["wins"]]:
            sg = slice(w, w + NEXC)
            v = a_arm[sg, j]
            d = np.asarray(z["da"], float)[sg, j]
            mn = min((v - lo[j]).min(), (hi[j] - v).min())
            mp = min((v + d - lo[j]).min(), (hi[j] - v - d).min())
            mm = min((v - d - lo[j]).min(), (hi[j] - v + d).min())
            e9 = np.percentile(err[sg, j], 99) if len(err) > w + NEXC else np.nan
            res = min(mp, mm) - e9
            good = res > 0
            ok &= bool(good)
            print(f"{jn:>5}{w:>6}{D(mn):>9.3f}°{D(mp):>9.3f}°{D(mm):>9.3f}°"
                  f"{D(e9):>9.3f}°{D(res):>+10.3f}°  {'✓' if good else '✗'}")
    # 命令削顶 = 编码器**钉在可达边界不动** 且 指令**在明显移动** ⇒ 控制器/硬件截断。
    #   · "在动"阈值取 1°/s: 早先用 1e-5 rad/帧(0.0006°) —— 指令的量化抖动都算在动, 误报
    #   · **按注入窗判门禁**: 基准轨迹别处的削顶(j2 指令到 −1.77° 而可达下界 −0.07°)是
    #     六臂共有的属性, 与探针无关, 只作信息报告, 不阻断
    clip = False
    if recA:
        from make_probe_b2 import NEXC, JOINTS
        winj = {j: [int(x) for x in seqs[jn]["wins"]] for jn, j in JOINTS.items()}
        for d in recA:
            q = np.asarray(d["state"], float)[:, :6] * MD
            c = np.asarray(d["cmd"], float)[:, :6]
            n = min(len(q), len(c))
            for j in range(6):
                still = np.abs(np.diff(q[:n, j], prepend=q[0, j])) * FPS < np.radians(0.5)
                out = ((q[:n, j] <= lo[j] + np.radians(0.1))
                       | (q[:n, j] >= hi[j] - np.radians(0.1)))
                mv = np.abs(np.diff(c[:n, j], prepend=c[0, j])) * FPS > np.radians(1.0)
                bad = still & out & mv
                inw = np.zeros(n, bool)
                for w in winj.get(j, []):
                    inw[w:w + NEXC] = True
                k_in, k_all = int((bad & inw).sum()), int(bad.sum())
                if k_in > 3:                       # 注入窗内削顶 ⇒ 阻断
                    clip = True
                    print(f"   ✗ 削顶(注入窗内): {d.get('arm')} j{j+1} {k_in} 帧 "
                          f"—— 该窗不得注入")
                elif k_all > 3:                    # 窗外削顶 ⇒ 只报告
                    print(f"   · 削顶(注入窗外, 不阻断): {d.get('arm')} j{j+1} {k_all} 帧 "
                          f"—— 基准轨迹本身的属性, 六臂共有")
        print(f"   命令削顶检查(注入窗内): {'✗ 发现削顶' if clip else '✓ 未发现'}")
    else:
        print(f"   命令削顶检查: — 需 A 实测记录")
    ok &= not clip
    print(f"   ⓪ {'✓ 全部窗口 residual 为正' if ok else '✗ 有窗口 residual ≤0 或发生削顶 — 该窗不得注入'}")
    return ok


def gate1_recording(recs):
    """① 记录质量 + 设备时间健全 + 时钟映射。"""
    print("\n① 记录质量 / 设备时间 / 时钟映射")
    print(f"{'臂':>14}{'样本':>13}{'速率':>9}{'缺样':>8}{'回退':>6}{'重复':>6}"
          f"{'晶振':>10}{'延迟p95':>10}{'发送跨度':>10}")
    ok_all = True
    for cfg, d in recs.items():
        a = np.asarray(d.get("imu100", []), float)
        if a.size == 0:
            print(f"{cfg:>14}  ✗ 无 100Hz 记录"); ok_all = False; continue
        ht, ms = a[:, 0], a[:, 1]
        dur = len(d["imu"]) / FPS
        dms = np.diff(ms)
        back, dup = int((dms < 0).sum()), int((dms == 0).sum())
        exp_n = dur * 100
        miss = max(0.0, 1 - len(a) / exp_n)
        rate = len(a) / max(ht[-1] - ht[0], 1e-9)
        _, ci = A.fit_clock(ht, ms, name=cfg)
        span = np.asarray(d.get("t_send_span", []), float)
        sp = np.nanmedian(span) * 1e3 if span.size and np.isfinite(span).any() else float("nan")
        ok = (back == 0 and dup == 0 and miss < 0.02 and ci["ok"]
              and abs(rate - 100) < 3)
        ok_all &= ok
        print(f"{cfg:>14}{len(a):>7}/{exp_n:>5.0f}{rate:>8.1f}Hz{miss*100:>7.1f}%"
              f"{back:>6}{dup:>6}{ci['rate_ppm']:>+8.0f}ppm{ci['lat_p95_ms']:>9.1f}ms"
              f"{sp:>9.2f}ms  {'✓' if ok else '✗'}")
        d["_clock"] = ci
    if ok_all:
        print("   ✓ 六臂记录全部达标")
    else:
        print("   ✗ 有记录不达标 —— 停机, 不得进入后续判读")
    return ok_all


def gate2_symmetry(recs, seq):
    """② 实际发送的 B− 与 −B+ 逐点一致（查 cmd, 不是查设计序列）。"""
    print("\n② 实际发送 B− ≟ −B+  (逐点, 查真实下发的 cmd)")
    base = np.asarray(seq["cmd7"], float)[:, :6]
    ok_all = True
    for jn in ("j2", "j3"):
        kp, km = f"{jn}_{AMP[0]}_p", f"{jn}_{AMP[0]}_m"
        if kp not in recs or km not in recs:
            continue
        cp = np.asarray(recs[kp]["cmd"], float)[:, :6]
        cm = np.asarray(recs[km]["cmd"], float)[:, :6]
        n = min(len(cp), len(cm), len(base))
        dp, dm = cp[:n] - base[:n], cm[:n] - base[:n]
        err = np.abs(dp + dm).max()                 # B+ 与 B− 的 δa 应严格互为相反数
        gr = np.abs(np.asarray(recs[kp]["cmd"], float)[:n, 6]
                    - np.asarray(recs[km]["cmd"], float)[:n, 6]).max()
        ok = err < 1e-9 and gr < 1e-9
        ok_all &= ok
        print(f"   {jn}: max|δa+ + δa−| = {err*1e6:.3f} µrad   夹爪差 {gr*1e6:.3f} µrad"
              f"   峰值 {np.abs(dp).max()*1e3:.2f} mrad  {'✓' if ok else '✗'}")
    return ok_all


def gate3_identity(recs, seq):
    """③ 起始关节状态与原始 command hash 一致。"""
    print("\n③ 起始状态 / 基准指令 hash 一致性")
    hs = {c: d.get("cmd_hash") for c, d in recs.items()}
    uniq = set(v for v in hs.values() if v)
    ok_h = len(uniq) == 1 and None not in hs.values()
    print(f"   command hash: {uniq if uniq else '缺失'}  {'✓ 全部一致' if ok_h else '✗ 不一致/缺失'}")
    st = {c: np.asarray(d.get("state_init", []), float)[:6] * MD for c, d in recs.items()}
    have = {c: v for c, v in st.items() if v.size == 6}
    if len(have) < 2:
        print("   ✗ 缺 state_init（需新版 replay_probe.py）"); return False
    M = np.stack(list(have.values()))
    spread = np.degrees(M.max(0) - M.min(0))
    ok_s = spread.max() < 0.5                        # 各臂起始位形差 < 0.5°
    print(f"   起始关节最大离散 {np.round(spread, 3)} °  (阈 0.5°)  {'✓' if ok_s else '✗'}")
    return ok_h and ok_s


AMP = ["4mrad"]


def gate456(recs, seq, env):
    """④奇分量>噪声 ⑤偶分量≈噪声 ⑥未逼近安全包络。"""
    freqs = np.asarray(seq["freqs"], float)
    jn = {1: "j2", 2: "j3"}[int(seq["joint"])]
    W = {k: A.arm_windows(recs[c], seq, sg) for k, (c, sg) in
         {"A1": ("A1", 0), "A2": ("A2", 0),
          "Bp": (f"{jn}_{AMP[0]}_p", +1), "Bm": (f"{jn}_{AMP[0]}_m", -1)}.items()}
    common = sorted(set.intersection(*[set(v) for v in W.values()]))
    if not common:
        print(f"\n④⑤ {jn}: 无共同可用窗 ✗"); return False, False, False

    def LI(arm, w):
        d = W[arm][w]
        return np.array([A.lockin_at(d["y"], d["t"], f, d["dt"]) for f in freqs])

    Yo = np.array([(LI("Bp", w) - LI("Bm", w)) / 2 for w in common])
    Ye = np.array([(LI("Bp", w) + LI("Bm", w)) / 2
                   - (LI("A1", w) + LI("A2", w)) / 2 for w in common])
    Yn = np.array([(LI("A2", w) - LI("A1", w)) / 2 for w in common])
    nw = len(common)
    # 噪声标准差(每窗每频点每轴, 复数量): 由 Yn 估。Yo 与 Yn 同方差 σ²/2;
    # Ye = (Yp+Ym)/2 − (YA1+YA2)/2 的方差是 σ² ⇒ 标准差 √2 倍。
    # **功率 SNR**（不是幅值比）:  SNR_p = E_odd,tones / E_noise,tones
    #   Y_odd = S + N_odd, D 方案下 var(N_odd) = var(N_noise) = σ²/2 ⇒ 可直接相减去偏
    #   与相干度的对应:  γ² = SNR_p/(1+SNR_p)   ⇒  SNR_p 3/5 ↔ γ² 0.75/0.83
    P = lambda X: float(np.mean(np.abs(X) ** 2))                 # 逐窗逐频点功率
    Pn = P(Yn)
    snr_o = max(P(Yo) - Pn, 0.0) / max(Pn, 1e-30)                # 功率 SNR(已去噪偏)
    # 偶分量: var = σ² = 2×var(Yn) ⇒ H0 下 P(Ye)/Pn ≈ 2
    snr_e = max(P(Ye) - 2 * Pn, 0.0) / max(2 * Pn, 1e-30)
    sd = np.sqrt(Pn)
    print(f"\n④⑤ {jn} {AMP[0]}  ({nw} 窗 @ {common}, 噪声底 {sd*1e3:.2f} mrad/s rms)")
    print(f"{'f(Hz)':>7}{'|奇|':>10}{'|偶|':>10}{'|噪声|':>10}{'SNR_p':>9}{'γ²(推)':>9}{'偶 SNR_p':>10}")
    for i, f in enumerate(freqs):
        pn = P(Yn[:, i])
        so = max(P(Yo[:, i]) - pn, 0.0) / max(pn, 1e-30)
        se = max(P(Ye[:, i]) - 2 * pn, 0.0) / max(2 * pn, 1e-30)
        print(f"{f:>7.1f}{np.sqrt(P(Yo[:, i]))*1e3:>9.2f}{np.sqrt(P(Ye[:, i]))*1e3:>9.2f}"
              f"{np.sqrt(pn)*1e3:>9.2f}{so:>9.1f}{so/(1+so):>9.2f}{se:>10.1f}")
    if nw < 2:
        print(f"   ⚠ {jn} 只有 1 个注入窗 —— 无窗间独立观测: H1 的 γ² 无意义(已置 NaN),"
              f" 不确定度只能由实测噪声窗给出; SNR_p/偶分量仍可判, 但离散度更大")
    # ④ 两档:
    #   门禁档 SNR≥3 —— 只问"注入信号在测量链路里看得见吗", 看不见说明链路有问题, 停机。
    #   校准档 SNR≥5 —— 够不够支撑 Ĝ 校准结论。4mrad 达不到是**预期内**的,
    #   那正是开放 8mrad 的理由, 不是停机理由(幅值本就是按 8mrad 设计的)。
    ok4, ok5 = snr_o >= 3.0, snr_e <= 3.0
    grade = snr_o >= 5.0
    print(f"   ④ 奇/噪 SNR = {snr_o:.1f}   链路档(≥3) {'✓' if ok4 else '✗ 信号埋在噪声里, 链路有问题'}"
          f"   校准档(≥5) {'✓' if grade else '— 未达; 属预期, 是开放 8mrad 的依据'}")
    print(f"   ⑤ 偶分量 SNR_p = {snr_e:.1f} (阈 ≤3, H0=0)  "
          f"{'✓ 与噪声相容' if ok5 else '✗ 有偶次非线性或漂移'}")
    # ⑤b 宽带偶分量。基频锁相**看不见二次非线性** —— y² 把能量搬到 DC、差频
    #    (0.5/1.0/1.5Hz) 和二次谐波(12-15Hz), 四个分析频点上一点都不落。
    #    故另在这两个带内比较偶分量与噪声的能量。
    tg = np.arange(0, A.NANA / A.FPS, 0.01)
    def onto(arm, w):
        d = W[arm][w]
        return np.stack([np.interp(tg, d["t"], d["y"][:, c]) for c in range(3)], 1)
    Ee, En = [], []
    for w in common:
        Ee.append((onto("Bp", w) + onto("Bm", w)) / 2 - (onto("A1", w) + onto("A2", w)) / 2)
        En.append((onto("A2", w) - onto("A1", w)) / 2)
    def band(X, lo, hi):
        F = np.fft.rfft(np.asarray(X), axis=1)
        fq = np.fft.rfftfreq(len(tg), 0.01)
        m = (fq >= lo) & (fq <= hi)
        return float(np.mean(np.abs(F[:, m]) ** 2))
    print(f"   ⑤b 宽带偶分量 (基频锁相对二次非线性是盲的)")
    ok5b = True
    for lbl, lo, hi in [("差频 0.3-1.7Hz", 0.3, 1.7), ("二次谐波 11.5-15.5Hz", 11.5, 15.5)]:
        r = band(Ee, lo, hi) / max(band(En, lo, hi), 1e-30)
        # H0: 偶分量方差 σ², 噪声窗方差 σ²/2 ⇒ 期望比值 2.0; 阈 4.0 留一倍余量
        good = r < 4.0
        ok5b &= good
        print(f"      {lbl:>22}: 偶/噪 能量比 {r:>6.2f} (H0≈2.0, 阈<4.0) {'✓' if good else '✗ 有二次非线性'}")
    ok5 = ok5 and ok5b
    print(f"      注: 4mrad 只验证**奇对称**; 开放 8mrad 还须偶分量与噪声相容、无削顶、响应峰值留余量")

    # ⑥ 安全包络
    print(f"\n⑥ {jn} {AMP[0]} 响应 vs 既有 20 集包络")
    gmax = np.array(env["gyro_max"]).max()

    def ratios(c):
        """该臂的 (gyro, 速度, 加速度) 相对既有包络的占比。"""
        a = np.asarray(recs[c]["imu100"], float)
        q = np.asarray(recs[c]["state"], float)[:, :6] * MD
        return np.array([
            np.abs(a[:, 5:8]).max() / gmax,
            (np.abs(np.diff(q, axis=0) * FPS) / np.array(env["vel_max"])).max(),
            (np.abs(np.diff(q, 2, axis=0) * FPS ** 2) / np.array(env["acc_max"])).max()])

    # 包络是**这 20 集自身的最大值**, 而探针复放的正是其中一集 —— 裸复放天然就在 ~100%。
    # 所以主判据是**探针带来的增量**(同一轨迹, B 减 A, 干净归因);
    # 绝对值只作越界护栏(不得明显越过历史包络)。
    rA = np.mean([ratios(c) for c in ("A1", "A2")], 0)
    lbl3 = ("gyro", "速度", "加速度")
    print(f"   {'臂':>14}" + "".join(f"{k:>22}" for k in lbl3))
    print(f"   {'裸复放 A1/A2':>14}" + "".join(f"{v*100:>21.0f}%" for v in rA))
    inc, absr = [], []
    for c in (f"{jn}_{AMP[0]}_p", f"{jn}_{AMP[0]}_m"):
        r = ratios(c)
        inc.append(r - rA); absr.append(r)
        print(f"   {c:>14}" + "".join(
            f"{r[k]*100:>14.0f}% ({(r[k]-rA[k])*100:+.0f}pp)" for k in range(3)))
    mi, ma = np.max(inc), np.max(absr)
    ok6 = mi < 0.10 and ma < 1.10
    print(f"   ⑥ 探针增量最大 {mi*100:+.1f}pp (阈 <+10pp) | "
          f"绝对占比最大 {ma*100:.0f}% (护栏 <110%)  {'✓' if ok6 else '✗ 停机'}")
    print(f"      注: 主判据是增量 —— 包络取自含本集在内的 20 集, 裸复放本就接近 100%;"
          f"\n          加速度的 20Hz 二阶差分在 7Hz 近奈奎斯特, 绝对值不可靠, 但同轨迹增量可比")
    return ok4, ok5, ok6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--only", default=None, help="只查这一条的记录质量(刚跑完时用)")
    ap.add_argument("--margin", action="store_true",
                    help="⓪ 逐窗余量门禁(任一 B 配置上机前必须过; 有 A 记录则用实测)")
    ap.add_argument("--envelope", action="store_true", help="只重算并打印安全包络")
    ap.add_argument("--amp", default="4mrad")
    a = ap.parse_args()
    AMP[0] = a.amp

    env = safety_envelope(rebuild=a.envelope)
    if a.envelope:
        print("既有 20 集安全包络:")
        print(f"  gyro 各集峰值的最大 {np.round(env['gyro_max'], 3)} rad/s")
        print(f"  关节速度上限 {np.round(np.degrees(env['vel_max']), 1)} °/s")
        print(f"  关节加速度上限 {np.round(np.degrees(env['acc_max']), 0)} °/s²")
        return 0

    if a.margin:
        rA = []
        for c in ("A1", "A2"):
            q = PB / f"replay_ep{a.ep}_{c}.json"
            if q.exists():
                rA.append(json.loads(q.read_text()))
        return 0 if gate0_margin(a.ep, a.amp, rA or None) else 1

    order = order_for(a.ep, a.amp)
    print(f"=== ep{a.ep} B′ pilot 门禁  (顺序 {' → '.join(order)}) ===")
    want = [a.only] if a.only else order
    recs, missing = {}, []
    for cfg in want:
        p = PB / f"replay_ep{a.ep}_{cfg}.json"
        if not p.exists():
            missing.append(cfg); continue
        recs[cfg] = json.loads(p.read_text())
    if missing:
        print(f"尚未采集: {missing}")
    if not recs:
        return 1

    g0 = gate0_margin(a.ep, a.amp, [recs[c] for c in ("A1", "A2") if c in recs] or None)
    g1 = gate1_recording(recs)
    if a.only or missing:
        print(f"\n（仅记录质量检查{'：' + a.only if a.only else ''}）"
              f"{'✓ 可继续下一条' if g1 else '✗ 停机排查'}")
        return 0 if g1 else 1

    g2 = gate2_symmetry(recs, np.load(PB / f"seq_ep{a.ep}_j2_{a.amp}_p.npz", allow_pickle=True))
    g3 = gate3_identity(recs, None)
    g4 = g5 = g6 = True
    for jn in ("j2", "j3"):
        seq = np.load(PB / f"seq_ep{a.ep}_{jn}_{a.amp}_p.npz", allow_pickle=True)
        r4, r5, r6 = gate456(recs, seq, env)
        g4 &= r4; g5 &= r5; g6 &= r6

    res = [("⓪ 逐窗 residual margin/削顶", g0), ("① 记录质量/设备时间", g1), ("② B− = −B+ 逐点", g2),
           ("③ 起始状态/hash 一致", g3), ("④ 奇分量 > 噪声", g4),
           ("⑤ 偶分量 ≈ 噪声", g5), ("⑥ 未逼近安全包络", g6)]
    print("\n" + "=" * 56)
    for lbl, v in res:
        print(f"  {'✓' if v else '✗'} {lbl}")
    allok = all(v for _, v in res)
    print("=" * 56)
    print("判决: " + ("✅ 六项全过 —— 可以讨论是否开放 8 mrad"
                     if allok else "❌ 未全过 —— 不开放 8 mrad"))
    print("提醒: 相位/增益/时延本阶段只作观察, 不据此下最终结论。")
    (PB / f"gate_ep{a.ep}.json").write_text(json.dumps(
        {"ep": a.ep, "order": order, "results": {l: bool(v) for l, v in res},
         "clocks": {c: d.get("_clock") for c, d in recs.items()}}, indent=1))
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
