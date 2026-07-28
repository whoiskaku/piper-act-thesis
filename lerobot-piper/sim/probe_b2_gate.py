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
RTAG = [""]                # 轮次文件前缀: 第1轮 "", 第2轮 "r2_"
ENV_CACHE = PB / "safety_envelope.json"
FPS = 20.0
MD = np.pi / 180 / 1000                       # 0.001° → rad

# 第 1 轮（已完成）: A1 → j2+ → j2− → j3− → j3+ → A2
# 第 2 轮 = **反向平衡块**（用户 2026-07-28 定）: A3 → j3+ → j3− → j2− → j2+ → A4
#   同时反转了关节顺序与符号顺序。若偶分量是真实探针效应, 两轮应同号复现;
#   若来自时间漂移或单次复放变异, 反向块会改变或抵消它。
#   必须**重复 B 对**, 只补 A3/A4 不足以分辨。
ROUND = {1: ["A1", "j2_{a}_p", "j2_{a}_m", "j3_{a}_m", "j3_{a}_p", "A2"],
         2: ["A3", "j3_{a}_p", "j3_{a}_m", "j2_{a}_m", "j2_{a}_p", "A4"]}


def order_for(ep, amp="4mrad", rnd=1):
    return [c.format(a=amp) for c in ROUND[rnd]]


def bare_arms(rnd=1):
    return [c for c in ROUND[rnd] if c.startswith("A")]


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
BARE = ["A1", "A2"]        # 该轮的两条裸复放名(第2轮是 A3/A4)


def gate456(recs, seq, env):
    """④奇分量>噪声 ⑤偶分量≈噪声 ⑥未逼近安全包络。"""
    freqs = np.asarray(seq["freqs"], float)
    jn = {1: "j2", 2: "j3"}[int(seq["joint"])]
    W = {k: A.arm_windows(recs[c], seq, sg) for k, (c, sg) in
         {"A1": (BARE[0], 0), "A2": (BARE[1], 0),
          "Bp": (f"{jn}_{AMP[0]}_p", +1), "Bm": (f"{jn}_{AMP[0]}_m", -1)}.items()}
    common = sorted(set.intersection(*[set(v) for v in W.values()]))
    if not common:
        print(f"\n④⑤ {jn}: 无共同可用窗 ✗"); return False, False, False

    def LI(arm, w):
        d = W[arm][w]
        return np.array([A.lockin_at(d["y"], d["t"], f, d["dt"]) for f in freqs])

    def LIf(arm, w, f):
        """任意频点的锁相(⑤b 的宽带诊断用), 各臂按自身时间戳。"""
        d = W[arm][w]
        return A.lockin_at(d["y"], d["t"], f, d["dt"])

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
    #   SNR_p ≥ 5 (γ²≈0.83) —— 用于更新 Ĝ 的**必要条件, 不是充分条件**。
    #     2026-07-28 用户澄清: SNR 只证明"响应可见"。可否用于稳定校准还另受限于
    #     裸复放对数(单对 A 的噪声底自由度太少)、每关节工作点数(j3 单窗无交叉对照)、
    #     以及 ⑤ 暴露的非平稳性。达不到 ≥5 则该频点确定不可用; 达到也不等于可用。
    ok4, ok5 = snr_o >= 3.0, snr_e <= 3.0
    grade = snr_o >= 5.0
    print(f"   ④ 奇分量 SNR_p = {snr_o:.1f} (γ²≈{snr_o/(1+snr_o):.2f})   "
          f"链路门(≥3) {'✓ 响应可见' if ok4 else '✗ 埋在噪声里, 暂停后续配置并排查'}")
    print(f"      可用于 Ĝ 的**必要条件**(≥5) "
          f"{'✓ 满足' if grade else '— 未达, 该频点确定不可用'}"
          f"  —— 非充分: 另受限于裸复放对数、每关节工作点数、⑤ 的非平稳性")
    print(f"   ⑤ 偶分量 SNR_p = {snr_e:.1f} (阈 ≤3, H0=0)  "
          f"{'✓ 与噪声相容' if ok5 else '✗ 有偶次非线性或漂移'}")
    # ⑤b 宽带偶分量 —— **完全不用插值**: 各臂按自身时间戳锁相, 再频域做奇偶组合。
    #   早先用 np.interp 把四臂搬到公共网格再相减: 插值误差正比于探针响应、四臂不相消,
    #   噪声窗里又没有探针信号 ⇒ 比值被系统性抬高(实测 二次谐波 6.02 → 去插值后 4.41,
    #   差频 2.22 → 1.38)。
    # 判读边界: 差频带干净**只能排除简单的无记忆二次模型**。带动态滤波的非线性
    #   (先非线性再线性滤波, 或先滤波再非线性) 仍可能只在和频/二次谐波带显现。
    # 另设**基带参照带**(无探针内容也无谐波内容): 若那里也超出, 说明超出不是探针的谐波,
    #   而是 B 对与 A 对之间的系统性差异(非平稳/单对A低估变异/探针宽带微扰)。
    BANDS = [("差频 0.5/1.0/1.5Hz", np.array([0.5, 1.0, 1.5])),
             ("和频/二次谐波 12-15Hz", np.arange(12.0, 15.01, 0.5)),
             ("基带参照 5/8.x Hz", np.array([5.0, 5.25, 8.25, 8.5, 8.75]))]
    print(f"   ⑤b 宽带偶分量 (无插值; 基频锁相对二次非线性是盲的)")
    ok5b = True
    for lbl, FS in BANDS:
        pe = np.mean([P((LIf("Bp", w, f) + LIf("Bm", w, f)) / 2
                        - (LIf("A1", w, f) + LIf("A2", w, f)) / 2)
                      for w in common for f in FS])
        pnb = np.mean([P((LIf("A2", w, f) - LIf("A1", w, f)) / 2)
                       for w in common for f in FS])
        r = pe / max(pnb, 1e-30)
        good = r < 4.0
        ok5b &= good
        print(f"      {lbl:>24}: 偶/噪 能量比 {r:>6.2f} (H0≈2.0, 阈<4.0) "
              f"{'✓' if good else '✗'}")
    print(f"      注: 差频干净仅排除**无记忆**二次模型; 动态滤波型非线性可能只现于和频带。")
    print(f"          基带参照超出 ⇒ 非探针谐波, 而是 B 对/A 对的系统性差异。")
    ok5 = ok5 and ok5b
    print(f"      注: 4mrad 只验证**奇对称**; 开放 8mrad 还须偶分量与噪声相容、无削顶、响应峰值留余量")

    # ⑥ 安全包络
    gmax = np.array(env["gyro_max"]).max()
    jj = int(seq["joint"])

    def ratios(c):
        """(gyro占比, 逐关节速度占比[6], 逐关节加速度占比[6])。"""
        a = np.asarray(recs[c]["imu100"], float)
        q = np.asarray(recs[c]["state"], float)[:, :6] * MD
        return (np.abs(a[:, 5:8]).max() / gmax,
                np.abs(np.diff(q, axis=0) * FPS).max(0) / np.array(env["vel_max"]),
                np.abs(np.diff(q, 2, axis=0) * FPS ** 2).max(0) / np.array(env["acc_max"]))

    rA = [ratios(c) for c in BARE]
    gA = float(np.mean([r[0] for r in rA]))
    vA = np.mean([r[1] for r in rA], 0)
    print(f"\n⑥ {jn} {AMP[0]} 响应 vs 既有 20 集**经验异常门**(非硬件安全认证)")
    print(f"   {'臂':>14}{'gyro':>16}{'被注入 j'+str(jj+1):>18}"
          + "".join(f"{'j'+str(k+1):>7}" for k in range(6) if k != jj) + "   (其余=诊断)")
    print(f"   {'裸复放 '+'/'.join(BARE):>14}{gA*100:>15.0f}%{vA[jj]*100:>17.0f}%"
          + "".join(f"{vA[k]*100:>6.0f}%" for k in range(6) if k != jj))
    # 裸复放之间的离散 = 该量自身的噪声底(判据分辨力)
    vspread = np.abs(rA[0][1] - rA[1][1])
    print(f"   {BARE[0]+'↔'+BARE[1]+' 离散':>14}{abs(rA[0][0]-rA[1][0])*100:>15.1f}pp{vspread[jj]*100:>16.1f}pp"
          + "".join(f"{vspread[k]*100:>5.1f}pp" for k in range(6) if k != jj))
    dg, dvj, absg = [], [], []
    for c in (f"{jn}_{AMP[0]}_p", f"{jn}_{AMP[0]}_m"):
        g, v, _ = ratios(c)
        dg.append(g - gA); dvj.append(v[jj] - vA[jj]); absg.append(g)
        print(f"   {c:>14}{g*100:>10.0f}% ({(g-gA)*100:+.1f}pp){v[jj]*100:>9.0f}% "
              f"({(v[jj]-vA[jj])*100:+.1f}pp)"
              + "".join(f"{v[k]*100:>6.0f}%" for k in range(6) if k != jj))
    # 主判据: 100Hz IMU gyro(带宽足够) + 被注入关节。
    #   其余关节的 20Hz 差分速度**降为诊断项** —— 7Hz 已近 20Hz 的奈奎斯特, 一阶差分同样
    #   不可靠(此前只把这条告诫用在了加速度上); 且实测 A1↔A2 在 j5/j6 上就差 13pp/10pp,
    #   与 +10pp 阈值同量级, 分辨力不足。**但仍全关节监控并打印**, 不排除任何关节。
    ok6 = max(dg) < 0.10 and max(absg) < 1.10 and max(dvj) < 0.25
    print(f"   ⑥ 主判据: gyro 增量 {max(dg)*100:+.1f}pp(阈<+10) · gyro 绝对 "
          f"{max(absg)*100:.0f}%(异常门<110%) · 被注入 j{jj+1} 速度增量 "
          f"{max(dvj)*100:+.1f}pp(阈<+25)  {'✓' if ok6 else '✗'}")
    print(f"      注: 「110% 历史包络」是**经验异常门**(20集实测的最大值), 不是硬件安全认证。")
    print(f"          其余关节 20Hz 差分速度仅诊断 —— 7Hz 近奈奎斯特, 且 A1↔A2 自身离散"
          f" 达 {vspread.max()*100:.0f}pp。")
    dvo = [max(abs(ratios(c)[1][k] - vA[k]) for k in range(6) if k != jj)
           for c in (f"{jn}_{AMP[0]}_p", f"{jn}_{AMP[0]}_m")]
    if max(dvo) > 0.10:
        print(f"      · 诊断: 非注入关节速度增量最大 {max(dvo)*100:+.0f}pp "
              f"(> A1↔A2 离散 {vspread.max()*100:.0f}pp 则值得复现确认)")
    return ok4, ok5, ok6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--round", type=int, default=1, choices=[1, 2],
                    help="1=首轮 A1..A2; 2=反向平衡块 A3 → j3+ → j3− → j2− → j2+ → A4")
    ap.add_argument("--only", default=None, help="只查这一条的记录质量(刚跑完时用)")
    ap.add_argument("--margin", action="store_true",
                    help="⓪ 逐窗余量门禁(任一 B 配置上机前必须过; 有 A 记录则用实测)")
    ap.add_argument("--envelope", action="store_true", help="只重算并打印安全包络")
    ap.add_argument("--amp", default="4mrad")
    a = ap.parse_args()
    AMP[0] = a.amp
    RTAG[0] = "" if a.round == 1 else f"r{a.round}_"

    env = safety_envelope(rebuild=a.envelope)
    if a.envelope:
        print("既有 20 集安全包络:")
        print(f"  gyro 各集峰值的最大 {np.round(env['gyro_max'], 3)} rad/s")
        print(f"  关节速度上限 {np.round(np.degrees(env['vel_max']), 1)} °/s")
        print(f"  关节加速度上限 {np.round(np.degrees(env['acc_max']), 0)} °/s²")
        return 0

    if a.margin:
        rA = []
        for c in bare_arms(a.round):
            q = PB / f"replay_ep{a.ep}_{RTAG[0]}{c}.json"
            if q.exists():
                rA.append(json.loads(q.read_text()))
        return 0 if gate0_margin(a.ep, a.amp, rA or None) else 1

    order = order_for(a.ep, a.amp, a.round)
    BARE[:] = bare_arms(a.round)
    print(f"=== ep{a.ep} 第{a.round}轮 B′ 门禁  (顺序 {' → '.join(order)}) ===")
    want = [a.only] if a.only else order
    recs, missing = {}, []
    for cfg in want:
        p = PB / f"replay_ep{a.ep}_{RTAG[0]}{cfg}.json"
        if not p.exists():
            missing.append(cfg); continue
        recs[cfg] = json.loads(p.read_text())
    if missing:
        print(f"尚未采集: {missing}")
    if not recs:
        return 1

    g0 = gate0_margin(a.ep, a.amp, [recs[c] for c in BARE if c in recs] or None)
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
    (PB / f"gate_ep{a.ep}_r{a.round}.json").write_text(json.dumps(
        {"ep": a.ep, "round": a.round, "amp": a.amp, "order": order, "results": {l: bool(v) for l, v in res},
         "clocks": {c: d.get("_clock") for c, d in recs.items()}}, indent=1))
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
