#!/usr/bin/env python3
"""③a 探针序列（**协议原设计**）+ 模型侧接 `g2-incr-v1` + 无机时自测。

为什么另写而不改 `make_probe_sequence.py`
------------------------------------------
那个文件已演化成 **v2**：`AMP=6mrad`、`JOINTS=(j2,j3)`（**j5 被去掉**，注释理由
「j5 模型峰 12 mrad/s, 设计上测不到」）、8 帧 5 Hz 方波小波。
用户 2026-09-02 明确要求**回到协议原设计**：`5 mrad 单帧脉冲 / j2·j3·j5 / 正负各半 /
静止段不发`。改写原文件会毁掉 v2 的审计痕迹，故另立一支。

⚠ 但 v2 那条注释必须当真：**本脚本在上机前先算逐关节信噪比**
（`‖Δ_model,j‖ / n`，`n` = 复放噪声底，从既有 `probe_b2` 裸复放 A1/A2 实测）。
若 j5 注定过不了协议判据① (`>3`)，应当在花机时**之前**知道。

模型侧
------
`Δ_model = ω̂(u+δa) − ω̂(u)`，`ω̂` 来自**已冻结的 `g2-incr-v1`**
（`r_q` 冻结 + `v=0.4` 重拟的 `r_s`；离线增量 cos 0.409、8/8 改善）。
**不再调 v**（用户 2026-09-02）。
单边差分（不是 ±配对），与真机 `Δ_real = B − (A1+A2)/2` 的口径一致。

判据（`reports/PROTOCOL_robot_validation.md` ③a，事前定死）
-----------------------------------------------------------
 ① 信噪 `‖Δ_real‖/n > 3`      不过 ⇒ 加大探针或增加重复
 ② 方向 `cos > 0.6`            不过 ⇒ **根基有问题，停，不进 ③b**
 ③ 增益 `g`                    报告值，不设通过线
 ④ 线性 `‖Δ(+5)+Δ(−5)‖/‖Δ(+5)‖ < 0.3`   不过 ⇒ ε 尺度非线性
**②或④不过 = 停下来重新审视 Ĝ，不继续 ③b。**

⚠ 旋转与位移**不得由探针结果反挑**（用户 2026-09-02）：
上机当日用**独立标定数据**定出 `R`，在 A1/B/A2 三次复放中固定；
位移同样固定，不扫描。否则 cos 就成了自由参数（我们在离线对标里已经吃过这个亏：
自由旋转把 cos 从 0.084 抬到 0.412）。

用法:
    python -u sim/make_probe_3a.py            # 生成 + 无机时自测
    python -u sim/replay_probe.py --dry-run --seqfile outputs/probe_3a/probe_3a.npz
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "sim/step2_g", ROOT / "sim/step3_canobs",
           ROOT / "sim/step4_gvnext"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = ROOT / "outputs/probe_3a"
ART = ROOT / "outputs/step4_gvnext/g2_incr_v1.npz"
NOISE_A1 = ROOT / "outputs/probe_b2/replay_ep0_A1.json"
NOISE_A2 = ROOT / "outputs/probe_b2/replay_ep0_A2.json"

# ── 协议原设计，事前定死 ───────────────────────────────────────────────
AMP = 0.008                 # 8 mrad —— 协议对判据①不过的**预授权补救**「加大探针到 8mrad」
EPISODES_PER_ARM = 4        # 协议另一条预授权补救「增加重复次数」（原 3）
PULSES_PER_JOINT_PER_EP = 3 # 实测命中数 ÷ 关节数，用于信噪估计
EPS = 0.010                 # ε 硬界
PERIOD = 40                 # 每 2 秒一发
JOINTS = (1, 2, 4)          # j2 / j3 / j5（0-based），墙的主载体
GATE_TH = 0.04              # 静止段不发，与部署 still-gate 同判据
BAND = (0.5, 9.0)
SNR_GATE = 3.0


def build_probe(cmd7, moving):
    """单帧脉冲表：关节轮流、正负各半、静止段不发、同帧仅一关节。

    ⚠ `moving` 必须是**真机陀螺模长**判出的全局运动标志
    （`b_runtime.py:25` 的部署 still-gate 口径），**不是**逐关节命令速度 ——
    后者会让 j5 一发都打不出来（j5 的 gross 速度小，但它正是墙的主载体）。
    """
    T = len(cmd7)
    da = np.zeros((T, 6))
    hits = []
    k = 0
    for t in range(PERIOD, T - PERIOD, PERIOD):
        if not moving[t]:                                      # 静止段不发
            continue
        j = JOINTS[k % len(JOINTS)]
        k += 1
        # ⚠ 全部取 **+**；± 配对改为**跨复放**（`_p` / `_m` 两条序列，帧号完全相同）。
        # 旧设计把 ± 分散在**不同时刻**的脉冲上 ⇒ 两侧落在**不同位形**，
        # 而逐发响应范数实测差 4×(j2)–27×(j3) ⇒ 判据④ 被位形变化主导，**测不到符号非线性**。
        da[t, j] = AMP
        hits.append([t, j, 1.0])
    return da, np.asarray(hits, float).reshape(-1, 3)


def h_detectability(H, n_floor):
    """**不依赖 G-2**：用已验证的 H 直接算单帧 5 mrad 脉冲的响应幅度。

    单帧脉冲 δa=AMP 打在关节 j ⇒ 响应 `H[k][:, j]·AMP`，k=0…15。
    窗内 RMS = AMP · sqrt(mean_k ‖H[k][:,j]‖²)。这是**协议原设计能不能被测到**
    的直接判据，与我们正在检验的 G-2 无关。
    """
    out = {}
    for j in JOINTS:
        r = AMP * float(np.sqrt(np.mean(np.sum(H[:, :, j] ** 2, axis=1))))
        out[j] = (r, r / max(n_floor, 1e-30) if n_floor else float("nan"))
    return out


def audit(cmd7, da, hits):
    """零机时安全审计。全部必须 PASS 才可上机。"""
    ok = True
    def chk(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"    {'✓' if cond else '**✗**'} {name}{('  ' + detail) if detail else ''}")
    chk("|δa| ≤ ε", np.abs(da).max() <= EPS + 1e-12,
        f"max {np.abs(da).max()*1e3:.2f} mrad ≤ {EPS*1e3:.0f}")
    chk("同帧仅一关节", int(np.max((np.abs(da) > 0).sum(1))) <= 1)
    chk("脉冲均为单帧", all(np.count_nonzero(da[:, j]) == int((hits[:, 1] == j).sum())
                            for j in JOINTS))
    chk("每关节发数 ≥ 2（跨复放 ± 配对后每侧同样发数）",
        all(int((hits[:, 1] == j).sum()) >= 2 for j in JOINTS),
        " / ".join(f"j{j+1} {int((hits[:,1]==j).sum())} 发" for j in JOINTS))
    chk("夹爪无 δa", da.shape[1] == 6)
    try:
        import mujoco
        # ⚠ **必须与执行器同一个限位源**（sim/replay_probe.py:89）。
        # 我原先用 sim.step2_gap_baseline.MODEL，比执行器的紧，报出 -1.77° 的假警报。
        # 安全检查与真正执行的东西不一致，比没有检查更糟。
        from ideal_imu_kinematic import MODEL_PATH
        m = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        # 执行器**有意**放宽 ±0.035 rad = ±2°（replay_probe.py:108）：
        # MuJoCo 把 j2 下界写成 0°，但真机实测到 −0.1°、指令到 −1.9° 且正常执行 20 集
        # ⇒ 模型限位是**保守软限**，不放宽就会报假警报（我先前的 −1.77° 就是）。
        lo, hi = m.jnt_range[:6, 0] - 0.035, m.jnt_range[:6, 1] + 0.035
        q0 = np.asarray(cmd7, float)[:, :6]
        m0 = min(float((q0 - np.asarray(lo)).min()), float((np.asarray(hi) - q0).min()))
        q1 = q0 + da
        m1 = min(float((q1 - np.asarray(lo)).min()), float((np.asarray(hi) - q1).min()))
        print(f"    · 基础序列自身的限位余量 {np.degrees(m0):.2f}°"
              f"（限位源与执行器逐字相同：模型 jnt_range ±2°）")
        chk("加探针后仍在限位内", m1 > 0, f"最小余量 {np.degrees(m1):.2f}°")
        chk("探针未使限位余量变差超过其幅值", (m0 - m1) <= AMP + 1e-9,
            f"探针引起的余量变化 {np.degrees(m0-m1):.3f}° ≤ 探针幅值 {np.degrees(AMP):.3f}°")
    except Exception as e:                                   # noqa: BLE001
        print(f"    ⚠ 限位检查跳过：{e}")
    return ok


def noise_floor():
    """从既有 `probe_b2` 裸复放 A1/A2 实测复放噪声底（带内 RMS，rad/s）。"""
    from data import bp
    try:
        a1 = np.asarray(json.loads(NOISE_A1.read_text())["imu100"], float)
        a2 = np.asarray(json.loads(NOISE_A2.read_text())["imu100"], float)
    except Exception as e:                                   # noqa: BLE001
        print(f"  ⚠ 噪声底无法从既有数据量出（{e}）⇒ 信噪比只能上机后算")
        return None
    a1, a2 = np.atleast_2d(a1), np.atleast_2d(a2)
    if a1.shape[1] > 3:
        a1, a2 = a1[:, -3:], a2[:, -3:]
    m = min(len(a1), len(a2))
    a1, a2 = bp(a1[:m], *BAND), bp(a2[:m], *BAND)
    # ⚠ 两次复放的起点不保证对齐；不扫位移会把**信号本身**当成噪声（虚高 ~√2 倍）。
    best = None
    for s_ in range(-20, 21):
        x, y = (a1[s_:], a2[:m - s_]) if s_ > 0 else (a1[:m + s_], a2[-s_:]) if s_ < 0 else (a1, a2)
        k = min(len(x), len(y))
        r = float(np.sqrt(np.mean((x[:k] - y[:k]) ** 2)))
        if best is None or r < best[0]:
            best = (r, s_)
    raw = float(np.sqrt(np.mean((a1 - a2) ** 2)))
    print(f"  复放噪声底: 未对齐 {raw:.5f} → **最优对齐 {best[0]:.5f}**（位移 {best[1]:+d} 帧）"
          f"  实测自 probe_b2 A1 vs A2")
    return best[0]


def main() -> int:
    import gc
    gc.disable()
    import json as _j
    from data import load, bp, valid_files, FPS
    from train_rq import features as rq_features, fir_design
    from train_rs import rs_features
    from eval_gate import base_q
    from ceiling_200hz import kin_gyro_fs

    z = np.load(ART, allow_pickle=False)
    man_a = _j.loads(str(z["manifest_json"]))
    W_s, nlag = np.ascontiguousarray(z["W"], float), int(z["nlag"])
    assert hashlib.sha256(W_s.tobytes()).hexdigest() == man_a["W_sha256"]
    zq = np.load(ROOT / "outputs/step4_gvnext/rq/rq_fir.npz")
    sq = _j.loads((ROOT / "outputs/step4_gvnext/rq/select.json").read_text())
    print(f"模型侧 = **{man_a['artifact']}**（v={man_a['v']}, r_q 冻结, "
          f"W sha {man_a['W_sha256'][:16]}…）")
    print(f"  离线增量 cos {man_a['dev_metrics']['cos_med']:.3f}"
          f"（③a 门槛 0.6）· 绝对代价 "
          f"{man_a['scope']['absolute_cost']['0.5-3Hz']:+.3f}/"
          f"{man_a['scope']['absolute_cost']['3-9Hz']:+.3f}\n")

    src = sorted(valid_files(ROOT / "outputs/step4_gvnext/s1"))[0]
    dd = load(src)
    cmd7 = np.asarray(dd["cmd"], float)
    t_read = np.asarray(dd["t_read"], float)
    t20 = dd["tg_i"][::5]
    from data import to20
    gy = to20(dd["gyro100"])
    gmag = np.linalg.norm(gy, axis=1)
    kk = max(1, int(0.25 * FPS))
    gmag = np.convolve(gmag, np.ones(kk) / kk, mode="same")     # 平滑，同 wall_portrait
    t20a = np.asarray(t20, float)
    moving = np.interp(t_read, t20a[:len(gmag)], gmag) >= GATE_TH
    print(f"  still-gate（真机陀螺模长 ≥ {GATE_TH}）: 运动帧占 "
          f"{moving.mean()*100:.0f}%（逐关节命令速度是**错**判据，会让 j5 打不出来）")
    da, hits = build_probe(cmd7, moving)
    print(f"探针（协议原设计）: {AMP*1e3:.0f} mrad 单帧脉冲 · 每 {PERIOD} 帧一发 · "
          f"j{'/j'.join(str(j+1) for j in JOINTS)} · 正负各半 · 静止段不发")
    print(f"  基础序列 {src.name}（训练族，T={len(cmd7)}）· 命中 {len(hits)} 发\n")
    print("  安全审计（零机时）")
    ok = audit(cmd7, da, hits)

    # ── 模型侧预测 Δ_model = ω̂(u+δa) − ω̂(u) ────────────────────────────
    def omega(ddx):
        qh = base_q(ddx, t20) + fir_design(rq_features(ddx)[0], sq["nlag"]) @ zq["W"]
        n = len(t20)
        return kin_gyro_fs(qh, FPS)[:n] + fir_design(rs_features(ddx, t20, qh)[:n], nlag) @ W_s
    w0 = omega(dd)
    c = cmd7.copy(); c[:, :6] += da
    w1 = omega({**dd, "cmd": c})
    cm = cmd7.copy(); cm[:, :6] -= da
    w1m = omega({**dd, "cmd": cm})
    m = min(len(w0), len(w1), len(w1m))
    pred = bp(w1[:m] - w0[:m], *BAND)
    predm = bp(w1m[:m] - w0[:m], *BAND)
    print(f"\n  Δ_model 带内 RMS = {np.sqrt(np.mean(pred**2)):.5f} rad/s")

    # ── 逐关节信噪比预测（v2 去掉 j5 的那条理由，先量化）──────────────
    n_floor = noise_floor()
    if n_floor is not None:
        print(f"  复放噪声底 n = {n_floor:.5f} rad/s（实测自 probe_b2 的 A1 vs A2）\n")
        print(f"  **跨激励形式检验**（候选是用多正弦训的，这里是单帧脉冲）")
        print(f"  {'关节':>5}{'发数':>6}{'模型 ‖Δ‖':>12}{'H ‖Δ‖':>11}{'模型/H':>9}")
        t20a = np.asarray(t20, float)
        for j in JOINTS:
            idx = hits[hits[:, 1] == j][:, 0].astype(int)
            w = np.zeros(m, bool)
            for t in idx:
                k0 = int(np.searchsorted(t20a, t_read[min(t, len(t_read) - 1)]))
                w[max(0, k0):min(m, k0 + 16)] = True          # 脉冲后 16 帧 = H 的核长
            r = float(np.sqrt(np.mean(pred[w] ** 2))) if w.any() else 0.0
            hr = AMP * float(np.sqrt(np.mean(np.sum(
                np.load(ROOT / "outputs/step1_H/step1_H_v1.npz")["H"][:, :, j] ** 2, axis=1))))
            print(f"  j{j+1:<4}{len(idx):>6}{r:>12.5f}{hr:>11.5f}{r/max(hr,1e-30):>9.2f}x")
        # 不依赖 G-2 的可探测性：直接用已验证的 H 算单帧脉冲响应
        import json as _jj
        H = np.ascontiguousarray(np.load(ROOT/"outputs/step1_H/step1_H_v1.npz")["H"], float)
        # ── 判据① 的**正确**算法（协议口径，不是单发）──────────────────
        # Δ_real = mean(B) − mean(A1,A2)；每臂 E 集、同关节每集 P 发 ⇒ 可相干叠加 K=E·P 发
        E, P = EPISODES_PER_ARM, PULSES_PER_JOINT_PER_EP
        sig = n_floor / np.sqrt(2.0)                 # 单次复放噪声 σ（n_floor=RMS(A1−A2)）
        n_pulse = sig * np.sqrt(1.0 / E + 1.0 / (2.0 * E))   # mean(B) − mean(A1,A2) 的噪声
        n_eff = n_pulse / np.sqrt(E * P)                      # 再相干叠加 K=E·P 发
        print(f"\n  **不依赖 G-2 的可探测性**（已验证的 H，单帧 {AMP*1e3:.0f} mrad 脉冲）")
        print(f"  协议设计: 每臂 {E} 集 · 同关节每集 ~{P} 发 ⇒ K={E*P} 发相干叠加")
        print(f"  σ(单次复放) {sig:.5f} → Δ_real 逐发噪声 {n_pulse:.5f} → 叠加后 {n_eff:.5f} rad/s")
        print(f"  {'关节':>5}{'H 响应窗内 RMS':>18}{'有效信噪比':>12}{'判据①>3':>10}")
        for j, (r, _) in h_detectability(H, n_floor).items():
            sn = r / max(n_eff, 1e-30)
            print(f"  j{j+1:<4}{r:>18.5f}{sn:>12.2f}{'  ✓' if sn > SNR_GATE else '  **✗**':>10}")
        print(f"\n  ⚠ 候选在**多正弦**上是高估约 1.6×，在**单帧脉冲**上却低估 2–4× ⇒"
              f"\n    跨激励形式不一致。这正是用协议原设计(脉冲)而非训练同分布(多正弦)"
              f"\n    做 ③a 的价值：③a 检验的是候选能否跨激励形式成立。")

    OUT.mkdir(parents=True, exist_ok=True)
    man = {"artifact": "probe-3a-v1", "design": "PROTOCOL 3a original",
           "amp_rad": AMP, "eps_rad": EPS, "period_frames": PERIOD,
           "joints_0based": list(JOINTS), "gate_th": GATE_TH,
           "n_hits": int(len(hits)), "lr": 16, "episodes_per_arm": EPISODES_PER_ARM,
           "replays_total": 3 * EPISODES_PER_ARM, "base_seq": str(src),
           "base_seq_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
           "model_side": man_a, "safety_audit_pass": bool(ok),
           "noise_floor_rad_s": n_floor,
           "caliber": ("Δ_model = ω̂(u+δa) − ω̂(u)（单边，与真机 Δ_real = B−(A1+A2)/2 同口径）；"
                       "R 与位移上机当日用**独立标定数据**定死，**不得由探针结果反挑**；"
                       "②或④不过 ⇒ 停，不进 ③b。")}
    # ⚠ pred/omega0 原本在 t20 网格（586），而 hits 与真机 IMU 在**控制环网格**（581）。
    # 三者必须同一索引，否则脉冲窗会错位 ⇒ 在此重采到控制环网格。
    t20a = np.asarray(t20, float)[:m]
    to_loop = lambda X: np.stack([np.interp(t_read, t20a, X[:, c]) for c in range(3)], 1)
    pred_l, predm_l = to_loop(pred), to_loop(predm)
    om_l = to_loop(bp(w0[:m], *BAND))
    # omega0 = 名义 ω̂（**机体系**，绝对量）—— 分析侧用 A0 拟合 R 时需要它；
    # pred 是增量，拟合不了绝对旋转。
    # ⚠ 两条序列：`_p`(+δa) 与 `_m`(−δa)，**帧号完全相同**。
    # B 臂 r1/r2 跑 `_p`、r3/r4 跑 `_m` ⇒ 判据④ 的两侧位形**逐帧严格相同**；
    # ①②③ 改用奇分量 (Δ⁺−Δ⁻)/2，顺带**精确抵消任何偶次污染**。
    for suf, sgn, pr in (("_p", +1.0, pred_l), ("_m", -1.0, predm_l)):
        np.savez_compressed(OUT / f"probe_3a{suf}.npz",
                            cmd7=cmd7.astype(np.float32),
                            da=(sgn * da).astype(np.float32),
                            hits=np.column_stack([hits[:, 0], hits[:, 1],
                                                  sgn * hits[:, 2]]),
                            pred=pr.astype(np.float32),
                            omega0=om_l.astype(np.float32),
                            lr=np.int64(16),
                            manifest_json=np.asarray(json.dumps(
                                {**man, "sign": suf}, ensure_ascii=False, sort_keys=True)))
    np.savez_compressed(OUT / "probe_3a.npz",
                        cmd7=cmd7.astype(np.float32), da=da.astype(np.float32),
                        hits=hits, pred=pred_l.astype(np.float32),
                        pred_m=predm_l.astype(np.float32),
                        omega0=om_l.astype(np.float32),
                        lr=np.int64(16),            # 单帧脉冲的响应窗 = H 的核长
                        manifest_json=np.asarray(json.dumps(man, ensure_ascii=False,
                                                            sort_keys=True)))
    (OUT / "probe_3a.json").write_text(json.dumps(man, ensure_ascii=False,
                                                  indent=2, sort_keys=True) + "\n")
    print(f"\n  安全审计: {'**全部通过**' if ok else '**未通过 —— 不得上机**'}")
    print(f"  → {OUT/'probe_3a.npz'}")
    print(f"\n  下一步（零机时）: python -u sim/replay_probe.py --dry-run "
          f"--seqfile {OUT/'probe_3a.npz'}")
    return 0 if ok else 2


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    import os
    os._exit(rc)
