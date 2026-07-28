#!/usr/bin/env python3
"""审计①：统一的约 1 帧相位偏差（2026-07-28，用户指出）。

前一版可表达性测试给出的最优时延在三个工作点上是 −44 / −48 / −48 ms，
**几乎正好一个 20 Hz 帧（50 ms），且一致到 ±2 ms**。这个签名更像**统一的时序或
索引偏差**，而不是位形相关动力学。在排除它之前，**不能**把约 78 pp 的相位残差
归因于极点结构。

本工具做两件事:

**(A) 共享 τ 的联合拟合**（用户给的目标函数）

    min_{τ, g_{p,m} ≥ 0}  Σ_{p,f} ‖ H_real(p,f)
                              − e^{−j2πfτ}( H_fixed(p,f) + Σ_m g_{p,m} H_m(p,f) ) ‖²

  单一 τ 跨全部工作点共享; g 逐工作点。报告最优 τ、其 CI、以及残差随 τ 的曲线。
  若最优 τ ≈ 50 ms 且残差在该处大幅下降 ⇒ 是时序问题, 不是结构问题。

**(B) 时序链逐环节核查**：t_send / 模型墙钟索引 / FIR 延迟 / 历史 LSHIFT

用法: python sim/audit_G_timing.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch

import analyze_probe_b2 as A
import train_pi_e2 as TP
from g_model import Rigid, load_ensemble
from audit_G_expressibility import model_paths, measured_H, OPS

ROOT = Path(__file__).resolve().parents[1]
FREQS = np.array([6.0, 6.5, 7.0, 7.5])


def joint_fit(P_all, H_all, taus):
    """共享 τ + 逐工作点非负 g 的联合拟合。返回 (τ*, 残差曲线, 各点 g)。"""
    curve, gs = [], {}
    for tau in taus:
        tot_r2, tot_y2, g_this = 0.0, 0.0, {}
        for key in P_all:
            P, Href = P_all[key], H_all[key]
            ph = np.exp(-2j * np.pi * FREQS * tau)[:, None]
            base = (P["刚体"] + P["直通D"]) * ph
            Ms = np.stack([P[f"模态{m+1}"] * ph for m in range(3)])
            y = (Href - base).ravel()
            Am = Ms.reshape(3, -1).T
            Ar = np.vstack([Am.real, Am.imag])
            yr = np.concatenate([y.real, y.imag])
            g, *_ = np.linalg.lstsq(Ar, yr, rcond=None)
            if (g < 0).any():                          # 非负活跃集
                best = None
                for mask in range(1, 8):
                    idx = [i for i in range(3) if mask >> i & 1]
                    sol, *_ = np.linalg.lstsq(Ar[:, idx], yr, rcond=None)
                    if (sol < 0).any():
                        continue
                    gg = np.zeros(3); gg[idx] = sol
                    r = np.linalg.norm(Ar @ gg - yr)
                    if best is None or r < best[0]:
                        best = (r, gg)
                g = best[1] if best else np.maximum(g, 0)
            tot_r2 += np.linalg.norm(Ar @ g - yr) ** 2
            tot_y2 += np.linalg.norm(Href.ravel()) ** 2
            g_this[key] = g
        curve.append(np.sqrt(tot_r2 / tot_y2))
        gs[float(tau)] = g_this
    curve = np.array(curve)
    i = int(curve.argmin())
    return taus[i], curve, gs[float(taus[i])]


def main():
    rig = Rigid()
    models = load_ensemble(rigid=rig)
    man = json.loads((TP.LROOT / "pp2-deploy" / "manifest.json").read_text())
    R = np.array(man["R"], np.float32)
    z = np.load(TP.LROOT / "pp2-deploy" / "ep000.npz")
    cmd7 = np.concatenate([z["a_arm"].astype(np.float32),
                           z["grip"].astype(np.float32)[:, None]], 1)
    km = len(models) // 2

    P_all, H_all = {}, {}
    for jn, w in OPS:
        j = {"j2": 1, "j3": 2}[jn]
        seq = np.load(A.PB / f"seq_ep0_{jn}_4mrad_p.npz", allow_pickle=True)
        if w not in [int(x) for x in seq["wins"]]:
            continue
        da = np.zeros((len(cmd7), 6), np.float32)
        da[:, j] = seq["da"][:, j]
        P_all[f"{jn}_w{w}"] = model_paths(models[km], cmd7, da, w, FREQS, R)
        H_all[f"{jn}_w{w}"], _ = measured_H(jn, w, FREQS)

    # ---------- (A) 共享 τ 联合拟合 ----------
    taus = np.linspace(-0.12, 0.12, 481)
    t_best, curve, g_best = joint_fit(P_all, H_all, taus)
    r0 = curve[np.abs(taus).argmin()]                       # τ=0 的残差
    print("=== (A) 共享 τ 的联合拟合 (单一 τ 跨全部工作点) ===")
    print(f"  最优 τ* = **{t_best*1e3:+.1f} ms**   联合残差 {curve.min()*100:.1f}%"
          f"   (τ=0 时 {r0*100:.1f}%)")
    print(f"  一个 20Hz 帧 = 50.0 ms;  τ* / 帧 = **{t_best/0.05:+.3f} 帧**")
    print(f"\n  残差随 τ 的曲线 (每 10ms):")
    print(f"  {'τ(ms)':>8}" + "".join(f"{t*1e3:>8.0f}" for t in
                                     np.arange(-0.10, 0.101, 0.02)))
    print(f"  {'残差%':>8}" + "".join(
        f"{curve[np.abs(taus-t).argmin()]*100:>8.1f}" for t in np.arange(-0.10, 0.101, 0.02)))
    # τ 的 CI: 残差平方增加 1/N 视作 1σ 的粗略代理
    n = sum(H_all[k].size * 2 for k in H_all)
    thr = curve.min() * np.sqrt(1 + 2.0 / n)
    inb = taus[curve <= thr]
    print(f"\n  τ 的粗略区间(残差平方 +2/N): [{inb.min()*1e3:+.1f}, {inb.max()*1e3:+.1f}] ms")
    print(f"  50 ms(整帧) 是否落在区间内: "
          f"{'✓ 落在内 ⇒ 与「一帧偏差」一致' if inb.min() <= 0.05 <= inb.max() else '✗ 不在内'}")
    print(f"  −50 ms 是否落在区间内: "
          f"{'✓' if inb.min() <= -0.05 <= inb.max() else '✗'}")
    print(f"\n  各工作点在 τ* 处的 g:")
    for k, g in g_best.items():
        print(f"    {k:>10}: g = {np.round(g,3)}")
    print(f"\n  ⇒ 相位残差被单一 τ 补回 {(r0-curve.min())*100:.1f}pp"
          f"  (从 {r0*100:.1f}% 降到 {curve.min()*100:.1f}%)")
    if curve.min() < 0.5 and abs(abs(t_best) - 0.05) < 0.015:
        print(f"  ⇒ **最优 τ 与整帧一致且残差大幅下降 ⇒ 高度疑似统一时序/索引偏差**,")
        print(f"     在 (B) 排清之前不得把相位残差归因于极点结构。")

    # ---------- (B) 时序链核查 ----------
    print("\n=== (B) 时序链逐环节核查 ===")
    print("  ① 真机侧: 窗起点 = t_send[w] (JointCtrl 实际写总线时刻),")
    print("     IMU 采样时刻 = device_ms 经最小延迟线映射到 host monotonic")
    d = json.loads((A.PB / "replay_ep0_j2_4mrad_p.json").read_text())
    ts = np.asarray(d["t_send"], float)
    print(f"     t_send 间隔 中位 {np.median(np.diff(ts))*1e3:.2f} ms "
          f"(名义 50.0)  抖动 sd {np.std(np.diff(ts))*1e3:.2f} ms")
    a = np.asarray(d["imu100"], float)
    _, ci = A.fit_clock(a[:, 0], a[:, 1])
    print(f"     时钟映射: 晶振 {ci['rate_ppm']:+.0f} ppm, 传输延迟 p95 {ci['lat_p95_ms']:.1f} ms")
    print(f"     ⚠ **最小传输延迟被截距吸收, 不可观测** ⇒ 绝对 τ 含此常数(USB 100Hz 典型 1-3ms)")
    print("\n  ② 模型侧: increment_fast 的输出按 20Hz 帧索引;")
    print("     本审计用 _bl_eval(y, w/FPS + tg) 求值, 即**把模型第 i 帧当作时刻 i/FPS**。")
    print("     若模型第 i 帧其实是'对第 i−1 帧指令的响应', 就是整整一帧的偏差。")
    print(f"\n  ③ 响应核的时间原点: LNEG={TP.LNEG} 帧(核允许非因果前伸), "
          f"LTOT={TP.LTOT}")
    print(f"     apply_G 的核由数值探针量出(T0=48 处打单位冲激, 取 [T0−LNEG, T0+LR_RESP))")
    print(f"     ⇒ 核的第 0 个样本对应 **t = −LNEG/FPS = {-TP.LNEG/20*1e3:.0f} ms**")
    print(f"     若下游把核的第 0 个样本当作 t=0, 就引入 {TP.LNEG/20*1e3:.0f} ms 偏差")
    print(f"     **注意: {TP.LNEG} 帧 = {TP.LNEG/20*1e3:.0f} ms, 与 τ* {t_best*1e3:+.0f} ms "
          f"的关系值得核对**")
    print("\n  ④ 历史 LSHIFT / 滞后口径: 记忆记载 'e(i) = real[i+3] − ideal[i]' 的 e-index 口径,")
    print("     以及'滞后重测 2-3 帧'。本审计的模型侧未做任何 shift ⇒ 若标签管线里有,")
    print("     则模型与实测之间会差那几帧。")
    print("\n  待办(需读代码确认, 本工具只列出可疑点):")
    print("    · increment_fast / apply_G 中核的时间原点是否与 _bl_eval 的假设一致")
    print("    · 标签管线 (kin_ideal_chunked / FeatureAssembler) 是否含 shift")
    print("    · b_runtime 流式实现的相位与离线是否一致")

    (ROOT / "outputs/probe_b2/audit_timing.json").write_text(json.dumps(
        {"tau_best_ms": float(t_best * 1e3), "resid_at_tau": float(curve.min()),
         "resid_at_zero": float(r0), "tau_ci_ms": [float(inb.min() * 1e3),
                                                   float(inb.max() * 1e3)],
         "frames": float(t_best / 0.05), "LNEG": int(TP.LNEG),
         "g_at_tau": {k: v.tolist() for k, v in g_best.items()}}, indent=1))
    print(f"\n→ outputs/probe_b2/audit_timing.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
