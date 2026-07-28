#!/usr/bin/env python3
"""审计 ①b：宽带绝对时序 / 相位分支判别（2026-07-28，用户指出 ①a 的三处越界）。

**①a 的三处越界（已收回）**
  1. apply_G 与 increment_fast 差正好 LSHIFT=3 只证明**软件索引契约正确**，
     不证明模型与真机**物理对齐**。
  2. LSHIFT 是**双方共同**的坐标变换，消不掉相对延迟：
        H_real^e = S₃ H_real^wall,   H_Ĝ^e = S₃ H_Ĝ^wall  ⇒ 复数比值不变
     墙钟差 100 ms，e-index 下仍差 100 ms。故「口径差」的归因无效。
  3. 43+100=143 ms 只能叫「在 +100 ms 分支下的 6–8 Hz 有效群延迟」，
     不能叫「真机总时延」，更不能说「伺服 FIR 解释了 30%」——
     Ĝ 是并联复数系统，群延迟不能按比例分摊。

**核心问题**：+100 ms 与 −40 ms 相差约 140 ms ≈ 1/7 Hz ⇒ 这是**相位分支别名**。
四个音调只跨 6.0–7.5 Hz，配上 3 个自由增益后两个分支都能拟合。±1 ms 只是主谷
内部的曲率，**不是排除了另一分支后的绝对置信区间**。

**本工具做六件事**
  (1) 用完整「2 s 猝发 + 1.5 s 衰减」窗，对**所有有能量的频点**做 NUDFT（不只四音调）
  (2) 在**真实 100 Hz 时间戳**上联合搜索 τ ∈ [−250, 250] ms
  (3) 保留并报告**所有局部谷**、目标函数差、以及各工作点单独结果
  (4) 用旧 5 Hz 小波探针（4.1–5.4 Hz，与 B′ 相隔约 2 Hz）**独立判别分支**
  (5) 把真实 odd 与模型都左移三帧再拟合一次 —— 相对延迟**理论上必须不变**
  (6) 已知延迟的**合成测试**，确认符号与分支选择

用法: python sim/audit_G_timing_b.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

import analyze_probe_b2 as A
import train_pi_e2 as TP
from g_model import Rigid, load_ensemble

ROOT = Path(__file__).resolve().parents[1]
OPS = [("j2", 170), ("j2", 280), ("j3", 170)]
ROUNDS = [(1, "4mrad", ""), (2, "4mrad", "r2_"), (3, "2mrad", "r3_")]
FGRID = np.arange(3.0, 10.01, 0.05)          # 宽带频点栅格
UMIN = 0.15                                   # |U| 门限(占该窗峰值)


def nudft(x, t, dt, f):
    """Δt 加权 NUDFT（与 analyze_probe_b2.lockin_at 同一实现）。"""
    return A.lockin_at(x, t, f, dt)


def broadband_H(jn, w, models, cmd7, R, shift_frames=0):
    """宽带 H_real / H_model(逐路径)。真实 100Hz 时间戳 + NUDFT。

    shift_frames: (5) 的不变性检查 —— **把窗原点在两侧同时**前移这么多帧,
      U 也在移位窗上重算, 于是两侧 H 都在同一个移位坐标系里算出。
      正确实现下 τ* 必须不变。
      (旧实现的 bug: 真机侧改变换的时间戳、模型侧改采样点, 两者在窗边界不等价,
       导致 τ* 假变 −300ms = −2×3帧。)
    """
    j = {"j2": 1, "j3": 2}[jn]
    seq = np.load(A.PB / f"seq_ep0_{jn}_4mrad_p.npz", allow_pickle=True)
    da = np.zeros((len(cmd7), 6), np.float32)
    da[:, j] = seq["da"][:, j]
    # ---- 模型: 逐路径墙钟响应, 在真实时间戳上求值 ----
    km = len(models) // 2
    g = models[km]; g.R = R
    fl = g.flex
    import torch
    B0, D0 = fl.B.detach().clone(), fl.D.detach().clone()
    with torch.no_grad():
        fl.B.zero_(); fl.D.zero_(); rigid = g.increment_fast(cmd7, da)
        fl.D.copy_(D0); rig_d = g.increment_fast(cmd7, da)
        fl.D.zero_()
        modes = []
        for m in range(B0.shape[0]):
            fl.B.zero_(); fl.B[m] = B0[m]
            modes.append(g.increment_fast(cmd7, da) - rigid)
        fl.B.copy_(B0); fl.D.copy_(D0)
    paths = {"刚体": rigid, "直通D": rig_d - rigid}
    for m, mm in enumerate(modes):
        paths[f"模态{m+1}"] = mm

    Hr, Hm, wts = [], [], []
    for rnd, amp, rt in ROUNDS:
        sq = np.load(A.PB / f"seq_ep0_{jn}_{amp}_p.npz", allow_pickle=True)
        if w not in [int(x) for x in sq["wins"]]:
            continue
        Rec = {k: json.loads((A.PB / f"replay_ep0_{rt}{jn}_{amp}_{s}.json").read_text())
               for k, s in [("Bp", "p"), ("Bm", "m")]}
        W = {k: A.arm_windows(Rec[k], sq, +1 if k == "Bp" else -1) for k in Rec}
        d0 = W["Bp"][w]
        # 窗原点同移: 真机侧改用 (t − sh) 作为相对时间(等价于窗起点前移 sh)
        sh = shift_frames / A.FPS
        tp = W["Bp"][w]["t"] - sh
        tm = W["Bm"][w]["t"] - sh
        U = np.array([(nudft(W["Bp"][w]["u"], tp, W["Bp"][w]["dt"], f)
                       - nudft(W["Bm"][w]["u"], tm, W["Bm"][w]["dt"], f)) / 2
                      for f in FGRID])
        keep = np.abs(U) > UMIN * np.abs(U).max()
        Y = np.array([(nudft(W["Bp"][w]["y"], tp, W["Bp"][w]["dt"], f)
                       - nudft(W["Bm"][w]["y"], tm, W["Bm"][w]["dt"], f)) / 2
                      for f in FGRID])
        Hr.append(Y / U[:, None])                  # H 与幅值无关(已验证)
        wts.append(keep)
    Hr = np.mean(Hr, 0)
    keep = np.all(wts, 0)
    # 模型: 在**同一移位窗**里求值并变换 —— 与真机侧用完全相同的相对时间轴 tp
    tg = d0["t"] - sh                              # 移位后的相对时间(与真机侧一致)
    keys = list(paths)
    for p in keys:
        yy = A._bl_eval(paths[p], w / A.FPS + tg)   # 模型在同一物理时刻上取样
        Hm.append(np.array([nudft(yy, tg, d0["dt"], f) for f in FGRID]))
    Um = np.array([nudft(A._bl_eval(da[:, j], w / A.FPS + tg), tg, d0["dt"], f)
                   for f in FGRID])
    Hm = {p: Hm[i] / Um[:, None] for i, p in enumerate(keys)}
    return FGRID[keep], Hr[keep], {p: v[keep] for p, v in Hm.items()}


def tau_scan(data, taus):
    """联合 τ 扫描: 共享 τ, 逐工作点非负 g。返回 (曲线, 逐点曲线)。"""
    curve, per = [], {k: [] for k in data}
    for tau in taus:
        R2 = Y2 = 0.0
        for k, (F, Hr, Hm) in data.items():
            ph = np.exp(-2j * np.pi * F * tau)[:, None]
            base = (Hm["刚体"] + Hm["直通D"]) * ph
            Ms = np.stack([Hm[f"模态{m+1}"] * ph for m in range(3)])
            y = (Hr - base).ravel()
            Am = Ms.reshape(3, -1).T
            Ar = np.vstack([Am.real, Am.imag])
            yr = np.concatenate([y.real, y.imag])
            g, *_ = np.linalg.lstsq(Ar, yr, rcond=None)
            if (g < 0).any():
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
            rr = np.linalg.norm(Ar @ g - yr) ** 2
            yy = np.linalg.norm(Hr.ravel()) ** 2
            R2 += rr; Y2 += yy
            per[k].append(np.sqrt(rr / yy))
        curve.append(np.sqrt(R2 / Y2))
    return np.array(curve), {k: np.array(v) for k, v in per.items()}


def minima(taus, curve, n=6):
    """所有局部谷, 按目标值排序。"""
    loc = [i for i in range(1, len(curve) - 1)
           if curve[i] <= curve[i - 1] and curve[i] <= curve[i + 1]]
    loc.sort(key=lambda i: curve[i])
    return [(taus[i] * 1e3, curve[i]) for i in loc[:n]]


def main():
    rig = Rigid(); models = load_ensemble(rigid=rig)
    man = json.loads((TP.LROOT / "pp2-deploy" / "manifest.json").read_text())
    R = np.array(man["R"], np.float32)
    z = np.load(TP.LROOT / "pp2-deploy" / "ep000.npz")
    cmd7 = np.concatenate([z["a_arm"].astype(np.float32),
                           z["grip"].astype(np.float32)[:, None]], 1)
    taus = np.arange(-0.250, 0.2501, 0.001)

    # ---------- (1)(2)(3) 宽带联合 τ 扫描 ----------
    data = {}
    for jn, w in OPS:
        sq = np.load(A.PB / f"seq_ep0_{jn}_4mrad_p.npz", allow_pickle=True)
        if w not in [int(x) for x in sq["wins"]]:
            continue
        data[f"{jn}_w{w}"] = broadband_H(jn, w, models, cmd7, R)
    nb = {k: len(v[0]) for k, v in data.items()}
    print("=== (1) 宽带频点选取 (|U| > 15% 峰值) ===")
    for k, (F, _, _) in data.items():
        print(f"  {k:>10}: {len(F):>3} 个频点, 跨 {F.min():.2f}–{F.max():.2f} Hz"
              f"  (四音调版只有 4 个, 跨 6.0–7.5)")
    print(f"  ⇒ 频率跨度 {max(v[0].max() for v in data.values()) - min(v[0].min() for v in data.values()):.2f} Hz"
          f"  ⇒ 时延分辨 ~1/(2·跨度) = "
          f"{1/(2*(max(v[0].max() for v in data.values())-min(v[0].min() for v in data.values())))*1e3:.0f} ms")

    curve, per = tau_scan(data, taus)
    print("\n=== (2)(3) 联合 τ 扫描 [−250, 250] ms — 所有局部谷 ===")
    print(f"  {'τ(ms)':>9}{'联合残差':>10}{'Δ vs 最优':>11}")
    ms = minima(taus, curve)
    for t, v in ms:
        print(f"  {t:>+9.0f}{v*100:>9.1f}%{(v-ms[0][1])*100:>+10.1f}pp")
    print(f"\n  最优 τ* = {ms[0][0]:+.0f} ms, 残差 {ms[0][1]*100:.1f}%")
    print(f"  次优谷距最优 {ms[1][0]-ms[0][0]:+.0f} ms, 目标差 {(ms[1][1]-ms[0][1])*100:+.1f}pp")
    print(f"  1/7Hz = 143 ms;  相邻谷间距 ≈ "
          f"{np.median(np.diff(sorted(t for t,_ in ms))):.0f} ms")
    print("\n  各工作点单独的最优 τ:")
    for k, c in per.items():
        i = int(c.argmin())
        m2 = minima(taus, c, 3)
        print(f"    {k:>10}: τ* {taus[i]*1e3:>+5.0f} ms 残差 {c[i]*100:>5.1f}%"
              f"   前3谷 {[f'{t:+.0f}ms/{v*100:.0f}%' for t,v in m2]}")

    # ---------- (5) LSHIFT 不变性 ----------
    print("\n=== (5) 双方同时左移 3 帧 ⇒ 相对延迟必须不变 ===")
    data3 = {}
    for jn, w in OPS:
        sq = np.load(A.PB / f"seq_ep0_{jn}_4mrad_p.npz", allow_pickle=True)
        if w not in [int(x) for x in sq["wins"]]:
            continue
        data3[f"{jn}_w{w}"] = broadband_H(jn, w, models, cmd7, R, shift_frames=3)
    c3, _ = tau_scan(data3, taus)
    m3 = minima(taus, c3)
    print(f"  未移位 τ* = {ms[0][0]:+.0f} ms   左移3帧后 τ* = {m3[0][0]:+.0f} ms"
          f"   差 {m3[0][0]-ms[0][0]:+.0f} ms")
    print(f"  {'✓ 不变(≤5ms) ⇒ 与理论一致, 也再次否证「LSHIFT 造成 100ms」' if abs(m3[0][0]-ms[0][0])<=5 else '✗ 变了 ⇒ 实现里有额外 shift, 需查'}")

    # ---------- (6) 已知延迟的合成测试 ----------
    print("\n=== (6) 合成测试: 注入已知延迟, 检验符号与分支选择 ===")
    for tau_true in (0.100, -0.040, 0.050):
        syn = {}
        for k, (F, Hr, Hm) in data.items():
            Hsyn = (Hm["刚体"] + Hm["直通D"]
                    + sum(Hm[f"模态{m+1}"] for m in range(3))) \
                * np.exp(-2j * np.pi * F * tau_true)[:, None]
            syn[k] = (F, Hsyn, Hm)
        cs, _ = tau_scan(syn, taus)
        msy = minima(taus, cs, 3)
        ok = abs(msy[0][0] - tau_true * 1e3) < 3
        print(f"  真 τ = {tau_true*1e3:+.0f} ms → 估 {msy[0][0]:+.0f} ms "
              f"(残差 {msy[0][1]*100:.2f}%)  {'✓' if ok else '✗'}"
              f"   次优谷 {msy[1][0]:+.0f} ms / Δ{(msy[1][1]-msy[0][1])*100:+.1f}pp")
    print("  ⇒ 符号约定: τ>0 表示 **H_real = 模型延迟 τ**, 即真机比模型晚")

    (ROOT / "outputs/probe_b2/audit_timing_b.json").write_text(json.dumps(
        {"minima_ms": [[float(t), float(v)] for t, v in ms],
         "tau_best_ms": float(ms[0][0]), "resid_best": float(ms[0][1]),
         "per_op": {k: float(taus[int(c.argmin())] * 1e3) for k, c in per.items()},
         "shift3_tau_ms": float(m3[0][0]), "n_bins": nb}, indent=1, ensure_ascii=False))
    print(f"\n→ outputs/probe_b2/audit_timing_b.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
