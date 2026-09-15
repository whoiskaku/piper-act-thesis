#!/usr/bin/env python3
"""**相位专项**（有边界，零机时）：G-2 与 H 的增量差异，是统一时基问题还是结构问题？

背景与本专项的定位（用户 2026-09-15）
--------------------------------------
增益量化的结果：汇总范数比 3.61 但 **cos 仅 0.118**、跨关节离散 12×
⇒ 按事前判据 **不是标量增益问题**。且**低频幅度比 1.21（接近 1）而方向不对**，
是相位误差的典型形态。

⚠ **不得预先称为「矛盾」**：③a 的 `L=0` 是 **pp2、j5 主轴、模型—真机**的相对对齐；
本次是 **ab4、五关节、G 相对 H** 的频域比较。两者对象不同。
**本专项正是用来判断差异来自统一时基、ab4 均匀 20 Hz 假设，还是模型结构。**

⚠ 主口径订正：**j4 不进主汇总**
--------------------------------
j4 在 pp2 中**没有运动数据**（`train_rq.MOVING` 明确排除），其范数比 12.05×
属**域外外推**，单独报告。主汇总只用 **j1/j2/j3/j5/j6**。
后续候选固定 **δa_j4 = 0**。

写死的口径（用户 2026-09-15）
-----------------------------
1. **只用 1 mrad RMS**，不再扫幅度（幅度非线性已基本排除：0.5/1/2 → 3.56/3.61/4.07）
2. 固定现有 `R` 与 `ALIGN_L=3`，**主结果不得重新扫描位移**
3. 五个受支持关节**分别**算逐频率：幅值比 · 复数相位差 · 相干
4. 在 **P0 ep0–2** 拟合「相位是否近似一条线」（= 是否等价于单一时间延迟）；
   **所得一个共享延迟固定后，仅在 ep3–4 验证**。**禁止每关节分别挑位移。**
5. 归因为**统一时基**问题需**同时**满足：
   · 相位—频率近似线性
   · 五关节估计延迟相差 **≤ 1 帧**
   · 固定共享延迟后，**ep3 与 ep4 的五关节汇总 cos 均 > 0.85**，且**≥4/5 关节通过**
6. 否则收口为「**增量结构 / 逐关节动态不匹配**」，**不再靠移位修补**

⚠ ab4 无时间戳 ⇒ G 前向在**假设均匀 20 Hz** 下跑（本专项的候选解释之一）。

用法:
    python -u sim/phase_G_vs_H.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "sim/step4_gvnext"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fit_step1_candidate as S1                                    # noqa: E402
import dir_transfer_g2 as D                                         # noqa: E402
import g2_on_ab4 as G                                               # noqa: E402
import gain_G_vs_H as GH                                            # noqa: E402

FS = 20.0
AMP = 0.001                       # 1 mrad RMS（固定，不扫）
SUP = (0, 1, 2, 4, 5)             # 受支持关节 j1/j2/j3/j5/j6（**j4 出主汇总**）
J4 = 3
FIT_EP, VAL_EP = (0, 1, 2), (3, 4)
FLO, FHI = 0.5, 9.0
COS_GATE, TAU_SPREAD_MAX, NPASS_MIN = 0.85, 1.0, 4
OUT = ROOT / "outputs/step1_H/phase_G_vs_H.json"


def pair(e, g2, R, H, base, j):
    """→ (Δ_H, Δ_G)，δa 屏蔽到关节 j 并缩放到 AMP。"""
    mu, sd = pair.mu, pair.sd
    v = GH.policy_da(e, pair.Th, mu, sd)
    m = np.zeros(6); m[j] = 1.0
    v = v * m
    cur = float(np.sqrt(np.mean(v ** 2)))
    if cur < 1e-15:
        return None, None
    v = np.clip(v * (AMP / cur), -D.EPS, D.EPS)
    v[e["contact"]] = 0.0
    c = np.array(e["dd"]["cmd"], float, copy=True)
    k = min(len(c), len(v)); c[:k, :6] += v[:k]
    dw = g2.omega({**e["dd"], "cmd": c})
    m0 = min(len(dw), len(base[e["name"]]))
    return S1.causal_conv(H, v[:e["n"]]), D.to_gap(dw[:m0] - base[e["name"]][:m0], R, e["n"])


def cross(a, b, nfft=256):
    """→ (f, 幅值比, 相位差, 相干)。多轴按通道求和的互谱。"""
    from scipy.signal import csd, welch
    f, Shh = welch(a, FS, nperseg=nfft, axis=0)
    _, Sgg = welch(b, FS, nperseg=nfft, axis=0)
    _, Sgh = csd(a, b, FS, nperseg=nfft, axis=0)      # a=H 参照, b=G
    Shh, Sgg, Sgh = Shh.sum(1), Sgg.sum(1), Sgh.sum(1)
    sel = (f >= FLO) & (f <= FHI)
    coh = np.abs(Sgh) ** 2 / np.maximum(Shh * Sgg, 1e-30)
    return f[sel], np.sqrt(Sgg[sel] / np.maximum(Shh[sel], 1e-30)), \
        np.angle(Sgh[sel]), coh[sel]


def fit_tau(f, ph, w):
    """相位—频率线性拟合 → (τ 帧, R²)。`ph = −2π f τ Δt` ⇒ τ = −slope·FS/(2π)。"""
    p = np.unwrap(ph)
    A = np.vstack([f, np.ones_like(f)]).T
    W = np.diag(w)
    beta = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ p, rcond=None)[0]
    pred = A @ beta
    ss = float((w * (p - pred) ** 2).sum()) / max(float((w * (p - p.mean()) ** 2).sum()), 1e-30)
    return -beta[0] * FS / (2 * np.pi), 1 - ss


def shift_fd(x, tau):
    """频域整体提前 τ 帧：`X(f) · e^{+2πi f τ/FS}`。"""
    n = len(x)
    Xf = np.fft.rfft(x, axis=0)
    fr = np.fft.rfftfreq(n, 1 / FS)
    return np.fft.irfft(Xf * np.exp(2j * np.pi * fr * tau / FS)[:, None], n=n, axis=0)


def main() -> int:
    import gc
    gc.disable()
    E, R, _ = D.load_P0()
    g2 = G.G2()
    H = np.ascontiguousarray(np.load(ROOT / "outputs/step1_H/step1_H_v1.npz")["H"], float)
    pair.mu = np.concatenate([E[i]["e0"] for i in D.TRAIN]).mean(0)
    pair.sd = np.concatenate([E[i]["e0"] for i in D.TRAIN]).std(0) + 1e-12
    pair.Th = np.random.default_rng(GH.DIR_SEED).choice([-1.0, 1.0], size=(6, 3))
    base = {e["name"]: g2.omega(e["dd"]) for e in E}
    print(f"P0 {len(E)} 集 · 1 mrad RMS（不扫幅度）· R 与 ALIGN_L={D.ALIGN_L} 固定")
    print(f"主汇总只用 j1/j2/j3/j5/j6（**j4 出主汇总**，单独报告）")
    print(f"⚠ ab4 无时间戳 ⇒ G 前向假设均匀 20 Hz\n")

    # ── 在 ep0–2 拟合逐关节 τ，并定一个**共享**延迟 ──────────────────
    print(f"③④ 逐关节频域（拟合集 P0 ep0–2）")
    print(f"  {'关节':>5}{'幅值比中位':>12}{'相干中位':>10}{'τ 估计[帧]':>12}{'相位线性 R²':>13}")
    taus, fits = {}, {}
    for j in list(SUP) + [J4]:
        F, MR, PH, CO = [], [], [], []
        for i in FIT_EP:
            dh, dg = pair(E[i], g2, R, H, base, j)
            if dh is None:
                continue
            f, mr, ph, co = cross(dh, dg)
            F.append(f); MR.append(mr); PH.append(ph); CO.append(co)
        f = F[0]; mr = np.median(MR, 0); ph = np.mean(PH, 0); co = np.mean(CO, 0)
        tau, r2 = fit_tau(f, ph, co)
        taus[j], fits[j] = tau, r2
        tag = f"j{j+1}" + ("*" if j == J4 else "")
        print(f"  {tag:>5}{np.median(mr):>12.2f}{np.median(co):>10.3f}"
              f"{tau:>12.2f}{r2:>13.3f}")
    print(f"  *j4 域外外推，不进主汇总\n")

    sup_tau = np.array([taus[j] for j in SUP])
    w = np.ones(len(SUP))
    tau_shared = float(np.average(sup_tau, weights=w))
    spread = float(sup_tau.max() - sup_tau.min())
    lin_ok = all(fits[j] > 0.5 for j in SUP)
    print(f"  共享延迟 τ = **{tau_shared:.2f} 帧**（{tau_shared/FS*1000:.0f} ms）· "
          f"五关节离散 {spread:.2f} 帧 · 相位线性 R² 全 >0.5: {lin_ok}")

    # ── 固定共享 τ 后，仅在 ep3–4 验证（**禁止逐关节挑位移**）────────
    print(f"\n⑤ 固定共享 τ={tau_shared:.2f} 帧后，仅在 P0 ep3/ep4 验证")
    print(f"  {'集':>9}{'汇总 cos':>10}{'通过关节':>10}   逐关节 cos")
    val = {}
    for i in VAL_EP:
        A_, B_, per = [], [], {}
        for j in SUP:
            dh, dg = pair(E[i], g2, R, H, base, j)
            dgs = shift_fd(dg, tau_shared)
            a, b = GH.bp(dh, FLO, FHI), GH.bp(dgs, FLO, FHI)
            per[f"j{j+1}"] = float((a * b).sum()
                                   / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
            A_.append(a); B_.append(b)
        a, b = np.vstack(A_), np.vstack(B_)
        agg = float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
        npass = sum(1 for v in per.values() if v > COS_GATE)
        val[E[i]["name"]] = {"agg_cos": agg, "n_pass": npass, "per_joint": per}
        print(f"  {E[i]['name']:>9}{agg:>10.3f}{npass:>7}/5   "
              + " ".join(f"{k} {v:.2f}" for k, v in per.items()))

    ok = (lin_ok and spread <= TAU_SPREAD_MAX
          and all(v["agg_cos"] > COS_GATE and v["n_pass"] >= NPASS_MIN
                  for v in val.values()))
    print(f"\n  ── 判据（事前写死，三条须同时满足）──")
    print(f"  · 相位—频率近似线性（R²>0.5 全通过）: {lin_ok}")
    print(f"  · 五关节 τ 离散 ≤ {TAU_SPREAD_MAX} 帧: {spread:.2f} "
          f"{'✓' if spread <= TAU_SPREAD_MAX else '**✗**'}")
    print(f"  · ep3/ep4 汇总 cos > {COS_GATE} 且 ≥{NPASS_MIN}/5 关节通过: "
          f"{all(v['agg_cos'] > COS_GATE and v['n_pass'] >= NPASS_MIN for v in val.values())}")
    print(f"\n  **判决: {'统一时基问题 —— 可进入时基修正' if ok else '**增量结构 / 逐关节动态不匹配** —— 收口，不再靠移位修补'}**")
    OUT.write_text(json.dumps(
        {"pass_unified_timebase": bool(ok), "tau_shared_frames": tau_shared,
         "tau_per_joint": {f"j{j+1}": taus[j] for j in list(SUP) + [J4]},
         "phase_linearity_r2": {f"j{j+1}": fits[j] for j in list(SUP) + [J4]},
         "tau_spread_frames": spread, "validation": val,
         "gates": {"cos": COS_GATE, "tau_spread_max": TAU_SPREAD_MAX,
                   "n_pass_min": NPASS_MIN},
         "caliber": ("**j4 出主汇总**(pp2 无运动数据, 域外外推), 单独报告; "
                     "后续候选固定 δa_j4=0; 只用 1mrad RMS; R 与 ALIGN_L=3 固定, "
                     "主结果**不重新扫描位移**; 共享 τ 在 ep0-2 拟合后固定, "
                     "**禁止逐关节挑位移**, 仅在 ep3-4 验证; "
                     "③a 的 L=0 是 pp2/j5 主轴的模型—真机对齐, 与本次(ab4/五关节/G vs H "
                     "频域)对象不同, **不得预先称为矛盾**; ab4 无时间戳⇒假设均匀 20Hz"),
         }, indent=1, ensure_ascii=False))
    print(f"\n  → {OUT}")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
