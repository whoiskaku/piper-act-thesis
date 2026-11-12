#!/usr/bin/env python3
"""**零机时门禁**：严格因果 + chunk 钳位 + 接触门控 + ε 约束下，
**加入实时 IMU 反馈**之后，δa 通道到底能交付多少？

为什么这不是重跑 Step-1
-----------------------
`reports/STEP1_RESULT_v1.md` §2.3-bis 已经跑过「正确因果滚动解」：
**3.16%（±2mrad）/ 4.34%（±3mrad，6.5% 触界）**。
但那一版的信息集**只有指令**（`W` 从 36 维指令特征预测 gap，留出可预测性 31.6%）——
它是**纯前馈**，而纯前馈正是已判否的 Step-1 路线。

本脚本唯一的合法差别是**信息集多了「实测到当前帧的真机 IMU」**：

    命令时刻 t 可见:  x̃[t]（chunk 钳位的指令 horizon）
                      + {e₀[s] : s ≤ t − ALIGN_L}      ← **新增**
    （因为 gap[t] = real[t+3] − ideal[t]，墙钟 t 只知道 real 到 t）

⇒ 预测 e₀[t+k] 的**前瞻**是 k+3 帧。在 2 Hz 处 3 帧 = 0.3 周期（好预测），
   9 Hz 处 = 1.35 周期（基本不可预测）⇒ **反馈的收益必然集中在低频**
   （与 `RESEARCH_PLAN_v5` §3 已记的「反馈带宽受 20 Hz/Nyquist 限」一致）。

结构（严格因果，逐帧滚动）
--------------------------
    ê₀[t+k] = W·x[t+k]  +  C_{k+3}·[r[t−3], …, r[t−3−J]]      r = e₀ − W·x
    d[t]    = argmin ‖ê₀[t:t+K] + resp[t:t+K] + H[0:K]·d‖² + λ‖d‖²
              s.t. |d| ≤ ε,  d = 0 在接触窗
    K       = min(M, cap[t] − t)                              ← chunk 钳位
    resp    = 已提交的 δa[<t] 的响应尾巴                        ← 正确滚动，不置零

**没有任何一步用到 t+k 的真实 e₀。** 用户 2026-09-02 明确要求：
「不能用整集未来的 e₀ 解一个非因果最优解，再让因果 π 模仿」。

⚠ 数据卫生（必须并读）
----------------------
ab4 的 23 集**没有一集是干净留出**：P1 拟合过 H、P0 ep0–2 进过 CV、
P0 ep3–4 已被 §2.3 的诊断用于模型选择、P2/P3 用过 β 门。
故本脚本是 **go/no-go 筛查**，**不是验证**。
报告逐集留一（LOO）折外数，但**不得**称之为独立留出。

⚠ e₀ 的口径
-----------
ab4 是**开环复放、没有 FF**，故此处 e₀ 是**裸墙**。
而候选 C = FF + 残差，实际要修的是**FF 之后**的墙（已小 33% 且形状不同）。
两者不可混称。

判读（用户 2026-09-02 定）
--------------------------
  · 仍只有 3–4%  ⇒ **立即停止**
  · 约 20%       ⇒ 才生成 δa* 并训练策略
（20% 来自实测传递率：离线→真机约剩 70%，真机要 15% 则离线需 ≈21%。）

用法:
    python -u sim/oracle_causal_feedback.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fit_step1_candidate as S1                                   # noqa: E402

H_ART = ROOT / "outputs/step1_H/step1_H_v1.npz"
OUT = ROOT / "outputs/step1_H/oracle_causal_feedback.json"
FB_LAGS = 8                    # J：反馈历史用 J+1 帧
W_MULT = 0.001                 # 与冻结 sidecar 同
FB_MULT = 1e-3                 # 反馈项 ridge（相对尺度）
EPS_LIST = (0.002, 0.003)      # ±2 / ±3 mrad
K_MULTS = (1.0, 0.01)          # 冻结值 / ×0.01（与 §2.3-bis 两行对齐）


def load_all():
    """→ [{name, e0, x, xt, cap, contact}]，e₀ 已扣掉 P1 的探针贡献。"""
    z = np.load(H_ART, allow_pickle=False)
    H = np.ascontiguousarray(z["H"], dtype=np.float64)
    logs = S1.discover_logs(ROOT / "outputs/ab4", ("P0", "P1"))
    man = json.loads((H_ART.with_suffix(".json")).read_text())
    calib = json.loads(Path(man["calibration"]["path"]).read_text())
    R = np.asarray(calib["R"], float)
    b = np.asarray(calib.get("b", [0, 0, 0]), float)
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    rend = IdealChunkRenderer()
    E = []
    for tier in ("P0", "P1"):
        for p in logs[tier]:
            ep = S1.load_episode(p, tier, R, b, rend, 550)
            pay = json.loads(p.read_text())
            _, _, cap = S1._render_logged_chunks(pay.get("chunks", []), rend,
                                                 len(pay.get("frames", [])))
            n = len(ep.gap)
            E.append({"name": f"{tier}_ep{ep.ep}", "tier": tier,
                      "e0": S1.baseline_target(ep, H),
                      "x": np.asarray(ep.x, float), "xt": np.asarray(ep.x_tilde, float),
                      "cap": np.asarray(cap[:n], int),
                      "contact": np.asarray(ep.contact_mask, bool)})
    return H, E


def fit_w(train):
    X = np.concatenate([e["x"] for e in train])
    Y = np.concatenate([e["e0"] for e in train])
    coef, _ = S1._ridge_solve(X, Y, W_MULT)
    return coef.T                                            # (3,36)


def fb_hist(r, t, j0=S1.ALIGN_L):
    """r[t−3], r[t−4], …（越界补零），拼成 3(J+1) 维。"""
    idx = [t - j0 - j for j in range(FB_LAGS + 1)]
    return np.concatenate([r[i] if i >= 0 else np.zeros(3) for i in idx])


def fit_fb(train, W):
    """按前瞻 ℓ = 3…3+M 各拟合一个 C_ℓ：从 r 的可见历史预测 r[s+ℓ]。"""
    Cs = {}
    for lead in range(S1.ALIGN_L, S1.ALIGN_L + S1.N_LAGS):
        Xs, Ys = [], []
        for e in train:
            r = e["e0"] - e["x"] @ W.T
            T = len(r)
            for s in range(FB_LAGS + S1.ALIGN_L, T - lead):
                Xs.append(fb_hist(r, s + S1.ALIGN_L))       # 可见到 s
                Ys.append(r[s + lead])
        coef, _ = S1._ridge_solve(np.asarray(Xs), np.asarray(Ys), FB_MULT)
        Cs[lead] = coef.T                                    # (3, 3(J+1))
    return Cs


def roll(e, H, W, Cs, eps, k_mult):
    """严格因果滚动解 → (δa, sat_pct)。Cs=None 即纯前馈臂。"""
    T = len(e["e0"])
    M1 = S1.N_LAGS
    hs_full = H.reshape(M1 * 3, 6)
    scale = max(float(np.trace(hs_full.T @ hs_full)) / 6.0, np.finfo(float).tiny)
    lam = k_mult * scale
    da = np.zeros((T, 6))
    resp = np.zeros((T + M1, 3))
    r = np.zeros((T, 3))                       # 反馈用的 FF 残差（**只在可见处填**）
    e0 = e["e0"]
    ff = e["x"] @ W.T
    nsat = 0
    for t in range(T):
        s = t - S1.ALIGN_L                     # 可见到 s
        if s >= 0:
            r[s] = e0[s] - ff[s] - resp[s]     # 扣掉自己已施加的贡献 ⇒ 纯扰动观测
        K = int(min(S1.M, e["cap"][t] - t))
        if K < 0:
            K = 0
        tgt = np.empty((K + 1, 3))
        for k in range(K + 1):
            j = min(t + k, T - 1)
            pred = ff[j].copy()
            if Cs is not None and s >= 0:
                pred = pred + Cs[min(k + S1.ALIGN_L, S1.ALIGN_L + S1.M)] @ fb_hist(r, t)
            tgt[k] = pred + resp[t + k]
        hs = H[:K + 1].reshape((K + 1) * 3, 6)
        d = -np.linalg.solve(hs.T @ hs + lam * np.eye(6), hs.T @ tgt.reshape(-1))
        nsat += int(np.any(np.abs(d) > eps))
        d = np.clip(d, -eps, eps)
        if e["contact"][t]:
            d[:] = 0.0
        da[t] = d
        for k in range(M1):
            if t + k < T + M1:
                resp[t + k] += H[k] @ d
    return da, nsat / max(T, 1)


def score(E, H, arms):
    """→ 每臂 (降幅, δa RMS, 触界%)，按能量池化（带通 0.5–9 Hz）。"""
    out = {}
    for name, das in arms.items():
        num = den = 0.0
        rms, sat = [], []
        for e, (da, sp) in zip(E, das):
            base = S1.bandpass41(e["e0"])
            new = S1.bandpass41(e["e0"] + S1.causal_conv(H, da))
            num += float(np.square(new).sum()); den += float(np.square(base).sum())
            rms.append(float(np.sqrt(np.mean(np.square(da)))) * 1e3)
            sat.append(sp * 100)
        out[name] = (1 - num / max(den, 1e-30), float(np.mean(rms)), float(np.mean(sat)))
    return out


def main() -> int:
    import gc
    gc.disable()
    H, E = load_all()
    print(f"ab4 载入 {len(E)} 集（P0 5 + P1 8）—— **无一集是干净留出**，本表是筛查非验证")
    print(f"  H 来自 {H_ART.name}（float64，与 step1-linear-v1 逐比特一致）")
    print(f"  反馈历史 J+1 = {FB_LAGS+1} 帧，可见到 t−{S1.ALIGN_L}；"
          f"预测前瞻 {S1.ALIGN_L}–{S1.ALIGN_L+S1.M} 帧\n")

    res = {}
    for k_mult in K_MULTS:
        for eps in EPS_LIST:
            arms = {"前馈(对照)": [], "前馈+反馈": []}
            for i, e in enumerate(E):                      # 逐集留一
                tr = [E[j] for j in range(len(E)) if j != i]
                W = fit_w(tr)
                Cs = fit_fb(tr, W)
                arms["前馈(对照)"].append(roll(e, H, W, None, eps, k_mult))
                arms["前馈+反馈"].append(roll(e, H, W, Cs, eps, k_mult))
            s = score(E, H, arms)
            tag = f"λ_K×{k_mult:g}, ±{eps*1e3:.0f}mrad"
            res[tag] = {k: {"reduction": v[0], "da_rms_mrad": v[1], "sat_pct": v[2]}
                        for k, v in s.items()}
            print(f"  {tag}")
            for k, v in s.items():
                print(f"    {k:<12} 降幅 {v[0]*100:>6.2f}%   δa RMS {v[1]:.2f} mrad"
                      f"   触界 {v[2]:.1f}%")
            print()

    ctl = res[f"λ_K×1, ±2mrad"]["前馈(对照)"]["reduction"] * 100
    fb = max(v["前馈+反馈"]["reduction"] for v in res.values()) * 100
    print(f"  ── 实现自检 ──")
    print(f"  前馈对照（λ_K 冻结, ±2mrad）= {ctl:.2f}%   "
          f"记录值 3.16%   差 {ctl-3.16:+.2f} pp")
    print(f"  {'✓ 实现可信' if abs(ctl-3.16) < 1.5 else '**✗ 与记录值偏离过大，先查实现再读反馈臂**'}")
    print(f"\n  ── 判读 ──")
    print(f"  前馈+反馈的最好一档 = **{fb:.2f}%**   门槛 ≈20%")
    print(f"  {'**≥20%：可以生成 δa* 并训练策略**' if fb >= 20 else ('**仍在 3–4% 量级 ⇒ 立即停止**' if fb < 6 else '**介于两者之间 ⇒ 需用户裁决**')}")
    OUT.write_text(json.dumps({
        "arms": res, "fb_lags": FB_LAGS + 1, "align_L": S1.ALIGN_L,
        "metric_band_hz": list(S1.METRIC_BAND), "protocol": "leave-one-episode-out",
        "caliber": ("ab4 开环复放、**无 FF**，故 e0 是裸墙；候选 C 实际修的是 FF 之后的墙。"
                    "ab4 23 集无一是干净留出（P1 拟合 H / P0 ep0-2 进 CV / P0 ep3-4 被诊断用于"
                    "模型选择 / P2P3 用过 β 门）⇒ 本表是 go/no-go 筛查，不是验证。"),
        "strict_causality": ("ê0[t+k] 只用 x̃[t]（chunk 钳位）与 e0[s<=t-3]；"
                             "滚动解含已提交 δa 的响应尾巴；从不使用 t+k 的真实 e0。"),
    }, indent=1, ensure_ascii=False))
    print(f"\n  → {OUT}")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
