#!/usr/bin/env python3
"""E-1 — 生产版 Ĝ 拟合与验收门。

结构(会话解耦):
  刚体+伺服: ω_rigid_sim(t) = Jr(q_t)·q̇̂(t),  q̇̂ = c_enc ∗ ȧ  (c_enc: 编码器独立拟合)
  柔性(仿真系, 会话无关): 残差 r_sim = Rᵀg − ω_rigid_sim,  激励 u = q̈̂
     模态式: s_m = k_m(ω_m,ζ_m) ∗ (w_mᵀu),  k_m[n]=e^{−ζωn dt}sin(ω_d n dt)/ω_d (解析冲激核, 可微)
     r̂_sim = Σ_m b_m s_m + D u
  使用: pred_real系(t) = R·(ω_rigid_sim + r̂_sim)
拟合: FIR脚手架(闭式)→频谱峰初始化→Adam精调(带通0.5-9Hz, 接触窗掩掉)。
验收门(预注册): ①gap-R²≥0.45 ②存在极点∈3.7±0.7Hz ③探针线性度<2% ④signed-cos≥0.747(GB基线)。
输出: outputs/p2_bakeoff/g_production.{json,npz}
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, filtfilt

from bakeoff_g import JOINTS, dual_r2, fit_c_enc, probe_model
from g_verdict import DT, KIN, LAGS, TRAIN_EPS, VAL_EPS, jr_at, load

torch.set_num_threads(4)
OUT = Path(__file__).resolve().parents[1] / "outputs/p2_bakeoff"
BP = butter(3, [0.5 / 10, 9.0 / 10], "bandpass")
BP28 = butter(3, [2.0 / 10, 8.0 / 10], "bandpass")
M_MODES, KKER = 3, 60


def contact_mask(ep):
    """夹爪开合±0.5s 掩码(与标签管线同法, 用指令夹爪列不可得 → 用 a 无夹爪; 退化: 全 False。
    真接触窗信息在标签npz里; 这里按数据集/集号读取。"""
    import numpy as np
    tag = ep["tag"]
    # 与 p2_labels 对齐(其长度=len−L, L=3): 补齐尾部
    try:
        idx = ep["_ep_idx"]
        z = np.load(f"../outputs/p2_labels/{tag}/ep{idx:03d}.npz")
        m = z["mask_contact"]
        full = np.zeros(len(ep["a"]), bool)
        full[:len(m)] = m
        return full
    except Exception:
        return np.zeros(len(ep["a"]), bool)


def rigid_parts(ep, c_enc):
    """q̇̂, q̈̂, ω_rigid_sim(仿真系), J_sim(t)。"""
    a = ep["a"]
    T = len(a)
    ad = np.vstack([np.zeros((1, 6)), np.diff(a, axis=0)]).astype(np.float32) / DT
    K = c_enc.shape[1]
    qdh = np.zeros((T, 6), np.float32)
    for k in range(K):
        qdh[k:] += c_enc[:, k][None, :] * ad[:T - k]
    qddh = np.vstack([np.zeros((1, 6)), np.diff(qdh, axis=0)]).astype(np.float32) / DT
    J = np.stack([jr_at(a[max(t - 2, 0)]) for t in range(T)]).astype(np.float32)  # 仿真系
    w_rigid = np.einsum("tcj,tj->tc", J, qdh)
    return qdh, qddh, w_rigid, J


def modal_kernels(omega, zeta, dt=DT, K=KKER):
    """解析冲激核 (M,K): 单位冲激进 ṡ 方程, 输出 s。可微。"""
    n = torch.arange(K, dtype=torch.float32)[None, :]
    om = omega[:, None]; ze = zeta[:, None]
    om_d = om * torch.sqrt(torch.clamp(1 - ze ** 2, min=1e-4))
    return torch.exp(-ze * om * n * dt) * torch.sin(om_d * n * dt) / om_d * dt


class FlexModal(torch.nn.Module):
    def __init__(self, f_init):
        super().__init__()
        self.log_om = torch.nn.Parameter(
            torch.log(torch.tensor(f_init, dtype=torch.float32) * 2 * np.pi))
        self.raw_ze = torch.nn.Parameter(torch.full((M_MODES,), -2.0))   # sigmoid→(0,0.5)
        self.W = torch.nn.Parameter(torch.randn(M_MODES, 6) * 0.1)
        self.B = torch.nn.Parameter(torch.randn(M_MODES, 3) * 0.1)
        self.D = torch.nn.Parameter(torch.zeros(3, 6))

    def forward(self, u):                                   # u (T,6) → r̂ (T,3)
        om = torch.exp(self.log_om)
        ze = torch.sigmoid(self.raw_ze) * 0.5
        ker = modal_kernels(om, ze)                          # (M,K)
        up = torch.einsum("mj,tj->tm", self.W, u)            # (T,M)
        x = up.T[None]                                       # (1,M,T)
        s = torch.nn.functional.conv1d(
            torch.nn.functional.pad(x, (KKER - 1, 0)),
            ker.flip(-1)[:, None, :], groups=M_MODES)[0].T   # (T,M) 因果卷积
        return s @ self.B + u @ self.D.T


def bp_t(x, ba):
    return torch.from_numpy(filtfilt(*ba, x.detach().numpy(), axis=0).copy()) \
        if isinstance(x, torch.Tensor) else filtfilt(*ba, x, axis=0)


def main():
    train = load(TRAIN_EPS, with_gap=True)
    val = load(VAL_EPS, with_gap=True)
    # 附集号供接触掩码定位(load 顺序: DATASETS×eps)
    per = len(TRAIN_EPS)
    for i, e in enumerate(train):
        e["_ep_idx"] = TRAIN_EPS[i % per]
    for i, e in enumerate(val):
        e["_ep_idx"] = VAL_EPS[i % len(VAL_EPS)]

    c_enc, S = fit_c_enc(train)
    print(f"c_enc 已拟(编码器裁判同源); 抽头能量前3滞后: "
          f"{np.round((c_enc**2).sum(0)[:3] / (c_enc**2).sum(), 2)}")

    # ---- 残差数据(仿真系) + FIR 脚手架找峰 ----
    def prep(eps):
        out = []
        for e in eps:
            qdh, qddh, w_rigid, J = rigid_parts(e, c_enc)
            g_sim = e["g"] @ e["R"]                        # Rᵀg (行向量右乘R)
            r = g_sim - w_rigid
            out.append({"u": qddh, "r": r, "mask": ~contact_mask(e), "ep": e,
                        "w_rigid": w_rigid, "J": J})
        return out
    ptr, pva = prep(train), prep(val)

    U = np.concatenate([bp_t(p["u"], BP)[p["mask"]] for p in ptr])
    Rr = np.concatenate([bp_t(p["r"], BP)[p["mask"]] for p in ptr])
    # 粗FIR(每输出通道对6输入×20抽头) → 频谱峰
    Kf = 20
    Xf = []
    for p in ptr:
        ub = bp_t(p["u"], BP)
        T = len(ub)
        X = np.zeros((T, 6 * Kf), np.float32)
        for k in range(Kf):
            X[k:, k::Kf] = ub[:T - k]
        Xf.append(X[p["mask"]])
    Xf = np.concatenate(Xf)
    Wfir, *_ = np.linalg.lstsq(Xf, Rr, rcond=None)
    kern = Wfir.reshape(6, Kf, 3)
    spec = np.abs(np.fft.rfft(kern.sum(0), n=64, axis=0))[:, :].sum(1)
    freqs = np.fft.rfftfreq(64, DT)
    band = (freqs > 1.5) & (freqs < 9)
    pk = freqs[band][np.argsort(spec[band])[::-1][:M_MODES]]
    print(f"FIR脚手架谱峰(模式初始化): {np.round(np.sort(pk), 2)} Hz")

    # ---- 模态精调 ----
    flex = FlexModal(sorted(pk))
    opt = torch.optim.Adam(flex.parameters(), lr=3e-2)
    uts = [torch.from_numpy(np.ascontiguousarray(p["u"])) for p in ptr]
    rts = [torch.from_numpy(np.ascontiguousarray(bp_t(p["r"], BP))) for p in ptr]
    mks = [torch.from_numpy(p["mask"]) for p in ptr]
    # 目标 r 已零相位带通(0.5-9Hz); pred 由带限谐振核+D·u 组成, 天然近带限 →
    # 直接时域 MSE(接触窗掩掉)。若 D 项渗入低频, 验收的 band28 指标会暴露。
    for it in range(400):
        loss = 0.0
        for u, r, m in zip(uts, rts, mks):
            d = flex(u) - r
            loss = loss + (d[m] ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % 100 == 0:
            print(f"  flex iter {it+1}: loss={float(loss):.5f}  "
                  f"极点={np.round(np.sort(torch.exp(flex.log_om).detach().numpy()/2/np.pi), 2)} Hz",
                  flush=True)

    # ---- 阶段3: 联合精调(c 抽头 + 模态参数, 陀螺目标端到端; 全线性→探针品质不变) ----
    Kc = c_enc.shape[1]
    # 变体开关(2026-07-15 用户裁决: A=基座): A=编码器c+阶段2柔性(导数冠军, 默认);
    # C=交替闭式(值略高但柔性糊/相位差, 已否, 留作对照)。
    import os
    ALT_ROUNDS = 2 if os.environ.get("G_VARIANT", "A").upper() == "C" else 0
    # 交替闭式坐标下降: c ← 陀螺LS(目标扣掉flex) ⇄ flex ← 残差重拟。每步全局最优。
    def ls_c(target_list):
        Xs, ys = [], []
        for p, tgt in zip(ptr, target_list):
            e = p["ep"]; a = e["a"]; T = len(a)
            ad = (np.vstack([np.zeros((1, 6)), np.diff(a, axis=0)]) / DT).astype(np.float32)
            X = np.zeros((T, 3, 6, Kc0), np.float32)
            for k in range(Kc0):
                sh = np.vstack([np.zeros((k, 6), np.float32), ad[:T - k]])
                X[:, :, :, k] = p["J"] * sh[:, None, :]
            Xs.append(X[p["mask"]].reshape(-1, 6 * Kc0))
            ys.append(tgt[p["mask"]].reshape(-1))
        w, *_ = np.linalg.lstsq(np.concatenate(Xs), np.concatenate(ys), rcond=None)
        return w.reshape(6, Kc0)

    Kc0 = c_enc.shape[1]
    g_sims = [(p["ep"]["g"] @ p["ep"]["R"]).astype(np.float32) for p in ptr]
    c_use = c_enc.copy()
    for rnd in range(ALT_ROUNDS):
        # flex 预测(用当前 c 的 q̈̂)
        flex_preds = []
        for p in ptr:
            _, qddh, _, _ = rigid_parts(p["ep"], c_use)
            with torch.no_grad():
                flex_preds.append(flex(torch.from_numpy(np.ascontiguousarray(qddh))).numpy())
        c_use = ls_c([g - fp for g, fp in zip(g_sims, flex_preds)])
        # flex 重拟(新 c 的残差, 带通口径同阶段2)
        rts2, uts2 = [], []
        for p, g in zip(ptr, g_sims):
            _, qddh, w_rig, _ = rigid_parts(p["ep"], c_use)
            rts2.append(torch.from_numpy(bp_t(g - w_rig, BP).astype(np.float32).copy()))
            uts2.append(torch.from_numpy(np.ascontiguousarray(qddh)))
        opt2 = torch.optim.Adam(flex.parameters(), lr=1e-2)
        for it in range(200):
            loss = 0.0
            for u, r, m in zip(uts2, rts2, mks):
                d = flex(u) - r
                loss = loss + (d[m] ** 2).mean()
            opt2.zero_grad(); loss.backward(); opt2.step()
        print(f"  交替轮{rnd+1}: flex_loss={float(loss):.5f}  "
              f"极点={np.round(np.sort(torch.exp(flex.log_om).detach().numpy()/2/np.pi), 2)} Hz",
              flush=True)

    poles = np.sort(torch.exp(flex.log_om).detach().numpy() / (2 * np.pi))
    zetas = (torch.sigmoid(flex.raw_ze) * 0.5).detach().numpy()

    # ---- 生产版预测函数(真机系) + 验收 ----
    def make_pred(ep):
        def f(a):
            e2 = dict(ep); e2["a"] = a
            qdh, qddh, w_rigid, _ = rigid_parts(e2, c_use)
            with torch.no_grad():
                r = flex(torch.from_numpy(np.ascontiguousarray(qddh))).numpy()
            return (w_rigid + r) @ ep["R"].T
        return f

    res = {"poles_hz": poles.tolist(), "zetas": zetas.tolist()}
    r2s_full, r2s_rig, band28 = [], [], []
    for p in pva:
        ep = p["ep"]
        f = make_pred(ep)
        pred = f(ep["a"])
        _, _, w_rig_fit, _ = rigid_parts(ep, c_use)
        pred_rig = w_rig_fit @ ep["R"].T
        r2s_full.append(dual_r2(pred, ep))
        r2s_rig.append(dual_r2(pred_rig, ep))
        gb, pb = bp_t(ep["gap"], BP28), bp_t(pred - (ep["g"] - ep["gap"]), BP28)
        band28.append(1 - ((pb - gb) ** 2).sum() / ((gb - gb.mean(0)) ** 2).sum())
    res["r2_raw_full"], res["r2_gap_full"] = [float(np.mean([r[i] for r in r2s_full])) for i in (0, 1)]
    res["r2_gap_rigid"] = float(np.mean([r[1] for r in r2s_rig]))
    res["r2_band28_gap"] = float(np.mean(band28))

    rng = np.random.default_rng(0)
    recs = []
    for p in pva:
        ep = p["ep"]
        sites = rng.integers(KIN, len(ep["a"]) - LAGS - 1, 10)
        recs += probe_model(make_pred(ep), ep, sites, S)
    res["signed"] = float(np.median([r["signed"] for r in recs]))
    res["anti_pct"] = float(np.mean([r["anti"] for r in recs]) * 100)
    res["lin_pct"] = float(np.median([r["lin"] for r in recs]) * 100)
    res["asym_pct"] = float(np.median([r["asym"] for r in recs]) * 100)
    res["mag"] = float(np.median([r["mag"] for r in recs]))

    gate = {"①gap-R²≥0.45": res["r2_gap_full"] >= 0.45,
            "②极点∈3.7±0.7Hz": bool(np.any(np.abs(poles - 3.7) <= 0.7)),
            "③线性度<2%": res["lin_pct"] < 2.0,
            "④signed≥0.747": res["signed"] >= 0.747}
    res["gate"] = {k: bool(v) for k, v in gate.items()}
    res["gate_pass"] = all(gate.values())

    np.savez(OUT / "g_production.npz", c_enc=c_enc, c_fit=c_use,
             log_om=flex.log_om.detach().numpy(), raw_ze=flex.raw_ze.detach().numpy(),
             W=flex.W.detach().numpy(), B=flex.B.detach().numpy(), D=flex.D.detach().numpy())
    (OUT / "g_production.json").write_text(json.dumps(res, indent=1))
    print(f"\n== 生产版 Ĝ 验收 ==")
    print(f"极点: {np.round(poles,2)} Hz (ζ={np.round(zetas,3)})")
    print(f"R²gap: 仅刚体={res['r2_gap_rigid']:.3f} → +柔性={res['r2_gap_full']:.3f}  "
          f"(2-8Hz带 gap-R²={res['r2_band28_gap']:.3f})")
    print(f"探针: signed={res['signed']:.3f} 反相={res['anti_pct']:.0f}% "
          f"线性度={res['lin_pct']:.1f}% 奇对称={res['asym_pct']:.1f}% mag={res['mag']:.4f}")
    for k, v in gate.items():
        print(f"  {'✓' if v else '✗'} {k}")
    print("→ 验收", "通过" if res["gate_pass"] else "未通过", flush=True)


if __name__ == "__main__":
    main()
