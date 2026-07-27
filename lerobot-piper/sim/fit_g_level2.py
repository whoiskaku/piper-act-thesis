#!/usr/bin/env python3
"""E-1c — Ĝ Level-2: 位形调度增益 + 激励修复(在变体A基座上)。

捆绑(2026-07-15 用户批准):
  修复: u = c_enc ∗ acc20,  acc20 = 200Hz样条精确 q̈_cmd → AA降采20Hz (换掉双差分)
  调度: 模态输入增益 α_m(t) = softplus(θ_mᵀ φ(t)),
        φ = [1, q2, q3, q2², q3², q2·q3, spd, grip] (名义轨迹, 标准化, 对δa冻结
        → 扰动通道严格保线性 — 探针线性度应保持≈0)
消融: A(0.397, 已录) / A+激励修复(不调度) / L2全量 — 隔离两件的贡献。
验收门同 E-1; 另做位形3×3分箱复查(0.32↔0.74 落差是否抹平)。
输出: outputs/p2_bakeoff/g_level2.{json,npz}; 结果追加 reports/E1_G_identification.md §6
"""
from __future__ import annotations
import json

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt

from bakeoff_g import dual_r2, fit_c_enc, probe_model
from compare_ideal_methods import load_episode
from fit_g_production import BP, BP28, OUT, FlexModal, bp_t, modal_kernels, rigid_parts
from g_verdict import DATASETS, DT, KIN, LAGS, TRAIN_EPS, VAL_EPS, load

torch.set_num_threads(4)
M2 = 3
FEAT_NAMES = ["1", "q2", "q3", "q2²", "q3²", "q2q3", "spd", "grip"]


def cmd_accel_20(a, fps=20.0, hi=200.0):
    """200Hz 样条精确加速度 → AA 降采 20Hz。"""
    T = len(a)
    t = np.arange(T) / fps
    sp = CubicSpline(t, a, axis=0)
    th = np.arange(int(round(t[-1] * hi)) + 1) / hi
    acc = sp(th, 2)
    ba = butter(4, 0.45 * fps / (hi / 2))
    return filtfilt(*ba, acc, axis=0)[::int(hi // fps)][:T].astype(np.float32)


def excitation(a, c_enc):
    """u = c_enc ∗ acc20 (伺服滤波后的精确关节加速度)。"""
    acc = cmd_accel_20(a)
    K = c_enc.shape[1]
    u = np.zeros_like(acc)
    for k in range(K):
        u[k:] += c_enc[:, k][None, :] * acc[:len(a) - k]
    return u


def raw_feats(a):
    q2, q3 = a[:, 1], a[:, 2]
    ad = np.vstack([np.zeros((1, 6)), np.diff(a[:, :6], axis=0)]) / DT
    spd = np.convolve(np.linalg.norm(ad, axis=1), np.ones(5) / 5, "same")
    grip = np.clip(a[:, 6] / 0.035, -2, 2) if a.shape[1] > 6 else np.zeros(len(a))
    return np.stack([np.ones(len(a)), q2, q3, q2 ** 2, q3 ** 2, q2 * q3, spd, grip], 1)


class FlexL2(torch.nn.Module):
    def __init__(self, f_init, nfeat=8, M=M2):
        super().__init__()
        self.log_om = torch.nn.Parameter(
            torch.log(torch.tensor(f_init, dtype=torch.float32) * 2 * np.pi))
        self.raw_ze = torch.nn.Parameter(torch.full((M,), -2.0))
        self.W = torch.nn.Parameter(torch.randn(M, 6) * 0.1)
        self.B = torch.nn.Parameter(torch.randn(M, 3) * 0.1)
        self.D = torch.nn.Parameter(torch.zeros(3, 6))
        th = torch.zeros(M, nfeat)
        th[:, 0] = 0.5413                      # softplus(0.5413)≈1 → 初始=无调度
        self.theta = torch.nn.Parameter(th)

    def forward(self, u, phi):                 # u (T,6), phi (T,F)
        om = torch.exp(self.log_om)
        ze = torch.sigmoid(self.raw_ze) * 0.5
        ker = modal_kernels(om, ze)
        alpha = torch.nn.functional.softplus(phi @ self.theta.T)   # (T,M)
        up = torch.einsum("mj,tj->tm", self.W, u) * alpha
        x = up.T[None]
        s = torch.nn.functional.conv1d(
            torch.nn.functional.pad(x, (ker.shape[1] - 1, 0)),
            ker.flip(-1)[:, None, :], groups=up.shape[1])[0].T
        return s @ self.B + u @ self.D.T


def contact_mask(tag, ei, n):
    try:
        z = np.load(f"../outputs/p2_labels/{tag}/ep{ei:03d}.npz")
        m = z["mask_contact"]
        full = np.zeros(n, bool)
        full[:len(m)] = m
        return full
    except Exception:
        return np.zeros(n, bool)


def fit_flex(module, uts, rts, phis, mks, iters=600, lr=1e-2):
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    for it in range(iters):
        loss = 0.0
        for u, r, ph, m in zip(uts, rts, phis, mks):
            d = module(u, ph) - r
            loss = loss + (d[m] ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % 200 == 0:
            print(f"    iter {it+1}: loss={loss.item():.5f} "
                  f"极点={np.round(np.sort(torch.exp(module.log_om).detach().numpy()/2/np.pi),2)}Hz",
                  flush=True)
    return module


def main():
    train = load(TRAIN_EPS, with_gap=True)
    val = load(VAL_EPS, with_gap=True)
    k = 0
    for grp, eplist in [(train, TRAIN_EPS), (val, VAL_EPS)]:
        pass
    # 附夹爪列+集号
    idx = 0
    for repo, tag in DATASETS:
        for ei in TRAIN_EPS:
            d = load_episode(repo, ei)
            train[idx]["a7"] = d["action"][:len(train[idx]["a"])].astype(np.float32)
            train[idx]["_ep"] = ei
            idx += 1
    idx = 0
    for repo, tag in DATASETS:
        for ei in VAL_EPS:
            d = load_episode(repo, ei)
            val[idx]["a7"] = d["action"][:len(val[idx]["a"])].astype(np.float32)
            val[idx]["_ep"] = ei
            idx += 1

    c_enc, S = fit_c_enc(train)

    # 特征标准化统计(训练集, 常数列除外)
    F_all = np.concatenate([raw_feats(e["a7"]) for e in train])
    mu, sd = F_all.mean(0), F_all.std(0) + 1e-8
    mu[0], sd[0] = 0.0, 1.0

    def feats(e):
        return ((raw_feats(e["a7"]) - mu) / sd).astype(np.float32)

    def prep(eps):
        out = []
        for e in eps:
            _, _, w_rigid, _ = rigid_parts(e, c_enc)
            u = excitation(e["a"], c_enc)
            g_sim = (e["g"] @ e["R"]).astype(np.float32)
            r = bp_t(g_sim - w_rigid, BP).astype(np.float32)
            m = ~contact_mask(e["tag"], e["_ep"], len(e["a"]))
            out.append({"e": e, "u": torch.from_numpy(u.copy()),
                        "r": torch.from_numpy(r.copy()),
                        "phi": torch.from_numpy(feats(e)),
                        "m": torch.from_numpy(m), "w_rigid": w_rigid})
        return out
    ptr, pva = prep(train), prep(val)

    # FIR 脚手架(新激励)找峰
    Kf = 20
    Xs, ys = [], []
    for p in ptr:
        ub = bp_t(p["u"].numpy(), BP)
        T = len(ub)
        X = np.zeros((T, 6 * Kf), np.float32)
        for kk in range(Kf):
            X[kk:, kk::Kf] = ub[:T - kk]
        Xs.append(X[p["m"].numpy()])
        ys.append(p["r"].numpy()[p["m"].numpy()])
    Wfir, *_ = np.linalg.lstsq(np.concatenate(Xs), np.concatenate(ys), rcond=None)
    kern = Wfir.reshape(6, Kf, 3)
    spec = np.abs(np.fft.rfft(kern.sum(0), n=64, axis=0)).sum(1)
    freqs = np.fft.rfftfreq(64, DT)
    band = (freqs > 1.5) & (freqs < 9.5)
    pk = np.sort(freqs[band][np.argsort(spec[band])[::-1][:M2]])
    print(f"新激励 FIR 谱峰: {np.round(pk, 2)} Hz")

    uts = [p["u"] for p in ptr]; rts = [p["r"] for p in ptr]
    phis = [p["phi"] for p in ptr]; mks = [p["m"] for p in ptr]
    phis0 = [torch.zeros_like(p) for p in phis]
    for p0 in phis0:
        p0[:, 0] = 1.0                          # 无调度: 只留常数列

    results = {}

    def evaluate(name, module, use_phi=True):
        r2s, band28s = [], []
        for p in pva:
            e = p["e"]
            n_ = len(p["phi"])
            phi_t = p["phi"] if use_phi else torch.cat(
                [torch.ones(n_, 1), torch.zeros(n_, 7)], 1)
            with torch.no_grad():
                r = module(p["u"], phi_t).numpy()
            pred = (p["w_rigid"] + r) @ e["R"].T
            r2s.append(dual_r2(pred, e))
            gb = bp_t(e["gap"], BP28)
            pb = bp_t(pred - (e["g"] - e["gap"]), BP28)
            band28s.append(1 - ((pb - gb) ** 2).sum() / ((gb - gb.mean(0)) ** 2).sum())
        poles = np.sort(torch.exp(module.log_om).detach().numpy() / (2 * np.pi))
        rec = {"r2_gap": float(np.mean([x[1] for x in r2s])),
               "r2_raw": float(np.mean([x[0] for x in r2s])),
               "band28": float(np.mean(band28s)), "poles": poles.tolist()}
        results[name] = rec
        print(f"[{name}] gap-R²={rec['r2_gap']:.3f} band28={rec['band28']:.3f} "
              f"极点={np.round(poles,2)}Hz", flush=True)
        return rec

    # --- 消融1: A+激励修复(不调度) ---
    print("== A+激励修复(不调度) ==")
    fx1 = FlexL2(pk.tolist())
    fx1 = fit_flex(fx1, uts, rts, phis0, mks)
    evaluate("A+激励修复", fx1, use_phi=False)

    # --- L2 全量: 调度+修复 ---
    print("== L2 全量(调度+修复) ==")
    fx2 = FlexL2(pk.tolist())
    fx2 = fit_flex(fx2, uts, rts, phis, mks)
    rec = evaluate("L2全量", fx2, use_phi=True)

    # --- 探针(调度冻结于名义轨迹 → 线性度应≈0) ---
    def make_pred(p):
        e = p["e"]
        phi_t = p["phi"]
        def f(a):
            e2 = dict(e); e2["a"] = a
            _, _, w_rigid, _ = rigid_parts(e2, c_enc)
            u = torch.from_numpy(excitation(a, c_enc).copy())
            with torch.no_grad():
                r = fx2(u, phi_t).numpy()
            return (w_rigid + r) @ e["R"].T
        return f
    rng = np.random.default_rng(0)
    recs = []
    for p in pva:
        sites = rng.integers(KIN, len(p["e"]["a"]) - LAGS - 1, 10)
        recs += probe_model(make_pred(p), p["e"], sites, S)
    probe = {"signed": float(np.median([r["signed"] for r in recs])),
             "anti_pct": float(np.mean([r["anti"] for r in recs]) * 100),
             "lin_pct": float(np.median([r["lin"] for r in recs]) * 100),
             "asym_pct": float(np.median([r["asym"] for r in recs]) * 100)}
    results["L2全量"].update(probe)
    print(f"探针: signed={probe['signed']:.3f} 反相={probe['anti_pct']:.0f}% "
          f"线性度={probe['lin_pct']:.1f}% 奇对称={probe['asym_pct']:.1f}%")

    # --- 位形分箱复查(train+val 全体, 与画像同法) ---
    allp = ptr + pva
    q2a = np.concatenate([p["e"]["a"][:, 1] for p in allp])
    q3a = np.concatenate([p["e"]["a"][:, 2] for p in allp])
    e2 = np.quantile(q2a, [0, 1/3, 2/3, 1]); e3 = np.quantile(q3a, [0, 1/3, 2/3, 1])
    rhos, gaps, masks2 = [], [], []
    for p in allp:
        e = p["e"]
        with torch.no_grad():
            r = fx2(p["u"], p["phi"]).numpy()
        pred = (p["w_rigid"] + r) @ e["R"].T
        rhos.append(e["g"] - pred); gaps.append(e["gap"])
    print("位形3×3复查(U值):")
    grid = []
    for i in range(3):
        row = []
        for j in range(3):
            num = den = 0.0
            for p, rho, gap in zip(allp, rhos, gaps):
                a = p["e"]["a"]
                mk = ((a[:, 1] >= e2[i]) & (a[:, 1] <= e2[i+1]) &
                      (a[:, 2] >= e3[j]) & (a[:, 2] <= e3[j+1]))
                num += (rho[mk] ** 2).sum(); den += (gap[mk] ** 2).sum()
            row.append(num / max(den, 1e-12))
        grid.append([float(v) for v in row])
        print("   " + "  ".join(f"{v:.2f}" for v in row))
    results["pose_grid_after"] = grid

    gate = {"①gap-R²≥0.45": rec["r2_gap"] >= 0.45,
            "②极点∈3.7±0.7Hz": bool(np.any(np.abs(np.array(rec["poles"]) - 3.7) <= 0.7)),
            "③线性度<2%": probe["lin_pct"] < 2.0,
            "④signed≥0.747": probe["signed"] >= 0.747}
    results["gate"] = {k: bool(v) for k, v in gate.items()}
    results["gate_pass"] = all(gate.values())
    for k, v in gate.items():
        print(f"  {'✓' if v else '✗'} {k}")
    print("→ 验收", "通过" if results["gate_pass"] else "未通过")

    np.savez(OUT / "g_level2.npz", c_enc=c_enc, mu=mu, sd=sd,
             log_om=fx2.log_om.detach().numpy(), raw_ze=fx2.raw_ze.detach().numpy(),
             W=fx2.W.detach().numpy(), B=fx2.B.detach().numpy(),
             D=fx2.D.detach().numpy(), theta=fx2.theta.detach().numpy())
    (OUT / "g_level2.json").write_text(json.dumps(results, indent=1))
    print(f"→ {OUT/'g_level2.json'}")


if __name__ == "__main__":
    main()
