#!/usr/bin/env python3
"""E-2 — π 对练训练框架(v1: 开环特征/闭环目标, 已向用户声明)。

loss = (1/K)Σ_k Σ_t w(t)·‖BP_{0.5-9}(e_sim^k)‖² + β‖δa‖² + γ‖Δδa‖²
  e_sim^k = e_log + Ĝ_δ^k[δa]  (增量式; Ĝ_δ = L2 扰动通道: 刚体线性化+调度模态柔性)
  w(t) = clip(gate/0.15, 0.2, 1) 运动段加权(2026-07-15 用户批准, 预注册);
  接触窗零权; K=5 集成均值训练/最差成员进评估。
特征 51 维(b_runtime, 含 q̇)。响应算子在 e-index 空间: δa@τ 影响 e[τ+ℓ−3], ℓ=0..15。
模式: --mode smoke(正确性: δa≡0 ⇒ 消减≡0) | sweep(β/γ 膝点, TCN) | race(7席×3种子)
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import firwin

from lerobot.common.policies.act.b_runtime import build_offline
from bakeoff_g import ep_J
from fit_g_production import modal_kernels
from fit_g_level2 import excitation, raw_feats

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(__file__).resolve().parents[1] / "outputs/p2_pi"
LROOT = Path(__file__).resolve().parents[1] / "outputs/p2_labels"
GROOT = Path(__file__).resolve().parents[1] / "outputs/p2_bakeoff"
TAGS = ["pp2-deploy", "ins-deploy"]
TRAIN_EPS, VAL_EPS = list(range(8)), [8, 9]
DT, EPS_B, LSHIFT = 0.05, 0.010, 3
LR_RESP, SEG, BATCH = 16, 96, 96
LNEG = 4                    # 负滞后格位: 训练路径 excitation() 用 filtfilt(零相位) → 核有~5%非因果能量
LTOT = LNEG + LR_RESP       # 核的总格数; 格 l ↔ 墙钟滞后 (l − LNEG)
FEAT = 51


# ---------------- Ĝ 集成 → 响应核 ----------------
class GEnsemble:
    def __init__(self):
        z = np.load(GROOT / "g_ensemble.npz")
        self.K = len(z["c_enc"])
        c = z["c_enc"]                                    # (K,6,10)
        M = z["log_om"].shape[1]
        # 响应核一律用**数值探针**从训练路径的同一份算子量出(它们对指令严格线性),
        # 不再解析推导 — 2026-07-21 修: 二次差分版比样条精确二阶导晚 1 帧, 且丢掉
        # filtfilt 的非因果部分(~5%能量), 曾令 psi 系统性偏离真值。
        N, T0 = 96, 48
        vk = np.zeros((self.K, 6, LTOT), np.float32)
        ek = np.zeros((self.K, 6, LTOT), np.float32)
        for k in range(self.K):
            for j in range(6):
                imp = np.zeros((N, 6), np.float32)
                imp[T0, j] = 1.0
                ad = np.vstack([np.zeros((1, 6), np.float32), np.diff(imp, axis=0)]) / DT
                qd = np.zeros((N, 6), np.float32)         # 刚体路: 同 rigid_parts
                for t in range(c.shape[2]):
                    qd[t:] += c[k][:, t][None, :] * ad[:N - t]
                vk[k, j] = qd[T0 - LNEG:T0 + LR_RESP, j]
                ek[k, j] = excitation(imp, c[k])[T0 - LNEG:T0 + LR_RESP, j]   # 柔性路激励
        # 模态路**不**预乘成单核: 训练时 alpha 夹在激励与谐振器之间(up=W@u*alpha 再卷积),
        # 预乘会迫使 alpha 退到输入端 → 数学不等价(实测误差 80-108%)。故分两段保留。
        km = np.stack([modal_kernels(torch.from_numpy(z["log_om"][k]).exp(),
                                     torch.sigmoid(torch.from_numpy(z["raw_ze"][k])) * 0.5).numpy()
                       for k in range(self.K)])            # (K,M,KKER)
        self.exck = torch.from_numpy(ek).to(DEV)          # (K,6,LTOT) 激励核(直通+模态共用)
        self.vker = torch.from_numpy(vk).to(DEV)          # (K,6,LTOT)
        self.km = torch.from_numpy(km).to(DEV)            # (K,M,KKER)
        self.Wt = torch.from_numpy(z["W"]).to(DEV)        # (K,M,6)
        self.KKER = km.shape[2]
        self.B = z["B"]; self.D = z["D"]; self.theta = z["theta"]
        self.mu = z["mu"]; self.sd = z["sd"]
        self.M = M

    def ep_context(self, a_arm, grip, R):
        """每集: epJ(n,3,6, 真机系), alpha(K,n,M), RB(K,M,3), RD(K,3,6)。"""
        n = len(a_arm)
        epj = ep_J({"a": a_arm, "R": R})                  # 真机系 R@Jr
        a7 = np.concatenate([a_arm, grip[:, None]], 1)
        alpha = np.zeros((self.K, n, self.M), np.float32)
        for k in range(self.K):
            phi = (raw_feats(a7) - self.mu[k]) / self.sd[k]
            alpha[k] = np.log1p(np.exp(phi @ self.theta[k].T))     # softplus
        RB = np.stack([self.B[k] @ R.T for k in range(self.K)]).astype(np.float32)
        RD = np.stack([R @ self.D[k] for k in range(self.K)]).astype(np.float32)
        return (torch.from_numpy(epj.astype(np.float32)).to(DEV),
                torch.from_numpy(alpha).to(DEV), torch.from_numpy(RB).to(DEV),
                torch.from_numpy(RD).to(DEV))


def _fir(x, kern, nfut):
    """x(B,C,S) ⊛ kern(C,L) — kern 的第 l 格代表滞后 l−nfut。返回 (B,C,S)。"""
    L = kern.shape[-1]
    xp = F.pad(x, (L - 1 - nfut, nfut))
    return F.conv1d(xp, kern.flip(-1).unsqueeze(1), groups=x.shape[1])


def apply_G(delta, epJ, alpha, RB, RD, G):
    """δa(B,S,6) → Δe(B,K,S,3)。与 FlexL2+rigid_parts 逐项等价的增量算子。

    三条通路(全部对 δa 严格线性):
      刚体  Δω = J(t)·(vker ⊛ δa)
      模态  Δω = RB·(km ⊛ [alpha·(W·(exck ⊛ δa))])   ← alpha 必须夹在两次卷积之间
      直通  Δω = RD·(exck ⊛ δa)
    最后整体左移 LSHIFT: 墙钟 t 的响应落在 e-索引 t−LSHIFT。"""
    Bz, S, _ = delta.shape
    K, M = G.K, G.M
    x = delta.transpose(1, 2).repeat(1, K, 1)                       # (B,K*6,S)
    qd = _fir(x, G.vker.reshape(K * 6, -1), LNEG).view(Bz, K, 6, S)  # 刚体: 关节速度
    ud = _fir(x, G.exck.reshape(K * 6, -1), LNEG).view(Bz, K, 6, S)  # 激励
    rig = torch.einsum("btcj,bkjt->bktc", epJ, qd)
    dir_ = torch.einsum("bkcj,bkjt->bktc", RD, ud)
    mi = torch.einsum("kmj,bkjt->bkmt", G.Wt, ud) * alpha.permute(0, 1, 3, 2)
    s = _fir(mi.reshape(Bz, K * M, S), G.km.reshape(K * M, -1), 0).view(Bz, K, M, S)
    flx = torch.einsum("bkmt,bkmc->bktc", s, RB)
    tot = rig + flx + dir_                                          # 墙钟索引
    de = torch.zeros_like(tot)
    de[:, :, :S - LSHIFT] = tot[:, :, LSHIFT:]                      # → e-索引
    return de


# ---------------- 数据 ----------------
def load_eps(eps_list, G):
    out = []
    for tag in TAGS:
        for ei in eps_list:
            z = np.load(LROOT / tag / f"ep{ei:03d}.npz")
            X, _, act_mask = build_offline(z, L=LSHIFT)
            R = np.array(json.loads((LROOT / tag / "manifest.json").read_text())["R"],
                         np.float32)
            epj, alpha, RB, RD = G.ep_context(z["a_arm"], z["grip"], R)
            gate = X[:, -1]
            w = np.clip(gate / 0.15, 0.2, 1.0).astype(np.float32)
            w[z["mask_contact"]] = 0.0
            out.append({
                "X": torch.from_numpy(X).to(DEV),
                "e": torch.from_numpy(z["e"][:, 3:6].astype(np.float32)).to(DEV),
                "w": torch.from_numpy(w).to(DEV),
                "motion": torch.from_numpy(~z["mask_still"]).to(DEV),
                "epJ": epj.unsqueeze(0), "alpha": alpha.unsqueeze(0), "RB": RB.unsqueeze(0),
                "RD": RD.unsqueeze(0), "tag": tag, "ep": ei})
    return out


def make_bp(lo=0.5, hi=9.0, taps=41):
    k = firwin(taps, [lo, hi], fs=20.0, pass_zero=False).astype(np.float32)
    w = torch.from_numpy(k).to(DEV).flip(0).reshape(1, 1, -1).repeat(3, 1, 1)
    def bp(x):                                             # (...,S,3)
        sh = x.shape
        y = x.reshape(-1, sh[-2], 3).transpose(1, 2)
        y = F.conv1d(F.pad(y, (20, 20)), w, groups=3)
        return y.transpose(1, 2).reshape(sh)
    return bp


BP = None


def batch_iter(eps, steps, bs=BATCH, seg=SEG):
    lens = np.array([len(e["e"]) - seg for e in eps], float)
    p = lens / lens.sum()
    for _ in range(steps):
        bi = np.random.choice(len(eps), bs, p=p)
        off = [np.random.randint(0, int(lens[i])) for i in bi]
        yield bi, off


def gather(eps, bi, off, seg=SEG):
    X = torch.stack([eps[i]["X"][o:o + seg] for i, o in zip(bi, off)])
    e = torch.stack([eps[i]["e"][o:o + seg] for i, o in zip(bi, off)])
    w = torch.stack([eps[i]["w"][o:o + seg] for i, o in zip(bi, off)])
    epJ = torch.cat([eps[i]["epJ"][:, o:o + seg] for i, o in zip(bi, off)])
    al = torch.cat([eps[i]["alpha"][:, :, o:o + seg] for i, o in zip(bi, off)])
    RB = torch.cat([eps[i]["RB"] for i in bi])
    RD = torch.cat([eps[i]["RD"] for i in bi])
    return X, e, w, epJ, al, RB, RD


def losses(delta, e, w, epJ, al, RB, RD, G, beta, gamma):
    de = apply_G(delta, epJ, al, RB, RD, G)
    esim = e.unsqueeze(1) + de
    b_sim = BP(esim)
    wk = w.unsqueeze(1).unsqueeze(-1)
    K = esim.shape[1]
    l1 = (b_sim ** 2 * wk).sum() / wk.sum() / 3 / K        # K成员均值(勿漏除, 否则虚增5×)
    base = (BP(e.unsqueeze(1)) ** 2 * wk).sum() / wk.sum() / 3
    l2 = beta * (delta ** 2).mean()
    l3 = gamma * ((delta[:, 1:] - delta[:, :-1]) ** 2).mean()
    return l1, l2, l3, float(1 - (l1 / base).item())


def train_one(cls, lr, seed, steps, eps_tr, G, beta, gamma, log_every=500):
    torch.manual_seed(seed); np.random.seed(seed)
    model = cls(d=FEAT).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for i, (bi, off) in enumerate(batch_iter(eps_tr, steps)):
        X, e, w, epJ, al, RB, RD = gather(eps_tr, bi, off)
        delta = model(X)
        l1, l2, l3, red = losses(delta, e, w, epJ, al, RB, RD, G, beta, gamma)
        loss = l1 + l2 + l3
        opt.zero_grad(); loss.backward(); opt.step()
        if (i + 1) % log_every == 0:
            print(f"    step {i+1}: 消减={red*100:.1f}% ‖δa‖={delta.detach().abs().mean().item()*1000:.2f}mrad",
                  flush=True)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, eps_va, G):
    reds, worst, motion_reds = [], [], []
    d_rms, d_sat, d_rough = [], [], []
    for ep in eps_va:
        n = len(ep["e"])
        delta = model(ep["X"].unsqueeze(0))
        de = apply_G(delta, ep["epJ"], ep["alpha"], ep["RB"], ep["RD"], G)
        esim = ep["e"].unsqueeze(0).unsqueeze(1) + de      # (1,K,n,3)
        b_sim, b_log = BP(esim), BP(ep["e"].view(1, 1, n, 3))
        m = ep["w"] > 0
        num_k = (b_sim[0][:, m] ** 2).sum((1, 2))
        den = (b_log[0, 0][m] ** 2).sum()
        rk = 1 - num_k / den
        reds.append(rk.mean().item()); worst.append(rk.min().item())
        mm = m & ep["motion"]
        nk = (b_sim[0][:, mm] ** 2).sum((1, 2))
        dd = (b_log[0, 0][mm] ** 2).sum()
        motion_reds.append((1 - nk / dd).mean().item())
        d = delta[0]
        d_rms.append((d ** 2).mean().sqrt().item() * 1000)
        d_sat.append((d.abs() > 0.95 * EPS_B).float().mean().item() * 100)
        d_rough.append((d[1:] - d[:-1]).abs().mean().item() * 1000)
    return {"red": float(np.mean(reds)), "worst": float(np.mean(worst)),
            "red_motion": float(np.mean(motion_reds)),
            "da_rms_mrad": float(np.mean(d_rms)), "sat_pct": float(np.mean(d_sat)),
            "rough_mrad": float(np.mean(d_rough))}


def get_zoo():
    from bakeoff_backbones import TCN, S4DNet
    from bakeoff_final3 import LSTMNet
    from bakeoff_more import GRUNet, MLPWin, TCNDeep, XFormer
    return {"TCN": TCN, "TCN-deep": TCNDeep, "GRU": GRUNet, "LSTM": LSTMNet,
            "S4D": S4DNet, "Transformer": XFormer, "MLP-window": MLPWin}


def main():
    global BP
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "sweep", "race"], required=True)
    ap.add_argument("--beta", type=float, default=1e3)
    ap.add_argument("--gamma", type=float, default=0.0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    BP = make_bp()
    print(f"device={DEV}")
    G = GEnsemble()
    eps_tr, eps_va = load_eps(TRAIN_EPS, G), load_eps(VAL_EPS, G)
    print(f"train {sum(len(e['e']) for e in eps_tr)} / val {sum(len(e['e']) for e in eps_va)} 帧")

    if args.mode == "smoke":
        from bakeoff_backbones import TCN
        # 正确性1: δa≡0 ⇒ 消减≡0
        class Zero(torch.nn.Module):
            def forward(self, x):
                return x.new_zeros(x.shape[0], x.shape[1], 6)
        z = evaluate(Zero().to(DEV), eps_va, G)
        print(f"[δa≡0] 消减={z['red']*100:.3f}% (应≈0)")
        t0 = time.time()
        m = train_one(TCN, 3e-3, 0, 200, eps_tr, G, args.beta, args.gamma, log_every=100)
        r = evaluate(m, eps_va, G)
        print(f"[TCN 200步] 消减={r['red']*100:.1f}% (运动段 {r['red_motion']*100:.1f}%, "
              f"最差成员 {r['worst']*100:.1f}%)  ‖δa‖={r['da_rms_mrad']:.2f}mrad "
              f"sat={r['sat_pct']:.0f}% 粗糙={r['rough_mrad']:.2f}  ({time.time()-t0:.0f}s)")
        return

    if args.mode == "sweep":
        from bakeoff_backbones import TCN
        res = {}
        for beta in (1e2, 1e3, 1e4):
            for gamma in (0.0, beta / 10):
                m = train_one(TCN, 3e-3, 0, 2000, eps_tr, G, beta, gamma, log_every=1000)
                r = evaluate(m, eps_va, G)
                key = f"b{beta:g}_g{gamma:g}"
                res[key] = r
                print(f"[{key}] 消减={r['red']*100:.1f}% 运动段={r['red_motion']*100:.1f}% "
                      f"‖δa‖={r['da_rms_mrad']:.2f} sat={r['sat_pct']:.0f}% 粗糙={r['rough_mrad']:.2f}",
                      flush=True)
        ok = {k: v for k, v in res.items() if v["da_rms_mrad"] <= 4 and v["sat_pct"] < 20}
        pick = max(ok or res, key=lambda k: (ok or res)[k]["red"])
        res["_pick"] = pick
        (OUT / "sweep.json").write_text(json.dumps(res, indent=1))
        print(f"膝点选择: {pick}")
        return

    if args.mode == "race":
        zoo = get_zoo()
        res = {}
        for name, cls in zoo.items():
            best = (None, -1e9)
            for lr in (1e-3, 3e-3):
                m = train_one(cls, lr, 0, 800, eps_tr, G, args.beta, args.gamma, log_every=10**9)
                r = evaluate(m, eps_va, G)
                if r["red"] > best[1]:
                    best = (lr, r["red"])
            recs = []
            t0 = time.time()
            for seed in (0, 1, 2):
                m = train_one(cls, best[0], seed, 6000, eps_tr, G, args.beta, args.gamma,
                              log_every=10**9)
                torch.save(m.state_dict(), OUT / f"pi_{name}_s{seed}.pt")
                recs.append(evaluate(m, eps_va, G))
            res[name] = {k: [float(np.mean([r[k] for r in recs])),
                             float(np.std([r[k] for r in recs]))] for k in recs[0]}
            res[name]["best_lr"] = best[0]
            r = res[name]
            print(f"[{name}] 消减={r['red'][0]*100:.1f}±{r['red'][1]*100:.1f}% "
                  f"运动段={r['red_motion'][0]*100:.1f}% 最差={r['worst'][0]*100:.1f}% "
                  f"‖δa‖={r['da_rms_mrad'][0]:.2f} sat={r['sat_pct'][0]:.0f}% "
                  f"(lr={best[0]:g}, {time.time()-t0:.0f}s)", flush=True)
            (OUT / "race.json").write_text(json.dumps(res, indent=1))
        print("done")


if __name__ == "__main__":
    main()
