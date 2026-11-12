#!/usr/bin/env python3
"""Ĝ 表达 bake-off v2 — 评分修订(2026-07-15, 用户对尺子的三条审稿全部采纳):
  ① 带号余弦: 编码器独立裁判(指令→关节速度FIR)导出符号轮廓 s_{j,k},
     真值方向 = s·R·Jr[:,j]; 另报反相占比(红线: >10% 出局); 轴共线|cos|降为辅报
  ② 双R²: raw 作参考, **gap-R²(靶=real−kin_ideal, 即墙)为入场券**(须≥ridge)
  ③ 双极性 ±{2.5,5,10}mrad: 六档比例一致性 + 奇对称偏差 ‖r(+5)+r(−5)‖/‖r(+5)‖
  ④ 全部模型权重落盘(以后改评分只重打探针, 不再重训)
名单(批1): Gray-box / Hybrid / TCN / GRU / S4D / MLP-window / ridge
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from g_verdict import DT, KIN, LAGS, TRAIN_EPS, VAL_EPS, GrayBox, jr_at, load

torch.set_num_threads(4)
OUT = Path(__file__).resolve().parents[1] / "outputs/p2_bakeoff"
MDL = OUT / "g_models"
LRS = [1e-3, 2e-3, 5e-3]
JOINTS = (0, 1, 2, 4)
AMPS = [0.0025, 0.005, 0.010, -0.0025, -0.005, -0.010]


# ---------------- NN 动物园(批1) ----------------
class TCNG(nn.Module):
    def __init__(self, h=32):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(6, h, 3, dilation=1),
                                    nn.Conv1d(h, h, 3, dilation=2),
                                    nn.Conv1d(h, h, 3, dilation=4)])
        self.pads = [2, 4, 8]
        self.out = nn.Linear(h, 3)

    def forward(self, u):
        x = u.transpose(1, 2)
        for c, p in zip(self.convs, self.pads):
            x = F.gelu(c(F.pad(x, (p, 0))))
        return self.out(x.transpose(1, 2))


class GRUG(nn.Module):
    def __init__(self, h=48):
        super().__init__()
        self.rnn = nn.GRU(6, h, num_layers=2, batch_first=True)
        self.out = nn.Linear(h, 3)

    def forward(self, u):
        y, _ = self.rnn(u)
        return self.out(y)


class S4DG(nn.Module):
    def __init__(self, h=32):
        super().__init__()
        from bakeoff_backbones import S4DLayer
        self.enc = nn.Linear(6, h)
        self.s1 = S4DLayer(h)
        self.n1 = nn.LayerNorm(h)
        self.out = nn.Linear(h, 3)

    def forward(self, u):
        x = F.gelu(self.enc(u))
        x = self.n1(x + F.gelu(self.s1(x)))
        return self.out(x)


class MLPG(nn.Module):
    def __init__(self, w=12):
        super().__init__()
        self.w = w
        self.mlp = nn.Sequential(nn.Linear(6 * w, 128), nn.GELU(), nn.Linear(128, 3))

    def forward(self, u):
        x = F.pad(u, (0, 0, self.w - 1, 0))
        xu = x.unfold(1, self.w, 1).reshape(u.shape[0], u.shape[1], -1)
        return self.mlp(xu)


class HybridG(nn.Module):
    """解析 Jr 管方向, depthwise 因果conv管每关节等效速度。J 由包装喂入(基准冻结)。"""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(6, 24, 3, dilation=1, groups=6)
        self.c2 = nn.Conv1d(24, 24, 3, dilation=2, groups=6)
        self.c3 = nn.Conv1d(24, 6, 1, groups=6)

    def forward(self, u, J):
        ad = torch.diff(u, dim=1, prepend=u[:, :1]) / DT
        x = ad.transpose(1, 2)
        x = F.gelu(self.c1(F.pad(x, (2, 0))))
        x = F.gelu(self.c2(F.pad(x, (4, 0))))
        v = self.c3(x).transpose(1, 2)
        return torch.einsum("btcj,btj->btc", J, v)


# ---------------- 工具 ----------------
def ep_J(ep):
    a = ep["a"]
    return np.stack([ep["R"] @ jr_at(a[max(t - 2, 0)]) for t in range(len(a))]).astype(np.float32)


def fit_c_enc(train_eps, K=10):
    """独立裁判: 指令→编码器关节速度 FIR; 返回(每关节抽头, 单帧脉冲符号轮廓)。"""
    C = np.zeros((6, K), np.float32)
    for j in range(6):
        Xs, ys = [], []
        for e in train_eps:
            a, q = e["a"][:, j], e["q"][:, j]
            ad = (np.concatenate([[0], np.diff(a)]) / DT).astype(np.float32)
            qd = (np.concatenate([[0], np.diff(q)]) / DT).astype(np.float32)
            T = len(a)
            X = np.zeros((T, K), np.float32)
            for k in range(K):
                X[k:, k] = ad[:T - k]
            Xs.append(X); ys.append(qd)
        C[j] = np.linalg.lstsq(np.vstack(Xs), np.concatenate(ys), rcond=None)[0]
    P = np.zeros((6, LAGS), np.float32)
    for j in range(6):
        for k in range(LAGS):
            ck = C[j, k] if k < K else 0.0
            ck1 = C[j, k - 1] if 0 <= k - 1 < K else 0.0
            P[j, k] = (ck - ck1) / DT
    S = (np.sign(P) * (np.abs(P) > 0.05 * np.abs(P).max(1, keepdims=True))).astype(np.float32)
    return C, S


def probe_model(f, ep, sites, S):
    a, R = ep["a"], ep["R"]
    base = f(a)
    recs = []
    for t0 in sites:
        for j in JOINTS:
            U = np.stack([R @ jr_at(a[min(t0 + k, len(a) - 1)])[:, j] for k in range(LAGS)])
            U /= np.linalg.norm(U, axis=1, keepdims=True) + 1e-12
            resp = {}
            for amp in AMPS:
                ap = a.copy(); ap[t0, j] += amp
                resp[amp] = (f(ap) - base)[t0:t0 + LAGS]
            r5 = resp[0.005]
            mag_k = np.linalg.norm(r5, axis=1)
            cosk = (r5 * U).sum(1) / (mag_k + 1e-12)          # 带号(相对+轴)
            sj = S[j]
            wk = mag_k * (sj != 0)
            signed = float((cosk * sj * wk).sum() / (wk.sum() + 1e-12))
            axis = float((np.abs(cosk) * mag_k).sum() / (mag_k.sum() + 1e-12))
            norms = {amp: float(np.linalg.norm(resp[amp])) for amp in AMPS}
            ratios = [norms[amp] / abs(amp) for amp in AMPS]
            lin = float((max(ratios) - min(ratios)) / (np.mean(ratios) + 1e-12))
            asym = float(np.linalg.norm(resp[0.005] + resp[-0.005]) / (norms[0.005] + 1e-12))
            recs.append({"signed": signed, "axis": axis, "anti": signed < 0,
                         "mag": float(mag_k.sum()), "lin": lin, "asym": asym})
    return recs


def make_f(model, ep, J=None):
    if isinstance(model, GrayBox):
        return lambda a: model.predict({"a": a, "R": ep["R"]})
    if isinstance(model, HybridG):
        Jt = torch.from_numpy(J[None])
        def f(a):
            with torch.no_grad():
                return model(torch.from_numpy(a[None]), Jt)[0].numpy()
        return f
    def f(a):
        with torch.no_grad():
            return model(torch.from_numpy(a[None]))[0].numpy()
    return f


def dual_r2(pred, ep):
    sse = float(((pred - ep["g"]) ** 2).sum())
    raw = 1 - sse / float(((ep["g"] - ep["g"].mean(0)) ** 2).sum())
    gap = 1 - sse / float(((ep["gap"] - ep["gap"].mean(0)) ** 2).sum())
    return raw, gap


def train_nn(cls, lr, seed, steps, train_eps, J_cache):
    torch.manual_seed(seed); np.random.seed(seed)
    model = cls()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lens = np.array([len(e["a"]) - 64 for e in train_eps], float)
    p = lens / lens.sum()
    hybrid = isinstance(model, HybridG)
    for _ in range(steps):
        bi = np.random.choice(len(train_eps), 64, p=p)
        xb, yb, jb = [], [], []
        for b in bi:
            o = np.random.randint(0, int(lens[b]))
            xb.append(train_eps[b]["a"][o:o + 64]); yb.append(train_eps[b]["g"][o:o + 64])
            if hybrid:
                jb.append(J_cache[id(train_eps[b])][o:o + 64])
        x, y = torch.from_numpy(np.stack(xb)), torch.from_numpy(np.stack(yb))
        pred = model(x, torch.from_numpy(np.stack(jb))) if hybrid else model(x)
        loss = F.mse_loss(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def agg(recs, key):
    return float(np.median([r[key] for r in recs]))


def evaluate_generic(name, model_iter, val, J_cache, site_map, S, res):
    rows = {"r2raw": [], "r2gap": [], "signed": [], "axis": [], "anti": [],
            "mag": [], "lin": [], "asym": []}
    for m in model_iter:
        r2s, recs = [], []
        for ep in val:
            f = make_f(m, ep, J_cache[id(ep)])
            r2s.append(dual_r2(f(ep["a"]), ep))
            recs += probe_model(f, ep, site_map[id(ep)], S)
        rows["r2raw"].append(np.mean([r[0] for r in r2s]))
        rows["r2gap"].append(np.mean([r[1] for r in r2s]))
        rows["signed"].append(agg(recs, "signed"))
        rows["axis"].append(agg(recs, "axis"))
        rows["anti"].append(float(np.mean([r["anti"] for r in recs]) * 100))
        rows["mag"].append(agg(recs, "mag"))
        rows["lin"].append(agg(recs, "lin") * 100)
        rows["asym"].append(agg(recs, "asym") * 100)
    res[name] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in rows.items()}
    r = res[name]
    print(f"[{name}] R²raw={r['r2raw'][0]:.3f} R²gap={r['r2gap'][0]:.3f}  "
          f"带号cos={r['signed'][0]:.3f} 反相={r['anti'][0]:.0f}%  mag={r['mag'][0]:.4f}  "
          f"线性度={r['lin'][0]:.1f}% 奇对称偏差={r['asym'][0]:.1f}%", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True); MDL.mkdir(parents=True, exist_ok=True)
    train, val = load(TRAIN_EPS, with_gap=True), load(VAL_EPS, with_gap=True)
    print(f"train {sum(len(e['a']) for e in train)} / val {sum(len(e['a']) for e in val)} 帧", flush=True)
    _, S = fit_c_enc(train)
    np.save(MDL / "judge_sign_profile.npy", S)
    J_cache = {id(e): ep_J(e) for e in train + val}
    rng = np.random.default_rng(0)
    site_map = {id(e): rng.integers(KIN, len(e["a"]) - LAGS - 1, 10) for e in val}
    res = {}

    gb = GrayBox(); gb.fit(train)
    np.save(MDL / "Gray-box_c.npy", gb.c)
    evaluate_generic("Gray-box", [gb], val, J_cache, site_map, S, res)

    # ridge(展平窗线性, 闭式)
    class RidgeF:
        def __init__(self, eps_):
            X, Y = [], []
            for e in eps_:
                a = e["a"]
                pad = np.vstack([np.zeros((KIN - 1, 6), np.float32), a])
                X.append(np.stack([pad[t:t + KIN].ravel() for t in range(len(a))]))
                Y.append(e["g"])
            X, Y = np.concatenate(X), np.concatenate(Y)
            lam = 1e-3 * len(X)
            self.W = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)
        def __call__(self, a):
            pad = np.vstack([np.zeros((KIN - 1, 6), np.float32), a])
            return np.stack([pad[t:t + KIN].ravel() for t in range(len(a))]) @ self.W
    rgf = RidgeF(train)
    r2s, recs = [], []
    for ep in val:
        r2s.append(dual_r2(rgf(ep["a"]), ep))
        recs += probe_model(rgf, ep, site_map[id(ep)], S)
    res["ridge"] = {"r2raw": [float(np.mean([r[0] for r in r2s])), 0.0],
                    "r2gap": [float(np.mean([r[1] for r in r2s])), 0.0],
                    "signed": [agg(recs, "signed"), 0.0], "axis": [agg(recs, "axis"), 0.0],
                    "anti": [float(np.mean([r['anti'] for r in recs]) * 100), 0.0],
                    "mag": [agg(recs, "mag"), 0.0], "lin": [agg(recs, "lin") * 100, 0.0],
                    "asym": [agg(recs, "asym") * 100, 0.0]}
    r = res["ridge"]
    print(f"[ridge] R²raw={r['r2raw'][0]:.3f} R²gap={r['r2gap'][0]:.3f}  "
          f"带号cos={r['signed'][0]:.3f} 反相={r['anti'][0]:.0f}%  mag={r['mag'][0]:.4f}  "
          f"线性度={r['lin'][0]:.1f}% 奇对称偏差={r['asym'][0]:.1f}%", flush=True)

    for name, cls in [("Hybrid", HybridG), ("TCN", TCNG), ("GRU", GRUG),
                      ("S4D", S4DG), ("MLP-window", MLPG)]:
        best = (None, -1e9)
        for lr in LRS:
            m = train_nn(cls, lr, 0, 1500, train, J_cache)
            v = np.mean([dual_r2(make_f(m, ep, J_cache[id(ep)])(ep["a"]), ep)[1] for ep in val])
            if v > best[1]:
                best = (lr, v)
        t0 = time.time()
        models = []
        for s in (0, 1, 2):
            m = train_nn(cls, best[0], s, 4000, train, J_cache)
            torch.save(m.state_dict(), MDL / f"{name}_s{s}.pt")
            models.append(m)
        evaluate_generic(name, models, val, J_cache, site_map, S, res)
        res[name]["best_lr"] = best[0]
        print(f"    (best_lr={best[0]:g}, {time.time()-t0:.0f}s)", flush=True)
        (OUT / "g_bakeoff.json").write_text(json.dumps(res, indent=1))

    (OUT / "g_bakeoff.json").write_text(json.dumps(res, indent=1))
    print(f"\n→ {OUT/'g_bakeoff.json'}")


if __name__ == "__main__":
    main()
