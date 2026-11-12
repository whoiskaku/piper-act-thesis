#!/usr/bin/env python3
"""P2.3 — 骨干 bake-off: TCN vs S4D-lite vs Gray-box (+predict-0/线性ridge 基线)。

同数据(outputs/p2_labels, ep0-7训/ep8-9验, 按集切防泄漏), 同流式输入(45维/步),
同输出头 tanh×ε, 同损失 Huber, 同预算(--steps)。
流式输入 u(t) = [e(t)6, ideal(t)6, ideal(t+4)6, ideal(t+8)6, a(t)6, a(t+4)6, a(t+8)6, phase3]
评分: ①val RMS(全帧/活动帧/j1) ②模型内预测消减(vs标签天花板) ③参数量
      ④CPU单步延迟 ⑤可解释产物(S4D极点/Gray-box增益, 打印)。
用法: python sim/bakeoff_backbones.py [--steps 3000]
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)
ROOT = Path(__file__).resolve().parents[1] / "outputs/p2_labels"
OUT = Path(__file__).resolve().parents[1] / "outputs/p2_bakeoff"
EPS_B = 0.010
DT = 0.05
TAGS = ["pp2-deploy", "ins-deploy"]
TRAIN_EPS, VAL_EPS = list(range(8)), [8, 9]
SEG = 64


def load_split(eps_list):
    xs, ys, acts = [], [], []
    for tag in TAGS:
        for ep in eps_list:
            z = np.load(ROOT / tag / f"ep{ep:03d}.npz")
            e, ideal, a, ph = z["e"], z["ideal"], z["a_arm"], z["phase"]
            lab, still = z["label"], z["mask_still"]
            n = len(e) - 8
            f = np.concatenate([
                e[:n], ideal[:n], ideal[4:n+4], ideal[8:n+8],
                a[:n], a[4:n+4], a[8:n+8], ph[:n]], axis=1)   # (n,45)
            xs.append(f.astype(np.float32))
            ys.append(lab[:n].astype(np.float32))
            acts.append(~still[:n])
    return xs, ys, acts


class Head(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 6)

    def forward(self, h):
        return torch.tanh(self.lin(h)) * EPS_B


class TCN(nn.Module):
    def __init__(self, d=45, h=32):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(d, h, 3, dilation=1), nn.Conv1d(h, h, 3, dilation=2),
            nn.Conv1d(h, h, 3, dilation=4)])
        self.pads = [2, 4, 8]
        self.head = Head(h)

    def forward(self, u):                       # u (B,T,45)
        x = u.transpose(1, 2)
        for c, p in zip(self.convs, self.pads):
            x = torch.nn.functional.gelu(c(torch.nn.functional.pad(x, (p, 0))))
        return self.head(x.transpose(1, 2))


class S4DLayer(nn.Module):
    """对角复数SSM: x_{t+1}=exp((−σ+iω)dt)x_t+B·u_t, y=Re(C x)+D u; ω初始化覆盖0.5-8Hz。"""

    def __init__(self, H=32, N=8):
        super().__init__()
        f = np.logspace(np.log10(0.5), np.log10(8.0), N)      # Hz
        self.log_sigma = nn.Parameter(torch.full((H, N), np.log(3.0)))   # 衰减 rad/s
        self.omega = nn.Parameter(torch.tensor(2*np.pi*f, dtype=torch.float32)
                                  .repeat(H, 1) * (1 + 0.01*torch.randn(H, N)))
        self.C = nn.Parameter(torch.randn(H, N, 2) / np.sqrt(N))
        self.D = nn.Parameter(torch.ones(H))

    def forward(self, u):                       # (B,T,H)
        B_, T, H = u.shape
        N = self.omega.shape[1]
        sig = torch.exp(self.log_sigma)
        dec = torch.exp(-sig * DT)
        cos, sin = torch.cos(self.omega*DT)*dec, torch.sin(self.omega*DT)*dec
        xr = u.new_zeros(B_, H, N); xi = u.new_zeros(B_, H, N)
        Cr, Ci = self.C[..., 0], self.C[..., 1]
        ys = []
        for t in range(T):
            ut = u[:, t, :].unsqueeze(-1)                     # (B,H,1)
            xr, xi = xr*cos - xi*sin + ut, xr*sin + xi*cos
            ys.append((xr*Cr + xi*Ci).sum(-1) + self.D*u[:, t, :])
        return torch.stack(ys, 1)


class S4DNet(nn.Module):
    def __init__(self, d=45, h=32):
        super().__init__()
        self.enc = nn.Linear(d, h)
        self.s1, self.s2 = S4DLayer(h), S4DLayer(h)
        self.n1, self.n2 = nn.LayerNorm(h), nn.LayerNorm(h)
        self.head = Head(h)

    def forward(self, u):
        x = torch.nn.functional.gelu(self.enc(u))
        x = self.n1(x + torch.nn.functional.gelu(self.s1(x)))
        x = self.n2(x + torch.nn.functional.gelu(self.s2(x)))
        return self.head(x)


class GrayBox(nn.Module):
    """固定三带通(1-2/2-4/4-8Hz biquad)滤 e_gyro → 9流; 上下文MLP出增益; W映射到6关节。"""

    def __init__(self):
        super().__init__()
        from scipy.signal import butter
        sos = [butter(1, [lo/10, hi/10], "bandpass", output="sos")[0]
               for lo, hi in [(1, 2), (2, 4), (4, 8)]]
        self.register_buffer("coef", torch.tensor(np.array(sos), dtype=torch.float32))
        self.gain_mlp = nn.Sequential(nn.Linear(18, 32), nn.GELU(), nn.Linear(32, 9))
        self.W = nn.Linear(9, 6)

    def forward(self, u):                       # (B,T,45)
        eg = u[..., 3:6]                        # e_gyro
        ctx = torch.cat([u[..., 6:12], u[..., 24:30], u[..., 42:45],
                         u[..., 18:21]], -1)    # ideal(t), a(t), phase, ideal_far_gyro
        B_, T, _ = u.shape
        outs = []
        z1 = eg.new_zeros(3, B_, 3, 2)          # 每带 biquad 状态(直接II型): (band,B,ch,2)
        for t in range(T):
            x = eg[:, t, :]
            bands = []
            for b in range(3):
                b0, b1, b2, _, a1, a2 = self.coef[b]
                w = x - a1*z1[b, ..., 0] - a2*z1[b, ..., 1]
                y = b0*w + b1*z1[b, ..., 0] + b2*z1[b, ..., 1]
                z1 = z1.clone()
                z1[b, ..., 1] = z1[b, ..., 0]
                z1[b, ..., 0] = w
                bands.append(y)
            z = torch.cat(bands, -1)            # (B,9)
            g = 2*torch.sigmoid(self.gain_mlp(ctx[:, t, :]))
            outs.append(torch.tanh(self.W(z*g)) * EPS_B)
        return torch.stack(outs, 1)


def batches(xs, ys, acts, bs=64, steps=1000):
    lens = [len(x)-SEG for x in xs]
    p = np.array(lens, float); p /= p.sum()
    for _ in range(steps):
        bi = np.random.choice(len(xs), bs, p=p)
        off = [np.random.randint(0, lens[i]) for i in bi]
        yield (torch.stack([torch.from_numpy(xs[i][o:o+SEG]) for i, o in zip(bi, off)]),
               torch.stack([torch.from_numpy(ys[i][o:o+SEG]) for i, o in zip(bi, off)]))


def evaluate(model, xs, ys, acts):
    model.eval()
    se = sa = n = na = 0.0
    sj1 = 0.0
    with torch.no_grad():
        for x, y, a in zip(xs, ys, acts):
            p = model(torch.from_numpy(x).unsqueeze(0))[0].numpy()
            d2 = (p - y) ** 2
            se += d2.sum(); n += d2.size
            sa += d2[a].sum(); na += d2[a].size
            sj1 += d2[:, 0].sum()
    return (np.sqrt(se/n)*1000, np.sqrt(sa/max(na, 1))*1000,
            np.sqrt(sj1/(n/6))*1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    xtr, ytr, atr = load_split(TRAIN_EPS)
    xva, yva, ava = load_split(VAL_EPS)
    print(f"train {sum(len(x) for x in xtr)} 帧 / val {sum(len(x) for x in xva)} 帧")

    res = {}
    # 基线0: 预测零
    z_all, z_act = [], []
    for y, a in zip(yva, ava):
        z_all.append((y**2)); z_act.append((y[a]**2))
    res["predict-0"] = {"val_rms": float(np.sqrt(np.concatenate([v.ravel() for v in z_all]).mean())*1000),
                        "val_rms_act": float(np.sqrt(np.concatenate([v.ravel() for v in z_act]).mean())*1000),
                        "params": 0, "lat_us": 0}
    # 基线1: 线性ridge(当前步45维)
    X = np.concatenate(xtr); Y = np.concatenate(ytr)
    lam = 1e-3*len(X)
    Wr = np.linalg.solve(X.T@X + lam*np.eye(45), X.T@Y)
    se = sa = n = na = 0.0
    for x, y, a in zip(xva, yva, ava):
        p = np.clip(x@Wr, -EPS_B, EPS_B)
        d2 = (p-y)**2; se += d2.sum(); n += d2.size
        sa += d2[a].sum(); na += d2[a].size
    res["ridge-45d"] = {"val_rms": float(np.sqrt(se/n)*1000),
                        "val_rms_act": float(np.sqrt(sa/na)*1000),
                        "params": 45*6, "lat_us": 1}

    models = {"TCN": TCN(), "S4D-lite": S4DNet(), "Gray-box": GrayBox()}
    for name, model in models.items():
        nparam = sum(p.numel() for p in model.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        t0 = time.time()
        model.train()
        for i, (xb, yb) in enumerate(batches(xtr, ytr, atr, steps=args.steps)):
            loss = torch.nn.functional.huber_loss(model(xb), yb, delta=0.005)
            opt.zero_grad(); loss.backward(); opt.step()
            if (i+1) % 1000 == 0:
                print(f"  [{name}] step {i+1} loss={loss.item():.2e}")
        tr_time = time.time()-t0
        v_all, v_act, v_j1 = evaluate(model, xva, yva, ava)
        # 单步CPU延迟(流式: 喂1帧上下文所需的最小序列)
        model.eval()
        seq = torch.zeros(1, 16, 45)
        with torch.no_grad():
            for _ in range(20): model(seq)
            t0 = time.perf_counter()
            for _ in range(200): model(seq)
            lat = (time.perf_counter()-t0)/200/16*1e6   # 摊到每帧, µs
        res[name] = {"val_rms": float(v_all), "val_rms_act": float(v_act),
                     "val_rms_j1": float(v_j1), "params": int(nparam),
                     "lat_us": float(lat), "train_s": round(tr_time, 1)}
        torch.save(model.state_dict(), OUT / f"{name.replace('/','_')}.pt")

    # 可解释产物
    s4 = models["S4D-lite"]
    f1 = (s4.s1.omega.detach().numpy()/(2*np.pi)).ravel()
    sg1 = np.exp(s4.s1.log_sigma.detach().numpy()).ravel()
    dom = f1[np.argsort(sg1)[:8]]
    res["_interp"] = {"s4d_slowest_poles_hz": sorted(np.round(dom, 2).tolist()),
                      "graybox_W_absmax": float(models['Gray-box'].W.weight.detach().abs().max())}
    (OUT / "bakeoff.json").write_text(json.dumps(res, indent=1))

    print("\n=========== BAKE-OFF (val 集: 每数据集 ep8-9) ===========")
    print(f"{'模型':10s} | {'valRMS(mrad)':12s} | {'活动帧RMS':9s} | {'j1 RMS':7s} | {'参数量':7s} | {'µs/帧':6s}")
    for k in ["predict-0", "ridge-45d", "TCN", "S4D-lite", "Gray-box"]:
        r = res[k]
        j1 = f"{r.get('val_rms_j1', float('nan')):7.3f}" if 'val_rms_j1' in r else "   -   "
        print(f"{k:10s} | {r['val_rms']:12.3f} | {r['val_rms_act']:9.3f} | {j1} | "
              f"{r['params']:7d} | {r['lat_us']:6.1f}")
    print(f"\nS4D 最慢(最持久)极点频率: {res['_interp']['s4d_slowest_poles_hz']} Hz")
    print(f"→ {OUT/'bakeoff.json'}")


if __name__ == "__main__":
    main()
