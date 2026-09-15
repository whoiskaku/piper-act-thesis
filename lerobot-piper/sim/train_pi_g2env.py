#!/usr/bin/env python3
"""**在冻结的 G-2 环境里训练 π**（零机时）。

结构（用户 2026-09-16 定稿）
----------------------------
    训练:  ACT 轨迹 → 冻结 G-2 环境 → π 出 δa → G-2 执行 u+δa
                                   → 用仿真 IMU 残差算损失 → 更新 π

    L = ‖ω_G2(u+δa) − ω_ideal‖²_band  +  λ_a‖δa‖²  +  λ_s‖Δδa‖²

    部署:  ACT + 真机传感器 → π → 安全门控/硬限幅 → u + δa
           **不需要 G-2 / H / Ĥ_θ / 解析求解器**

「G-2 的控制知识保存在 π 的权重里」—— 这才是严格意义上的「用改进后的 G 训练 π」。

为什么 v1 是**纯前馈**（`o_t` 里没有 IMU/编码器）
------------------------------------------------
**G-2 是指令的确定性函数**，没有随机性、没有未建模扰动。所以仿真里的 IMU/编码器
都是 π 已经看得到的指令的确定性函数，**携带不了额外信息** —— 放进去 π 学不到
任何反馈行为，反而可能去拟合仿真特有的特征，迁移更糟。
要让 π 学会用 IMU，必须往仿真里注入 π 观测不到的扰动，那是 v2 的事。
附带好处：D 与 C 都是纯前馈 ⇒ `D−C` 干净地只比「学出的策略 vs 解析求解器」。

⚠ 梯度来源与它的已知缺陷
------------------------
G-2 前向不可微，故用 G-2 **自己的**雅可比线性化：`ω_G2(u+δa) ≈ ω_G2(u) + J⊛δa`。
`J` 的适用性检验**未过事前门禁**（全带 cos 0.900 / 范数比 0.980 /
**相对误差 0.444** vs 门槛 0.30）。但用 J 求解在**真 G-2** 上实测降 **+22.6%**
⇒ 作梯度来源够用。**训练完必须用真 G-2 复核**（`--verify-g2`），不能只看 J 的损失。

⚠ 事前预测（跑之前写下，用于事后判读）
--------------------------------------
实测：要抵消同一面墙，G-2 只需 **0.13 mrad** 的 δa，而真机需要 **0.85 mrad**
—— **约 7 倍**；且 `Δ_G` 与 `Δ_H` 的 `cos` 仅 **0.12**（方向也对不上）。
⇒ **预测 π 在真机上严重欠补，`D−A` 落在 −1% ~ −2%（对照 `C−A` = −8.0%），
   甚至可能 ≈0 或为正（补反）。** 这不是「训练没跑起来」，而是
   **策略在 G-2 中有效、迁移到真机失败** —— 正是本实验要测的东西。

用法:
    python -u sim/train_pi_g2env.py --steps 4000
    python -u sim/train_pi_g2env.py --verify-g2          # 训完用真 G-2 复核
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "sim/step4_gvnext"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

ENV = ROOT / "outputs/p2_pi/g2env"
OUT = ROOT / "outputs/p2_pi"
EPS = 0.010            # **瞬时峰值**硬界（tanh 结构性保证，部署时外面仍限幅兜底）
CHUNK, MLAG = 100, 16
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class PiNet(nn.Module):
    """指令序列 → 整 chunk 的 δa。**chunk 内非因果是合法的**（整 chunk 指令 t₀ 已知）。"""

    def __init__(self, cin=21, ch=64, out=6):
        super().__init__()
        ks, dil = 5, (1, 2, 4, 8)
        layers, c = [], cin
        for d in dil:
            layers += [nn.Conv1d(c, ch, ks, padding=d * (ks - 1) // 2, dilation=d),
                       nn.GELU()]
            c = ch
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(ch, out, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)  # 从 δa≡0 起步

    def forward(self, x):                      # x (B, cin, 200) → (B, 100, 6)
        y = self.head(self.body(x))            # (B, 6, 200)
        return torch.tanh(y[:, :, -CHUNK:]).transpose(1, 2) * EPS


def load_env(limit=0):
    # index.json 只在预计算**全部跑完**才写 —— 缺它时直接扫盘, 方便中途试训
    ix = ENV / "index.json"
    if ix.exists():
        names = json.loads(ix.read_text())["episodes"]
    else:
        names = [f.stem.replace("__", "/") for f in sorted(ENV.glob("*.npz"))]
        print(f"⚠ 无 index.json（预计算未完成）⇒ 直接扫到 {len(names)} 集")
    eps = []
    for name in names:
        f = ENV / (name.replace("/", "__") + ".npz")
        if not f.exists():
            continue
        z = np.load(f)
        eps.append({"name": name, "u": z["u"], "tgt": z["tgt"], "J": z["J"],
                    "contact": z["contact"], "starts": z["starts"], "n": int(z["n"])})
        if limit and len(eps) >= limit:
            break
    return eps


def build_inputs(u: np.ndarray, starts, n):
    """每 chunk 的输入 = [上一 chunk 指令; 本 chunk 指令] 各 100 帧，附一阶差分。"""
    xs, spans = [], []
    for ci, s0 in enumerate(starts):
        if s0 >= n:
            break
        lc = min(CHUNK, n - s0)
        cur = np.zeros((CHUNK, 7), np.float32); cur[:lc] = u[s0:s0 + lc]
        if ci > 0:
            p0 = int(starts[ci - 1]); lp = min(CHUNK, n - p0)
            prev = np.zeros((CHUNK, 7), np.float32); prev[:lp] = u[p0:p0 + lp]
        else:
            prev = np.zeros((CHUNK, 7), np.float32)
        seq = np.concatenate([prev, cur], 0)                       # (200,7)
        # ⚠ 墙主要由**chunk 接缝的跳变**驱动，绝对位姿信息量低且集间差异大 ⇒
        #   显式给出一阶/二阶差分，并把绝对指令减去本 chunk 首帧（去掉位姿偏置）
        d1 = np.concatenate([np.zeros((1, 7), np.float32), np.diff(seq, axis=0)], 0)
        d2 = np.concatenate([np.zeros((1, 7), np.float32), np.diff(d1, axis=0)], 0)
        rel = seq - seq[CHUNK:CHUNK + 1]
        xs.append(np.concatenate([rel, d1, d2], 1).T)              # (21,200)
        spans.append((s0, lc))
    return np.stack(xs), spans


def response(J: torch.Tensor, da: torch.Tensor, n: int) -> torch.Tensor:
    """Σ_k J[t,k] @ δa[t] 落在 ω[t+k] ——「ω 时间轴」，与 tgt 同轴。"""
    out = torch.zeros(n + MLAG, 3, device=da.device, dtype=da.dtype)
    for k in range(MLAG):
        m = n - k
        if m <= 0:
            break
        out[k:k + m] += torch.einsum("tcj,tj->tc", J[:m, k], da[:m])
    return out[:n]


def episode_loss(net, e, bp, lam_a, lam_s, dev, want_da=False):
    x, spans = e["_x"], e["_spans"]
    da_chunks = net(x)                                             # (C,100,6)
    n = e["n"]
    da = torch.zeros(n, 6, device=dev)
    for i, (s0, lc) in enumerate(spans):
        da[s0:s0 + lc] = da_chunks[i, :lc]
    da = da * e["_free"]                                           # 接触门控
    resid = e["_tgt"] + response(e["_J"], da, n)
    num = (bp(resid.unsqueeze(0))[0] ** 2).sum()
    loss = num / e["_den"]
    reg = lam_a * (da ** 2).mean() / (EPS ** 2)
    if lam_s > 0:
        reg = reg + lam_s * ((da[1:] - da[:-1]) ** 2).mean() / (EPS ** 2)
    return (loss + reg, loss.detach(), da.detach()) if want_da else (loss + reg, loss.detach())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam-a", type=float, default=0.0)
    ap.add_argument("--lam-s", type=float, default=1e-2)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="pi_g2env_v1")
    a = ap.parse_args()

    import train_pi_e2 as TP
    bp = TP.make_bp()
    torch.manual_seed(a.seed)
    eps = load_env(a.limit)
    if not eps:
        print("没有预计算的环境 —— 先跑 sim/pi_g2env_precompute.py"); return 1
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(eps))
    nval = max(1, int(len(eps) * a.val_frac))
    val_i, tr_i = set(order[:nval].tolist()), order[nval:].tolist()

    for e in eps:
        e["_xraw"], e["_spans"] = build_inputs(e["u"], e["starts"], e["n"])
    # 通道标准化：统计**只用训练集**（防泄漏）
    cat = np.concatenate([eps[i]["_xraw"] for i in tr_i], 0)
    mu = cat.mean((0, 2), keepdims=True); sd = cat.std((0, 2), keepdims=True) + 1e-8
    print(f"输入标准化: {cat.shape[1]} 通道（统计只用训练集）")
    for e in eps:                                                  # 一次性搬上设备
        x, spans = (e["_xraw"] - mu) / sd, e["_spans"]
        e["_x"] = torch.from_numpy(x.astype(np.float32)).to(DEV)
        e["_spans"] = spans
        e["_tgt"] = torch.from_numpy(e["tgt"].astype(np.float32)).to(DEV)
        e["_J"] = torch.from_numpy(e["J"]).to(DEV)
        e["_free"] = torch.from_numpy((~e["contact"]).astype(np.float32)).to(DEV)[:, None]
        e["_den"] = (bp(e["_tgt"].unsqueeze(0))[0] ** 2).sum().clamp_min(1e-30)

    net = PiNet().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    npar = sum(p.numel() for p in net.parameters())
    print(f"π: {npar} 参数 · 设备 {DEV} · 训练 {len(tr_i)} 集 / 验证 {len(val_i)} 集")
    print(f"λ_a={a.lam_a} λ_s={a.lam_s} · ε={EPS*1e3:.0f} mrad(tanh 结构硬界)")
    print(f"⚠ 损失里的被控对象是 **J**(相对误差 0.444, 未过事前门禁) ⇒ 训完须用真 G-2 复核\n")
    print(f"{'step':>6}{'训练 loss':>11}{'验证 loss':>11}{'验证降幅':>10}{'δa RMS':>9}")

    def evaluate(idx):
        net.eval(); tot = []; rms = []
        with torch.no_grad():
            for i in idx:
                _, l, da = episode_loss(net, eps[i], bp, 0, 0, DEV, want_da=True)
                tot.append(float(l)); rms.append(float((da ** 2).mean().sqrt()) * 1e3)
        net.train()
        return float(np.mean(tot)), float(np.mean(rms))

    best, best_state, t0 = 1e9, None, time.perf_counter()
    for step in range(1, a.steps + 1):
        i = int(rng.choice(tr_i))
        loss, _ = episode_loss(net, eps[i], bp, a.lam_a, a.lam_s, DEV)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % max(a.steps // 20, 1) == 0 or step == a.steps:
            vl, vr = evaluate(sorted(val_i))
            tl, _ = evaluate(tr_i[:min(len(tr_i), 10)])
            print(f"{step:>6}{tl:>11.4f}{vl:>11.4f}{(1-vl)*100:>9.1f}%{vr:>9.3f}")
            if vl < best:
                best = vl
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    if best_state is not None:
        net.load_state_dict(best_state)
    vl, vr = evaluate(sorted(val_i))
    print(f"\n  **最佳验证**: loss {best:.4f} ⇒ 在 **J 口径**下降幅 {(1-best)*100:.1f}%"
          f" · δa RMS {vr:.3f} mrad · 用时 {time.perf_counter()-t0:.0f}s")
    print(f"  ⚠ 这是 **J 口径**的数字。真 G-2 口径须跑 --verify-g2。")

    ck = OUT / f"{a.tag}.pt"
    torch.save({"state_dict": net.state_dict(), "eps_rad": EPS,
                "arch": "PiNet(cin=21,ch=64,dil=1/2/4/8,k=5); 输入=[rel,d1,d2]",
                "val_episodes": [eps[i]["name"] for i in sorted(val_i)],
                "norm_mu": mu, "norm_sd": sd}, ck)
    (OUT / f"{a.tag}.json").write_text(json.dumps(
        {"artifact": a.tag, "n_params": npar, "device": DEV,
         "train_episodes": [eps[i]["name"] for i in tr_i],
         "val_episodes": [eps[i]["name"] for i in sorted(val_i)],
         "hparams": {"steps": a.steps, "lr": a.lr, "lam_a": a.lam_a,
                     "lam_s": a.lam_s, "seed": a.seed, "eps_rad": EPS},
         "val_loss_J": best, "val_reduction_J_percent": (1 - best) * 100,
         "val_da_rms_mrad": vr,
         "caliber": ("**损失里的被控对象是 J**(G-2 自己的雅可比, 相对误差 0.444 "
                     "未过事前 0.30 门禁) ⇒ 此处降幅是 **J 口径**, 非真 G-2、更非真机; "
                     "o_t 只含指令(纯前馈) —— 因为 G-2 是指令的确定性函数, "
                     "仿真 IMU/编码器携带不了额外信息; "
                     "ε 由 tanh 结构性保证, 部署时外面仍限幅+接触门控兜底"),
         "prereg_prediction": ("G-2 只需 0.13 mrad 而真机需 0.85 mrad(**约 7×**), "
                               "且 Delta_G 与 Delta_H 的 cos 仅 0.12 "
                               "=> **预测真机 D−A 落在 −1%~−2%, 甚至 ≈0 或为正**; "
                               "对照 C−A = −8.0%"),
         }, ensure_ascii=False, indent=1))
    print(f"  → {ck}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    import os
    os._exit(rc)
