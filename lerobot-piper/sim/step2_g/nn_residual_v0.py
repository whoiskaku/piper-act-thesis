#!/usr/bin/env python3
"""**最小神经残差 v0** —— 命令→关节轨迹层的残差。物理底座 = 冻结的 G-π-phys-v1。

```
q̂ = q_cmd + Δ,   Δ = NN(近期指令, 模拟状态)          ← 有界输出
q̂ → 刚体传感器模型 → IMU
```

**为什么放在这一层**：A 段是主要缺口；B 段喂入真实关节角后能稳定解释
约 **68%** 的陀螺能量（残差/实测 0.3245，17/17 一致）⇒ 暂不需端到端预测 IMU。

## 防过拟合的六条（执行前已定）

1. **廉价对照**：同特征的**线性时序残差**（岭回归 + 时间滞后窗）并列报告；
2. **小型单层 GRU**（hidden 32，单层），不堆容量；
3. **输入只含推理时可得的量** —— 历史指令与**模拟状态**（G-π-phys-v1 的
   `q_phys`，它是指令的确定性函数）。**真实编码器在训练与验证时都不进输入**；
4. **输出有界的 `Δq_ref`**（`tanh × BOUND`），**不直接预测 IMU**；
5. **按整集划分**；验证是**从第 0 帧起的整集前向滚动** —— GRU 只带自己的隐状态，
   **不持续喂真实关节状态**；
6. 同时检查**关节 RMS** 与**最终 IMU 四区域指标**。

划分用**留一集交叉验证**（17 折）：每集都拿到一份**折外**预测，
「多数集改善 + 中位为正」在全部 17 集上评判，比单次 12/5 划分稳健得多。

## 门槛

**只有验证（折外）上多数集改善且中位改善为正，才冻结神经残差并采最终 holdout。
否则保留 G-π-phys-v1，神经路线判否。**

用法:
    python sim/step2_g/nn_residual_v0.py --run
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "sim/step2_g"))

from sim.step1_v2_loss import ALIGN_L, bandpass41, load_calib, render_ideal  # noqa: E402
from sim.step2_gap_baseline import bare_logs                                  # noqa: E402
from sim.step2_fidelity import CALIB, SEAM_W, regions, fidelity               # noqa: E402
from ab_decompose import kin_gyro                                             # noqa: E402
import g_pi_phys_v1 as PHYS                                                   # noqa: E402

CACHE = ROOT / "outputs/step2_g/nn_v0"
BOUND = 20.0          # 输出界(度) —— 真实残差 max 18.7°, 取 20 足够
HID = 32              # 单层 GRU 隐藏维
LAGS = (1, 2, 4, 8)   # 线性对照的时间滞后窗(帧)


def build_cache():
    """每集算一次：指令 / 实测编码器 / 物理底座轨迹 / 实测 IMU / chunks。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / "feat.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return list(z["eps"])
    eps = []
    for p in bare_logs():
        d = json.loads(p.read_text())
        fr = d["frames"]
        act = np.asarray([f_["a"] for f_ in fr], float)
        cmd = np.degrees(act[:, :6])
        enc = np.asarray([f_["state"] for f_ in fr], float)[:, :6] / 1000.0
        imu = np.asarray([f_["imu"] for f_ in fr], float)[:, 3:6]
        qph = PHYS.predict_qpos(act)                    # 模拟状态(指令的确定性函数)
        kin = render_ideal(d["chunks"], len(fr))
        k = min(len(cmd), len(qph), len(enc) - ALIGN_L, len(kin))
        eps.append({"name": p.stem, "cmd": cmd[:k], "qph": qph[:k],
                    "enc": enc[ALIGN_L:ALIGN_L + k], "imu": imu[ALIGN_L:ALIGN_L + k],
                    "kin": kin[:k], "seams": [c["start"] - ALIGN_L
                                              for c in d["chunks"][1:]
                                              if 0 <= c["start"] - ALIGN_L < k]})
        print(f"  cache {p.stem}  {k} 帧", flush=True)
    np.savez_compressed(f, eps=np.array(eps, dtype=object))
    return eps


def feats(e):
    """逐帧特征（**只含推理时可得的量**）：指令、指令差分、模拟状态、模拟−指令。"""
    c, q = e["cmd"], e["qph"]
    dc = np.vstack([np.zeros((1, 6)), np.diff(c, axis=0)])
    return np.hstack([c, dc, q, q - c]).astype(np.float32)      # (T, 24)


def target(e):
    return (e["enc"] - e["cmd"]).astype(np.float32)             # (T, 6)


# ────────────────────── 廉价对照：线性时序残差 ──────────────────────
def lin_fit(tr):
    X = np.vstack([_lin_x(feats(e)) for e in tr])
    Y = np.vstack([target(e) for e in tr])
    A = X.T @ X + 1e-2 * len(X) * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Y)


def _lin_x(F):
    cols = [F, np.ones((len(F), 1), np.float32)]
    for L in LAGS:
        cols.append(np.vstack([np.repeat(F[:1], L, 0), F[:-L]]))
    return np.hstack(cols)


def lin_pred(W, e):
    return np.clip(_lin_x(feats(e)) @ W, -BOUND, BOUND)


# ────────────────────── 小型单层 GRU ──────────────────────
def gru_fit(tr, seed, epochs=250):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.g = nn.GRU(24, HID, 1, batch_first=True)
            s.h = nn.Linear(HID, 6)

        def forward(s, x):
            y, _ = s.g(x)
            return torch.tanh(s.h(y)) * BOUND      # **有界输出**

    net = Net().to(dev)
    mu = np.vstack([feats(e) for e in tr]).mean(0)
    sd = np.vstack([feats(e) for e in tr]).std(0) + 1e-6
    X = [torch.tensor((feats(e) - mu) / sd, device=dev)[None] for e in tr]
    Y = [torch.tensor(target(e), device=dev)[None] for e in tr]
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    for ep in range(epochs):
        opt.zero_grad()
        loss = sum(torch.nn.functional.smooth_l1_loss(net(x), y)
                   for x, y in zip(X, Y)) / len(X)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    return net, mu, sd, dev


def gru_pred(model, e):
    import torch
    net, mu, sd, dev = model
    with torch.no_grad():
        x = torch.tensor((feats(e) - mu) / sd, device=dev)[None]
        return net(x)[0].cpu().numpy()            # 整集一次前向 = 完整滚动


# ────────────────────── 评价：关节 RMS + IMU 四区域 ──────────────────────
def imu_metric(e, q_hat, R):
    """q̂ → 刚体传感器 → 陀螺 → 四区域相对变化（接缝后 10 帧 / 难处前 10%）。"""
    g = kin_gyro(q_hat)
    n = min(len(g), len(e["kin"]), len(e["imu"]))
    tr = {"ideal": bandpass41(e["kin"][:n, 3:6] @ R.T),
          "real": bandpass41(e["imu"][:n]),
          "dyn": bandpass41(g[:n] @ R.T),
          "seams": [s for s in e["seams"] if s < n], "n": n, "name": e["name"]}
    rg = regions(tr)
    return (fidelity(tr, rg[f"接缝后{SEAM_W}帧"])[2] * 100,
            fidelity(tr, rg["gap能量前10%"])[2] * 100)


def rms(q_hat, e):
    return float(np.sqrt(np.mean((q_hat - e["enc"]) ** 2)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--epochs", type=int, default=250)
    a_ = ap.parse_args()
    if not a_.run:
        print(__doc__.split("用法:")[0]); return 0
    R, _ = load_calib(CALIB)
    eps = build_cache()
    N = len(eps)
    print(f"\n最小神经残差 v0 —— **留一集交叉验证**（{N} 折，每集一份折外预测）")
    print(f"  输入 24 维：指令 / 指令差分 / 模拟状态 q_phys / q_phys−指令")
    print(f"  **真实编码器不进输入**（训练与验证都不进）；输出 tanh×{BOUND:g}° 有界")
    print(f"  基线 = 冻结的 G-π-phys-v1（关节 RMS 0.924，主指标逐集中位 −19.1%）\n")

    rows = []
    for i in range(N):
        tr = [eps[j] for j in range(N) if j != i]
        te = eps[i]
        Wl = lin_fit(tr)
        ql = te["cmd"] + lin_pred(Wl, te)
        mg = gru_fit(tr, seed=1000 + i, epochs=a_.epochs)
        qg = te["cmd"] + gru_pred(mg, te)
        base = te["qph"]                                  # G-π-phys-v1
        r = dict(name=te["name"], base=rms(base, te), lin=rms(ql, te),
                 gru=rms(qg, te))
        r["base_i"] = imu_metric(te, base, R)
        r["lin_i"] = imu_metric(te, ql, R)
        r["gru_i"] = imu_metric(te, qg, R)
        rows.append(r)
        print(f"  [{i+1:2d}/{N}] {te['name']:22s} 关节RMS 底座 {r['base']:.3f} · "
              f"线性 {r['lin']:.3f} · GRU {r['gru']:.3f}   |  主指标 "
              f"{r['base_i'][0]:+6.1f} → 线性 {r['lin_i'][0]:+6.1f} · "
              f"GRU {r['gru_i'][0]:+6.1f}", flush=True)

    B = np.array([[r["base"], r["lin"], r["gru"]] for r in rows])
    S = np.array([[r["base_i"][0], r["lin_i"][0], r["gru_i"][0]] for r in rows])
    T = np.array([[r["base_i"][1], r["lin_i"][1], r["gru_i"][1]] for r in rows])
    print(f"\n{'═'*78}\n【折外结果】{N} 集")
    print(f"  关节 RMS(度) 均值: 底座 {B[:,0].mean():.3f} · "
          f"线性 {B[:,1].mean():.3f} · **GRU {B[:,2].mean():.3f}**")
    for k, lab in ((1, "线性对照"), (2, "GRU")):
        dj = B[:, 0] - B[:, k]
        ds = S[:, k] - S[:, 0]
        dt = T[:, k] - T[:, 0]
        print(f"\n  ── {lab} vs 底座 ──")
        print(f"    关节 RMS  改善 **{int((dj>0).sum())}/{N}**  "
              f"中位 **{np.median(dj/B[:,0])*100:+.1f}%**")
        print(f"    主指标    改善 **{int((ds>0).sum())}/{N}**  "
              f"中位 **{np.median(ds):+.1f}pp**")
        print(f"    难处      改善 {int((dt>0).sum())}/{N}  中位 {np.median(dt):+.1f}pp")

    print(f"\n{'═'*78}\n【门槛】折外上多数集改善 **且** 中位改善为正")
    ok = {}
    for k, lab in ((1, "线性对照"), (2, "GRU")):
        dj = B[:, 0] - B[:, k]; ds = S[:, k] - S[:, 0]
        gj = (dj > 0).sum() > N / 2 and np.median(dj) > 0
        gs = (ds > 0).sum() > N / 2 and np.median(ds) > 0
        ok[lab] = gj and gs
        print(f"  {lab:8s} 关节 {'✓' if gj else '✗'} · IMU 主指标 "
              f"{'✓' if gs else '✗'}  ⇒ {'**通过**' if ok[lab] else '**未通过**'}")
    print("\n⇒ " + ("**GRU 通过 —— 冻结神经残差并采最终 holdout**" if ok["GRU"] else
                    "**未通过 —— 保留 G-π-phys-v1，神经路线判否**"))
    if ok["线性对照"] and not ok["GRU"]:
        print("   （注：线性对照通过而 GRU 未通过 ⇒ 收益不需要非线性，"
              "应先用线性时序残差，不引入网络。）")
    json.dump(rows, open(CACHE / "loo_result.json", "w"),
              ensure_ascii=False, indent=1, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
