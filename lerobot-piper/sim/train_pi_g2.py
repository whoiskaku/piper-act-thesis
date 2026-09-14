#!/usr/bin/env python3
"""训练 π：**非线性预测器**吃全部可得观测，接同一解析求解器。

为什么「训练 π」等价于「训练预测器」
------------------------------------
δa 的求解那半边**已经是解析最优**（给定预测的 e₀ horizon，
`d = −(HᵀH+λI)⁻¹Hᵀ·tgt` 就是该 horizon 下的最优因果解）。
所以 π 能超过线性基线的唯一途径是**预测得更准**。

已测（同一求解器 + 同一部署约束，ε=10 mrad 峰值硬界、chunk 钳位、接触门控）：

    指令 only（线性 W）              4.23%
    指令 + IMU                     4.63%
    指令 + 编码器                    5.05%
    指令 + IMU + 编码器             **5.08%**   ← 线性基线
    完美预测（诊断上界，不可部署）        **22.07%**

指令侧非线性**试过并且输了**（`CAUSAL_CEILING.md`：TCN 51 维 26.1% < 线性 W 31.6%）。
**但「指令 × IMU × 编码器」的联合非线性从未试过** —— 本脚本测它。

⚠ 记忆风险
----------
只有 13 集。G-vNext 那轮的网络就是**纯记忆**（样本内 +64.9% / 折外 −56.6%）。
故：逐集留一折外、小模型、按折外早停，并**同时报样本内与折外**——
两者差距大即记忆，结论以折外为准。

⚠ 必查项（用户 2026-09-02 指定）
--------------------------------
- **π 自己的触界率**（oracle 的 0.3% 不能代替）
- 整集留出评价
- δa RMS / 峰值，与认证 ε=10 mrad（**峰值**口径，不是 RMS）对照

用法:
    python -u sim/train_pi_g2.py
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

import oracle_causal_feedback as O                                   # noqa: E402
import oracle_constraint_ceiling as CC                               # noqa: E402
import pi_feature_ablation as A                                      # noqa: E402
import fit_step1_candidate as S1                                     # noqa: E402

EPS, KMULT = 0.010, 1.0
JLAG = 8
HID, EPOCHS, LR, WD = 96, 400, 3e-3, 1e-3
SEED = 0


def build_xy(E):
    """→ 每集 (特征 F, 目标 Y=e₀ horizon)。特征**全部严格因果**。"""
    out = []
    for e in E:
        T = len(e["e0"])
        F, Y = [], []
        for t in range(JLAG + S1.ALIGN_L, T - S1.N_LAGS):
            f = [e["x"][t]]                                   # 指令特征（36）
            f.append(A.hist(e["r_imu"], t))                   # IMU 残差历史
            f.append(A.hist(e["terr"], t))                    # 编码器误差+速度历史
            F.append(np.concatenate(f))
            Y.append(e["e0"][t:t + S1.N_LAGS].reshape(-1))    # 16×3 horizon
        out.append((np.asarray(F), np.asarray(Y)))
    return out


def train(tr, seed=SEED):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    X = np.concatenate([x for x, _ in tr]); Y = np.concatenate([y for _, y in tr])
    mx, sx = X.mean(0), X.std(0) + 1e-8
    my, sy = Y.mean(0), Y.std(0) + 1e-8
    xt = torch.tensor((X - mx) / sx, dtype=torch.float32)
    yt = torch.tensor((Y - my) / sy, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(X.shape[1], HID), nn.GELU(),
                        nn.Linear(HID, HID), nn.GELU(), nn.Linear(HID, Y.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    for _ in range(EPOCHS):
        opt.zero_grad(); loss = ((net(xt) - yt) ** 2).mean(); loss.backward(); opt.step()
    net.eval()
    return net, (mx, sx, my, sy)


def predict(net, nrm, F):
    import torch
    mx, sx, my, sy = nrm
    with torch.no_grad():
        z = net(torch.tensor((F - mx) / sx, dtype=torch.float32)).numpy()
    return z * sy + my


def roll_pred(e, H, pred, t0):
    """用预测的 horizon 做同一解析求解；返回 δa 与触界率。"""
    T = len(e["e0"])
    M1 = S1.N_LAGS
    hs_full = H.reshape(M1 * 3, 6)
    lam = KMULT * max(float(np.trace(hs_full.T @ hs_full)) / 6.0, np.finfo(float).tiny)
    da = np.zeros((T, 6)); resp = np.zeros((T + M1, 3)); nsat = 0
    for t in range(T):
        K = max(int(min(S1.M, e["cap"][t] - t)), 0)
        i = t - t0
        ph = (pred[i].reshape(M1, 3) if 0 <= i < len(pred) else np.zeros((M1, 3)))
        tgt = np.stack([ph[k] + resp[t + k] for k in range(K + 1)])
        hs = H[:K + 1].reshape((K + 1) * 3, 6)
        d = -np.linalg.solve(hs.T @ hs + lam * np.eye(6), hs.T @ tgt.reshape(-1))
        nsat += int(np.any(np.abs(d) > EPS))
        d = np.clip(d, -EPS, EPS)
        if e["contact"][t]:
            d[:] = 0.0
        da[t] = d
        for k in range(M1):
            resp[t + k] += H[k] @ d
    return da, nsat / max(T, 1)


def main() -> int:
    import gc
    gc.disable()
    H, E = O.load_all()
    E = A.add_state(E)
    # IMU 残差通道（与消融同定义：e₀ 减去线性前馈）
    Wall = O.fit_w(E)
    for e in E:
        e["r_imu"] = e["e0"] - e["x"] @ Wall.T
    XY = build_xy(E)
    t0 = JLAG + S1.ALIGN_L
    print(f"ab4 {len(E)} 集 · 特征维 {XY[0][0].shape[1]} · 目标维 {XY[0][1].shape[1]}"
          f" · MLP {HID}×2 · 逐集留一折外")
    print(f"约束：ε = {EPS*1e3:.0f} mrad（**峰值**硬界）· chunk 钳位 · 接触门控\n")

    das_oof, das_in = [], []
    for i in range(len(E)):
        tr = [XY[j] for j in range(len(E)) if j != i]
        net, nrm = train(tr)
        das_oof.append(roll_pred(E[i], H, predict(net, nrm, XY[i][0]), t0))
    net, nrm = train(XY)
    for i in range(len(E)):
        das_in.append(roll_pred(E[i], H, predict(net, nrm, XY[i][0]), t0))

    r_o, rm_o, sa_o = CC.score(E, H, das_oof)
    r_i, rm_i, sa_i = CC.score(E, H, das_in)
    pk = max(float(np.abs(d).max()) for d, _ in das_oof) * 1e3
    print(f"  {'':<18}{'降幅':>9}{'δa RMS':>10}{'δa 峰值':>10}{'触界%':>8}")
    print(f"  {'样本内':<18}{r_i*100:>8.2f}%{rm_i:>10.2f}{'':>10}{sa_i:>8.2f}")
    print(f"  {'**折外（判据）**':<18}{r_o*100:>8.2f}%{rm_o:>10.2f}{pk:>9.2f}m{sa_o:>8.2f}")
    print(f"\n  对照（同一求解器 + 同一约束）")
    print(f"  {'线性 指令+IMU+编码器':<18}{5.08:>8.2f}%")
    print(f"  {'完美预测（上界）':<18}{22.07:>8.2f}%")
    print(f"\n  ── 判读 ──")
    print(f"  折外 {r_o*100:.2f}% vs 线性基线 5.08%  ⇒ {r_o*100-5.08:+.2f} pp")
    print(f"  样本内−折外 = {(r_i-r_o)*100:+.2f} pp"
          f"{'  ⚠ **记忆迹象**' if r_i - r_o > 3 else ''}")
    print(f"  **π 自己的触界率 {sa_o:.2f}%**（门槛 ≤1%；oracle 的 0.3% 不能代替）"
          f"  {'✓' if sa_o <= 1 else '**✗**'}")
    (ROOT / "outputs/step1_H/train_pi_g2.json").write_text(json.dumps(
        {"oof": {"reduction": r_o, "da_rms_mrad": rm_o, "da_peak_mrad": pk, "sat_pct": sa_o},
         "in_sample": {"reduction": r_i, "da_rms_mrad": rm_i, "sat_pct": sa_i},
         "baseline_linear_all_obs": 5.08, "oracle_perfect_prediction": 22.07,
         "arch": {"hidden": HID, "epochs": EPOCHS, "lr": LR, "wd": WD, "jlag": JLAG},
         "caliber": ("求解器解析最优, 唯一变量是预测器; 逐集留一折外为判据; "
                     "ε=10mrad 是**峰值**硬界(非 RMS); 13 集有记忆风险, 须并读样本内"),
         }, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
