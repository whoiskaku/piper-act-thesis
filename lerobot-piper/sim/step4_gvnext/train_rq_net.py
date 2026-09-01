#!/usr/bin/env python3
"""**`r_q` 的一个固定规模小网络** —— `PREREG_G_VNEXT.md` §4 允许的那一次。

触发条件（已满足）：FIR 的折外 3–9 Hz 改善 **−3.4% < 5%**。
预注册限制：**只试一个固定规模**，**不做架构搜索、不调超参网格**。

固定配置（事前写死，不搜）
--------------------------
- 输入 = 与 FIR **完全相同**的因果窗（nlag=16 × 32 特征 = 512 维 + 常数）
  ⇒ 与 FIR 的对比是**同样的信息、只换非线性**。
- 结构 = 512 → 64 (tanh) → 6，单隐层。
- 损失 = 与 FIR 同规：`0.5·MSE(残差) + 0.5·MSE(bp_{3-9}(残差))`。
- 优化 = Adam(lr=1e-3)、weight_decay=1e-4、400 epoch 全批、种子 20260902。
- 折外划分 = 与 FIR **逐字相同**（按集，K=6）。

用法:
    python -u sim/step4_gvnext/train_rq_net.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data import load, bp, valid_files                     # noqa: E402
from train_rq import (features, fir_design, band_err,      # noqa: E402
                      MOVING, BAND, W_BAND, KFOLD)

S1 = ROOT / "outputs/step4_gvnext/s1"
OUT = ROOT / "outputs/step4_gvnext/rq"
NLAG, HID, EPOCHS, LR, WD, SEED = 16, 64, 400, 1e-3, 1e-4, 20260902


def main() -> int:
    import gc
    gc.disable()
    import torch
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    files = valid_files(S1)
    EP = []
    for f in files:
        dd = load(f)
        X, Y, _ = features(dd)
        EP.append({"name": dd["name"], "D": fir_design(X, NLAG), "Y": Y})
    F = EP[0]["D"].shape[1]
    print(f"固定规模小网络 {F} → {HID} (tanh) → 6   "
          f"Adam lr={LR} wd={WD} {EPOCHS}ep seed={SEED}")
    print(f"训练族 {len(EP)} 条 · 折外按集 K={KFOLD}（与 FIR 逐字相同）\n")

    # 标准化用**折内**统计（否则泄漏）
    folds = [list(range(i, len(EP), KFOLD)) for i in range(KFOLD)]
    n39 = d39 = nb = db = 0.0
    pj = {j: [0.0, 0.0] for j in MOVING}
    ins39 = insd = 0.0
    for k, te in enumerate(folds):
        tr = [i for i in range(len(EP)) if i not in te]
        Dtr = np.vstack([EP[i]["D"] for i in tr])
        mu, sd = Dtr.mean(0), Dtr.std(0) + 1e-8
        net = torch.nn.Sequential(torch.nn.Linear(F, HID), torch.nn.Tanh(),
                                  torch.nn.Linear(HID, 6)).double()
        opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
        # **目标也要标准化** —— 残差量级约 1e-3 rad，而标准初始化的网络输出是
        # O(0.1–1)，差 100–1000 倍。首次运行因此样本内就 −591%（**没收敛**，
        # 不是模型类不行）。这是数值调理，不是超参搜索。
        ysd = float(np.vstack([EP[i]["Y"] for i in tr]).std()) + 1e-12
        # ⚠ 带内项也要**按自身尺度归一**，否则它的 MSE 小三个数量级、
        # 「0.5 权重」形同虚设（与 FIR 侧同一处修正）。
        bsd = float(np.vstack([bp(EP[i]["Y"], *BAND) for i in tr]).std()) + 1e-12
        Xs = [torch.tensor((EP[i]["D"] - mu) / sd) for i in tr]
        Ys = [torch.tensor(EP[i]["Y"] / ysd) for i in tr]
        Yb = [torch.tensor(bp(EP[i]["Y"], *BAND) / bsd) for i in tr]
        # 带通是线性算子 ⇒ 对网络输出做同一带通即可
        from scipy.signal import firwin
        ker = torch.tensor(firwin(41, list(BAND), fs=20.0, pass_zero=False)).double()
        def bpt(y):
            yp = torch.nn.functional.pad(y.T[:, None, :], (20, 20))
            return torch.nn.functional.conv1d(yp, ker.flip(0)[None, None, :])[:, 0].T
        for ep in range(EPOCHS):
            opt.zero_grad()
            loss = 0.0
            for x, y, yb in zip(Xs, Ys, Yb):
                p = net(x)
                loss = loss + (1 - W_BAND) * ((p - y) ** 2).mean() \
                            + W_BAND * ((bpt(p) * (ysd / bsd) - yb) ** 2).mean()
            (loss / len(Xs)).backward()
            opt.step()
            if ep in (0, EPOCHS // 2, EPOCHS - 1):
                tr_loss = float(loss) / len(Xs)
                if ep == 0: l0 = tr_loss
                if ep == EPOCHS - 1:
                    print(f"    折{k+1} 损失 {l0:.4f} → {tr_loss:.4f}"
                          f"  (1.0 = 预测全零)", flush=True)
        with torch.no_grad():
            for i in tr:                                   # 样本内(该折)
                P = net(torch.tensor((EP[i]["D"] - mu) / sd)).numpy() * ysd
                a, b = band_err(EP[i]["Y"], P, *BAND); ins39 += a; insd += b
            for i in te:                                   # 折外
                P = net(torch.tensor((EP[i]["D"] - mu) / sd)).numpy() * ysd
                a, b = band_err(EP[i]["Y"], P, *BAND); n39 += a; d39 += b
                a, b = band_err(EP[i]["Y"], P, 0.5, 9.0); nb += a; db += b
                A, B = bp(EP[i]["Y"], *BAND), bp(P, *BAND)
                for j in MOVING:
                    pj[j][0] += float(((A - B)[:, j] ** 2).sum())
                    pj[j][1] += float((A[:, j] ** 2).sum())


    r39 = 1 - n39 / max(d39, 1e-30)
    rb = 1 - nb / max(db, 1e-30)
    ri = 1 - ins39 / max(insd, 1e-30)
    print(f"\n  {'':<22}{'3–9 Hz 削减':>14}{'宽带削减':>12}")
    print(f"  {'小网络 折外':<22}{r39*100:>13.1f}%{rb*100:>11.1f}%")
    print(f"  {'小网络 样本内':<22}{ri*100:>13.1f}%{'':>12}")
    print(f"  {'FIR 折外(最好)':<22}{-3.4:>13.1f}%{-5.9:>11.1f}%")
    print(f"  {'FIR 样本内(最好)':<22}{8.8:>13.1f}%{'':>12}")
    print(f"\n  逐关节折外 3–9 Hz 削减")
    for j in MOVING:
        print(f"    j{j+1}: {(1 - pj[j][0]/max(pj[j][1],1e-30))*100:>6.1f}%")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "net.json").write_text(json.dumps(
        {"arch": f"{F}->{HID}(tanh)->6", "epochs": EPOCHS, "lr": LR, "wd": WD,
         "seed": SEED, "nlag": NLAG, "w_band": W_BAND, "kfold": KFOLD,
         "oof_reduction_3_9": float(r39), "oof_reduction_broad": float(rb),
         "insample_reduction_3_9": float(ri),
         "per_joint_3_9": {f"j{j+1}": float(1 - pj[j][0]/max(pj[j][1],1e-30))
                           for j in MOVING},
         "note": "预注册允许的唯一一次网络尝试; 未做架构搜索或超参网格",
         }, indent=1, ensure_ascii=False))
    print(f"\n  → {OUT/'net.json'}")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
