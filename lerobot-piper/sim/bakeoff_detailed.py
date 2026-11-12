#!/usr/bin/env python3
"""P2.3b — 细致训练对比: 逐模型lr扫描 + 3种子×6000步 + 收敛曲线 + 切片。

阶段1: 每模型 lr∈{1e-3,2e-3,5e-3} ×1500步(seed0) → 各自最优lr(公平)
阶段2: 最优lr × seed∈{0,1,2} × 6000步, 每500步记录 val RMS(全帧/活动/接触)
输出: outputs/p2_bakeoff/detailed.json
"""
from __future__ import annotations
import json
import time

import numpy as np
import torch

from bakeoff_backbones import (EPS_B, ROOT, TAGS, TRAIN_EPS, VAL_EPS, GrayBox,
                               S4DNet, TCN, batches, load_split, OUT)

MODELS = {"TCN": TCN, "S4D-lite": S4DNet, "Gray-box": GrayBox}
LRS = [1e-3, 2e-3, 5e-3]
SWEEP_STEPS, FULL_STEPS, EVAL_EVERY = 1500, 6000, 500


def load_masks(eps_list):
    """val 帧级掩码: (active, contact, is_pp2) 与 load_split 同序同长。"""
    act, ct, pp2 = [], [], []
    for tag in TAGS:
        for ep in eps_list:
            z = np.load(ROOT / tag / f"ep{ep:03d}.npz")
            n = len(z["e"]) - 8
            act.append(~z["mask_still"][:n])
            ct.append(z["mask_contact"][:n])
            pp2.append(np.full(n, tag == "pp2-deploy"))
    return act, ct, pp2


def val_eval(model, xva, yva, mact, mct, mpp2):
    model.eval()
    agg = {k: [0.0, 0] for k in ("all", "act", "contact", "pp2", "ins")}
    with torch.no_grad():
        for x, y, a, c, p in zip(xva, yva, mact, mct, mpp2):
            pr = model(torch.from_numpy(x).unsqueeze(0))[0].numpy()
            d2 = (pr - y) ** 2
            for key, m in (("all", np.ones(len(y), bool)), ("act", a),
                           ("contact", c), ("pp2", p), ("ins", ~p)):
                agg[key][0] += d2[m].sum(); agg[key][1] += d2[m].size
    return {k: float(np.sqrt(v / max(c_, 1)) * 1000) for k, (v, c_) in agg.items()}


def train_one(cls, lr, seed, steps, xtr, ytr, atr, evalf):
    torch.manual_seed(seed); np.random.seed(seed)
    model = cls()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    curve = []
    model.train()
    for i, (xb, yb) in enumerate(batches(xtr, ytr, atr, steps=steps)):
        loss = torch.nn.functional.huber_loss(model(xb), yb, delta=0.005)
        opt.zero_grad(); loss.backward(); opt.step()
        if (i + 1) % EVAL_EVERY == 0:
            curve.append({"step": i + 1, **evalf(model)})
            model.train()
    return model, curve


def main():
    xtr, ytr, atr = load_split(TRAIN_EPS)
    xva, yva, _ = load_split(VAL_EPS)
    mact, mct, mpp2 = load_masks(VAL_EPS)
    evalf = lambda m: val_eval(m, xva, yva, mact, mct, mpp2)
    out = {"sweep": {}, "runs": {}}

    for name, cls in MODELS.items():
        best = (None, 1e9)
        out["sweep"][name] = {}
        for lr in LRS:
            t0 = time.time()
            _, curve = train_one(cls, lr, 0, SWEEP_STEPS, xtr, ytr, atr, evalf)
            v = curve[-1]["all"]
            out["sweep"][name][str(lr)] = v
            print(f"[sweep {name}] lr={lr:g} → val {v:.3f} mrad ({time.time()-t0:.0f}s)", flush=True)
            if v < best[1]:
                best = (lr, v)
        out["runs"][name] = {"best_lr": best[0], "seeds": []}
        for seed in (0, 1, 2):
            t0 = time.time()
            _, curve = train_one(cls, best[0], seed, FULL_STEPS, xtr, ytr, atr, evalf)
            out["runs"][name]["seeds"].append({"seed": seed, "curve": curve})
            f = curve[-1]
            print(f"[full {name}] seed{seed} lr={best[0]:g} → all {f['all']:.3f} act {f['act']:.3f} "
                  f"contact {f['contact']:.3f} pp2 {f['pp2']:.3f} ins {f['ins']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        (OUT / "detailed.json").write_text(json.dumps(out, indent=1))

    print("\n===== 终点(6000步) 跨种子 mean±std, val全帧 RMS(mrad) =====")
    for name in MODELS:
        fin = [s["curve"][-1] for s in out["runs"][name]["seeds"]]
        arr = np.array([f["all"] for f in fin])
        act = np.array([f["act"] for f in fin])
        ct = np.array([f["contact"] for f in fin])
        print(f"{name:9s}: all {arr.mean():.3f}±{arr.std():.3f}  act {act.mean():.3f}±{act.std():.3f}  "
              f"contact {ct.mean():.3f}±{ct.std():.3f}  (best_lr={out['runs'][name]['best_lr']:g})")
    print(f"→ {OUT/'detailed.json'}")


if __name__ == "__main__":
    main()
