#!/usr/bin/env python3
"""**补上缺的那一格**：用 G-2 预测墙，在同一求解器与同一部署约束下能交付多少？（零机时）

已知的两个锚点（同一解析求解器 + 同一约束：ε=10 mrad 峰值硬界 / chunk 钳位 / 接触门控，
被控对象 = 已验证的 H）：

    线性 W      可预测性 31.6%  →  交付 **5.08%**
    **G-2**     可预测性 **52.1%**  →  **？**   ← 本脚本
    完美预测    可预测性 100%    →  交付 **22.07%**

⚠ 口径统一（**与旧的 5.08 / 22.07 不同**）
------------------------------------------
旧的两个数是在 **13 集（P0 5 + P1 8）**上测的，而 **P1 的 `e₀ = gap − H⊛da` 依赖 H**。
本脚本改为 **P0-only（5 集，无探针 ⇒ `e₀ = gap`，H 不进 `e₀`）**，
并在**同一批 P0** 上**重算三臂**，得到自洽比较。旧的 13 集数字仅作背景，不可并列。

三臂
----
1. **线性 W**（指令侧，逐集留一折外拟合）
2. **G-2 预测**（`ê₀ = calib(ω̂_G2) − ideal`；G-2 **冻结，不在本数据上拟合任何东西**）
3. **完美预测**（诊断上界，用真实 e₀ 的 horizon；不可部署）

G-2 是**纯前馈**（只吃命令）且**跨数据集**（拟合于 step4_gvnext，评价于 ab4）
⇒ 无需折外。

⚠ `ALIGN_L=3` 与标定 `(R,b)` **事前固定**，不扫位移、不重拟旋转。
⚠ ab4 无时间戳 ⇒ G 前向在**假设均匀 20 Hz** 下跑。

用法:
    python -u sim/delivery_g2_predictor.py
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

import fit_step1_candidate as S1                                    # noqa: E402
import g2_on_ab4 as G                                               # noqa: E402

ALIGN_L, EPS, KMULT = S1.ALIGN_L, 0.010, 1.0
OUT = ROOT / "outputs/step1_H/delivery_g2_predictor.json"


def load():
    """P0 5 集：e₀ / x（W 用）/ contact / cap / G-2 的 ê₀。"""
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    calib = json.loads(
        (ROOT / "outputs/step1_H/step1_H_v1.json").read_text())["calibration"]
    R32 = np.asarray(calib["R"], np.float32)
    b32 = np.asarray(calib.get("b", [0, 0, 0]), np.float32)
    rend = IdealChunkRenderer()
    g2 = G.G2()
    E = []
    for p in S1.discover_logs(ROOT / "outputs/ab4", ("P0",))["P0"]:
        ep = S1.load_episode(p, "P0", np.asarray(R32, np.float64),
                             np.asarray(b32, np.float64), rend, 550)
        pay = json.loads(p.read_text())
        fr = pay["frames"]
        _, ideal_sim, cap = S1._render_logged_chunks(pay.get("chunks", []), rend, len(fr))
        ideal = S1.calibrate_ideal_imu(ideal_sim.astype(np.float32), R32, b32)
        w = g2.omega(G.ab4_to_dd(p, p.stem))
        w6 = np.zeros((len(w), 6), np.float32); w6[:, 3:] = w
        ws = S1.calibrate_ideal_imu(w6, R32, b32)[:, 3:]
        n = min(len(ep.gap), len(ws) - ALIGN_L, len(ideal))
        E.append({"name": p.stem, "n": n,
                  "e0": np.asarray(ep.gap, float)[:n],
                  "e0h": ws[ALIGN_L:ALIGN_L + n].astype(float) - ideal[:n, 3:6].astype(float),
                  "x": np.asarray(ep.x, float)[:n],
                  "contact": np.asarray(ep.contact_mask, bool)[:n],
                  "cap": np.asarray(cap[:n], int)})
    return E


def roll(e, H, pred):
    """严格因果滚动 + chunk 钳位 + 接触门控 + ε。`pred` 提供 horizon 的 ê₀。"""
    n, M1 = e["n"], S1.N_LAGS
    hs_full = H.reshape(M1 * 3, 6)
    lam = KMULT * max(float(np.trace(hs_full.T @ hs_full)) / 6.0, np.finfo(float).tiny)
    da = np.zeros((n, 6)); resp = np.zeros((n + M1, 3)); nsat = 0
    for t in range(n):
        K = max(int(min(S1.M, e["cap"][t] - t)), 0)
        tgt = np.stack([pred[min(t + k, n - 1)] + resp[t + k] for k in range(K + 1)])
        hs = H[:K + 1].reshape((K + 1) * 3, 6)
        d = -np.linalg.solve(hs.T @ hs + lam * np.eye(6), hs.T @ tgt.reshape(-1))
        nsat += int(np.any(np.abs(d) > EPS))
        d = np.clip(d, -EPS, EPS)
        if e["contact"][t]:
            d[:] = 0.0
        da[t] = d
        for k in range(M1):
            resp[t + k] += H[k] @ d
    return da, nsat / max(n, 1)


def score(E, H, das):
    num = den = 0.0
    rms, sat = [], []
    for e, (da, sp) in zip(E, das):
        a = S1.bandpass41(e["e0"])
        b = S1.bandpass41(e["e0"] + S1.causal_conv(H, da))
        num += float(np.square(b).sum()); den += float(np.square(a).sum())
        rms.append(float(np.sqrt(np.mean(da ** 2))) * 1e3); sat.append(sp * 100)
    return 1 - num / max(den, 1e-30), float(np.mean(rms)), float(np.mean(sat))


def main() -> int:
    import gc
    gc.disable()
    E = load()
    H = np.ascontiguousarray(np.load(ROOT / "outputs/step1_H/step1_H_v1.npz")["H"], float)
    print(f"ab4 **P0-only** {len(E)} 集（e₀ = gap，H 不进 e₀）· "
          f"ε={EPS*1e3:.0f} mrad 峰值 · chunk 钳位 · 接触门控 · ALIGN_L={ALIGN_L} 固定")
    print(f"⚠ 与旧的 5.08 / 22.07（13 集，P1 的 e₀ 依赖 H）**不可并列**；本表三臂同批重算")
    print(f"⚠ ab4 无时间戳 ⇒ G 前向假设均匀 20 Hz\n")

    # 可预测性（同 g2_predicts_wall 的定义）
    def predictability(E, preds):
        n_ = d_ = 0.0
        for e, p in zip(E, preds):
            a, b = S1.bandpass41(e["e0"]), S1.bandpass41(p)
            n_ += float(np.square(a - b).sum()); d_ += float(np.square(a).sum())
        return 1 - n_ / max(d_, 1e-30)

    # 臂1：线性 W（逐集留一折外）
    W_pred = []
    for i in range(len(E)):
        tr = [E[j] for j in range(len(E)) if j != i]
        X = np.concatenate([t["x"] for t in tr]); Y = np.concatenate([t["e0"] for t in tr])
        coef, _ = S1._ridge_solve(X, Y, 0.001)
        W_pred.append(E[i]["x"] @ coef)
    arms = {
        "线性 W（折外）": W_pred,
        "**G-2 预测**（冻结，跨数据集）": [e["e0h"] for e in E],
        "完美预测（诊断上界）": [e["e0"] for e in E],
    }
    print(f"  {'预测器':<28}{'可预测性':>10}{'交付降幅':>10}{'δa RMS':>9}{'触界%':>8}")
    res = {}
    for name, preds in arms.items():
        pr = predictability(E, preds)
        das = [roll(e, H, p) for e, p in zip(E, preds)]
        r, rms, sa = score(E, H, das)
        res[name.replace("*", "")] = {"predictability": pr, "reduction": r,
                                "da_rms_mrad": rms, "sat_pct": sa}
        print(f"  {name:<28}{pr*100:>9.1f}%{r*100:>9.2f}%{rms:>9.2f}{sa:>8.2f}")

    k = list(res)
    w, g, o = res[k[0]], res[k[1]], res[k[2]]
    print(f"\n  ── 判读 ──")
    print(f"  G-2 相对线性 W：可预测性 {w['predictability']*100:.1f}% → "
          f"{g['predictability']*100:.1f}%　交付 {w['reduction']*100:.2f}% → "
          f"**{g['reduction']*100:.2f}%**（{(g['reduction']-w['reduction'])*100:+.2f} pp）")
    span = o["reduction"] - w["reduction"]
    got = g["reduction"] - w["reduction"]
    print(f"  占「W → 完美」可得空间的 **{got/max(span,1e-12)*100:.0f}%**"
          f"（完美 {o['reduction']*100:.2f}%）")
    print(f"\n  ⚠ 这是**离线**数字，被控对象是 H（模型），**不是真机验证**。")
    print(f"  ⚠ 训练/评价数据是**裸 ACT**；C 臂的 π 实际修的是 **FF 之后**的墙。")
    OUT.write_text(json.dumps(
        {"arms": res, "eps_rad": EPS, "n_episodes": len(E),
         "caliber": ("ab4 **P0-only**(无探针⇒e0=gap, H 不进 e0); 三臂**同批重算**, "
                     "与旧的 5.08/22.07(13集, P1 的 e0 依赖 H)**不可并列**; "
                     "G-2 冻结于 step4_gvnext, 在 ab4 上评价 ⇒ **跨数据集**且**纯前馈**, 无需折外; "
                     "W 逐集留一折外; ALIGN_L=3 与 (R,b) 事前固定不扫; "
                     "ab4 无时间戳⇒假设均匀 20Hz; **离线数字, 被控对象是 H 模型, 非真机验证**"),
         }, indent=1, ensure_ascii=False))
    print(f"\n  → {OUT}")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
