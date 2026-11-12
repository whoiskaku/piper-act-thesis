#!/usr/bin/env python3
"""**用真 G-2 复核 π**（零机时）—— 训练损失用的是 J，这里必须换成真前向。

为什么必须做
------------
训练时把 `ω_G2(u+δa)` 线性化成 `ω_G2(u) + J⊛δa`，而 `J` 的适用性
**未过事前门禁**（相对误差 0.444 vs 门槛 0.30）。
π 完全可能**钻 J 的不准**：在 J 口径下很好看，真跑 G-2 却不行。
（征兆已经有了：π 的 δa RMS ≈ 2.0 mrad，而 J 解析解只用 0.13 mrad。）

三条对照（同一批**留出**集，被控对象都是**真 G-2**）
---------------------------------------------------
| 臂 | δa 来自 |
|---|---|
| `zero` | δa ≡ 0（基线） |
| **`pi`** | **训练好的 π**（部署时就是它） |
| `Jsolve` | 用 G-2 自己的雅可比做解析求解（环境内的「教师上限」参照） |

另报 **J 口径**的同一组数字 ⇒ 两者之差 = **π 钻 J 不准的程度**。

用法:
    python -u sim/verify_pi_g2env.py [--tag pi_g2env_v1] [--n 8]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "sim/step4_gvnext"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fit_step1_candidate as S1                                     # noqa: E402
from freeze_pi_g2wall import G2v0                                    # noqa: E402
import train_pi_g2env as T                                           # noqa: E402

L, M, EPS = S1.ALIGN_L, S1.M, T.EPS
OUT = ROOT / "outputs/p2_pi/verify_pi_g2env.json"


def omega_cal(g2, u, R, b, da=None):
    from lerobot.common.policies.act.step1_correction_runtime import calibrate_ideal_imu
    cmd = np.array(u, np.float64, copy=True)
    if da is not None:
        cmd[:len(da), :6] += da
    n = len(cmd)
    dd = {"name": "x", "cmd": cmd, "t_send": np.arange(n) / 20.0,
          "t_read": np.arange(n) / 20.0,
          "tg_i": np.arange(n * 5) / 100.0, "gyro100": np.zeros((n * 5, 3))}
    w = g2.omega(dd)
    w6 = np.zeros((len(w), 6), np.float32); w6[:, 3:] = w
    return calibrate_ideal_imu(w6, R, b)[:, 3:].astype(np.float64)


def roll_J(tgt, Jd, n):
    """与 H 求解器同构、但每帧用 `Jd[t]`（ω 时间轴）。"""
    da = np.zeros((n, 6)); resp = np.zeros((n + M + 1, 3))
    for t in range(n):
        K = min(M, n - 1 - t)
        hs = Jd[t][:K + 1].reshape((K + 1) * 3, 6).astype(np.float64)
        lam = max(float(np.trace(hs.T @ hs)) / 6.0, 1e-30)
        y = np.stack([tgt[min(t + k, n - 1)] + resp[t + k] for k in range(K + 1)])
        d = -np.linalg.solve(hs.T @ hs + lam * np.eye(6), hs.T @ y.reshape(-1))
        da[t] = np.clip(d, -EPS, EPS)
        for k in range(M + 1):
            resp[t + k] += Jd[t][k] @ da[t]
    return da


def main() -> int:
    import gc
    gc.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pi_g2env_v1")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()

    import train_pi_e2 as TP
    bp_t = TP.make_bp()
    ck = torch.load(ROOT / f"outputs/p2_pi/{a.tag}.pt", map_location=T.DEV,
                    weights_only=False)   # 自产 ckpt, 内含 numpy 的归一化统计
    net = T.PiNet().to(T.DEV); net.load_state_dict(ck["state_dict"]); net.eval()
    mu, sd = ck["norm_mu"], ck["norm_sd"]
    val_names = ck["val_episodes"][:a.n]
    calib = json.loads(
        (ROOT / "outputs/step1_H/step1_H_v1.json").read_text())["calibration"]
    R = np.asarray(calib["R"], np.float32); b = np.asarray(calib.get("b", [0, 0, 0]), np.float32)
    g2 = G2v0()

    print(f"用**真 G-2** 复核 `{a.tag}` · 留出集 {len(val_names)} 集")
    print(f"⚠ 训练损失用的是 J（相对误差 0.444）⇒ 此处才是 G-2 口径的真数字\n")
    print(f"{'集':<18}{'基线':>9}{'π(真G2)':>10}{'降幅':>8}"
          f"{'Jsolve(真G2)':>13}{'降幅':>8}{'π 的 J 口径':>12}{'δa RMS':>8}")

    agg = {k: 0.0 for k in ("base", "pi", "js", "pi_J")}
    rms_all = []
    for name in val_names:
        z = np.load(ROOT / "outputs/p2_pi/g2env" / (name.replace("/", "__") + ".npz"))
        u, tgt, Jd, contact = z["u"], z["tgt"].astype(np.float64), z["J"], z["contact"]
        n = int(z["n"]); starts = z["starts"]
        x, spans = T.build_inputs(u, starts, n)
        xn = torch.from_numpy(((x - mu) / sd).astype(np.float32)).to(T.DEV)
        with torch.no_grad():
            dch = net(xn).cpu().numpy()
        da = np.zeros((n, 6))
        for i, (s0, lc) in enumerate(spans):
            da[s0:s0 + lc] = dch[i, :lc]
        da[contact[:n]] = 0.0
        da_js = roll_J(tgt, Jd, n)
        da_js[contact[:n]] = 0.0

        def energy_true(d):
            w = omega_cal(g2, u, R, b, d)
            m = min(len(w) - L, n)
            # 与训练同一目标：ω 轴上 ω(u+δa)[s] 对 ideal[s−L]
            r = np.zeros((m, 3))
            r[:] = w[L:L + m] - (tgt[L:L + m] - 0)  # 占位, 下面用真 ideal 重算
            return None

        # 真 ideal = ω_G2(u)[s] − tgt[s]（由预计算定义反解，零误差）
        w0 = omega_cal(g2, u, R, b)
        m0 = min(len(w0), n)
        ideal_w = w0[:m0] - tgt[:m0]                       # ω 轴上的理想

        def ener(d):
            w = omega_cal(g2, u, R, b, d) if d is not None else w0
            m = min(len(w), m0)
            r = w[:m] - ideal_w[:m]
            r[:L] = 0.0                                    # s<L 无 ideal 对应
            return float((S1.bandpass41(r) ** 2).sum())

        e_base, e_pi, e_js = ener(None), ener(da), ener(da_js)
        # J 口径（训练时看到的那个）
        rj = tgt[:n] + np.stack([np.zeros(3)] * n)
        resp = np.zeros((n + M, 3))
        for k in range(M):
            mm = n - k
            resp[k:k + mm] += np.einsum("tcj,tj->tc", Jd[:mm, k].astype(np.float64), da[:mm])
        e_piJ = float((S1.bandpass41(tgt[:n] + resp[:n]) ** 2).sum())
        e_baseJ = float((S1.bandpass41(tgt[:n]) ** 2).sum())

        agg["base"] += e_base; agg["pi"] += e_pi; agg["js"] += e_js
        agg["pi_J"] += e_piJ
        rms_all.append(float(np.sqrt((da ** 2).mean()) * 1e3))
        print(f"{name:<18}{e_base:>9.4g}{e_pi:>10.4g}{(1-e_pi/e_base)*100:>7.1f}%"
              f"{e_js:>13.4g}{(1-e_js/e_base)*100:>7.1f}%"
              f"{(1-e_piJ/e_baseJ)*100:>11.1f}%{rms_all[-1]:>8.2f}")

    r_pi = 1 - agg["pi"] / agg["base"]
    r_js = 1 - agg["js"] / agg["base"]
    print(f"\n  ── 汇总（留出集，被控对象 = **真 G-2**）──")
    print(f"  **π（部署件）**          {r_pi*100:+.1f}%")
    print(f"  Jsolve（环境内参照）    {r_js*100:+.1f}%")
    print(f"  π 的 δa RMS             {np.mean(rms_all):.2f} mrad")
    print(f"\n  ⚠ 对照：H 解析解（当前 C 臂的做法）在**真 G-2** 上是 **−289%**，"
          f"\n     但在**真机**上是 **−8.0%** —— 这正是 G-2 与真机的增量差异。")
    OUT.write_text(json.dumps(
        {"tag": a.tag, "val_episodes": val_names,
         "true_g2": {"pi_reduction": r_pi, "jsolve_reduction": r_js},
         "pi_da_rms_mrad": float(np.mean(rms_all)),
         "caliber": ("留出集; 被控对象 = **真 G-2**(每集真跑前向); "
                     "基线 = δa≡0; 接触门控已施加; ε 由 tanh 结构保证; "
                     "**这仍是仿真内的数字, 不是真机**"),
         }, ensure_ascii=False, indent=1))
    print(f"\n  → {OUT}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    import os
    os._exit(rc)
