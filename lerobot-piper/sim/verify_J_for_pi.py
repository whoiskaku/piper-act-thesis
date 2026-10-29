#!/usr/bin/env python3
"""**J 适用性检验**：能不能用 `J = ∂ω_G2/∂δa` 做 π 训练的可微环境？（零机时）

为什么要查
----------
用户设计的训练结构是「π 在冻结的 G-2 环境里学」：

    loss = ‖ ω_G2(u+δa) − ω_ideal ‖² + λ_a‖δa‖² + λ_s‖Δδa‖²

但 G-2 的前向含 MuJoCo 伺服 rollout 与逐帧运动学，**不可微**。
可行的做法是用 G-2 **自己的雅可比** `J` 把损失线性化：

    ω_G2(u+δa) − ω_G2(u)  ≈  J ⊛ δa

⚠ **仓里有前科**：J 的线性化曾按事前门禁被判否（未参与抽取的**单发脉冲** 7/10 失败）。
但那测的是单发脉冲这种极端输入，与 π 实际产生的**连续小幅 δa** 分布不同。
本脚本换成 π 的真实工作分布重测。

口径（事前写死）
----------------
- `u` = `ab5 C` 臂日志里的 `chunks[].raw`（原始 ACT 指令）
- `δa` = **同一集里 g2wall 真正施加过的 δa**（RMS ~0.54 mrad）⇒ 正是 π 的工作分布
- 真值 = **真跑两遍 G-2**：`G2(u+δa) − G2(u)`（**单边**，与部署一致）
- 另报 `odd` = `½[G2(u+δa) − G2(u−δa)]`（J 建模的对象）⇒ 两者之差 = 二阶效应
- J 用 `g2_on_ab4.extract/dense`（AMP=1 mrad 脉冲，步长 M+1=16，沿 t 线性插值）
- 频带 0.5–3 / 3–9 / 0.5–9 Hz

**事前门禁**（这是「J 是不是 G-2 自身的好线性化」的自洽性检验，故标准从严）
--------------------------------------------------------------------------
    cos ≥ 0.90   ·   范数比 ∈ [0.80, 1.25]   ·   相对误差 ≤ 0.30

不过 ⇒ 不能用 J 做训练环境，须回头改用可微重写或别的方案。

用法:
    python -u sim/verify_J_for_pi.py [--n 5]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "sim/step4_gvnext"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import g2_on_ab4 as G                                                # noqa: E402
import gain_G_vs_H as GH                                             # noqa: E402
from freeze_pi_g2wall import G2v0                                    # noqa: E402

BANDS = ((0.5, 3.0), (3.0, 9.0), (0.5, 9.0))
GATE = {"cos": 0.90, "ratio": (0.80, 1.25), "relerr": 0.30}
OUT = ROOT / "outputs/p2_pi/verify_J_for_pi.json"


def conv_J(Jd: np.ndarray, da: np.ndarray) -> np.ndarray:
    """`Jd[t, k, c, j]` = t 时刻单位脉冲在 t+k 的第 c 轴响应 → 卷积得总响应。"""
    T, K = Jd.shape[0], Jd.shape[1]
    out = np.zeros((T, 3))
    for k in range(K):
        contrib = np.einsum("tcj,tj->tc", Jd[:T - k, k], da[:T - k])
        out[k:k + len(contrib)] += contrib
    return out


def main() -> int:
    import gc
    gc.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()

    g2 = G2v0()
    logs = sorted((ROOT / "outputs/ab5").glob("C_ep*.json"),
                  key=lambda p: int(p.stem.split("ep")[1]))[:a.n]
    print(f"J 适用性检验 · {len(logs)} 集 · δa 取**真机实际施加**的那一条（π 的工作分布）")
    print(f"事前门禁: cos ≥ {GATE['cos']} · 范数比 ∈ {GATE['ratio']} · "
          f"相对误差 ≤ {GATE['relerr']}\n")

    acc = {b: [0.0, 0.0, 0.0, 0.0] for b in BANDS}     # Σ<t,p> Σ‖t‖² Σ‖p‖² Σ‖t−p‖²
    acc_odd = {b: [0.0, 0.0] for b in BANDS}
    for p in logs:
        d = json.loads(p.read_text())
        chunks = sorted(d["chunks"], key=lambda c: int(c["start"]))
        u = np.concatenate([np.asarray(c["raw"], float) for c in chunks])
        da_log = np.asarray([f["da"] for f in d["frames"]], float)
        T = min(len(u), len(da_log))
        u, da = u[:T], np.zeros((T, 6))
        da[:] = da_log[:T]
        dd = {"name": p.stem, "cmd": u, "t_send": np.arange(T) / 20.0,
              "t_read": np.arange(T) / 20.0,
              "tg_i": np.arange(T * 5) / 100.0, "gyro100": np.zeros((T * 5, 3))}
        t0 = time.perf_counter()
        w0 = g2.omega(dd)
        up = np.array(u, copy=True); up[:, :6] += da
        um = np.array(u, copy=True); um[:, :6] -= da
        wp = g2.omega({**dd, "cmd": up})
        wm = g2.omega({**dd, "cmd": um})
        m = min(len(w0), len(wp), len(wm), T)
        true1 = wp[:m] - w0[:m]                     # 单边（部署时真实发生的）
        odd = 0.5 * (wp[:m] - wm[:m])               # 奇分量（J 建模的对象）
        t0s, J = G.extract(g2, dd, m)
        Jd = G.dense(t0s, J, m)
        pred = conv_J(Jd, da[:m])
        ms = (time.perf_counter() - t0) * 1e3
        for b in BANDS:
            x, y, o = GH.bp(true1, *b), GH.bp(pred, *b), GH.bp(odd, *b)
            acc[b][0] += float((x * y).sum()); acc[b][1] += float((x ** 2).sum())
            acc[b][2] += float((y ** 2).sum()); acc[b][3] += float(((x - y) ** 2).sum())
            acc_odd[b][0] += float(((x - o) ** 2).sum()); acc_odd[b][1] += float((x ** 2).sum())
        print(f"  {p.stem:<10} T={m}  δa RMS {np.sqrt((da[:m]**2).mean())*1e3:.2f} mrad  "
              f"({ms/1000:.1f}s)")

    print(f"\n  {'频带':<10}{'cos':>8}{'范数比':>9}{'相对误差':>10}{'判决':>8}")
    res, ok_all = {}, True
    for b in BANDS:
        dot, nt, np_, err = acc[b]
        cos = dot / max(np.sqrt(nt * np_), 1e-30)
        ratio = np.sqrt(np_ / max(nt, 1e-30))
        rel = np.sqrt(err / max(nt, 1e-30))
        ok = (cos >= GATE["cos"] and GATE["ratio"][0] <= ratio <= GATE["ratio"][1]
              and rel <= GATE["relerr"])
        ok_all &= ok
        res[f"{b[0]}-{b[1]}"] = {"cos": float(cos), "norm_ratio": float(ratio), "rel_err": float(rel), "pass": bool(ok)}
        print(f"  {b[0]}-{b[1]:<6}{cos:>8.3f}{ratio:>9.3f}{rel:>10.3f}{'✓' if ok else '**✗**':>8}")

    print(f"\n  二阶效应（单边 vs 奇分量的相对差）:")
    for b in BANDS:
        e, n = acc_odd[b]
        print(f"    {b[0]}-{b[1]} Hz: {np.sqrt(e/max(n,1e-30)):.3f}")

    print(f"\n  ── {'**J 可用作 π 的训练环境**' if ok_all else '**J 不可用 ⇒ 须改方案**'} ──")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"bands": res, "all_pass": bool(ok_all), "gate": GATE,
         "n_episodes": len(logs),
         "second_order": {f"{b[0]}-{b[1]}": float(np.sqrt(acc_odd[b][0] /
                          max(acc_odd[b][1], 1e-30))) for b in BANDS},
         "caliber": ("u=ab5 C 的 chunks[].raw; δa=**同集真机实际施加**的那一条(π 工作分布); "
                     "真值=真跑两遍 G-2 的**单边**差(与部署一致); "
                     "J 用 g2_on_ab4.extract/dense(AMP=1mrad, 步长16, 沿 t 线性插值); "
                     "**这是 J 对 G-2 自身的自洽性检验, 不是 J 对真机**"),
         }, ensure_ascii=False, indent=1))
    print(f"  → {OUT}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    import os
    os._exit(rc)
