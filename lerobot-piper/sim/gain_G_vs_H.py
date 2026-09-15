#!/usr/bin/env python3
"""**量化 G-2 的增量增益相对实测 H**（零机时）—— 在任何新 π 训练之前完成。

为什么必须先做
--------------
方向性实验(a)里，SPSA 停在 `δa≈0.01 mrad`（工作区是 ~1 mrad），
地形探测显示 G 的目标函数在 `δa≈0` 取最小。由此**反推**出「G 的增量增益可能偏大约 7 倍」
—— 但那是**由损失反推**的，**不能当作增益结论**。本脚本直接量它。

口径（用户 2026-09-15 定，全部写死）
------------------------------------
- 同一组、**同一时刻**的 δa，取自**这次 18 参数策略类**的固定扰动方向，
  缩放到 **0.5 / 1 / 2 mrad RMS**。
- `Δ_G = G2(u+δa) − G2(u)`（**单边差分**，非 ±配对）；`Δ_H = H * δa`。
- **固定**独立的 `R` 与 `ALIGN_L=3`：`Δgap_G[t] = Δω_G[t+3] @ R.T`。
  **不重新扫描位移、旋转或增益。**
- 频带 **0.5–3 / 3–9 / 0.5–9 Hz** 分别报。
- 四个量：范数比 `‖Δ_G‖/‖Δ_H‖` · 方向 `cos` ·
  **有符号投影增益** `⟨Δ_G,Δ_H⟩/‖Δ_H‖²` · 相对误差。
- **除汇总外逐关节报告**（δa 屏蔽到单关节，时刻与缩放同）。

判读（用户写死）
----------------
**只有范数比大且 cos 也高，才能称为「系统性增益偏大」。**
若 **cos 低**，或**跨关节差异很大**，就**不是**一个标量增益问题。

⚠ 即便确认是稳定的 ~7 倍失配，**也不能直接除以 7 继续** ——
该修正来自 **H**，会改变「**仅用 G 训练**」的口径。届时须把它作为
**G 的增量标定步骤**预注册，并用**独立数据**验证。
若不是稳定标量失配，则应回到 **G 的增量结构或时序**问题。

⚠ ab4 无时间戳 ⇒ G 前向在**假设均匀 20 Hz** 下跑。

用法:
    python -u sim/gain_G_vs_H.py
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
import dir_transfer_g2 as D                                         # noqa: E402
import g2_on_ab4 as G                                               # noqa: E402

AMPS = (0.0005, 0.001, 0.002)             # 0.5 / 1 / 2 mrad **RMS**
BANDS = ((0.5, 3.0), (3.0, 9.0), (0.5, 9.0))
DIR_SEED = 7                              # 固定扰动方向（18 参数策略类）
OUT = ROOT / "outputs/step1_H/gain_G_vs_H.json"


def bp(x, lo, hi):
    from scipy.signal import firwin
    k = firwin(41, [lo, hi], fs=20.0, pass_zero=False)
    return np.stack([np.convolve(x[:, c], k, mode="same") for c in range(3)], 1)


def policy_da(e, Th, mu, sd):
    """18 参数策略类产生的 δa（未缩放）。"""
    n = e["n"]
    rt = (e["e0"] - mu) / sd
    v = np.zeros((n, 6))
    v[D.ALIGN_L:] = D.S_SCALE * (rt[:n - D.ALIGN_L] @ Th.T)
    v[e["contact"]] = 0.0
    return v


def metrics(a, b):
    """a=Δ_H（参照）, b=Δ_G。→ (范数比, cos, 有符号投影增益, 相对误差)。"""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    dot = float((a * b).sum())
    return (nb / max(na, 1e-30),
            dot / max(na * nb, 1e-30),
            dot / max(float((a * a).sum()), 1e-30),
            float(np.linalg.norm(b - a) / max(na, 1e-30)))


def main() -> int:
    import gc
    gc.disable()
    E, R, _ = D.load_P0()
    g2 = G.G2()
    H = np.ascontiguousarray(np.load(ROOT / "outputs/step1_H/step1_H_v1.npz")["H"], float)
    mu = np.concatenate([E[i]["e0"] for i in D.TRAIN]).mean(0)
    sd = np.concatenate([E[i]["e0"] for i in D.TRAIN]).std(0) + 1e-12
    Th = np.random.default_rng(DIR_SEED).choice([-1.0, 1.0], size=(6, 3))
    print(f"P0 {len(E)} 集 · 固定扰动方向 seed={DIR_SEED} · 单边差分 · "
          f"R 与 ALIGN_L={D.ALIGN_L} 固定不重扫")
    print(f"⚠ ab4 无时间戳 ⇒ G 前向假设均匀 20 Hz\n")

    base = {e["name"]: g2.omega(e["dd"]) for e in E}     # G2(u)，缓存
    res = {}

    def run(mask, tag):
        rows = {}
        for amp in AMPS:
            acc = {b: [np.zeros((0, 3)), np.zeros((0, 3))] for b in BANDS}
            for e in E:
                v = policy_da(e, Th, mu, sd)
                if mask is not None:
                    m = np.zeros(6); m[mask] = 1.0
                    v = v * m
                cur = float(np.sqrt(np.mean(v ** 2)))
                if cur < 1e-15:
                    continue
                v = v * (amp / cur)
                v = np.clip(v, -D.EPS, D.EPS)
                v[e["contact"]] = 0.0
                c = np.array(e["dd"]["cmd"], float, copy=True)
                k = min(len(c), len(v)); c[:k, :6] += v[:k]
                dw = g2.omega({**e["dd"], "cmd": c})
                n = e["n"]
                m0 = min(len(dw), len(base[e["name"]]))
                d_g = D.to_gap(dw[:m0] - base[e["name"]][:m0], R, n)
                d_h = S1.causal_conv(H, v[:n])
                for b in BANDS:
                    acc[b][0] = np.vstack([acc[b][0], bp(d_h, *b)])
                    acc[b][1] = np.vstack([acc[b][1], bp(d_g, *b)])
            rows[amp] = {f"{b[0]}-{b[1]}": metrics(*acc[b]) for b in BANDS}
        return rows

    print(f"  {'':<10}{'带':<10}{'范数比':>9}{'cos':>8}{'投影增益':>10}{'相对误差':>10}")
    for tag, mask in [("**汇总**", None)] + [(f"j{j+1}", j) for j in range(6)]:
        rows = run(mask, tag)
        res[tag.strip("*")] = {f"{a*1e3:.1f}mrad": {k: list(v) for k, v in r.items()}
                               for a, r in rows.items()}
        for a in AMPS:
            for b in BANDS:
                key = f"{b[0]}-{b[1]}"
                nr, cs, pg, re_ = rows[a][key]
                lbl = f"{tag} {a*1e3:.1f}m" if b == BANDS[0] else ""
                print(f"  {lbl:<10}{key:<10}{nr:>9.2f}{cs:>8.3f}{pg:>10.2f}{re_:>10.2f}")
        print()

    ag = res["汇总"][f"{AMPS[1]*1e3:.1f}mrad"]["0.5-9.0"]
    nr, cs = ag[0], ag[1]
    print("  ── 判读（用户 2026-09-15 写死）──")
    print(f"  汇总 @1 mrad · 0.5–9 Hz：范数比 {nr:.2f} · cos {cs:.3f}")
    cj = [res[f"j{j+1}"][f"{AMPS[1]*1e3:.1f}mrad"]["0.5-9.0"][0] for j in range(6)]
    spread = max(cj) / max(min(cj), 1e-12)
    print(f"  逐关节范数比 {np.round(cj,2).tolist()} · 跨关节离散 {spread:.1f}×")
    if nr > 2 and cs > 0.6 and spread < 3:
        print(f"  ⇒ **范数比大且 cos 高、跨关节一致 ⇒ 可称「系统性增益偏大」**")
        print(f"     ⚠ 但**不得直接除以它继续** —— 该修正来自 H，会改变「仅用 G 训练」的口径。"
              f"\n       须作为 **G 的增量标定步骤**预注册，并用**独立数据**验证。")
    else:
        print(f"  ⇒ **不是一个标量增益问题**"
              f"（cos {'低' if cs <= 0.6 else '尚可'}；跨关节离散 {spread:.1f}×）"
              f"\n     ⇒ 应回到 **G 的增量结构或时序**问题，不是除以一个标量能解决的。")
    OUT.write_text(json.dumps(
        {"result": res, "amps_rad_rms": list(AMPS), "bands": [list(b) for b in BANDS],
         "dir_seed": DIR_SEED, "align_L": D.ALIGN_L,
         "metrics_order": ["norm_ratio", "cos", "signed_projection_gain", "rel_err"],
         "caliber": ("单边差分 Δ_G=G2(u+δa)−G2(u); Δ_H=H*δa; "
                     "R 与 ALIGN_L=3 **固定不重扫**; δa 取自 18 参数策略类的固定方向, "
                     "缩放到 0.5/1/2 mrad RMS, 同一时刻; "
                     "**只有范数比大且 cos 也高才能称系统性增益偏大**; "
                     "ab4 无时间戳⇒假设均匀 20Hz"),
         }, indent=1, ensure_ascii=False))
    print(f"\n  → {OUT}")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
