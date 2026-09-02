#!/usr/bin/env python3
"""**阶段 1：j5 单铰网格** —— 预注册 `PREREG_FLEX_J5.md` §3。

固定预算 **35 格**（ω 7 × ζ 5），**只搜 j5**，结束后不追加、不外延。
选格**只用训练族**（会话1 有效 17 条）；开发验证集不参与。

选格口径（本脚本对预注册的明确解释）
------------------------------------
目标 = **底座层**的腕部 IMU 3–9 Hz ρ²（柔性模型自己的陀螺，**不叠 `r_s`**）。
理由：`r_s` 是对着 `kin(q̂)` 拟合的，柔性底座的 IMU 来自它**自己的陀螺**，
直接叠旧 `r_s` 会把柔性铰已经产生的那部分**重复计入**。
阶段 2 会在选定结构上**重拟** `r_s`，那时才评价完整模型。
（与单点探针同一约定，可直接比较。）

约束：编码器宽带损失 `L_broad ≤ 1.05 × 底座`，防止靠拖垮关节层来刷 IMU。

用法:
    python -u sim/step4_gvnext/grid_flex_j5.py --time-one   # 先测单格耗时
    python -u sim/step4_gvnext/grid_flex_j5.py              # 跑完整 35 格
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g", ROOT / "sim/step3_canobs", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data import load, bp, q_at, best_fit, valid_files, FPS      # noqa: E402
from probe_flex_j5 import make_item, roll_both                    # noqa: E402
from train_rq import MOVING                                       # noqa: E402

JFLEX = 4                                    # j5（0-based）
W_GRID = np.round(np.exp(np.linspace(np.log(3.0), np.log(16.0), 7)), 3)
Z_GRID = (0.1, 0.2, 0.4, 0.7, 1.2)
ENC_TOL = 1.05
OUT = ROOT / "outputs/step4_gvnext/flex_j5"
S1 = ROOT / "outputs/step4_gvnext/s1"


def prep():
    """训练族的 item + 真值，只算一次。"""
    E = []
    for f in valid_files(S1):
        dd = load(f)
        t20 = dd["tg_i"][::5]
        E.append({"name": dd["name"], "it": make_item(dd), "t20": t20,
                  "tr": np.asarray(dd["t_read"], float), "dd": dd,
                  "qt": q_at(dd, t20)})
    from train_rs import sensor_target
    for e in E:
        e["A"], _, _ = sensor_target(e["dd"], e["t20"])
        e["travel"] = e["qt"].max(0) - e["qt"].min(0)
        e.pop("dd")
    return E


def score(E, xml, theta, flex_joint):
    """→ (IMU 3–9 Hz ρ² 中位, IMU 0.5–3 Hz ρ² 中位, 编码器 L_broad, 铰 RMS)。"""
    r39, r13, per, fl = [], [], [], []
    for e in E:
        q, g, f_ = roll_both(xml, e["it"], theta, flex_joint)
        S = np.stack([np.interp(e["t20"], e["tr"], g[:, c]) for c in range(3)], 1)
        m = min(len(e["A"]), len(S))
        r39.append(best_fit(e["A"][:m], S[:m], 3.0, 9.0)[1])
        r13.append(best_fit(e["A"][:m], S[:m], 0.5, 3.0)[1])
        qi = np.stack([np.interp(e["t20"], e["tr"], q[:, j]) for j in range(6)], 1)
        err = qi - e["qt"]
        per.append(float(np.mean([np.sqrt(np.mean(err[:, j] ** 2))
                                  / max(e["travel"][j], 1e-9) for j in MOVING])))
        if f_ is not None:
            fl.append(float(np.sqrt(np.mean(
                bp(np.interp(e["t20"], e["tr"], f_)[:, None], 3.0, 9.0)[:, 0] ** 2))))
    return (float(np.median(r39)), float(np.median(r13)),
            float(np.mean(per)), float(np.mean(fl)) if fl else 0.0)


def main() -> int:
    import gc
    gc.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-one", action="store_true")
    a = ap.parse_args()
    from g_q_v1 import THETA_STAR
    from sim.step2_gap_baseline import MODEL
    from structure_compare import flex_xml

    E = prep()
    print(f"训练族 {len(E)} 条 · 网格 {len(W_GRID)}×{len(Z_GRID)} = "
          f"{len(W_GRID)*len(Z_GRID)} 格（固定预算，不追加不外延）")
    print(f"  ω 网格 {list(W_GRID)} Hz")
    print(f"  ζ 网格 {list(Z_GRID)}\n")

    t0 = time.time()
    b39, b13, bL, _ = score(E, MODEL, THETA_STAR, None)
    dt1 = time.time() - t0
    print(f"底座（无铰）: IMU 3–9 Hz ρ² {b39:.4f}   0.5–3 Hz {b13:.4f}   "
          f"编码器 L_broad {bL:.5f}   [{dt1:.0f}s]")
    print(f"  编码器约束线 = {bL*ENC_TOL:.5f}\n")
    if a.time_one:
        t0 = time.time()
        xml = flex_xml(JFLEX, float(W_GRID[3]), Z_GRID[2], THETA_STAR)
        r = score(E, xml, THETA_STAR, f"joint{JFLEX+1}_flex")
        print(f"单格耗时 {time.time()-t0:.0f}s ⇒ 35 格约 "
              f"{(time.time()-t0)*35/60:.0f} 分钟")
        print(f"  样例格 ω={W_GRID[3]} ζ={Z_GRID[2]}: ρ²(3–9) {r[0]:.4f}  "
              f"L_broad {r[2]:.5f}  铰 RMS {np.degrees(r[3]):.4f}°")
        return 0

    rows = []
    for w in W_GRID:
        for z in Z_GRID:
            xml = flex_xml(JFLEX, float(w), float(z), THETA_STAR)
            r39, r13, L, fl = score(E, xml, THETA_STAR, f"joint{JFLEX+1}_flex")
            ok = L <= bL * ENC_TOL
            rows.append({"w": float(w), "z": float(z), "r39": r39, "r13": r13,
                         "L": L, "flex_deg": float(np.degrees(fl)), "enc_ok": bool(ok)})
            print(f"  ω={w:>6.3f} ζ={z:<4} ρ²(3–9) {r39:.4f}  ρ²(0.5–3) {r13:.4f}  "
                  f"L {L:.5f} {'✓' if ok else '**✗超编码器约束**'}  "
                  f"铰 {np.degrees(fl):.4f}°", flush=True)
    OK = [r for r in rows if r["enc_ok"]]
    print(f"\n  满足编码器约束的格: {len(OK)}/{len(rows)}")
    if not OK:
        print("  **无可用格 —— 本条线终止**")
    else:
        best = max(OK, key=lambda r: r["r39"])
        edge = (best["w"] in (W_GRID[0], W_GRID[-1])) or (best["z"] in (Z_GRID[0], Z_GRID[-1]))
        print(f"  **选中 ω={best['w']:.3f} Hz  ζ={best['z']}**   "
              f"ρ²(3–9) {best['r39']:.4f}（底座 {b39:.4f}，"
              f"{(best['r39']-b39)/b39*100:+.1f}%）")
        print(f"    ρ²(0.5–3) {best['r13']:.4f}（底座 {b13:.4f}）  "
              f"L_broad {best['L']:.5f}  铰 RMS {best['flex_deg']:.4f}°")
        if edge:
            print("    ⚠ **选中格落在网格边界** —— 按预注册如实报告为局限，**不外延**")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "grid.json").write_text(json.dumps(
        {"base": {"r39": b39, "r13": b13, "L": bL}, "enc_tol": ENC_TOL,
         "w_grid": [float(x) for x in W_GRID], "z_grid": list(Z_GRID),
         "rows": rows,
         "select_rule": ("训练族底座层 IMU 3–9 Hz ρ² 最大，约束 L_broad ≤ 1.05×底座；"
                         "不叠 r_s（避免与柔性铰重复计入），阶段 2 再重拟"),
         "caliber": "训练族=会话1 有效 17 条；开发验证集未参与",
         }, indent=1, ensure_ascii=False))
    print(f"\n  → {OUT/'grid.json'}")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
