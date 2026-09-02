#!/usr/bin/env python3
"""**阶段 2：在选中的 j5 柔性结构上重拟 `r_s`，并直接评价完整模型**。

预注册 `PREREG_FLEX_J5.md` §4/§5。结构取自阶段 1（`flex_j5/grid.json`）。

构造（对预注册 §4 一句话的**落实**，见 `RESULT_FLEX_J5.md` §附录）
--------------------------------------------------------------------
预注册写「目标仍是 `IMU100_real − kin_flex(q200_real)`」，但柔性铰是一个
**仿真自由度**，`kin_flex(q200_real)` 并无定义（真机编码器给不出铰的角度）。
落实为：

    Δ(u)  = gyro_flex(u) − gyro_base(u)      两次 MuJoCo 前向，同 θ*、同命令
    ω̂     = kin(q̂) + Δ + r_s′               q̂ = G-q-v1(u) + r_q（**r_q 不变**）
    目标 Y′ = A_body − kin(q_true) − Δ

- Δ ≡ 0 时**逐比特退回 G-2** ⇒ 两者可直接比较。
- 关节层与 G-2 **完全相同**（同一 `base_q` + 同一 `r_q` 权重），
  柔性铰的全部作用都走 Δ 这一条 ⇒ 不会重复计入。
- Δ 以**单位增益**进入，不作为 `r_s′` 的特征、不拟合缩放 ⇒ 不新增自由度。
- 口径一致：Δ 的两端都是 MuJoCo 陀螺经 9 Hz 低通后取 20 Hz，
  与 `kin_gyro_fs` 的抗混叠约定相同。

用法:
    python -u sim/step4_gvnext/stage2_flex.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g", ROOT / "sim/step3_canobs", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data import load, bp, q_at, best_fit, valid_files, FPS            # noqa: E402
from train_rq import (features as rq_features, fir_design, stack_band,  # noqa: E402
                      fit, band_err, MOVING, BAND, W_BAND, KFOLD)
from train_rs import rs_features, sensor_target, band_err3, LAGS, LAMS  # noqa: E402
from eval_gate import base_q                                            # noqa: E402
from ceiling_200hz import kin_gyro_fs                                   # noqa: E402
from probe_flex_j5 import make_item, roll_both                          # noqa: E402

S1, S2 = ROOT / "outputs/step4_gvnext/s1", ROOT / "outputs/step4_gvnext/s2"
RQ, RS = ROOT / "outputs/step4_gvnext/rq", ROOT / "outputs/step4_gvnext/rs"
OUT = ROOT / "outputs/step4_gvnext/flex_j5"
JFLEX, WIN = 4, 80
GATE = {"r39": 0.560, "r13": 0.826, "n_imp": 6, "joint_rms": 0.05, "hard": 0.5}


def build(files, base_xml, xml, theta, zq, sq, oof=None):
    """→ 每集 dict：Δ、q̂、特征、两个目标、真机。"""
    E = []
    for f in files:
        dd = load(f)
        t20 = dd["tg_i"][::5]
        tr = np.asarray(dd["t_read"], float)
        it = make_item(dd)
        _, gb, _ = roll_both(base_xml, it, theta, None)
        _, gf, fl = roll_both(xml, it, theta, f"joint{JFLEX+1}_flex")
        G = lambda g: np.stack([np.interp(t20, tr, g[:, c]) for c in range(3)], 1)
        D = G(gf) - G(gb)                                   # Δ：柔性铰的增量贡献
        Xq, Yq, _ = rq_features(dd)
        qb = base_q(dd, t20)
        key = f"oof_{dd['name']}"
        Pq = oof[key] if (oof is not None and key in oof) else fir_design(Xq, sq["nlag"]) @ zq["W"]
        qh = qb + Pq
        Ab, Kt, Y_old = sensor_target(dd, t20)
        n = min(len(Y_old), len(D))
        E.append({"name": dd["name"], "D": D[:n], "qb": qb, "qh": qh,
                  "Xs": rs_features(dd, t20, qh)[:n], "Xq": Xq, "Yq": Yq, "Pq": Pq,
                  "Y_old": Y_old[:n], "Y_new": Y_old[:n] - D[:n], "A": Ab[:n],
                  "flex_deg": float(np.degrees(np.sqrt(np.mean(
                      bp(np.interp(t20, tr, fl)[:, None], *BAND)[:, 0] ** 2))))})
    return E


def main() -> int:
    import gc
    gc.disable()
    from g_q_v1 import THETA_STAR
    from sim.step2_gap_baseline import MODEL
    from structure_compare import flex_xml

    g = json.loads((OUT / "grid.json").read_text())
    sel = max([r for r in g["rows"] if r["enc_ok"]], key=lambda r: r["r39"])
    W_HZ, ZETA = sel["w"], sel["z"]
    xml = flex_xml(JFLEX, W_HZ, ZETA, THETA_STAR)
    print(f"阶段 1 选中：j5 下游有效柔性模态  ω={W_HZ:.3f} Hz  ζ={ZETA}"
          f"   （训练族选，开发验证集未参与）")
    print(f"  训练族底座 IMU 3–9 Hz ρ² {g['base']['r39']:.4f} → {sel['r39']:.4f}\n")

    zq, zs = np.load(RQ / "rq_fir.npz"), np.load(RS / "rs_fir.npz")
    sq, ss = (json.loads((p / "select.json").read_text()) for p in (RQ, RS))

    # ── 训练族：折外重拟 r_s′（q̂ 用 r_q 的折外预测，防泄漏）────────────
    TR = build(valid_files(S1), MODEL, xml, THETA_STAR, zq, sq, oof=zq)
    print(f"  训练族 {len(TR)} 条 · 特征维 {TR[0]['Xs'].shape[1]} · 折外 K={KFOLD}")
    print(f"  柔性铰相对运动 3–9 Hz RMS {np.mean([e['flex_deg'] for e in TR]):.4f}°")
    print(f"  Δ 的带内 RMS = {np.mean([np.sqrt(np.mean(bp(e['D'],*BAND)**2)) for e in TR]):.5f} rad/s")

    folds = [list(range(i, len(TR), KFOLD)) for i in range(KFOLD)]
    print(f"\n  {'nlag':>5}{'lam':>8}{'折外 3-9Hz 削减':>16}{'折外 0.5-3Hz 削减':>18}")
    best = None
    for nlag in LAGS:
        D_ = [fir_design(e["Xs"], nlag) for e in TR]
        for lam in LAMS:
            n39 = d39 = n13 = d13 = 0.0
            for te in folds:
                tr = [i for i in range(len(TR)) if i not in te]
                W = fit([stack_band(D_[i], TR[i]["Y_new"], W_BAND)[0] for i in tr],
                        [stack_band(D_[i], TR[i]["Y_new"], W_BAND)[1] for i in tr], lam)
                for i in te:
                    P = D_[i] @ W
                    a, b = band_err3(TR[i]["Y_new"], P, *BAND); n39 += a; d39 += b
                    a, b = band_err3(TR[i]["Y_new"], P, 0.5, 3.0); n13 += a; d13 += b
            r39, r13 = 1 - n39 / max(d39, 1e-30), 1 - n13 / max(d13, 1e-30)
            print(f"  {nlag:>5}{lam:>8.0e}{r39*100:>15.1f}%{r13*100:>17.1f}%")
            if best is None or r39 > best[0]:
                best = (r39, r13, nlag, lam)
    o39, o13, nlag, lam = best
    print(f"\n  **重拟 r_s′：nlag={nlag}, lam={lam:.0e}** —— 折外 3–9 Hz 削减 "
          f"{o39*100:.1f}%（G-2 的 r_s 是 {ss['oof_reduction_3_9']*100:.1f}%）")
    D_ = [fir_design(e["Xs"], nlag) for e in TR]
    Wf = fit([stack_band(D_[i], TR[i]["Y_new"], W_BAND)[0] for i in range(len(TR))],
             [stack_band(D_[i], TR[i]["Y_new"], W_BAND)[1] for i in range(len(TR))], lam)

    # ── 对照：相位模板 / 零输入，都建在**柔性底座**上（同结构，只比残差学习）──
    Lmin = min(min(len(e["Yq"]), len(e["Y_new"])) for e in TR)
    tpl_q = np.mean([e["Yq"][:Lmin] for e in TR], 0)
    tpl_s = np.mean([e["Y_new"][:Lmin] for e in TR], 0)
    zin_q = fit([stack_band(fir_design(np.zeros_like(e["Xq"]), sq["nlag"]), e["Yq"], W_BAND)[0] for e in TR],
                [stack_band(fir_design(np.zeros_like(e["Xq"]), sq["nlag"]), e["Yq"], W_BAND)[1] for e in TR], sq["lam"])
    zin_s = fit([stack_band(fir_design(np.zeros_like(e["Xs"]), nlag), e["Y_new"], W_BAND)[0] for e in TR],
                [stack_band(fir_design(np.zeros_like(e["Xs"]), nlag), e["Y_new"], W_BAND)[1] for e in TR], lam)

    # ── 开发验证集（8 条，未参与选格与拟合）────────────────────────────
    DV = build(valid_files(S2, "replay_pp2f*.json"), MODEL, xml, THETA_STAR, zq, sq)
    print(f"\n  开发验证集 {len(DV)} 条 —— **不是密封 holdout**（探针已看过）\n")
    ARMS = ("G-q-v1", "相位模板(柔)", "零输入(柔)", "G-2", "G-2F")
    BANDS = ((0.5, 3.0), (3.0, 9.0))
    r2 = {a: {b: [] for b in BANDS} for a in ARMS}
    win = {a: [0, 0] for a in ARMS}
    win2 = [0, 0]
    for e in DV:
        n = len(e["Y_new"])
        L = min(Lmin, n)
        kb, kh = kin_gyro_fs(e["qb"], FPS)[:n], kin_gyro_fs(e["qh"], FPS)[:n]
        pred = {
            "G-q-v1": kb,
            "相位模板(柔)": kin_gyro_fs(e["qb"][:L] + tpl_q[:L], FPS) + e["D"][:L] + tpl_s[:L],
            "零输入(柔)": (kin_gyro_fs(e["qb"] + fir_design(np.zeros_like(e["Xq"]), sq["nlag"]) @ zin_q, FPS)[:n]
                          + e["D"] + fir_design(np.zeros_like(e["Xs"]), nlag) @ zin_s),
            "G-2": kh + fir_design(e["Xs"], ss["nlag"]) @ zs["W"],
            "G-2F": kh + e["D"] + fir_design(e["Xs"], nlag) @ Wf,
        }
        A9 = bp(e["A"], *BAND)
        base9 = bp(kb, *BAND)
        g2_9 = bp(pred["G-2"], *BAND)
        for a, S in pred.items():
            m = min(len(e["A"]), len(S))
            for b_ in BANDS:
                r2[a][b_].append(best_fit(e["A"][:m], S[:m], *b_, rotate=False)[1])
            S9 = bp(S[:m], *BAND)
            ne = m // WIN
            en = [float((A9[k*WIN:(k+1)*WIN] ** 2).sum()) for k in range(ne)]
            for k in np.argsort(en)[-max(1, ne // 5):]:
                sl = slice(k * WIN, (k + 1) * WIN)
                win[a][0] += int(float(((A9[sl]-S9[sl])**2).sum())
                                 < float(((A9[sl]-base9[sl])**2).sum()))
                win[a][1] += 1
                if a == "G-2F":
                    win2[0] += int(float(((A9[sl]-S9[sl])**2).sum())
                                   < float(((A9[sl]-g2_9[sl])**2).sum()))
                    win2[1] += 1

    print(f"  {'':<14}{'0.5–3 Hz ρ²':>14}{'3–9 Hz ρ²':>12}{'最难20%窗 vs G-q-v1':>22}")
    for a in ARMS:
        print(f"  {a:<14}" + "".join(f"{np.median(r2[a][b]):>14.3f}" for b in BANDS)
              + f"{win[a][0]}/{win[a][1]:>18}")
    n_imp = sum(int(x > y) for x, y in zip(r2["G-2F"][BANDS[1]], r2["G-2"][BANDS[1]]))
    v39, v13 = (float(np.median(r2["G-2F"][b])) for b in (BANDS[1], BANDS[0]))
    g2_39 = float(np.median(r2["G-2"][BANDS[1]]))
    print(f"\n  逐集 3–9 Hz（G-2 → G-2F）:")
    for e, x, y in zip(DV, r2["G-2"][BANDS[1]], r2["G-2F"][BANDS[1]]):
        print(f"    {e['name']:<16}{x:.3f} → {y:.3f}   {'✓' if y > x else '×'}")

    # ── 门禁（PREREG_FLEX_J5.md §5）────────────────────────────────────
    ctl = all(np.median(r2["G-2F"][b]) > np.median(r2[a][b])
              for b in BANDS for a in ("G-q-v1", "相位模板(柔)", "零输入(柔)"))
    G = {
        "1 · IMU 3–9 Hz ρ² 中位": (f"{v39:.3f}（G-2 {g2_39:.3f}）", f"≥{GATE['r39']}", v39 >= GATE["r39"]),
        "2 · 逐集改善": (f"{n_imp}/8", f"≥{GATE['n_imp']}/8", n_imp >= GATE["n_imp"]),
        "3 · IMU 0.5–3 Hz ρ² 中位": (f"{v13:.3f}", f"≥{GATE['r13']}", v13 >= GATE["r13"]),
        "4 · 关节层宽带 RMS": ("与 G-2 完全相同（r_q 冻结、关节层未改）", "≤+5%", True),
        "5 · 两带均优于三对照": ("全部严格优" if ctl else "**有未过**", "全部严格优", ctl),
        "6 · 最难 20% 窗改善": (f"{win['G-2F'][0]}/{win['G-2F'][1]} = "
                              f"{win['G-2F'][0]/max(win['G-2F'][1],1)*100:.0f}%",
                              ">50%", win["G-2F"][0] > win["G-2F"][1] * 0.5),
    }
    print(f"\n  {'判据':<26}{'实测':<40}{'门槛':<12}判")
    for k in sorted(G):
        s, t, ok = G[k]
        print(f"  {k:<26}{s:<40}{t:<12}{'✓' if ok else '**✗**'}")
    allok = all(v[2] for v in G.values())
    print(f"\n  总判: {'**通过**' if allok else '**未通过 —— 本条线终止**'}")
    print(f"  （附：最难 20% 窗 **vs G-2** {win2[0]}/{win2[1]} = "
          f"{win2[0]/max(win2[1],1)*100:.0f}%，非门禁项）")

    np.savez(OUT / "rs_flex.npz", W=Wf, nlag=nlag, lam=lam, w_hz=W_HZ, zeta=ZETA)
    (OUT / "stage2.json").write_text(json.dumps(
        {"pass": bool(allok), "w_hz": W_HZ, "zeta": ZETA,
         "rs_nlag": int(nlag), "rs_lam": float(lam), "rs_oof_3_9": float(o39),
         "criteria": {k: {"got": v[0], "thr": v[1], "ok": bool(v[2])} for k, v in G.items()},
         "rho2_median": {a: {f"{b[0]}-{b[1]}": float(np.median(r2[a][b])) for b in BANDS}
                         for a in ARMS},
         "per_episode_3_9": {e["name"]: {"G-2": float(x), "G-2F": float(y)}
                             for e, x, y in zip(DV, r2["G-2"][BANDS[1]], r2["G-2F"][BANDS[1]])},
         "hard_win_vs_g2": win2,
         "caliber": ("8 条为**开发验证集**，非密封 holdout（探针已看过）；"
                     "结构由训练族选；r_q 冻结；Δ 单位增益；"
                     "ρ² 不可线性相加，本表全部来自完整模型直接评价"),
         }, indent=1, ensure_ascii=False))
    print(f"  → {OUT/'stage2.json'}")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
