#!/usr/bin/env python3
"""**G-vNext 的完整门禁**（`PREREG_G_VNEXT.md` §6）—— 未见轨迹 + 跨会话留出。

    q̂ = G-q-v1(u) + r_q(·)        ω̂ = kin(q̂) + r_s(·)

留出 = 会话2 的 pp2f + pp2f60（轨迹未参与训练、会话也未参与训练）。
两层的 nlag/lam 均由**训练族折外**选定，留出集未参与任何选择。

三个对照（`PREREG_G_VNEXT.md` §6）
  · **G-q-v1**    r ≡ 0
  · **相位模板**   训练族上按**帧序**平均的残差 —— 记忆基线
  · **零输入对照** 同一 FIR 结构但输入置零 ⇒ 只能输出常数

用法:
    python -u sim/step4_gvnext/eval_gate.py
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

from data import load, to20, bp, q_at, best_fit, valid_files, FPS   # noqa: E402
from train_rq import (features as rq_features, fir_design, stack_band,  # noqa: E402
                      fit, band_err, MOVING, BAND, W_BAND)
from train_rs import rs_features, sensor_target, band_err3          # noqa: E402
from ceiling_200hz import kin_gyro_fs                               # noqa: E402

S1 = ROOT / "outputs/step4_gvnext/s1"
S2 = ROOT / "outputs/step4_gvnext/s2"
RQ, RS = ROOT / "outputs/step4_gvnext/rq", ROOT / "outputs/step4_gvnext/rs"
OUT = ROOT / "outputs/step4_gvnext"
FRZ = {(0.5, 3.0): 0.789, (3.0, 9.0): 0.707}
WIN = 80                                   # 4 s @ 20 Hz


def base_q(dd, t20):
    import g_q_v1 as GQ
    qf = np.radians(GQ.predict_qpos(dd["cmd"], t_send=dd["t_send"],
                                    t_read=dd["t_read"]))
    return np.stack([np.interp(t20, dd["t_read"], qf[:, j]) for j in range(6)], 1)


def main() -> int:
    import gc
    gc.disable()
    zq, zs = np.load(RQ / "rq_fir.npz"), np.load(RS / "rs_fir.npz")
    sq = json.loads((RQ / "select.json").read_text())
    ss = json.loads((RS / "select.json").read_text())
    print(f"r_q: FIR nlag={sq['nlag']} lam={sq['lam']:.0e}   "
          f"r_s: FIR nlag={ss['nlag']} lam={ss['lam']:.0e}\n")

    # ── 相位模板 & 零输入对照：只用训练族拟合 ─────────────────────────
    TR = []
    for f in valid_files(S1):
        dd = load(f)
        t20 = dd["tg_i"][::5]
        Xq, Yq, _ = rq_features(dd)
        qb = base_q(dd, t20)
        _, _, Ys = sensor_target(dd, t20)
        TR.append({"Xq": Xq, "Yq": Yq, "Ys": Ys,
                   "Xs": rs_features(dd, t20, qb)[:len(Ys)]})
    Lmin = min(min(len(t["Yq"]), len(t["Ys"])) for t in TR)
    tpl_q = np.mean([t["Yq"][:Lmin] for t in TR], 0)     # 按帧序平均 = 相位模板
    tpl_s = np.mean([t["Ys"][:Lmin] for t in TR], 0)
    zin_q = fit([stack_band(fir_design(np.zeros_like(t["Xq"]), sq["nlag"]),
                            t["Yq"], W_BAND)[0] for t in TR],
                [stack_band(fir_design(np.zeros_like(t["Xq"]), sq["nlag"]),
                            t["Yq"], W_BAND)[1] for t in TR], sq["lam"])
    zin_s = fit([stack_band(fir_design(np.zeros_like(t["Xs"]), ss["nlag"]),
                            t["Ys"], W_BAND)[0] for t in TR],
                [stack_band(fir_design(np.zeros_like(t["Xs"]), ss["nlag"]),
                            t["Ys"], W_BAND)[1] for t in TR], ss["lam"])

    # ── 留出 ──────────────────────────────────────────────────────────
    HOLD = valid_files(S2, "replay_pp2f*.json")
    print(f"  留出 {len(HOLD)} 条（会话2, 未见轨迹）\n")
    ARMS = ("G-q-v1", "相位模板", "零输入", "G-vNext")
    r2 = {a: {b: [] for b in FRZ} for a in ARMS}
    win = {a: [0, 0] for a in ARMS}
    n39 = d39 = nb = db = 0.0
    pj = {j: [0.0, 0.0] for j in MOVING}
    rms = [0.0, 0.0]
    for f in HOLD:
        dd = load(f)
        t20 = dd["tg_i"][::5]
        Xq, Yq, _ = rq_features(dd)
        qb = base_q(dd, t20)
        Pq = fir_design(Xq, sq["nlag"]) @ zq["W"]
        Ab, Kt, Ys = sensor_target(dd, t20)
        n = len(Ys)
        # 判据 4/5/6（关节层，r_q）
        a, b = band_err(Yq, Pq, *BAND); n39 += a; d39 += b
        A_, B_ = bp(Yq, *BAND), bp(Pq, *BAND)
        for j in MOVING:
            pj[j][0] += float(((A_ - B_)[:, j] ** 2).sum())
            pj[j][1] += float((A_[:, j] ** 2).sum())
        rms[0] += float((Yq[:, MOVING] ** 2).sum())
        rms[1] += float(((Yq - Pq)[:, MOVING] ** 2).sum())
        # 四条臂的 ω̂（都在机体系；真机已 Procrustes 旋入）
        qh = qb + Pq
        Xs = rs_features(dd, t20, qh)[:n]
        Ps = fir_design(Xs, ss["nlag"]) @ zs["W"]
        L = min(Lmin, n)
        pred = {
            "G-q-v1":  kin_gyro_fs(qb, FPS)[:n],
            "相位模板": (kin_gyro_fs(qb[:L] + tpl_q[:L], FPS) + tpl_s[:L]),
            "零输入":   (kin_gyro_fs(qb + fir_design(np.zeros_like(Xq), sq["nlag"]) @ zin_q,
                                    FPS)[:n]
                        + fir_design(np.zeros_like(Xs), ss["nlag"]) @ zin_s),
            "G-vNext": kin_gyro_fs(qh, FPS)[:n] + Ps,
        }
        for aname, S in pred.items():
            m = min(len(Ab), len(S))
            for b_ in FRZ:
                r2[aname][b_].append(best_fit(Ab[:m], S[:m], *b_, rotate=False)[1])
            # 判据 7：最难 20% 窗口（按实测带内能量）
            A9, S9 = bp(Ab[:m], *BAND), bp(S[:m], *BAND)
            A0, S0 = bp(Ab[:m], *BAND), bp(kin_gyro_fs(qb, FPS)[:m], *BAND)
            ne = m // WIN
            en = [float((A9[k*WIN:(k+1)*WIN] ** 2).sum()) for k in range(ne)]
            hard = np.argsort(en)[-max(1, ne // 5):]
            for k in hard:
                sl = slice(k * WIN, (k + 1) * WIN)
                e_new = float(((A9[sl] - S9[sl]) ** 2).sum())
                e_base = float(((A0[sl] - S0[sl]) ** 2).sum())
                win[aname][0] += int(e_new < e_base); win[aname][1] += 1

    print(f"  {'':<12}{'0.5–3 Hz ρ²':>14}{'3–9 Hz ρ²':>12}{'最难20%窗改善':>15}")
    for a in ARMS:
        print(f"  {a:<12}" + "".join(f"{np.median(r2[a][b]):>14.3f}" for b in FRZ)
              + f"{win[a][0]}/{win[a][1]:<14}")
    print(f"  {'冻结门槛':<12}" + "".join(f"{FRZ[b]:>14.3f}" for b in FRZ))

    g = {}
    for i, b in enumerate(FRZ):
        v = np.median(r2["G-vNext"][b])
        g[f"判据{i+1}"] = (f"{b[0]}–{b[1]} Hz ρ² {v:.3f}", FRZ[b], v >= FRZ[b])
    ok3 = all(np.median(r2["G-vNext"][b]) > np.median(r2[a][b])
              for b in FRZ for a in ("G-q-v1", "相位模板", "零输入"))
    g["判据3"] = ("两带均优于三个对照", "全部严格优", ok3)
    v4 = 1 - n39 / max(d39, 1e-30)
    g["判据4"] = (f"r_q 3–9 Hz 关节误差降 {v4*100:.1f}%", "≥20%", v4 >= 0.20)
    imp = [j for j in MOVING if 1 - pj[j][0] / max(pj[j][1], 1e-30) > 0]
    g["判据5"] = (f"逐关节改善 {len(imp)}/5", "≥4/5", len(imp) >= 4)
    v6 = np.sqrt(rms[1] / max(rms[0], 1e-30)) - 1
    g["判据6"] = (f"宽带关节 RMS {v6*100:+.1f}%", "≤+5%", v6 <= 0.05)
    v7 = win["G-vNext"][0] / max(win["G-vNext"][1], 1)
    g["判据7"] = (f"最难20%窗改善 {v7*100:.0f}%", ">50%", v7 > 0.5)

    print(f"\n  {'判据':<8}{'实测':<34}{'门槛':<14}判")
    for k in sorted(g):
        s, t, ok = g[k]
        print(f"  {k:<8}{s:<34}{str(t):<14}{'✓' if ok else '**✗**'}")
    allok = all(v[2] for v in g.values())
    print(f"\n  总判: {'**通过 —— 可冻结 G-vNext**' if allok else '**未通过**'}")
    (OUT / "gate.json").write_text(json.dumps(
        {"pass": bool(allok),
         "criteria": {k: {"got": v[0], "thr": str(v[1]), "ok": bool(v[2])}
                      for k, v in g.items()},
         "rho2_median": {a: {f"{b[0]}-{b[1]}": float(np.median(r2[a][b]))
                             for b in FRZ} for a in ARMS},
         "hard_windows": {a: win[a] for a in ARMS},
         "caliber": ("留出=会话2 pp2f+pp2f60(未见轨迹+跨会话); 两层 nlag/lam 由训练族"
                     "折外选定; 真机逐集 Procrustes 旋入机体系; 逐系列最优位移; "
                     "不可与 17 集开发基准集并列"),
         }, indent=1, ensure_ascii=False))
    print(f"  → {OUT/'gate.json'}")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
