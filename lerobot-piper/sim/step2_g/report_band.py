#!/usr/bin/env python3
"""**带权目标重辨识的门禁评价** —— 预注册 `PREREG_BAND_WEIGHTED_IDENT.md` §5。

选参**只用训练损失**（3 seed 里训练损失最低者），与 `G_Q_V1_FREEZE.md` 同规。
诊断集**从未参与选参**，是主门禁。

判据
----
主 1  诊断集腕部陀螺 3–9 Hz ρ² **≥ 0.35**（kin20 渲染口径；G-q-v1 = 0.204）
主 2  4 条里**至少 3 条**改善，且逐集中位为正
防 3  诊断集腕部陀螺 0.5–3 Hz ρ² **≥ 0.70**（G-q-v1 = 0.735）
防 4  逐关节 0.5–3 Hz 位置 ρ² **≥ 0.99**（G-q-v1 = 0.993–1.000）
防 5  留出 A/B 的**宽带**位置损失 ≤ **1.05 ×** 冻结 θ* 的同集数值

口径
----
诊断集 = 4 条 pp2（A1/A3 × ep0/ep1，200 Hz 关节 + 抗混叠 IMU），
逐系列各自最优位移，逐集正交 Procrustes 拟合装着旋转。
**不可与 17 集 pp2 开发基准集的数字并列。**

用法:
    python -u sim/step2_g/report_band.py
"""
from __future__ import annotations
import gc, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g", ROOT / "sim/step3_canobs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DIAG = [(a, e) for a in ("A1", "A3") for e in ("ep0", "ep1")]
FS_I, FPS = 100.0, 20.0
GATE = {"r2_39": 0.35, "r2_lo": 0.70, "pos_lo": 0.99, "broad_ratio": 1.05,
        "n_improve": 3}
BASE = {"r2_39": 0.204, "r2_lo": 0.735}          # G-q-v1 在同口径下的基准


def main() -> int:
    gc.disable()
    import mujoco
    from scipy.signal import butter, filtfilt, firwin
    from scipy.linalg import orthogonal_procrustes
    import optuna_ident_band as B
    import optuna_ident_enc as E
    import g_q_v1 as GQ
    from g_q_v1 import THETA_STAR
    from ceiling_200hz import load_streams, kin_gyro_fs
    from sim.step1_v2_loss import load_calib
    from sim.step2_fidelity import CALIB
    from sim.step2_gap_baseline import MODEL

    lp = lambda x, fc, fs: filtfilt(*butter(4, fc / (fs / 2)), x, axis=0)

    def bp(x, lo, hi):
        k = firwin(41, [lo, hi], fs=FPS, pass_zero=False)
        x = np.atleast_2d(x.T).T
        return np.stack([np.convolve(x[:, j], k, mode="same")
                         for j in range(x.shape[1])], 1)

    def r2_shift(A0, S0, lo, hi, maxL=3):
        """逐系列最优位移下的 ρ²（含逐集 Procrustes 装着旋转）。"""
        out = 0.0
        for L in range(maxL + 1):
            n = min(len(A0) - L, len(S0))
            A, S = A0[L:L + n], S0[:n]
            Sr = S @ orthogonal_procrustes(S, A)[0] if S.shape[1] == 3 else S
            a, s = bp(A, lo, hi), bp(Sr, lo, hi)
            out = max(out, float((a * s).sum() ** 2
                                 / max((a * a).sum() * (s * s).sum(), 1e-30)))
        return out

    # ── 1. 选参：3 seed 里训练损失最低者，**只看训练损失** ──────────────
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    best = None
    print("训练损失（选参唯一依据）")
    for seed in E.SEEDS:
        try:
            st = optuna.load_study(study_name=E.study_name(seed), storage=E.storage(seed))
        except Exception as e:                                   # noqa: BLE001
            print(f"  seed {seed}: 载入失败 {type(e).__name__}")
            continue
        ok = [t for t in st.trials if t.value is not None]
        if not ok:
            print(f"  seed {seed}: 无完成 trial")
            continue
        b = min(ok, key=lambda t: t.value)
        print(f"  seed {seed}: {len(ok):4d} trial   最优 L = {b.value:.6f}  (#{b.number})")
        if best is None or b.value < best[1]:
            best = (seed, b.value, E.theta_of(b))
    if best is None:
        print("\n  **没有可用的 trial —— 先跑 --run**")
        return 2
    seed, Lbest, TH = best
    print(f"\n  选中 seed {seed}，训练 L = {Lbest:.6f}")

    R = load_calib(CALIB)[0]
    m = mujoco.MjModel.from_xml_path(str(MODEL))

    # ── 2. 训练/留出上的分解 ───────────────────────────────────────────
    tr, ha, hb = E.split()
    print(f"\n{'集':<10}{'':>4}{'L_broad':>12}{'L_39':>10}{'加权 L':>10}")
    hold = {}
    for tag, names in (("训练", tr), ("留出A", ha), ("留出B", hb)):
        items = B.prep_band(names, m.opt.timestep)
        r0 = B.evaluate_band(m, items, THETA_STAR, R)
        r1 = B.evaluate_band(m, items, TH, R)
        hold[tag] = (r0["broad"], r1["broad"])
        print(f"{tag:<10}{'v1':>4}{r0['broad']:>12.5f}{r0['band']:>10.4f}{r0['loss']:>10.4f}")
        print(f"{tag:<10}{'新':>4}{r1['broad']:>12.5f}{r1['band']:>10.4f}{r1['loss']:>10.4f}"
              f"   宽带比 {r1['broad']/max(r0['broad'],1e-12):.3f}")

    # ── 3. 诊断集（主门禁）─────────────────────────────────────────────
    print(f"\n诊断集（4 条 pp2，200 Hz + 抗混叠；**从未参与选参**）")
    print(f"  {'集':<10}{'3–9 Hz ρ²':>22}{'0.5–3 Hz ρ²':>22}")
    print(f"  {'':<10}{'v1':>10}{'新':>11}{'v1':>11}{'新':>11}")
    d39, dlo, posr2 = {"v1": [], "新": []}, {"v1": [], "新": []}, {"v1": [], "新": []}
    for arm, ep in DIAG:
        d = json.loads((ROOT / f"outputs/probe/replay_{ep}_{arm}.json").read_text())
        g100, q200, tg_i, tg_q = load_streams(ep, arm)
        cmd = np.asarray(d["cmd"], float)
        ts, tv = np.asarray(d["t_send"], float), np.asarray(d["t"], float)
        A20 = lp(g100, 9.0, FS_I)[::5]
        tg20 = tg_i[::5]
        q_true = np.stack([np.interp(tg20, tg_q, lp(q200, 9.0, 200.0)[:, j])
                           for j in range(6)], 1)
        row = {}
        for tag, th in (("v1", None), ("新", TH)):
            q_f = (GQ.predict_qpos(cmd, t_send=ts, t_read=tv) if th is None
                   else GQ.predict_with_theta(cmd, th, t_send=ts, t_read=tv)[0])
            q_f = np.radians(q_f)
            q_i = np.stack([np.interp(tg20, tv, q_f[:, j]) for j in range(6)], 1)
            S = kin_gyro_fs(q_i, FPS)
            a = r2_shift(A20, S, 3.0, 9.0); b = r2_shift(A20, S, 0.5, 3.0)
            d39[tag].append(a); dlo[tag].append(b); row[tag] = (a, b)
            # 逐关节 0.5–3 Hz 位置 ρ²（防退化判据 4）
            pj = []
            for j in range(6):
                if np.degrees(q_true[:, j].max() - q_true[:, j].min()) < 5:
                    continue
                pj.append(r2_shift(q_true[:, j:j + 1], q_i[:, j:j + 1], 0.5, 3.0))
            posr2[tag].append(min(pj))
        print(f"  {arm}/{ep:<6}{row['v1'][0]:>10.3f}{row['新'][0]:>11.3f}"
              f"{row['v1'][1]:>11.3f}{row['新'][1]:>11.3f}")
    m39 = {k: float(np.mean(v)) for k, v in d39.items()}
    mlo = {k: float(np.mean(v)) for k, v in dlo.items()}
    print(f"  {'均值':<10}{m39['v1']:>10.3f}{m39['新']:>11.3f}"
          f"{mlo['v1']:>11.3f}{mlo['新']:>11.3f}")
    imp = [n > v for n, v in zip(d39["新"], d39["v1"])]
    med = float(np.median([n - v for n, v in zip(d39["新"], d39["v1"])]))
    pmin = float(min(posr2["新"]))

    # ── 4. 门禁 ────────────────────────────────────────────────────────
    ratio = max(hold["留出A"][1] / max(hold["留出A"][0], 1e-12),
                hold["留出B"][1] / max(hold["留出B"][0], 1e-12))
    C = [("主1 3–9 Hz ρ² ≥ 0.35", m39["新"], GATE["r2_39"], m39["新"] >= GATE["r2_39"]),
         (f"主2 ≥3/4 集改善 且中位>0", f"{sum(imp)}/4, 中位 {med:+.3f}", "3/4 & >0",
          sum(imp) >= GATE["n_improve"] and med > 0),
         ("防3 0.5–3 Hz ρ² ≥ 0.70", mlo["新"], GATE["r2_lo"], mlo["新"] >= GATE["r2_lo"]),
         ("防4 逐关节位置 ρ² ≥ 0.99", pmin, GATE["pos_lo"], pmin >= GATE["pos_lo"]),
         ("防5 留出宽带 ≤ 1.05×", ratio, GATE["broad_ratio"], ratio <= GATE["broad_ratio"])]
    print(f"\n{'判据':<28}{'实测':>16}{'门槛':>12}   判")
    for nm, got, thr, ok in C:
        g = f"{got:.3f}" if isinstance(got, float) else str(got)
        t = f"{thr:.2f}" if isinstance(thr, float) else str(thr)
        print(f"{nm:<28}{g:>16}{t:>12}   {'✓' if ok else '**✗**'}")
    allok = all(c[3] for c in C)
    print(f"\n  总判: {'**通过 —— 可冻结为 G-q-v2**' if allok else '**未通过**'}")
    out = ROOT / "outputs/step2_g/optuna_band"
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate.json").write_text(json.dumps({
        "seed": seed, "train_loss": Lbest, "theta": TH.tolist(),
        "diag_r2_39": d39, "diag_r2_lo": dlo, "pos_r2_min": posr2,
        "holdout_broad": hold, "broad_ratio": ratio,
        "gate": {c[0]: bool(c[3]) for c in C}, "pass": bool(allok),
        "caliber": ("诊断集 4 条 pp2/2 会话, 200 Hz 关节 + 抗混叠 IMU, "
                    "kin20 渲染, 逐系列最优位移, 逐集 Procrustes; "
                    "不可与 17 集开发基准集并列"),
    }, indent=1))
    print(f"  → {out/'gate.json'}")
    return 0 if allok else 1


if __name__ == "__main__":
    _c = main()
    sys.stdout.flush()
    import os
    os._exit(_c)
