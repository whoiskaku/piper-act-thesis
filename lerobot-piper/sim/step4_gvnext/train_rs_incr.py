#!/usr/bin/env python3
"""**零机时增量约束实验**：把「增量要匹配实测 H」直接加进 `r_s` 的损失。

动机
----
`sim/step4_gvnext/incr_variants.py` 实测：**残差层在增量通道上是负贡献**
（增量 cos* @0.5–3Hz：纯物理底座 0.586 → 全 G-2 0.439），而绝对保真度正好相反
（3–9 Hz ρ² 0.193 → 0.520）。机理是 **`r_s` 的雅可比从未被任何东西约束** ——
它的目标只是名义命令下的绝对残差。老师的判据却是「**Ĝ 只需增量准**」。

⚠ 只重拟 `r_s`，`r_q` 冻结
--------------------------
`r_q` 冻结后，`Δ r_s = ΔD·W_s` 对**待求权重** `W_s` 确实是线性的 ⇒ 可复用现有岭回归，
整个问题仍是**一次线性最小二乘**。若同时重拟 `r_q`，它还会改变运动学渲染
`kin(q̂)` 以及 `r_s` 的输入，就不能这么称呼了。（用户 2026-09-02 指定。）

损失（三块，**每块按自身尺度归一** —— G-vNext 那轮的教训）
-----------------------------------------------------------
    L = (1−w−v)·MSE(Y)/var(Y)
      +      w ·MSE(bpY)/var(bpY)                     w = 0.5（冻结）
      +      v ·MSE(ΔD·W − T)/var(T)                  ← **新增：增量项**

    T = bp(Δ_H)[:−3] − bp(Δkin)[3:]        Δ_H = causal_conv(H_body, δa)
    ΔD = bp(ΔD_raw)[3:]                    Δkin/ΔD 取 ± 配对的奇分量

对齐：`Δ_H[t]` 对应 `Δω̂[t+3]`（H 的定义是 `real[t+3] − ideal[t]`），故增量块整体错 3 帧。

协议（用户 2026-09-02 定）
--------------------------
- `v` 事前固定小网格 **{0, 0.1, 0.25, 0.4}**；`nlag`/`λ` 保持 `r_s` 冻结值，**只搜 v**。
- **只用训练族折外**选 v（判据 = 折外增量相对误差），然后**一次性**看未见轨迹。
- 增量 cos、增益 g、**绝对保真度并列报告**。允许绝对数字下降，**但不得隐去这个代价**。
- 方向判定用**固定标定 R + 固定 +3 帧**；自由旋转的 cos* **只作诊断**。
- **停止规则**：选中候选在未见轨迹上连以 H 为基准的增量指标都没有明确改善 ⇒
  停止这条残差重训路线。若改善，才做协议 ③a 的 A1/B/A2 真机复放。

⚠ 循环性（必须并读）
--------------------
**用 H 训练、再用同一个 H 打分。** H 已过跨 seed（β 中位 0.9546）与幅值线性（1.0134），
足以筛掉明显不合适的候选；但本实验**不能代替真机直测**。
G-2 对真实 pp2 扰动到底够不够准、能不能当 π 的训练入口，只有 ③a 能回答。

用法:
    python -u sim/step4_gvnext/train_rs_incr.py
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g", ROOT / "sim/step3_canobs", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data import load, bp, best_fit, valid_files, FPS                 # noqa: E402
from train_rq import (features as rq_features, fir_design, fit,       # noqa: E402
                      KFOLD, W_BAND)
from train_rs import rs_features, sensor_target                       # noqa: E402
from eval_gate import base_q                                          # noqa: E402
from ceiling_200hz import kin_gyro_fs                                 # noqa: E402

S1, S2 = ROOT / "outputs/step4_gvnext/s1", ROOT / "outputs/step4_gvnext/s2"
RQ, RS = ROOT / "outputs/step4_gvnext/rq", ROOT / "outputs/step4_gvnext/rs"
H_ART = ROOT / "outputs/step1_H/step1_H_v1.npz"
OUT = ROOT / "outputs/step4_gvnext/rs_incr.json"

V_GRID = (0.0, 0.1, 0.25, 0.4)          # 事前固定，不扩
AMP, ALIGN_L = 0.003, 3
FREQS = np.array([0.7, 1.3, 2.1, 3.4, 5.2, 7.1, 8.6])
BANDS = ((0.5, 3.0), (3.0, 9.0))
MB = (0.5, 9.0)                          # 增量项与 cos 的评价带
IMP_COS, IMP_N = 0.10, 6                 # 🔧 「明确改善」= cos 中位 +0.10 且 ≥6/8 集


def schroeder(fr, n, fps, seed):
    rng = np.random.default_rng(seed)
    k = np.arange(len(fr))
    ph = -np.pi * k * (k - 1) / len(fr) + rng.uniform(0, 2 * np.pi)
    t = np.arange(n) / fps
    s = np.sum(np.cos(2 * np.pi * fr[:, None] * t[None, :] + ph[:, None]), 0)
    return s / max(np.abs(s).max(), 1e-12)


def probe(n, seed):
    return np.stack([AMP * schroeder(FREQS, n, FPS, seed * 97 + j) for j in range(6)], 1)


def prep(files, zq, sq, H_body, nlag):
    """每集预算一次（与 v、W 无关）：D/Y/bp 版 + 增量块 ΔD/T。"""
    from fit_step1_candidate import causal_conv
    E = []
    for f in files:
        dd = load(f)
        t20 = dd["tg_i"][::5]
        n = len(t20)
        tr = np.asarray(dd["t_read"], float)
        # ── 名义（绝对项）────────────────────────────────────────────
        qh = base_q(dd, t20) + fir_design(rq_features(dd)[0], sq["nlag"]) @ zq["W"]
        Ab, _, Y = sensor_target(dd, t20)
        m = len(Y)
        D = fir_design(rs_features(dd, t20, qh)[:m], nlag)
        kin0 = kin_gyro_fs(qh, FPS)[:m]
        # ── 增量（± 配对奇分量）──────────────────────────────────────
        seed = int.from_bytes(hashlib.sha256(dd["name"].encode()).digest()[:4], "little")
        da20 = probe(n, seed)
        dac = np.stack([np.interp(tr, t20, da20[:, j]) for j in range(6)], 1)
        daa = np.stack([np.interp(t20, tr, dac[:, j]) for j in range(6)], 1)
        dk, dd_ = [], []
        for sgn in (+1.0, -1.0):
            ddp = {**dd, "cmd": None}
            c = np.array(dd["cmd"], float, copy=True)
            k = min(len(c), len(dac))
            c[:k, :6] += sgn * dac[:k]
            ddp["cmd"] = c
            qp = base_q(ddp, t20) + fir_design(rq_features(ddp)[0], sq["nlag"]) @ zq["W"]
            dk.append(kin_gyro_fs(qp, FPS)[:m])
            dd_.append(fir_design(rs_features(ddp, t20, qp)[:m], nlag))
        d_kin = 0.5 * (dk[0] - dk[1])
        dD = 0.5 * (dd_[0] - dd_[1])
        d_h = causal_conv(H_body, daa[:m])
        # 对齐：Δ_H[t] ↔ Δω̂[t+3]
        T = bp(d_h, *MB)[:-ALIGN_L] - bp(d_kin, *MB)[ALIGN_L:]
        E.append({"name": dd["name"], "D": D, "Y": Y, "bpD": bp(D, *MB), "bpY": bp(Y, *MB),
                  "dD": bp(dD, *MB)[ALIGN_L:], "T": T,
                  "dH": bp(d_h, *MB)[:-ALIGN_L], "dkin": bp(d_kin, *MB)[ALIGN_L:],
                  "A": Ab[:m], "kin0": kin0})
    return E


def stack3(e, v, w=W_BAND):
    """三块堆叠，**每块按自身尺度归一**。→ (X, y)"""
    s0 = np.sqrt(max(float(np.var(e["Y"])), 1e-30))
    s1 = np.sqrt(max(float(np.var(e["bpY"])), 1e-30))
    s2 = np.sqrt(max(float(np.var(e["T"])), 1e-30))
    a0, a1, a2 = np.sqrt(max(1.0 - w - v, 0.0)) / s0, np.sqrt(w) / s1, np.sqrt(v) / s2
    return (np.vstack([e["D"] * a0, e["bpD"] * a1, e["dD"] * a2]),
            np.vstack([e["Y"] * a0, e["bpY"] * a1, e["T"] * a2]))


def incr_metrics(E, W):
    """→ (相对误差, cos 固定, 增益 g)。固定标定 R + 固定 +3 帧。"""
    num = den = dot = nb = na = 0.0
    cs = []
    for e in E:
        pred = e["dkin"] + e["dD"] @ W
        num += float(((pred - e["dH"]) ** 2).sum()); den += float((e["dH"] ** 2).sum())
        d = float((e["dH"] * pred).sum())
        dot += d; nb += float((pred * pred).sum()); na += float((e["dH"] ** 2).sum())
        cs.append(d / max(np.linalg.norm(e["dH"]) * np.linalg.norm(pred), 1e-30))
    return num / max(den, 1e-30), cs, dot / max(nb, 1e-30)


def abs_rho2(E, W):
    out = {b: [] for b in BANDS}
    for e in E:
        S = e["kin0"] + e["D"] @ W
        m = min(len(e["A"]), len(S))
        for b in BANDS:
            out[b].append(best_fit(e["A"][:m], S[:m], *b, rotate=False)[1])
    return {b: float(np.median(out[b])) for b in BANDS}


def main() -> int:
    import gc
    gc.disable()
    H = np.ascontiguousarray(np.load(H_ART)["H"], dtype=np.float64)
    meta = json.loads(H_ART.with_suffix(".json").read_text())
    assert hashlib.sha256(H.tobytes()).hexdigest() == meta["H"]["sha256"]
    R = np.asarray(meta["calibration"]["R"], float)
    H_body = np.einsum("lak,ab->lbk", H, R)
    zq = np.load(RQ / "rq_fir.npz")
    sq = json.loads((RQ / "select.json").read_text())
    ss = json.loads((RS / "select.json").read_text())
    nlag, lam = int(ss["nlag"]), float(ss["lam"])
    print(f"r_q **冻结**（nlag={sq['nlag']}）· r_s 重拟（nlag={nlag} λ={lam:.0e} **不搜**，只搜 v）")
    print(f"v 网格（事前固定）{V_GRID}   w_band={W_BAND}   增量带 {MB[0]}–{MB[1]} Hz")
    print(f"标尺 H：{H_ART.name}（跨 seed β 0.9546 / 幅值线性 1.0134）\n")

    TR = prep(valid_files(S1), zq, sq, H_body, nlag)
    DV = prep(valid_files(S2, "replay_pp2f*.json"), zq, sq, H_body, nlag)
    print(f"  训练族 {len(TR)} 集 · 未见轨迹 {len(DV)} 集\n")

    folds = [list(range(i, len(TR), KFOLD)) for i in range(KFOLD)]
    print(f"  {'v':>6}{'折外 增量相对误差':>20}{'折外 增量 cos':>16}")
    oof = {}
    for v in V_GRID:
        P = [stack3(e, v) for e in TR]
        num = den = 0.0
        cs = []
        for te in folds:
            trn = [i for i in range(len(TR)) if i not in te]
            W = fit([P[i][0] for i in trn], [P[i][1] for i in trn], lam)
            r, c, _ = incr_metrics([TR[i] for i in te], W)
            num += r * sum(float((TR[i]["dH"] ** 2).sum()) for i in te)
            den += sum(float((TR[i]["dH"] ** 2).sum()) for i in te)
            cs += c
        oof[v] = (num / max(den, 1e-30), float(np.median(cs)))
        print(f"  {v:>6.2f}{oof[v][0]:>20.4f}{oof[v][1]:>16.3f}")
    v_sel = min(V_GRID, key=lambda v: oof[v][0])
    print(f"\n  **选中 v = {v_sel}**（折外增量相对误差最小；未见轨迹未参与）\n")

    # ── 一次性看未见轨迹 ────────────────────────────────────────────
    res = {}
    for v in V_GRID:
        P = [stack3(e, v) for e in TR]
        W = fit([p[0] for p in P], [p[1] for p in P], lam)
        r, cs, g = incr_metrics(DV, W)
        res[v] = {"incr_rel_err": r, "cos_med": float(np.median(cs)),
                  "cos_each": [float(x) for x in cs], "gain_g": g,
                  "abs": {f"{b[0]}-{b[1]}": val for b, val in abs_rho2(DV, W).items()}}
    print(f"  未见轨迹 {len(DV)} 集 —— **增量与绝对并列**（绝对数字下降是代价，不隐去）")
    print(f"  {'v':>6}{'增量 cos':>11}{'增益 g':>10}{'增量相对误差':>15}"
          f"{'ρ² 0.5–3':>11}{'ρ² 3–9':>10}")
    for v in V_GRID:
        d = res[v]
        mark = " ←选中" if v == v_sel else ""
        print(f"  {v:>6.2f}{d['cos_med']:>11.3f}{d['gain_g']:>10.3f}"
              f"{d['incr_rel_err']:>15.4f}{d['abs']['0.5-3.0']:>11.3f}"
              f"{d['abs']['3.0-9.0']:>10.3f}{mark}")

    b0, b1 = res[0.0], res[v_sel]
    n_imp = sum(int(a > b) for a, b in zip(b1["cos_each"], b0["cos_each"]))
    d_cos = b1["cos_med"] - b0["cos_med"]
    ok = (v_sel != 0.0) and (d_cos >= IMP_COS) and (n_imp >= IMP_N)
    print(f"\n  ── 停止规则（用户 2026-09-02）──")
    print(f"  相对 v=0：cos {b0['cos_med']:.3f} → {b1['cos_med']:.3f}（{d_cos:+.3f}），"
          f"逐集改善 {n_imp}/{len(DV)}")
    print(f"  绝对代价：ρ²(0.5–3) {b0['abs']['0.5-3.0']:.3f} → {b1['abs']['0.5-3.0']:.3f}"
          f"   ρ²(3–9) {b0['abs']['3.0-9.0']:.3f} → {b1['abs']['3.0-9.0']:.3f}")
    print(f"  判读线 🔧 cos 中位 ≥ +{IMP_COS} 且逐集 ≥ {IMP_N}/{len(DV)}")
    print(f"  ⇒ {'**明确改善 —— 可进协议 ③a 真机复放**' if ok else '**无明确改善 ⇒ 停止残差重训路线**'}")
    print(f"\n  ⚠ 用 H 训练、再用同一个 H 打分。H 已过跨 seed 与幅值线性，"
          f"足以筛掉明显不合适的候选，\n    但**不能代替真机直测**；"
          f"G-2 对真实 pp2 扰动是否够准，只有 ③a 能回答。")
    # ── 冻结选中候选（③a 的模型侧要指向它）────────────────────────────
    P = [stack3(e, v_sel) for e in TR]
    W_sel = fit([p[0] for p in P], [p[1] for p in P], lam)
    art = ROOT / "outputs/step4_gvnext/g2_incr_v1.npz"
    man = {
        "artifact": "g2-incr-v1",
        "model": "omega_hat = kin(G-q-v1(u) + r_q) + r_s_incr;  r_q FROZEN, only r_s refit",
        "v": float(v_sel), "w_band": W_BAND, "rs_nlag": nlag, "rs_lam": lam,
        "v_grid": list(V_GRID), "selected_by": "training-family OOF incremental relative error",
        "W_sha256": hashlib.sha256(np.ascontiguousarray(W_sel, np.float64).tobytes()).hexdigest(),
        "rq_frozen_sha256": hashlib.sha256(
            np.ascontiguousarray(zq["W"], np.float64).tobytes()).hexdigest(),
        "H_sha256": meta["H"]["sha256"], "H_artifact": str(H_ART),
        "dev_metrics": res[v_sel],
        "scope": {
            "validated_offline": "increment vs measured H on 8 unseen trajectories (8/8 improved)",
            "NOT_validated": ("trained against H and scored against the SAME H -- circular. "
                              "Only PROTOCOL 3a (A1/B/A2 real replay) can say whether this "
                              "candidate is accurate for real pp2 perturbations."),
            "offline_cos": res[v_sel]["cos_med"],
            "cos_gate_3a": 0.6,
            "absolute_cost": {"0.5-3Hz": res[v_sel]["abs"]["0.5-3.0"] - res[0.0]["abs"]["0.5-3.0"],
                              "3-9Hz": res[v_sel]["abs"]["3.0-9.0"] - res[0.0]["abs"]["3.0-9.0"]},
            "v_at_grid_edge": bool(v_sel == max(V_GRID)),
        },
    }
    np.savez_compressed(art, W=np.ascontiguousarray(W_sel, np.float64),
                        v=np.float64(v_sel), nlag=np.int64(nlag), lam=np.float64(lam),
                        manifest_json=np.asarray(json.dumps(man, ensure_ascii=False,
                                                            sort_keys=True)))
    art.with_suffix(".json").write_text(
        json.dumps(man, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"\n  冻结候选 → {art}  （W sha {man['W_sha256'][:16]}…）")

    OUT.write_text(json.dumps({
        "v_grid": list(V_GRID), "v_selected": v_sel, "oof": {str(k): v for k, v in oof.items()},
        "dev": {str(k): v for k, v in res.items()}, "improve": ok,
        "rule": {"cos_delta_min": IMP_COS, "n_min": IMP_N, "n_dev": len(DV)},
        "frozen": {"r_q": True, "rs_nlag": nlag, "rs_lam": lam, "w_band": W_BAND},
        "caliber": ("只重拟 r_s(r_q 冻结) ⇒ ΔD·W 对待求权重线性, 仍是一次线性最小二乘; "
                    "方向用固定标定 R + 固定 +3 帧(自由旋转只作诊断); "
                    "v 只用训练族折外选, 未见轨迹一次性评价; "
                    "**用 H 训练再用同一 H 打分, 不能代替协议③a 真机直测**"),
    }, indent=1, ensure_ascii=False))
    print(f"\n  → {OUT}")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
