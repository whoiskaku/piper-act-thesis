#!/usr/bin/env python3
"""③a 分析（单帧脉冲 · 四重复 · **跨复放 ± 配对**）—— `reports/CHECKLIST_3a.md` 的分析端。

为什么另写而不改 `analyze_probe.py`
------------------------------------
那个文件已演化成 **v2** 形态，与 ③a 的产物四处不兼容：硬编码单目录（四重复会互相覆盖）·
期望 `pred` 是 `(K,T,3)` 集成 · `LR=24`（为 8 帧小波调的）· **不施加 R**。

⚠ 为什么 ± 配对必须**跨复放**而不是跨时间
------------------------------------------
协议原设计把 ± 分散在**不同时刻**的脉冲上。但实测：同一关节的逐发模型响应范数
差 **4×(j2) – 27×(j3)**（每发落在不同臂位形）。只有 1–2 发/符号时，
`‖mean(+)+mean(−)‖` 被**位形变化**主导 ⇒ **判据④ 测不到符号非线性**（自测发现）。

改为 `_p`(+δa) / `_m`(−δa) 两条序列、**帧号完全相同**，B 臂 r1/r2 跑 `_p`、r3/r4 跑 `_m`：

    Δ⁺ = mean(Bp) − mean(A1,A2)      Δ⁻ = mean(Bm) − mean(A1,A2)
    ④  = ‖Δ⁺ + Δ⁻‖ / ‖Δ⁺‖            两侧位形**逐帧严格相同** ⇒ 干净的符号非线性
    Δ_odd = (Δ⁺ − Δ⁻)/2              ①②③ 用它；裸臂项在奇分量里**精确抵消**

机时不变（仍 4 条 B）。这更好地实现了协议「顺带验奇对称性（线性度的直接检验）」的原意。

⚠ 旋转 `R` 与位移：**只由独立的 A0 定，之后全程固定**
- `A0` 是不含探针的裸复放，与 A1/B/A2 分开采、**不进入 Δ**。
- 由 `A0` 拟合 `R`（`ω̂_model @ R ≈ real`）与整数位移 `L`，之后**不重拟、不扫描**。
- 理由：自由旋转会把 `cos` 变成自由参数（离线同一份数据：冻结 R 给 0.084，自由旋转给 0.412）。

判据（`PROTOCOL_robot_validation.md` ③a）
 ① 信噪 > 3 · ② cos > 0.6（不过 ⇒ 停）· ③ `g` **只报告** · ④ 奇对称 < 0.3（不过 ⇒ 停）

⚠ `g = cos × 范数比`：方向不对时大幅低估范数也能乘出 `g≈1` 的假象
⇒ **波形 / cos / 范数比 / g 四者分开报**。

用法:
    python -u sim/analyze_probe_3a.py --self-test              # 线性用例，④ 应通过
    python -u sim/analyze_probe_3a.py --self-test --nonlin 0.5 # 注入非线性，④ 应抓到
    python -u sim/analyze_probe_3a.py                          # 真机数据
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

ROOTD = ROOT / "outputs/probe_3a"
SEQ = ROOTD / "probe_3a.npz"
FIG = ROOTD / "fig_3a_pulse.png"
REPS_P, REPS_M, REPS_A = ("r1", "r2"), ("r3", "r4"), ("r1", "r2", "r3", "r4")
SHIFTS = range(0, 7)
SNR_GATE, COS_GATE, ASYM_GATE = 3.0, 0.6, 0.3


def gyro(p: Path) -> np.ndarray:
    return np.asarray(json.loads(p.read_text())["imu"], float)[:, 3:6]


def load_arm(arm: str, root: Path, reps) -> list[np.ndarray]:
    out = [gyro(root / r / f"replay_{arm}.json") for r in reps
           if (root / r / f"replay_{arm}.json").exists()]
    if not out:
        raise SystemExit(f"找不到 {arm}：{root}/{{{','.join(reps)}}}/replay_{arm}.json")
    T = min(len(x) for x in out)
    return [x[:T] for x in out]


def fit_frame(a0: np.ndarray, om: np.ndarray):
    """**只用 A0** 定 (R, L)：`ω̂_model[t] @ R ≈ real[t+L]`。"""
    from scipy.linalg import orthogonal_procrustes
    best = None
    for L in SHIFTS:
        m = min(len(a0) - L, len(om))
        x, y = om[:m], a0[L:L + m]
        R, _ = orthogonal_procrustes(x, y)
        c = float((y * (x @ R)).sum()
                  / max(np.linalg.norm(y) * np.linalg.norm(x @ R), 1e-30))
        if best is None or c > best[0]:
            best = (c, L, R)
    c, L, R = best
    return R, L, c, float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def stack(x, hits, j, LR):
    """把某关节的脉冲窗按已知帧号叠加（`_p` 里符号全为 +，无需符号整合）。"""
    acc, n = np.zeros((LR, 3)), 0
    for t, jj, _ in hits:
        if int(jj) == j and int(t) + LR <= len(x):
            acc += x[int(t):int(t) + LR]; n += 1
    return acc / max(n, 1), n


def main() -> int:
    import gc
    gc.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOTD)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--nonlin", type=float, default=0.0,
                    help="自测用：注入符号非线性比例 f（Δ⁻ 幅值 ×(1+f)）；④ 应回收 ≈f")
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()

    z = np.load(SEQ, allow_pickle=False)
    pred, pred_m = np.asarray(z["pred"], float), np.asarray(z["pred_m"], float)
    om0, hits = np.asarray(z["omega0"], float), np.asarray(z["hits"], float)
    LR = int(z["lr"])

    if a.self_test:
        rng = np.random.default_rng(0)
        th = np.radians(37.0)
        R_t = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0],
                        [0, 0, 1.0]])
        L_t, g_t, sig, f = 3, 1.35, 0.004, a.nonlin
        T = len(pred)
        base = rng.normal(0, 0.25, (T, 3)).cumsum(0) * 0.02
        sh = lambda X: np.vstack([np.zeros((L_t, 3)), X[:T - L_t]])
        mk = lambda e: base + sh(e) + rng.normal(0, sig, (T, 3))
        A0 = [base + sh(om0 @ R_t) + rng.normal(0, sig, (T, 3))]
        A1 = [mk(np.zeros_like(pred)) for _ in REPS_A]
        A2 = [mk(np.zeros_like(pred)) for _ in REPS_A]
        # ⚠ 用**奇分量**合成，f=0 才是**严格**反对称。若直接用 pred / pred_m，
        # 会把 **G-2 自身的偶次含量**当成"真机非线性"注入进来（自测曾因此误报 ④ 失灵）。
        P_odd = 0.5 * ((pred - pred_m) @ R_t)
        Bp = [mk(g_t * P_odd) for _ in REPS_P]
        Bm = [mk(-g_t * (1 + f) * P_odd) for _ in REPS_M]
        print(f"=== 自检（注入 R={np.degrees(th):.0f}°, L={L_t}, g={g_t}, σ={sig}, "
              f"**符号非线性 f={f}**）===")
    else:
        cal = a.root / "calib"
        A0 = load_arm("A0", cal if cal.exists() else a.root, REPS_A + ("",))
        A1 = load_arm("A1", a.root, REPS_A)
        A2 = load_arm("A2", a.root, REPS_A)
        Bp = load_arm("Bp", a.root, REPS_P)
        Bm = load_arm("Bm", a.root, REPS_M)
        print(f"复放: A0 {len(A0)} · A1 {len(A1)} · A2 {len(A2)} · "
              f"Bp {len(Bp)} · Bm {len(Bm)}")

    R, L, c0, ang = fit_frame(np.mean(A0, 0), om0)
    print(f"\n  **(R, L) 只由独立 A0 定，之后全程固定**")
    print(f"  位移 L = {L} 帧 · 旋转角 {ang:.1f}° · A0 拟合 cos = {c0:.3f}")

    T = min(min(len(x) for x in A1 + A2 + Bp + Bm), len(pred))
    m1, m2 = np.mean([x[:T] for x in A1], 0), np.mean([x[:T] for x in A2], 0)
    mp, mm = np.mean([x[:T] for x in Bp], 0), np.mean([x[:T] for x in Bm], 0)
    E1, E2, Ep, Em = len(A1), len(A2), len(Bp), len(Bm)
    d_p, d_m = mp - 0.5 * (m1 + m2), mm - 0.5 * (m1 + m2)
    d_odd = 0.5 * (d_p - d_m)
    nz = m1 - m2
    sigma = float(np.sqrt(np.mean(nz ** 2))) / np.sqrt(1 / E1 + 1 / E2)   # 单次复放 σ
    n_odd = sigma * 0.5 * np.sqrt(1 / Ep + 1 / Em)      # 奇分量里裸臂**精确抵消**
    n_sum = sigma * np.sqrt(1 / Ep + 1 / Em + (1 / E1 + 1 / E2))          # Δ⁺+Δ⁻
    shift = lambda X: np.vstack([np.zeros((L, 3)), X[:T - L]])
    p_odd = shift(0.5 * ((pred - pred_m) @ R))
    print(f"  σ(单次复放) {sigma:.5f} → 奇分量噪声 {n_odd:.5f} · Δ⁺+Δ⁻ 噪声 {n_sum:.5f}")

    print(f"\n  {'关节':>5}{'发数':>6}{'信噪':>8}{'cos':>8}{'范数比':>9}{'g':>8}"
          f"{'④奇对称':>10}{'(噪声预期)':>12}")
    rows = {}
    for j in sorted({int(x) for x in hits[:, 1]}):
        s, n_ = stack(d_odd, hits, j, LR)
        p, _ = stack(p_odd, hits, j, LR)
        sp, _ = stack(d_p, hits, j, LR)
        sm, _ = stack(d_m, hits, j, LR)
        snr = float(np.linalg.norm(s) / max(n_odd * np.sqrt(LR * 3 / n_), 1e-30))
        dot = float((s * p).sum())
        cos = dot / max(np.linalg.norm(s) * np.linalg.norm(p), 1e-30)
        nr = float(np.linalg.norm(s) / max(np.linalg.norm(p), 1e-30))
        g = dot / max(float((p * p).sum()), 1e-30)
        asym = float(np.linalg.norm(sp + sm) / max(np.linalg.norm(sp), 1e-30))
        a_exp = float(n_sum * np.sqrt(LR * 3 / n_) / max(np.linalg.norm(sp), 1e-30))
        rows[j] = {"n": n_, "snr": snr, "cos": cos, "norm_ratio": nr, "g": g,
                   "asym": asym, "asym_noise_expected": a_exp, "s": s, "p": p}
        bad4 = asym > max(ASYM_GATE, 2 * a_exp)
        print(f"  j{j+1:<4}{n_:>6}{snr:>8.2f}{cos:>8.3f}{nr:>9.3f}{g:>8.3f}"
              f"{asym:>10.3f}{a_exp:>12.3f}"
              f"{'' if snr > SNR_GATE else ' ✗①'}{'' if cos > COS_GATE else ' ✗②'}"
              f"{' ✗④' if bad4 else ''}")

    agg = {k: float(np.mean([r[k] for r in rows.values()]))
           for k in ("snr", "cos", "norm_ratio", "g")}
    nonlin = [r["asym"] > max(ASYM_GATE, 2 * r["asym_noise_expected"])
              for r in rows.values()]
    amax = max(r["asym"] for r in rows.values())
    print(f"\n  信噪   = {agg['snr']:.2f}   "
          f"{'✓' if agg['snr'] > SNR_GATE else '**✗ 判据① — 加大探针/增加复放**'}")
    print(f"  cos    = {agg['cos']:.3f}  "
          f"{'✓' if agg['cos'] > COS_GATE else '**✗ 判据② — 停，不进 ③b**'}")
    print(f"  范数比 = {agg['norm_ratio']:.3f}  ‖Δ_real‖/‖Δ_model‖")
    print(f"  增益 g = {agg['g']:.3f}  = cos × 范数比 "
          f"({agg['cos']:.3f} × {agg['norm_ratio']:.3f} = "
          f"{agg['cos']*agg['norm_ratio']:.3f})  **只报告，不进通过条件**")
    print(f"  ④奇对称 最大 = {amax:.3f}（与 max(0.3, 2×噪声预期) 比）  "
          f"{'✓' if not any(nonlin) else '**✗ 判据④ — 停**'}")
    ok = agg["snr"] > SNR_GATE and agg["cos"] > COS_GATE and not any(nonlin)
    print(f"\n  **③a 判决: {'通过 — 可进 ③b' if ok else '未通过 — 不进 ③b'}**")

    if not a.no_fig:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for f_ in ("Noto Sans CJK JP", "Noto Sans CJK SC", "IPAexGothic"):
            if any(f_ in x.name for x in font_manager.fontManager.ttflist):
                plt.rcParams["font.family"] = f_; break
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(len(rows), 3, figsize=(13, 2.6 * len(rows)), squeeze=False)
        t = np.arange(LR) / 20.0
        for i, (j, r) in enumerate(sorted(rows.items())):
            for c in range(3):
                ax[i][c].plot(t, r["s"][:, c], color="#1f6fb2", lw=2.0, label="Δ_real 実機")
                ax[i][c].plot(t, r["p"][:, c], color="#d62728", lw=1.6, ls="--",
                              label="Δ_model (g2-incr-v1)")
                ax[i][c].axhline(0, color="0.7", lw=0.6)
                ax[i][c].grid(alpha=.25); ax[i][c].tick_params(labelsize=8)
                if i == 0 and c == 0:
                    ax[i][c].legend(fontsize=8.5)
                ax[i][c].set_title(f"j{j+1}  軸 {'xyz'[c]}" + (
                    f"   cos {r['cos']:.3f} · 範数比 {r['norm_ratio']:.2f} · g {r['g']:.2f}"
                    if c == 1 else ""), fontsize=9)
                if i == len(rows) - 1:
                    ax[i][c].set_xlabel("脈衝後の時間 [s]", fontsize=9)
                if c == 0:
                    ax[i][c].set_ylabel("角速度 [rad/s]", fontsize=9)
        fig.suptitle(f"③a 脈衝応答の重ね図（奇分量 (Δ⁺−Δ⁻)/2 · 跨複放 ± 対 · "
                     f"R と位移は独立 A0 で固定）    cos {agg['cos']:.3f}（門槛 0.6）",
                     fontsize=12, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(FIG, dpi=150)
        print(f"  → {FIG}")

    if a.self_test:
        # 注入 f 后奇分量幅值变成 (1+(1+f))/2 = (2+f)/2 倍 ⇒ 期望 g = 1.35·(2+f)/2。
        # 先前直接拿 1.35 比，导致 f≠0 的用例误报"核心回收 ✗"。
        g_exp = 1.35 * (2 + a.nonlin) / 2
        e_g, e_a = abs(agg["g"] - g_exp) / g_exp, abs(ang - 37.0)
        print(f"\n  自检: g {agg['g']:.3f} vs 期望 {g_exp:.3f}（误差 {e_g*100:.1f}%）· "
              f"旋转 {ang:.1f}° vs 37° · 位移 {L} vs 3 · cos {agg['cos']:.3f} · "
              f"④ {amax:.3f} vs 注入 f={a.nonlin}")
        core = e_g < 0.05 and e_a < 2.0 and L == 3 and agg["cos"] > 0.95
        g4 = (not any(nonlin)) if a.nonlin == 0 else any(nonlin)
        print(f"  核心回收 {'✓' if core else '✗'} · "
              f"判据④ {'✓ 正确' + ('通过' if a.nonlin == 0 else '抓到非线性') if g4 else '✗ 失灵'}")
        print(f"  {'✅ 端到端自测通过' if core and g4 else '❌ 自测未过'}")
        return 0 if (core and g4) else 2
    (a.root / "result_3a.json").write_text(json.dumps(
        {"pass": bool(ok), "aggregate": agg,
         "per_joint": {f"j{j+1}": {k: v for k, v in r.items() if k not in ("s", "p")}
                       for j, r in rows.items()},
         "frame": {"R": R.tolist(), "shift": int(L), "rot_deg": ang, "a0_fit_cos": c0,
                   "determined_by": "independent A0 only; fixed across A1/A2/Bp/Bm"},
         "design": "cross-replay +/- pairing; odd component for 1/2/3; sum for 4",
         "gates": {"snr": SNR_GATE, "cos": COS_GATE, "asym": ASYM_GATE,
                   "note": "g is reported only, NOT part of the pass condition"},
         }, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    import os
    os._exit(rc)
