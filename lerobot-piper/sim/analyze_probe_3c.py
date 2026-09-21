#!/usr/bin/env python3
"""③c 分析：**连续波形** · 单侧 / 奇分量 / 偶分量**分别报告**（零机时）。

与 ③a 的分析端（`analyze_probe_3a.py`）的差别
---------------------------------------------
③a 是**单帧脉冲**，按 `hits` 对齐叠加 `LR=24` 帧的窗口。
③c 的补正是**整段连续**的 ⇒ 没有可叠加的「发」，统计量直接在**整段有效窗**上算。

不确定性按**整次复放**估计
--------------------------
⚠ **不把帧当独立样本。** 除了解析式，另做**整次复放的 jackknife**（留一条复放）。

⚠ **两侧共享同一条裸基线 ⇒ Δ⁺ 与 Δ⁻ 相关，必须保留这个相关性。**

    base = ½(mean A1 + mean A2)      Δ⁺ = mean(Bp) − base      Δ⁻ = mean(Bm) − base

    Var(Δ±)     = σ²/N + σ²/(2M)
    Cov(Δ⁺,Δ⁻)  = Var(base) = σ²/(2M)        ← **正相关，不可忽略**
    Δ_odd = ½(Δ⁺−Δ⁻) = ½(mean Bp − mean Bm)   ← **裸基线精确抵消**
      Var  = σ²/(2N)
    E     = Δ⁺+Δ⁻ = mean Bp + mean Bm − (mean A1 + mean A2)
      Var  = 2σ²/N + 2σ²/M                    ← **裸基线噪声进来两次**

⚠⚠ 三条口径约束（用户 2026-09-21，**分析端必须遵守**）⚠⚠
-----------------------------------------------------------
1. **两侧发送量并非严格反对称** —— `±δa` 独立取整 ⇒ 指令层就有
   `e_cmd = c₊ + c₋ − 2c₀ ≠ 0`（已冻结在 `outputs/probe_3c/sent_counts.npz`）。
   **分析不得假设严格反对称。**
2. **不得用尚待验证的 `H` 扣掉这部分响应，再把剩余量称为独立的线性验证。**
3. 偶分量若明显存在，**只能先报告「系统整体的正负非对称」**，
   其来源**可能包括输入量化、固件整形和机械响应**，
   **不能自动归因于机械非线性**。

允许的三种判读
--------------
    ① 比较冻结模型与真实响应（奇分量信噪足够时）
    ② **无法判定** —— 信噪不足或重复性无法可靠估计；
       **不得**归为「模型失败」，**也不得**归为「真实响应不稳定」
    ③ 偶分量明显存在 ⇒ 报告「系统整体的正负非对称」（来源未归因）

用法:
    python -u sim/analyze_probe_3c.py --self-test      # 五个用例
    python -u sim/analyze_probe_3c.py --root outputs/probe_3c
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

SNR_GATE = 3.0            # 奇分量：低于此 ⇒ **无法判定**
COS_GATE = 0.6            # 方向判据（仅在信噪足够时才适用）
EVEN_RATIO_GATE = 0.3     # 偶分量的容忍线 —— **噪声底高于它时该判据无判别力**
AXES = ["gx", "gy", "gz"]


# ══════════════════════════════════════════════════════════════════════
def components(A1, A2, Bp, Bm):
    """单侧 / 奇 / 偶 —— 三者**分别**返回，附共享基线的相关性。"""
    M1, M2, N1, N2 = len(A1), len(A2), len(Bp), len(Bm)
    T = min(min(len(x) for x in A1 + A2 + Bp + Bm), 10 ** 9)
    m1 = np.mean([x[:T] for x in A1], 0); m2 = np.mean([x[:T] for x in A2], 0)
    mp = np.mean([x[:T] for x in Bp], 0); mm = np.mean([x[:T] for x in Bm], 0)
    base = 0.5 * (m1 + m2)
    d_p, d_m = mp - base, mm - base
    d_odd = 0.5 * (d_p - d_m)
    e_even = d_p + d_m                       # **不除以二**
    # σ：由裸复放之间的差异估计（**整次复放**，不是帧）
    nz = m1 - m2
    sigma = float(np.sqrt(np.mean(nz ** 2)) / np.sqrt(1.0 / M1 + 1.0 / M2))
    M = 0.5 * (M1 + M2); N = 0.5 * (N1 + N2)
    var_side = sigma ** 2 * (1.0 / N + 1.0 / (2 * M))
    cov = sigma ** 2 / (2 * M)               # ← 共享裸基线造成的正相关
    return {"T": T, "d_p": d_p, "d_m": d_m, "d_odd": d_odd, "e_even": e_even,
            "base": base, "sigma": sigma, "M": M, "N": N,
            "sd_side": float(np.sqrt(var_side)),
            "cov_pm": float(cov),
            "corr_pm": float(cov / var_side) if var_side > 0 else float("nan"),
            "sd_odd": float(sigma * np.sqrt(1.0 / (2 * N))),
            "sd_even": float(sigma * np.sqrt(2.0 / N + 2.0 / M))}


def jackknife(A1, A2, Bp, Bm):
    """**留一条整次复放**的 jackknife —— 不把帧当独立样本。"""
    arms = {"A1": A1, "A2": A2, "Bp": Bp, "Bm": Bm}
    vals = {"odd": [], "even": []}
    for k, lst in arms.items():
        if len(lst) <= 1:
            continue
        for i in range(len(lst)):
            sub = {kk: (vv[:i] + vv[i + 1:] if kk == k else vv) for kk, vv in arms.items()}
            c = components(sub["A1"], sub["A2"], sub["Bp"], sub["Bm"])
            vals["odd"].append(float(np.linalg.norm(c["d_odd"])))
            vals["even"].append(float(np.linalg.norm(c["e_even"])))
    out = {}
    for k, v in vals.items():
        n = len(v)
        out[k] = {"n_leave_one": n,
                  "mean": float(np.mean(v)) if n else float("nan"),
                  # jackknife 方差：(n−1)/n · Σ(v_i − v̄)²
                  "sd": float(np.sqrt((n - 1) / n * np.sum((np.asarray(v) - np.mean(v)) ** 2)))
                  if n > 1 else float("nan")}
    return out


def report(c, jk, pred_odd=None, tag="", cmd_even_rms=None):
    T = c["T"]
    nrm = lambda x: float(np.linalg.norm(x))
    rms = lambda x: float(np.sqrt((np.asarray(x) ** 2).mean()))
    print(f"\n{'=' * 78}\n{tag}\n{'=' * 78}")
    print(f"  复放数 A1/A2 各 {c['M']:.0f} · Bp/Bm 各 {c['N']:.0f} · 有效帧 {T}")
    print(f"  σ(单次复放, 由裸复放之差估) = {c['sigma']:.6f} rad/s")

    print(f"\n  ── 单侧（**分别报告**）──")
    for nm, d in (("Δ⁺", c["d_p"]), ("Δ⁻", c["d_m"])):
        print(f"    {nm}  ‖·‖ {nrm(d):.5f} · RMS {rms(d):.6f}  逐轴 RMS "
              + " ".join(f"{a} {rms(d[:, i]):.6f}" for i, a in enumerate(AXES)))
    print(f"    不确定度 sd(Δ±) = {c['sd_side']:.6f} rad/s")
    print(f"    ⚠ **两侧共享同一条裸基线** ⇒ Cov(Δ⁺,Δ⁻) = {c['cov_pm']:.3e}，"
          f"**相关系数 {c['corr_pm']:+.3f}** —— 不可当独立处理")

    print(f"\n  ── 奇分量 Δ_odd = ½(Δ⁺−Δ⁻)（裸基线**精确抵消**）──")
    snr = nrm(c["d_odd"]) / (c["sd_odd"] * np.sqrt(3 * T))
    print(f"    ‖Δ_odd‖ {nrm(c['d_odd']):.5f} · RMS {rms(c['d_odd']):.6f} · "
          f"噪声 {c['sd_odd']:.6f} ⇒ **信噪 {snr:.2f}**（门槛 {SNR_GATE:g}）")
    print(f"    jackknife(留一条整次复放, n={jk['odd']['n_leave_one']}): "
          f"‖Δ_odd‖ 均值 {jk['odd']['mean']:.5f} · sd {jk['odd']['sd']:.5f}")
    four = None
    if pred_odd is not None:
        p = np.asarray(pred_odd)[:T]
        cs = float((c["d_odd"] * p).sum() / max(np.linalg.norm(c["d_odd"]) *
                                                np.linalg.norm(p), 1e-30))
        nr = nrm(c["d_odd"]) / max(nrm(p), 1e-30)
        four = {"cos": cs, "norm_ratio": nr, "g": cs * nr}
        print(f"    对冻结模型：**cos {cs:+.3f} · 范数比 {nr:.3f} · g {cs*nr:.3f}**")
        print(f"    ⚠ **四者分开看**（波形/cos/范数比/g）—— 方向不对时大幅低估范数"
              f"也能乘出 g≈1")

    print(f"\n  ── 偶分量 E = Δ⁺+Δ⁻（**不除以二**）──")
    e_ratio = nrm(c["e_even"]) / max(nrm(c["d_p"]), 1e-30)
    noise_ratio = (c["sd_even"] * np.sqrt(3 * T)) / max(nrm(c["d_p"]), 1e-30)
    print(f"    ‖E‖ {nrm(c['e_even']):.5f} · 比值 ‖E‖/‖Δ⁺‖ = **{e_ratio:.3f}**")
    print(f"    噪声底（同一比值口径）= **{noise_ratio:.3f}** ｜ 容忍线 {EVEN_RATIO_GATE}")
    informative = noise_ratio < EVEN_RATIO_GATE
    print(f"    ⇒ 该判据{'**有**' if informative else '**没有**'}判别力"
          + ("" if informative else "（噪声底已高于容忍线）"))
    if cmd_even_rms is not None:
        print(f"    ⚠ **指令层本身就非反对称**：e_cmd RMS {cmd_even_rms*1e3:.5f} mrad"
              f" —— 这是**输入量化**的确定性产物，**不是**被控对象的非线性")

    # ── 判读 ─────────────────────────────────────────────────────────
    print(f"\n  ── 判读 ──")
    if snr < SNR_GATE:
        verdict = "无法判定"
        print(f"    **② 无法判定** —— 奇分量信噪 {snr:.2f} < {SNR_GATE:g}。")
        print(f"       ⚠ **不得**归为「模型失败」，**也不得**归为「真实响应不稳定」。")
    else:
        verdict = "可比较"
        print(f"    **① 可比较冻结模型与真实响应**（信噪 {snr:.2f} ≥ {SNR_GATE:g}）")
        if four:
            print(f"       方向 cos {four['cos']:+.3f} "
                  f"{'过' if four['cos'] > COS_GATE else '**未过**'} {COS_GATE} 门槛")
    if e_ratio > EVEN_RATIO_GATE and informative:
        print(f"    **③ 偶分量明显存在** ⇒ 只能报告「**系统整体的正负非对称**」。")
        print(f"       ⚠ 来源**可能包括输入量化、固件整形和机械响应**，"
              f"**不能自动归因于机械非线性**。")
    elif not informative:
        print(f"    偶分量：**检出能力不足，无法判断是否足够小** —— "
              f"**不得**因此宣称「线性通过」。")
    return {"snr_odd": float(snr), "verdict": verdict, "even_ratio": float(e_ratio),
            "even_noise_ratio": float(noise_ratio), "even_informative": bool(informative),
            "four": four, "corr_pm": c["corr_pm"], "sigma": c["sigma"],
            "jackknife": jk}


# ══════════════════════════════════════════════════════════════════════
def self_test() -> int:
    """五个用例。**该落入「无法判定」的必须落进去**，不得被判成模型失败或线性通过。"""
    rng = np.random.default_rng(0)
    T, M, N = 560, 6, 10
    t = np.arange(T) / 20.0
    truth = np.stack([0.05 * np.sin(2 * np.pi * 2.0 * t),
                      0.03 * np.sin(2 * np.pi * 3.5 * t + 1.0),
                      0.02 * np.sin(2 * np.pi * 5.0 * t)], 1)
    drift0 = np.cumsum(rng.normal(0, 0.02, (T, 3)), 0) * 0.02

    def mk(sig, sd, drift=0.0, k=1):
        return [drift0 * drift + sig + rng.normal(0, sd, (T, 3)) for _ in range(k)]

    cases = []
    # ① 零响应：应 **无法判定** 或「可比较但无响应」，**绝不能**判模型失败
    cases.append(("① 零响应（δa 完全没引起变化）", 0.0, 0.004, 0.0, None, True))
    # ② 已知响应：信噪高，应可比较且 cos≈1
    cases.append(("② 已知响应（信噪充足）", 1.0, 0.004, 0.0, truth, False))
    # ③ 输入量化不对称：注入一个**纯偶**分量（模拟 e_cmd 的响应）
    cases.append(("③ 输入量化不对称（纯偶分量）", 1.0, 0.004, 0.0, truth, False))
    # ④ 漂移：A1/A2 之间有系统性漂移
    cases.append(("④ 漂移（A1/A2 系统性不同）", 1.0, 0.004, 1.0, truth, False))
    # ⑤ 噪声不足：σ 大到信噪 < 3 ⇒ **必须落「无法判定」**
    cases.append(("⑤ 噪声不足（应落『无法判定』）", 1.0, 0.060, 0.0, truth, True))

    out, ok = [], True
    for i, (nm, gain, sd, dr, pred, expect_undecided) in enumerate(cases, 1):
        sig = gain * truth
        A1 = mk(0.0, sd, dr * 0.0, M)
        A2 = mk(0.0, sd, dr * 1.0, M)          # ④ 让 A2 带漂移
        Bp = mk(+sig, sd, dr * 0.5, N)
        Bm = mk(-sig, sd, dr * 0.5, N)
        if i == 3:                              # 纯偶分量注入两侧**同号**
            even = 0.4 * truth
            Bp = [x + even for x in Bp]; Bm = [x + even for x in Bm]
        c = components(A1, A2, Bp, Bm)
        jk = jackknife(A1, A2, Bp, Bm)
        r = report(c, jk, pred_odd=pred, tag=f"自测 {nm}",
                   cmd_even_rms=1.05e-5 if i == 3 else None)
        undecided = r["verdict"] == "无法判定"
        good = (undecided == expect_undecided)
        if i == 1:
            good &= undecided or (r["four"] is None)
        if i == 2:
            good &= (not undecided) and r["four"]["cos"] > 0.9
        if i == 3:
            good &= r["even_ratio"] > EVEN_RATIO_GATE   # 偶分量应被看到
        print(f"\n  {'✓' if good else '**✗**'} 用例期望"
              f"{'「无法判定」' if expect_undecided else '「可比较」'}，"
              f"实得「{r['verdict']}」")
        ok &= good
        out.append({"case": nm, "pass": bool(good), **{k: v for k, v in r.items()
                                                       if k != "jackknife"}})
    print(f"\n{'=' * 78}\n自测：{sum(o['pass'] for o in out)}/{len(out)} 通过\n{'=' * 78}")
    print("  关键：⑤ 噪声不足**落进「无法判定」**，"
          "**没有**被判成「模型失败」或「线性通过」。")
    (ROOT / "outputs/probe_3c").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs/probe_3c/selftest_3c.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if ok else 1


def main() -> int:
    import gc
    gc.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--root", default="outputs/probe_3c")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    print("真机数据尚未采集（**运动采集未放行**）。本轮只提供 --self-test。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
