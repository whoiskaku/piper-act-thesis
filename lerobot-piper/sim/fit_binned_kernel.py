#!/usr/bin/env python3
"""**位形分箱响应核 vs 固定核**（零机时）—— 严格按 `reports/PREREG_BINNED_KERNEL.md`。

⚠ 本轮**只比响应核**，`G-2` 不进拟合也不进评分（预注册 §2.0）：

    ĝap_m = H_m ⊛ δa
    E_{m,e} = ‖bp(gap_e − ĝap_{m,e})‖²_F / (3·T_e)

**`G₂(u) + K_φ⊛δa` 是「后续组合形式」，不是本轮评价的模型。**
报告里**不得**把结论写成「对 `ω̂ = G₂ + K⊛δa` 的评价」。

数据角色（依**数据设计**固定，不依结果 —— 预注册 §1）
------------------------------------------------------
| P1 | 1.5 mrad · 8 个**各不相同**的种子 | **拟合 + 内部选参** |
| P2 | 1.5 mrad · **3 个未见种子各 ×2** | **主判据** |
| P3 | 3.0 mrad · **复用 P1 的种子 1000–1003** | **跨幅值压力验证**（次要判据） |

⚠ P3 既改幅值又复用训练种子 ⇒ **不是与 P2 等价的独立折**。
⚠ **不做「P2 拟合、P3 评价」支线** —— 会混淆种子变化与幅值变化。

门槛（事前写死，预注册 §3/§4）
------------------------------
    P2 通过 ⇔ 三个 d_s 全正 且 d̄ ≥ 5%
    P3 通过 ⇔ 四个 d_s 全正 且 d̄ ≥ 5%
`5%` 是**是否值得投入后续工作的筛查阈**，不是统计显著性、也不是可辨识性证明。
⚠ **不得**拿真机 `C−A = −8.0%` 支撑它 —— 预测误差与控制后真机误差**不是同一指标**。
**本轮不报 p 值作主判据**（P2 只有 3 个独立种子，定位是开发筛查）。

用法:
    python -u sim/fit_binned_kernel.py            # 全流程
    python -u sim/fit_binned_kernel.py --selftest # 只跑自测
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

H_JSON = ROOT / "outputs/step1_H/step1_H_v1.json"
OUT = ROOT / "outputs/step1_H/binned_kernel_v1.json"

THR = 0.10234            # 分界：**仅由 P1 名义指令**算的中位数（预注册 §2 写死）
EDGE = 20                # 有效帧：首尾各丢 20（≥ N_LAGS−1=15 且 ≥ 滤波半窗 20）
GATE = 0.05              # 筛查阈 5%
JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]
SUBBANDS = {"0.5-9Hz": (0.5, 9.0), "0.5-3Hz": (0.5, 3.0), "3-9Hz": (3.0, 9.0)}


# ══════════════════════════════════════════════════════════════════════
#  设计矩阵：固定核 = 6 列/滞后；分箱核 = 12 列/滞后（按**输入时刻**的箱分流）
# ══════════════════════════════════════════════════════════════════════
def design(da: np.ndarray, binv: np.ndarray | None, n_lags: int) -> np.ndarray:
    """`binv=None` → 固定核 (T, 16·6)；否则分箱核 (T, 16·12)。

    ⚠ **在输入发生时选箱**：滞后 τ 那一列用 `bin[t−τ]`，
    等价于先把 `δa` 按箱拆成两路**再各自**做因果卷积
    ⇒ **不存在「把不连续的同箱片段拼接后再卷积」**（预注册 §2）。
    """
    import fit_step1_candidate as S1
    if binv is None:
        return S1.causal_design(da, n_lags)
    z0 = np.where(binv[:, None] == 0, da, 0.0)
    z1 = np.where(binv[:, None] == 1, da, 0.0)
    return np.concatenate([S1.causal_design(z0, n_lags),
                           S1.causal_design(z1, n_lags)], axis=1)


def fit(eps, binned: bool, mult: float, n_lags: int):
    """只用给定集拟合。逐集常数从两侧扣除（与 `fit_H` 一致，**不是**带通）。"""
    import fit_step1_candidate as S1
    xs, ys = [], []
    for e in eps:
        X = design(e.da, e.binv if binned else None, n_lags)
        xs.append(X - X.mean(0, keepdims=True))
        ys.append(np.asarray(e.gap, float) - np.mean(e.gap, 0, keepdims=True))
    coef, lam = S1._ridge_solve(np.concatenate(xs), np.concatenate(ys), mult)
    return coef, lam


def predict(e, coef, binned: bool, n_lags: int) -> np.ndarray:
    return design(e.da, e.binv if binned else None, n_lags) @ coef


def energy(e, coef, binned: bool, n_lags: int, band) -> float:
    """预注册 §2.0.1 的评分口径，**顺序写死**：

        ① res = gap − ĝap（raw 对齐空间）
        ② 扣该集 res 的常数（在**全集**上算）   ← ⚠ 这是**离线评分处理**
        ③ bandpass41
        ④ 取有效帧（首尾各丢 EDGE）
        ⑤ 能量 / (3·T_e)

    ⚠ ② 用到了验证集自身的数据，**不是部署时可获得的信息**；
      沿用它只为与 `fit_H` 的 nuisance 处理同口径。
    """
    import fit_step1_candidate as S1
    res = np.asarray(e.gap, float) - predict(e, coef, binned, n_lags)
    res = res - res.mean(0, keepdims=True)
    old = S1.METRIC_BAND
    try:
        S1.METRIC_BAND = band
        res = S1.bandpass41(res)
    finally:
        S1.METRIC_BAND = old
    v = res[EDGE:-EDGE]
    if len(v) <= 0:
        raise ValueError(f"{e.tier} ep{e.ep}: 有效帧为 0")
    return float((v ** 2).sum() / (3 * len(v)))


def by_seed_energy(eps, coef, binned, n_lags, band) -> dict[int, float]:
    """**先对同一种子的各集误差取平均**（预注册 §3）。"""
    acc = {}
    for e in eps:
        acc.setdefault(e.seed, []).append(energy(e, coef, binned, n_lags, band))
    return {s: float(np.mean(v)) for s, v in acc.items()}


def select_multiplier(p1, binned: bool, n_lags: int, grid):
    """P1 内部 leave-one-seed-out；**两个模型同一套折、同一网格、同一指标**。

    平局取**更大**的 multiplier（更强正则化，保守侧）。
    端点被选中时**照实报告、不外延网格**。
    """
    seeds = sorted({e.seed for e in p1})
    rows = []
    for m in grid:
        tot = []
        for s in seeds:
            tr = [e for e in p1 if e.seed != s]
            te = [e for e in p1 if e.seed == s]
            coef, _ = fit(tr, binned, m, n_lags)
            tot.append(np.mean([energy(e, coef, binned, n_lags, SUBBANDS["0.5-9Hz"])
                                for e in te]))
        rows.append({"mult": float(m), "cv": float(np.mean(tot))})
    best = min(r["cv"] for r in rows)
    # 平局取更大 ⇒ 在并列最优里取 mult 最大者
    chosen = max(r["mult"] for r in rows if r["cv"] <= best * (1 + 1e-12))
    at_edge = chosen in (min(grid), max(grid))
    return chosen, rows, at_edge


# ══════════════════════════════════════════════════════════════════════
def load(S1, rend, R, b):
    logs = S1.discover_logs(ROOT / "outputs/ab4", ("P1", "P2", "P3"))
    eps = {t: [S1.load_episode(p, t, R, b, rend, 550) for p in ps]
           for t, ps in logs.items()}
    S1.validate_protocol(eps)
    for t, ps in logs.items():
        for e, p in zip(eps[t], ps):
            pay = json.loads(p.read_text())
            cmd, _, _ = S1._render_logged_chunks(
                sorted(pay["chunks"], key=lambda c: int(c["start"])), rend,
                len(pay["frames"]))
            s = cmd[:len(e.da), 1] + cmd[:len(e.da), 2]      # q̄2 + q̄3（名义指令）
            e.binv = (s >= THR).astype(np.int8)
    return eps


def selftest(S1, eps, BC, n_lags) -> bool:
    """跑拟合之前的自测 —— 每条都对着预注册的一句话。"""
    print("\n" + "=" * 72 + "\n自测（对照预注册）\n" + "=" * 72)
    ok = {}

    def chk(k, c, d=""):
        ok[k] = bool(c); print(f"  {'✓' if c else '**✗**'} {k}{('   ' + d) if d else ''}")

    # 1 §4：P3 必须用日志实际(裁剪后)的 da，**不得**用 2× P1 波形代替
    e3 = eps["P3"][0]
    n = len(e3.da)
    real3 = BC.make_probe(e3.seed, 3.0)[:n].astype(float)
    fake3 = 2.0 * BC.make_probe(e3.seed, 1.5)[:n].astype(float)
    chk("1 P3 的 da == make_probe(seed, 3.0)（日志实际值）",
        np.abs(e3.da - real3).max() == 0.0)
    chk("1b P3 的 da **≠** 2 × make_probe(seed, 1.5)（触界使其不成比例）",
        np.abs(e3.da - fake3).max() > 0,
        f"最大差 {np.abs(e3.da-fake3).max()*1e3:.3f} mrad · "
        f"不相等的帧 {int((np.abs(e3.da-fake3)>0).any(1).sum())}/{n}")

    # 2 §2：分箱设计矩阵 = 固定核的**无损分流**（两路相加复原）
    e = eps["P1"][0]
    Xf = design(e.da, None, n_lags)
    Xb = design(e.da, e.binv, n_lags)
    k = n_lags * 6
    chk("2 分箱设计矩阵两路相加 == 固定核设计矩阵（无损分流）",
        np.abs((Xb[:, :k] + Xb[:, k:]) - Xf).max() < 1e-12)
    chk("2b 同一行不会两路同时非零（箱互斥）",
        not np.any((Xb[:, :k] != 0) & (Xb[:, k:] != 0)))

    # 3 §2：在**输入发生时**选箱 —— lag τ 的那一列必须用 bin[t−τ]
    tau, j = 5, 2
    col = Xb[:, k + tau * 6 + j]                       # 箱1 · lag τ · 关节 j
    want = np.zeros(len(e.da))
    want[tau:] = np.where(e.binv[:-tau] == 1, e.da[:-tau, j], 0.0)
    chk("3 lag τ 的列用的是 bin[t−τ] 而非 bin[t]",
        np.abs(col - want).max() < 1e-12)

    # 4 §2.0.1：有效帧与滤波半窗
    chk("4 EDGE ≥ N_LAGS−1 且 ≥ 滤波半窗 20",
        EDGE >= n_lags - 1 and EDGE >= 20, f"EDGE={EDGE}, N_LAGS−1={n_lags-1}")

    # 5 §2.0.1：去均值顺序 ①②③ 可交换（线性），换序不改结果
    import fit_step1_candidate as S1m
    coef, _ = fit(eps["P1"], False, 1e-3, n_lags)
    r = np.asarray(e.gap, float) - predict(e, coef, False, n_lags)
    a1 = S1m.bandpass41(r - r.mean(0, keepdims=True))[EDGE:-EDGE]
    a2 = (S1m.bandpass41(r) - S1m.bandpass41(np.tile(r.mean(0), (len(r), 1))))[EDGE:-EDGE]
    chk("5 去均值与带通可交换（顺序只为可复现）", np.abs(a1 - a2).max() < 1e-9)

    # 6 复现冻结 H：固定核 + mult=1e-3 + 只用 P1 ⇒ 必须逐位等于冻结版
    import hashlib
    cf, _ = fit(eps["P1"], False, 1e-3, n_lags)
    Hf = cf.reshape(n_lags, 6, 3).transpose(0, 2, 1)
    sha = hashlib.sha256(np.ascontiguousarray(Hf, np.float64).tobytes()).hexdigest()
    chk("6 固定核(mult=1e-3, 只用P1) 复现冻结 H 的 sha256",
        sha == "89d118d4b69837f05636c1a73f3aefbe8d8c2a2fe3f35e37ba647a6c0c4164c2",
        sha[:16] + "…")

    # 7 §1：数据角色的种子/幅值确实如预注册所写
    chk("7 P1 八个种子各不相同 · P2 三个未见种子各×2 · P3 复用 P1 的 1000–1003",
        len({e.seed for e in eps["P1"]}) == 8
        and sorted(e.seed for e in eps["P2"]) == [2000, 2000, 2001, 2001, 2002, 2002]
        and sorted(e.seed for e in eps["P3"]) == [1000, 1001, 1002, 1003])

    # 8 两箱在每层都非空（否则该层不足以比较）
    cov = {t: [int(sum((e.binv == b).sum() for e in eps[t])) for b in (0, 1)]
           for t in ("P1", "P2", "P3")}
    chk("8 三层两箱都非空", all(min(v) > 0 for v in cov.values()), str(cov))
    print(f"\n  ⇒ 自测 {sum(ok.values())}/{len(ok)} 通过")
    return all(ok.values())


def main() -> int:
    import gc
    gc.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    import fit_step1_candidate as S1
    from lerobot.common.policies.act import b_correction_deploy as BC
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    n_lags, grid = S1.N_LAGS, S1.RIDGE_GRID

    calib = json.loads(H_JSON.read_text())["calibration"]
    R = np.asarray(calib["R"], np.float64)
    b = np.asarray(calib.get("b", [0, 0, 0]), np.float64)
    eps = load(S1, IdealChunkRenderer(), R, b)
    print(f"载入 " + " · ".join(f"{t} {len(v)} 集" for t, v in sorted(eps.items()))
          + f"   分界 {THR:+.5f} rad · EDGE {EDGE} · 网格 {grid}")

    if not selftest(S1, eps, BC, n_lags):
        print("\n**✗ 自测未全过 —— 不进行拟合**")
        return 1
    if a.selftest:
        return 0

    # ── 选参：P1 内部 leave-one-seed-out，两个模型各自选 ──────────────
    print("\n" + "=" * 72 + "\n选参（P1 内部 leave-one-seed-out；选参绝不看 P2/P3）\n" + "=" * 72)
    sel = {}
    for binned, name in ((False, "固定核"), (True, "分箱核")):
        m, rows, edge = select_multiplier(eps["P1"], binned, n_lags, grid)
        print(f"  {name}: 选中 multiplier = {m:g}"
              + ("   ⚠ **落在网格端点，照实报告、不外延网格**" if edge else ""))
        print("      CV: " + " · ".join(f"{r['mult']:g}→{r['cv']:.6g}" for r in rows))
        sel[name] = {"mult": m, "cv": rows, "at_edge": edge}

    # ── 拟合（只用 P1）+ 评价 ────────────────────────────────────────
    coefs = {n: fit(eps["P1"], n == "分箱核", sel[n]["mult"], n_lags)[0]
             for n in ("固定核", "分箱核")}
    res = {"threshold_rad": THR, "edge": EDGE, "n_lags": n_lags,
           "grid": list(grid), "selection": sel, "gate": GATE, "tiers": {}}

    print("\n" + "=" * 72 + "\n评价（ĝap = H⊛δa；G-2 不进评分）\n" + "=" * 72)
    verdict = {}
    for tier in ("P2", "P3"):
        role = "**主判据**" if tier == "P2" else "**次要判据：跨幅值压力验证**"
        print(f"\n  {tier} —— {role}")
        tv = {}
        for bname, band in SUBBANDS.items():
            ef = by_seed_energy(eps[tier], coefs["固定核"], False, n_lags, band)
            eb = by_seed_energy(eps[tier], coefs["分箱核"], True, n_lags, band)
            ds = {s: (ef[s] - eb[s]) / ef[s] for s in sorted(ef)}
            dbar = float(np.mean(list(ds.values())))
            allpos = all(v > 0 for v in ds.values())
            mark = "  ⇒ **通过**" if (allpos and dbar >= GATE) else ""
            print(f"    [{bname:>8}] d̄ = {dbar*100:+6.2f}%   逐种子 " +
                  " ".join(f"{s}:{v*100:+6.2f}%" for s, v in ds.items()) + mark)
            tv[bname] = {"d_bar": dbar, "d_s": {str(k): v for k, v in ds.items()},
                         "all_positive": allpos, "pass": bool(allpos and dbar >= GATE),
                         "E_fixed": {str(k): v for k, v in ef.items()},
                         "E_binned": {str(k): v for k, v in eb.items()}}
        verdict[tier] = tv["0.5-9Hz"]["pass"]
        res["tiers"][tier] = tv

    # ── 辅助诊断（不参与判决）──────────────────────────────────────
    print("\n  辅助诊断（**不参与判决**）")
    aux = {}
    for tier in ("P2", "P3"):
        row = {}
        for bname, binned in (("固定核", False), ("分箱核", True)):
            num = den = 0.0
            for e in eps[tier]:
                pr = predict(e, coefs[bname], binned, n_lags)
                pr = pr - pr.mean(0, keepdims=True)
                me = np.asarray(e.gap, float) - np.mean(e.gap, 0, keepdims=True)
                num += float((me * pr).sum()); den += float((pr * pr).sum())
            row[bname] = num / den if den else float("nan")
        print(f"    {tier} beta_ols  固定核 {row['固定核']:.3f} · 分箱核 {row['分箱核']:.3f}")
        aux[tier] = row
    print("    ⚠ beta_ols 是**过原点的投影尺度**；真实响应若有两个正交分量而模型只抓到一个，"
          "β 仍可为 1 ⇒ **不能独自判定**。")
    res["aux_beta_ols"] = aux

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n→ {OUT}")

    # ── 判读：按预注册 §4 的四格表，**不许自由发挥** ──────────────────
    print("\n" + "=" * 72 + "\n判读（预注册 §3/§4 事前写死）\n" + "=" * 72)
    p2, p3 = verdict["P2"], verdict["P3"]
    print(f"  P2 主判据（三个 d_s 全正 且 d̄ ≥ 5%）：{'**通过**' if p2 else '**不通过**'}")
    print(f"  P3 压力验证（四个 d_s 全正 且 d̄ ≥ 5%）：{'**通过**' if p3 else '**不通过**'}")
    print("  ⇒ " + {
        (True, True): "支持进入**独立确认**；**仍不直接授权部署**。",
        (True, False): "「**小幅值开发条件下有收益，跨幅值未获支持**」。",
        (False, True): "**不能用 P3 挽救主判据** ⇒ 记为**主判据不通过**。",
        (False, False): "**不通过**。",
    }[(p2, p3)])
    print("\n  ⚠ 不通过**不能否定所有位形依赖**，只表示本轮没有证据支持增加复杂度。")
    print("  ⚠ 通过只证明「**六输入**分箱模型有预测收益」，收益**可能来自 j4** ⇒ "
          "推不出「固定 δa_j4=0 的五关节部署候选也会获益」。")
    print("  ⚠ 本轮评价的是**响应核**，不是 `ω̂ = G₂ + K⊛δa` 这个组合形式。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
