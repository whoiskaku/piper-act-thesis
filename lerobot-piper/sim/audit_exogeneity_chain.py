#!/usr/bin/env python3
"""**折外残差诊断 + 探针数据链路审计**（零机时）。

⚠⚠ 本脚本**不能证明外生性**，也不产出「外生性通过／失败」的结论 ⚠⚠
--------------------------------------------------------------------
判读**只允许**写成：

    「发现 / 未发现 冻结 H 尚未解释的探针相关结构。」

理由（用户 2026-09-20 指出，已核对）：残差

    r_H = gap − H⊛δa = e_base + (Δω_real − H⊛δa)
                        ↑不可观测    ↑H 的响应预测误差

**两项不可分离**。于是：

- 残差与探针**相关** ⇒ 可能是 H 漏了位形依赖，**也可能**是任务基线相关，**区分不了**；
- 残差与探针**不相关** ⇒ **也不能**证明外生性，可能只是检出力不足。

输入—残差相关是**发现模型未解释响应**的工具，不是**证明扰动来源独立**的工具。
闭环链路存在 ≠ 已证明估计有偏。

⚠ 为什么 P1 只作背景：`fit_H` 是（ridge）最小二乘，**拟合残差与设计矩阵近似正交是
代数结果**，不是外生性的证据。所以主诊断放在 **P2/P3（折外）**，P1 只用来展示
「在折内这个量会被压到多小」——它是**标尺，不是证据**。

口径
----
- `H` = 冻结的 `outputs/step1_H/step1_H_v1.npz`（**只由 P1 的 8 集拟合**），本脚本**不重新拟合**。
- 残差空间与 `fit_H` 一致：raw aligned gyro gap，**逐集常数**作为 nuisance 扣除。
  （`--band` 另给 0.5–9 Hz 带通视图作参照。）
- 位形指标 `s[t] = q̄₂[t] + q̄₃[t]`，`q̄` 取**名义指令**（`chunks[].raw`，不含 δa）；
  **分界中位数只由 P1 算**。
- **在输入发生时选箱**：滞后 τ 的输入用 `s[t−τ]` 所属的箱。
- **不把帧当独立样本**：逐集算 → 按 **seed** 取平均 → 跨 seed 汇总。
  P2 只有 3 个独立 seed、P3 只有 4 个 ⇒ 跨 seed 检验**几乎没有功率**，如实报告。
- 多重比较：2 箱 × 6 关节 × 16 滞后 × 3 轴 = **576 个统计量**，
  **不允许挑一个显著点下结论** ⇒ 用**循环移位零分布**校准**族内最大值**。
  移位时 `(s, δa)` **整体同移**（它们同属「t−τ 时刻的输入」），只打断与 `r_H` 的对齐。

用法:
    python -u sim/audit_exogeneity_chain.py [--nperm 500] [--seed 0]
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

H_NPZ = ROOT / "outputs/step1_H/step1_H_v1.npz"
H_JSON = ROOT / "outputs/step1_H/step1_H_v1.json"
FROZEN_H_SHA = "89d118d4b69837f05636c1a73f3aefbe8d8c2a2fe3f35e37ba647a6c0c4164c2"
OUT = ROOT / "outputs/step1_H/audit_exogeneity_chain.json"
JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]
AXES = ["gx", "gy", "gz"]


# ══════════════════════════════════════════════════════════════════════
#  A. 数据链路审计：生成值 → 日志值 → 下发值
# ══════════════════════════════════════════════════════════════════════
def audit_chain(paths_by_tier, S1, BC):
    """核对探针的三个环节，并明确**研究对象是「请求补正」还是「执行后补正」**。"""
    print("\n" + "=" * 74)
    print("A. 探针数据链路审计（生成值 → 日志值 → 下发值）")
    print("=" * 74)
    rows, ok = [], True
    for tier in ("P1", "P2", "P3"):
        for p in paths_by_tier[tier]:
            pay = json.loads(p.read_text())
            summ = pay["summary"]
            seed, mrad = int(summ["probe_seed"]), float(summ["probe_mrad"])
            frames = pay["frames"]
            da_log = np.asarray([f["da"] for f in frames], float)
            a_log = np.asarray([f["a"] for f in frames], float)

            # A1 生成值 vs 日志值 —— 重新生成波形逐帧对拍
            gen = BC.make_probe(seed, mrad)[:len(frames)].astype(np.float64)
            d1 = float(np.abs(gen - da_log).max())

            # A2 日志值 vs 下发值 —— a[:6] − 名义指令 是否恰等于 da
            chunks = sorted(pay["chunks"], key=lambda c: int(c["start"]))
            tot = max(int(c["start"]) + len(c["raw"]) for c in chunks)
            cmd = np.full((tot, 7), np.nan)
            for c in chunks:
                s0 = int(c["start"]); r = np.asarray(c["raw"], float)
                cmd[s0:s0 + len(r)] = r
            n = len(frames)
            d2 = float(np.abs((a_log[:n, :6] - cmd[:n, :6]) - da_log[:n]).max())

            # A3 波形触界（P3 是 P1 的 2× 幅值，理应触界）
            clip = float(np.mean(np.abs(gen) >= BC.EPS_B * 0.999) * 100)
            hash_ok = summ.get("probe_hash") == BC.probe_hash(seed)
            rows.append({"tier": tier, "ep": S1._episode_number(p), "seed": seed,
                         "mrad": mrad, "gen_vs_log": d1, "log_vs_emit": d2,
                         "clip_pct": clip, "hash_ok": hash_ok,
                         "wrapped": int(summ.get("probe_wrapped", 0)),
                         "overrun": int(summ.get("probe_overrun", 0))})
            # ⚠ 门槛不能比 float32 还紧：`a` 以 float32 落盘，在 ~1 rad 量级上
            #   相对精度 2^-24 ⇒ 差值 ~6e-8 rad (6e-5 mrad) 是**序列化舍入**，不是链路问题。
            ok &= (d1 == 0.0) and (d2 < 1e-6) and hash_ok

    for tier in ("P1", "P2", "P3"):
        r = [x for x in rows if x["tier"] == tier]
        print(f"  {tier} ({len(r)} 集, {r[0]['mrad']:g} mrad)  "
              f"生成↔日志 max {max(x['gen_vs_log'] for x in r):.2e} rad · "
              f"日志↔下发 max {max(x['log_vs_emit'] for x in r):.2e} rad · "
              f"波形触界 {r[0]['clip_pct']:.2f}% · "
              f"hash {sum(x['hash_ok'] for x in r)}/{len(r)} · "
              f"wrap/overrun {sum(x['wrapped'] for x in r)}/{sum(x['overrun'] for x in r)}")
    print(f"\n  {'✓' if ok else '**✗**'} 三环节逐帧一致"
          f"（生成的波形 == 日志的 da == 下发动作减名义指令）")
    print("\n  ⚠ 可核对的到此为止。日志有 `state`（编码器读数，mdeg），"
          "但**没有伺服侧的指令回读**")
    print("     ⇒ 「下发之后伺服实际执行了什么」**无法从现有日志验证**。")
    print("     ⇒ **本审计与 H 的研究对象都是「请求补正」δa_req 的响应**，"
          "不是执行后补正的响应。")
    print("     ⇒ 若伺服存在未知的限幅/整形，它被**吸收进 H**，"
          "而不是表现为 δa 的误差。P3 的 2× 幅值线性检验（β 比 1.0134）")
    print("        只能说明该环节在 1.5→3.0 mrad 区间**近似线性**，不能排除静态整形。")
    return rows, ok


# ══════════════════════════════════════════════════════════════════════
#  B. 折外残差诊断
# ══════════════════════════════════════════════════════════════════════
def build(eps_by_tier, S1, H, thr, band=False):
    """逐集算 r_H 与输入流 (s 的箱号, δa)。"""
    out = {}
    for tier, eps in eps_by_tier.items():
        lst = []
        for e in eps:
            gap = np.asarray(e.gap, float)
            pred = S1.causal_conv(H, e.da)
            r = gap - pred
            if band:
                r = S1.bandpass41(r)
            r = r - r.mean(0, keepdims=True)          # 与 fit_H 的逐集常数 nuisance 一致
            s = e.x_cmd[:, 1] + e.x_cmd[:, 2]         # q̄2 + q̄3（名义指令, rad）
            lst.append({"ep": e.ep, "seed": e.seed, "r": r,
                        "da": np.asarray(e.da, float),
                        "bin": (s >= thr).astype(np.int8), "s": s})
        out[tier] = lst
    return out


def cosines(rec, n_lags, shift=0, shift_bin=0):
    """返回 `(cos, slope)`，形状均为 (3, n_lags, 6 关节, 3 轴)。

    第 0 维：`0`/`1` = 两个箱，`2` = **不分箱**（全体帧）。
    - `cos`  = 过原点余弦，用于族内显著性
    - `slope`= `Σx·r / Σx²`，**有量纲**，用于幅值标度检验（见 `amplitude_probe`）

    ⚠ **两个零分布，对应两个不同的零假设** —— 用错会得出相反结论：

    | 参数 | 移动什么 | 零假设 | 适用统计量 |
    |---|---|---|---|
    | `shift` | `(bin, da)` **联动** | δa 与 `r_H` **毫无关系** | 不分箱 / 分箱的绝对相关 |
    | `shift_bin` | **只移 bin** | 有 LTI 关系但**与位形无关** | **箱1−箱0 的对比** |

    为什么对比项必须用后者：折外的 `r_H` 里**本来就有** `(K−H)⊛δa` 这一 LTI 项
    （ridge 收缩 + 估计误差保证它非零）。联动移位把这一项一并打掉 ⇒ 零分布偏窄 ⇒
    **把「H 整体不完美」误判成「位形依赖」**。只移 bin 则保住 `(δa, r_H)` 的对齐，
    该 LTI 项在两箱中同等出现、在对比里抵消，零分布正好刻画它的有限样本涨落。
    """
    r, da, bb = rec["r"], rec["da"], rec["bin"]
    T = len(r)
    if shift:
        da = np.roll(da, shift, axis=0)
        bb = np.roll(bb, shift)
    if shift_bin:
        bb = np.roll(bb, shift_bin)
    den_r = np.einsum("tc,tc->c", r, r)               # (3,)
    cos = np.zeros((3, n_lags, 6, 3)); slp = np.zeros((3, n_lags, 6, 3))
    for b in (0, 1, 2):
        Z = da if b == 2 else (bb == b)[:, None] * da
        for tau in range(n_lags):
            m = T - tau
            if m <= 1:
                break
            zz = Z[:m]
            zz = zz - zz.mean(0, keepdims=True)       # 与 fit_H 的中心化一致
            num = zz.T @ r[tau:]                      # (6,3)
            den_z = np.einsum("tj,tj->j", zz, zz)     # (6,)
            cos[b, tau] = num / np.sqrt(np.maximum(den_z[:, None] * den_r[None, :], 1e-300))
            slp[b, tau] = num / np.maximum(den_z[:, None], 1e-300)
    return cos, slp


def by_seed(recs, n_lags, shifts=None, shifts_bin=None):
    """逐集算 → **按 seed 平均** → 跨 seed 平均。返回 (跨seed均值cos, 各seed cos, seeds, 跨seed均值slope)。"""
    pc, ps = {}, {}
    for i, rec in enumerate(recs):
        c, s = cosines(rec, n_lags, shifts[i] if shifts is not None else 0,
                       shifts_bin[i] if shifts_bin is not None else 0)
        pc.setdefault(rec["seed"], []).append(c)
        ps.setdefault(rec["seed"], []).append(s)
    seeds = sorted(pc)
    cm = np.stack([np.mean(pc[s], 0) for s in seeds])   # (n_seed, 3, L, 6, 3)
    sm = np.stack([np.mean(ps[s], 0) for s in seeds])
    return cm.mean(0), cm, seeds, sm.mean(0)


def coverage(recs, thr):
    """逐箱报告跨集/跨种子覆盖与激励 —— 覆盖不足时这个诊断本身不成立。"""
    rows = []
    for b in (0, 1):
        fr, eps_hit, seeds_hit, en = 0, 0, set(), np.zeros(6)
        for rec in recs:
            m = rec["bin"] == b
            if m.sum() > 0:
                eps_hit += 1
                seeds_hit.add(rec["seed"])
            fr += int(m.sum())
            en += (rec["da"][m] ** 2).sum(0)
        rows.append({"bin": b, "frames": fr, "episodes": eps_hit,
                     "seeds": sorted(seeds_hit),
                     "rms_mrad": (np.sqrt(en / max(fr, 1)) * 1e3).round(3).tolist()})
    return rows


def main() -> int:
    import gc
    gc.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--nperm", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    import fit_step1_candidate as S1
    from lerobot.common.policies.act import b_correction_deploy as BC
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer

    H = np.ascontiguousarray(np.load(H_NPZ)["H"], np.float64)
    sha = hashlib.sha256(H.tobytes()).hexdigest()
    assert sha == FROZEN_H_SHA, f"H 不是冻结版: {sha}"
    n_lags = H.shape[0]
    print(f"冻结 H  shape {H.shape}  sha {sha[:16]}…  （**只由 P1 的 8 集拟合**，本脚本不重拟合）")

    logs = S1.discover_logs(ROOT / "outputs/ab4", ("P1", "P2", "P3"))
    calib = json.loads(H_JSON.read_text())["calibration"]
    R = np.asarray(calib["R"], np.float64); b = np.asarray(calib.get("b", [0, 0, 0]), np.float64)
    rend = IdealChunkRenderer()
    eps = {t: [S1.load_episode(p, t, R, b, rend, 550) for p in ps] for t, ps in logs.items()}
    S1.validate_protocol(eps)

    # 名义指令（chunks[].raw，不含 δa）补挂到 Episode 上
    for t, ps in logs.items():
        for e, p in zip(eps[t], ps):
            pay = json.loads(p.read_text())
            cmd, _, _ = S1._render_logged_chunks(
                sorted(pay["chunks"], key=lambda c: int(c["start"])), rend, len(pay["frames"]))
            e.x_cmd = cmd[:len(e.da), :6]

    chain_rows, chain_ok = audit_chain(logs, S1, BC)

    # ── 分界：**只由 P1 的名义指令**算中位数 ─────────────────────────
    s_p1 = np.concatenate([e.x_cmd[:, 1] + e.x_cmd[:, 2] for e in eps["P1"]])
    thr = float(np.median(s_p1))
    print("\n" + "=" * 74)
    print("B. 折外残差诊断")
    print("=" * 74)
    print(f"  分箱 s = q̄2 + q̄3，分界 = **仅 P1 名义指令**的中位数 = {thr:+.5f} rad"
          f"   (P1 范围 {s_p1.min():+.3f} ~ {s_p1.max():+.3f})")

    res = {"H_sha": sha, "threshold_rad": thr, "chain": chain_rows,
           "chain_ok": bool(chain_ok), "nperm": a.nperm, "views": {}}

    for band in (False, True):
        tag = "带通 0.5–9 Hz" if band else "拟合空间 raw"
        D = build(eps, S1, H, thr, band=band)
        print(f"\n  ── 视图：{tag} " + "─" * (52 - len(tag)))

        view = {}
        for tier in ("P2", "P3", "P1"):
            recs = D[tier]
            role = "**折外主诊断**" if tier in ("P2", "P3") else "折内背景（代数上必然很小，是标尺不是证据）"
            cov = coverage(recs, thr)
            obs, per_seed, seeds, slope = by_seed(recs, n_lags)
            # ⚠ **分箱对比**才是指向位形调度的量。两箱都相关但**相关得一样**，
            #   只说明 H 在折外整体不完美，**不**构成位形依赖的证据。
            contrast = obs[1] - obs[0]                          # (L,6,3)

            # 两个零分布，见 `cosines` 的表
            def _within(ps):
                """(n_seed,L,6,3) 的对比图样 → 平均两两余弦。"""
                v = [float((ps[i] * ps[j]).sum() /
                           max(np.linalg.norm(ps[i]) * np.linalg.norm(ps[j]), 1e-300))
                     for i in range(len(ps)) for j in range(i + 1, len(ps))]
                return float(np.mean(v)) if v else float("nan"), len(v)

            null_max = np.empty(a.nperm); null_unc = np.empty(a.nperm)
            null_con = np.empty(a.nperm); null_wit = np.empty(a.nperm)
            for k in range(a.nperm):
                sh = [int(rng.integers(50, len(r["r"]) - 50)) for r in recs]
                nm, _, _, _ = by_seed(recs, n_lags, shifts=sh)
                null_max[k] = np.abs(nm[:2]).max()
                null_unc[k] = np.abs(nm[2]).max()
                # 对比项用**只移 bin** 的零分布（保住 (δa, r_H) 的 LTI 关系）
                shb = [int(rng.integers(50, len(r["r"]) - 50)) for r in recs]
                nb, nbs, _, _ = by_seed(recs, n_lags, shifts_bin=shb)
                null_con[k] = np.abs(nb[1] - nb[0]).max()
                null_wit[k] = _within(nbs[:, 1] - nbs[:, 0])[0]
            omax = float(np.abs(obs[:2]).max())
            ounc = float(np.abs(obs[2]).max())
            ocon = float(np.abs(contrast).max())
            pfw = float((null_max >= omax).mean())
            punc = float((null_unc >= ounc).mean())
            pcon = float((null_con >= ocon).mean())
            idx = np.unravel_index(np.argmax(np.abs(obs[:2])), obs[:2].shape)
            icon = np.unravel_index(np.argmax(np.abs(contrast)), contrast.shape)

            print(f"\n  {tier}（{len(recs)} 集 / {len(seeds)} 个独立 seed）—— {role}")
            for c in cov:
                print(f"      箱{c['bin']}: {c['frames']:5d} 帧 · {c['episodes']}/{len(recs)} 集 · "
                      f"seed {c['seeds']} · δa RMS(mrad) {c['rms_mrad']}")
            if any(c["episodes"] < len(recs) or len(c["seeds"]) < len(seeds) for c in cov):
                print("      ⚠ **该箱缺少跨集/跨种子覆盖 ⇒ 不足以比较**（不事后重切）")
            print(f"      [不分箱] 最大 |cos| = {ounc:.4f} @ {JOINTS[np.unravel_index(np.argmax(np.abs(obs[2])), obs[2].shape)[1]]}"
                  f"  · 零分布 95% {np.quantile(null_unc, .95):.4f}  ⇒ 族内 p = {punc:.3f}")
            print(f"      [分箱  ] 最大 |cos| = {omax:.4f} @ 箱{idx[0]} {JOINTS[idx[2]]} "
                  f"lag{idx[1]} {AXES[idx[3]]}  · 零分布 95% {np.quantile(null_max, .95):.4f} "
                  f" ⇒ 族内 p = {pfw:.3f}   (2×6×16×3 = 576 个统计量)")
            print(f"      [箱1−箱0] 最大 |Δcos| = {ocon:.4f} @ {JOINTS[icon[1]]} lag{icon[0]} "
                  f"{AXES[icon[2]]}  · 零分布 95% {np.quantile(null_con, .95):.4f} "
                  f" ⇒ 族内 p = {pcon:.3f}   （零分布=**只移 bin**）")
            print(f"        ↑ **只有这一行指向位形调度**；前两行显著而对比不显著 ⇒ "
                  f"只是 H 整体不完美，与位形无关")
            # 跨 seed 一致性：最强那一格在各 seed 上的值
            vals = per_seed[(slice(None),) + idx]
            vcon = per_seed[:, 1][(slice(None),) + icon] - per_seed[:, 0][(slice(None),) + icon]
            print(f"      逐 seed（分箱最强格） {np.round(vals, 4).tolist()} ⇒ "
                  f"{'同号' if np.all(np.sign(vals) == np.sign(vals[0])) else '**异号**'}"
                  f"   ·（对比最强格） {np.round(vcon, 4).tolist()} ⇒ "
                  f"{'同号' if np.all(np.sign(vcon) == np.sign(vcon[0])) else '**异号**'}")
            # 层**内**跨 seed 的图样一致性 —— 跨层 cos 必须与它对照才读得懂：
            # 层内也只有 ~0.1 ⇒ 图样整体就是噪声；层内高而跨层低 ⇒ 结构不可迁移。
            within, npair = _within(per_seed[:, 1] - per_seed[:, 0])
            pwit = float((null_wit >= within).mean())
            print(f"      对比图样的**层内**跨 seed 一致性：平均两两 cos = {within:+.3f}"
                  f"  ({npair} 对) · 零分布(只移bin) 中位 {np.median(null_wit):+.3f} "
                  f"95% {np.quantile(null_wit, .95):+.3f}  ⇒ p = {pwit:.3f}")
            view[tier] = {"n_ep": len(recs), "seeds": seeds, "coverage": cov,
                          "max_abs_cos": omax, "argmax": [int(i) for i in idx],
                          "null_q95": float(np.quantile(null_max, .95)),
                          "p_familywise": pfw,
                          "contrast_within_tier_cos": within, "contrast_within_p": pwit,
                          "unconditional": {"max_abs_cos": ounc, "p_familywise": punc,
                                            "null_q95": float(np.quantile(null_unc, .95))},
                          "contrast": {"max_abs_dcos": ocon, "p_familywise": pcon,
                                       "argmax": [int(i) for i in icon],
                                       "null_q95": float(np.quantile(null_con, .95)),
                                       "per_seed_at_argmax": [float(v) for v in vcon]},
                          "per_seed_at_argmax": [float(v) for v in vals],
                          "_slope": slope, "_contrast": contrast}

        # ── 幅值标度：**机制线索，不是判决** ────────────────────────────
        # r_H = e_base + u，u = 响应预测误差（对 δa 线性）。斜率 = Σx·r / Σx²：
        #   · u 驱动的部分 ∝ δa² / δa² ⇒ **与幅值无关**
        #   · e_base 驱动的部分 ∝ δa¹ / δa² ⇒ **与幅值成反比**
        # P2(1.5 mrad) 与 P3(3.0 mrad) **都在折外**，故可比（种子不同，但带限白噪声
        # 的不同实现在统计上可交换）。⚠ 两者是不同集的不同任务执行，e_base 本身也不同。
        s2, s3 = view["P2"]["_slope"], view["P3"]["_slope"]
        c2, c3 = view["P2"]["_contrast"], view["P3"]["_contrast"]
        for k in ("P1", "P2", "P3"):
            view[k].pop("_slope"); view[k].pop("_contrast")
        num = float((s2 * s3).sum()); den = float((s2 * s2).sum())
        proj = num / den if den else float("nan")
        ratio = float(np.linalg.norm(s3) / max(np.linalg.norm(s2), 1e-300))
        scos = num / max(np.linalg.norm(s2) * np.linalg.norm(s3), 1e-300)
        print(f"\n    幅值标度（P3 3.0 mrad 对 P2 1.5 mrad，均折外）："
              f"斜率投影尺度 **{proj:.3f}** · 范数比 {ratio:.3f} · 形状 cos {scos:.3f}")
        print(f"      参照：≈1.00 ⇒ 相关来自**对 δa 线性的响应预测误差**；"
              f"≈0.50 ⇒ 来自**固定大小的任务基线**")
        if abs(scos) < 0.5:
            print(f"      ⚠ **形状 cos 只有 {scos:.3f} ⇒ 两档幅值的残差斜率图样本身就对不上，"
                  f"这个幅值检验在此不成立**（它预设同一套底层结构）。")
        print(f"      ⚠ 这是**机制线索，不是判决** —— 两个来源可以并存，"
              f"且 P2/P3 是不同集、e_base 本身也不同。")

        # ── 跨层可复现性：P2 与 P3 的**对比图样**是否一致 ────────────────
        # 若两箱差异是稳定的位形调度核属性，它应在两个折外层上给出**同一张图样**。
        ccos = float((c2 * c3).sum() / max(np.linalg.norm(c2) * np.linalg.norm(c3), 1e-300))
        print(f"\n    对比图样的跨层可复现性（P2 vs P3，{c2.size} 格）：cos = **{ccos:+.3f}**")
        print(f"      P2 最强格 {JOINTS[np.unravel_index(np.argmax(np.abs(c2)), c2.shape)[1]]}"
              f" lag{np.unravel_index(np.argmax(np.abs(c2)), c2.shape)[0]}"
              f" {AXES[np.unravel_index(np.argmax(np.abs(c2)), c2.shape)[2]]}"
              f"  ·  P3 最强格 {JOINTS[np.unravel_index(np.argmax(np.abs(c3)), c3.shape)[1]]}"
              f" lag{np.unravel_index(np.argmax(np.abs(c3)), c3.shape)[0]}"
              f" {AXES[np.unravel_index(np.argmax(np.abs(c3)), c3.shape)[2]]}")
        view["amplitude_probe"] = {"proj_scale_P3_on_P2": proj, "norm_ratio": ratio,
                                   "shape_cos": scos}
        view["contrast_reproducibility_P2_P3"] = ccos
        res["views"]["band" if band else "raw"] = view

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n→ {OUT}")
    print("\n" + "=" * 74)
    print("判读（只允许这一种措辞）")
    print("=" * 74)
    for band in (False, True):
        v = res["views"]["band" if band else "raw"]
        tag = "带通 0.5–9 Hz" if band else "拟合空间 raw"
        hit = [t for t in ("P2", "P3") if v[t]["p_familywise"] < 0.05]
        hcon = [t for t in ("P2", "P3") if v[t]["contrast"]["p_familywise"] < 0.05]
        print(f"  [{tag}]")
        print(f"    · " + ("**发现**" if hit else "**未发现**") +
              " 冻结 H 尚未解释的探针相关结构"
              f"（折外 P2 p={v['P2']['p_familywise']:.3f} · P3 p={v['P3']['p_familywise']:.3f}）")
        print(f"    · " + ("**发现**" if hcon else "**未发现**") +
              " 该结构在预定两箱之间**有差异**"
              f"（折外 P2 p={v['P2']['contrast']['p_familywise']:.3f} · "
              f"P3 p={v['P3']['contrast']['p_familywise']:.3f}）")
    print("\n  ⚠ 以上**不是**「外生性通过／失败」。r_H 同时含 e_base 与 H 的响应预测误差，"
          "两者不可分离；")
    print("     不显著也可能只是检出力不足（P2 仅 3 个独立 seed、P3 仅 4 个）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
