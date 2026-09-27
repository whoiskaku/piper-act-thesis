#!/usr/bin/env python3
"""会话均衡采样 —— 严格按 `reports/PREREG_FABS_SESSION_BALANCE.md`（**已冻结**）。

**只检验 H_bal（训练会话不均衡是否影响跨会话泛化）。不换网络、不进 `F_inc`、不训 π。**

设计（预注册 §1–§2）
--------------------
- **两臂走这一份代码**：`--sampler episode`（逐集均匀，对照）/ `session`（会话均衡）。
- **网络 / 输入 / 步数 / 口径 / 留一会话评价 / `G-2` 全部复用 `train_fabs_g0.py`** ——
  **不复制、不改写**那份归档脚本（它的自测 11/11 与阶段 A 结论保持有效）。
- **每个「种子 × 留出会话」两臂共享**：初始权重、训练数据、归一化统计、训练设置。
  **唯一差异 = 抽样方式。**

保证「只有采样方式不同」的三条实现（自测逐条验证）
--------------------------------------------------
1. `_build()` 里 **`torch.manual_seed(seed)` 之后模型是第一个消费 torch RNG 的对象**
   ⇒ 两臂初始权重**逐位相同**。
2. 抽样用**独立的** `np.random.default_rng(seed)`，训练循环中**没有别的随机源**
   （无 dropout / 无增广，Adam 确定）。
3. 归一化统计只由 `tr` 决定（同折同数据）⇒ 两臂**逐位相同**。
⚠ 另置 `cudnn.deterministic=True, benchmark=False`，使上述在**数值上**成立。
   **该设置与上一轮归档运行不同** ⇒ 臂 E `seed=0` 与历史值之差属**复现差异**
   （预注册 §5：**不得**称为运行噪声的「下界」）。

用法:
    python -u sim/train_fabs_balance.py --self-test
    python -u sim/train_fabs_balance.py
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "sim/step4_gvnext"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train_fabs_g0 as A          # **复用**：features / FAbs / err_energy / bp_kernel /
                                   # load_all / g2_score / SESSIONS / EDGE / STEPS / LR / G0_GATE

OUT = ROOT / "outputs/fabs_balance"
SEEDS = (0, 1, 2)                  # **事前固定，不得追加、不得挑选**
ARMS = ("episode", "session")
GUARD_PP = 0.05                    # 护栏 5 个百分点（基准 = **本轮** R_E）


# ── 唯一的改动：抽样器 ────────────────────────────────────────────────────
def make_sampler(tr, arm, seed):
    """臂 E = 逐集均匀；臂 S = 先均匀抽会话、再从该会话抽一集。"""
    rng = np.random.default_rng(seed)
    if arm == "episode":
        n = len(tr)
        return lambda: tr[int(rng.integers(n))]
    if arm != "session":
        raise ValueError(f"未知的 sampler: {arm}")
    sess_list = sorted({e.sess for e in tr})          # **固定排序 ⇒ 可复现**
    by = {s: [e for e in tr if e.sess == s] for s in sess_list}
    ns = len(sess_list)

    def draw():
        s = sess_list[int(rng.integers(ns))]
        g = by[s]
        return g[int(rng.integers(len(g)))]
    return draw


def _build(seed, dev):
    """**先定种子再建模** ⇒ 模型是第一个消费 torch RNG 的对象 ⇒ 两臂初始权重逐位相同。"""
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    return A.FAbs(dev)


def norm_stats(tr):
    """归一化统计**只用训练会话**；与臂、种子无关 ⇒ 两臂逐位相同。"""
    X = np.concatenate([e.x for e in tr])
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def run_fold(tr, te, K, dev, arm, seed, steps=A.STEPS, log=None):
    """训一折。**固定取第 `steps` 步权重；留出只在最后评分一次。**"""
    import torch
    m = _build(seed, dev)                     # ① 初始权重：只由 seed 决定
    draw = make_sampler(tr, arm, seed)        # ② 唯一的臂间差异
    mu, sd = norm_stats(tr)                   # ③ 归一化：只由 tr 决定
    Kt = torch.tensor(K, device=dev).view(1, 1, -1).repeat(3, 1, 1)

    def band(r):
        return torch.nn.functional.conv1d(r.transpose(1, 2), Kt, padding=20, groups=3
                                          ).transpose(1, 2)

    opt = torch.optim.Adam(m.net.parameters(), lr=A.LR)
    t0 = time.time()
    for it in range(steps):
        e = draw()
        x = torch.tensor((e.x - mu) / sd, device=dev).T[None]
        y = torch.tensor(e.y, device=dev)[None]
        r = y - m(x)
        r = r - r.mean(1, keepdim=True)                       # 逐集常数
        v = band(r)[:, A.EDGE:-A.EDGE]
        loss = (v ** 2).sum() / (3 * v.shape[1])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.net.parameters(), 1.0); opt.step()
        if log and (it + 1) % 1500 == 0:
            log(f"        step {it+1}/{steps}  loss {loss.item():.6g}  {time.time()-t0:.0f}s")
    m.net.eval(); num = den = 0.0; per = []
    with torch.no_grad():
        for e in te:
            x = torch.tensor((e.x - mu) / sd, device=dev).T[None]
            p = m(x)[0].cpu().numpy()
            a_, _ = A.err_energy(e.y - p, K); b_, _ = A.err_energy(e.y, K)
            num += a_; den += b_
            per.append({"ep": e.path, "num": a_, "den": b_})
    return {"redu": 1.0 - num / den, "num": num, "den": den,
            "episodes": per, "sec": time.time() - t0}


# ── 判据（预注册 §3）──────────────────────────────────────────────────────
def verdict(rbar):
    """`rbar[arm][sess]` = 该会话的**3 种子均值**。⚠ 护栏基准 = **本轮** `R_E`。"""
    M, R = {}, {}
    for a in ARMS:
        ss = sorted(rbar[a])
        M[a] = float(min(rbar[a][s] for s in ss))
        R[a] = float(np.mean([rbar[a][s] for s in ss]))
    dM = M["session"] - M["episode"]
    g0 = bool(M["session"] > 0 and R["session"] >= A.G0_GATE)
    guard = bool(R["session"] >= R["episode"] - GUARD_PP)      # ← **本轮 R_E**
    return {"M": M, "R": R, "dM": dM, "g0_session": g0, "guard": guard,
            "advance": bool(dM > 0 and g0 and guard)}


def phrase(v):
    """预注册 §3.1：两种情况的**冻结措辞**。"""
    out = []
    if v["dM"] > 0 and not v["g0_session"] and v["M"]["session"] <= 0:
        out.append("最差折**有改善，未通过 G0**（**不得**写成「未测到改善」）")
    if v["dM"] <= 0:
        out.append("**在本预算、本采样方式下未测到改善**"
                   "（**不得**写成「会话不均衡无影响」，也不得据此判 ab2 分布外）")
    if v["M"]["episode"] > 0 and v["M"]["session"] > 0:
        out.append("**两臂都全正 ⇒ 不得说「均衡采样修复了原来的失败」**，"
                   "只能报告本轮两臂差异")
    if v["g0_session"]:
        out.append("臂 S 通过 G0 ⇒ **必须记为「第 2 次配置尝试」并披露多重性**，"
                   "**不得**改写上一轮的「不通过」")
    return out


# ── 自测 ─────────────────────────────────────────────────────────────────
def self_test(eps, K, dev):
    print("\n" + "=" * 78 + "\n自测\n" + "=" * 78)
    ok = {}

    def chk(k, c, d=""):
        ok[k] = bool(c); print(f"  {'✓' if c else '**✗**'} {k}{('  ' + d) if d else ''}")

    byS = {}
    for e in eps:
        byS.setdefault(e.sess, []).append(e)
    hold = "ab2/A"
    tr = [e for e in eps if e.sess != hold]
    n_by = {s: len([e for e in tr if e.sess == s]) for s in sorted({e.sess for e in tr})}

    # ① 抽样频率：臂 E → 集数占比；臂 S → 1/K
    NS = 200000
    for arm, exp in (("episode", {s: n / len(tr) for s, n in n_by.items()}),
                     ("session", {s: 1 / len(n_by) for s in n_by})):
        d = make_sampler(tr, arm, 0); cnt = {s: 0 for s in n_by}
        for _ in range(NS):
            cnt[d().sess] += 1
        err = max(abs(cnt[s] / NS - exp[s]) for s in n_by)
        chk(f"1 抽样频率[{arm}] 收敛到事前权重表", err < 0.005,
            " · ".join(f"{s.split('/')[0]} {cnt[s]/NS*100:.1f}%(期望{exp[s]*100:.1f})"
                       for s in sorted(n_by)) + f"  max|Δ|={err:.4f}")

    # ② 臂 E 与归档脚本 train_fabs_g0 的抽样序列**逐位一致**
    d = make_sampler(tr, "episode", 7)
    mine = [id(d()) for _ in range(500)]
    rng = np.random.default_rng(7)
    theirs = [id(tr[int(rng.integers(len(tr)))]) for _ in range(500)]
    chk("2 **臂 E 的抽样序列 == 归档脚本 `train_fabs_g0` 的抽样序列**", mine == theirs,
        f"{len(mine)} 步逐位一致")

    # ③ 两臂初始权重逐位相同
    import torch
    for sd_ in SEEDS:
        m1, m2 = _build(sd_, dev), _build(sd_, dev)
        dmax = max((p1 - p2).abs().max().item()
                   for p1, p2 in zip(m1.net.parameters(), m2.net.parameters()))
        if dmax != 0.0:
            break
    chk("3 **同 seed ⇒ 初始权重逐位相同**（模型是第一个消费 torch RNG 的对象）",
        dmax == 0.0, f"seeds {SEEDS} · max|Δ| = {dmax}")
    m3 = _build(SEEDS[0] + 100, dev)
    dif = max((p1 - p3).abs().max().item()
              for p1, p3 in zip(_build(SEEDS[0], dev).net.parameters(), m3.net.parameters()))
    chk("3b 不同 seed ⇒ 初始权重确实不同（上一条不是恒真）", dif > 0, f"max|Δ| = {dif:.3e}")

    # ④ 归一化统计与臂、种子无关
    mu1, sd1 = norm_stats(tr); mu2, sd2 = norm_stats(tr)
    chk("4 **归一化统计只由 tr 决定 ⇒ 两臂逐位相同**",
        np.array_equal(mu1, mu2) and np.array_equal(sd1, sd2))
    chk("4b 归一化**不含留出会话**（与全量不同）",
        np.abs(mu1 - np.concatenate([e.x for e in eps]).mean(0)).max() > 0)

    # ⑤ 臂 S 只抽训练会话，且每个训练会话都抽得到
    d = make_sampler(tr, "session", 1)
    seen = {d().sess for _ in range(5000)}
    chk("5 **臂 S 覆盖全部训练会话且不含留出**",
        seen == set(n_by) and hold not in seen, f"{sorted(seen)} · 留出 {hold}")

    # ⑥ 口径函数确实复用归档脚本（同一对象）
    chk("6 **口径/网络/常量复用 `train_fabs_g0`**（同一对象，未复制）",
        run_fold.__globals__["A"].err_energy is A.err_energy and A.EDGE == 20
        and A.STEPS == 3000 and A.G0_GATE == 0.30,
        f"EDGE={A.EDGE} STEPS={A.STEPS} GATE={A.G0_GATE} LR={A.LR}")

    # ⑦ 判据函数单元测试
    def mk(e_, s_):
        return {"episode": dict(zip("vwxyz", e_)), "session": dict(zip("vwxyz", s_))}
    v = verdict(mk([-.22, .63, .58, .66, .39], [-.05, .60, .55, .60, .42]))
    chk("7 ΔM>0 但 M_S<0 ⇒ 不通过 G0、不推进",
        abs(v["dM"] - 0.17) < 1e-12 and not v["g0_session"] and not v["advance"],
        f"ΔM={v['dM']:+.4f} R_S={v['R']['session']:.4f}")
    chk("7b 该情形的措辞 = 「有改善，未通过 G0」",
        any("未通过 G0" in p for p in phrase(v)))
    v2 = verdict(mk([.10, .40, .40, .40, .40], [.20, .45, .45, .45, .45]))
    chk("7c 三项全满足 ⇒ 推进", v2["advance"] and v2["g0_session"] and v2["guard"])
    # **判别用例**：正确护栏（本轮 R_E）必须拦住，而误用历史基准则会放行
    HIST = 0.4070                      # 上一轮归档均值，**仅用于本断言**
    v3 = verdict(mk([.10, .725, .725, .725, .725], [.20, .45, .45, .45, .45]))
    chk("7d **护栏基准 = 本轮 R_E**（R_E=60%,R_S=40% ⇒ 拦住；"
        "误用历史基准则会放行 ⇒ 本例可判别）",
        (not v3["guard"]) and (not v3["advance"])
        and v3["dM"] > 0 and v3["g0_session"]                 # 只被护栏拦住
        and v3["R"]["session"] >= HIST - GUARD_PP,            # 历史基准会放行
        f"R_E={v3['R']['episode']*100:.2f}% R_S={v3['R']['session']*100:.2f}% "
        f"ΔM={v3['dM']*100:+.2f} G0={v3['g0_session']} 护栏={v3['guard']}")
    v4 = verdict(mk([.30, .40, .40, .40, .40], [.25, .45, .45, .45, .45]))
    chk("7e 两臂都全正 ⇒ 措辞禁止「修复了原来的失败」",
        any("修复" in p for p in phrase(v4)) and v4["dM"] < 0)
    chk("7f ΔM≤0 的措辞 = 「未测到改善」且禁止推广",
        any("未测到改善" in p for p in phrase(v4)))

    # ⑧ 判据函数内不得出现任何历史常数；护栏必须引用本轮 R["episode"]
    import inspect
    vsrc = inspect.getsource(verdict)
    chk("8 **`verdict()` 内无历史常数，护栏引用本轮 `R[\"episode\"]`**",
        "0.4070" not in vsrc and "40.70" not in vsrc
        and 'R["session"] >= R["episode"] - GUARD_PP' in vsrc)

    chk("9 种子事前固定为 0/1/2 且两臂同种子", SEEDS == (0, 1, 2) and len(ARMS) == 2,
        f"{SEEDS} × {ARMS} × 5 折 = {len(SEEDS)*len(ARMS)*5} 次训练")
    print(f"\n  ⇒ 自测 {sum(ok.values())}/{len(ok)} 通过")
    return all(ok.values())


def main() -> int:
    import gc; gc.disable()
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--steps", type=int, default=A.STEPS)
    a = ap.parse_args()
    import fit_step1_candidate as S1
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    cal = json.loads((ROOT / "outputs/step1_H/step1_H_v1.json").read_text())["calibration"]
    R = np.asarray(cal["R"], np.float64); b = np.asarray(cal.get("b", [0, 0, 0]), np.float64)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    eps = A.load_all(S1, IdealChunkRenderer(), R, b)
    K = A.bp_kernel(S1)
    byS = {}
    for e in eps:
        byS.setdefault(e.sess, []).append(e)
    print(f"裸集 {len(eps)} 集 / {len(byS)} 会话  " +
          " · ".join(f"{k} {len(v)}" for k, v in byS.items()) + f"   device={dev}")
    if a.self_test:
        return 0 if self_test(eps, K, dev) else 1

    OUT.mkdir(parents=True, exist_ok=True)
    part = OUT / "fabs_balance_partial.json"
    print("\n" + "=" * 78 +
          f"\n3 种子 × 2 臂 × 5 折 = {len(SEEDS)*len(ARMS)*len(byS)} 次训练"
          f"（步数 {a.steps}；每折固定取第 {a.steps} 步；留出只最后评分一次）\n" + "=" * 78)

    g2 = {}                       # `G-2` 冻结 ⇒ 与臂、种子无关，每折只算一次
    for s in byS:
        g2[s] = A.g2_score(byS[s], K, R, b, S1)
    print("  G-2（冻结，同口径）  " + " · ".join(f"{s} {g2[s]*100:+.2f}%" for s in byS))

    t_all = time.time(); runs = []
    for seed in SEEDS:
        for arm in ARMS:
            for s in byS:
                tr = [e for e in eps if e.sess != s]
                r = run_fold(tr, byS[s], K, dev, arm, seed, steps=a.steps,
                             log=lambda m: print(m))
                r.update({"arm": arm, "seed": seed, "session": s,
                          "n_te": len(byS[s]), "n_tr": len(tr)})
                runs.append(r)
                print(f"  seed {seed} · 臂 {arm:<8} · 留出 {s:<11} "
                      f"**redu = {r['redu']*100:+7.2f}%**   [{r['sec']:.0f}s]")
                part.write_text(json.dumps({"runs": runs}, indent=2, ensure_ascii=False))

    # ── 汇总（预注册 §3/§4）────────────────────────────────────────────
    sess = sorted(byS)
    rbar = {a_: {s: float(np.mean([r["redu"] for r in runs
                                   if r["arm"] == a_ and r["session"] == s]))
                 for s in sess} for a_ in ARMS}
    v = verdict(rbar)
    per_seed = {}
    for seed in SEEDS:
        per_seed[seed] = {}
        for a_ in ARMS:
            rr = {s: [r["redu"] for r in runs
                      if r["arm"] == a_ and r["seed"] == seed and r["session"] == s][0]
                  for s in sess}
            per_seed[seed][a_] = {
                "folds": rr, "mean": float(np.mean([rr[s] for s in sess])),
                "min": float(min(rr.values())),
                "g0": bool(min(rr.values()) > 0 and np.mean([rr[s] for s in sess]) >= A.G0_GATE)}
    hrs = (time.time() - t_all) / 3600

    print("\n" + "=" * 78 + "\n③ 逐种子（另报）\n" + "=" * 78)
    for seed in SEEDS:
        for a_ in ARMS:
            d_ = per_seed[seed][a_]
            print(f"  seed {seed} 臂 {a_:<8} " +
                  " ".join(f"{s.split('/')[0]}{d_['folds'][s]*100:+7.2f}" for s in sess) +
                  f"  │ 均值{d_['mean']*100:+7.2f}%  最差{d_['min']*100:+7.2f}%  "
                  f"G0 {'通过' if d_['g0'] else '**不通过**'}")

    print("\n" + "=" * 78 + "\n② 主判据（先对每会话取 3 种子均值）\n" + "=" * 78)
    print(f"  {'会话':<12}{'臂 E 均衡前':>14}{'臂 S 均衡后':>14}{'Δ':>10}{'G-2':>10}")
    for s in sess:
        print(f"  {s:<12}{rbar['episode'][s]*100:>13.2f}%{rbar['session'][s]*100:>13.2f}%"
              f"{(rbar['session'][s]-rbar['episode'][s])*100:>+9.2f}{g2[s]*100:>9.2f}%")
    print(f"  {'M（最差）':<10}{v['M']['episode']*100:>13.2f}%{v['M']['session']*100:>13.2f}%"
          f"{v['dM']*100:>+9.2f}")
    print(f"  {'R（均值）':<10}{v['R']['episode']*100:>13.2f}%{v['R']['session']*100:>13.2f}%"
          f"{(v['R']['session']-v['R']['episode'])*100:>+9.2f}")
    print(f"\n  采样改善量 ΔM = M_S − M_E = **{v['dM']*100:+.2f} 个百分点**"
          f"　⇒ ΔM>0：{'**是**' if v['dM'] > 0 else '**否**'}")
    print(f"  G0（M_S>0 且 R_S≥{A.G0_GATE*100:.0f}%）："
          f"{'**通过**' if v['g0_session'] else '**不通过**'}"
          f"　（M_S={v['M']['session']*100:+.2f}% · R_S={v['R']['session']*100:+.2f}%）")
    print(f"  整体护栏（R_S ≥ **本轮** R_E − {GUARD_PP*100:.0f}pp = "
          f"{(v['R']['episode']-GUARD_PP)*100:.2f}%）："
          f"{'**满足**' if v['guard'] else '**不满足**'}")
    print(f"\n  **本轮配置推进（三项同时满足）："
          f"{'**是**' if v['advance'] else '**否**'}**")
    for p in phrase(v):
        print(f"    ⚠ {p}")
    print(f"\n  用时 {hrs:.2f} GPU·小时（预算 2）"
          + ("" if hrs <= 2 else "   ⚠ **超预算 ⇒ 记为未完成**"))

    (OUT / "fabs_balance_v1.json").write_text(json.dumps(
        {"runs": runs, "rbar": rbar, "verdict": v, "per_seed": per_seed, "g2": g2,
         "seeds": list(SEEDS), "arms": list(ARMS), "steps": a.steps,
         "guard_pp": GUARD_PP, "gate": A.G0_GATE, "gpu_hours": hrs,
         "phrasing_rules": phrase(v),
         "note": "护栏基准 = 本轮重跑的 R_E，非历史 40.70%；"
                 "臂 E seed=0 与历史之差 = **复现差异**，不得称运行噪声下界。"},
        indent=2, ensure_ascii=False))
    part.unlink(missing_ok=True)
    print(f"→ {OUT/'fabs_balance_v1.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
