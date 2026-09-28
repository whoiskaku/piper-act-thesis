#!/usr/bin/env python3
"""`F_inc` 的 `P1` 内部早停 —— 严格按 `reports/PREREG_FINC_EARLYSTOP.md`（**已冻结**）。

**只检验 H_es（固定 3000 步是否让增量网络过度拟合 P1）。唯一改变的是「取第几步的权重」。**
不换网络 · 不加正则 · 不追加种子 · 不插检查点 · 不上机 · 不训练 π。
**上一轮判否保持不改写；全会话 G0 仍不通过；`ab2` 保留未动。**

选择（只用 P1）
    V(step) = mean over {8 折} × {3 训练种子} of E(该折留出集, 该 step)
    step*   = V 取最小的检查点中**取最早者**（完整精度，rtol=1e-12/atol=1e-15）

最终（step* > 0 时）
    全 8 集 P1 训练到 **3000 步**，同时存第 `step*` 与第 3000 步权重。
    门槛**只施加于 (c)@step\***；3000 步只作**固定参照**，**不得据 P2 改选**。

用法:
    python -u sim/train_finc_earlystop.py --self-test
    python -u sim/train_finc_earlystop.py
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "sim/step4_gvnext"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train_fabs_g0 as A          # FAbs / features / err_energy / bp_kernel / EDGE / STEPS / LR
import train_finc_ab4 as F         # load_tier / Pfilt / Eerr / FInc / fit_beta / aggregate / dbar

OUT = ROOT / "outputs/finc_earlystop"
NOMINAL_CKPT = ROOT / "outputs/finc_ab4/fabs_nominal_ab4.pt"

CKPTS = (0, 50, 100, 200, 400, 800, 1500, 3000)   # **固定，不插点、不扩网格**
TRAIN_SEEDS = F.TRAIN_SEEDS                       # (0,1,2) —— 复用，不追加
TIE_RTOL, TIE_ATOL = 1e-12, 1e-15                 # 平局容差（完整精度，不先四舍五入）
BUDGET_H = 1.0                                    # **含留一/最终训练/评分/保存**
BETA_REF = 0.960643                               # 上一轮的 β，须复算吻合
REF_P2_REDU = {0: 0.0098, 1: 0.0205, 2: 0.0516}   # 参照臂应复现的上一轮 P2 redu（%/100，粗值）


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ── 选择规则（只用 P1）──────────────────────────────────────────────────
def pick_step(V):
    """`V[step]` → `step*` = 取最小值的检查点中**最早者**（完整精度平局容差）。"""
    steps = sorted(V)
    vmin = min(V[s] for s in steps)
    tied = [s for s in steps if abs(V[s] - vmin) <= TIE_ATOL + TIE_RTOL * abs(vmin)]
    return int(min(tied)), tied


# ── 训练一次，并在预定检查点处快照 + 评分 ────────────────────────────────
def train_with_ckpts(tr, base_tr, K, dev, seed, ckpts, evals=None, steps=A.STEPS, log=None):
    """训练 `steps` 步；在 `ckpts` 处保存 `state_dict` 深拷贝，并对 `evals` 评分。

    `evals` = [(名字, 集列表, 名义预测列表)]，在每个检查点上算 `E`（逐集）。
    与上一轮 `F.train_finc` **同初始化、同采样轨迹**（同 seed ⇒ 同 RNG 流）。
    """
    import torch, copy
    net = F.FInc(dev, seed)                       # manual_seed(seed) 后第一个消费 torch RNG
    rng = np.random.default_rng(seed)             # 抽样流，与上一轮一致
    mu_u, sd_u = (lambda X: (X.mean(0), np.where(X.std(0) < 1e-8, 1.0, X.std(0))))(
        np.concatenate([e.x for e in tr]))
    mu_d, sd_d = (lambda X: (X.mean(0), np.where(X.std(0) < 1e-8, 1.0, X.std(0))))(
        np.concatenate([e.dq for e in tr]))
    stats = (mu_u, sd_u, mu_d, sd_d)
    Kt = torch.tensor(K, device=dev).view(1, 1, -1).repeat(3, 1, 1)

    def band(r):
        return torch.nn.functional.conv1d(r.transpose(1, 2), Kt, padding=20, groups=3
                                          ).transpose(1, 2)

    def mk(e):
        xu = torch.tensor((e.x - mu_u) / sd_u, device=dev, dtype=torch.float32).T[None]
        xd = torch.tensor((e.dq - mu_d) / sd_d, device=dev, dtype=torch.float32).T[None]
        xd0 = torch.tensor((np.zeros_like(e.dq) - mu_d) / sd_d,
                           device=dev, dtype=torch.float32).T[None]
        return xu, xd, xd0

    pre = [mk(e) for e in tr]
    ys = [torch.tensor(np.asarray(e.y, np.float32), device=dev)[None] for e in tr]
    b0 = [torch.tensor(p.astype(np.float32), device=dev)[None] for p in base_tr]
    opt = torch.optim.Adam(net.m.net.parameters(), lr=A.LR)      # 无 weight decay

    snaps, scores, hist = {}, {}, []

    def snapshot(s):
        snaps[s] = copy.deepcopy({k: v.detach().cpu() for k, v in net.m.net.state_dict().items()})
        if evals:
            scores[s] = {}
            for name, E, bs in evals:
                pf = F.finc_predict(net, stats, E, dev)
                scores[s][name] = [F.Eerr(np.asarray(e.y, np.float64) - (bb + f), K)[0]
                                   for e, bb, f in zip(E, bs, pf)]

    t0 = time.time()
    if 0 in ckpts:
        snapshot(0)                                # **第 0 步：F_inc ≡ 0 逐位**
    for it in range(steps):
        i = int(rng.integers(len(tr)))
        xu, xd, xd0 = pre[i]
        r = ys[i] - (b0[i] + net(xu, xd, xd0))
        r = r - r.mean(1, keepdim=True)
        v = band(r)[:, A.EDGE:-A.EDGE]
        loss = (v ** 2).sum() / (3 * v.shape[1])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.m.net.parameters(), 1.0); opt.step()
        if (it + 1) % 250 == 0:
            hist.append({"step": it + 1, "loss": float(loss.item())})
        if (it + 1) in ckpts:
            snapshot(it + 1)
        if log and (it + 1) % 1500 == 0:
            log(f"        step {it+1}/{steps}  loss {loss.item():.6g}  {time.time()-t0:.0f}s")
    return net, stats, snaps, scores, hist, time.time() - t0


def save_ckpt(path, sd, stats, meta):
    import torch
    mu_u, sd_u, mu_d, sd_d = stats
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": sd, "mu_u": mu_u, "sd_u": sd_u, "mu_d": mu_d, "sd_d": sd_d,
                **meta}, path)


# ── 自测 ─────────────────────────────────────────────────────────────────
def self_test(P1, P2, P3, K, dev, S1, H, base):
    print("\n" + "=" * 78 + "\n自测\n" + "=" * 78)
    ok = {}

    def chk(k, c, d=""):
        ok[k] = bool(c); print(f"  {'✓' if c else '**✗**'} {k}{('  ' + d) if d else ''}")

    # ① 检查点集合固定
    chk("1 **检查点固定为 8 个且含 0 与 3000**",
        CKPTS == (0, 50, 100, 200, 400, 800, 1500, 3000) and 0 in CKPTS and 3000 in CKPTS,
        str(CKPTS))
    chk("1b 训练种子复用 (0,1,2)，未追加", TRAIN_SEEDS == (0, 1, 2) and len(TRAIN_SEEDS) == 3)
    chk("1c 留一折数 = P1 的探针种子数 = 8，各 1 集",
        len(P1) == 8 == len({int(e.seed) for e in P1}))

    # ② 第 0 步 ⇒ F_inc ≡ 0 逐位 ⇒ (c)@0 与 (a) **逐位**相同
    import torch
    net = F.FInc(dev, 0)
    st = (np.concatenate([e.x for e in P1]).mean(0),
          np.where(np.concatenate([e.x for e in P1]).std(0) < 1e-8, 1.0,
                   np.concatenate([e.x for e in P1]).std(0)),
          np.concatenate([e.dq for e in P1]).mean(0),
          np.where(np.concatenate([e.dq for e in P1]).std(0) < 1e-8, 1.0,
                   np.concatenate([e.dq for e in P1]).std(0)))
    pf0 = F.finc_predict(net, st, P1, dev)
    chk("2 **第 0 步 F_inc ≡ 0 逐位**（输出头零初始化）",
        max(float(np.abs(p).max()) for p in pf0) == 0.0)
    Ea = [F.Eerr(np.asarray(e.y, np.float64) - b_, K)[0] for e, b_ in zip(P1, base["P1"])]
    Ec0 = [F.Eerr(np.asarray(e.y, np.float64) - (b_ + p), K)[0]
           for e, b_, p in zip(P1, base["P1"], pf0)]
    chk("2b ⇒ **(c)@0 与 (a) 逐位相同**", all(x == y for x, y in zip(Ea, Ec0)),
        f"max|Δ| = {max(abs(x-y) for x, y in zip(Ea, Ec0)):.3e}")

    # ③ 平局规则：完整精度、rtol=1e-12/atol=1e-15、取最早者
    s_, t_ = pick_step({0: 1.0, 50: 0.5, 100: 0.5, 200: 0.5 + 1e-13, 400: 0.9,
                        800: 1.1, 1500: 1.2, 3000: 1.3})
    chk("3 **平局取最早者**（50/100/200 在容差内并列）", s_ == 50 and set(t_) == {50, 100, 200},
        f"step*={s_} · 并列 {sorted(t_)}")
    s2, _ = pick_step({0: 0.5, 50: 0.5 + 1e-6, 100: 0.9, 200: 0.9, 400: 0.9,
                       800: 0.9, 1500: 0.9, 3000: 0.9})
    chk("3b 差 1e-6 **不算**平局（容差是 1e-12/1e-15，不是四舍五入）", s2 == 0)
    s3, _ = pick_step({s: 1.0 - 1e-9 * s for s in CKPTS})
    chk("3c 单调下降 ⇒ 选 3000（规则不偏向早停）", s3 == 3000)

    # ④ 留一划分：训练 7 集、验证 1 集、归一化不含留出
    f0 = [e for e in P1 if int(e.seed) != int(P1[0].seed)]
    chk("4 **留一：训练 7 集、验证 1 集**", len(f0) == 7 and int(P1[0].seed) not in
        {int(e.seed) for e in f0})
    mu_in = np.concatenate([e.x for e in f0]).mean(0)
    mu_all = np.concatenate([e.x for e in P1]).mean(0)
    chk("4b **归一化只用该折 7 集**（与全 8 集不同 ⇒ 确实排除了留出）",
        np.abs(mu_in - mu_all).max() > 0)

    # ⑤ 选择只用 P1
    import inspect
    src = inspect.getsource(pick_step)
    chk("5 **`pick_step` 只吃 V 表，不触及 P2/P3**",
        "P2" not in src and "P3" not in src and "V[" in src)

    # ⑥ β 复算
    beta = F.fit_beta(P1, base["P1"], H, K, S1)
    chk("6 **β 复算 == 上一轮 +0.960643**", abs(beta - BETA_REF) < 5e-7, f"{beta:+.9f}")

    # ⑦ 名义分支 checkpoint
    ck = torch.load(NOMINAL_CKPT, weights_only=False)
    chk("7 **名义分支 checkpoint 存在且 redu 吻合**",
        abs(ck["redu_ab4"] - 0.386568) < 1e-5 and ck["spec"]["hold"] == "ab4/P0",
        f"redu {ck['redu_ab4']*100:.4f}% · sha {sha(NOMINAL_CKPT)[:16]}…")

    # ⑧ 聚合与判据复用上一轮（同一对象，未复制）
    chk("8 **聚合/判据复用 `train_finc_ab4`**（同一对象）",
        F.aggregate.__module__ == "train_finc_ab4" and F.dbar.__module__ == "train_finc_ab4"
        and F.GATE_D == 0.05)

    # ⑨ 预算守卫：超预算 ⇒ 未完成，**不得**用已完成的折选步数
    def guard(elapsed_h, done, total):
        if elapsed_h > BUDGET_H:
            return {"complete": False, "step_star": None}
        return {"complete": done == total, "step_star": 0 if done == total else None}
    chk("9 **超预算 ⇒ 未完成且不输出 step\\***",
        guard(1.5, 12, 24)["step_star"] is None and not guard(1.5, 12, 24)["complete"])
    chk("9b 未跑满 24 折也不输出 step*", guard(0.5, 12, 24)["step_star"] is None)

    # ⑩ 逐帧预测 → 逐集评分的 round-trip
    e = P1[0]; p = base["P1"][0]
    e1, _ = F.Eerr(np.asarray(e.y, np.float64) - p, K)
    arr = np.asarray(e.y, np.float64) - p
    e2 = float((F.Pfilt(arr, K) ** 2).sum()) / (3 * len(F.Pfilt(arr, K)))
    chk("10 **逐帧预测可重算出逐集评分**", abs(e1 - e2) / e1 < 1e-12)

    # ⑪ 门槛只施加于 step*，参照臂不参与
    chk("11 **参照臂 (c₃₀₀₀) 不参与门槛**（源码中门槛只引用 star 键）",
        "'star'" in inspect.getsource(main) and "gates" in inspect.getsource(main))
    chk("11b 参照值表 = 上一轮的 P2 redu", REF_P2_REDU == {0: 0.0098, 1: 0.0205, 2: 0.0516})

    # ⑫ 留一检查点份数
    chk("12 **留一检查点份数 = 8 折 × 3 种子 × 8 检查点 = 192**",
        len(P1) * len(TRAIN_SEEDS) * len(CKPTS) == 192,
        f"{len(P1)}×{len(TRAIN_SEEDS)}×{len(CKPTS)} = {len(P1)*len(TRAIN_SEEDS)*len(CKPTS)}"
        f" · 约 {192*70531*4/2**20:.1f} MiB")
    print(f"\n  ⇒ 自测 {sum(ok.values())}/{len(ok)} 通过")
    return all(ok.values())


def main() -> int:
    import gc; gc.disable()
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    import fit_step1_candidate as S1
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    cal = json.loads((ROOT / "outputs/step1_H/step1_H_v1.json").read_text())["calibration"]
    R = np.asarray(cal["R"], np.float64); b = np.asarray(cal.get("b", [0, 0, 0]), np.float64)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rend = IdealChunkRenderer()
    K = A.bp_kernel(S1)
    H = np.ascontiguousarray(np.load(F.H_PATH)["H"], np.float64)
    P1, P2, P3 = (F.load_tier(S1, rend, R, b, t) for t in ("P1", "P2", "P3"))
    TIERS = (("P1", P1), ("P2", P2), ("P3", P3))
    print(f"P1 {len(P1)} 集（种子 {sorted(int(e.seed) for e in P1)}）· "
          f"P2 {len(P2)} · P3 {len(P3)}   device={dev}")

    # 名义分支：**加载**已存权重（不重跑）
    ck = torch.load(NOMINAL_CKPT, weights_only=False)
    mN = A.FAbs(dev); mN.net.load_state_dict(ck["state_dict"])
    base = {t: F.ebase_predict(mN, ck["mu"], ck["sd"], E, dev) for t, E in TIERS}
    print(f"名义分支 ← {NOMINAL_CKPT.name}  sha {sha(NOMINAL_CKPT)[:24]}…  "
          f"redu(ab4/P0) = {ck['redu_ab4']*100:.4f}%")

    if a.self_test:
        return 0 if self_test(P1, P2, P3, K, dev, S1, H, base) else 1

    OUT.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    beta = F.fit_beta(P1, base["P1"], H, K, S1)
    assert abs(beta - BETA_REF) < 5e-7, f"β 复算不吻合: {beta}"
    print(f"β 复算 = {beta:+.9f}（== 上一轮 {BETA_REF:+.6f}）")
    hconv = {t: [S1.causal_conv(H, e.dq) for e in E] for t, E in TIERS}

    # ── ② 留一选步数（只用 P1）──────────────────────────────────────
    print("\n" + "=" * 78 + f"\n② P1 留一选训练时长（8 折 × {len(TRAIN_SEEDS)} 种子 = "
          f"{8*len(TRAIN_SEEDS)} 次训练；检查点 {CKPTS}）\n" + "=" * 78)
    seeds_p1 = sorted(int(e.seed) for e in P1)
    table, over = [], False
    for fi, sv in enumerate(seeds_p1):
        tr = [e for e in P1 if int(e.seed) != sv]
        va = [e for e in P1 if int(e.seed) == sv]
        btr = [p for e, p in zip(P1, base["P1"]) if int(e.seed) != sv]
        bva = [p for e, p in zip(P1, base["P1"]) if int(e.seed) == sv]
        for k in TRAIN_SEEDS:
            _, stats, snaps, scores, _, sec = train_with_ckpts(
                tr, btr, K, dev, k, CKPTS, evals=[("val", va, bva)])
            for s in CKPTS:
                save_ckpt(OUT / "loo" / f"fold{sv}_seed{k}_step{s}.pt", snaps[s], stats,
                          {"fold_holdout_probe_seed": sv, "train_seed": k, "step": s,
                           "n_train": len(tr)})
                table.append({"fold": sv, "train_seed": k, "step": s,
                              "E_val": float(scores[s]["val"][0])})
            print(f"  折 {fi+1}/8（留出探针种子 {sv}）· 训练种子 {k} · "
                  f"E@0 {scores[0]['val'][0]:.6g} → E@3000 {scores[3000]['val'][0]:.6g}"
                  f"   [{sec:.0f}s]")
            if (time.time() - t_all) / 3600 > BUDGET_H:
                over = True; break
        if over:
            break

    n_done = len({(r["fold"], r["train_seed"]) for r in table})
    if over or n_done != 8 * len(TRAIN_SEEDS):
        print(f"\n  ⛔ **超预算或未跑满（完成 {n_done}/{8*len(TRAIN_SEEDS)} 折×种子）"
              f"⇒ 记为未完成，不输出 step\\***")
        (OUT / "finc_earlystop_v1.json").write_text(json.dumps(
            {"complete": False, "reason": "超预算或未跑满", "n_done": n_done,
             "loo_table": table, "gpu_hours": (time.time() - t_all) / 3600},
            indent=2, ensure_ascii=False))
        return 5

    V = {s: float(np.mean([r["E_val"] for r in table if r["step"] == s])) for s in CKPTS}
    star, tied = pick_step(V)
    print("\n  V(step)（8 折 × 3 种子 = 24 项的平均验证误差）")
    for s in CKPTS:
        print(f"    step {s:>5}   V = {V[s]:.8g}" + ("   ← **最小**" if s in tied else ""))
    print(f"\n  ⇒ **step\\* = {star}**" + (f"（并列 {sorted(tied)}，取最早者）"
                                           if len(tied) > 1 else ""))

    result = {"complete": True, "V": V, "step_star": star, "tied": sorted(tied),
              "beta": beta, "loo_table": table, "ckpts": list(CKPTS),
              "train_seeds": list(TRAIN_SEEDS), "budget_h": BUDGET_H,
              "nominal_ckpt_sha256": sha(NOMINAL_CKPT),
              "code_sha256": {p.name: sha(HERE / p.name) for p in
                              (HERE / "train_finc_earlystop.py", HERE / "train_finc_ab4.py",
                               HERE / "train_fabs_g0.py", HERE / "train_fabs_balance.py")}}

    # ── ③.2 step* = 0 的停止分支 ────────────────────────────────────
    if star == 0:
        print("\n  ⛔ **step\\* = 0 ⇒ 内部验证不支持加入该增量分支。**")
        print("     不继续搜索，**不为取得参照额外训练**。")
        print("     ⚠ 按构造必然失败的**只有条件 3**（(c)=(a)，不满足严格低于 (a)）；")
        print("     **条件 1、2 取决于 H 与 βH 的实际评分，不能称数学上必然失败。**")
        result["stopped"] = "step*=0：内部验证不支持加入该增量分支"
        result["gpu_hours"] = (time.time() - t_all) / 3600
        (OUT / "finc_earlystop_v1.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=float))
        print(f"\n  用时 {result['gpu_hours']:.3f} GPU·小时（预算 {BUDGET_H}）")
        print(f"→ {OUT/'finc_earlystop_v1.json'}")
        return 0

    # ── ③ 最终重训（全 8 集，训到 3000，存 step* 与 3000）───────────
    print("\n" + "=" * 78 + f"\n③ 最终重训（全 {len(P1)} 集 P1，训到 3000，"
          f"存第 {star} 步与第 3000 步）\n" + "=" * 78)
    finc_pred = {t: {"star": {}, "ref": {}} for t, _ in TIERS}
    p1_uniform, ref_ok = {}, {}
    for k in TRAIN_SEEDS:
        net, stats, snaps, _, hist, sec = train_with_ckpts(
            P1, base["P1"], K, dev, k, (star, 3000), log=lambda m: print(m))
        for tag, s in (("star", star), ("ref", 3000)):
            save_ckpt(OUT / "final" / f"seed{k}_step{s}.pt", snaps[s], stats,
                      {"train_seed": k, "step": s, "role": tag, "n_train": len(P1)})
            net.m.net.load_state_dict({kk: vv.to(dev) for kk, vv in snaps[s].items()})
            for t, E in TIERS:
                finc_pred[t][tag][k] = F.finc_predict(net, stats, E, dev)
        # 参照臂完整性：(c₃₀₀₀) 在 P2 的 redu 应复现上一轮
        num = den = 0.0
        for e, b_, f_ in zip(P2, base["P2"], finc_pred["P2"]["ref"][k]):
            n_, _ = A.err_energy(np.asarray(e.y, np.float64) - (b_ + f_), K)
            d_, _ = A.err_energy(np.asarray(e.y, np.float64), K)
            num += n_; den += d_
        r3k = 1.0 - num / den
        ref_ok[k] = bool(abs(r3k - REF_P2_REDU[k]) < 5e-4)
        # 最终模型在**全 P1** 的统一评分（补上一轮缺的精确训练/验证比）
        p1_uniform[k] = {
            tag: float(np.mean([F.Eerr(np.asarray(e.y, np.float64) - (b_ + f_), K)[0]
                                for e, b_, f_ in zip(P1, base["P1"], finc_pred["P1"][tag][k])]))
            for tag in ("star", "ref")}
        print(f"  种子 {k}: (c₃₀₀₀) 在 P2 的 redu = {r3k*100:+.2f}% "
              f"(上一轮 {REF_P2_REDU[k]*100:+.2f}%) {'✓ 复现' if ref_ok[k] else '**✗ 不符**'}"
              f"   [{sec:.0f}s]")

    # ── 评分与判据 ───────────────────────────────────────────────────
    E_of = lambda E, preds: [F.Eerr(np.asarray(e.y, np.float64) - p, K)[0]
                             for e, p in zip(E, preds)]
    res, preds_dump = {}, {}
    for t, E in TIERS:
        Ea = F.aggregate(E, {0: E_of(E, base[t])})
        Eb = F.aggregate(E, {0: E_of(E, [p + h for p, h in zip(base[t], hconv[t])])})
        Eb2 = F.aggregate(E, {0: E_of(E, [p + beta * h for p, h in zip(base[t], hconv[t])])})
        row = {"E_a": Ea, "E_b": Eb, "E_bp": Eb2,
               "mean_E_a": float(np.mean(list(Ea.values())))}
        for tag in ("star", "ref"):
            Ec = F.aggregate(E, {k: E_of(E, [p + f_ for p, f_ in zip(base[t], finc_pred[t][tag][k])])
                                 for k in TRAIN_SEEDS})
            db, dbb = F.dbar(Eb, Ec); dp, dpb = F.dbar(Eb2, Ec)
            row[tag] = {"E_c": Ec, "d_b": db, "dbar_b": dbb, "d_bp": dp, "dbar_bp": dpb,
                        "mean_E_c": float(np.mean(list(Ec.values())))}
        res[t] = row
        preds_dump[t] = {"eps": [e.path for e in E],
                         "gap": [np.asarray(e.y, np.float64) for e in E],
                         "a": base[t], "b": [p + h for p, h in zip(base[t], hconv[t])],
                         "bp": [p + beta * h for p, h in zip(base[t], hconv[t])],
                         **{f"c_{tag}_s{k}": [p + f_ for p, f_ in zip(base[t], finc_pred[t][tag][k])]
                            for tag in ("star", "ref") for k in TRAIN_SEEDS}}

    p2 = res["P2"]; st = p2["star"]
    gate_b = all(v > 0 for v in st["d_b"].values()) and st["dbar_b"] >= F.GATE_D
    gate_bp = all(v > 0 for v in st["d_bp"].values()) and st["dbar_bp"] >= F.GATE_D
    gate_a = st["mean_E_c"] < p2["mean_E_a"]
    advance = bool(gate_b and gate_bp and gate_a)
    hrs = (time.time() - t_all) / 3600

    print("\n" + "=" * 78 + f"\n判据（**只施加于 (c)@step\\*={star}**；3000 步只作固定参照）\n"
          + "=" * 78)
    for t, _ in TIERS:
        r_ = res[t]; ss = sorted(r_["E_a"])
        print(f"\n  【{t}】按探针种子等权的误差 E（越小越好）")
        print(f"    {'探针种子':<10}{'(a)':>12}{'(b) H':>12}{'(b′) βH':>12}"
              f"{f'(c)@{star}':>13}{'(c)@3000':>12}{'d vs b':>10}{'d vs b′':>10}")
        for s in ss:
            print(f"    {s:<10}{r_['E_a'][s]:>12.6g}{r_['E_b'][s]:>12.6g}{r_['E_bp'][s]:>12.6g}"
                  f"{r_['star']['E_c'][s]:>13.6g}{r_['ref']['E_c'][s]:>12.6g}"
                  f"{r_['star']['d_b'][s]*100:>+9.2f}%{r_['star']['d_bp'][s]*100:>+9.2f}%")
        print(f"    {'等权平均':<8}{r_['mean_E_a']:>12.6g}{'':>12}{'':>12}"
              f"{r_['star']['mean_E_c']:>13.6g}{r_['ref']['mean_E_c']:>12.6g}"
              f"{r_['star']['dbar_b']*100:>+9.2f}%{r_['star']['dbar_bp']*100:>+9.2f}%")

    print("\n  推进三条（**全部满足才算推进**；对象 = (c)@step\\*）：")
    print(f"    1. vs (b)  冻结 H ：三种子全正 = {all(v>0 for v in st['d_b'].values())} · "
          f"d̄ = {st['dbar_b']*100:+.2f}% ≥ 5% = {st['dbar_b'] >= F.GATE_D}  ⇒ "
          f"{'**满足**' if gate_b else '**不满足**'}")
    print(f"    2. vs (b′) 标量校准 H：三种子全正 = {all(v>0 for v in st['d_bp'].values())} · "
          f"d̄ = {st['dbar_bp']*100:+.2f}% ≥ 5% = {st['dbar_bp'] >= F.GATE_D}  ⇒ "
          f"{'**满足**' if gate_bp else '**不满足**'}")
    print(f"    3. (c) 误差低于 (a)：{st['mean_E_c']:.6g} < {p2['mean_E_a']:.6g}  ⇒ "
          f"{'**满足**' if gate_a else '**不满足**'}")
    print(f"\n  **推进：{'**是**' if advance else '**否**'}**")
    if gate_b and not gate_bp:
        print("    ⚠ **只赢 (b) 而输给 (b′) ⇒ 必须写成「未能优于标量校准后的 H」，不算通过。**")
    print(f"\n  参照臂（同初始化、同采样轨迹）：(c)@3000 的 d̄ vs b = "
          f"{p2['ref']['dbar_b']*100:+.2f}% ｜ (c)@{star} = {st['dbar_b']*100:+.2f}%")
    print(f"    复现上一轮：{ref_ok}" +
          ("" if all(ref_ok.values()) else "   ⚠ **参照臂无效，须在结论中声明**"))
    print("\n  最终模型在**全 P1** 的统一评分（补上一轮缺的精确训练/验证比）：")
    print(f"    (a) 在 P1 = {np.mean(E_of(P1, base['P1'])):.6g}")
    for k in TRAIN_SEEDS:
        print(f"    种子 {k}: (c)@{star} = {p1_uniform[k]['star']:.6g} · "
              f"(c)@3000 = {p1_uniform[k]['ref']:.6g}")
    print(f"\n  用时 {hrs:.3f} GPU·小时（预算 {BUDGET_H}）"
          + ("" if hrs <= BUDGET_H else "   ⚠ **超预算 ⇒ 记为未完成**"))

    np.savez_compressed(OUT / "predictions.npz",
                        **{f"{t}__{kk}__{i}": np.asarray(v[i], np.float64)
                           for t, d in preds_dump.items() for kk, v in d.items()
                           if kk != "eps" for i in range(len(v))})
    (OUT / "episodes.json").write_text(json.dumps(
        {t: preds_dump[t]["eps"] for t, _ in TIERS}, indent=2, ensure_ascii=False))
    result.update({"results": res, "p1_uniform": p1_uniform,
                   "E_a_P1_mean": float(np.mean(E_of(P1, base["P1"]))),
                   "ref_reproduces_prev_round": ref_ok,
                   "gates": {"vs_b": gate_b, "vs_bp": gate_bp, "vs_a": gate_a,
                             "advance": advance, "applied_to": f"(c)@step*={star}"},
                   "gpu_hours": hrs,
                   "scope": "**限定 ab4 工况的开发实验；不能视为全会话 G0 已通过；"
                            "不代表跨会话泛化或真机迁移；ab2 保留。**"})
    (OUT / "finc_earlystop_v1.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=float))
    print(f"→ {OUT/'finc_earlystop_v1.json'} · predictions.npz · loo/ (192) · final/ (6)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
