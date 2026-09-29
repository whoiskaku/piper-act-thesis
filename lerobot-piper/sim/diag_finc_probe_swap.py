#!/usr/bin/env python3
"""扰动依赖性诊断（**不重训**）——「把输入的扰动波形换掉，网络是否仍然输出原来那份任务残差预测？」

事前限定（用户 2026-09-29 指定，运行前冻结）
--------------------------------------------
1. 固定 **24 个留一模型的第 3000 步检查点**，**只检查各自的 `P1` 训练集**（各 7 集），
   **不再看 `P2`/`P3`**。
2. **保持名义指令不变**；用**事先固定、无自身配对**的置换把 `δa` 换成另一集的探针。
   **统一有效时段，不伸缩波形。**
3. 比较**原配对 / 替换探针 / 零扰动**三种输入下的**预测变化**，
   以及对**原观测残差**的拟合程度。

判读（**事前限定，不得事后放宽**）
----------------------------------
- **换探针后仍保持很好的拟合** ⇒ 与「**`δa` 主要充当开关，网络记住了任务残差**」
  的假说**相容**，可作为下一步**约束扰动通道结构**的依据，**但不是机制证明**。
- **换后拟合崩塌** ⇒ 只能说明**网络依赖输入配对**，
  仍**分不清**「真实响应学习」与「探针指纹记忆」，**不能据此认可模型**。

⚠ 两条必须公开的限制
--------------------
- **没有替换探针后的真实 IMU** ⇒ **置换后的拟合误差<u>不是</u>反事实预测准确率**。
  它衡量的只是「预测是否仍然对得上**原来那份**观测残差」。
- **闭环名义指令可能携带原探针的信息**（`u_e` 来自被 `δa_e` 扰动过的闭环 rollout）
  ⇒ 即使换掉 `δa`，网络仍可能从 `u` 复原原残差。

**不扫描置换、不追加训练、不挽救旧判决。只出一页诊断。**

用法:
    python -u sim/diag_finc_probe_swap.py --self-test
    python -u sim/diag_finc_probe_swap.py
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

import train_fabs_g0 as A
import train_finc_ab4 as F
import train_finc_earlystop as ES

OUT = ROOT / "outputs/finc_probe_swap"
LOO = ES.OUT / "loo"
STEP = 3000                      # **固定用第 3000 步检查点**
COND = ("paired", "perm", "zero")


def perm_map(seeds):
    """**事先固定、无自身配对**的置换：按探针种子升序做**循环移位 1**。

    8 个种子构成单个 8-循环 ⇒ **无不动点**（也无互换对）。**确定性，不随机、不扫描。**
    """
    ss = sorted(seeds)
    return {s: ss[(i + 1) % len(ss)] for i, s in enumerate(ss)}


def load_finc(path, dev):
    """载入某个留一模型的第 3000 步权重 + **它自己的**归一化统计。"""
    import torch
    ck = torch.load(path, weights_only=False)
    net = F.FInc(dev, 0)
    net.m.net.load_state_dict({k: v.to(dev) for k, v in ck["state_dict"].items()})
    net.m.net.eval()
    return net, (ck["mu_u"], ck["sd_u"], ck["mu_d"], ck["sd_d"]), ck


def predict(net, stats, x, da, dev):
    """`F_inc(u, δa)` —— 名义特征 `x` 与扰动 `da` **分开传**，便于替换 `da` 而保持 `x` 不变。"""
    import torch
    mu_u, sd_u, mu_d, sd_d = stats
    with torch.no_grad():
        xu = torch.tensor((x - mu_u) / sd_u, device=dev, dtype=torch.float32).T[None]
        xd = torch.tensor((da - mu_d) / sd_d, device=dev, dtype=torch.float32).T[None]
        # **零扰动分支：原始 δa 置 0 后再归一化**（与训练时一致）
        xd0 = torch.tensor((np.zeros_like(da) - mu_d) / sd_d,
                           device=dev, dtype=torch.float32).T[None]
        return net(xu, xd, xd0)[0].cpu().numpy().astype(np.float64)


def band_stats(fa, fb, K):
    """带通后比较两份预测：`cos`、`‖b‖/‖a‖`、`‖b−a‖/‖a‖`（与损失同口径）。"""
    pa, pb = F.Pfilt(fa, K), F.Pfilt(fb, K)
    na, nb = float(np.linalg.norm(pa)), float(np.linalg.norm(pb))
    if na == 0.0:
        return {"cos": float("nan"), "ratio": float("nan"), "rel_diff": float("nan"),
                "norm_a": na, "norm_b": nb}
    return {"cos": float((pa * pb).sum() / (na * nb)) if nb > 0 else 0.0,
            "ratio": nb / na, "rel_diff": float(np.linalg.norm(pb - pa) / na),
            "norm_a": na, "norm_b": nb}


def self_test(P1, base1, K, dev, L):
    print("\n" + "=" * 78 + "\n自测\n" + "=" * 78)
    ok = {}

    def chk(k, c, d=""):
        ok[k] = bool(c); print(f"  {'✓' if c else '**✗**'} {k}{('  ' + d) if d else ''}")

    seeds = sorted(int(e.seed) for e in P1)
    pm = perm_map(seeds)
    chk("1 **置换无自身配对**（无不动点）", all(pm[s] != s for s in seeds), str(pm))
    chk("1b 置换是双射且值域 = 原集合", sorted(pm.values()) == seeds)
    chk("1c **确定性**（重复调用逐位相同，未用随机数）", perm_map(seeds) == pm)
    chk("1d 单个 8-循环 ⇒ 也无互换对",
        len({tuple(sorted((s, pm[s]))) for s in seeds}) == len(seeds))

    # ② 统一有效时段：截断而非伸缩
    ns = [len(e.y) for e in P1]
    chk("2 **统一有效时段 = min 帧数，截断不伸缩**", L == min(ns) and len(set(ns)) > 1,
        f"L = {L} · 各集原始帧数 {ns}")
    import inspect
    # ⚠ 只查**真正干活的函数**：整模块源码含自测自己写的关键词列表（自指陷阱）
    work = "".join(inspect.getsource(f_) for f_ in (perm_map, predict, band_stats, main))
    chk("2b 干活的函数中**无重采样/插值**（不伸缩波形）",
        all(t not in work for t in ("interp", "resample", "CubicSpline", "zoom")))
    chk("2c 截断是纯切片 `[:L]`（长度一致，值逐位取自原数组）",
        all(np.array_equal(np.asarray(e_.dq)[:L], np.asarray(e_.dq[:L])) and len(e_.dq[:L]) == L
            for e_ in P1))

    # ③ 名义指令在三种条件下逐位不变
    p = sorted(LOO.glob(f"*_step{STEP}.pt"))[0]
    net, stats, ck = load_finc(p, dev)
    e = P1[0]; d_other = np.ascontiguousarray(P1[1].dq[:L], np.float64)
    x = np.ascontiguousarray(e.x[:L], np.float32)
    f_pair = predict(net, stats, x, np.ascontiguousarray(e.dq[:L], np.float64), dev)
    f_perm = predict(net, stats, x, d_other, dev)
    f_zero = predict(net, stats, x, np.zeros((L, 6)), dev)
    chk("3 **零扰动条件下 F_inc 逐位为 0**", float(np.abs(f_zero).max()) == 0.0)
    chk("3b 原配对与替换探针的输出**确实不同**（置换生效）",
        float(np.abs(f_perm - f_pair).max()) > 0)
    chk("3c 替换的 `δa` **确实来自另一集**（逐位不等于原 `δa`）",
        float(np.abs(d_other - np.asarray(e.dq[:L], np.float64)).max()) > 0)

    # ④ 用**该模型自己的**归一化统计，不是 P1 全局
    mu_all = np.concatenate([x_.dq for x_ in P1]).mean(0)
    chk("4 **归一化统计来自该折的 checkpoint**（与 P1 全局不同 ⇒ 确实是逐折的）",
        float(np.abs(np.asarray(stats[2]) - mu_all).max()) > 0)

    # ⑤ 检查点齐备 & 只用第 3000 步
    chk("5 **24 个留一模型的第 3000 步检查点齐备**",
        len(list(LOO.glob(f"*_step{STEP}.pt"))) == 24,
        f"{len(list(LOO.glob(f'*_step{STEP}.pt')))} 个")
    j = json.load(open(ES.OUT / "finc_earlystop_v1.json"))
    chk("5b 权重哈希与早停轮归档一致",
        ES.sha(p) == j["loo_ckpt_sha256"][p.name], f"{p.name} {ES.sha(p)[:16]}…")

    # ⑥ 只评各自的 7 集训练集，不含留出集、不含 P2/P3
    sv = int(p.stem.split("fold")[1].split("_")[0])
    tr = [x_ for x_ in P1 if int(x_.seed) != sv]
    chk("6 **每个模型只评自己的 7 集训练集**（排除该折留出集）",
        len(tr) == 7 and sv not in {int(x_.seed) for x_ in tr}, f"留出 {sv}")
    chk("6b **干活的函数中不出现 P2/P3**（本轮只看 P1）",
        all(t not in work for t in ('"P2"', "'P2'", '"P3"', "'P3'")))

    # ⑦ 口径复用
    chk("7 **E/P 口径复用 `train_finc_ab4`**（同一对象）",
        F.Eerr.__module__ == "train_finc_ab4" and F.Pfilt.__module__ == "train_finc_ab4"
        and A.EDGE == 20)
    bs = band_stats(f_pair, f_pair, K)
    chk("7b `band_stats` 自比 ⇒ cos=1、ratio=1、rel_diff=0",
        abs(bs["cos"] - 1) < 1e-12 and abs(bs["ratio"] - 1) < 1e-12 and bs["rel_diff"] < 1e-12)
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
    K = A.bp_kernel(S1)
    P1 = F.load_tier(S1, IdealChunkRenderer(), R, b, "P1")
    L = min(len(e.y) for e in P1)                      # **统一有效时段（截断，不伸缩）**
    ckN = torch.load(ES.NOMINAL_CKPT, weights_only=False)
    mN = A.FAbs(dev); mN.net.load_state_dict(ckN["state_dict"])
    base1 = F.ebase_predict(mN, ckN["mu"], ckN["sd"], P1, dev)
    print(f"P1 {len(P1)} 集（探针种子 {sorted(int(e.seed) for e in P1)}）· "
          f"统一有效时段 L = {L} 帧（各集原始 {[len(e.y) for e in P1]}）· device={dev}")

    if a.self_test:
        return 0 if self_test(P1, base1, K, dev, L) else 1

    OUT.mkdir(parents=True, exist_ok=True)
    seeds = sorted(int(e.seed) for e in P1)
    pm = perm_map(seeds)
    by = {int(e.seed): e for e in P1}
    bb = {int(e.seed): p for e, p in zip(P1, base1)}
    print(f"\n置换（事先固定、无自身配对、循环移位 1）：" +
          " ".join(f"{s}→{pm[s]}" for s in seeds))

    rows, t0 = [], time.time()
    for sv in seeds:
        for k in ES.TRAIN_SEEDS:
            p = LOO / f"fold{sv}_seed{k}_step{STEP}.pt"
            net, stats, _ = load_finc(p, dev)
            for s in seeds:
                if s == sv:
                    continue                            # **只评该折的 7 集训练集**
                e = by[s]
                x = np.ascontiguousarray(e.x[:L], np.float32)
                da = np.ascontiguousarray(e.dq[:L], np.float64)
                dn = np.ascontiguousarray(by[pm[s]].dq[:L], np.float64)
                r = np.asarray(e.y[:L], np.float64) - bb[s][:L]     # **原观测残差**
                f = {"paired": predict(net, stats, x, da, dev),
                     "perm": predict(net, stats, x, dn, dev),
                     "zero": predict(net, stats, x, np.zeros((L, 6)), dev)}
                E = {c: F.Eerr(r - f[c], K)[0] for c in COND}
                rows.append({"fold": sv, "train_seed": k, "ep_seed": s,
                             "donor_seed": pm[s], "donor_is_holdout": bool(pm[s] == sv),
                             **{f"E_{c}": float(E[c]) for c in COND},
                             **{f"pred_{q}": v for q, v in
                                band_stats(f["paired"], f["perm"], K).items()}})
            print(f"  折 {sv} · 训练种子 {k} 完成（7 集）")

    # ── 汇总 ─────────────────────────────────────────────────────────
    g = lambda key: np.array([r_[key] for r_ in rows], float)
    E0, Ep, Es = g("E_zero"), g("E_paired"), g("E_perm")
    summ = {
        "n_pairs": len(rows), "L": int(L), "step": STEP,
        "perm": {str(s): pm[s] for s in seeds},
        "E_zero_mean": float(E0.mean()), "E_paired_mean": float(Ep.mean()),
        "E_perm_mean": float(Es.mean()),
        "fit_gain_paired": float(1 - Ep.mean() / E0.mean()),
        "fit_gain_perm": float(1 - Es.mean() / E0.mean()),
        "frac_pairs_perm_better_than_zero": float((Es < E0).mean()),
        "frac_pairs_paired_better_than_zero": float((Ep < E0).mean()),
        "pred_cos_median": float(np.nanmedian(g("pred_cos"))),
        "pred_ratio_median": float(np.nanmedian(g("pred_ratio"))),
        "pred_rel_diff_median": float(np.nanmedian(g("pred_rel_diff"))),
        "sec": time.time() - t0,
    }
    m = np.array([r_["donor_is_holdout"] for r_ in rows], bool)
    summ["fit_gain_perm_donor_seen"] = float(1 - Es[~m].mean() / E0[~m].mean())
    summ["fit_gain_perm_donor_holdout"] = float(1 - Es[m].mean() / E0[m].mean())

    print("\n" + "=" * 78 + "\n诊断结果（24 模型 × 各自 7 集训练集 = "
          f"{len(rows)} 个 (模型,集) 对）\n" + "=" * 78)
    print(f"\n  对**原观测残差** `r = gap − ê_base` 的拟合（带内误差 E，越小越好）")
    print(f"    {'条件':<22}{'E 均值':>14}{'相对零扰动的拟合改善':>24}")
    print(f"    {'零扰动 F_inc≡0':<20}{E0.mean():>14.6g}{'—':>22}")
    print(f"    {'原配对 δa_e':<21}{Ep.mean():>14.6g}{summ['fit_gain_paired']*100:>21.2f}%")
    print(f"    {'替换探针 δa_σ(e)':<19}{Es.mean():>14.6g}{summ['fit_gain_perm']*100:>21.2f}%")
    print(f"\n    逐对优于零扰动的比例：原配对 {summ['frac_pairs_paired_better_than_zero']*100:.1f}%"
          f" · 替换探针 {summ['frac_pairs_perm_better_than_zero']*100:.1f}%")
    print(f"\n  **预测变化**（带通后，替换探针 vs 原配对）")
    print(f"    cos 中位数      = {summ['pred_cos_median']:+.4f}")
    print(f"    ‖perm‖/‖paired‖ = {summ['pred_ratio_median']:.4f}")
    print(f"    ‖Δ‖/‖paired‖    = {summ['pred_rel_diff_median']:.4f}")
    print(f"\n  附：置换供体来自该折留出集的 {int(m.sum())} 对 —— "
          f"拟合改善 {summ['fit_gain_perm_donor_holdout']*100:+.2f}%"
          f"（供体为训练集内的 {int((~m).sum())} 对：{summ['fit_gain_perm_donor_seen']*100:+.2f}%）")
    print(f"\n  用时 {summ['sec']:.0f}s（**未重训**）")

    (OUT / "probe_swap_v1.json").write_text(json.dumps(
        {"summary": summ, "rows": rows,
         "prespecified": "用户 2026-09-29 指定，运行前冻结：固定 24 个留一模型的第 3000 步"
                         "检查点、只看各自 P1 训练集、名义指令不变、事先固定无自身配对的置换、"
                         "统一有效时段不伸缩、比较原配对/替换探针/零扰动。",
         "limits": ["**没有替换探针后的真实 IMU** ⇒ 置换后的拟合误差**不是**反事实预测准确率",
                    "**闭环名义指令可能携带原探针的信息** ⇒ 换掉 δa 后仍可能从 u 复原原残差",
                    "**换后拟合崩塌只能说明网络依赖输入配对**，分不清真实响应学习与探针指纹记忆",
                    "**换后仍拟合良好只与「δa 充当开关」的假说相容，不是机制证明**"]},
        indent=2, ensure_ascii=False))
    print(f"→ {OUT/'probe_swap_v1.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
