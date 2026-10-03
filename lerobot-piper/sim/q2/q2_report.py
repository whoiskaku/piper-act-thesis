"""阶段二汇总 —— `reports/PREREG_Q2_STAGE2.md` §6、§7、§11。

会话误差比 `ρ = Σ_e ‖P(gap − ê)‖² / Σ_e ‖P(gap)‖²`（留出会话的补正运行内按集池化）。
**汇总顺序固定**：先对种子平均误差，得每折 ρ̄；`d_f = (ρ̄_H,f − ρ̄_候选,f) / ρ̄_H,f`，三折等权平均。
**判据**（每个候选单独判断，三条都满足才通过）：三折平均 d ≥ 5%；每折 ρ̄_候选 < ρ̄_H；每折 ρ̄_候选 < ρ̄_零响应。
**两种形式之间**（事前定死，不逐折挑赢家）：只有一个通过 ⇒ 用它；都通过 ⇒ 默认条件线性，只有非线性差分三折都更好、
且三折平均相对误差再降 ≥ 2%（分母 = 条件线性的 ρ̄）才保留非线性差分；都不通过 ⇒ 阶段二不通过。
⚠ 开发检验，不是密封确认；5% 与 2% 是事前的工程阈值，不等于统计显著，也不能换算成真机改善。
汇总前强制核验（任一不满足即拒绝汇总，并列出全部问题）：
- 3 折 × 3 种子 × {名义, 两种增量} 的内层表与终训产物、3 个 G-2 参照齐全；每份产物逐集覆盖冻结名单上该折该类的集。
- 全部 `code_sha` = 当前代码、`config_hash` = 冻结的选模规则；输入自 prepare 以来未变（数据、标定 R/b、固定 H、依赖、G-2
  逐一重算哈希）；每个作业的输入指纹与当前一致（终训含内层表哈希；增量含该折名义权重的哈希）。
- 由内层评分表重算 V 与 step*，与记录一致；`.pt` 的步数 = step*、权重哈希与记录一致；预测文件哈希一致。
本轮未完成（有「未完成」标记、缺截止时间记录、或任一产物晚于截止时间）⇒ 只写「未完成」，**不作通过判决**。
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent / "q1", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import q1_run as R1            # noqa: E402
import q1_train as TR1         # noqa: E402
import q2_data as D2           # noqa: E402
import q2_run as RUN           # noqa: E402
import q2_train as TR          # noqa: E402

FOLDS = D2.FOLDS
FORMS = RUN.FORMS
SEEDS = RUN.SEEDS
PREDICTORS = ("zero", "H") + FORMS
AUX_FOLD = "ab4"
AUX_ARMS = D2.AUX[AUX_FOLD]
LABEL = {"zero": "零响应（只用名义 2a）", "H": "固定 H", "nonlinear": "非线性差分", "condlinear": "条件线性",
         "nominal": "名义 2a"}


# ── 应有的产物与冻结名单 ────────────────────────────────────────────────────
def artifacts(out) -> dict:
    """→ {键: json 路径}：应有的全部 84 份产物（3 参照 + 18 名义内层 + 9 名义终训 + 36 增量内层 + 18 增量终训）。"""
    A = {}
    for f in FOLDS:
        A[("refs", f)] = RUN.ref_json(out, f)
        for s in SEEDS:
            for v in RUN.train_sessions(f):
                A[("nom-inner", f, v, s)] = RUN.nom_inner_json(out, f, v, s)
            A[("nom-final", f, s)] = RUN.nom_final_base(out, f, s).with_suffix(".json")
            for form in FORMS:
                for v in RUN.train_sessions(f):
                    A[("inc-inner", form, f, v, s)] = RUN.inc_inner_json(out, form, f, v, s)
                A[("inc-final", form, f, s)] = RUN.inc_final_base(out, form, f, s).with_suffix(".json")
    return A


def expected(fold: str, kind: str) -> list:
    """冻结名单上该会话该类（bare / corr / aux）的全部集名。"""
    arms = {"bare": D2.BARE, "corr": D2.CORR, "aux": D2.AUX}[kind][fold]
    return sorted(n for a in arms for n in D2.ROSTER.get(f"{fold}/{a}", []))


def _ident(key) -> dict:
    ph = key[0]
    if ph == "refs":
        return {"kind": "g2", "fold": key[1]}
    if ph == "nom-inner":
        return {"kind": ph, "fold": key[1], "val": key[2], "seed": key[3]}
    if ph == "nom-final":
        return {"kind": ph, "fold": key[1], "seed": key[2]}
    if ph == "inc-inner":
        return {"kind": ph, "form": key[1], "fold": key[2], "val": key[3], "seed": key[4]}
    return {"kind": ph, "form": key[1], "fold": key[2], "seed": key[3]}


# 每类产物里的逐集行 → 应覆盖的（会话, 类）
ROWS = {"refs": {"episodes": "bare"},
        "nom-final": {"q1": "bare", "zero": "corr", "H": "corr", "aux_zero": "aux", "aux_H": "aux", "aux_H_beta": "aux"},
        "inc-final": {"cand": "corr", "aux_cand": "aux", "aux_beta": "aux"}}


# ── 核验 ──────────────────────────────────────────────────────────────────
def _check_rows(P: list, path, j: dict, ph: str, fold: str) -> None:
    for key, kind in ROWS.get(ph, {}).items():
        want = expected(AUX_FOLD if kind == "aux" else fold, kind)
        rows = j.get(key)
        got = sorted(r["name"] for r in rows) if isinstance(rows, list) else []
        if got != want:
            P.append(f"{path}: {key} 的集与冻结名单不符（{len(got)} 集，应为 {len(want)}）")


def _check_inputs(P: list, out, I: dict, key, path, j: dict) -> None:
    ph = key[0]
    if ph == "refs":
        exp = RUN.ref_inputs(I)
    elif ph == "nom-inner":
        exp = RUN.nom_inner_inputs(I, key[1], key[2])
    elif ph == "nom-final":
        exp = RUN.nom_final_inputs(out, I, key[1], key[2])
    elif ph == "inc-inner":
        exp = RUN.inc_inner_inputs(out, I, *key[1:])
    else:
        exp = RUN.inc_final_inputs(out, I, *key[1:])
    if exp is None or j.get("inputs") != exp or j.get("input_fp") != RUN.fp_of(exp):
        P.append(f"{path}: 输入指纹与当前的数据 / 标定 / H / 内层表 / 名义权重不符")
    if ph in ("nom-inner", "inc-inner"):
        fold, val = (key[1], key[2]) if ph == "nom-inner" else (key[2], key[3])
        if j.get("train_sessions") != [RUN.inner_train(fold, val)]:
            P.append(f"{path}: 内层训练会话 {j.get('train_sessions')} 不对")


def _check_final(P: list, out, key, path, j: dict) -> None:
    """终训：由内层表重算的 V 与 step*、边界标记、权重与预测文件。"""
    import torch
    if key[0] == "nom-final":
        fold, seed, form = key[1], key[2], None
        inner = [RUN.nom_inner_json(out, fold, v, seed) for v in RUN.train_sessions(fold)]
    else:
        form, fold, seed = key[1], key[2], key[3]
        inner = [RUN.inc_inner_json(out, form, fold, v, seed) for v in RUN.train_sessions(fold)]
    if j.get("train_sessions") != list(RUN.train_sessions(fold)):
        P.append(f"{path}: 训练会话 {j.get('train_sessions')} 不对")
    try:
        V, step = RUN.selection(inner, TR.CONFIG["ckpts"])
    except (RuntimeError, KeyError, TypeError, ValueError) as err:
        P.append(f"{path}: 无法由内层表重算 step*（{err}）")
        return
    if j.get("V") != {str(s): V[s] for s in sorted(V)} or j.get("step_star") != step:
        P.append(f"{path}: 记录的 V / step* 与内层评分表重算不符")
    if j.get("boundary") != (step == TR.CONFIG["max_steps"]):
        P.append(f"{path}: 边界标记与 step* 不符")
    pt, npz = path.with_suffix(".pt"), path.with_suffix(".npz")
    if not pt.exists() or RUN.file_sha(pt) != j.get("pt_sha256"):
        P.append(f"{pt}: 缺失或与记录的 pt_sha256 不符")
    else:
        ck = torch.load(pt, weights_only=False)
        if ck.get("step") != step or ck.get("form", form) != form or \
                TR1.state_dict_sha(ck["state_dict"]) != j.get("state_sha256"):
            P.append(f"{pt}: 权重的步数、形式或哈希与 step* 记录不符")
    if not npz.exists() or RUN.file_sha(npz) != j.get("npz_sha256"):
        P.append(f"{npz}: 缺失或与记录的 npz_sha256 不符")


def verify(out) -> list:
    """完整性核验 → 问题列表（空 = 通过）。"""
    out = Path(out)
    try:
        I = RUN.read_inputs(out)
    except RuntimeError as err:
        return [str(err)]
    P = []
    for group in ("data_files", "calib_files", "h_files", "deps_files", "g2_files"):
        for rel, h in I.get(group, {}).items():
            f = RUN.ROOT / rel
            if not f.exists() or RUN.file_sha(f) != h:
                P.append(f"输入文件自 prepare 以来已变或缺失：{rel}")
    code_now, cfg_now = TR.code_sha(), TR.config_hash()
    if I.get("code_sha") != code_now:
        P.append("代码自 prepare 以来已变")
    if I.get("env") != R1.env_versions():
        P.append("库版本自 prepare 以来已变")
    if I.get("roster") != D2.ROSTER:
        P.append("冻结名单自 prepare 以来已变")
    d = RUN._load(RUN.deadline_file(out))
    if d and (d.get("code_sha"), d.get("config_hash")) != (code_now, cfg_now):
        P.append("deadline.json 属于另一版冻结的实验")
    for key, path in artifacts(out).items():
        j = RUN._load(path)
        if j is None:
            P.append(f"缺产物或无法解析：{path}")
            continue
        ident = _ident(key)
        if {k: j.get(k) for k in ident} != ident:
            P.append(f"{path}: 身份字段与路径不符")
        if j.get("code_sha") != code_now:
            P.append(f"{path}: code_sha 与当前代码不一致")
        if j.get("config_hash") != cfg_now:
            P.append(f"{path}: config_hash 与冻结的选模规则不符")
        _check_inputs(P, out, I, key, path, j)
        _check_rows(P, path, j, key[0], ident["fold"])
        if key[0] in ("nom-final", "inc-final"):
            _check_final(P, out, key, path, j)
    return P


def run_status(out) -> tuple:
    """本轮是否在截止时间内完成：有「未完成」标记、缺截止时间记录、或任一产物晚于截止时间 ⇒ 未完成。"""
    out = Path(out)
    mark = RUN.incomplete_file(out)
    if mark.exists():
        return False, json.loads(mark.read_text()).get("reason", "已标记未完成")
    d = RUN._load(RUN.deadline_file(out))
    if not d:
        return False, "缺本轮截止时间记录（deadline.json）"
    late = []
    for p in artifacts(out).values():
        j = RUN._load(p)
        if j is not None and j.get("finished_at", float("inf")) > d["deadline"]:
            late.append(p)
    if late:
        return False, f"{len(late)} 份产物晚于本轮截止时间（如 {late[0].relative_to(out)}）"
    return True, ""


# ── 判据（纯函数）─────────────────────────────────────────────────────────
def seed_mean_rho(per_seed: dict) -> dict:
    """{fold: {seed: 逐集行}} → {fold: 各种子池化 ρ 的平均}（先对种子平均误差，不平均预测）。"""
    return {f: float(np.mean([TR.pooled_rho(rows) for rows in by_seed.values()])) for f, by_seed in per_seed.items()}


def per_seed_rho(per_seed: dict) -> dict:
    return {f: {s: TR.pooled_rho(rows) for s, rows in by_seed.items()} for f, by_seed in per_seed.items()}


def gate(rb: dict) -> dict:
    """rb = {预测: {fold: ρ̄}} → 每个候选的 d_f、三折平均 d、逐折比较与是否通过。"""
    out = {}
    for form in FORMS:
        c, h, z = rb[form], rb["H"], rb["zero"]
        d = {f: (h[f] - c[f]) / h[f] for f in FOLDS}
        vs_h = {f: bool(c[f] < h[f]) for f in FOLDS}
        vs_z = {f: bool(c[f] < z[f]) for f in FOLDS}
        d_mean = float(np.mean([d[f] for f in FOLDS]))
        out[form] = {"d": d, "d_mean": d_mean, "per_fold_vs_H": vs_h, "per_fold_vs_zero": vs_z,
                     "pass": bool(d_mean >= TR.CONFIG["gate_d"] and all(vs_h.values()) and all(vs_z.values()))}
    return out


def nonlin_vs_cond(rb: dict) -> dict:
    """非线性差分相对条件线性的逐折相对误差降低（分母 = 条件线性的 ρ̄）与三折平均；三折是否都更好。"""
    n, c = rb["nonlinear"], rb["condlinear"]
    rel = {f: (c[f] - n[f]) / c[f] for f in FOLDS}
    return {"rel": rel, "rel_mean": float(np.mean([rel[f] for f in FOLDS])),
            "better_all": all(n[f] < c[f] for f in FOLDS)}


def choose(g: dict, rb: dict):
    """→ "nonlinear" / "condlinear" / None（两个都不通过 ⇒ 阶段二不通过）。"""
    ok = [f for f in FORMS if g[f]["pass"]]
    if not ok:
        return None
    if len(ok) == 1:
        return ok[0]
    v = nonlin_vs_cond(rb)
    return "nonlinear" if v["better_all"] and v["rel_mean"] >= TR.CONFIG["nonlin_margin"] else "condlinear"


def beta(rows):
    """投影尺度 β = Σ_e 分子 / Σ_e 分母；增量恒为 0（分母 = 0）⇒ None（N/A）。"""
    num, den = sum(r["num"] for r in rows), sum(r["den"] for r in rows)
    return None if den == 0 else num / den


def beta_seed_mean(by_seed: dict) -> dict:
    """{seed: 逐集行} → 逐种子 β、有定义种子的平均与个数（全部 N/A ⇒ 平均 None）。"""
    per = {s: beta(rows) for s, rows in by_seed.items()}
    ok = [b for b in per.values() if b is not None]
    return {"per_seed": per, "mean": float(np.mean(ok)) if ok else None, "n_defined": len(ok)}


def lead_rho(by_seed: dict) -> dict:
    """{seed: 逐集行} → 按前瞻分组的 ρ（各组分子分母先对种子平均再池化）。"""
    acc, k = {}, len(by_seed)
    for rows in by_seed.values():
        for r in rows:
            for b, (n, d) in r["lead"].items():
                a = acc.setdefault(b, [0.0, 0.0])
                a[0] += n / k
                a[1] += d / k
    return {b: n / d for b, (n, d) in acc.items()}


# ── 汇总 ──────────────────────────────────────────────────────────────────
def collect(out) -> dict:
    """读终训与参照产物 → 逐种子的行（调用前须已通过 `verify`）。"""
    A = {k: RUN._load(p) for k, p in artifacts(out).items()}
    held = {p: {f: {} for f in FOLDS} for p in PREDICTORS}
    aux = {p: {f: {} for f in FOLDS} for p in PREDICTORS}
    braw = {p: {f: {} for f in FOLDS} for p in ("H",) + FORMS}
    q1 = {f: {} for f in FOLDS}
    steps = {b: {f: {} for f in FOLDS} for b in ("nominal",) + FORMS}
    for f in FOLDS:
        for s in SEEDS:
            n = A[("nom-final", f, s)]
            held["zero"][f][s], held["H"][f][s] = n["zero"], n["H"]
            aux["zero"][f][s], aux["H"][f][s], braw["H"][f][s] = n["aux_zero"], n["aux_H"], n["aux_H_beta"]
            q1[f][s] = n["q1"]
            steps["nominal"][f][s] = {"step_star": n["step_star"], "boundary": n["boundary"]}
            for form in FORMS:
                c = A[("inc-final", form, f, s)]
                held[form][f][s], aux[form][f][s], braw[form][f][s] = c["cand"], c["aux_cand"], c["aux_beta"]
                steps[form][f][s] = {"step_star": c["step_star"], "boundary": c["boundary"]}
    return {"held": held, "aux": aux, "beta": braw, "q1": q1, "steps": steps,
            "g2": {f: A[("refs", f)]["episodes"] for f in FOLDS}}


def aux_table(aux: dict, braw: dict) -> dict:
    """ab4 P2 / P3 分开：每折终训模型上的 ρ̄（种子平均）与 β（逐种子 + 有定义种子的平均）。"""
    arm = lambda rows, a: [r for r in rows if r["arm"] == a]
    return {a: {"rho": {p: seed_mean_rho({f: {s: arm(rows, a) for s, rows in aux[p][f].items()} for f in FOLDS})
                        for p in PREDICTORS},
                "beta": {p: {f: beta_seed_mean({s: arm(rows, a) for s, rows in braw[p][f].items()}) for f in FOLDS}
                         for p in braw}}
            for a in AUX_ARMS}


def summarize(out) -> dict:
    C = collect(out)
    rb = {p: seed_mean_rho(C["held"][p]) for p in PREDICTORS}
    g = gate(rb)
    choice = choose(g, rb)
    return {"rho_bar": rb, "rho_per_seed": {p: per_seed_rho(C["held"][p]) for p in PREDICTORS},
            "gate": g, "nonlin_vs_cond": nonlin_vs_cond(rb), "choice": choice, "stage2_pass": choice is not None,
            "q1_recheck": {"nominal": seed_mean_rho(C["q1"]), "g2": {f: TR.pooled_rho(C["g2"][f]) for f in FOLDS}},
            "aux": aux_table(C["aux"], C["beta"]),
            "lead": {p: lead_rho({s: [r for f in FOLDS for r in C["held"][p][f][s]] for s in SEEDS})
                     for p in PREDICTORS},
            "steps": C["steps"]}


def markdown(s: dict) -> str:
    rho = lambda x: f"{x:.4f}"
    pct = lambda x: f"{x * 100:+.2f}%"
    fold_head = "| " + " | ".join(FOLDS) + " |"
    L = ["## 阶段二（Q2）开发检验汇总", "",
         "⚠ 这是开发检验，不是密封确认，也不是补正因果效应的直接验证。5% 与 2% 是事前的工程推进 / 选择阈值，"
         "不等于统计显著，也不能换算成真机改善。结论只在「维持 ab5/ab6 装置状态」的前提下有意义。", "",
         "### 留出会话补正运行上的 ρ̄（先对种子平均；越小越好）", "", "| 预测 " + fold_head, "|---" * 4 + "|"]
    for p in PREDICTORS:
        L.append(f"| {LABEL[p]} | " + " | ".join(rho(s["rho_bar"][p][f]) for f in FOLDS) + " |")
    L += ["", "### 判据（每个候选单独判断，三条都满足才通过）", "",
          "| 候选 | " + " | ".join(f"d_{f}" for f in FOLDS) + " | 三折平均 d | 逐折 < 固定 H | 逐折 < 零响应 | 判定 |",
          "|---" * 8 + "|"]
    for form in FORMS:
        g = s["gate"][form]
        why = [] if g["pass"] else (
            ([f"平均 d < {TR.CONFIG['gate_d'] * 100:.0f}%"] if g["d_mean"] < TR.CONFIG["gate_d"] else [])
            + [f"{f} 不优于 H" for f in FOLDS if not g["per_fold_vs_H"][f]]
            + [f"{f} 不优于零响应" for f in FOLDS if not g["per_fold_vs_zero"][f]])
        yn = lambda d: "是" if all(d.values()) else "否（" + "、".join(f for f in FOLDS if not d[f]) + "）"
        L.append(f"| {LABEL[form]} | " + " | ".join(pct(g["d"][f]) for f in FOLDS) + f" | {pct(g['d_mean'])} | "
                 f"{yn(g['per_fold_vs_H'])} | {yn(g['per_fold_vs_zero'])} | "
                 + ("**过**" if g["pass"] else "不过（" + "；".join(why) + "）") + " |")
    v = s["nonlin_vs_cond"]
    verdict = {None: "两个候选都不通过 ⇒ **阶段二不通过**，不进入阶段三。",
               "condlinear": "保留**条件线性**。", "nonlinear": "保留**非线性差分**（三折都更好，且平均相对误差再降 ≥ 2%）。"}
    L += ["", "### 两种形式之间（事前定死；不逐折挑赢家）", "",
          "- 非线性差分相对条件线性的相对误差降低（分母 = 条件线性的 ρ̄）：" + "、".join(
              f"{f} {pct(v['rel'][f])}" for f in FOLDS) + f"；三折平均 {pct(v['rel_mean'])}；三折都更好："
          + ("是" if v["better_all"] else "否"),
          f"- 结论：{verdict[s['choice']]}",
          "- 对照结论只能是「两套具体响应模型谁的折外预测更好」。两者还差着扰动记忆长度（61 对 20 帧）、四基核约束与参数化，"
          "不能把胜负单独解释成非线性是否有用。",
          "- 通过只意味着可以进入阶段三的离线 π 研究，不代表 π 会在真机上有效。",
          "", "## 只报告（不作判据）", "",
          "### 留出裸跑上的 Q1 复查（名义 2a 种子平均 ρ；G-2 同口径窗口重预测作背景）", "",
          "| 预测 " + fold_head, "|---" * 4 + "|",
          f"| {LABEL['nominal']} | " + " | ".join(rho(s["q1_recheck"]["nominal"][f]) for f in FOLDS) + " |",
          "| G-2 | " + " | ".join(rho(s["q1_recheck"]["g2"][f]) for f in FOLDS) + " |",
          "", "### 逐种子 ρ（留出补正运行；每格为种子 " + " / ".join(str(k) for k in SEEDS) + "）", "",
          "| 预测 " + fold_head, "|---" * 4 + "|"]
    for p in PREDICTORS:
        ps = s["rho_per_seed"][p]
        L.append(f"| {LABEL[p]} | " + " | ".join(" / ".join(rho(ps[f][k]) for k in SEEDS) for f in FOLDS) + " |")
    L += ["", "### ab4 辅助参考（旧装置；复用每折终训模型；P1 不参与；不用于选步、选模或判决）", ""]
    na = lambda b: "N/A" if b["mean"] is None else (
        f"{b['mean']:.3f}" + ("" if b["n_defined"] == len(SEEDS) else f"（{b['n_defined']}/{len(SEEDS)} 个种子有定义）"))
    for a in AUX_ARMS:
        t = s["aux"][a]
        L += [f"**{a}**：ρ̄（种子平均）与投影尺度 β（有定义种子的平均）；列 = 各折的终训模型", "",
              "| 预测 | " + " | ".join(f"ρ̄ {f}" for f in FOLDS) + " | " + " | ".join(f"β {f}" for f in FOLDS) + " |",
              "|---" * 7 + "|"]
        for p in PREDICTORS:
            bc = [na(t["beta"][p][f]) if p in t["beta"] else "—" for f in FOLDS]
            L.append(f"| {LABEL[p]} | " + " | ".join(rho(t["rho"][p][f]) for f in FOLDS) + " | " + " | ".join(bc) + " |")
        L.append("")
    L += ["- β 只是投影尺度，不是「方向和大小都正确」：β ≈ 1 时仍可能漏掉很大的正交响应；增量恒为 0（选中第 0 步）时 β 不定义，记 N/A。",
          "- 闭环运行下，随机探针不能自动保证与名义预测误差无关；扣掉名义预测后的残差不能称为「已分离的真实增量」。",
          "- P3/P2 的 β 比值不是严格线性检验（指令、种子不同，P3 还有裁剪）；P3 是跨幅值的压力参考。"
          "历史的 0.955、1.013 只作历史记录，不与本表直接比较。",
          "", "### 按前瞻分组的 ρ（留出补正运行按集池化、分子分母先对种子平均；带通混合相邻帧 ⇒ 近似）", ""]
    bins = list(s["lead"][PREDICTORS[0]])
    L += ["| 预测 | " + " | ".join(bins) + " |", "|---" * (len(bins) + 1) + "|"]
    for p in PREDICTORS:
        L.append(f"| {LABEL[p]} | " + " | ".join(rho(s["lead"][p][b]) for b in bins) + " |")
    top = TR.CONFIG["max_steps"]
    L += ["", "### step*（每格为种子 " + " / ".join(str(k) for k in SEEDS) + f"；† = 落在搜索边界 {top} 步）", "",
          "| 分支 " + fold_head, "|---" * 4 + "|"]
    hit = zero = False
    for b in ("nominal",) + FORMS:
        cells = []
        for f in FOLDS:
            ks = [s["steps"][b][f][k] for k in SEEDS]
            hit |= any(k["boundary"] for k in ks)
            zero |= b != "nominal" and any(k["step_star"] == 0 for k in ks)
            cells.append(" / ".join(f"{k['step_star']}{'†' if k['boundary'] else ''}" for k in ks))
        L.append(f"| {LABEL[b]} | " + " | ".join(cells) + " |")
    if hit:
        L += ["", f"† 最佳点落在搜索边界（{top} 步），尚不能确认训练充分；按预注册不追加步数。"]
    if zero:
        L += ["", "增量选中第 0 步 = 保留零响应（未训练），是有效选择；该处候选与零响应逐集相同。"]
    return "\n".join(L)


def _key_str(obj):
    """JSON 只接受字符串键：把整数种子键转成字符串。"""
    if isinstance(obj, dict):
        return {str(k): _key_str(v) for k, v in obj.items()}
    return obj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    ok, why = run_status(out)
    if not ok:
        R1.atomic_json(out / "summary.json", {"status": "未完成", "reason": why})
        print(f"**本轮未完成**：{why} —— 按预注册记为未完成，不作通过判决。", flush=True)
        return 3
    problems = verify(out)
    if problems:
        R1.atomic_json(out / "summary.json", {"status": "核验未通过", "problems": problems})
        print("产物不完整或不一致，拒绝汇总：\n  " + "\n  ".join(problems), flush=True)
        return 4
    s = {"status": "完成", **summarize(out)}
    R1.atomic_json(out / "summary.json", _key_str(s))
    md = markdown(s)
    (out / "summary.md").write_text(md + "\n")
    print(md, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
