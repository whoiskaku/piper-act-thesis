"""阶段一汇总：逐会话误差比、推进判据（§5 #1）、b 相对 a、前瞻分层、辅助起点。

误差比 `ρ_s = Σ_e ‖P(gap−ê)‖² / Σ_e ‖P(gap)‖²`（会话 s 内按集池化；`redu = 1 − ρ`）。
训练种子：取各种子 `ρ_s` 的平均（= 平均误差，不平均预测）。
**推进判据**（每个候选单独判，不要求六个全过）：5 个留出会话都 `ρ_s ≤ ρ_s(G-2)`，
且 5 会话等权平均 `mean ρ_s < mean ρ_s(G-2)`。G-2 = 同口径窗口重预测。
⚠ 开发筛查规则，不代表统计上已证明泛化。
汇总前强制核验（任一不满足即拒绝汇总，并列出全部问题）：
- 6 模型 × 3 种子 × 5 折 + 3 个参照 × 5 折；每份产物恰好覆盖同一组 100 集；训练会话不含留出会话。
- 候选终训的 `config_hash` = **冻结的选模规则**（不要求各作业的 step* 相同）；全部 `code_sha` = 当前代码。
- 输入自 prepare 以来未变（数据、依赖、G-2、传感器层文件逐一重算哈希）；每个作业的输入指纹与当前一致。
- 由内层评分表重算 `V` 与 step*，与记录一致；`.pt` 的步数 = step*、权重哈希与记录一致；预测文件哈希一致。
- 每份产物逐会话覆盖**冻结名单**（`q1_roster.json`）上的集，不只是总数 100。
本轮未完成（有「未完成」标记、缺截止时间记录、或任一产物晚于截止时间）⇒ 只写「未完成」，**不作通过判决**。
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import q1_data as D            # noqa: E402
import q1_run as RUN           # noqa: E402
import q1_train as TR          # noqa: E402

CANDIDATES = ("1a", "1b", "2a", "2b", "3a", "3b")
REFS = ("g2", "g2_whole", "ar")
SEEDS = ("0", "1", "2")
SESS = tuple(f"{s}/{a}" for s, a in D.SESSIONS)
N_EPISODES = sum(D.EXPECTED_COUNTS.values())


def file_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_final(j: dict, out: Path, I: dict, P: list) -> None:
    """候选终训的选模记录：配置、输入指纹、由内层评分表重算的 step*、权重与预测文件。"""
    import torch
    m, fold, seed, path = j["model"], j["fold"], int(j["seed"]), Path(j["_path"])
    if j.get("config_hash") != TR.config_hash():
        P.append(f"{path}: config_hash 与冻结的选模规则不符")
    if j.get("train_sessions") != list(RUN.train_sessions(fold)):
        P.append(f"{path}: 训练会话 {j.get('train_sessions')} 不对")
    exp = RUN.final_inputs(out, I, m, fold, seed)
    if exp is None or j.get("inputs") != exp or j.get("input_fp") != RUN.fp_of(exp):
        P.append(f"{path}: 输入指纹与当前的内层评分表 / 传感器层 / 数据不符")
    try:
        V, step = RUN.selection(out, m, fold, seed)
    except RuntimeError as err:
        P.append(str(err))
        return
    if j.get("V") != {str(s): V[s] for s in sorted(V)} or j.get("step_star") != step:
        P.append(f"{path}: 记录的 V / step* 与内层评分表重算不符")
    if j.get("boundary") != (step == TR.CONFIG["max_steps"]):
        P.append(f"{path}: 边界标记与 step* 不符")
    pt, npz = path.with_suffix(".pt"), path.with_suffix(".npz")
    if not pt.exists() or file_sha(pt) != j.get("pt_sha256"):
        P.append(f"{pt}: 缺失或与记录的 pt_sha256 不符")
    else:
        ck = torch.load(pt, weights_only=False)
        if ck.get("step") != step or TR.state_dict_sha(ck["state_dict"]) != j.get("state_sha256"):
            P.append(f"{pt}: 权重的步数或哈希与 step* 记录不符")
    if not npz.exists() or file_sha(npz) != j.get("npz_sha256"):
        P.append(f"{npz}: 缺失或与记录的 npz_sha256 不符")


def verify(raw, out) -> list:
    """完整性检查 → 问题列表（空 = 通过）。"""
    out = Path(out)
    try:
        I = RUN.read_inputs(out)
    except RuntimeError as err:
        return [str(err)]
    P = []
    for key in ("data_files", "deps_files", "g2_files"):
        for rel, h in I.get(key, {}).items():
            f = RUN.ROOT / rel
            if not f.exists() or file_sha(f) != h:
                P.append(f"输入文件自 prepare 以来已变或缺失：{rel}")
    for st, h in I["sensor"].items():
        f = out / "sensor" / f"{st}.npz"
        if not f.exists() or file_sha(f) != h:
            P.append(f"传感器层文件自 prepare 以来已变或缺失：{f.name}")
    code_now = TR.code_sha()
    for m in CANDIDATES + REFS:
        by_seed = raw.get(m, {})
        want = SEEDS if m in CANDIDATES else ("-",)
        if sorted(by_seed) != sorted(want):
            P.append(f"{m}: 种子 {sorted(by_seed)}，应为 {list(want)}")
        for s, js in by_seed.items():
            if sorted(j["fold"] for j in js) != sorted(SESS):
                P.append(f"{m} seed {s}: 折 {sorted(j['fold'] for j in js)} 不齐")
            names = [r["name"] for j in js for r in j["episodes"]]
            if len(names) != N_EPISODES or len(set(names)) != N_EPISODES:
                P.append(f"{m} seed {s}: {len(set(names))} 个不同的集，应为 {N_EPISODES}")
            for x in SESS:
                got = sorted(r["name"] for j in js for r in j["episodes"] if r["sess"] == x)
                if got != sorted(D.ROSTER[x]):
                    P.append(f"{m} seed {s}: {x} 的集与冻结名单不符（{len(got)} 集，应为 {len(D.ROSTER[x])}）")
            for j in js:
                if any(r["sess"] != j["fold"] for r in j["episodes"]):
                    P.append(f"{j['_path']}: 含非留出会话的集")
                if j.get("code_sha") != code_now:
                    P.append(f"{j['_path']}: code_sha 与当前代码不一致")
                if m in CANDIDATES:
                    _check_final(j, out, I, P)
                    continue
                if j.get("input_fp") != RUN.fp_of(RUN.refs_inputs(I)):
                    P.append(f"{j['_path']}: 参照的输入指纹与当前不符")
                ts = j.get("train_sessions", [])
                if m == "ar" and (j["fold"] in ts or len(ts) != len(SESS) - 1):
                    P.append(f"{j['_path']}: AR 的训练会话 {ts} 不对")
    sets = {tuple(sorted(r["name"] for j in js for r in j["episodes"]))
            for by_seed in raw.values() for js in by_seed.values()}
    if len(sets) > 1:
        P.append("各产物覆盖的集合不一致")
    return P


def _read_jobs(out) -> dict:
    raw: dict = {}
    for p in sorted((Path(out) / "jobs").glob("*/seed*/*.json")) + \
            sorted((Path(out) / "refs").glob("*/*.json")):
        j = json.loads(p.read_text())
        j["_path"] = str(p)
        raw.setdefault(j["model"], {}).setdefault(str(j["seed"]), []).append(j)
    return raw


def load_runs(out) -> dict:
    """→ {model: {seed: [逐集行]}}。先做完整性检查；任何问题 ⇒ RuntimeError（列出全部）。"""
    raw = _read_jobs(out)
    problems = verify(raw, out)
    if problems:
        raise RuntimeError("产物不完整或不一致：\n  " + "\n  ".join(problems))
    return {m: {s: [r for j in js for r in j["episodes"]] for s, js in by_seed.items()}
            for m, by_seed in raw.items()}


def run_status(out) -> tuple[bool, str]:
    """本轮是否在截止时间内完成：有「未完成」标记、缺截止时间记录、或任一产物晚于截止时间 ⇒ 未完成。"""
    out = Path(out)
    mark = RUN.incomplete_file(out)
    if mark.exists():
        return False, json.loads(mark.read_text()).get("reason", "已标记未完成")
    dp = RUN.deadline_file(out)
    if not dp.exists():
        return False, "缺本轮截止时间记录（deadline.json）"
    deadline = json.loads(dp.read_text())["deadline"]
    files = (sorted((out / "jobs").glob("*/seed*/*.json")) + sorted((out / "refs").glob("*/*.json"))
             + sorted((out / "inner").glob("*/seed*/*.json")))
    late = [p for p in files if json.loads(p.read_text()).get("finished_at", float("inf")) > deadline]
    if late:
        return False, f"{len(late)} 份产物晚于本轮截止时间（如 {late[0].relative_to(out)}）"
    return True, ""


def load_selection(out) -> dict:
    """→ {model: {fold: {seed: {step_star, boundary, V}}}}（候选终训的选步记录）。"""
    sel: dict = {}
    for m, by_seed in _read_jobs(out).items():
        if m not in CANDIDATES:
            continue
        for s, js in by_seed.items():
            for j in js:
                sel.setdefault(m, {}).setdefault(j["fold"], {})[s] = {
                    "step_star": j["step_star"], "boundary": j["boundary"], "V": j["V"]}
    return sel


def rho_by_session(rows, key: str = "num") -> dict:
    acc: dict = {}
    for r in rows:
        a = acc.setdefault(r["sess"], [0.0, 0.0])
        a[0] += r[key]
        a[1] += r["den"]
    return {s: n / d for s, (n, d) in acc.items()}


def seed_mean(by_seed, key: str = "num") -> dict:
    per = [rho_by_session(rows, key) for rows in by_seed.values()]
    return {s: float(np.mean([p[s] for p in per])) for s in per[0]}


def gate(rho_c: dict, rho_g2: dict) -> dict:
    per = {s: bool(rho_c[s] <= rho_g2[s]) for s in rho_g2}
    mc = float(np.mean([rho_c[s] for s in rho_g2]))
    mg = float(np.mean(list(rho_g2.values())))
    return {"per_session": per, "mean": mc, "mean_g2": mg, "pass": all(per.values()) and mc < mg}


def b_vs_a(rho_a: dict, rho_b: dict) -> dict:
    """逐会话相对变化 `(ρ_b − ρ_a)/ρ_a`；负 = b 更好。**只报告，不替代推进判据。**"""
    return {s: (rho_b[s] - rho_a[s]) / rho_a[s] for s in rho_a}


def lead_rho(by_seed) -> dict:
    acc: dict = {}
    k = len(by_seed)
    for rows in by_seed.values():
        for r in rows:
            for b, (n, d) in r["lead"].items():
                a = acc.setdefault(b, [0.0, 0.0])
                a[0] += n / k
                a[1] += d / k
    return {b: n / d for b, (n, d) in acc.items()}


def summarize(runs, selection=None) -> dict:
    g2 = seed_mean(runs["g2"])
    out = {"g2": g2, "ar": seed_mean(runs["ar"]), "g2_whole": seed_mean(runs["g2_whole"]),
           "lead_refs": {m: lead_rho(runs[m]) for m in ("g2", "ar")},
           "aux_refs": {m: seed_mean(runs[m], "num_aux") for m in ("g2", "ar")},
           "candidates": {}, "b_vs_a": {}}
    for m in CANDIDATES:
        if m in runs:
            rho = seed_mean(runs[m])
            out["candidates"][m] = {
                "rho": rho, "gate": gate(rho, g2), "rho_aux": seed_mean(runs[m], "num_aux"),
                "rho_per_seed": {s: rho_by_session(r) for s, r in runs[m].items()},
                "lead": lead_rho(runs[m])}
    for f in ("1", "2", "3"):
        c = out["candidates"]
        if f + "a" in c and f + "b" in c:
            out["b_vs_a"][f] = b_vs_a(c[f + "a"]["rho"], c[f + "b"]["rho"])
    if selection is not None:
        out["selection"] = selection
    return out


def markdown(s: dict) -> str:
    pct = lambda r: f"{(1 - r) * 100:+.2f}%"
    head = "| 模型 | " + " | ".join(SESS) + " | 等权平均 | 推进判据 |"
    sep = "|" + "---|" * (len(SESS) + 3)
    L = ["### 留出会话 redu（= 1 − ρ，种子平均）", "", head, sep]

    def row(name, rho, verdict=""):
        return f"| {name} | " + " | ".join(pct(rho[x]) for x in SESS) + \
            f" | {pct(float(np.mean([rho[x] for x in SESS])))} | {verdict} |"

    L.append(row("G-2（同口径重预测）", s["g2"], "比较对象"))
    for m, c in s["candidates"].items():
        g = c["gate"]
        bad = [x for x in SESS if not g["per_session"][x]]
        L.append(row(m, c["rho"], "**过**" if g["pass"] else f"不过（{', '.join(bad) or '平均未严格改善'}）"))
    L.append(row("折内 AR 参照", s["ar"], "参照"))
    L.append(row("G-2 旧口径（整集前向，只对照）", s["g2_whole"], "不作判据"))
    L += ["", "### 状态历史：b 相对 a 的逐会话变化（负 = b 更好；只报告）", "",
          "| 结构 | " + " | ".join(SESS) + " |", "|" + "---|" * (len(SESS) + 1)]
    for f, d in s["b_vs_a"].items():
        L.append(f"| {f}b vs {f}a | " + " | ".join(f"{d[x] * 100:+.1f}%" for x in SESS) + " |")
    sel = s.get("selection")
    if sel:
        top = TR.CONFIG["max_steps"]
        L += ["", "### step*（内层留一会话选出；每格为种子 0 / 1 / 2；† = 落在上限）", "",
              "| 模型 | " + " | ".join(SESS) + " |", "|" + "---|" * (len(SESS) + 1)]
        hit = False
        for m in CANDIDATES:
            if m not in sel:
                continue
            cells = []
            for x in SESS:
                ks = [sel[m][x][k] for k in sorted(sel[m].get(x, {}))]
                hit |= any(k["boundary"] for k in ks)
                cells.append(" / ".join(f"{k['step_star']}{'†' if k['boundary'] else ''}" for k in ks))
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        if hit:
            L += ["", f"† 最佳点落在搜索边界（{top} 步），尚不能确认训练充分；按预注册不追加步数。"]
        L += ["", "第 0 步的含义不统一：端到端 = 零预测（误差输出为零）；分层 = 伺服先验 + 已拟合的传感器层"
              "（关节层未训练）。不套用 F_inc「step* = 0」的解释。"]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ok, why = run_status(a.out)
    if not ok:
        (Path(a.out) / "summary.json").write_text(json.dumps(
            {"status": "未完成", "reason": why}, indent=1, ensure_ascii=False))
        print(f"**本轮未完成**：{why} —— 按预注册记为未完成，不作通过判决。")
        return 3
    s = {"status": "完成", **summarize(load_runs(a.out), load_selection(a.out))}
    (Path(a.out) / "summary.json").write_text(json.dumps(s, indent=1, ensure_ascii=False))
    print(markdown(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
