"""阶段一汇总：逐会话误差比、推进判据（§5 #1）、b 相对 a、前瞻分层、辅助起点。

误差比 `ρ_s = Σ_e ‖P(gap−ê)‖² / Σ_e ‖P(gap)‖²`（会话 s 内按集池化；`redu = 1 − ρ`）。
训练种子：取各种子 `ρ_s` 的平均（= 平均误差，不平均预测）。
**推进判据**（每个候选单独判，不要求六个全过）：5 个留出会话都 `ρ_s ≤ ρ_s(G-2)`，
且 5 会话等权平均 `mean ρ_s < mean ρ_s(G-2)`。G-2 = 同口径窗口重预测。
⚠ 开发筛查规则，不代表统计上已证明泛化。
汇总前强制完整性检查：6 模型 × 3 种子 × 5 折 + 3 个参照 × 5 折；每个作业的权重与预测文件存在且
哈希与记录一致；候选的 config_hash 唯一；全部产物的 code_sha = 当前代码；每份产物恰好覆盖同一组 100 集。
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
import q1_train as TR          # noqa: E402

CANDIDATES = ("1a", "1b", "2a", "2b", "3a", "3b")
REFS = ("g2", "g2_whole", "ar")
SEEDS = ("0", "1", "2")
SESS = tuple(f"{s}/{a}" for s, a in D.SESSIONS)
N_EPISODES = sum(D.EXPECTED_COUNTS.values())


def file_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(raw) -> list:
    """完整性检查 → 问题列表（空 = 通过）。"""
    P = []
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
            for j in js:
                if any(r["sess"] != j["fold"] for r in j["episodes"]):
                    P.append(f"{j['_path']}: 含非留出会话的集")
                if j.get("code_sha") != code_now:
                    P.append(f"{j['_path']}: code_sha 与当前代码不一致")
                if m not in CANDIDATES:
                    continue
                ts = j.get("train_sessions", [])
                if j["fold"] in ts or len(ts) != len(SESS) - 1:
                    P.append(f"{j['_path']}: 训练会话 {ts} 不对")
                for key, suf in (("pt_sha256", ".pt"), ("npz_sha256", ".npz")):
                    f = Path(j["_path"]).with_suffix(suf)
                    if not f.exists():
                        P.append(f"缺 {f}")
                    elif file_sha(f) != j.get(key):
                        P.append(f"{f}: 与记录的 {key} 不符")
    cfg = {j["config_hash"] for m in CANDIDATES for js in raw.get(m, {}).values() for j in js}
    if len(cfg) > 1:
        P.append(f"候选作业的 config_hash 不唯一：{sorted(cfg)}")
    sets = {tuple(sorted(r["name"] for j in js for r in j["episodes"]))
            for by_seed in raw.values() for js in by_seed.values()}
    if len(sets) > 1:
        P.append("各产物覆盖的集合不一致")
    return P


def load_runs(out) -> dict:
    """→ {model: {seed: [逐集行]}}。先做完整性检查；任何问题 ⇒ RuntimeError（列出全部）。"""
    raw: dict = {}
    for p in sorted((Path(out) / "jobs").glob("*/seed*/*.json")) + \
            sorted((Path(out) / "refs").glob("*/*.json")):
        j = json.loads(p.read_text())
        j["_path"] = str(p)
        raw.setdefault(j["model"], {}).setdefault(str(j["seed"]), []).append(j)
    problems = verify(raw)
    if problems:
        raise RuntimeError("产物不完整或不一致：\n  " + "\n  ".join(problems))
    return {m: {s: [r for j in js for r in j["episodes"]] for s, js in by_seed.items()}
            for m, by_seed in raw.items()}


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


def summarize(runs) -> dict:
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
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    s = summarize(load_runs(a.out))
    (Path(a.out) / "summary.json").write_text(json.dumps(s, indent=1, ensure_ascii=False))
    print(markdown(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
