"""B4 的方块位置、分组、录制顺序与集号登记（方案 `reports/PLAN_b4.md` §5.2；实现计划任务 5）。

  python -u sim/b4_positions.py gen                       # 生成 outputs/b4/positions.json（已存在且不同就停）
  python -u sim/b4_positions.py block check|record <块>   # 给 scripts/run_b4.sh：下一块各条，每行「id cm 摆放说明」
  python -u sim/b4_positions.py register                  # 第二步录完、标完成败后：集号 → 位置 id，列出要重录的

位置沿示教那条线，在 0…+4 cm 内均匀抽、取到 0.1 cm（尺子能摆的精度）；坐标同桌上的标记（−4、−2、0、+2、+4 cm）。
用户 10-29 先定 −4…+4（正式基线验证过的范围），第一步检查后改为只放正的一侧：负侧的示教数据不好，ACT 在那里夹起后会卡在盒子旁。
打印时换成「从哪个标记往哪边挪多少 cm」。主位置 50 个：排序后每 10 个一段，
段内随机 6 : 2 : 2 分到训练 / 验证 / 测试；另备 6 个训练位置（spare）；第一步检查另用 10 个位置。
录制顺序：各组内随机排，再按组内名次均匀交错（三份交替录，避免某一份集中在某几天）。
集号对应：按录制顺序排队，成败标完后，失败（review_eval 阶段 < 3）的位置依次排到队尾重录；第 i 集 = 队列第 i 个。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs/b4"
POS = OUT / "positions.json"
REC = OUT / "recordings.json"
LR = Path.home() / ".cache/huggingface/lerobot/enolab-guo"
REVIEW = HERE.parent / "outputs/review"
SEED, LO, HI = 20261029, 0.0, 4.0
MARKS = (-4.0, -2.0, 0.0, 2.0, 4.0)                              # 1–5 号标记（cm）
CHECK_REPO = {1: "eval_pickcube_b4check_raw", 2: "eval_pickcube_b4check_shift", 3: "eval_pickcube_b4check_shift",
              4: "eval_pickcube_b4check_raw"}                     # 第一步 ABBA：A = 不提前，B = Shift
RECORD_REPO = "eval_pickcube_b4rec"
FINAL_ROT = [("shift", "net", "raw"), ("net", "raw", "shift"), ("raw", "shift", "net"), ("shift", "net", "raw")]


def final_blocks(d: dict) -> list[tuple[str, list[str]]]:
    """第五步 12 块：位置块 A（测试位置 1–5）、B（6–10）按 A、B、A、B；每个位置块内三个条件按 FINAL_ROT 轮换。→ [(条件, 5 个 id)]。"""
    split = {p["id"]: p["split"] for p in d["pool"]}
    ids = [i for i in d["order"] if split[i] == "test"]
    return [(c, ids[:5] if pb % 2 == 0 else ids[5:]) for pb in range(4) for c in FINAL_ROT[pb]]


def final_plan(d: dict | None = None) -> list[tuple[str, list[float]]]:
    d = d or load()
    cm = {p["id"]: p["cm"] for p in d["pool"]}
    return [(c, [cm[i] for i in ids]) for c, ids in final_blocks(d)]


def generate(seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    draw = lambda n: [float(x) for x in np.round(rng.uniform(LO, HI, n), 1)]   # noqa: E731
    main, spare, check = draw(50), draw(6), draw(10)
    split = [""] * 50
    for b in range(5):                                              # 排序后每 10 个一段，段内 6 : 2 : 2
        idx = sorted(range(50), key=lambda i: (main[i], i))[10 * b:10 * b + 10]
        for i, s in zip(rng.permutation(idx), ["train"] * 6 + ["val"] * 2 + ["test"] * 2):
            split[i] = s
    pool = ([{"id": f"p{i:02d}", "cm": c, "split": s} for i, (c, s) in enumerate(zip(main, split))]
            + [{"id": f"p{50 + i:02d}", "cm": c, "split": "spare"} for i, c in enumerate(spare)])
    key = {}
    for s in ("train", "val", "test", "spare"):                     # 组内随机名次 j → (j + U) / m，按它交错
        ids = [p["id"] for p in pool if p["split"] == s]
        for j, i in enumerate(rng.permutation(ids)):
            key[str(i)] = (j + rng.uniform()) / len(ids)
    return {"seed": seed, "check": check, "pool": pool, "order": sorted(key, key=key.get)}


def mark_desc(cm: float) -> str:
    """摆放说明（用胶带上的刻度）：从最近的标记（两个一样近时取靠中间的）往哪个标记那边挪多少。"""
    lab = lambda m: "0" if m == 0 else f"{m:+g}"   # noqa: E731
    k = min(range(len(MARKS)), key=lambda i: (abs(cm - MARKS[i]), abs(MARKS[i])))
    d = round(cm - MARKS[k], 1)
    if d == 0:
        return f"正对 {lab(MARKS[k])}"
    return f"从 {lab(MARKS[k])} 往 {lab(MARKS[k + 1] if d > 0 else MARKS[k - 1])} 那边挪 {abs(d):g} cm"


def _queue(order: list[str], stages: list[int]) -> list[str]:
    q = list(order)
    for i, st in enumerate(stages):
        if i >= len(q):
            raise SystemExit(f"⛔ 第 {i} 集没有排到位置：录的集数比队列长")
        if st < 3:
            q.append(q[i])
    return q


def assign(order: list[str], stages: list[int]) -> dict[str, int]:
    """stages = 第 0、1 … 集的阶段（全部标完）→ 成功的位置 id → 集号。"""
    q = _queue(order, stages)
    return {q[i]: i for i, st in enumerate(stages) if st >= 3}


def next_positions(d: dict, mode: str, n_done: int, stages: dict[int, int] | None = None,
                   block: int | None = None) -> list[tuple[str, float]]:
    """下一块（≤ 5 条）的 [(id, cm)]，按数据集里已录的集数 n_done 接着排。
    check：第 b 块（1–4）的数据集里第 i 集摆在 check[i]；前两块用 check[0:5]、后两块用 check[5:10]。
    record：先按顺序录完全部位置（一块不跨进重录）；之后的重录要等已录的集都标完成败。"""
    if mode == "check":
        lo, hi = (0, 5) if block in (1, 2) else (5, 10)
        return [(f"c{i}", d["check"][i]) for i in range(max(lo, n_done), hi)]
    cm = {p["id"]: p["cm"] for p in d["pool"]}
    if mode == "final":                               # 这个条件的数据集里，第 pb 个位置块占第 5·pb … 5·pb + 4 集
        _, ids = final_blocks(d)[block - 1]
        pb = (block - 1) // 3
        if n_done < 5 * pb:
            raise SystemExit(f"⛔ 第 {block} 块之前同一条件的块还没录完（已录 {n_done} 集）")
        return [(i, cm[i]) for i in ids[max(0, n_done - 5 * pb):]]
    order = d["order"]
    if n_done < len(order):
        return [(i, cm[i]) for i in order[n_done:n_done + 5]]   # 只取原顺序：第一轮的块不会混进重录
    stages = stages or {}
    missing = [i for i in range(n_done) if i not in stages]
    if missing:
        raise SystemExit(f"已录 {n_done} 集，还有 {len(missing)} 集没标成败（第 {missing[0]} 集起）：先用 review_eval.py 标完、"
                         "再 register，看要重录哪些")
    q = _queue(order, [stages[i] for i in range(n_done)])
    return [(i, cm[i]) for i in q[n_done:n_done + 5]]


def n_recorded(repo: str) -> int:
    info = LR / repo / "meta/info.json"
    return json.loads(info.read_text())["total_episodes"] if info.exists() else 0


def review_stages(repo: str) -> dict[int, int]:
    """scripts/review_eval.py 标的阶段：{集号: 阶段 0–3}。"""
    p = REVIEW / f"enolab-guo__{repo}_eval.csv"
    return {int(r[0]): int(r[1]) for r in csv.reader(p.open()) if len(r) >= 2} if p.exists() else {}


def load() -> dict:
    if not POS.exists():
        raise SystemExit(f"⛔ 没有 {POS}：先运行 python -u sim/b4_positions.py gen")
    return json.loads(POS.read_text())


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["gen"]:
        text = json.dumps(generate(), indent=1)
        if POS.exists() and POS.read_text() != text:
            raise SystemExit(f"⛔ {POS} 已存在且内容不同，不覆盖")
        POS.parent.mkdir(parents=True, exist_ok=True)
        POS.write_text(text)
        d = json.loads(text)
        n = {s: sum(p["split"] == s for p in d["pool"]) for s in ("train", "val", "test", "spare")}
        print(f"{POS}：训练 {n['train']} / 验证 {n['val']} / 测试 {n['test']} / 备用 {n['spare']}，第一步检查 {len(d['check'])} 个位置")
    elif argv[:1] == ["block"] and len(argv) == 3:
        mode, b = argv[1], int(argv[2])
        if mode == "final" and b > 12:
            return 0
        repo = (CHECK_REPO[b] if mode == "check" else
                f"eval_pickcube_b4final_{final_blocks(load())[b - 1][0]}" if mode == "final" else RECORD_REPO)
        pos = next_positions(load(), mode, n_recorded(repo), review_stages(repo), b)
        print("\n".join(f"{i} {c:+g} {mark_desc(c)}" for i, c in pos))
    elif argv[:1] == ["cond"] and len(argv) == 2:       # 第五步第 b 块的条件（shift / net / raw），给 run_b4.sh
        print(final_blocks(load())[int(argv[1]) - 1][0])
    elif argv[:1] == ["register"]:
        d, stages = load(), review_stages(RECORD_REPO)
        n_done = n_recorded(RECORD_REPO)
        if any(i not in stages for i in range(n_done)):
            raise SystemExit(f"⛔ 已录 {n_done} 集，标了 {len(stages)} 集：先用 review_eval.py 标完")
        st = [stages[i] for i in range(n_done)]
        ok = assign(d["order"], st)
        REC.write_text(json.dumps({i: {"repo": RECORD_REPO, "episode": e, "stage": st[e]} for i, e in ok.items()}, indent=1))
        todo = _queue(d["order"], st)[n_done:]
        fails = sum(s < 3 for s in st)
        print(f"已录 {n_done} 集：成功 {len(ok)}、失败 {fails}（闭环 Shift 的失败率 {fails / max(n_done, 1):.1%}）→ {REC}")
        print(f"还要重录 {len(todo)} 个位置：{' '.join(todo) or '无'}")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
