"""`b4_positions.py`（方块位置、分组、录制顺序与集号登记，方案 `reports/PLAN_b4.md` §5.2）的测试。纯计算，不读数据。"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b4_positions as P  # noqa: E402


def test_positions_deterministic_and_in_range():
    """用户 10-29（第一步检查之后）：位置只放在正的一侧 0 … +4 cm（负侧的示教数据不好，ACT 在那里容易卡住）。"""
    a = P.generate()
    assert a == P.generate()
    cms = [p["cm"] for p in a["pool"]] + a["check"]
    assert all(0 <= c <= 4 for c in cms) and all(round(c, 1) == c for c in cms)
    assert min(cms) < 1 and max(cms) > 3                                   # 真的铺满整段


def test_mark_desc_names_nearest_mark_and_direction():
    """摆放说明用胶带上的刻度（−4、−2、0、+2、+4）：从最近的标记往哪个标记那边挪多少；
    正好在两个标记中间时，以靠中间（0）的那个为准。"""
    assert P.mark_desc(1.4) == "从 +2 往 0 那边挪 0.6 cm"                 # 离 +2 只有 0.6 cm，比离 0 近
    assert P.mark_desc(0.7) == "从 0 往 +2 那边挪 0.7 cm"
    assert P.mark_desc(-3.6) == "从 -4 往 -2 那边挪 0.4 cm"
    assert P.mark_desc(2.0) == "正对 +2"
    assert P.mark_desc(-3.0) == "从 -2 往 -4 那边挪 1 cm"
    assert P.mark_desc(4.0) == "正对 +4" and P.mark_desc(0.0) == "正对 0"


def test_split_is_6_2_2_per_bin_plus_6_spares():
    pool = P.generate()["pool"]
    main = sorted((p for p in pool if p["split"] != "spare"), key=lambda p: p["cm"])
    assert len(main) == 50 and sum(p["split"] == "spare" for p in pool) == 6
    for i in range(0, 50, 10):
        assert Counter(p["split"] for p in main[i:i + 10]) == {"train": 6, "val": 2, "test": 2}


def test_order_is_a_shuffle_of_all_56_with_splits_interleaved():
    d = P.generate()
    split = {p["id"]: p["split"] for p in d["pool"]}
    assert sorted(d["order"]) == sorted(split) and len(d["order"]) == 56
    assert {"train", "val", "test"} <= {split[i] for i in d["order"][:15]}


def test_register_requeues_failures():
    """第 1 集（p01）失败 → p01 排到队尾，第 3 集重录到它。"""
    assert P.assign(["p00", "p01", "p02"], [3, 1, 3, 3]) == {"p00": 0, "p02": 2, "p01": 3}


def test_next_positions_resume_from_recorded_count():
    """录制按数据集里已有的集数接着排：中途断掉的一块重跑时，位置不会和集号错开。"""
    d = P.generate()
    cm = {p["id"]: p["cm"] for p in d["pool"]}
    assert P.next_positions(d, "record", n_done=7, stages={}) == [(i, cm[i]) for i in d["order"][7:12]]
    assert P.next_positions(d, "record", n_done=53, stages={}) == [(i, cm[i]) for i in d["order"][53:]]   # 第一轮不跨进重录
    for n_done, stages in ((56, {}), (57, {i: 3 for i in range(56)})):       # 要重录了，但还有没标成败的集
        with pytest.raises(SystemExit):
            P.next_positions(d, "record", n_done=n_done, stages=stages)
    stages = {i: 3 for i in range(56)} | {4: 1}
    assert P.next_positions(d, "record", n_done=56, stages=stages) == [(d["order"][4], cm[d["order"][4]])]
    assert P.next_positions(d, "record", n_done=56, stages={i: 3 for i in range(56)}) == []   # 都成功：录完了


def test_check_blocks_pair_positions_abba():
    """第一步 ABBA：两个数据集的第 i 集都摆在 check[i]；已录满的块跳过。"""
    d = P.generate()
    c = d["check"]
    assert P.next_positions(d, "check", n_done=0, block=1) == [(f"c{i}", c[i]) for i in range(5)]
    assert P.next_positions(d, "check", n_done=3, block=2) == [(f"c{i}", c[i]) for i in range(3, 5)]
    assert P.next_positions(d, "check", n_done=5, block=3) == [(f"c{i}", c[i]) for i in range(5, 10)]
    assert P.next_positions(d, "check", n_done=5, block=2) == []
