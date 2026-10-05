"""重复精度 25 次的分析与判定测试：合成数据，全部离线。"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pytest

gc.disable()
HERE = Path(__file__).resolve().parent
for _p in (HERE / "q1", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import analyze_repeat_precision25 as Y          # noqa: E402
import run_repeat_precision25 as R              # noqa: E402
import testkit_repeat_precision25 as K          # noqa: E402

RUNS = R.expected_runs()
NON_A0 = [r["run_id"] for r in RUNS if r["run_id"] != "A0"]


# ── 分半基线不共享 ────────────────────────────────────────────────────────────
def test_halves_follow_the_block_table_and_never_share_runs():
    a, b = Y.halves(RUNS, (1, 3)), Y.halves(RUNS, (2, 4))
    assert a["P+"] == ["b1_Pp", "b3_Pp"] and b["P+"] == ["b2_Pp", "b4_Pp"]
    assert a["bare"] == ["b1_bare1", "b1_bare2", "b3_bare1", "b3_bare2"]
    assert b["bare"] == ["b2_bare1", "b2_bare2", "b4_bare1", "b4_bare2"]
    for split in Y.SPLITS:
        ha, hb = Y.halves(RUNS, split[0]), Y.halves(RUNS, split[1])
        ra = {x for v in ha.values() for x in v}
        rb = {x for v in hb.values() for x in v}
        assert ra.isdisjoint(rb) and sorted(ra | rb) == sorted(NON_A0)
        for h, blocks in ((ha, split[0]), (hb, split[1])):
            assert all(len(h[c]) == 2 for c in Y.CONDITIONS) and len(h["bare"]) == 4
            blk = {r["run_id"]: r["block"] for r in RUNS}
            assert all(blk[x] in blocks for v in h.values() for x in v)


def test_perturbing_bare_runs_of_blocks_2_and_4_leaves_the_13_half_unchanged():
    rng = np.random.default_rng(0)
    X = {rid: rng.standard_normal((50, 3)) for rid in NON_A0}
    h13, h24 = Y.halves(RUNS, (1, 3)), Y.halves(RUNS, (2, 4))
    before13, before24 = Y.r_half(X, h13, "Q-"), Y.r_half(X, h24, "Q-")
    for rid in ("b2_bare1", "b2_bare2", "b4_bare1", "b4_bare2"):
        X[rid] = X[rid] + 5.0
    np.testing.assert_array_equal(Y.r_half(X, h13, "Q-"), before13)
    assert not np.allclose(Y.r_half(X, h24, "Q-"), before24)


# ── 数值：手算 ────────────────────────────────────────────────────────────────
def test_split_R_hand_values():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert Y.split_R(a, a) == 1.0
    assert Y.split_R(a, -a) == -1.0
    assert Y.split_R(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])) == 0.0
    assert np.isnan(Y.split_R(np.zeros((2, 2)), np.zeros((2, 2))))


def test_main_metrics_match_hand_computation():
    X = {rid: np.zeros((3, 1)) for rid in NON_A0}
    for r in RUNS:
        if r["condition"] == "P+":
            X[r["run_id"]] = np.full((3, 1), float(r["block"]))         # 区组 b 的 P+ 复放 = b
    mm = Y.main_metrics(X, RUNS)
    assert mm["sigma_c"]["P+"] == pytest.approx(np.sqrt(5.0 / 3.0))     # 1,2,3,4：离差平方和 5，df 3
    assert mm["sigma_c"]["P-"] == 0.0 and mm["sigma_bare"] == 0.0
    # {1,3}|{2,4}：r = 2 对 3 ⇒ 1 − 1/13；{1,4}|{2,3}：2.5 对 2.5 ⇒ 1
    assert mm["R"]["P+"]["splits"] == pytest.approx([1 - 1 / 13, 1.0])
    assert mm["R_c"]["P+"] == pytest.approx((1 - 1 / 13 + 1.0) / 2)
    assert np.isnan(mm["R_c"]["P-"])                                    # 没有响应也没有噪声：R 无定义
    assert mm["sigma_pool"] == pytest.approx(np.sqrt(5.0 / 12.0))       # 合并 df 12


# ── 判定：三种出口与边界 ──────────────────────────────────────────────────────
def R4(a, b, c, d):
    return {"P+": a, "P-": b, "Q+": c, "Q-": d}


@pytest.mark.parametrize("complete,R_c,want", [
    (True, R4(0.6, 0.6, 0.6, 0.6), "support"),
    (True, R4(0.99, 0.95, 0.9, 0.61), "support"),
    (True, R4(0.6, 0.6, 0.6, 0.5999), "indeterminate"),
    (True, R4(0.3, 0.3, 0.3, 0.3), "indeterminate"),                     # 平均恰为 0.3：不是 < 0.3
    (True, R4(0.2999, 0.2999, 0.2999, 0.2999), "stop"),
    (True, R4(0.95, 0.95, 0.0, 0.0), "indeterminate"),                   # 平均 0.475：落在中间
    (True, R4(0.95, 0.95, -0.4, -0.4), "stop"),                          # 平均 0.275 < 0.3
    (True, R4(0.1, -0.2, 0.05, 0.0), "stop"),
    (True, R4(0.9, 0.9, 0.9, float("nan")), "indeterminate"),
    (False, R4(0.99, 0.99, 0.99, 0.99), "indeterminate"),
    (False, None, "indeterminate"),
])
def test_verdict_exits_and_boundaries(complete, R_c, want):
    v = Y.verdict(complete, R_c)
    assert v["verdict"] == want and v["label"] == {"support": "支持另行设计正式采集", "stop": "停止推进",
                                                   "indeterminate": "无法判定"}[want]


# ── 核验：完整会话、篡改、指纹 ────────────────────────────────────────────────
@pytest.fixture
def full(tmp_path, monkeypatch):
    src, out = tmp_path / "pilot", tmp_path / "out"
    pins = K.make_inputs(src, 560)
    for name, value in (("PILOT_SRC", src), ("OUT", out), ("RELEASE", tmp_path / "release.json"),
                        ("FROZEN_INPUTS", pins), ("CODE_FILES", (Path(__file__),))):
        monkeypatch.setattr(R, name, value)
    R.prepare()
    K.write_release(R.RELEASE, R.OUT, R.CODE_FILES)
    return tmp_path


def test_repeatable_responses_give_support(full):
    K.build_session(R)
    res = Y.analyze(R.OUT, R.RELEASE)
    assert res["execution"]["complete"] is True and res["execution"]["reasons"] == []
    assert all(res["main"]["R_c"][c] > 0.9 for c in Y.CONDITIONS)
    assert res["verdict"]["verdict"] == "support"
    assert str(full) not in json.dumps(res)                             # 产物不含绝对路径


def test_no_response_gives_stop(full):
    K.build_session(R, gyro_for=lambda run, i: K.response_gyro(run, i, amp=0.0))
    res = Y.analyze(R.OUT, R.RELEASE)
    assert res["verdict"]["verdict"] == "stop"


def test_tampered_record_hash_is_indeterminate_without_main_metrics(full):
    K.build_session(R)
    with (R.OUT / "records" / "replay_b3_Qm.json").open("a") as f:
        f.write(" ")
    res = Y.analyze(R.OUT, R.RELEASE)
    assert res["verdict"]["verdict"] == "indeterminate" and "main" not in res
    assert any("b3_Qm" in x for x in res["execution"]["reasons"])


def test_invalid_record_is_indeterminate(full):
    K.build_session(R)
    p = R.OUT / "records" / "replay_b2_bare2.json"
    rec = json.loads(p.read_text())
    rec["imu"][100][3] += 1e-6                                          # 20 Hz 样本不在 100 Hz 流里
    p.write_text(json.dumps(rec))
    s = json.loads((R.OUT / "session.json").read_text())
    for a in s["attempts"]:
        if a["run_id"] == "b2_bare2":
            a["record_sha256"] = K.sha(p)                               # 哈希一致，但内容无效
    (R.OUT / "session.json").write_text(json.dumps(s))
    res = Y.analyze(R.OUT, R.RELEASE)
    assert res["verdict"]["verdict"] == "indeterminate" and "main" not in res
    assert any("b2_bare2" in x for x in res["execution"]["reasons"])


@pytest.mark.parametrize("edit", ["drop_last", "halted", "failed"])
def test_incomplete_execution_is_indeterminate(full, edit):
    K.build_session(R)
    s = json.loads((R.OUT / "session.json").read_text())
    if edit == "drop_last":
        s["attempts"] = s["attempts"][:-1]
    elif edit == "halted":
        s["halted"] = True
    else:
        s["attempts"][7]["status"] = "failed"
    (R.OUT / "session.json").write_text(json.dumps(s))
    res = Y.analyze(R.OUT, R.RELEASE)
    assert res["verdict"]["verdict"] == "indeterminate" and "main" not in res


def test_code_fingerprint_mismatch_refuses_to_run(full):
    K.build_session(R)
    K.write_release(R.RELEASE, R.OUT, R.CODE_FILES, code_sha256={"x": "0" * 64})
    with pytest.raises(ValueError, match="指纹"):
        Y.analyze(R.OUT, R.RELEASE)


def test_release_changed_after_collection_breaks_the_binding(full):
    K.build_session(R)
    K.write_release(R.RELEASE, R.OUT, R.CODE_FILES, note="edited after collection")
    res = Y.analyze(R.OUT, R.RELEASE)
    assert res["verdict"]["verdict"] == "indeterminate" and "main" not in res


# ── 本机 OpenBLAS 的 20 线程池在负载下让一次 182×182 求解耗时约 3.5 s：入口与分析须钉住单线程 ─────
@pytest.mark.parametrize("module", ["run_repeat_precision25", "analyze_repeat_precision25"])
def test_entry_modules_pin_blas_to_one_thread_before_numpy_loads(module):
    import os
    import subprocess
    env = {k: v for k, v in os.environ.items() if not k.endswith("_NUM_THREADS")}
    code = (f"import sys; sys.path[:0] = [{str(HERE)!r}, {str(HERE / 'q1')!r}]; import {module}; "
            "from threadpoolctl import threadpool_info; "
            "print(max(i['num_threads'] for i in threadpool_info() if i.get('internal_api') == 'openblas'))")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "1"
