import json

import pytest

import q1_data as D
import q1_report as REP
import q1_train as TR

S = REP.SESS
BINS = ("3-6", "7-12", "13-27", "28-52", "53-96", "97-102")


def _rho(vals):
    return dict(zip(S, vals))


def test_gate_ties_pass_per_session_but_mean_must_strictly_improve():
    g2 = _rho([0.6, 0.6, 0.6, 0.6, 0.6])
    assert not REP.gate(_rho([0.6] * 5), g2)["pass"]                     # 全相等：平均未严格改善
    assert REP.gate(_rho([0.6, 0.6, 0.6, 0.6, 0.5]), g2)["pass"]
    g = REP.gate(_rho([0.1, 0.1, 0.1, 0.1, 0.61]), g2)
    assert not g["pass"] and g["per_session"][S[4]] is False            # 平均好很多也不能救一个会话


def test_seed_mean_averages_errors_not_predictions():
    rows = lambda num: [{"sess": s, "num": num, "den": 1.0} for s in S]
    m = REP.seed_mean({"0": rows(0.2), "1": rows(0.4)})
    assert set(m) == set(S) and all(abs(v - 0.3) < 1e-12 for v in m.values())


def test_b_vs_a_sign():
    d = REP.b_vs_a(_rho([0.5] * 5), _rho([0.4] * 5))
    assert all(abs(v + 0.2) < 1e-12 for v in d.values())


def _episodes(fold):
    return [{"name": f"{fold}_ep{i}", "sess": fold, "num": 0.4, "den": 1.0, "f3": 3, "num_aux": 0.5,
             "lead": {b: [0.1, 0.2] for b in BINS}} for i in range(D.EXPECTED_COUNTS[fold])]


def _tree(tmp):
    """完整产物树：6 模型 × 3 种子 × 5 折（含 .pt/.npz）+ 3 个参照 × 5 折。"""
    code = TR.code_sha()
    for m in REP.CANDIDATES:
        for seed in REP.SEEDS:
            for f in S:
                base = tmp / "jobs" / m / f"seed{seed}" / f.replace("/", "_")
                base.parent.mkdir(parents=True, exist_ok=True)
                base.with_suffix(".pt").write_bytes(f"pt{m}{seed}{f}".encode())
                base.with_suffix(".npz").write_bytes(f"npz{m}{seed}{f}".encode())
                base.with_suffix(".json").write_text(json.dumps({
                    "model": m, "seed": int(seed), "fold": f, "config_hash": "cfg", "code_sha": code,
                    "train_sessions": [x for x in S if x != f],
                    "pt_sha256": REP.file_sha(base.with_suffix(".pt")),
                    "npz_sha256": REP.file_sha(base.with_suffix(".npz")), "episodes": _episodes(f)}))
    for r in REP.REFS:
        for f in S:
            p = tmp / "refs" / r / (f.replace("/", "_") + ".json")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"model": r, "seed": "-", "fold": f, "config_hash": code[:16],
                                     "code_sha": code, "episodes": _episodes(f)}))


def test_load_runs_accepts_complete_tree(tmp_path):
    _tree(tmp_path)
    s = REP.summarize(REP.load_runs(tmp_path))
    assert set(s["candidates"]) == set(REP.CANDIDATES) and set(s["b_vs_a"]) == {"1", "2", "3"}


@pytest.mark.parametrize("damage", ["missing_npz", "tampered_pt", "mixed_config", "missing_seed",
                                    "missing_episode", "stale_code", "missing_ref", "leaky_train"])
def test_load_runs_rejects_incomplete_or_inconsistent(tmp_path, damage):
    _tree(tmp_path)
    j1 = tmp_path / "jobs" / "2b" / "seed1" / "ab6_A.json"
    if damage == "missing_npz":
        j1.with_suffix(".npz").unlink()
    elif damage == "tampered_pt":
        j1.with_suffix(".pt").write_bytes(b"changed")
    elif damage == "missing_seed":
        for p in (tmp_path / "jobs" / "3a" / "seed2").iterdir():
            p.unlink()
    elif damage == "missing_ref":
        (tmp_path / "refs" / "ar" / "ab7_Apost.json").unlink()
    else:
        j = json.loads(j1.read_text())
        if damage == "mixed_config":
            j["config_hash"] = "other"
        elif damage == "missing_episode":
            j["episodes"] = j["episodes"][1:]
        elif damage == "stale_code":
            j["code_sha"] = "0" * 64
        elif damage == "leaky_train":
            j["train_sessions"] = list(S)
        j1.write_text(json.dumps(j))
    with pytest.raises(RuntimeError):
        REP.load_runs(tmp_path)


def test_code_sha_covers_the_report():
    """判据（gate / seed_mean / verify）在 q1_report.py ⇒ 它也必须在冻结的代码指纹里。"""
    assert "q1_report.py" in TR.CODE_FILES
