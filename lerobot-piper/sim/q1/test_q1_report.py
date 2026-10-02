import json

import pytest

import q1_data as D
import q1_report as REP
import q1_run as RUN
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


def test_code_sha_covers_the_report():
    """判据（gate / seed_mean / verify）在 q1_report.py ⇒ 它也必须在冻结的代码指纹里。"""
    assert "q1_report.py" in TR.CODE_FILES


# ── 完整产物树（用 q1_run 的路径与指纹函数构造，保证与真实产物同构）──────────
def _episodes(fold):
    return [{"name": f"{fold}_ep{i}", "sess": fold, "num": 0.4, "den": 1.0, "f3": 3, "num_aux": 0.5,
             "lead": {b: [0.1, 0.2] for b in BINS}} for i in range(D.EXPECTED_COUNTS[fold])]


def _curve(model, fold, val, seed):
    """各检查点的验证 ρ：3b 单调下降（最佳点落在上限），其余在中间取最小。"""
    ck = TR.CONFIG["ckpts"]
    if model == "3b":
        return {s: 0.9 - 0.01 * i for i, s in enumerate(ck)}
    k = (S.index(fold) + seed + len(val)) % (len(ck) - 2) + 1
    return {s: 0.5 + 0.01 * abs(i - k) for i, s in enumerate(ck)}


def _tree(tmp):
    import torch
    code, cfg = TR.code_sha(), TR.config_hash()
    data = tmp / "d.json"
    data.write_text("{}")
    dfiles, dfp = RUN.files_fp([data])
    I = {"code_sha": code, "data": dfp, "deps": RUN.fp_of({"files": {}, "env": {}}), "g2": RUN.files_fp([])[1],
         "data_files": dfiles, "deps_files": {}, "env": {}, "g2_files": {}, "sensor": {}}
    for ss in RUN.sensor_sets():
        f = RUN.sensor_set_file(tmp, ss)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(RUN.set_tag(ss).encode())
        I["sensor"][RUN.set_tag(ss)] = RUN.file_sha(f)
    RUN.atomic_json(RUN.inputs_file(tmp), I)
    for m in REP.CANDIDATES:
        for seed in (0, 1, 2):
            for f in S:
                for v in RUN.train_sessions(f):
                    inputs = RUN.inner_inputs(I, m, f, v)
                    sc = {str(s): {"num": r, "den": 1.0, "rho": r, "state_sha256": "x"}
                          for s, r in _curve(m, f, v, seed).items()}
                    RUN.atomic_json(RUN.inner_json(tmp, m, f, v, seed),
                                    {"model": m, "fold": f, "val": v, "seed": seed, "config_hash": cfg,
                                     "code_sha": code, "inputs": inputs, "input_fp": RUN.fp_of(inputs),
                                     "train_sessions": list(RUN.inner_train(f, v)), "n_val": 1, "scores": sc})
                V, step = RUN.selection(tmp, m, f, seed)
                base = RUN.final_base(tmp, m, f, seed)
                base.parent.mkdir(parents=True, exist_ok=True)
                sd = {"w": torch.full((2,), float(step))}
                torch.save({"state_dict": sd, "step": step}, base.with_suffix(".pt"))
                base.with_suffix(".npz").write_bytes(f"npz{m}{seed}{f}".encode())
                inputs = RUN.final_inputs(tmp, I, m, f, seed)
                RUN.atomic_json(base.with_suffix(".json"), {
                    "model": m, "fold": f, "seed": seed, "config_hash": cfg, "code_sha": code,
                    "inputs": inputs, "input_fp": RUN.fp_of(inputs),
                    "train_sessions": list(RUN.train_sessions(f)), "V": {str(s): V[s] for s in sorted(V)},
                    "step_star": step, "boundary": step == TR.CONFIG["max_steps"],
                    "state_sha256": TR.state_dict_sha(sd), "pt_sha256": RUN.file_sha(base.with_suffix(".pt")),
                    "npz_sha256": RUN.file_sha(base.with_suffix(".npz")), "episodes": _episodes(f)})
    for r in REP.REFS:
        for f in S:
            inputs = RUN.refs_inputs(I)
            RUN.atomic_json(tmp / "refs" / r / f"{RUN.tag(f)}.json", {
                "model": r, "seed": "-", "fold": f, "config_hash": code[:16], "code_sha": code,
                "inputs": inputs, "input_fp": RUN.fp_of(inputs),
                "train_sessions": [x for x in S if x != f] if r == "ar" else [], "episodes": _episodes(f)})
    return data


def test_load_runs_accepts_complete_tree_and_reports_selection(tmp_path):
    _tree(tmp_path)
    s = REP.summarize(REP.load_runs(tmp_path), REP.load_selection(tmp_path))
    assert set(s["candidates"]) == set(REP.CANDIDATES) and set(s["b_vs_a"]) == {"1", "2", "3"}
    sel = s["selection"]
    assert all(sel["3b"][f][str(k)]["boundary"] for f in S for k in (0, 1, 2))
    assert not any(sel["2a"][f][str(k)]["boundary"] for f in S for k in (0, 1, 2))
    md = REP.markdown(s)
    assert "step*" in md and "最佳点落在搜索边界" in md and "伺服先验" in md


def _edit(path, **kv):
    j = json.loads(path.read_text())
    j.update(kv)
    path.write_text(json.dumps(j))


@pytest.mark.parametrize("damage", ["missing_inner", "tampered_inner", "wrong_step", "stale_config",
                                    "tampered_pt", "pt_step", "missing_npz", "leaky_train", "missing_seed",
                                    "missing_episode", "missing_ref", "data_changed", "sensor_changed"])
def test_load_runs_rejects_incomplete_or_inconsistent(tmp_path, damage):
    import torch
    data = _tree(tmp_path)
    j1 = RUN.final_base(tmp_path, "2b", "ab6/A", 1).with_suffix(".json")
    inner = RUN.inner_json(tmp_path, "2b", "ab6/A", "ab2/A", 1)
    if damage == "missing_inner":
        inner.unlink()
    elif damage == "tampered_inner":
        j = json.loads(inner.read_text())
        j["scores"]["0"]["rho"] = 0.0
        inner.write_text(json.dumps(j))
    elif damage == "wrong_step":
        _edit(j1, step_star=TR.CONFIG["ckpts"][1] if json.loads(j1.read_text())["step_star"] == 0 else 0)
    elif damage == "stale_config":
        _edit(j1, config_hash="0" * 16)
    elif damage == "tampered_pt":
        j1.with_suffix(".pt").write_bytes(b"changed")
    elif damage == "pt_step":
        pt = j1.with_suffix(".pt")
        ck = torch.load(pt, weights_only=False)
        ck["step"] = ck["step"] + 1
        torch.save(ck, pt)
        _edit(j1, pt_sha256=RUN.file_sha(pt))
    elif damage == "missing_npz":
        j1.with_suffix(".npz").unlink()
    elif damage == "leaky_train":
        _edit(j1, train_sessions=list(S))
    elif damage == "missing_seed":
        for p in (tmp_path / "jobs" / "3a" / "seed2").iterdir():
            p.unlink()
    elif damage == "missing_episode":
        j = json.loads(j1.read_text())
        _edit(j1, episodes=j["episodes"][1:])
    elif damage == "missing_ref":
        (tmp_path / "refs" / "ar" / "ab7_Apost.json").unlink()
    elif damage == "data_changed":
        data.write_text('{"x": 1}')
    elif damage == "sensor_changed":
        next((tmp_path / "sensor").iterdir()).write_bytes(b"changed")
    with pytest.raises(RuntimeError):
        REP.load_runs(tmp_path)
