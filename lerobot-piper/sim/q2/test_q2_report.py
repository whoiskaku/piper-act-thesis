import json

import numpy as np
import pytest

import q2_data as D2
import q2_report as REP
import q2_run as RUN
import q2_testing as T
import q2_train as TR

F = D2.FOLDS


def _rb(zero, H, nl, cl):
    return {"zero": dict(zip(F, zero)), "H": dict(zip(F, H)), "nonlinear": dict(zip(F, nl)), "condlinear": dict(zip(F, cl))}


# ── 判据与选择规则（纯函数）─────────────────────────────────────────────────
def test_gate_requires_mean_five_percent_and_every_fold_beating_both_baselines():
    rb = _rb([0.60] * 3, [0.50] * 3, [0.47, 0.47, 0.47], [0.49, 0.49, 0.49])
    g = REP.gate(rb)
    assert g["nonlinear"]["pass"] and abs(g["nonlinear"]["d_mean"] - 0.06) < 1e-12
    assert not g["condlinear"]["pass"] and g["condlinear"]["per_fold_vs_H"] == {"ab5": True, "ab6": True, "ab7": True}
    assert abs(g["condlinear"]["d_mean"] - 0.02) < 1e-12                    # 逐折都优于 H，但平均不到 5%
    rb2 = _rb([0.60] * 3, [0.50] * 3, [0.40, 0.40, 0.51], [0.47] * 3)        # 平均好很多也救不了一折
    assert not REP.gate(rb2)["nonlinear"]["pass"]
    rb3 = _rb([0.45] * 3, [0.50] * 3, [0.46] * 3, [0.40] * 3)                # 优于 H 但不优于零响应
    assert not REP.gate(rb3)["nonlinear"]["pass"] and REP.gate(rb3)["condlinear"]["pass"]


def test_gate_aggregates_relative_reduction_per_fold_then_averages():
    rb = _rb([1.0] * 3, [0.5, 0.4, 0.8], [0.45, 0.36, 0.72], [0.5] * 3)
    assert abs(REP.gate(rb)["nonlinear"]["d_mean"] - 0.10) < 1e-12           # 每折都降 10%（相对），不是百分点


def test_choose_prefers_condlinear_unless_nonlinear_wins_every_fold_by_two_percent():
    both = _rb([0.60] * 3, [0.50] * 3, [0.44, 0.44, 0.44], [0.46, 0.46, 0.46])
    g = REP.gate(both)
    assert REP.choose(g, both) == "nonlinear"                               # (0.46 − 0.44)/0.46 = 4.3% ≥ 2%
    close = _rb([0.60] * 3, [0.50] * 3, [0.455, 0.455, 0.455], [0.46] * 3)
    assert REP.choose(REP.gate(close), close) == "condlinear"               # 只低 1.1%
    split = _rb([0.60] * 3, [0.50] * 3, [0.40, 0.40, 0.465], [0.46] * 3)
    assert REP.choose(REP.gate(split), split) == "condlinear"               # 有一折更差
    only_nl = _rb([0.60] * 3, [0.50] * 3, [0.44] * 3, [0.49] * 3)
    assert REP.choose(REP.gate(only_nl), only_nl) == "nonlinear"
    none = _rb([0.60] * 3, [0.50] * 3, [0.49] * 3, [0.49] * 3)
    assert REP.choose(REP.gate(none), none) is None


def test_seed_mean_then_fold_order():
    rows = lambda num: [{"num": num, "den": 1.0}]
    per_seed = {"ab5": {0: rows(0.2), 1: rows(0.4)}}
    assert abs(REP.seed_mean_rho(per_seed)["ab5"] - 0.3) < 1e-12


def test_beta_is_na_when_the_increment_is_identically_zero():
    assert REP.beta([{"num": 0.0, "den": 0.0}]) is None
    assert abs(REP.beta([{"num": 1.0, "den": 2.0}, {"num": 1.0, "den": 2.0}]) - 0.5) < 1e-12


def test_beta_across_seeds_averages_only_defined_seeds():
    b = REP.beta_seed_mean({0: [{"num": 1.0, "den": 2.0}], 1: [{"num": 0.0, "den": 0.0}], 2: [{"num": 3.0, "den": 4.0}]})
    assert b["per_seed"] == {0: 0.5, 1: None, 2: 0.75} and b["n_defined"] == 2 and abs(b["mean"] - 0.625) < 1e-12
    assert REP.beta_seed_mean({0: [{"num": 0.0, "den": 0.0}]})["mean"] is None


def test_fold_mean_is_equal_weight_over_defined_folds():
    m = REP.fold_mean({"ab5": 0.2, "ab6": None, "ab7": 0.4})
    assert m["n_defined"] == 2 and abs(m["mean"] - 0.3) < 1e-12
    assert REP.fold_mean({"ab5": None, "ab6": None, "ab7": None}) == {"mean": None, "n_defined": 0}


def test_report_entry_disables_gc(tmp_path):
    """入口须关 GC：verify 会导入 MuJoCo 与 scipy，二者共存时 GC 会段错误。"""
    import gc
    gc.enable()
    try:
        assert REP.main(["--out", str(tmp_path)]) == 3
        assert not gc.isenabled()
    finally:
        gc.disable()


def test_code_fingerprint_covers_the_report():
    assert "q2_report.py" in TR.CODE_FILES_Q2


# ── 整条流水线在合成数据上跑一遍，再核验与汇总 ──────────────────────────────
@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    import signal
    out = tmp_path_factory.mktemp("q2out")
    mp = pytest.MonkeyPatch()
    eps = T.synthetic_set(2)

    def fake_load(folds=D2.FOLDS, kinds=("bare", "corr", "aux"), check=True):
        o = [e for e in eps if e.kind in kinds and (e.kind == "aux" or e.fold in folds)]
        if check:
            D2.check_roster(o, keys=sorted({f"{e.fold}/{e.arm}" for e in o}))
        return o

    data = out / "data.json"
    data.write_text("{}")
    mp.setattr(D2, "load", fake_load)
    mp.setattr(D2, "ROSTER", T.roster_of(eps))
    mp.setattr(RUN, "roster_files", lambda: [data])
    mp.setitem(TR.CONFIG, "ckpts", (0, 1, 2))
    mp.setitem(TR.CONFIG, "max_steps", 2)
    mp.setattr(signal, "alarm", lambda sec: None)
    import q1_g2ref as G
    mp.setattr(G, "load_g2", lambda: None)
    mp.setattr(G, "episode_ehat", lambda g2, e, ws, R, b: np.zeros_like(e.gap))
    pre = out / "PREREG.md"
    pre.write_text(f"**状态：已冻结（测试）**\ncode_sha={TR.code_sha()[:16]}，config_hash={TR.config_hash()}\n")
    RUN.cmd_prepare(out, prereg=pre)
    for f in F:
        RUN.cmd_refs(out, f, prereg=pre)
        for s in TR.CONFIG["seeds"]:
            for v in RUN.train_sessions(f):
                RUN.cmd_nom_inner(out, f, v, s, prereg=pre)
            RUN.cmd_nom_final(out, f, s, prereg=pre)
            for form in TR.CONFIG["forms"]:
                for v in RUN.train_sessions(f):
                    RUN.cmd_inc_inner(out, form, f, v, s, prereg=pre)
                RUN.cmd_inc_final(out, form, f, s, prereg=pre)
    yield out, mp
    mp.undo()


def test_complete_tree_verifies_and_aggregates(tree):
    out, _ = tree
    assert REP.verify(out) == []
    s = REP.summarize(out)
    rb = s["rho_bar"]
    assert set(rb) == {"zero", "H", "nonlinear", "condlinear"} and set(rb["zero"]) == set(F)
    j = [json.loads(RUN.nom_final_base(out, "ab6", k).with_suffix(".json").read_text()) for k in (0, 1, 2)]
    want = np.mean([sum(r["num"] for r in x["zero"]) / sum(r["den"] for r in x["zero"]) for x in j])
    assert abs(rb["zero"]["ab6"] - want) < 1e-12                           # 先种子平均再按折
    assert s["choice"] in (None, "nonlinear", "condlinear") and set(s["steps"]) == {"nominal", "nonlinear", "condlinear"}
    assert set(s["aux"]) == {"P2", "P3"} and isinstance(s["aux"]["P2"]["beta"]["H"]["ab5"]["mean"], float)
    assert set(s["q1_recheck"]) == {"nominal", "g2"} and set(s["lead"]) == set(rb)
    md = REP.markdown(s)
    assert "开发检验" in md and "不能把胜负单独解释成非线性是否有用" in md and "step*" in md and "不等于统计显著" in md


def test_ab4_reports_the_three_fold_average_and_per_seed_beta(tree):
    """设计稿第 5 项：ab4 逐折报告并给出三折等权平均；β 保留逐种子值与有效种子数（summary.json 与 summary.md 都有）。"""
    out, _ = tree
    s = REP.summarize(out)
    for a in ("P2", "P3"):
        t = s["aux"][a]
        for p in ("zero", "H", "nonlinear", "condlinear"):
            assert abs(t["rho_fold_mean"][p] - np.mean([t["rho"][p][f] for f in F])) < 1e-12
        b = t["beta_fold_mean"]["H"]
        assert b["n_defined"] == 3 and abs(b["mean"] - np.mean([t["beta"]["H"][f]["mean"] for f in F])) < 1e-12
    md = REP.markdown(s)
    assert "三折等权平均" in md and "3/3：" in md


def test_a_diverged_inner_checkpoint_keeps_the_run_verifiable(tree, tmp_path):
    """内层某检查点的验证 ρ 为 NaN（训练发散）：不被选中、终训照常、核验仍通过，汇总里注明。"""
    import shutil
    src, _ = tree
    out = tmp_path / "copy"
    shutil.copytree(src, out)
    p = RUN.inc_inner_json(out, "condlinear", "ab7", "ab5", 1)
    j = json.loads(p.read_text())
    j["scores"]["1"]["rho"] = float("nan")
    p.write_text(json.dumps(j))
    RUN.cmd_inc_final(out, "condlinear", "ab7", 1, prereg=out / "PREREG.md")
    f = json.loads(RUN.inc_final_base(out, "condlinear", "ab7", 1).with_suffix(".json").read_text())
    assert f["V"]["1"] == float("inf") and f["step_star"] != 1
    assert REP.verify(out) == []
    assert "非有限" in REP.markdown(REP.summarize(out))


@pytest.mark.parametrize("damage", ["missing_final", "tampered_pt", "tampered_npz", "changed_inner", "stale_config",
                                    "wrong_rows", "stale_inputs", "late"])
def test_damaged_trees_are_rejected(tree, damage, tmp_path):
    import shutil
    src, _ = tree
    out = tmp_path / "copy"
    shutil.copytree(src, out)
    jf = RUN.inc_final_base(out, "condlinear", "ab7", 1).with_suffix(".json")
    if damage == "missing_final":
        jf.unlink()
    elif damage == "tampered_pt":
        jf.with_suffix(".pt").write_bytes(b"changed")
    elif damage == "changed_inner":
        p = RUN.nom_inner_json(out, "ab5", "ab6", 2)
        j = json.loads(p.read_text()); j["scores"]["0"]["rho"] = -1.0; p.write_text(json.dumps(j))
    elif damage == "stale_config":
        j = json.loads(jf.read_text()); j["config_hash"] = "0" * 16; jf.write_text(json.dumps(j))
    elif damage == "wrong_rows":
        j = json.loads(jf.read_text()); j["cand"] = j["cand"][1:]; jf.write_text(json.dumps(j))
    elif damage == "tampered_npz":
        jf.with_suffix(".npz").write_bytes(b"changed")
    elif damage == "stale_inputs":                                      # 标定 R/b 文件的记录哈希与现在不符
        p = RUN.inputs_file(out)
        I = json.loads(p.read_text()); k = sorted(I["calib_files"])[0]; I["calib_files"][k] = "0" * 64
        p.write_text(json.dumps(I))
    elif damage == "late":
        d = json.loads(RUN.deadline_file(out).read_text())
        j = json.loads(jf.read_text()); j["finished_at"] = d["deadline"] + 1; jf.write_text(json.dumps(j))
    if damage == "late":
        assert REP.run_status(out)[0] is False and REP.main(["--out", str(out)]) == 3
    else:
        assert REP.verify(out) != [] and REP.main(["--out", str(out)]) == 4
        s = json.loads((out / "summary.json").read_text())
        assert s["status"] == "核验未通过" and "rho_bar" not in s


def test_main_marks_incomplete_runs_and_gives_no_verdict(tree, tmp_path, capsys):
    import shutil
    src, _ = tree
    out = tmp_path / "copy"
    shutil.copytree(src, out)
    RUN.mark_incomplete(out, "测试")
    assert REP.main(["--out", str(out)]) == 3
    s = json.loads((out / "summary.json").read_text())
    assert s["status"] == "未完成" and "rho_bar" not in s
    assert REP.main(["--out", str(src)]) == 0
    assert json.loads((src / "summary.json").read_text())["status"] == "完成"
