import json

import numpy as np
import pytest

import q1_data as D1
import q2_data as D2
import q2_run as RUN
import q2_testing as T
import q2_train as TR

TINY = (0, 1, 2)
ALARMS = []


@pytest.fixture()
def syn(monkeypatch, tmp_path):
    """合成数据：3 个近期会话 × {裸跑, 补正} × 2 集 + ab4 P2/P3 各 2 集；检查点缩成 (0,1,2)；闹钟只记录。"""
    import signal
    eps = T.synthetic_set(2)

    def fake_load(folds=D2.FOLDS, kinds=("bare", "corr", "aux"), check=True):
        out = [e for e in eps if e.kind in kinds and (e.kind == "aux" or e.fold in folds)]
        if check:
            D2.check_roster(out, keys=sorted({f"{e.fold}/{e.arm}" for e in out}))
        return out

    data = tmp_path / "data.json"
    data.write_text("{}")
    monkeypatch.setattr(D2, "load", fake_load)
    monkeypatch.setattr(D2, "ROSTER", T.roster_of(eps))
    monkeypatch.setattr(RUN, "roster_files", lambda: [data])
    monkeypatch.setitem(TR.CONFIG, "ckpts", TINY)
    monkeypatch.setitem(TR.CONFIG, "max_steps", TINY[-1])
    ALARMS.clear()
    monkeypatch.setattr(signal, "alarm", lambda sec: ALARMS.append(sec))
    return eps, data


def _prereg(tmp_path, status="**状态：已冻结（测试）**", sha=None, cfg=None):
    p = tmp_path / "PREREG.md"
    p.write_text(f"# 预注册\n\n{status}\ncode_sha={sha or TR.code_sha()[:16]}，config_hash={cfg or TR.config_hash()}\n")
    return p


def _run_all(out, pre, forms=("nonlinear", "condlinear"), folds=("ab6",), seeds=(0,)):
    for f in folds:
        for s in seeds:
            for v in RUN.train_sessions(f):
                RUN.cmd_nom_inner(out, f, v, s, prereg=pre)
            RUN.cmd_nom_final(out, f, s, prereg=pre)
            for form in forms:
                for v in RUN.train_sessions(f):
                    RUN.cmd_inc_inner(out, form, f, v, s, prereg=pre)
                RUN.cmd_inc_final(out, form, f, s, prereg=pre)


# ── 闸门 ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("case", ["missing", "draft", "stale_sha", "stale_config"])
def test_formal_commands_refuse_unless_frozen(tmp_path, monkeypatch, case):
    def boom(*a, **k):
        raise AssertionError("闸门必须在读数据之前")
    monkeypatch.setattr(D2, "load", boom)
    pre = {"missing": tmp_path / "none.md",
           "draft": _prereg(tmp_path, status="**状态：草稿（测试）**"),
           "stale_sha": _prereg(tmp_path, sha="0" * 16),
           "stale_config": _prereg(tmp_path, cfg="0" * 16)}[case]
    for call in (lambda: RUN.cmd_prepare(tmp_path, prereg=pre),
                 lambda: RUN.cmd_refs(tmp_path, "ab5", prereg=pre),
                 lambda: RUN.cmd_nom_inner(tmp_path, "ab5", "ab6", 0, prereg=pre),
                 lambda: RUN.cmd_nom_final(tmp_path, "ab5", 0, prereg=pre),
                 lambda: RUN.cmd_inc_inner(tmp_path, "condlinear", "ab5", "ab6", 0, prereg=pre),
                 lambda: RUN.cmd_inc_final(tmp_path, "condlinear", "ab5", 0, prereg=pre)):
        with pytest.raises(RUN.NotFrozenError):
            call()
    assert RUN.main(["prepare", "--out", str(tmp_path), "--prereg", str(pre)]) == 2


def test_formal_commands_take_no_training_overrides(tmp_path):
    for argv in (["nom-inner", "--out", str(tmp_path), "--steps", "5", "ab5", "ab6", "0"],
                 ["inc-final", "--out", str(tmp_path), "--steps", "5", "condlinear", "ab5", "0"]):
        with pytest.raises(SystemExit):
            RUN.main(argv)


def test_code_fingerprint_covers_the_run_entry():
    assert "q2_run.py" in TR.CODE_FILES_Q2


def test_inner_split_mapping():
    assert RUN.train_sessions("ab5") == ("ab6", "ab7")
    assert RUN.inner_train("ab5", "ab6") == "ab7" and RUN.inner_train("ab5", "ab7") == "ab6"
    with pytest.raises(ValueError):
        RUN.inner_train("ab5", "ab5")


# ── prepare：名单、截止时间、输入指纹 ─────────────────────────────────────
def test_prepare_records_inputs_including_calibration_and_h(tmp_path, syn):
    RUN.cmd_prepare(tmp_path, prereg=_prereg(tmp_path))
    I = RUN.read_inputs(tmp_path)
    assert set(I["calib_files"]) == set(RUN.CALIB_FILES) and set(I["h_files"]) == set(RUN.H_FILES)
    assert {"data", "calib", "h", "deps", "g2"} <= set(I)
    d = json.loads(RUN.deadline_file(tmp_path).read_text())
    assert d["deadline"] - d["started_at"] == 24 * 3600 and 0 < ALARMS[-1] <= 24 * 3600 + 1


def test_a_roster_mismatch_stops_prepare_before_the_clock_starts(tmp_path, syn, monkeypatch):
    eps, _ = syn
    bad = {k: v for k, v in T.roster_of(eps).items()}
    bad["ab6/D"] = bad["ab6/D"][1:]                                    # 名单少一集
    monkeypatch.setattr(D2, "ROSTER", bad)
    with pytest.raises(D1.DataContractError):
        RUN.cmd_prepare(tmp_path, prereg=_prereg(tmp_path))
    assert not RUN.deadline_file(tmp_path).exists()


def test_jobs_refuse_when_inputs_changed_after_prepare(tmp_path, syn):
    _, data = syn
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    data.write_text('{"changed": true}')
    with pytest.raises(RuntimeError):
        RUN.cmd_nom_inner(tmp_path, "ab6", "ab5", 0, prereg=pre)


def test_passed_deadline_stops_every_formal_command(tmp_path, syn):
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    d = json.loads(RUN.deadline_file(tmp_path).read_text())
    d["deadline"] = d["started_at"] - 1
    RUN.deadline_file(tmp_path).write_text(json.dumps(d))
    for call in (lambda: RUN.cmd_nom_inner(tmp_path, "ab6", "ab5", 0, prereg=pre),
                 lambda: RUN.cmd_inc_final(tmp_path, "condlinear", "ab6", 0, prereg=pre),
                 lambda: RUN.cmd_refs(tmp_path, "ab6", prereg=pre)):
        with pytest.raises(RUN.DeadlineExceeded):
            call()
    assert json.loads(RUN.incomplete_file(tmp_path).read_text())["status"] == "未完成"
    assert RUN.main(["nom-inner", "--out", str(tmp_path), "--prereg", str(pre), "ab6", "ab5", "0"]) == 3


# ── 作业链 ────────────────────────────────────────────────────────────────
def test_nominal_inner_and_final(tmp_path, syn):
    import torch
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    with pytest.raises(RuntimeError):
        RUN.cmd_nom_final(tmp_path, "ab6", 0, prereg=pre)            # 内层表不全
    for v in RUN.train_sessions("ab6"):
        RUN.cmd_nom_inner(tmp_path, "ab6", v, 0, prereg=pre)
    j = json.loads(RUN.nom_inner_json(tmp_path, "ab6", "ab5", 0).read_text())
    assert j["train_sessions"] == ["ab7"] and sorted(int(s) for s in j["scores"]) == list(TINY)
    RUN.cmd_nom_final(tmp_path, "ab6", 0, prereg=pre)
    base = RUN.nom_final_base(tmp_path, "ab6", 0)
    f = json.loads(base.with_suffix(".json").read_text())
    V, step = RUN.selection([RUN.nom_inner_json(tmp_path, "ab6", v, 0) for v in ("ab5", "ab7")], TINY)
    assert f["step_star"] == step and f["V"] == {str(k): V[k] for k in sorted(V)}
    assert torch.load(base.with_suffix(".pt"), weights_only=False)["step"] == step
    assert {r["fold"] for r in f["q1"]} == {"ab6"} and {r["arm"] for r in f["q1"]} == {"A"}
    assert {r["arm"] for r in f["zero"]} == {"D"} and len(f["H"]) == len(f["zero"])
    assert {r["arm"] for r in f["aux_zero"]} == {"P2", "P3"} and all(b["den"] > 0 for b in f["aux_H_beta"])
    assert f["pt_sha256"] == RUN.file_sha(base.with_suffix(".pt"))


def test_increment_chain_and_zero_step_keeps_zero_response(tmp_path, syn):
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    with pytest.raises(RuntimeError):
        RUN.cmd_inc_inner(tmp_path, "condlinear", "ab6", "ab5", 0, prereg=pre)   # 名义还没终训
    _run_all(tmp_path, pre)
    for form in ("nonlinear", "condlinear"):
        f = json.loads(RUN.inc_final_base(tmp_path, form, "ab6", 0).with_suffix(".json").read_text())
        assert {r["arm"] for r in f["cand"]} == {"D"} and f["train_sessions"] == ["ab5", "ab7"]
    # 把内层表改成第 0 步最好 ⇒ 终训保留零响应：候选与零响应逐集相同
    for v in ("ab5", "ab7"):
        p = RUN.inc_inner_json(tmp_path, "condlinear", "ab6", v, 0)
        j = json.loads(p.read_text())
        j["scores"]["0"]["rho"] = -1.0
        p.write_text(json.dumps(j))
    RUN.cmd_inc_final(tmp_path, "condlinear", "ab6", 0, prereg=pre)
    f = json.loads(RUN.inc_final_base(tmp_path, "condlinear", "ab6", 0).with_suffix(".json").read_text())
    z = json.loads(RUN.nom_final_base(tmp_path, "ab6", 0).with_suffix(".json").read_text())["zero"]
    assert f["step_star"] == 0 and [r["num"] for r in f["cand"]] == [r["num"] for r in z]
    assert all(b["den"] == 0.0 for b in f["aux_beta"])                 # 增量恒为 0 ⇒ β 不定义


def test_jobs_train_and_select_only_on_the_allowed_runs(tmp_path, syn, monkeypatch):
    """真正传给训练与内层评分的集（不只是记录的字段）：名义只用训练会话的裸跑、增量只用训练会话的补正运行；
    内层只在内层训练会话上训练、在内层验证会话上评分；留出会话与 ab4 从不进入训练或选步。"""
    trained, scored = [], []
    real = {"nom": TR.train_nominal, "inc": TR.train_increment, "sn": RUN.score_nominal, "si": RUN.score_increment}

    def nom(eps, seed, ckpts=None, log=None):
        trained.append(("nom", {(e.fold, e.kind) for e in eps}))
        return real["nom"](eps, seed, ckpts=ckpts, log=log)

    def inc(form, n, eps, seed, ckpts=None, log=None):
        trained.append((form, {(e.fold, e.kind) for e in eps}))
        return real["inc"](form, n, eps, seed, ckpts=ckpts, log=log)

    def sn(model, st, eps, K):
        scored.append(("nom", {(e.fold, e.kind) for e in eps}))
        return real["sn"](model, st, eps, K)

    def si(form, model, r, nom_st, eps, e_nom, K):
        scored.append((form, {(e.fold, e.kind) for e in eps}))
        return real["si"](form, model, r, nom_st, eps, e_nom, K)

    monkeypatch.setattr(TR, "train_nominal", nom)
    monkeypatch.setattr(TR, "train_increment", inc)
    monkeypatch.setattr(RUN, "score_nominal", sn)
    monkeypatch.setattr(RUN, "score_increment", si)
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    _run_all(tmp_path, pre, forms=("condlinear",))                     # 外层折 ab6、种子 0
    assert trained == [("nom", {("ab7", "bare")}), ("nom", {("ab5", "bare")}),
                       ("nom", {("ab5", "bare"), ("ab7", "bare")}),
                       ("condlinear", {("ab7", "corr")}), ("condlinear", {("ab5", "corr")}),
                       ("condlinear", {("ab5", "corr"), ("ab7", "corr")})]
    n_ck = len(TINY)
    assert scored == ([("nom", {("ab5", "bare")})] * n_ck + [("nom", {("ab7", "bare")})] * n_ck   # 内层：验证会话
                      + [("nom", {("ab6", "bare")})]                                         # 终训：留出裸跑 Q1 复查
                      + [("condlinear", {("ab5", "corr")})] * n_ck + [("condlinear", {("ab7", "corr")})] * n_ck)


def test_selection_treats_a_non_finite_validation_rho_as_never_best(tmp_path):
    """训练发散的检查点（验证 ρ 非有限）按 +∞ 处理：永不被选中，记录值可经 JSON 往返，核验时比较一致。"""
    tables = []
    for v, rhos in (("a", {0: 0.9, 1: float("nan"), 2: 0.5}), ("b", {0: 0.8, 1: 0.1, 2: 0.6})):
        p = tmp_path / f"{v}.json"
        p.write_text(json.dumps({"scores": {str(k): {"rho": r} for k, r in rhos.items()}}))
        tables.append(p)
    V, step = RUN.selection(tables, (0, 1, 2))
    assert V[1] == float("inf") and step == 2
    rec = {str(k): V[k] for k in V}
    assert json.loads(json.dumps(rec)) == rec                          # 终训记录的 V 与核验时重算的 V 可逐值比较


def test_list_jobs_counts_and_requeue_on_input_change(tmp_path, syn, capsys):
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    n = {ph: len(RUN.jobs(tmp_path, ph)) for ph in RUN.PHASES}
    assert n == {"refs": 3, "nom-inner": 18, "nom-final": 9, "inc-inner": 36, "inc-final": 18}
    _run_all(tmp_path, pre)
    assert ["ab6", "ab5", "0"] not in RUN.jobs(tmp_path, "nom-inner")
    assert ["condlinear", "ab6", "0"] not in RUN.jobs(tmp_path, "inc-final")
    p = RUN.nom_final_base(tmp_path, "ab6", 0).with_suffix(".pt")       # 名义产物变了 ⇒ 下游增量重排
    p.write_bytes(p.read_bytes() + b"x")
    assert ["condlinear", "ab6", "ab5", "0"] in RUN.jobs(tmp_path, "inc-inner")


def test_final_jobs_requeue_when_weights_or_predictions_are_missing_or_corrupt(tmp_path, syn):
    """终训的 JSON 指纹一致还不够：.pt、.npz 缺失或哈希与记录不符 ⇒ 重新排队。"""
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    _run_all(tmp_path, pre, forms=("condlinear",))
    assert ["ab6", "0"] not in RUN.jobs(tmp_path, "nom-final")
    assert ["condlinear", "ab6", "0"] not in RUN.jobs(tmp_path, "inc-final")
    nom, inc = RUN.nom_final_base(tmp_path, "ab6", 0), RUN.inc_final_base(tmp_path, "condlinear", "ab6", 0)
    keep = {p: p.read_bytes() for p in (nom.with_suffix(".npz"), inc.with_suffix(".pt"), inc.with_suffix(".npz"))}
    nom.with_suffix(".npz").write_bytes(b"corrupt")                    # 名义的预测文件损坏
    assert ["ab6", "0"] in RUN.jobs(tmp_path, "nom-final")
    nom.with_suffix(".npz").write_bytes(keep[nom.with_suffix(".npz")])
    assert ["ab6", "0"] not in RUN.jobs(tmp_path, "nom-final")
    inc.with_suffix(".pt").unlink()                                     # 增量权重缺失
    assert ["condlinear", "ab6", "0"] in RUN.jobs(tmp_path, "inc-final")
    inc.with_suffix(".pt").write_bytes(keep[inc.with_suffix(".pt")])
    inc.with_suffix(".npz").write_bytes(keep[inc.with_suffix(".npz")] + b"x")    # 增量预测文件被改
    assert ["condlinear", "ab6", "0"] in RUN.jobs(tmp_path, "inc-final")


def test_changed_code_cannot_continue_an_old_run(tmp_path, syn, monkeypatch):
    """冻结代码改动后须重新冻结，且不能在旧目录里续跑：截止时间记录绑定旧指纹 ⇒ prepare 与作业都拒绝。"""
    RUN.cmd_prepare(tmp_path, prereg=_prereg(tmp_path))
    monkeypatch.setattr(TR, "code_sha", lambda: "f" * 64)              # 模拟改了冻结代码
    pre2 = _prereg(tmp_path)                                            # 已按新指纹重新冻结
    for call in (lambda: RUN.cmd_prepare(tmp_path, prereg=pre2),
                 lambda: RUN.cmd_nom_inner(tmp_path, "ab6", "ab5", 0, prereg=pre2)):
        with pytest.raises(RuntimeError, match="另一版冻结的实验"):
            call()


@pytest.mark.parametrize("steps", [0, -1, 31, 10000])
def test_timing_refuses_out_of_range_steps_before_loading_or_training(steps, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("越界必须在加载数据、训练之前拒绝")
    monkeypatch.setattr(D2, "load", boom)
    monkeypatch.setattr(TR, "train_nominal", boom)
    with pytest.raises(RUN.TimingRefused):
        RUN.cmd_timing(steps)
    assert RUN.main(["timing", "--steps", str(steps)]) == 2


def test_timing_needs_no_prereg(syn, monkeypatch, capsys):
    monkeypatch.setattr(RUN, "PREREG", D1.ROOT / "reports" / "does_not_exist.md")
    RUN.cmd_timing(1)
    out = capsys.readouterr().out
    assert "s/步" in out and "核·小时" in out and "condlinear" in out


@pytest.mark.slow
def test_refs_on_synthetic(tmp_path, syn):
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    RUN.cmd_refs(tmp_path, "ab7", prereg=pre)
    j = json.loads(RUN.ref_json(tmp_path, "ab7").read_text())
    assert {r["fold"] for r in j["episodes"]} == {"ab7"} and j["config_hash"] == TR.config_hash()
    mtime = RUN.ref_json(tmp_path, "ab7").stat().st_mtime_ns
    RUN.cmd_refs(tmp_path, "ab7", prereg=pre)                          # 已是最新 ⇒ 跳过
    assert RUN.ref_json(tmp_path, "ab7").stat().st_mtime_ns == mtime
