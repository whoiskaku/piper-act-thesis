import json

import numpy as np
import pytest

import q1_data as D
import q1_run as RUN
import q1_testing as T
import q1_train as TR

TINY = (0, 1, 2)


def _fake_fit(tr, R):
    """快速替身（形状正确、随训练集变化）；真实拟合由 test_q1_sensor 与下面的 slow 用例覆盖。"""
    import q1_sensor as S
    return (np.full((S.NLAG * S.N_FEAT + 1, 3), 1e-4 * len(tr)),
            np.full((100, 3 * D.HIST, 3), 1e-3 * len(tr)))


ALARMS = []


@pytest.fixture()
def syn_data(monkeypatch):
    """5 个会话 × 2 集合成数据；检查点缩成 (0, 1, 2) 让端到端用例跑得快；名单换成合成集名；
    闹钟只记录、不真设（免得到点打断后面的用例）。"""
    import signal
    eps = [T.synthetic_episode(i, sess=f"{s}/{a}") for i, (s, a) in
           enumerate([x for x in D.SESSIONS for _ in range(2)])]
    roster = {}
    for e in eps:
        roster.setdefault(e.sess, []).append(e.name)
    monkeypatch.setattr(D, "ROSTER", roster)
    ALARMS.clear()
    monkeypatch.setattr(signal, "alarm", lambda sec: ALARMS.append(sec))
    monkeypatch.setattr(D, "load_bare", lambda *a, **k: eps)
    monkeypatch.setattr(D, "calibration", lambda: (np.eye(3), np.zeros(3)))
    monkeypatch.setattr(RUN, "data_files", lambda e: [D.CALIB_JSON])
    monkeypatch.setitem(TR.CONFIG, "ckpts", TINY)
    monkeypatch.setitem(TR.CONFIG, "max_steps", TINY[-1])
    monkeypatch.setattr(RUN, "fit_sensor", _fake_fit)
    return eps


def _prereg(tmp_path, status="**状态：已冻结（测试）**", sha=None, cfg=None):
    p = tmp_path / "PREREG.md"
    p.write_text(f"# 预注册\n\n{status}\n代码 code_sha={sha or TR.code_sha()[:16]}，"
                 f"config_hash={cfg or TR.config_hash()}\n")
    return p


# ── 冻结闸门与命令行 ───────────────────────────────────────────────────────
@pytest.mark.parametrize("case", ["missing", "draft", "stale_sha", "stale_config"])
def test_formal_commands_refuse_unless_frozen(tmp_path, monkeypatch, case):
    def boom(*a, **k):
        raise AssertionError("闸门必须在读数据之前")
    monkeypatch.setattr(D, "load_bare", boom)
    pre = {"missing": tmp_path / "none.md",
           "draft": _prereg(tmp_path, status="**状态：草稿（测试），待用户冻结。**"),
           "stale_sha": _prereg(tmp_path, sha="0" * 16),
           "stale_config": _prereg(tmp_path, cfg="0" * 16)}[case]
    calls = [lambda: RUN.cmd_prepare(tmp_path, prereg=pre),
             lambda: RUN.cmd_refs(tmp_path, prereg=pre),
             lambda: RUN.cmd_inner(tmp_path, "2a", "ab2/A", "ab4/P0", 0, prereg=pre),
             lambda: RUN.cmd_final(tmp_path, "2a", "ab2/A", 0, prereg=pre)]
    for call in calls:
        with pytest.raises(RUN.NotFrozenError):
            call()
    assert RUN.main(["prepare", "--out", str(tmp_path), "--prereg", str(pre)]) == 2


def test_formal_commands_take_no_training_overrides(tmp_path):
    """训练配置只来自冻结的 CONFIG：正式入口不接受 --steps 之类的覆盖。"""
    for argv in (["inner", "--out", str(tmp_path), "--steps", "5", "2a", "ab2/A", "ab4/P0", "0"],
                 ["final", "--out", str(tmp_path), "--steps", "5", "2a", "ab2/A", "0"],
                 ["prepare", "--out", str(tmp_path), "--steps", "5"]):
        with pytest.raises(SystemExit):
            RUN.main(argv)


def test_timing_needs_no_prereg(syn_data, monkeypatch, capsys):
    monkeypatch.setattr(RUN, "PREREG", D.ROOT / "reports" / "does_not_exist.md")
    RUN.cmd_timing(1)
    out = capsys.readouterr().out
    assert "s/步" in out and "s/集" in out and "核·小时" in out


def test_code_sha_covers_the_run_entry():
    """`q1_run.py` 决定写什么进产物（逐集计分、哈希）⇒ 它也必须在代码指纹里。"""
    assert "q1_run.py" in TR.CODE_FILES


# ── 会话集合与传感器层 ────────────────────────────────────────────────────
def test_session_sets_cover_outer_inner_and_all():
    sets = RUN.sensor_sets()
    # 内层三会话集合由无序对 {外层折, 内层验证会话} 决定 ⇒ C(5,2) = 10 个；外层四会话 5 个；全部 1 个
    assert len(sets) == 16 and len(set(sets)) == 16
    assert sorted(len(s) for s in sets) == [3] * 10 + [4] * 5 + [5] and tuple(sorted(RUN.SESS)) in sets
    assert RUN.train_sessions("ab5/A") == tuple(s for s in RUN.SESS if s != "ab5/A")
    assert RUN.train_sessions(RUN.ALL) == RUN.SESS
    assert RUN.seeds_of(RUN.ALL) == (0,) and RUN.seeds_of("ab5/A") == RUN.SEEDS


def test_prepare_fits_every_session_set_once_and_skips_current_files(tmp_path, syn_data):
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    I = RUN.read_inputs(tmp_path)
    assert len(I["sensor"]) == 16 and {"data", "deps", "g2"} <= set(I)
    inner = ("ab2/A", "ab4/P0", "ab7/Apost")
    Wrs, C, man = RUN.load_sensor_set(tmp_path, inner)
    assert man["train_sessions"] == sorted(inner) and man["n_train"] == 6
    stamp = {p.name: p.stat().st_mtime_ns for p in (tmp_path / "sensor").iterdir()}
    RUN.cmd_prepare(tmp_path, prereg=pre)                                 # 续跑：全是最新 ⇒ 不重拟合
    assert {p.name: p.stat().st_mtime_ns for p in (tmp_path / "sensor").iterdir()} == stamp


def test_load_sensor_set_rejects_a_file_fitted_on_other_sessions(tmp_path, syn_data):
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    src = RUN.sensor_set_file(tmp_path, ("ab2/A", "ab4/P0", "ab6/A", "ab7/Apost"))
    dst = RUN.sensor_set_file(tmp_path, ("ab2/A", "ab4/P0", "ab7/Apost"))
    dst.write_bytes(src.read_bytes())                                     # 四会话文件冒充三会话
    with pytest.raises(RuntimeError):
        RUN.load_sensor_set(tmp_path, ("ab2/A", "ab4/P0", "ab7/Apost"))


# ── 内层 / 终训 / 续跑 ────────────────────────────────────────────────────
@pytest.mark.parametrize("model", ["1b", "2a"])
def test_inner_scores_every_checkpoint_on_the_held_out_inner_session(tmp_path, syn_data, model):
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    RUN.cmd_inner(tmp_path, model, "ab5/A", "ab6/A", 0, prereg=pre)
    j = json.loads(RUN.inner_json(tmp_path, model, "ab5/A", "ab6/A", 0).read_text())
    assert j["train_sessions"] == ["ab7/Apost", "ab2/A", "ab4/P0"] and j["val"] == "ab6/A"
    assert sorted(int(s) for s in j["scores"]) == list(TINY)
    assert all(r["den"] > 0 and np.isclose(r["rho"], r["num"] / r["den"]) for r in j["scores"].values())
    I = RUN.read_inputs(tmp_path)
    want = I["sensor"][RUN.set_tag(("ab7/Apost", "ab2/A", "ab4/P0"))] if model == "1b" else None
    assert j["inputs"]["sensor"] == want                                   # 用的是三会话的传感器层
    assert j["config_hash"] == TR.config_hash() and j["input_fp"] == RUN.fp_of(j["inputs"])
    if model == "2a":
        assert j["scores"]["0"]["rho"] == 1.0                               # 端到端第 0 步 = 零预测


def _all_inner(tmp_path, pre, model, fold, seed=0):
    for v in RUN.train_sessions(fold):
        RUN.cmd_inner(tmp_path, model, fold, v, seed, prereg=pre)


@pytest.mark.parametrize("model", ["1a", "3a"])
def test_final_trains_to_the_selected_step_and_records_the_mapping(tmp_path, syn_data, model):
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    with pytest.raises(RuntimeError):
        RUN.cmd_final(tmp_path, model, "ab2/A", 0, prereg=pre)            # 内层评分表不全
    _all_inner(tmp_path, pre, model, "ab2/A")
    RUN.cmd_final(tmp_path, model, "ab2/A", 0, prereg=pre)
    base = RUN.final_base(tmp_path, model, "ab2/A", 0)
    j = json.loads(base.with_suffix(".json").read_text())
    V, step = RUN.selection(tmp_path, model, "ab2/A", 0)
    assert j["step_star"] == step == TR.pick_step(V) and j["V"] == {str(s): V[s] for s in sorted(V)}
    assert j["boundary"] == (step == TINY[-1])
    assert {r["sess"] for r in j["episodes"]} == {"ab2/A"} and "ab2/A" not in j["train_sessions"]
    import torch
    ck = torch.load(base.with_suffix(".pt"), weights_only=False)
    assert ck["step"] == step and TR.state_dict_sha(ck["state_dict"]) == j["state_sha256"]
    assert j["pt_sha256"] == RUN.file_sha(base.with_suffix(".pt"))
    assert j["npz_sha256"] == RUN.file_sha(base.with_suffix(".npz"))
    assert set(j["inputs"]["inner"]) == {RUN.tag(v) for v in RUN.train_sessions("ab2/A")}


def test_list_jobs_binds_config_and_input_fingerprints(tmp_path, syn_data, capsys):
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    capsys.readouterr()
    RUN.cmd_list_jobs(tmp_path, "inner")
    lines = [x for x in capsys.readouterr().out.split("\n") if x]
    assert len(lines) == 6 * (5 * 4 * 3 + 5) and "2a ab5/A ab6/A 0" in lines and "2a all ab6/A 0" in lines
    RUN.cmd_inner(tmp_path, "2a", "ab5/A", "ab6/A", 0, prereg=pre)
    RUN.cmd_list_jobs(tmp_path, "inner")
    assert "2a ab5/A ab6/A 0" not in capsys.readouterr().out.split("\n")
    p = RUN.inner_json(tmp_path, "2a", "ab5/A", "ab6/A", 0)                 # 输入指纹变了 ⇒ 重排
    j = json.loads(p.read_text()); j["input_fp"] = "0" * 16; p.write_text(json.dumps(j))
    RUN.cmd_list_jobs(tmp_path, "inner")
    assert "2a ab5/A ab6/A 0" in capsys.readouterr().out.split("\n")
    RUN.cmd_list_jobs(tmp_path, "final")
    finals = [x for x in capsys.readouterr().out.split("\n") if x]
    assert len(finals) == 6 * (5 * 3 + 1) and "2a all 0" in finals


def test_full_data_fit_is_selected_by_five_session_inner_loso(tmp_path, syn_data, monkeypatch):
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    _all_inner(tmp_path, pre, "2a", RUN.ALL)
    RUN.cmd_final(tmp_path, "2a", RUN.ALL, 0, prereg=pre)
    model, st, Wrs, C = RUN.load_fullfit(tmp_path, "2a")
    j = json.loads(RUN.final_base(tmp_path, "2a", RUN.ALL, 0).with_suffix(".json").read_text())
    assert len(j["V"]) == len(TINY) and "episodes" not in j and Wrs is None
    monkeypatch.setitem(TR.CONFIG, "tie_rtol", 1e-9)                         # 规则变了 ⇒ 作图拒绝旧模型
    with pytest.raises(RuntimeError):
        RUN.load_fullfit(tmp_path, "2a")


@pytest.mark.slow
def test_refs_end_to_end_on_synthetic(tmp_path, syn_data):
    """参照（G-2 同口径 / 旧口径 / AR）在合成数据上走通；产物格式满足汇总的核验；续跑时跳过。"""
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    RUN.cmd_refs(tmp_path, prereg=pre)
    stamp = {}
    for name in ("g2", "g2_whole", "ar"):
        for fold in RUN.FOLDS:
            p = tmp_path / "refs" / name / f"{RUN.tag(fold)}.json"
            j = json.loads(p.read_text())
            assert j["code_sha"] == TR.code_sha() and {r["sess"] for r in j["episodes"]} == {fold}
            assert j["input_fp"] == RUN.fp_of(RUN.refs_inputs(RUN.read_inputs(tmp_path)))
            assert all(set(r["lead"]) == {"3-6", "7-12", "13-27", "28-52", "53-96", "97-102"}
                       for r in j["episodes"])
            if name == "ar":
                assert fold not in j["train_sessions"] and len(j["train_sessions"]) == 4
            stamp[p] = p.stat().st_mtime_ns
    RUN.cmd_refs(tmp_path, prereg=pre)
    assert all(p.stat().st_mtime_ns == t for p, t in stamp.items())


@pytest.mark.slow
def test_prepare_with_the_real_sensor_fit(tmp_path, syn_data, monkeypatch):
    """真实的传感器层拟合走一遍 prepare（快速用例里用的是替身）。"""
    import signal
    import q1_sensor as S
    roster = dict(D.ROSTER)                       # 夹具换上的合成名单
    monkeypatch.undo()
    monkeypatch.setattr(D, "ROSTER", roster)
    monkeypatch.setattr(signal, "alarm", lambda sec: ALARMS.append(sec))
    monkeypatch.setattr(D, "load_bare", lambda *a, **k: syn_data)
    monkeypatch.setattr(D, "calibration", lambda: (np.eye(3), np.zeros(3)))
    monkeypatch.setattr(RUN, "data_files", lambda e: [D.CALIB_JSON])
    RUN.cmd_prepare(tmp_path, prereg=_prereg(tmp_path))
    Wrs, C, man = RUN.load_sensor_set(tmp_path, ("ab2/A", "ab4/P0", "ab7/Apost"))
    assert Wrs.shape == (S.NLAG * S.N_FEAT + 1, 3) and len(C) == 100 and np.abs(Wrs).max() > 0


# ── 数据名单 ──────────────────────────────────────────────────────────────
def _swap(eps):
    """ab5 少一集、ab6 多一集（总数不变）。"""
    extra = T.synthetic_episode(77, sess="ab6/A")
    return [e for e in eps if e.name != eps[0].name] + [extra]


def test_prepare_refuses_a_roster_swap_that_keeps_the_total(tmp_path, syn_data, monkeypatch):
    sw = _swap(syn_data)
    assert len(sw) == len(syn_data)
    monkeypatch.setattr(D, "load_bare", lambda *a, **k: sw)
    with pytest.raises(D.DataContractError):
        RUN.cmd_prepare(tmp_path, prereg=_prereg(tmp_path))


def test_jobs_refuse_data_that_no_longer_matches_the_roster(tmp_path, syn_data, monkeypatch):
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    monkeypatch.setattr(D, "load_bare", lambda *a, **k: _swap(syn_data))
    with pytest.raises(D.DataContractError):
        RUN.cmd_inner(tmp_path, "2a", "ab5/A", "ab6/A", 0, prereg=pre)


# ── 24 小时上限 ───────────────────────────────────────────────────────────
def test_prepare_records_the_deadline_once_and_resume_does_not_reset_it(tmp_path, syn_data):
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    d1 = json.loads(RUN.deadline_file(tmp_path).read_text())
    assert d1["deadline"] - d1["started_at"] == TR.CONFIG["budget_hours"] * 3600 == 24 * 3600
    RUN.cmd_prepare(tmp_path, prereg=pre)
    assert json.loads(RUN.deadline_file(tmp_path).read_text()) == d1
    assert 0 < ALARMS[-1] <= 24 * 3600 + 1                                  # 作业到点会被中止


def test_passed_deadline_stops_every_formal_command_and_marks_incomplete(tmp_path, syn_data):
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    d = json.loads(RUN.deadline_file(tmp_path).read_text())
    d["deadline"] = d["started_at"] - 1
    RUN.deadline_file(tmp_path).write_text(json.dumps(d))
    for call in (lambda: RUN.cmd_prepare(tmp_path, prereg=pre),
                 lambda: RUN.cmd_refs(tmp_path, prereg=pre),
                 lambda: RUN.cmd_inner(tmp_path, "2a", "ab5/A", "ab6/A", 0, prereg=pre),
                 lambda: RUN.cmd_final(tmp_path, "2a", "ab5/A", 0, prereg=pre)):
        with pytest.raises(RUN.DeadlineExceeded):
            call()
    assert json.loads(RUN.incomplete_file(tmp_path).read_text())["status"] == "未完成"
    assert RUN.main(["inner", "--out", str(tmp_path), "--prereg", str(pre), "2a", "ab5/A", "ab6/A", "0"]) == 3


def test_alarm_aborts_a_running_job_and_marks_incomplete(tmp_path, syn_data):
    import signal
    RUN.cmd_prepare(tmp_path, prereg=_prereg(tmp_path))
    handler = signal.getsignal(signal.SIGALRM)
    with pytest.raises(RUN.DeadlineExceeded):
        handler(signal.SIGALRM, None)
    assert RUN.incomplete_file(tmp_path).exists()


def test_deadline_of_another_frozen_experiment_is_not_reused(tmp_path, syn_data):
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    d = json.loads(RUN.deadline_file(tmp_path).read_text())
    d["config_hash"] = "0" * 16
    RUN.deadline_file(tmp_path).write_text(json.dumps(d))
    with pytest.raises(RuntimeError):
        RUN.cmd_prepare(tmp_path, prereg=pre)


def test_jobs_record_when_they_finished(tmp_path, syn_data):
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    RUN.cmd_inner(tmp_path, "2a", "ab5/A", "ab6/A", 0, prereg=pre)
    d = json.loads(RUN.deadline_file(tmp_path).read_text())
    j = json.loads(RUN.inner_json(tmp_path, "2a", "ab5/A", "ab6/A", 0).read_text())
    assert d["started_at"] <= j["finished_at"] <= d["deadline"]


def test_a_roster_failure_does_not_start_the_clock(tmp_path, syn_data, monkeypatch):
    monkeypatch.setattr(D, "load_bare", lambda *a, **k: _swap(syn_data))
    with pytest.raises(D.DataContractError):
        RUN.cmd_prepare(tmp_path, prereg=_prereg(tmp_path))
    assert not RUN.deadline_file(tmp_path).exists()
