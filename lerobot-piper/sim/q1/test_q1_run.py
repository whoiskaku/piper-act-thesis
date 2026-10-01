import json

import numpy as np
import pytest

import q1_data as D
import q1_run as RUN
import q1_testing as T
import q1_train as TR


@pytest.fixture()
def syn_data(monkeypatch):
    eps = [T.synthetic_episode(i, sess=f"{s}/{a}") for i, (s, a) in
           enumerate([x for x in D.SESSIONS for _ in range(2)])]
    monkeypatch.setattr(D, "load_bare", lambda *a, **k: eps)
    monkeypatch.setattr(D, "calibration", lambda: (np.eye(3), np.zeros(3)))
    return eps


def _prereg(tmp_path, status="**状态：已冻结（测试）**", sha=None):
    p = tmp_path / "PREREG.md"
    p.write_text(f"# 预注册\n\n{status}\n代码 code_sha={sha or TR.code_sha()[:16]}\n")
    return p


@pytest.mark.parametrize("case", ["missing", "draft", "stale_sha"])
def test_formal_commands_refuse_unless_frozen(tmp_path, monkeypatch, case):
    def boom(*a, **k):
        raise AssertionError("闸门必须在读数据之前")
    monkeypatch.setattr(D, "load_bare", boom)
    pre = {"missing": tmp_path / "none.md",
           "draft": _prereg(tmp_path, status="**状态：草稿（测试），待用户冻结。**"),
           "stale_sha": _prereg(tmp_path, sha="0" * 16)}[case]
    calls = [lambda: RUN.cmd_prepare(tmp_path, prereg=pre),
             lambda: RUN.cmd_job(tmp_path, "2a", "ab2/A", 0, 2, prereg=pre),
             lambda: RUN.cmd_refs(tmp_path, prereg=pre),
             lambda: RUN.cmd_fullfit(tmp_path, "2a", 2, prereg=pre)]
    for call in calls:
        with pytest.raises(RUN.NotFrozenError):
            call()
    assert RUN.main(["prepare", "--out", str(tmp_path), "--prereg", str(pre)]) == 2


def test_timing_needs_no_prereg(syn_data, monkeypatch, capsys):
    monkeypatch.setattr(RUN, "PREREG", D.ROOT / "reports" / "does_not_exist.md")
    RUN.cmd_timing(1)
    assert "s/步" in capsys.readouterr().out


def test_code_sha_covers_the_run_entry():
    """`q1_run.py` 决定写什么进产物（逐集计分、哈希）⇒ 它也必须在代码指纹里。"""
    assert "q1_run.py" in TR.CODE_FILES


def test_list_jobs_skips_done_and_requeues_stale(tmp_path, capsys):
    ch = TR.config_hash(5)
    RUN.atomic_json(RUN.job_json(tmp_path, "1a", "ab5/A", 0), {"config_hash": ch})
    RUN.atomic_json(RUN.job_json(tmp_path, "1a", "ab6/A", 0), {"config_hash": "stale"})
    RUN.cmd_list_jobs(tmp_path, 5)
    lines = capsys.readouterr().out.split("\n")
    assert "1a ab5/A 0" not in lines and "1a ab6/A 0" in lines
    assert len([x for x in lines if x]) == 6 * 5 * 3 - 1


def test_load_sensor_rejects_leaky_manifest(tmp_path):
    man = {"fold": "ab5/A", "train_sessions": ["ab5/A", "ab6/A"], "Wrs_sha256": RUN.sha(np.zeros(2)),
           "C_sha256": RUN.sha(np.zeros(2)), "code": TR.code_sha()}
    RUN.atomic_npz(RUN.sensor_file(tmp_path, "ab5/A"), Wrs=np.zeros(2), C=np.zeros(2),
                   manifest=json.dumps(man))
    with pytest.raises(RuntimeError):
        RUN.load_sensor(tmp_path, "ab5/A")


@pytest.mark.parametrize("model", ["1b", "2a"])
def test_prepare_and_job_end_to_end_on_synthetic(tmp_path, syn_data, model):
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    Wrs, C, man = RUN.load_sensor(tmp_path, "ab2/A")
    assert "ab2/A" not in man["train_sessions"] and len(man["train_sessions"]) == 4
    RUN.cmd_job(tmp_path, model, "ab2/A", 0, steps=2, prereg=pre)
    base = RUN.job_json(tmp_path, model, "ab2/A", 0)
    j = json.loads(base.read_text())
    assert {r["sess"] for r in j["episodes"]} == {"ab2/A"} and "ab2/A" not in j["train_sessions"]
    assert j["config_hash"] == TR.config_hash(2) and j["code_sha"] == TR.code_sha()
    assert j["pt_sha256"] == RUN.file_sha(base.with_suffix(".pt"))
    assert j["npz_sha256"] == RUN.file_sha(base.with_suffix(".npz"))
    r = j["episodes"][0]
    assert 0 <= r["num"] and r["den"] > 0 and set(r["lead"]) == {"3-6", "7-12", "13-27", "28-52", "53-102"}
