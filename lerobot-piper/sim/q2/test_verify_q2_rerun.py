"""同种子复跑核对与驱动的测试：合成数据 + 进程内启动器（可注入崩溃、静默算错、截止时间）。"""
import json
import signal
import sys

import numpy as np
import pytest

import q2_data as D2
import q2_report as REP
import q2_run as RUN
import q2_testing as T
import q2_train as TR
import verify_q2_rerun as V


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """合成数据、检查点 (0,1,2)、冻结的测试预注册；闹钟只记录。"""
    eps = T.synthetic_set(2)

    def fake_load(folds=D2.FOLDS, kinds=("bare", "corr", "aux"), check=True):
        o = [e for e in eps if e.kind in kinds and (e.kind == "aux" or e.fold in folds)]
        if check:
            D2.check_roster(o, keys=sorted({f"{e.fold}/{e.arm}" for e in o}))
        return o

    data = tmp_path / "data.json"
    data.write_text("{}")
    monkeypatch.setattr(D2, "load", fake_load)
    monkeypatch.setattr(D2, "ROSTER", T.roster_of(eps))
    monkeypatch.setattr(RUN, "roster_files", lambda: [data])
    monkeypatch.setitem(TR.CONFIG, "ckpts", (0, 1, 2))
    monkeypatch.setitem(TR.CONFIG, "max_steps", 2)
    monkeypatch.setattr(signal, "alarm", lambda sec: None)
    import q1_g2ref as G
    monkeypatch.setattr(G, "load_g2", lambda: None)
    monkeypatch.setattr(G, "episode_ehat", lambda g2, e, ws, R, b: np.zeros_like(e.gap))
    pre = tmp_path / "PREREG.md"
    pre.write_text(f"**状态：已冻结（测试）**\ncode_sha={TR.code_sha()[:16]}，config_hash={TR.config_hash()}\n")
    return tmp_path, pre


class InProc:
    """进程内启动器：直接调用各入口的 main；hook(kind, out, args, try_k) 可返回退出码以模拟失败，或在作业后改写产物。"""

    def __init__(self, pre, hook=None, after=None):
        self.pre, self.hook, self.after, self.calls = pre, hook, after, []

    def start(self, task, try_k):
        kind, out, args = task
        self.calls.append((kind, str(out), list(args), try_k))
        rc = self.hook(kind, out, args, try_k) if self.hook else None
        if rc is None:
            if kind == "report":
                rc = REP.main(["--out", str(out)])
            elif kind == "fig":
                rc = 0
            else:
                rc = RUN.main([kind, "--out", str(out), "--prereg", str(self.pre), *args])
            if self.after:
                self.after(kind, out, args, try_k)
        return V.Done(rc)


def _formal(tmp_path):
    return tmp_path / "formal", tmp_path / "verify"


# ── 摘要与上游复制 ───────────────────────────────────────────────────────────
def test_digest_ignores_timing_and_file_bytes_but_not_results(tmp_path):
    key = ("inc-final", "condlinear", "ab5", 0)
    base = RUN.inc_final_base(tmp_path, "condlinear", "ab5", 0)
    base.parent.mkdir(parents=True)
    rec = {"V": {"0": 0.5}, "step_star": 0, "boundary": False, "state_sha256": "s", "cand": [{"name": "x", "num": 1.0}],
           "aux_cand": [], "aux_beta": [], "sec": 1.0, "finished_at": 10.0, "pt_sha256": "a", "npz_sha256": "b"}
    base.with_suffix(".json").write_text(json.dumps(rec))
    np.savez(base.with_suffix(".npz"), cand__x=np.arange(3.0))
    d0 = V.digest(tmp_path, key)
    base.with_suffix(".json").write_text(json.dumps({**rec, "sec": 9.0, "finished_at": 99.0, "pt_sha256": "z"}))
    assert V.digest(tmp_path, key) == d0
    base.with_suffix(".json").write_text(json.dumps({**rec, "cand": [{"name": "x", "num": 1.0000001}]}))
    assert V.digest(tmp_path, key) != d0
    base.with_suffix(".json").write_text(json.dumps(rec))
    np.savez(base.with_suffix(".npz"), cand__x=np.arange(3.0) + 1e-12)
    assert V.digest(tmp_path, key) != d0
    base.with_suffix(".npz").write_bytes(b"broken")
    assert V.digest(tmp_path, key) is None


# ── 驱动 ───────────────────────────────────────────────────────────────────
def test_clean_run_double_checks_every_job_then_reports(env):
    tmp_path, pre = env
    F, VER = _formal(tmp_path)
    L = InProc(pre)
    assert V.run(F, VER, L, jobs=3) == 0
    st = json.loads((VER / "state.json").read_text())
    assert len(st) == 84 and {s["status"] for s in st.values()} == {"match"}
    assert json.loads((F / "summary.json").read_text())["status"] == "完成"
    assert [c[0] for c in L.calls][-2:] == ["report", "fig"]
    for ph in V.PHASES:                                                 # 每个正式作业都在核对目录里再跑过一遍
        assert sum(1 for c in L.calls if c[0] == ph and c[1] == str(VER / ph)) == sum(1 for k in REP.artifacts(F) if k[0] == ph)
    rel = lambda d: sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file() and p.suffix in (".json", ".pt")
                           and p.name not in ("inputs.json", "deadline.json"))
    nf = rel(VER / "nom-final")                                         # 核对目录只拿到本阶段的上游产物
    assert any(x.startswith("nom/inner/") for x in nf) and not any(x.startswith("inc/") for x in nf)
    ii = rel(VER / "inc-inner")
    assert any(x.startswith("nom/final/") and x.endswith(".pt") for x in ii) and not any(x.startswith("nom/inner/") for x in ii)
    fi = rel(VER / "inc-final")
    assert any(x.startswith("inc/") and "/inner/" in x for x in fi) and any(x.startswith("nom/final/") for x in fi)


def test_crashed_jobs_are_rerun_until_they_succeed(env):
    tmp_path, pre = env
    F, VER = _formal(tmp_path)
    crash = lambda kind, out, args, k: 139 if (kind, list(args), k) in (("nom-inner", ["ab6", "ab5", "0"], 1),
                                                                         ("nom-inner", ["ab6", "ab5", "0"], 2)) else None
    L = InProc(pre, hook=crash)
    assert V.run(F, VER, L, jobs=2) == 0
    tries = [c[3] for c in L.calls if c[0] == "nom-inner" and c[2] == ["ab6", "ab5", "0"] and c[1] == str(F)]
    assert tries == [1, 2, 3]


def test_a_silently_wrong_formal_result_is_detected_and_replaced(env):
    """正式那遍静默算错（文件合法、数值错）：核对两遍一致而与正式不同 ⇒ 正式产物移到 _corrupted/、重跑正式作业，最终全部一致。"""
    tmp_path, pre = env
    F, VER = _formal(tmp_path)
    hit = []

    def corrupt(kind, out, args, k):
        if kind == "nom-inner" and list(args) == ["ab7", "ab5", "1"] and str(out) == str(F) and not hit:
            p = RUN.nom_inner_json(out, "ab7", "ab5", 1)
            j = json.loads(p.read_text())
            j["scores"]["1"]["rho"] *= 1.5
            p.write_text(json.dumps(j))
            hit.append(1)

    assert V.run(F, VER, InProc(pre, after=corrupt), jobs=2) == 0
    s = json.loads((VER / "state.json").read_text())["nom-inner|ab7|ab5|1"]
    assert s["status"] == "match" and [r["who"] for r in s["runs"]] == ["formal", "verify", "verify", "formal"]
    assert list((F / "_corrupted").rglob("*.json"))
    assert json.loads((F / "summary.json").read_text())["status"] == "完成"


def test_a_job_that_never_agrees_stops_the_run_as_unstable(env):
    tmp_path, pre = env
    F, VER = _formal(tmp_path)
    n = []

    def always_different(kind, out, args, k):
        if kind == "nom-inner" and list(args) == ["ab5", "ab6", "0"]:
            p = RUN.nom_inner_json(out, "ab5", "ab6", 0)
            j = json.loads(p.read_text())
            n.append(1)
            j["scores"]["2"]["rho"] += len(n)
            p.write_text(json.dumps(j))

    assert V.run(F, VER, InProc(pre, after=always_different), jobs=2, max_runs=4) == 2
    assert json.loads((VER / "state.json").read_text())["nom-inner|ab5|ab6|0"]["status"] == "unstable"
    assert not (F / "summary.json").exists()


def test_deadline_and_persistent_failures_stop_without_a_verdict(env):
    tmp_path, pre = env
    F, VER = _formal(tmp_path)
    late = lambda kind, out, args, k: 3 if kind == "nom-final" and str(out) == str(F) else None
    assert V.run(F, VER, InProc(pre, hook=late), jobs=2) == 3
    assert not (F / "summary.json").exists()
    F2, VER2 = tmp_path / "f2", tmp_path / "v2"
    broken = lambda kind, out, args, k: 1 if kind == "refs" and list(args) == ["ab7"] else None
    assert V.run(F2, VER2, InProc(pre, hook=broken), jobs=2, tries=3) == 1
    assert not (F2 / "summary.json").exists()


def test_subprocess_launcher_keeps_a_log_and_exit_code_per_try(tmp_path):
    """真实启动器：每次尝试一个独立日志，status.tsv 记录时间、阶段、参数、退出码、第几次。"""
    script = tmp_path / "fake.py"
    script.write_text("import sys; print('ran', sys.argv[1:]); sys.exit(int(sys.argv[-1]))\n")
    L = V.Launcher(tmp_path / "logs", scripts={"nom-inner": script})
    failed = V.run_jobs([("nom-inner", tmp_path, ["ab5", "ab6", "1"]), ("nom-inner", tmp_path, ["ab5", "ab7", "0"])],
                        L.start, jobs=2, tries=2, poll=0.05)
    assert failed == [(("nom-inner", tmp_path, ["ab5", "ab6", "1"]), 1)]
    rows = [r.split("\t") for r in (tmp_path / "logs" / "status.tsv").read_text().splitlines()]
    assert sorted((r[1], r[2], r[3], r[4]) for r in rows) == [("nom-inner", "ab5 ab6 1", "1", "1"),
                                                               ("nom-inner", "ab5 ab6 1", "1", "2"),
                                                               ("nom-inner", "ab5 ab7 0", "0", "1")]
    assert (tmp_path / "logs" / "nom-inner_ab5_ab6_1_try1.log").read_text().startswith("ran")
    assert (tmp_path / "logs" / "nom-inner_ab5_ab6_1_try2.log").exists()


def test_not_part_of_the_code_fingerprint():
    assert "verify_q2_rerun.py" not in TR.CODE_FILES_Q2
