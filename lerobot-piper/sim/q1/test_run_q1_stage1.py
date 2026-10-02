"""正式运行脚本的空跑测试：假解释器只记参数，**不训练、不读数据**。"""
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "run_q1_stage1.sh"


def _fake(tmp_path, inner, final, fail_pattern=None):
    calls = tmp_path / "calls.log"
    lines = ["#!/usr/bin/env bash", f'echo "$* | BLAS=$OPENBLAS_NUM_THREADS" >> "{calls}"']
    if fail_pattern:
        lines.append(f'case "$*" in {fail_pattern}) exit 1 ;; esac')
    for phase, todo in (("inner", inner), ("final", final)):
        body = "".join(t + "\\n" for t in todo)
        lines.append(f'case "$*" in *"--phase {phase}"*) [ -f "{tmp_path}/listed_{phase}" ] || '
                     f'{{ touch "{tmp_path}/listed_{phase}"; printf "{body}"; }} ;; esac')
    fake = tmp_path / "fakepy.sh"
    fake.write_text("\n".join(lines) + "\n")
    fake.chmod(0o755)
    return fake, calls


def _run(tmp_path, fake):
    env = {**os.environ, "PY": str(fake), "OUT": str(tmp_path / "out"), "JOBS": "2"}
    return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)


def _is(line, sub):
    return f"q1_run.py {sub} " in line


def _cmds(calls, sub):
    return [line.split("|")[0].split() for line in calls.read_text().splitlines() if _is(line, sub)]


def test_runs_all_inner_jobs_before_any_final_job(tmp_path):
    fake, calls = _fake(tmp_path, ["1a ab5/A ab6/A 0", "2b all ab7/Apost 0"], ["1a ab5/A 0", "2b all 0"])
    r = _run(tmp_path, fake)
    assert r.returncode == 0, r.stdout + r.stderr
    assert sorted(c[-4:] for c in _cmds(calls, "inner")) == [["1a", "ab5/A", "ab6/A", "0"],
                                                             ["2b", "all", "ab7/Apost", "0"]]
    assert sorted(c[-3:] for c in _cmds(calls, "final")) == [["1a", "ab5/A", "0"], ["2b", "all", "0"]]
    log = calls.read_text().splitlines()
    last_inner = max(i for i, x in enumerate(log) if _is(x, "inner"))
    first_final = min(i for i, x in enumerate(log) if _is(x, "final"))
    assert last_inner < first_final


def test_resume_with_nothing_left_still_finishes(tmp_path):
    """作业已全部完成后续跑：不得以空参数调用作业，且要走完参照、汇总和 6 张图。"""
    fake, calls = _fake(tmp_path, [], [])
    r = _run(tmp_path, fake)
    assert r.returncode == 0, r.stdout + r.stderr
    log = calls.read_text()
    assert not _cmds(calls, "inner") and not _cmds(calls, "final")
    assert "q1_report.py" in log and log.count("fig_pp2_q1.py") == 6


def test_refs_run_before_jobs(tmp_path):
    """参照先跑：它在真实数据上出错时，还没花掉内层作业。"""
    fake, calls = _fake(tmp_path, ["1a ab5/A ab6/A 0"], [])
    assert _run(tmp_path, fake).returncode == 0
    log = calls.read_text().splitlines()
    first_job = next(i for i, line in enumerate(log) if _is(line, "inner"))
    assert any(_is(line, "refs") for line in log[:first_job])


def test_failed_inner_job_stops_the_run(tmp_path):
    fake, calls = _fake(tmp_path, ["1a ab5/A ab6/A 0"], ["1a ab5/A 0"], fail_pattern="*inner*1a*")
    r = _run(tmp_path, fake)
    assert r.returncode != 0
    assert not _cmds(calls, "final") and "fig_pp2_q1.py" not in calls.read_text()


def test_blas_threads_are_pinned_for_bit_stable_fits(tmp_path):
    fake, calls = _fake(tmp_path, [], [])
    assert _run(tmp_path, fake).returncode == 0
    assert all(line.endswith("BLAS=1") for line in calls.read_text().splitlines())
