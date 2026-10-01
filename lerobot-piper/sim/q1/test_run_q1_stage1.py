"""正式运行脚本的空跑测试：假解释器只记参数，**不训练、不读数据**。"""
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "run_q1_stage1.sh"


def _fake(tmp_path, todo, fail_pattern=None):
    calls = tmp_path / "calls.log"
    lines = ["#!/usr/bin/env bash", f'echo "$*" >> "{calls}"']
    if fail_pattern:
        lines.append(f'case "$*" in {fail_pattern}) exit 1 ;; esac')
    body = "".join(t + "\\n" for t in todo)
    lines.append(f'case "$*" in *list-jobs*) [ -f "{tmp_path}/listed" ] || '
                 f'{{ touch "{tmp_path}/listed"; printf "{body}"; }} ;; esac')
    fake = tmp_path / "fakepy.sh"
    fake.write_text("\n".join(lines) + "\n")
    fake.chmod(0o755)
    return fake, calls


def _run(tmp_path, fake):
    env = {**os.environ, "PY": str(fake), "OUT": str(tmp_path / "out"), "JOBS": "2"}
    return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)


def test_dispatches_listed_jobs(tmp_path):
    fake, calls = _fake(tmp_path, ["1a ab5/A 0", "2b ab7/Apost 2"])
    r = _run(tmp_path, fake)
    assert r.returncode == 0, r.stdout + r.stderr
    jobs = sorted(line.split()[-3:] for line in calls.read_text().splitlines() if " job " in line)
    assert jobs == [["1a", "ab5/A", "0"], ["2b", "ab7/Apost", "2"]]


def test_resume_with_nothing_left_still_finishes(tmp_path):
    """作业已全部完成后续跑：不得以空参数调用 job，且要走完参照、全数据拟合、汇总和 6 张图。"""
    fake, calls = _fake(tmp_path, [])
    r = _run(tmp_path, fake)
    assert r.returncode == 0, r.stdout + r.stderr
    log = calls.read_text()
    assert " job " not in log and "q1_report.py" in log and log.count("fig_pp2_q1.py") == 6


def test_refs_run_before_jobs(tmp_path):
    """参照先跑：它在真实数据上出错时，还没花掉 90 个作业。"""
    fake, calls = _fake(tmp_path, ["1a ab5/A 0"])
    assert _run(tmp_path, fake).returncode == 0
    log = calls.read_text().splitlines()
    first_job = next(i for i, line in enumerate(log) if " job " in line)
    assert any(" refs " in line for line in log[:first_job])


def test_failed_fullfit_stops_the_run(tmp_path):
    """后台的全数据拟合失败必须让脚本以非零退出，不能被 wait 吞掉。"""
    fake, calls = _fake(tmp_path, [], fail_pattern="*fullfit*1b*")
    r = _run(tmp_path, fake)
    assert r.returncode != 0
    assert "fig_pp2_q1.py" not in calls.read_text()
