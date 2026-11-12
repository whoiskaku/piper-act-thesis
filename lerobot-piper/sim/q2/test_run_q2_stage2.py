"""正式运行脚本的空跑测试：假解释器只记参数，**不训练、不读数据**。"""
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "run_q2_stage2.sh"
PHASES = ("refs", "nom-inner", "nom-final", "inc-inner", "inc-final")
TODO = {"refs": ["ab5", "ab7"], "nom-inner": ["ab5 ab6 0", "ab7 ab5 2"], "nom-final": ["ab5 0"],
        "inc-inner": ["condlinear ab5 ab6 0", "nonlinear ab7 ab6 1"], "inc-final": ["nonlinear ab6 2"]}


def _fake(tmp_path, todo, extra="", silent=None):
    """假解释器：记录每次调用。list-jobs 列出该阶段尚未完成的作业；作业成功退出即记为完成
    （silent 阶段除外：成功退出却不记完成，模拟作业没写出产物）。extra：插在最前面的 shell 片段，用来注入失败。"""
    calls = tmp_path / "calls.log"
    for ph, jobs in todo.items():
        (tmp_path / f"todo_{ph}.txt").write_text("".join(j + "\n" for j in jobs))
    keep = "" if silent is None else f'[ "$sub" = "{silent}" ] && exit 0\n  '
    fake = tmp_path / "fakepy.sh"
    fake.write_text(f"""#!/usr/bin/env bash
echo "$* | BLAS=$OPENBLAS_NUM_THREADS" >> "{calls}"
{extra}
if [ "$2" = "sim/q2/q2_run.py" ]; then
  sub=$3
  case "$sub" in
    prepare) echo "prepared"; exit 0 ;;
    list-jobs)
      f="{tmp_path}/todo_$7.txt"; [ -f "$f" ] || exit 0
      while IFS= read -r l; do [ -f "{tmp_path}/done_$7_$(echo "$l" | tr ' ' _)" ] || echo "$l"; done < "$f"
      exit 0 ;;
  esac
  shift 5
  echo "ran $sub $*"
  {keep}touch "{tmp_path}/done_${{sub}}_$(echo "$*" | tr ' ' _)"
  exit 0
fi
echo "ran $*"
""")
    fake.chmod(0o755)
    return fake, calls


def _run(tmp_path, fake):
    env = {**os.environ, "PY": str(fake), "OUT": str(tmp_path / "out"), "JOBS": "2"}
    return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)


def _is(line, sub):
    return f"q2_run.py {sub} " in line


def _cmds(calls, sub):
    return [line.split("|")[0].split() for line in calls.read_text().splitlines() if _is(line, sub)]


def _status(attempt):
    """status.tsv → [[阶段, 参数, 退出码]]（第一列是时间）。"""
    return [line.split("\t")[1:] for line in (attempt / "status.tsv").read_text().splitlines()]


def test_phases_run_in_order_with_their_arguments(tmp_path):
    fake, calls = _fake(tmp_path, TODO)
    r = _run(tmp_path, fake)
    assert r.returncode == 0, r.stdout + r.stderr
    for ph in PHASES:
        n = len(TODO[ph][0].split())
        assert sorted(c[-n:] for c in _cmds(calls, ph)) == sorted(t.split() for t in TODO[ph])
        assert all(c[c.index(ph) + 1] == "--out" for c in _cmds(calls, ph))
    log = calls.read_text().splitlines()
    pos = {ph: [i for i, x in enumerate(log) if _is(x, ph)] for ph in PHASES}
    for a, b in zip(PHASES, PHASES[1:]):
        assert max(pos[a]) < min(pos[b]), f"{a} 必须全部先于 {b}"
    report = next(i for i, x in enumerate(log) if "q2_report.py" in x)
    assert max(pos["inc-final"]) < report < next(i for i, x in enumerate(log) if "fig_q2.py" in x)


def test_resume_with_nothing_left_still_reports_and_draws(tmp_path):
    fake, calls = _fake(tmp_path, {})
    r = _run(tmp_path, fake)
    assert r.returncode == 0, r.stdout + r.stderr
    assert not any(_cmds(calls, ph) for ph in PHASES)
    assert "q2_report.py" in calls.read_text() and "fig_q2.py" in calls.read_text()


def test_prepare_refusal_stops_before_any_job(tmp_path):
    """预注册未冻结 ⇒ prepare 退出码 2 ⇒ 不跑任何作业；退出码记进本次尝试的 status.tsv。"""
    fake, calls = _fake(tmp_path, TODO, extra='case "$*" in *"q2_run.py prepare"*) exit 2 ;; esac')
    r = _run(tmp_path, fake)
    assert r.returncode != 0 and not any(_cmds(calls, ph) for ph in PHASES)
    assert _status(tmp_path / "out" / "logs" / "attempt_001") == [["prepare", "", "2"]]


def test_failed_job_stops_the_run_before_later_phases(tmp_path):
    fake, calls = _fake(tmp_path, TODO, extra='case "$*" in *"q2_run.py nom-final "*) exit 1 ;; esac')
    r = _run(tmp_path, fake)
    assert r.returncode != 0 and "nom-final" in r.stdout
    assert not _cmds(calls, "inc-inner") and "q2_report.py" not in calls.read_text()


def test_a_job_stopped_by_the_deadline_stops_the_run_with_a_clear_message(tmp_path):
    fake, calls = _fake(tmp_path, TODO, extra='case "$*" in *"q2_run.py inc-inner "*) exit 3 ;; esac')
    r = _run(tmp_path, fake)
    assert r.returncode != 0 and "INCOMPLETE.json" in r.stdout
    assert not _cmds(calls, "inc-final") and "q2_report.py" not in calls.read_text()


def test_jobs_still_listed_after_a_phase_stop_the_run(tmp_path):
    """作业成功退出、list-jobs 却仍列出它（如作业没写出产物）⇒ 停止。"""
    fake, calls = _fake(tmp_path, TODO, silent="nom-final")
    r = _run(tmp_path, fake)
    assert r.returncode != 0 and "todo_nom-final_after" in r.stdout and not _cmds(calls, "inc-inner")


def test_each_attempt_keeps_its_own_logs_and_exit_status(tmp_path):
    """续跑不覆盖失败证据：每次运行新建 attempt 目录，逐作业日志 + status.tsv（时间、阶段、参数、退出码）。"""
    fail = 'case "$*" in *"q2_run.py nom-final "*) echo "boom $*"; exit 1 ;; esac'
    fake, calls = _fake(tmp_path, TODO, extra=fail)
    assert _run(tmp_path, fake).returncode != 0
    fake.write_text(fake.read_text().replace(fail, ""))                 # 修好后续跑
    r = _run(tmp_path, fake)
    assert r.returncode == 0, r.stdout + r.stderr
    logs = tmp_path / "out" / "logs"
    a1, a2 = logs / "attempt_001", logs / "attempt_002"
    assert sorted(p.name for p in logs.iterdir()) == ["attempt_001", "attempt_002"]
    assert "boom" in (a1 / "nom-final_ab5_0.log").read_text()          # 第一次的失败日志还在
    assert "ran nom-final ab5 0" in (a2 / "nom-final_ab5_0.log").read_text()
    s1, s2 = _status(a1), _status(a2)
    assert ["prepare", "", "0"] in s1 and ["nom-final", "ab5 0", "1"] in s1
    assert not any(row[0] == "inc-inner" for row in s1)                # 第一次在名义终训失败处就停了
    assert ["nom-final", "ab5 0", "0"] in s2 and ["report", "", "0"] in s2 and ["fig", "", "0"] in s2
    assert not any(row[0] == "nom-inner" for row in s2)                 # 已完成的作业续跑时跳过
    assert str(a2) in r.stdout


def test_blas_threads_are_pinned(tmp_path):
    fake, calls = _fake(tmp_path, TODO)
    assert _run(tmp_path, fake).returncode == 0
    assert all(line.endswith("BLAS=1") for line in calls.read_text().splitlines())
