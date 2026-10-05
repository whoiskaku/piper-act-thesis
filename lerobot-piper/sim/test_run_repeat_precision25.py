"""重复精度 25 次的采集入口测试：全部离线，不导入设备后端，不读写真实的 outputs/repeat_precision25。"""
import contextlib
import gc
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

gc.disable()
HERE = Path(__file__).resolve().parent
for _p in (HERE / "q1", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import run_repeat_precision25 as R              # noqa: E402
import testkit_repeat_precision25 as K          # noqa: E402

T = 560
PILOT = HERE.parent / "outputs/pilot_response14"
_sha = K.sha


@pytest.fixture
def env(tmp_path, monkeypatch):
    src, out = tmp_path / "pilot", tmp_path / "out"
    pins = K.make_inputs(src, T)
    monkeypatch.setattr(R, "PILOT_SRC", src)
    monkeypatch.setattr(R, "OUT", out)
    monkeypatch.setattr(R, "RELEASE", tmp_path / "release.json")
    monkeypatch.setattr(R, "FROZEN_INPUTS", pins)
    monkeypatch.setattr(R, "CODE_FILES", (Path(__file__),))
    return tmp_path


def write_release(env, **over):
    return K.write_release(R.RELEASE, R.OUT, R.CODE_FILES, **over)


# ── 顺序 ──────────────────────────────────────────────────────────────────────
DESIGN_TABLE = [  # 方案第 4 节，逐字抄录
    ["bare", "P+", "Q-", "bare", "P-", "Q+"],
    ["bare", "Q+", "P-", "bare", "Q-", "P+"],
    ["bare", "P-", "Q+", "bare", "P+", "Q-"],
    ["bare", "Q-", "P+", "bare", "Q+", "P-"],
]
SEQ_OF = {"bare": "A.npz", "P+": "P_plus.npz", "P-": "P_minus.npz", "Q+": "Q_plus.npz", "Q-": "Q_minus.npz"}


def test_order_is_the_frozen_design_table():
    runs = R.expected_runs()
    assert len(runs) == 25 and len({r["run_id"] for r in runs}) == 25
    a0 = runs[0]
    assert (a0["run_id"], a0["block"], a0["condition"], a0["sequence"], a0["use_probe"]) == ("A0", 0, "A0", "A.npz", False)
    for b, row in enumerate(DESIGN_TABLE, 1):
        got = runs[1 + 6 * (b - 1): 1 + 6 * b]
        assert [r["condition"] for r in got] == row
        assert all(r["block"] == b for r in got)
        assert [r["sequence"] for r in got] == [SEQ_OF[c] for c in row]
        assert [r["use_probe"] for r in got] == [c != "bare" for c in row]
        assert [r["sign"] for r in got] == [0 if c == "bare" else (1 if c.endswith("+") else -1) for c in row]


def test_order_balances_positions_by_sequence_number():
    runs = R.expected_runs()
    pos = {}
    for i, r in enumerate(runs, 1):
        pos.setdefault(r["condition"], []).append(i)
    for c in ("P+", "P-", "Q+", "Q-"):
        assert len(pos[c]) == 4 and np.mean(pos[c]) == 14.0
    assert len(pos["bare"]) == 8 and np.mean(pos["bare"]) == 12.5


# ── prepare ───────────────────────────────────────────────────────────────────
def test_prepare_copies_inputs_with_pinned_hashes_and_writes_manifest(env):
    m = R.prepare()
    for name, digest in R.FROZEN_INPUTS.items():
        assert _sha(R.OUT / name) == digest
    disk = json.loads((R.OUT / "manifest.json").read_text())
    assert disk == m
    assert m["runs"] == R.expected_runs() and m["files"] == R.FROZEN_INPUTS
    assert m["budget"] == {"max_attempts": 25, "max_elapsed_s": 3000}
    assert m["nominal_frames"] == T and m["prehold_frames"] == 4 and m["preflight_complete"] is True


def test_prepare_refuses_tampered_source(env):
    with (R.PILOT_SRC / "Q_plus.npz").open("ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError):
        R.prepare()
    assert not (R.OUT / "manifest.json").exists()


def test_prepare_is_idempotent_but_never_overwrites_a_different_manifest(env):
    R.prepare()
    R.prepare()                                                         # 相同内容：允许
    p = R.OUT / "manifest.json"
    m = json.loads(p.read_text())
    m["budget"]["max_attempts"] = 26
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError):
        R.prepare()
    assert json.loads(p.read_text())["budget"]["max_attempts"] == 26


# ── load_frozen ───────────────────────────────────────────────────────────────
def test_load_frozen_accepts_consistent_release(env):
    R.prepare()
    write_release(env)
    release, m = R.load_frozen()
    assert release["run_set_id"] == R.RUN_SET_ID and len(m["runs"]) == 25


def test_load_frozen_rejects_changed_artifact_bytes(env):
    R.prepare()
    write_release(env)
    with (R.OUT / "P_minus.npz").open("ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="冻结文件被改动"):
        R.load_frozen()


def test_load_frozen_rejects_changed_code(env):
    R.prepare()
    write_release(env, code_sha256={str(Path(__file__).resolve()): "0" * 64})
    with pytest.raises(ValueError, match="代码"):
        R.load_frozen()


def test_load_frozen_rejects_changed_manifest(env):
    R.prepare()
    write_release(env)
    p = R.OUT / "manifest.json"
    m = json.loads(p.read_text())
    m["note"] = "edited"
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="manifest"):
        R.load_frozen()


def test_motion_needs_explicit_authorization_but_offline_check_does_not(env):
    R.prepare()
    write_release(env, motion_authorized=False, waive_missing_rate_limit_only=False)
    with pytest.raises(ValueError, match="运动未授权"):
        R.load_frozen(require_motion=True)
    release, _ = R.load_frozen(require_motion=False)
    assert release["motion_authorized"] is False


def test_escaping_artifact_name_rejected_even_if_pinned(env, monkeypatch):
    R.prepare()
    pins = dict(R.FROZEN_INPUTS, **{"../outside.npz": "0" * 64})
    monkeypatch.setattr(R, "FROZEN_INPUTS", pins)
    p = R.OUT / "manifest.json"
    m = json.loads(p.read_text())
    m["files"] = pins
    p.write_text(json.dumps(m))
    write_release(env)
    with pytest.raises(ValueError, match="普通文件名"):
        R.load_frozen()


@pytest.mark.parametrize("kw,why", [(dict(minus_scale=-0.9), "不成对"), (dict(bare_probe=True), "裸序列含探针"),
                                    (dict(j4=True), "不合法"), (dict(peak=0.011), "不合法")])
def test_invalid_sequences_rejected(env, monkeypatch, kw, why):
    pins = K.make_inputs(R.PILOT_SRC, T, **kw)
    monkeypatch.setattr(R, "FROZEN_INPUTS", pins)
    with pytest.raises(ValueError, match=why):
        R.prepare()


# ── 真实冻结输入（只读）──────────────────────────────────────────────────────
@pytest.mark.skipif(not (PILOT / "manifest.json").exists(), reason="pilot 原始输入不在本机")
def test_real_pilot_inputs_match_the_pins_and_pass_validation(tmp_path, monkeypatch):
    for name, digest in R.FROZEN_INPUTS.items():
        assert _sha(PILOT / name) == digest, name
    pilot_manifest = json.loads((PILOT / "manifest.json").read_text())
    assert dict(pilot_manifest["files"]) == R.FROZEN_INPUTS
    monkeypatch.setattr(R, "OUT", tmp_path / "out")
    m = R.prepare()                                                     # 复制到临时目录并逐项核验
    assert m["nominal_frames"] == 587


# ── 会话、预算、锁停（移植 pilot 入口测试，改为 25 条）────────────────────────
class Clock:
    def __init__(self):
        self.mono, self.unix = 1000.0, 1.7e9

    def advance(self, s):
        self.mono += s
        self.unix += s


@pytest.fixture
def ses(env, monkeypatch):
    R.prepare()
    write_release(env)
    clock = Clock()
    monkeypatch.setattr(R, "now_mono", lambda: clock.mono)
    monkeypatch.setattr(R.time, "time", lambda: clock.unix)
    monkeypatch.setattr(R, "boot_id", lambda: "test-boot")
    monkeypatch.setattr(R, "_safety_audit", lambda *a: None)
    return clock


def fake_replay(clock, *, mutate=None, code=0, advance=60.0):
    """按冻结格式写一条完全合法的记录；可以先改坏再写。"""
    def replay(**kw):
        run, outdir = kw["run"], Path(kw["outdir"])
        with np.load(kw["seqfile"], allow_pickle=False) as z:
            scope = R.validate_replay_context(kw["context"], kw["seqfile"], run["run_id"], z["cmd7"], z["da"],
                                              run["use_probe"])
        rec = K.good_record(run, R.OUT, scope["deadline_monotonic"], t0=clock.mono + 10.0,
                            rotation=np.eye(3))
        if mutate:
            mutate(rec)
        (outdir / f"replay_{run['run_id']}.json").write_text(json.dumps(rec))
        clock.advance(advance)
        return code
    return replay


def run_one(clock, **kw):
    return R.run_next(operator_ready=True, replay_fn=fake_replay(clock, **kw))


def test_operator_readiness_required(ses):
    with pytest.raises(ValueError):
        R.start_session()
    assert not (R.OUT / "session.json").exists()


def test_clock_starts_once_and_survives_restart(ses):
    R.start_session(operator_ready=True)
    with pytest.raises(ValueError):
        R.start_session(operator_ready=True)
    assert R.read_session()["deadline_monotonic"] == 1000.0 + 3000


def test_unauthorized_release_refuses_session_and_creates_no_ledger(ses, env):
    write_release(env, motion_authorized=False, waive_missing_rate_limit_only=False)
    with pytest.raises(ValueError, match="运动未授权"):
        R.start_session(operator_ready=True)
    assert not (R.OUT / "session.json").exists()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert R.main(["--dry-run"]) == 0
    assert "未授权" in buf.getvalue()


def test_existing_records_or_alignment_block_a_new_session(ses):
    (R.OUT / "records").mkdir()
    (R.OUT / "records" / "replay_A0.json").write_text("{}")
    with pytest.raises(ValueError):
        R.start_session(operator_ready=True)
    (R.OUT / "records" / "replay_A0.json").unlink()
    (R.OUT / "alignment_A0.json").write_text("{}")
    with pytest.raises(ValueError):
        R.start_session(operator_ready=True)
    assert not (R.OUT / "session.json").exists()


def test_one_call_one_attempt_no_automatic_next(ses):
    R.start_session(operator_ready=True)
    assert run_one(ses) == 0
    s = R.read_session()
    assert [a["status"] for a in s["attempts"]] == ["complete"] and s["attempts"][0]["run_id"] == "A0"
    assert s["attempts"][0]["record_sha256"] == _sha(R.OUT / "records" / "replay_A0.json")


def test_failure_counts_and_prevents_retry(ses):
    R.start_session(operator_ready=True)
    assert run_one(ses, code=3) == 3
    s = R.read_session()
    assert len(s["attempts"]) == 1 and s["halted"] is True
    with pytest.raises(ValueError):
        run_one(ses)


def test_exception_counts_and_halts(ses):
    R.start_session(operator_ready=True)

    def boom(**kw):
        raise RuntimeError("fake backend error")
    assert R.run_next(operator_ready=True, replay_fn=boom) != 0
    s = R.read_session()
    assert s["attempts"][0]["status"] == "failed" and s["halted"] is True


def test_less_than_60s_remaining_refuses_without_backend_or_attempt(ses):
    R.start_session(operator_ready=True)
    ses.advance(3000 - 59)

    def forbidden(**kw):
        raise AssertionError("backend invoked")
    with pytest.raises(ValueError):
        R.run_next(operator_ready=True, replay_fn=forbidden)
    assert R.read_session()["attempts"] == []


def test_replay_ending_after_deadline_is_failure(ses):
    R.start_session(operator_ready=True)
    ses.advance(3000 - 100)
    assert run_one(ses, advance=200.0) != 0                             # 复放结束时已过截止
    s = R.read_session()
    assert s["attempts"][0]["status"] == "failed" and s["halted"] is True


def test_incomplete_attempt_never_resumed(ses):
    R.start_session(operator_ready=True)
    s = R.read_session()
    s["attempts"] = [dict(run_id="A0", status="running")]
    (R.OUT / "session.json").write_text(json.dumps(s))
    with pytest.raises(ValueError):
        run_one(ses)


def test_reboot_refused(ses, monkeypatch):
    R.start_session(operator_ready=True)
    monkeypatch.setattr(R, "boot_id", lambda: "different-boot")
    with pytest.raises(ValueError):
        run_one(ses)


def test_forged_context_without_reserved_attempt_rejected(ses):
    R.start_session(operator_ready=True)
    with np.load(R.OUT / "A.npz") as z:
        with pytest.raises(ValueError):
            R.validate_replay_context({}, str(R.OUT / "A.npz"), "A0", z["cmd7"], z["da"], False)


def test_short_record_is_failure_not_success(ses):
    R.start_session(operator_ready=True)
    assert run_one(ses, mutate=lambda r: r.update(imu=r["imu"][:10])) != 0
    assert R.read_session()["halted"] is True


def test_missing_send_times_do_not_count_as_complete(ses):
    R.start_session(operator_ready=True)
    assert run_one(ses, mutate=lambda r: r.pop("t_send")) != 0
    assert R.read_session()["halted"] is True


def test_probe_needs_frozen_a0(ses, monkeypatch):
    monkeypatch.setattr(R, "freeze_a0", lambda: None)                  # 不产生对齐文件
    R.start_session(operator_ready=True)
    assert run_one(ses) == 0                                            # A0
    assert run_one(ses) == 0                                            # b1_bare1
    with pytest.raises(ValueError):                                     # b1_Pp：没有冻结的 A0 对齐
        run_one(ses)


def test_executor_side_validation_also_requires_a0_alignment(ses, monkeypatch):
    monkeypatch.setattr(R, "freeze_a0", lambda: None)
    R.start_session(operator_ready=True)
    run_one(ses)
    run_one(ses)
    s = R.read_session()
    s["attempts"].append(dict(run_id="b1_Pp", status="running"))      # 直接伪造一个已预留的探针条
    (R.OUT / "session.json").write_text(json.dumps(s))
    with np.load(R.OUT / "P_plus.npz") as z:
        with pytest.raises(ValueError, match="A0"):
            R.validate_replay_context(dict(token=s["token"], run_id="b1_Pp"), str(R.OUT / "P_plus.npz"), "b1_Pp",
                                      z["cmd7"], z["da"], True)


def test_a0_alignment_comes_from_a0_only(ses):
    R.start_session(operator_ready=True)
    ang = np.radians(30.0)
    rot = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])

    def replay(**kw):
        run = kw["run"]
        with np.load(kw["seqfile"], allow_pickle=False) as z:
            scope = R.validate_replay_context(kw["context"], kw["seqfile"], run["run_id"], z["cmd7"], z["da"], False)
        rec = K.good_record(run, R.OUT, scope["deadline_monotonic"], t0=ses.mono + 10, rotation=rot)
        (Path(kw["outdir"]) / f"replay_{run['run_id']}.json").write_text(json.dumps(rec))
        return 0
    assert R.run_next(operator_ready=True, replay_fn=replay) == 0
    a = json.loads((R.OUT / "alignment_A0.json").read_text())
    assert a["determined_by"] == "independent A0 only" and abs(a["angle_deg"] - 30.0) < 1.0
    assert R.read_session()["alignment_sha256"] == _sha(R.OUT / "alignment_A0.json")


def test_all_twenty_five_in_frozen_order_and_no_twenty_sixth(ses):
    R.start_session(operator_ready=True)
    for _ in range(25):
        assert run_one(ses, advance=60.0) == 0
    s = R.read_session()
    assert [a["run_id"] for a in s["attempts"]] == [r["run_id"] for r in R.expected_runs()]
    assert all(a["status"] == "complete" for a in s["attempts"])
    with pytest.raises(ValueError):
        run_one(ses)


def test_twenty_five_attempts_is_a_hard_limit(ses):
    R.start_session(operator_ready=True)
    s = R.read_session()
    s["attempts"] = [dict(run_id=r["run_id"], status="complete") for r in R.expected_runs()]
    (R.OUT / "session.json").write_text(json.dumps(s))
    with pytest.raises(ValueError):
        run_one(ses)


def test_changed_release_stops_existing_session(ses, env):
    R.start_session(operator_ready=True)
    rel = json.loads(R.RELEASE.read_text())
    rel["note"] = "changed"
    R.RELEASE.write_text(json.dumps(rel))
    with pytest.raises(ValueError):
        run_one(ses)


def test_wrong_loaded_wave_rejected_before_send(ses):
    R.start_session(operator_ready=True)

    def wrong(**kw):
        with np.load(R.OUT / "P_plus.npz") as z:
            with pytest.raises(ValueError):
                R.validate_replay_context(kw["context"], kw["seqfile"], "A0", z["cmd7"], z["da"], False)
        return 4
    assert R.run_next(operator_ready=True, replay_fn=wrong) == 4


def test_wrong_probe_flag_or_run_id_rejected(ses):
    R.start_session(operator_ready=True)

    def wrong(**kw):
        with np.load(kw["seqfile"]) as z:
            for rid, probe in (("A0", True), ("b1_Pp", False)):
                with pytest.raises(ValueError):
                    R.validate_replay_context(kw["context"], kw["seqfile"], rid, z["cmd7"], z["da"], probe)
        return 4
    assert R.run_next(operator_ready=True, replay_fn=wrong) == 4


def test_session_order_corruption_stops_before_backend(ses):
    R.start_session(operator_ready=True)
    run_one(ses)
    s = R.read_session()
    s["attempts"][0]["run_id"] = "b1_bare1"
    (R.OUT / "session.json").write_text(json.dumps(s))

    def forbidden(**kw):
        raise AssertionError("backend invoked")
    with pytest.raises(ValueError):
        R.run_next(operator_ready=True, replay_fn=forbidden)


def test_cli_has_no_force_or_abbreviation_bypass():
    ap = R.build_parser()
    for bad in (["--run-next", "--force"], ["--run-next", "--operator"], ["--retry"]):
        with pytest.raises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                ap.parse_args(bad)


# ── 记录完整性核验（Task 3）：从一条合法记录出发，逐项破坏 ─────────────────────
@pytest.fixture
def rec_env(env):
    m = R.prepare()
    run = R.expected_runs()[2]                                          # b1_Pp：探针条（指令含 δa）
    assert run["run_id"] == "b1_Pp"
    (R.OUT / "records").mkdir(exist_ok=True)

    def write(mutate=None, which=run):
        rec = K.good_record(which, R.OUT, 4000.0, t0=200.0)
        if mutate:
            mutate(rec)
        (R.OUT / "records" / f"replay_{which['run_id']}.json").write_text(json.dumps(rec))
        return which
    return m, run, write


def test_valid_record_passes_and_reports_coverage(rec_env):
    m, run, write = rec_env
    write()
    info = R.check_record(run, m)
    assert info["sha256"] == _sha(R.OUT / "records" / "replay_b1_Pp.json")
    assert info["coverage_lead_s"] >= 0.2 - 0.005 - 1e-9                  # 100 Hz 从首帧前 0.25 s 开始
    assert info["coverage_tail_s"] > 0 and info["imu100_max_gap_ms"] == 10.0
    assert info["send_interval_max_s"] == pytest.approx(0.05)


def test_valid_bare_and_a0_records_pass(rec_env):
    m, _, write = rec_env
    for run in (R.expected_runs()[0], R.expected_runs()[1]):            # A0、b1_bare1
        write(which=run)
        R.check_record(run, m)


def _set(path, value):
    def f(rec):
        d = rec
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value
    return f


def _shift_send_from(k, dt):
    def f(rec):
        ts = np.asarray(rec["t_send"])
        ts[k:] += dt
        rec["t_send"] = ts.tolist()
        rec["t"] = (ts + 0.002).tolist()
    return f


def _perturb_cmd(dx):
    def f(rec):
        rec["cmd"][300][0] += dx
    return f


def _nextafter_cmd(rec):
    v = np.float32(rec["cmd"][300][0])
    rec["cmd"][300][0] = float(np.nextafter(v, np.float32(np.inf)))


def _drop_imu100(sl):
    def f(rec):
        i = np.asarray(rec["imu100"])
        rec["imu100"] = np.delete(i, sl, axis=0).tolist()
    return f


def _imu100_ms(rec):
    i = np.asarray(rec["imu100"])
    i[500, 1] = i[499, 1]
    rec["imu100"] = i.tolist()


def _alter_20hz(rec):
    rec["imu"][123][4] += 1e-6


def _nan_imu(rec):
    rec["imu"][10][3] = float("nan")


def _t_not_increasing(rec):
    rec["t"][50] = rec["t"][49]


def _prehold_frames(rec):
    for k in ("t", "t_send", "t_send_span", "cmd"):
        rec["prehold"][k] = rec["prehold"][k][1:]
    rec["prehold"]["frames"] = 3


def _prehold_target_is_first_formal(rec):
    rec["prehold"]["target"] = rec["cmd"][0]
    rec["prehold"]["cmd"] = [rec["cmd"][0]] * 4


def _prehold_late(rec):
    rec["prehold"]["t_send"][-1] = rec["t_send"][0] + 0.01


def _prehold_mixed_into_formal(rec):
    rec["cmd"] = rec["prehold"]["cmd"] + rec["cmd"]


BREAKS = [
    ("stopped", _set(["stopped"], "用户中断"), "未完成"),
    ("return_blocked", _set(["return_blocked"], True), "归位"),
    ("return_deadline", _set(["return_deadline"], True), "归位"),
    ("short", lambda r: r.update(imu=r["imu"][:-1]), "帧数"),
    ("nan", _nan_imu, "有限"),
    ("t_not_increasing", _t_not_increasing, "时间戳"),
    ("identity_run", _set(["scope", "run_id"], "b1_Pm"), "身份"),
    ("identity_seq_sha", _set(["scope", "seq_sha256"], "0" * 64), "身份"),
    ("identity_release", _set(["scope", "release_id"], "pilot-response14-20260930"), "身份"),
    ("counts", _perturb_cmd(1e-4), "发送计数"),
    ("exact_cmd", _nextafter_cmd, "正式指令"),
    ("imu100_gap", _drop_imu100(slice(700, 702)), "15 ms"),
    ("imu100_ms_not_monotonic", _imu100_ms, "设备毫秒"),
    ("20hz_not_in_100hz", _alter_20hz, "20 Hz"),
    ("send_interval", _shift_send_from(280, 0.051), "写总线间隔"),
    ("coverage", _drop_imu100(slice(0, 25)), "覆盖"),
    ("prehold_missing", lambda r: r.pop("prehold"), "前置保持"),
    ("prehold_frames", _prehold_frames, "前置保持"),
    ("prehold_target", _prehold_target_is_first_formal, "目标"),
    ("prehold_late", _prehold_late, "前置保持"),
    ("prehold_mixed", _prehold_mixed_into_formal, "帧数"),
]


@pytest.mark.parametrize("name,mutate,why", BREAKS, ids=[b[0] for b in BREAKS])
def test_every_integrity_break_is_rejected(rec_env, name, mutate, why):
    m, run, write = rec_env
    write(mutate)
    with pytest.raises(ValueError, match=why):
        R.check_record(run, m)


def test_integrity_failure_halts_the_session(ses):
    R.start_session(operator_ready=True)
    assert run_one(ses, mutate=_alter_20hz) != 0
    s = R.read_session()
    assert s["halted"] is True and "20 Hz" in s["attempts"][0]["exception"]
    with pytest.raises(ValueError):
        run_one(ses)
