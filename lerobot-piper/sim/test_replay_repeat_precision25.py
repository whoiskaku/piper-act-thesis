"""重复精度 25 次的单次复放执行器测试：假设备 + 假时钟，全部离线；任何测试都不得到达真实设备工厂。

移植自 pilot 执行器的离线回归（test_replay_pilot_scope.py），另加：归位受截止约束、0.2 秒前置保持单独记录。
"""
import contextlib
import gc
import io
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

gc.disable()
HERE = Path(__file__).resolve().parent
for _p in (HERE / "q1", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import replay_repeat_precision25 as X           # noqa: E402

torch = pytest.importorskip("torch")
PILOT = HERE.parent / "outputs/pilot_response14"
T = 120
RUN_P = dict(run_id="b1_Pp", block=1, group="P", sign=1, condition="P+", sequence="P_plus.npz", use_probe=True)
RUN_BARE = dict(run_id="b1_bare1", block=1, group="A", sign=0, condition="bare", sequence="A.npz", use_probe=False)
RUN_A0 = dict(run_id="A0", block=0, group="A", sign=0, condition="A0", sequence="A.npz", use_probe=False)


def seq(n=T):
    cmd = np.zeros((n, 7), np.float32)
    cmd[:, 0] = 0.3                                                     # 离 safe_disable 足够远：三段归位都要动
    cmd[:, 3] = 0.1
    cmd[:, 4] = 0.52
    cmd[:, 5] = 0.2
    cmd[:, 6] = 0.02
    da = np.zeros((n, 6), np.float32)
    da[:, 0] = 0.001
    return cmd, da


class Harness:
    def __init__(self, cmd, *, stale_after_send=None, fail_capture=None, offset_start=False, connect_failure=False,
                 jump_at_send=None, jump_to=None, start=100.0):
        self.now = start
        self.cmd = cmd
        self.events, self.sent = [], []
        self.stale_after_send, self.fail_capture, self.offset_start = stale_after_send, fail_capture, offset_start
        self.connect_failure, self.jump_at_send, self.jump_to = connect_failure, jump_at_send, jump_to
        self.robot = None

    def sleep(self, t):
        self.now += max(float(t), 0.0)


class FakeIMU:
    """100 Hz 网格（相位 3 ms）；样本值为 float32，与真实读取器一致。"""
    def __init__(self, h):
        self.h, self.phase, self.next = h, 0.003, None

    def _k(self):
        return int(np.floor((self.h.now - self.phase) / 0.01))

    def _sample(self, k):
        host = self.phase + 0.01 * k
        g = np.float32([0.1 * np.sin(2 * np.pi * 1.3 * host), 0.05 * np.cos(2 * np.pi * 2.1 * host), 0.02])
        return host, 9000 + 10 * k, np.float32([0.0, 0.0, 9.8]), g

    def get_latest(self):
        host, ms, a, g = self._sample(self._k())
        r = self.h.robot
        if self.h.stale_after_send is not None and r is not None and r.sends >= self.h.stale_after_send:
            host -= 0.1
        return host, ms, a, g

    def start_recording(self):
        self.next = self._k() + 1

    def drain_recording(self):
        if self.next is None:
            return []
        k1 = self._k()
        out = [self._sample(k) for k in range(self.next, k1 + 1)]
        self.next = k1 + 1
        return out

    def stop_recording(self):
        self.next = None

    def stop(self):
        self.h.events.append("imu_stop")


class FakeRobot:
    def __init__(self, h):
        self.h = h
        h.robot = self
        self.is_connected = not h.connect_failure
        self.q = h.cmd[0, :6].copy()
        self.arm = types.SimpleNamespace(last_jointctrl_t=None,
                                         piper=types.SimpleNamespace(DisconnectPort=lambda: h.events.append("can_close")))
        self.imu_reader = FakeIMU(h)
        self.sends = self.captures = 0

    def connect(self):
        self.is_connected = True
        raise RuntimeError("fake partial connect failure")

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.h.events.append("UNSAFE_destructor_home_disable")

    def capture_observation(self):
        self.captures += 1
        if self.captures == self.h.fail_capture:
            raise RuntimeError("fake capture failure")
        q = self.q.copy()
        if self.h.offset_start and self.captures >= 102:
            q[0] += 0.1
        _, _, a, g = self.imu_reader.get_latest()
        return {"observation.state": torch.from_numpy(np.concatenate([q * 180 / np.pi * 1000, [0.0]]).astype(np.float32)),
                "observation.imu_real": torch.from_numpy(np.concatenate([a, g]).astype(np.float32))}

    def send_action(self, act):
        a = act.numpy().copy()
        self.q = a[:6].copy()
        self.sends += 1
        self.h.sent.append((self.h.now, a))
        self.arm.last_jointctrl_t = (self.h.now, self.h.now + 0.0002)
        if self.sends == self.h.jump_at_send:
            self.h.now = self.h.jump_to


def fake_can(h, decode_failure=False):
    mod = types.ModuleType("can_hs_recorder")

    class Rec:
        def start_recording(self): return True
        def wait_first(self): return True
        def drain_recording(self): return []
        def stop_recording(self): h.events.append("can_rec_stop")
        def meta(self, _): return {}
        err = None
    mod.CanHighSpdRecorder = Rec

    def decode(_):
        if decode_failure:
            raise RuntimeError("fake decode failure")
        return [], []
    mod.decode = decode
    return mod


@contextlib.contextmanager
def offline(h, *, scope=None, validation_error=None, blockers=None, degraded=None, factory=None,
            decode_failure=False, cmd=None, da=None):
    validator = types.ModuleType("run_repeat_precision25")
    calls = []

    def validate(*args):
        calls.append(args)
        if validation_error:
            raise validation_error
        return dict(scope)
    validator.validate_replay_context = validate
    utils = types.ModuleType("lerobot.common.robot_devices.robots.utils")
    utils.make_robot = factory or (lambda *a, **k: FakeRobot(h))
    mb = types.ModuleType("make_probe_b2")
    mb.reach_limits = lambda: (np.full(6, -10.0), np.full(6, 10.0))
    c, d = seq() if cmd is None else (cmd, da)
    with patch.dict("sys.modules", {"run_repeat_precision25": validator, utils.__name__: utils,
                                    "make_probe_b2": mb, "can_hs_recorder": fake_can(h, decode_failure)}), \
         patch.object(X, "_load", return_value=(c, d)), \
         patch.object(X, "safety_checks", return_value=([], degraded or [],
                                                         [X.MISSING_RATE_LIMIT_BLOCKER] if blockers is None else blockers)), \
         patch.object(X.time, "monotonic", side_effect=lambda: h.now), \
         patch.object(X.time, "perf_counter", side_effect=lambda: h.now), \
         patch.object(X.time, "sleep", side_effect=h.sleep), \
         contextlib.redirect_stdout(io.StringIO()):
        yield calls


DEADLINE = 200.0                                                        # 假时钟从 100 s 开始；完整一条约 15 s + 归位 10.5 s


def scope_for(run, deadline=DEADLINE):
    return dict(release_id="repeat-precision25", run_id=run["run_id"], sequence=run["sequence"],
                seq_sha256="ab" * 32, deadline_monotonic=deadline)


def fake_run(tmp_path, run=RUN_P, *, return_mode="ok", deadline=DEADLINE, decode_failure=False, cmd=None, da=None, **hk):
    c, d = seq() if cmd is None else (cmd, da)
    h = Harness(c, **hk)
    real_return = X.staged_return_guarded

    def returned(robot, torch_, grip, dl):
        h.events.append(("return", dl))
        if return_mode == "raise":
            raise RuntimeError("fake return failure")
        if return_mode == "real":
            return real_return(robot, torch_, grip, dl)
        return "blocked" if return_mode == "blocked" else "ok"
    with offline(h, scope=scope_for(run, deadline), decode_failure=decode_failure, cmd=c, da=d), \
         patch.object(X, "staged_return_guarded", side_effect=returned):
        rc = X.run(run, seqfile=str(tmp_path / run["sequence"]), outdir=tmp_path,
                   context=dict(token="t", run_id=run["run_id"]), canhs=decode_failure)
    rec = json.loads((tmp_path / f"replay_{run['run_id']}.json").read_text())
    if h.robot is not None:
        h.robot.__del__()                                               # 与真实包装器的析构一致（不产生物理动作）
        assert "UNSAFE_destructor_home_disable" not in h.events
        assert "imu_stop" in h.events and "can_close" in h.events
        assert rec["cleanup"]["wrapper_destructor_disarmed"] is True
        assert rec["cleanup"]["motion_or_disable_sent_by_cleanup"] is False
    return rc, rec, h


# ── 作用域核验：拒绝都发生在设备工厂之前 ─────────────────────────────────────
def _forbidden(*a, **k):
    raise AssertionError("device factory must not be reached")


@pytest.mark.parametrize("kw,expected", [
    (dict(validation_error=ValueError("wrong sha")), 4),
    (dict(blockers=[X.MISSING_RATE_LIMIT_BLOCKER, "position check missing"]), 4),
    (dict(degraded=["missing EPS"]), 2),
], ids=["invalid_release", "other_blocker_not_waived", "degraded_dependencies"])
def test_refusals_happen_before_the_device_factory(tmp_path, kw, expected):
    h = Harness(seq()[0])
    with offline(h, scope=scope_for(RUN_P), factory=_forbidden, **kw):
        rc = X.run(RUN_P, seqfile=str(tmp_path / "P_plus.npz"), outdir=tmp_path,
                   context=dict(token="t", run_id="b1_Pp"))
    assert rc == expected


def test_missing_context_refuses_without_validation_or_device(tmp_path):
    h = Harness(seq()[0])
    with offline(h, scope=scope_for(RUN_P), factory=_forbidden) as calls:
        rc = X.run(RUN_P, seqfile=str(tmp_path / "P_plus.npz"), outdir=tmp_path, context=None)
    assert rc == 4 and calls == []


def test_insufficient_time_refuses_before_device(tmp_path):
    h = Harness(seq()[0])
    with offline(h, scope=scope_for(RUN_P, deadline=100.0 + 59.0), factory=_forbidden):
        rc = X.run(RUN_P, seqfile=str(tmp_path / "P_plus.npz"), outdir=tmp_path,
                   context=dict(token="t", run_id="b1_Pp"))
    assert rc == 4


def test_time_is_rechecked_after_slow_imports_before_connecting(tmp_path):
    """与 pilot 相同：导入 torch / make_robot 可能耗时数秒，连接之前再查一次余量。"""
    h = Harness(seq()[0])
    called = []

    class SlowUtils(types.ModuleType):
        def __getattr__(self, name):
            if name == "make_robot":
                h.now += 50.0                                           # 导入期间时间流逝，余量跌破 60 s
                return lambda *a, **k: called.append(1) or FakeRobot(h)
            raise AttributeError(name)
    slow = SlowUtils("lerobot.common.robot_devices.robots.utils")
    with offline(h, scope=scope_for(RUN_P, deadline=100.0 + 100.0)), \
         patch.dict("sys.modules", {slow.__name__: slow}):
        rc = X.run(RUN_P, seqfile=str(tmp_path / "P_plus.npz"), outdir=tmp_path,
                   context=dict(token="t", run_id="b1_Pp"))
    assert rc == 4 and called == []


def test_valid_scope_waives_only_the_known_blocker_and_reaches_the_device(tmp_path):
    rc, rec, h = fake_run(tmp_path)
    assert rc == 0 and h.robot is not None and rec["scope"]["run_id"] == "b1_Pp"


# ── pilot 的 11 种故障路径 ────────────────────────────────────────────────────
def test_complete_record_and_metadata(tmp_path):
    rc, rec, _ = fake_run(tmp_path)
    cmd, da = seq()
    assert rc == 0 and rec["stopped"] is None and len(rec["imu"]) == T
    assert rec["started_at_utc"].endswith("Z")
    np.testing.assert_array_equal(np.asarray(rec["cmd"], np.float32)[:, :6], cmd[:, :6] + da)


def test_return_blocked_is_failure(tmp_path):
    rc, rec, _ = fake_run(tmp_path, return_mode="blocked")
    assert rc != 0 and rec["return_blocked"] is True


def test_return_exception_is_failure_and_recorded(tmp_path):
    rc, rec, _ = fake_run(tmp_path, return_mode="raise")
    assert rc != 0 and "fake return failure" in rec["return_error"]


def test_deadline_reserve_in_premove_records_and_returns(tmp_path):
    rc, rec, h = fake_run(tmp_path, jump_at_send=1, jump_to=DEADLINE - 14.0)
    assert rc != 0 and "预算" in rec["stopped"] and rec["imu"] == []
    assert ("return", DEADLINE) in h.events


def test_deadline_reserve_in_replay_records_and_returns(tmp_path):
    rc, rec, _ = fake_run(tmp_path, jump_at_send=61 + 100 + 4 + 3, jump_to=DEADLINE - 14.0)
    assert rc != 0 and "预算" in rec["stopped"] and 0 < len(rec["imu"]) < T


def test_capture_after_settle_failure_preserved(tmp_path):
    rc, rec, _ = fake_run(tmp_path, fail_capture=102)
    assert rc != 0 and "fake capture failure" in rec["stopped"]


def test_premove_pose_gate_failure_preserved(tmp_path):
    rc, rec, _ = fake_run(tmp_path, offset_start=True)
    assert rc != 0 and "未到位" in rec["stopped"]


def test_stale_imu_stops_before_next_command(tmp_path):
    rc, rec, h = fake_run(tmp_path, stale_after_send=1)
    assert rc != 0 and "IMU" in rec["stopped"] and rec["imu"] == []


def test_partial_connect_cleanup_has_no_home_or_disable(tmp_path):
    rc, rec, h = fake_run(tmp_path, connect_failure=True)
    assert rc != 0 and "partial connect" in rec["stopped"]
    assert not any(isinstance(e, tuple) and e[0] == "return" for e in h.events)


def test_initial_capture_cleanup_has_no_home_or_disable(tmp_path):
    rc, rec, h = fake_run(tmp_path, fail_capture=1)
    assert rc != 0 and "capture failure" in rec["stopped"]
    assert not any(isinstance(e, tuple) and e[0] == "return" for e in h.events)


def test_decode_failure_still_returns_and_cleans(tmp_path):
    rc, rec, h = fake_run(tmp_path, decode_failure=True)
    assert rc != 0 and "decode failure" in rec["stopped"]
    assert any(isinstance(e, tuple) and e[0] == "return" for e in h.events)


# ── 订正 1：归位受截止约束 ────────────────────────────────────────────────────
def _cosine(n):
    return 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, n)))


def test_guarded_return_stops_sending_at_the_deadline_without_speeding_up():
    cmd, _ = seq()
    h = Harness(cmd, start=500.0)
    robot = FakeRobot(h)
    q0 = robot.q.copy()
    deadline = 500.0 + 3.5 + 1.0                                        # 第一段（j6）走完，第二段走到一半
    with patch.object(X.time, "monotonic", side_effect=lambda: h.now), \
         patch.object(X.time, "sleep", side_effect=h.sleep), contextlib.redirect_stdout(io.StringIO()):
        status = X.staged_return_guarded(robot, torch, 0.02, deadline)
    assert status == "deadline"
    times = np.array([t for t, _ in h.sent])
    assert times.max() < deadline and len(times) > 70                  # 到期后零新指令
    stage1 = np.array([a for _, a in h.sent[:70]])
    want_j6 = q0[5] + (0.0 - q0[5]) * _cosine(60)                       # 原速余弦插值（RET_S = 3 s，20 Hz）
    np.testing.assert_allclose(stage1[:60, 5], want_j6, atol=1e-6)
    assert np.all(stage1[60:, 5] == stage1[59, 5])                       # 0.5 s 到位停留
    stage2 = np.array([a for _, a in h.sent[70:]])
    want_j4 = q0[3] + (0.0 - q0[3]) * _cosine(60)[:len(stage2)]
    np.testing.assert_allclose(stage2[:, 3], want_j4, atol=1e-6)        # 第二段同样原速，只是被截断


def test_return_deadline_after_capture_is_recorded_as_failure(tmp_path):
    n_before_return = 61 + 100 + 4 + T
    rc, rec, h = fake_run(tmp_path, return_mode="real", jump_at_send=n_before_return + 1, jump_to=DEADLINE - 0.5)
    assert rc != 0 and rec["return_deadline"] is True and "截止" in rec["stopped"]
    assert len(rec["imu"]) == T                                         # 采集本身完整，失败来自归位到期
    assert max(t for t, _ in h.sent) < DEADLINE


def test_pre_capture_abort_return_is_also_guarded(tmp_path):
    rc, rec, h = fake_run(tmp_path, return_mode="real", jump_at_send=1, jump_to=DEADLINE - 5.0)
    assert rc != 0 and rec["return_deadline"] is True and rec["imu"] == []
    assert max(t for t, _ in h.sent) < DEADLINE


# ── 订正 2：0.2 秒前置保持 ────────────────────────────────────────────────────
@pytest.mark.parametrize("run", [RUN_P, RUN_BARE, RUN_A0], ids=["probe", "bare", "A0"])
def test_prehold_is_four_frames_of_the_settle_target_recorded_separately(tmp_path, run):
    rc, rec, h = fake_run(tmp_path, run=run)
    cmd, da = seq()
    hold = np.concatenate([cmd[0, :6], cmd[0, 6:7]]).astype(np.float32)
    pre = rec["prehold"]
    assert rc == 0 and pre["frames"] == 4 and len(pre["t_send"]) == 4 and len(pre["cmd"]) == 4
    np.testing.assert_array_equal(np.asarray(pre["cmd"], np.float32), np.tile(hold, (4, 1)))
    np.testing.assert_array_equal(np.asarray(pre["target"], np.float32), hold)
    assert max(pre["t_send"]) < rec["t_send"][0]
    sent = cmd.copy()
    if run["use_probe"]:
        sent[:, :6] += da
    np.testing.assert_array_equal(np.asarray(rec["cmd"], np.float32), sent)   # 正式帧数与冻结波形不变
    assert len(rec["cmd"]) == len(rec["t_send"]) == T
    sends = np.array([a for _, a in h.sent])
    first_formal = 61 + 100 + 4
    np.testing.assert_array_equal(sends[first_formal - 5], hold)        # 稳定段最后一帧
    np.testing.assert_array_equal(sends[first_formal - 4:first_formal], np.tile(hold, (4, 1)))   # 目标不变


def test_100hz_stream_starts_well_before_the_first_formal_frame(tmp_path):
    rc, rec, _ = fake_run(tmp_path)
    first_host = rec["imu100"][0][0]
    assert rc == 0 and rec["t_send"][0] - first_host >= 0.19


def test_record_contains_no_absolute_paths(tmp_path):
    rc, rec, _ = fake_run(tmp_path)
    text = json.dumps(rec)
    assert rc == 0 and str(tmp_path) not in text and rec["scope"]["sequence"] == "P_plus.npz"


# ── 与入口完整性核验的衔接（真实冻结序列，587 帧）────────────────────────────
@pytest.mark.skipif(not (PILOT / "manifest.json").exists(), reason="pilot 原始输入不在本机")
def test_executor_record_passes_the_runner_integrity_check(tmp_path, monkeypatch):
    import run_repeat_precision25 as R
    monkeypatch.setattr(R, "OUT", tmp_path / "out")
    m = R.prepare()
    run = m["runs"][2]
    with np.load(R.OUT / run["sequence"], allow_pickle=False) as z:
        cmd, da = z["cmd7"].copy(), z["da"].copy()
    h = Harness(cmd)
    sc = dict(scope_for(run), seq_sha256=m["files"][run["sequence"]])
    (R.OUT / "records").mkdir()
    with offline(h, scope=sc, cmd=cmd, da=da), patch.object(X, "staged_return_guarded", return_value="ok"):
        rc = X.run(run, seqfile=str(R.OUT / run["sequence"]), outdir=R.OUT / "records",
                   context=dict(token="t", run_id=run["run_id"]))
    assert rc == 0
    rec = json.loads((R.OUT / "records" / f"replay_{run['run_id']}.json").read_text())
    assert len(rec["cmd"]) == 587
    info = R.check_record(run, m)
    assert info["coverage_lead_s"] >= 0.18
