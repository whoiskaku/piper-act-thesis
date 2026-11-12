"""重复精度 25 次的端到端回归：入口 → 执行器 → 记录 → 分析，假设备 + 假时钟，全部离线。

- 真实冻结输入（复制到临时目录）、真实 `safety_checks`（每个序列只算一次）、真实执行器与分析；
- 测试专用 release 只放在临时目录；设备工厂只返回假设备；CAN 记录器为假；任何测试都不连接机械臂。
"""
import contextlib
import gc
import hashlib
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
import analyze_repeat_precision25 as Y          # noqa: E402
import replay_probe as RP                       # noqa: E402
import replay_repeat_precision25 as X           # noqa: E402
import run_repeat_precision25 as R              # noqa: E402

torch = pytest.importorskip("torch")
PILOT = HERE.parent / "outputs/pilot_response14"
pytestmark = pytest.mark.skipif(not (PILOT / "manifest.json").exists(), reason="pilot 冻结输入不在本机")

_SAFETY_CACHE = {}
_real_safety = RP.safety_checks


def cached_safety(cmd7, da, model_path=None, probe_kind="single_joint_pulse"):
    key = (np.ascontiguousarray(cmd7).tobytes(), np.ascontiguousarray(da).tobytes(), probe_kind)
    if key not in _SAFETY_CACHE:
        _SAFETY_CACHE[key] = _real_safety(cmd7, da, model_path, probe_kind)
    return _SAFETY_CACHE[key]


class World:
    """一个现场：假时钟 + 每条复放一台新的假设备。"""
    def __init__(self, omega0, amp_p, amp_q, seed, stale_at_run=None):
        self.now, self.unix0 = 1000.0, 1.7e9
        self.omega0, self.amp = omega0, {"P": amp_p, "Q": amp_q}
        self.seed, self.stale_at_run = seed, stale_at_run
        self.run_index, self.run = None, None
        self.sends_after_deadline = 0
        self.deadline = None

    def sleep(self, t):
        self.now += max(float(t), 0.0)


class FakeIMU:
    def __init__(self, w, robot):
        self.w, self.robot = w, robot
        self.phase = 0.0037 + 0.0011 * (w.run_index % 7)               # 每条复放的 100 Hz 网格相位不同
        self.next, self.t0 = None, None
        self.rng = np.random.default_rng(w.seed * 1000 + w.run_index)
        self.noise = {}

    def _k(self):
        return int(np.floor((self.w.now - self.phase) / 0.01))

    def _sample(self, k):
        host = self.phase + 0.01 * k
        g = np.zeros(3)
        if self.t0 is not None:
            tau = host - self.t0
            T = len(self.w.omega0)
            g = np.array([np.interp(tau * 20.0, np.arange(T), self.w.omega0[:, j]) for j in range(3)])
            run = self.w.run
            if run["use_probe"]:
                f = 3.0 if run["group"] == "P" else 4.5
                wv = np.array([np.sin(2 * np.pi * f * tau), np.cos(2 * np.pi * (f + 0.7) * tau),
                               np.sin(2 * np.pi * (f + 1.9) * tau)])
                g = g + run["sign"] * self.w.amp[run["group"]] * wv
        if k not in self.noise:
            self.noise[k] = 0.02 * self.rng.standard_normal(3)
        g = (g + self.noise[k]).astype(np.float32)
        return host, 20000 + 10 * k, np.float32([0.0, 0.0, 9.8]), g

    def get_latest(self):
        host, ms, a, g = self._sample(self._k())
        if self.w.stale_at_run == self.w.run_index and self.robot.sends >= 200:
            host -= 0.1
        return host, ms, a, g

    def start_recording(self):
        self.next = self._k() + 1
        self.t0 = self.w.now + X.PREHOLD_FRAMES / X.FPS               # 第一个正式帧的发送时刻

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
        pass


class FakeRobot:
    def __init__(self, w):
        self.w = w
        self.is_connected = True
        self.q = np.array(RP.SAFE_DISABLE, np.float32)                   # 停在 safe_disable 位姿
        self.arm = types.SimpleNamespace(last_jointctrl_t=None, piper=types.SimpleNamespace(DisconnectPort=lambda: None))
        self.imu_reader = FakeIMU(w, self)
        self.sends = 0

    def capture_observation(self):
        _, _, a, g = self.imu_reader.get_latest()
        return {"observation.state": torch.from_numpy(np.concatenate([self.q * 180 / np.pi * 1000, [0.0]]).astype(np.float32)),
                "observation.imu_real": torch.from_numpy(np.concatenate([a, g]).astype(np.float32))}

    def send_action(self, act):
        if self.w.deadline is not None and self.w.now >= self.w.deadline:
            self.w.sends_after_deadline += 1
        self.q = act.numpy()[:6].copy()
        self.sends += 1
        self.arm.last_jointctrl_t = (self.w.now, self.w.now + 0.0002)


def fake_can():
    """只替换 CAN 记录器（叶子模块）；设备工厂在真实模块上只替换 make_robot，限位用真实 make_probe_b2。"""
    can = types.ModuleType("can_hs_recorder")

    class Rec:
        err = None
        def start_recording(self): return True
        def wait_first(self): return True
        def drain_recording(self): return []
        def stop_recording(self): pass
        def meta(self, _): return {}
    can.CanHighSpdRecorder = Rec
    can.decode = lambda raw: ([], [])
    return {"can_hs_recorder": can}


def session(tmp, monkeypatch, *, amp_p, amp_q, seed=7, stale_at_run=None, cap_after=None, reset_s=30.0):
    """跑一个完整现场（或在锁停 / 时钟触顶处停下），再分析。返回 (analysis, session, world)。"""
    for name, value in (("OUT", tmp / "out"), ("RELEASE", tmp / "release.json"), ("PILOT_SRC", PILOT)):
        monkeypatch.setattr(R, name, value)
    R.prepare()
    rel = dict(schema=R.RELEASE_SCHEMA, run_set_id=R.RUN_SET_ID, budget=R.BUDGET, motion_authorized=True,
               waive_missing_rate_limit_only=True, manifest_sha256=R.sha(R.OUT / "manifest.json"),
               code_sha256=R.code_fingerprints(), note="test-only release in a temporary directory")
    R.RELEASE.write_text(json.dumps(rel))
    with np.load(R.OUT / "calibration_reference.npz") as z:
        w = World(z["omega0"].copy(), amp_p, amp_q, seed, stale_at_run)
    refused = None
    import lerobot.common.robot_devices.robots.utils as U
    # 只替换 sys.modules 里这一个键：patch.dict 退出时会删掉期间新导入的全部模块，
    # 而 safetensors / pyarrow 这类扩展不能在同一进程里二次初始化（现场每条 --run-next 是新进程，不受影响）。
    monkeypatch.setitem(sys.modules, "can_hs_recorder", fake_can()["can_hs_recorder"])
    with patch.object(U, "make_robot", side_effect=lambda *a, **k: FakeRobot(w)), \
         patch.object(RP, "safety_checks", side_effect=cached_safety), \
         patch.object(X, "safety_checks", side_effect=cached_safety), \
         patch.object(X, "_utc", return_value="2026-10-05T00:00:00Z"), \
         patch.object(R, "boot_id", return_value="e2e-boot"), \
         patch.object(R, "utc", return_value="2026-10-05T00:00:00+00:00"), \
         patch.object(X.time, "monotonic", side_effect=lambda: w.now), \
         patch.object(X.time, "perf_counter", side_effect=lambda: w.now), \
         patch.object(X.time, "sleep", side_effect=w.sleep), \
         patch.object(X.time, "time", side_effect=lambda: w.unix0 + w.now), \
         contextlib.redirect_stdout(io.StringIO()):
        s = R.start_session(operator_ready=True)
        w.deadline = s["deadline_monotonic"]
        for i, run in enumerate(R.expected_runs()):
            if cap_after is not None and i == cap_after:
                w.now = w.deadline - 50.0                               # 剩余不足 60 s
            w.run_index, w.run = i, run
            try:
                rc = R.run_next(operator_ready=True)
            except ValueError as e:
                refused = str(e)
                break
            if rc != 0:
                break
            w.now += reset_s                                            # 现场复位与等待
        res = Y.analyze(R.OUT, R.RELEASE, pilot_root=tmp / "no_pilot")
        after_lock = None
        if R.read_session()["halted"]:
            try:
                R.run_next(operator_ready=True)
            except ValueError as e:
                after_lock = str(e)
    return res, R.read_session(), w, refused, after_lock


def digest(res):
    return hashlib.sha256(json.dumps(res, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@pytest.mark.parametrize("amp_p,amp_q,want", [(0.1, 0.1, "support"), (0.0, 0.0, "stop"), (0.1, 0.0, "indeterminate")],
                         ids=["repeatable_response", "no_response", "P_only"])
def test_full_sessions_reach_each_verdict_and_are_reproducible(tmp_path, monkeypatch, amp_p, amp_q, want):
    res1, s1, w1, refused, _ = session(tmp_path / "a", monkeypatch, amp_p=amp_p, amp_q=amp_q)
    assert refused is None and len(s1["attempts"]) == 25 and not s1["halted"]
    assert all(a["status"] == "complete" for a in s1["attempts"])
    assert res1["execution"]["complete"] is True and res1["verdict"]["verdict"] == want
    assert w1.sends_after_deadline == 0
    if want == "indeterminate":
        assert res1["main"]["R_c"]["P+"] > 0.6 and res1["main"]["R_c"]["Q+"] < 0.3
    for a in s1["attempts"]:
        assert a["record_check"]["coverage_lead_s"] >= 0.18            # 前置保持带来的覆盖余量
    _SAFETY_CACHE_SIZE = len(_SAFETY_CACHE)
    res2, *_ = session(tmp_path / "b", monkeypatch, amp_p=amp_p, amp_q=amp_q)
    assert digest(res1) == digest(res2)                                 # 两遍完全一致
    assert len(_SAFETY_CACHE) == _SAFETY_CACHE_SIZE


def test_stale_imu_on_the_third_run_halts_and_is_indeterminate(tmp_path, monkeypatch):
    res, s, w, refused, after_lock = session(tmp_path, monkeypatch, amp_p=0.1, amp_q=0.1, stale_at_run=2)
    assert len(s["attempts"]) == 3 and s["attempts"][2]["status"] == "failed" and s["halted"] is True
    rec = json.loads((R.OUT / "records" / "replay_b1_Pp.json").read_text())
    assert "IMU" in rec["stopped"]
    assert after_lock is not None and "锁停" in after_lock               # 锁停后入口拒绝
    assert res["verdict"]["verdict"] == "indeterminate" and "main" not in res


def test_clock_cap_refuses_the_next_run_without_consuming_an_attempt(tmp_path, monkeypatch):
    res, s, w, refused, _ = session(tmp_path, monkeypatch, amp_p=0.1, amp_q=0.1, cap_after=5)
    assert refused is not None and "时间预算" in refused
    assert len(s["attempts"]) == 5 and all(a["status"] == "complete" for a in s["attempts"])
    assert res["verdict"]["verdict"] == "indeterminate" and "main" not in res
