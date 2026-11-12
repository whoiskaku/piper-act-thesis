"""Standalone offline regression; no repository hardware pytest fixtures."""
import contextlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import replay_probe as RP


class ReplayPilotScopeTests(unittest.TestCase):
    def setUp(self):
        self.cmd = np.zeros((4, 7), np.float32)
        self.cmd[:, 4] = 0.52
        self.cmd[:, 6] = 0.02
        self.da = np.zeros((4, 6), np.float32)
        self.da[:, 0] = 0.001
        self.context = dict(release_id="pilot14", seq_sha256="abc", run_id="B_Pp_r1",
                            deadline_monotonic=1000.0)

    def run_dry(self, context, blockers=None, degraded=None, validation_error=None, dry_run=True):
        validator = types.ModuleType("run_pilot_response14")
        calls = []
        def validate(*args):
            calls.append(args)
            if validation_error:
                raise validation_error
            return dict(self.context)
        validator.validate_replay_context = validate
        utils = types.ModuleType("lerobot.common.robot_devices.robots.utils")
        def forbidden_factory(*args, **kwargs):
            raise AssertionError("device factory must not be reached")
        utils.make_robot = forbidden_factory
        with patch.dict("sys.modules", {"run_pilot_response14": validator, utils.__name__: utils}), \
             patch.object(RP, "load_seq", return_value=(self.cmd, self.da, {})), \
             patch.object(RP, "safety_checks", return_value=([], degraded or [], blockers or [])), \
             patch.object(RP.time, "monotonic", return_value=100.0), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = RP.run("B_Pp_r1", dry_run=dry_run, seqfile="frozen.npz",
                        probe_kind="multi_joint_continuous", pilot_context=context)
        return rc, calls

    def test_legacy_blocker_still_denies(self):
        rc, calls = self.run_dry(None, [RP.MISSING_RATE_LIMIT_BLOCKER])
        self.assertEqual(rc, 4)
        self.assertFalse(calls)

    def test_scoped_valid_release_handles_only_known_blocker(self):
        rc, calls = self.run_dry(self.context, [RP.MISSING_RATE_LIMIT_BLOCKER])
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][-1])

    def test_other_blocker_is_not_waived(self):
        rc, _ = self.run_dry(self.context, [RP.MISSING_RATE_LIMIT_BLOCKER, "position check missing"])
        self.assertEqual(rc, 4)

    def test_invalid_release_denies(self):
        rc, _ = self.run_dry(self.context, [RP.MISSING_RATE_LIMIT_BLOCKER],
                             validation_error=ValueError("wrong sha"))
        self.assertEqual(rc, 4)

    def test_degraded_dependencies_deny(self):
        rc, _ = self.run_dry(self.context, [RP.MISSING_RATE_LIMIT_BLOCKER], degraded=["missing EPS"])
        self.assertEqual(rc, 2)

    def test_live_entry_guards_do_not_reach_factory(self):
        for context, blockers, degraded, expected in (
            (None, [RP.MISSING_RATE_LIMIT_BLOCKER], [], 4),
            (self.context, [RP.MISSING_RATE_LIMIT_BLOCKER, "missing position"], [], 4),
            (self.context, [RP.MISSING_RATE_LIMIT_BLOCKER], ["missing EPS"], 2)):
            rc, _ = self.run_dry(context, blockers, degraded, dry_run=False)
            self.assertEqual(rc, expected)

    def test_insufficient_time_denies(self):
        self.context["deadline_monotonic"] = 120.0
        rc, _ = self.run_dry(self.context, [RP.MISSING_RATE_LIMIT_BLOCKER])
        self.assertEqual(rc, 4)

    def test_single_joint_assertion_preserved(self):
        fake = types.ModuleType("lerobot.common.policies.act.b_runtime")
        fake.EPS_B = 0.01
        da = self.da.copy()
        da[:, 1] = 0.001
        with patch.dict("sys.modules", {fake.__name__: fake}):
            with self.assertRaisesRegex(AssertionError, "同帧多关节"):
                RP.safety_checks(self.cmd, da, probe_kind="single_joint_pulse")

    def test_fresh_imu_passes(self):
        robot = types.SimpleNamespace(imu_reader=types.SimpleNamespace(
            get_latest=lambda: (99.98, 1, np.zeros(3), np.zeros(3))))
        with patch.object(RP.time, "monotonic", return_value=100.0):
            RP.pilot_check_imu_freshness(robot)

    def test_stale_or_missing_imu_denies(self):
        for latest in (None, (99.9, 1, np.zeros(3), np.zeros(3)),
                       (float("nan"), 1, np.zeros(3), np.zeros(3)),
                       (100.1, 1, np.zeros(3), np.zeros(3))):
            robot = types.SimpleNamespace(imu_reader=types.SimpleNamespace(get_latest=lambda: latest))
            with patch.object(RP.time, "monotonic", return_value=100.0):
                with self.assertRaises(RuntimeError):
                    RP.pilot_check_imu_freshness(robot)

    def fake_run(self, return_mode="ok", jump_at_send=None, fail_capture=None,
                 offset_start=False, stale_after_send=None, decode_failure=False,
                 connect_failure=False):
        import torch
        clock = types.SimpleNamespace(now=100.0)
        def sleep(t):
            clock.now += max(float(t), 0)
        outer = self
        events = []
        class FakeRobot:
            is_connected = True
            def __init__(self):
                self.is_connected = not connect_failure
                self.q = outer.cmd[0, :6].copy()
                self.arm = types.SimpleNamespace(last_jointctrl_t=None,
                    piper=types.SimpleNamespace(DisconnectPort=lambda: events.append("can_close")))
                self.imu_reader = types.SimpleNamespace(
                    get_latest=lambda: (clock.now - (0.1 if stale_after_send is not None
                        and self.sends >= stale_after_send else 0), 0, np.zeros(3), np.zeros(3)),
                    stop=lambda: events.append("imu_stop"))
                self.sends = 0
                self.captures = 0
            def connect(self):
                self.is_connected = True
                raise RuntimeError("fake partial connect failure")
            def __del__(self):
                if getattr(self, "is_connected", False):
                    events.append("UNSAFE_destructor_home_disable")
            def capture_observation(self):
                self.captures += 1
                if self.captures == fail_capture:
                    raise RuntimeError("fake capture failure")
                q = self.q.copy()
                if offset_start and self.captures >= 102:
                    q[0] += 0.1
                return {"observation.state": torch.from_numpy(q * 180 / np.pi * 1000),
                        "observation.imu_real": torch.zeros(6)}
            def send_action(self, act):
                self.q = act.numpy()[:6].copy()
                self.sends += 1
                if self.sends == jump_at_send:
                    clock.now = 990.0
        robot = FakeRobot()
        utils = types.ModuleType("lerobot.common.robot_devices.robots.utils")
        utils.make_robot = lambda *args, **kwargs: robot
        mb = types.ModuleType("make_probe_b2")
        mb.reach_limits = lambda: (np.full(6, -10.0), np.full(6, 10.0))
        validator = types.ModuleType("run_pilot_response14")
        validator.validate_replay_context = lambda *args: dict(self.context)
        canmod = types.ModuleType("can_hs_recorder")
        class FakeCanRecorder:
            def start_recording(self): return True
            def wait_first(self): return True
            def drain_recording(self): return []
            def stop_recording(self): events.append("can_rec_stop")
            def meta(self, _): return {}
        canmod.CanHighSpdRecorder = FakeCanRecorder
        def broken_decode(_):
            raise RuntimeError("fake decode failure")
        canmod.decode = broken_decode
        def returned(*args):
            if return_mode == "raise":
                raise RuntimeError("fake return failure")
            return return_mode != "blocked"
        with tempfile.TemporaryDirectory() as td, \
             patch.dict("sys.modules", {utils.__name__: utils, "make_probe_b2": mb,
                                        "run_pilot_response14": validator, "can_hs_recorder": canmod}), \
             patch.object(RP, "load_seq", return_value=(self.cmd, self.da, {})), \
             patch.object(RP, "safety_checks", return_value=([], [], [RP.MISSING_RATE_LIMIT_BLOCKER])), \
             patch.object(RP, "staged_return", side_effect=returned) as ret, \
             patch.object(RP.time, "monotonic", side_effect=lambda: clock.now), \
             patch.object(RP.time, "perf_counter", side_effect=lambda: clock.now), \
             patch.object(RP.time, "sleep", side_effect=sleep), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = RP.run("B_Pp_r1", seqfile="frozen.npz", outdir=Path(td),
                        canhs=decode_failure, probe_kind="multi_joint_continuous", pilot_context=self.context)
            rec = json.loads((Path(td) / "replay_B_Pp_r1.json").read_text())
            self.assertEqual(ret.call_count, 0 if connect_failure or fail_capture == 1 else 1)
            robot.__del__()  # Match the real wrapper's destructor, without physical motion.
            self.assertNotIn("UNSAFE_destructor_home_disable", events)
            self.assertIn("imu_stop", events)
            self.assertIn("can_close", events)
            self.assertTrue(rec["pilot_cleanup"]["wrapper_destructor_disarmed"])
        return rc, rec

    def test_fake_backend_complete_and_metadata(self):
        rc, rec = self.fake_run()
        self.assertEqual(rc, 0)
        self.assertEqual(len(rec["imu"]), 4)
        self.assertEqual(rec["pilot"]["run_id"], "B_Pp_r1")
        self.assertTrue(rec["started_at_utc"].endswith("Z"))
        np.testing.assert_allclose(np.asarray(rec["cmd"])[:, :6], self.cmd[:, :6] + self.da)

    def test_return_blocked_is_failure(self):
        rc, rec = self.fake_run("blocked")
        self.assertNotEqual(rc, 0)
        self.assertTrue(rec["return_blocked"])

    def test_return_exception_is_failure_and_recorded(self):
        rc, rec = self.fake_run("raise")
        self.assertNotEqual(rc, 0)
        self.assertIn("fake return failure", rec["return_error"])

    def test_deadline_in_premove_records_and_returns(self):
        rc, rec = self.fake_run(jump_at_send=1)
        self.assertNotEqual(rc, 0)
        self.assertIn("预算", rec["stopped"])
        self.assertEqual(len(rec["imu"]), 0)

    def test_deadline_in_replay_records_and_returns(self):
        rc, rec = self.fake_run(jump_at_send=164)
        self.assertNotEqual(rc, 0)
        self.assertIn("预算", rec["stopped"])
        self.assertLess(len(rec["imu"]), 4)

    def test_capture_after_settle_failure_preserved(self):
        rc, rec = self.fake_run(fail_capture=102)
        self.assertNotEqual(rc, 0)
        self.assertIn("fake capture failure", rec["stopped"])

    def test_premove_pose_gate_failure_preserved(self):
        rc, rec = self.fake_run(offset_start=True)
        self.assertNotEqual(rc, 0)
        self.assertIn("未到位", rec["stopped"])

    def test_pilot_stale_imu_stops_before_next_command(self):
        rc, rec = self.fake_run(stale_after_send=1)
        self.assertNotEqual(rc, 0)
        self.assertIn("IMU", rec["stopped"])
        self.assertEqual(len(rec["imu"]), 0)

    def test_partial_connect_cleanup_has_no_home_or_disable(self):
        rc, rec = self.fake_run(connect_failure=True)
        self.assertNotEqual(rc, 0)
        self.assertIn("partial connect", rec["stopped"])

    def test_initial_capture_cleanup_has_no_home_or_disable(self):
        rc, rec = self.fake_run(fail_capture=1)
        self.assertNotEqual(rc, 0)
        self.assertIn("capture failure", rec["stopped"])

    def test_decode_failure_still_returns_and_cleans(self):
        rc, rec = self.fake_run(decode_failure=True)
        self.assertNotEqual(rc, 0)
        self.assertIn("decode failure", rec["stopped"])


if __name__ == "__main__":
    unittest.main()
