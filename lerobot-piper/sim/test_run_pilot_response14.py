"""Offline only: never imports a robot backend or the repository hardware fixtures."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import run_pilot_response14 as pilot


class PilotBudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.release = self.out / 'release.json'
        for name, value in [('OUT', self.out), ('RELEASE', self.release),
                            ('CODE_FILES', (Path(__file__),))]:
            p = patch.object(pilot, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.clock = 1000.0
        for name, fn in [('now_mono', lambda: self.clock),
                         ('boot_id', lambda: 'test-boot'),
                         ('_safety_audit', lambda *args: None)]:
            p = patch.object(pilot, name, fn)
            p.start()
            self.addCleanup(p.stop)
        self.cmd = np.zeros((560, 7), np.float32)
        self.da = np.zeros((560, 6), np.float32)
        self.da[:, 0] = 0.001
        for name, da in [('A.npz', self.da*0), ('P_plus.npz', self.da),
                         ('P_minus.npz', -self.da), ('Q_plus.npz', self.da*0.7),
                         ('Q_minus.npz', -self.da*0.7)]:
            np.savez(self.out/name, cmd7=self.cmd, da=da)
        np.savez(self.out/'calibration_reference.npz', omega0=np.ones((560, 3)))
        self.manifest = dict(schema='pilot-response14-v1', pilot_id=pilot.PILOT_ID,
                             budget=dict(max_attempts=14, max_elapsed_s=2400),
                             nominal_frames=560, runs=pilot.expected_runs(),
                             preflight_complete=True)
        self.manifest['files'] = {p.name: pilot.sha(p) for p in self.out.glob('*.npz')}
        self.write_json(self.out/'manifest.json', self.manifest)
        self.release_data = dict(pilot_id=pilot.PILOT_ID, motion_authorized=True,
                                 authority='user explicit 同意; engineering decision, not safety proof',
                                 manifest_sha256=pilot.sha(self.out/'manifest.json'),
                                 waive_missing_rate_limit_only=True,
                                 code_sha256={str(Path(__file__).resolve()): pilot.sha(Path(__file__))})
        self.write_json(self.release, self.release_data)

    @staticmethod
    def write_json(path, payload):
        path.write_text(json.dumps(payload))

    def fake_replay(self, **kwargs):
        with np.load(kwargs['seqfile']) as z:
            checked = pilot.validate_replay_context(kwargs['pilot_context'], kwargs['seqfile'],
                                                    kwargs['arm'], z['cmd7'], z['da'],
                                                    kwargs['arm'].startswith('B'))
            self.assertEqual(checked['run_id'], kwargs['arm'])
            cmd = z['cmd7'].copy()
            if kwargs['arm'].startswith('B'):
                cmd[:, :6] += z['da']
        payload = dict(stopped=None, cmd=cmd.tolist(), imu=np.zeros((560, 6)).tolist(),
                       t=np.arange(560).tolist(), t_send=np.arange(560).tolist(),
                       pilot=checked)
        self.write_json(Path(kwargs['outdir'])/f"replay_{kwargs['arm']}.json", payload)
        return 0

    def start(self):
        pilot.start_session(operator_ready=True)

    def run_one(self, replay=None):
        with patch.object(pilot, 'freeze_a0', return_value=None):
            return pilot.run_next(operator_ready=True, replay_fn=replay or self.fake_replay)

    def test_frozen_artifacts_validate(self):
        self.assertEqual(len(pilot.load_frozen()[1]['runs']), 14)

    def test_changed_bytes_rejected(self):
        with (self.out/'P_plus.npz').open('ab') as f:
            f.write(b'changed')
        with self.assertRaises(ValueError):
            pilot.load_frozen()

    def test_operator_readiness_required(self):
        with self.assertRaises(ValueError):
            pilot.start_session()
        self.assertFalse((self.out/'session.json').exists())

    def test_clock_starts_once_and_survives_restart(self):
        self.start()
        with self.assertRaises(ValueError):
            pilot.start_session(operator_ready=True)
        self.assertEqual(pilot.read_session()['deadline_monotonic'], 3400.)

    def test_one_call_one_attempt_no_automatic_next(self):
        self.start()
        self.assertEqual(self.run_one(), 0)
        self.assertEqual(len(pilot.read_session()['attempts']), 1)
        self.assertEqual(pilot.read_session()['attempts'][0]['status'], 'complete')

    def test_failure_counts_and_prevents_retry(self):
        self.start()
        self.assertEqual(self.run_one(lambda **kw: 3), 3)
        self.assertEqual(len(pilot.read_session()['attempts']), 1)
        with self.assertRaises(ValueError):
            self.run_one()

    def test_exception_counts_and_halts(self):
        self.start()
        def fail(**kwargs):
            raise RuntimeError('fake backend error')
        self.assertNotEqual(self.run_one(fail), 0)
        self.assertEqual(pilot.read_session()['attempts'][0]['status'], 'failed')

    def test_deadline_refuses_before_device(self):
        self.start()
        self.clock = 3390.
        with self.assertRaises(ValueError):
            self.run_one(lambda **kw: self.fail('backend invoked'))
        self.assertEqual(len(pilot.read_session()['attempts']), 0)

    def test_incomplete_attempt_never_resumed(self):
        self.start()
        session = pilot.read_session()
        session['attempts'] = [dict(run_id='A0', status='running')]
        self.write_json(self.out/'session.json', session)
        with self.assertRaises(ValueError):
            self.run_one()

    def test_reboot_refused(self):
        self.start()
        with patch.object(pilot, 'boot_id', return_value='different-boot'):
            with self.assertRaises(ValueError):
                self.run_one()

    def test_forged_context_without_reserved_attempt_rejected(self):
        self.start()
        with self.assertRaises(ValueError):
            pilot.validate_replay_context({}, str(self.out/'A.npz'), 'A0',
                                          self.cmd, self.da*0, False)

    def test_short_record_is_failure_not_success(self):
        self.start()
        def short(**kw):
            self.fake_replay(**kw)
            p = Path(kw['outdir'])/'replay_A0.json'
            data = json.loads(p.read_text())
            data['imu'] = data['imu'][:10]
            self.write_json(p, data)
            return 0
        self.assertNotEqual(self.run_one(short), 0)
        self.assertTrue(pilot.read_session()['halted'])

    def test_probe_needs_frozen_a0(self):
        self.start()
        for _ in range(3):
            self.run_one()
        with self.assertRaises(ValueError):
            self.run_one()

    def test_fourteen_is_hard_limit(self):
        self.start()
        session = pilot.read_session()
        session['attempts'] = [dict(run_id=r['run_id'], status='complete')
                               for r in pilot.expected_runs()]
        self.write_json(self.out/'session.json', session)
        with self.assertRaises(ValueError):
            self.run_one()

    def test_artifact_path_escape_rejected(self):
        self.manifest['files']['../outside.npz'] = 'fake'
        self.write_json(self.out/'manifest.json', self.manifest)
        self.release_data['manifest_sha256'] = pilot.sha(self.out/'manifest.json')
        self.write_json(self.release, self.release_data)
        with self.assertRaises(ValueError):
            pilot.load_frozen()

    def test_all_fourteen_signs_match_and_no_fifteenth(self):
        def alignment():
            self.write_json(self.out/'alignment_A0.json', dict(
                determined_by='independent A0 only',
                A0_sha256=pilot.sha(self.out/'records/replay_A0.json'),
                reference_sha256=pilot.sha(self.out/'calibration_reference.npz')))
        self.start()
        with patch.object(pilot, 'freeze_a0', side_effect=alignment):
            for _ in range(14):
                self.assertEqual(pilot.run_next(operator_ready=True, replay_fn=self.fake_replay), 0)
                self.clock += 60
        session = pilot.read_session()
        self.assertEqual([a['run_id'] for a in session['attempts']],
                         [r['run_id'] for r in pilot.expected_runs()])
        self.assertEqual(session['deadline_monotonic'], 3400.)
        with self.assertRaises(ValueError):
            self.run_one()

    def test_changed_release_stops_existing_session(self):
        self.start()
        self.release_data['authority'] += ' changed'
        self.write_json(self.release, self.release_data)
        with self.assertRaises(ValueError):
            self.run_one()

    def test_wrong_loaded_wave_rejected_before_send(self):
        self.start()
        def wrong(**kwargs):
            with self.assertRaises(ValueError):
                pilot.validate_replay_context(kwargs['pilot_context'], kwargs['seqfile'],
                                              kwargs['arm'], self.cmd, self.da, False)
            return 4
        self.assertEqual(self.run_one(wrong), 4)

    def test_wrong_probe_arm_cannot_be_silently_bare(self):
        self.start()
        def wrong(**kwargs):
            with self.assertRaises(ValueError):
                pilot.validate_replay_context(kwargs['pilot_context'], kwargs['seqfile'],
                                              'P+', self.cmd, self.da*0, False)
            return 4
        self.assertEqual(self.run_one(wrong), 4)

    def test_session_order_corruption_stops_before_backend(self):
        self.start()
        self.run_one()
        s = pilot.read_session()
        s['attempts'][0]['run_id'] = 'A5'
        self.write_json(self.out/'session.json', s)
        with self.assertRaises(ValueError):
            self.run_one(lambda **kw: self.fail('backend invoked'))

    def test_missing_send_times_do_not_count_as_complete(self):
        self.start()
        def missing(**kw):
            self.fake_replay(**kw)
            p = Path(kw['outdir'])/'replay_A0.json'
            rec = json.loads(p.read_text())
            rec.pop('t_send')
            self.write_json(p, rec)
            return 0
        self.assertNotEqual(self.run_one(missing), 0)


if __name__ == '__main__':
    unittest.main()
