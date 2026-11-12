#!/usr/bin/env python3
"""One explicitly requested replay at a time; no automatic retry or robot discovery.

The dated user release is an engineering decision, not a numerical safety proof.
Only its frozen pilot may waive the named missing-rate-source blocker. Ordinary
replay_probe calls and archived 3c remain blocked. Python guards prevent mistakes;
they are not a security boundary against editing code/files.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
OUT = ROOT / 'outputs/pilot_response14'
RELEASE = ROOT / 'reports/RELEASE_PILOT_RESPONSE_14.json'
PILOT_ID = 'pilot-response14-20260930'
CODE_FILES = tuple(ROOT / p for p in (
    'sim/run_pilot_response14.py', 'sim/replay_probe.py',
    'sim/make_pilot_response14.py', 'sim/analyze_probe_3a.py',
    'lerobot/common/robot_devices/motors/piper.py',
    'lerobot/common/robot_devices/robots/piper.py'))
BUDGET = dict(max_attempts=14, max_elapsed_s=2400)
MIN_START_S = 60.0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now_mono():
    return time.monotonic()


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def utc():
    return datetime.now(timezone.utc).isoformat()


def expected_runs():
    items = [('A0', 'A', 0, 0), ('A1', 'A', 0, 1), ('A2', 'A', 0, 2),
             ('BP_p_r1', 'P', 1, 1), ('BP_m_r1', 'P', -1, 1),
             ('BP_m_r2', 'P', -1, 2), ('BP_p_r2', 'P', 1, 2),
             ('A3', 'A', 0, 3), ('BQ_m_r1', 'Q', -1, 1),
             ('BQ_p_r1', 'Q', 1, 1), ('BQ_p_r2', 'Q', 1, 2),
             ('BQ_m_r2', 'Q', -1, 2), ('A4', 'A', 0, 4), ('A5', 'A', 0, 5)]
    return [dict(run_id=i, arm=i, group=g, sign=s, repeat=r,
                 sequence='A.npz' if not s else f'{g}_{"plus" if s > 0 else "minus"}.npz')
            for i, g, s, r in items]


def _artifact(name):
    p = OUT / name
    if Path(name).name != name or p.resolve().parent != OUT.resolve():
        raise ValueError(f'Artifact must be an ordinary file inside pilot directory: {name}')
    return p


def load_frozen():
    """Read and verify the fixed release/manifest; never import a hardware backend."""
    release = json.loads(RELEASE.read_text())
    if (release.get('pilot_id') != PILOT_ID or release.get('motion_authorized') is not True
            or release.get('waive_missing_rate_limit_only') is not True):
        raise ValueError('No matching, explicit pilot motion release')
    if release.get('manifest_sha256') != sha(OUT/'manifest.json'):
        raise ValueError('Manifest changed after release')
    expected_code = {str(p.resolve()): sha(p) for p in CODE_FILES}
    if release.get('code_sha256') != expected_code:
        raise ValueError('Execution code changed after release; offline re-verification required')
    m = json.loads((OUT/'manifest.json').read_text())
    if (m.get('schema') != 'pilot-response14-v1' or m.get('pilot_id') != PILOT_ID
            or m.get('budget') != BUDGET or m.get('preflight_complete') is not True):
        raise ValueError('Pilot identity/budget/preflight mismatch')
    expected = expected_runs()
    if len(m.get('runs', [])) != 14:
        raise ValueError('Pilot must contain exactly the frozen 14 entries')
    for got, want in zip(m['runs'], expected):
        for key in ('run_id', 'arm', 'group', 'sign', 'sequence'):
            if got.get(key) != want[key]:
                raise ValueError(f'Frozen order mismatch: {key} at {want["run_id"]}')
    files = m.get('files', {})
    needed = {r['sequence'] for r in expected} | {'calibration_reference.npz'}
    if not needed <= files.keys():
        raise ValueError('Required frozen artifact missing')
    for name, digest in files.items():
        if sha(_artifact(name)) != digest:
            raise ValueError(f'Artifact changed: {name}')
    with np.load(OUT/'A.npz', allow_pickle=False) as z:
        base = z['cmd7'].copy()
    T = m['nominal_frames']
    if not isinstance(T, int) or T < 550 or base.shape != (T, 7):
        raise ValueError('Nominal trajectory/frame count mismatch')
    waves = {}
    for name in {r['sequence'] for r in expected}:
        with np.load(OUT/name, allow_pickle=False) as z:
            cmd, da = z['cmd7'], z['da']
            if (cmd.dtype != np.float32 or da.dtype != np.float32
                    or not np.array_equal(cmd, base) or da.shape != (T, 6)
                    or not np.isfinite(cmd).all() or not np.isfinite(da).all()
                    or np.abs(da).max() > .010000001 or np.any(da[:, 3])):
                raise ValueError(f'Invalid frozen sequence: {name}')
            waves[name] = da.copy()
    if np.any(waves['A.npz']):
        raise ValueError('Bare sequence contains a probe')
    for g in ('P', 'Q'):
        if (not np.any(waves[f'{g}_plus.npz'])
                or not np.array_equal(waves[f'{g}_minus.npz'], -waves[f'{g}_plus.npz'])):
            raise ValueError('Floating probe sign pair mismatch')
    return release, m


def _safety_audit(manifest):
    from replay_probe import safety_checks, MISSING_RATE_LIMIT_BLOCKER
    for name in sorted({r['sequence'] for r in manifest['runs']}):
        with np.load(OUT/name, allow_pickle=False) as z:
            _, degraded, blockers = safety_checks(z['cmd7'], z['da'],
                                                  probe_kind='multi_joint_continuous')
        if degraded or any(b != MISSING_RATE_LIMIT_BLOCKER for b in blockers):
            raise ValueError(f'{name}: unwaived safety failure: {degraded!r}, {blockers!r}')


def _write(path, data, exclusive=False):
    """Durable journal update; output artifacts, not source-file editing."""
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open('x') as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
    else:
        tmp = path.with_name(path.name + '.tmp')
        with tmp.open('w') as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def _lock():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'session.lock').open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise ValueError('Another pilot process owns the session') from e
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def read_session():
    return json.loads((OUT/'session.json').read_text())


def _check_clock(s, required=MIN_START_S):
    if s['boot_id'] != boot_id():
        raise ValueError('Machine rebooted; this pilot session cannot resume')
    remaining = min(s['deadline_monotonic'] - now_mono(),
                    s['started_unix'] + 2400 - time.time())
    if remaining < required or now_mono() < s['started_monotonic']:
        raise ValueError('Pilot time budget exhausted/clock invalid; do not start another motion')
    return remaining


def start_session(operator_ready=False):
    if not operator_ready:
        raise ValueError('Onsite start requires --operator-ready; clock includes resets and waiting')
    with _lock():
        _, m = load_frozen()
        _safety_audit(m)
        if (OUT/'session.json').exists():
            raise ValueError('Session already exists; budget may not be reset')
        if (OUT/'records').exists() and any((OUT/'records').iterdir()):
            raise ValueError('Unjournaled records exist; refuse a fresh budget')
        start = now_mono()
        s = dict(pilot_id=PILOT_ID, token=uuid.uuid4().hex, boot_id=boot_id(),
                 release_sha256=sha(RELEASE), manifest_sha256=sha(OUT/'manifest.json'),
                 started_utc=utc(), started_unix=time.time(), started_monotonic=start,
                 deadline_monotonic=start+2400, attempts=[], halted=False)
        _write(OUT/'session.json', s, exclusive=True)
        return s


def _check_session(s):
    if (s.get('pilot_id') != PILOT_ID or s.get('release_sha256') != sha(RELEASE)
            or s.get('manifest_sha256') != sha(OUT/'manifest.json')):
        raise ValueError('Session/release binding changed')
    if s.get('halted'):
        raise ValueError('Pilot halted after an anomaly; no automatic retry')
    runs = expected_runs()
    if len(s['attempts']) > len(runs):
        raise ValueError('Attempt budget exceeded')
    for a, r in zip(s['attempts'], runs):
        if a.get('run_id') != r['run_id']:
            raise ValueError('Attempt ledger order differs from frozen pilot')
    _check_clock(s)


def _validate_alignment():
    p = OUT/'alignment_A0.json'
    if not p.exists():
        raise ValueError('Freeze independent A0 alignment before any P/Q replay')
    a = json.loads(p.read_text())
    if (a['A0_sha256'] != sha(OUT/'records/replay_A0.json')
            or a['reference_sha256'] != sha(OUT/'calibration_reference.npz')
            or a['determined_by'] != 'independent A0 only'):
        raise ValueError('A0 alignment/source changed')
    s = read_session()
    if s.get('alignment_sha256') != sha(p):
        raise ValueError('Frozen A0 alignment changed after journaling')


def freeze_a0():
    """Uses only the completed A0 and pre-frozen nominal reference, never P/Q."""
    from analyze_probe_3a import fit_frame
    p = OUT/'records/replay_A0.json'
    a0 = np.asarray(json.loads(p.read_text())['imu'], float)[:, 3:6]
    with np.load(OUT/'calibration_reference.npz', allow_pickle=False) as z:
        om = z['omega0']
    if a0.shape != om.shape or not np.isfinite(a0).all() or not np.isfinite(om).all():
        raise ValueError('A0/reference shape or finite check failed')
    R, L, cos, angle = fit_frame(a0, om)
    if not np.isfinite(R).all() or np.linalg.det(R) < 0:
        raise ValueError('A0 did not yield a proper finite rotation; no probing')
    a = dict(determined_by='independent A0 only', frozen_utc=utc(),
             A0_sha256=sha(p), reference_sha256=sha(OUT/'calibration_reference.npz'),
             method='analyze_probe_3a.fit_frame; SHIFTS=0..6; raw orthogonal Procrustes',
             R=R.tolist(), L=int(L), cosine=float(cos), angle_deg=float(angle))
    _write(OUT/'alignment_A0.json', a, exclusive=True)
    return a


def validate_replay_context(context, seqfile, arm, cmd7, da, use_probe):
    """Called by replay_probe immediately before backend construction.

    An arbitrary context dict is not a release: disk hashes, active reservation,
    exact arm/arrays and persistent budget must match independently.
    """
    _, m = load_frozen()
    s = read_session()
    _check_session(s)
    attempts = s['attempts']
    if not attempts or len(attempts) > 14:
        raise ValueError('No budget reservation for this replay')
    a, want = attempts[-1], m['runs'][len(attempts)-1]
    if (a.get('status') != 'running' or a.get('run_id') != want['run_id']
            or context.get('token') != s['token'] or context.get('run_id') != want['run_id']
            or arm != want['arm'] or bool(use_probe) != (want['sign'] != 0)
            or Path(seqfile).resolve() != (OUT/want['sequence']).resolve()):
        raise ValueError('Reserved pilot replay/context mismatch')
    if any(old.get('status') != 'complete' for old in attempts[:-1]):
        raise ValueError('Previous replay incomplete')
    with np.load(OUT/want['sequence'], allow_pickle=False) as z:
        if not np.array_equal(cmd7, z['cmd7']) or not np.array_equal(da, z['da']):
            raise ValueError('Loaded replay arrays differ from frozen sequence')
    if use_probe:
        _validate_alignment()
    return dict(release_id=PILOT_ID, seq_sha256=m['files'][want['sequence']],
                run_id=want['run_id'], deadline_monotonic=s['deadline_monotonic'])


def _complete_record(run, manifest):
    p = OUT/'records'/f'replay_{run["arm"]}.json'
    rec = json.loads(p.read_text())
    T = manifest['nominal_frames']
    imu, cmd = np.asarray(rec.get('imu', [])), np.asarray(rec.get('cmd', []))
    if (rec.get('stopped') is not None or rec.get('return_blocked') or rec.get('return_error')
            or imu.shape != (T, 6) or cmd.shape != (T, 7)
            or not np.isfinite(imu).all() or not np.isfinite(cmd).all()):
        raise ValueError('Incomplete/invalid capture or unsuccessful return')
    for key in ('t', 't_send'):
        stamps = np.asarray(rec.get(key, []), float)
        if stamps.shape != (T,) or not np.isfinite(stamps).all() or np.any(np.diff(stamps) <= 0):
            raise ValueError(f'Incomplete/non-increasing capture timestamps: {key}')
    identity = rec.get('pilot', {})
    if (identity.get('run_id') != run['run_id'] or identity.get('release_id') != PILOT_ID
            or identity.get('seq_sha256') != manifest['files'][run['sequence']]):
        raise ValueError('Capture pilot identity mismatch')
    with np.load(OUT/run['sequence'], allow_pickle=False) as z:
        expected = z['cmd7'].copy()
        if run['sign']:
            expected[:, :6] += z['da']
    if not np.array_equal(cmd.astype(np.float32), expected):
        raise ValueError('Recorded requested commands differ from frozen waveform')
    return sha(p)


def run_next(operator_ready=False, replay_fn=None):
    if not operator_ready:
        raise ValueError('Confirm the scene/reset is ready for THIS replay using --operator-ready')
    with _lock():
        _, m = load_frozen()
        _safety_audit(m)
        s = read_session()
        _check_session(s)
        if len(s['attempts']) >= 14:
            raise ValueError('14 attempts consumed; no extra/replacement replay authorized')
        if any(a.get('status') != 'complete' for a in s['attempts']):
            raise ValueError('An attempt is incomplete/failed; no automatic retry')
        run = m['runs'][len(s['attempts'])]
        if run['sign']:
            _validate_alignment()
        output = OUT/'records'/f'replay_{run["arm"]}.json'
        if output.exists():
            raise ValueError('Result already exists; never overwrite or silently repeat')
        output.parent.mkdir(parents=True, exist_ok=True)
        attempt = dict(run_id=run['run_id'], status='running', started_utc=utc(),
                       started_monotonic=now_mono(), sequence=run['sequence'])
        s['attempts'].append(attempt)
        _write(OUT/'session.json', s)  # Before make_robot/connect, even startup failures count.
        code = 5
        try:
            if replay_fn is None:
                from replay_probe import run as replay_fn
            code = replay_fn(arm=run['arm'], seqfile=str(OUT/run['sequence']),
                             outdir=output.parent, probe_kind='multi_joint_continuous',
                             pilot_context=dict(token=s['token'], run_id=run['run_id']))
            if code == 0:
                attempt['record_sha256'] = _complete_record(run, m)
                if run['run_id'] == 'A0':
                    freeze_a0()
                    if (OUT/'alignment_A0.json').exists():
                        s['alignment_sha256'] = sha(OUT/'alignment_A0.json')
                _check_clock(s, required=0)
            attempt['status'] = 'complete' if code == 0 else 'failed'
        except BaseException as e:
            attempt.update(status='failed', exception=f'{type(e).__name__}: {e}',
                           traceback=traceback.format_exc())
            code = 130 if isinstance(e, KeyboardInterrupt) else 5
        finally:
            attempt.update(exit_code=int(code), ended_utc=utc(), ended_monotonic=now_mono())
            s['halted'] = attempt['status'] != 'complete'
            _write(OUT/'session.json', s)
        print(f'{run["run_id"]}: {attempt["status"]}; attempts {len(s["attempts"])}/14. '
              'No further replay started. Inspect scene and reset before the next command.')
        return int(code)


def main():
    import gc
    gc.disable()
    ap = argparse.ArgumentParser(description=__doc__)
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument('--dry-run', action='store_true')
    action.add_argument('--status', action='store_true')
    action.add_argument('--start-session', action='store_true')
    action.add_argument('--run-next', action='store_true')
    ap.add_argument('--operator-ready', action='store_true')
    a = ap.parse_args()
    try:
        if a.dry_run:
            _, m = load_frozen()
            _safety_audit(m)
            print('Offline preflight passed. No backend imported; no session started; no motion.')
            return 0
        if a.status:
            print(json.dumps(read_session() if (OUT/'session.json').exists() else
                             {'status': 'not_started', 'motion_performed_by_this_command': False},
                             ensure_ascii=False, indent=2))
            return 0
        if a.start_session:
            s = start_session(a.operator_ready)
            print(f'Session clock started {s["started_utc"]}; maximum 40 minutes including resets. '
                  'No device connected. Next: --run-next --operator-ready (A0 only).')
            return 0
        return run_next(a.operator_ready)
    except (OSError, ValueError, KeyError, AssertionError) as e:
        print(f'Pilot refused: {type(e).__name__}: {e}', file=sys.stderr)
        return 4


if __name__ == '__main__':
    raise SystemExit(main())
