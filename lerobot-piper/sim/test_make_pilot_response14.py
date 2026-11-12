"""Only offline unit tests; never collect the repository's hardware tests."""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_pilot_response14 as M


def test_frozen_order_and_budget():
    rows = M.build_runs()
    assert [r['run_id'] for r in rows] == [
        'A0', 'A1', 'A2', 'BP_p_r1', 'BP_m_r1', 'BP_m_r2', 'BP_p_r2',
        'A3', 'BQ_m_r1', 'BQ_p_r1', 'BQ_p_r2', 'BQ_m_r2', 'A4', 'A5']
    assert len(rows) == 14 and len({r['run_id'] for r in rows}) == 14
    assert M.BUDGET == {'max_attempts': 14, 'max_elapsed_s': 2400}
    assert [r['group'] for r in rows].count('A') == 6
    assert all(r['arm'].startswith('B') for r in rows if r['sign'])
    for group in ('P', 'Q'):
        for sign in (-1, 1):
            assert sorted(r['repeat'] for r in rows
                          if r['group'] == group and r['sign'] == sign) == [1, 2]


def test_probe_preserves_p1_amplitude_before_gate_without_rescaling():
    seen = []
    raw = np.arange(2000 * 6, dtype=np.float32).reshape(2000, 6) / 1e7
    def fake(seed, mrad, n):
        seen.append((seed, mrad, n))
        return raw.copy()
    gate = np.array([False, True, False, False])
    got = M.generate_probe(3000, gate, probe_factory=fake)
    expected = raw[:4].copy()
    expected[gate] = 0
    expected[:, 3] = 0
    np.testing.assert_array_equal(got, expected)
    assert seen == [(3000, 1.5, 2000)]
    assert got.dtype == np.float32


def test_counts_follow_runtime_float32_then_python_round():
    cmd = np.array([[0.1234567, -0.0098765, 1.1, 0, -0.4, 0.2, 0.02]], np.float32)
    da = np.array([[0.00123456, 0.00222222, -0.00333333, 0, 0.00444444, -0.00555555]], np.float32)
    total = (cmd[:, :6] + da).astype(np.float32).tolist()
    expected = np.array([[round(x * 57324.840764) for x in row] for row in total])
    np.testing.assert_array_equal(M.planned_counts(cmd, da), expected)
    assert M.planned_counts(cmd, da).dtype == np.int64


def test_nominal_alignment_input_uses_planned_integer_targets():
    cmd = np.array([[0.1234567, -.21, .1, 0, .54, -.3, .0123456]], np.float32)
    counts = M.planned_counts(cmd, np.zeros((1, 6), np.float32))
    restored = M.quantized_nominal(cmd, counts)
    np.testing.assert_array_equal(restored[:, :6], counts.astype(float) / M.JOINT_FACTOR)
    assert restored[0, 6] == round(float(cmd[0, 6]) * 1_000_000) / 1_000_000
    assert not np.array_equal(restored[:, :6], cmd[:, :6].astype(float))
    assert restored.dtype == np.float64


def test_nominal_alignment_gripper_matches_abs_rounded_write_for_negative_input():
    cmd = np.zeros((3, 7), np.float32)
    cmd[:, 6] = [-0.00018105842173099518, 0.0123456, 0.0]
    counts = M.planned_counts(cmd, np.zeros((3, 6), np.float32))
    restored = M.quantized_nominal(cmd, counts)
    expected = np.array([abs(round(float(x) * 1000 * 1000)) / 1_000_000
                         for x in cmd[:, 6]])
    np.testing.assert_array_equal(restored[:, 6], expected)
    assert restored[0, 6] > 0


def test_source_check_requires_identical_float32_nominal_and_gate_length():
    cmd = np.zeros((7, 7), np.float32)
    M.check_source_arrays(cmd, cmd.copy(), np.zeros(7, bool))
    altered = cmd.copy()
    altered[3, 0] = np.nextafter(np.float32(0), np.float32(1))
    with pytest.raises(ValueError, match='nominal'):
        M.check_source_arrays(cmd, altered, np.zeros(7, bool))
    with pytest.raises(ValueError, match='gate'):
        M.check_source_arrays(cmd, cmd.copy(), np.zeros(6, bool))


def test_preflight_only_allows_the_specific_engineering_review_blocker():
    result = M.check_replay_result(['ok'], [], [M.KNOWN_BLOCKER])
    assert result['pending_engineering_authorization'] == [M.KNOWN_BLOCKER]
    with pytest.raises(ValueError, match='blocker'):
        M.check_replay_result([], [], [M.KNOWN_BLOCKER, 'unrecognized problem'])
    with pytest.raises(ValueError, match='degraded'):
        M.check_replay_result([], ['missing position check'], [M.KNOWN_BLOCKER])


def test_probe_validation_does_not_clip_or_repair_an_invalid_factory():
    def oversized(seed, mrad, n):
        return np.ones((n, 6), np.float32) * .011
    with pytest.raises(ValueError, match='peak'):
        M.generate_probe(3000, np.zeros(8, bool), probe_factory=oversized)


def test_frozen_npz_is_deterministic_and_valid():
    fields = {'da': np.zeros((4, 6), np.float32), 'cmd7': np.zeros((4, 7), np.float32)}
    first = M.npz_bytes(**fields)
    assert first == M.npz_bytes(**dict(reversed(list(fields.items()))))
    with np.load(io.BytesIO(first), allow_pickle=False) as archive:
        assert set(archive.files) == {'cmd7', 'da'}
        np.testing.assert_array_equal(archive['da'], fields['da'])


def test_freeze_is_idempotent_and_never_rewrites_identical_files(tmp_path):
    payloads = {'A.npz': b'sequence', 'manifest.json': b'{"schema":"test"}\n'}
    M.freeze_files(tmp_path, payloads)
    timestamps = {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()}
    M.freeze_files(tmp_path, payloads)
    assert timestamps == {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()}


def test_freeze_conflict_is_detected_before_any_new_file(tmp_path):
    M.freeze_files(tmp_path, {'old.npz': b'old'})
    with pytest.raises(FileExistsError):
        M.freeze_files(tmp_path, {'new.npz': b'new', 'old.npz': b'changed'})
    assert not (tmp_path / 'new.npz').exists()
    assert (tmp_path / 'old.npz').read_bytes() == b'old'


def test_canonical_json_has_no_nan_and_is_stable():
    assert M.canonical_json({'z': 2, 'a': 1}) == M.canonical_json({'a': 1, 'z': 2})
    assert json.loads(M.canonical_json({'a': 1})) == {'a': 1}
    with pytest.raises(ValueError):
        M.canonical_json({'value': float('nan')})
