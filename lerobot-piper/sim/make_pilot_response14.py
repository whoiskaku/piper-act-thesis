#!/usr/bin/env python3
"""Freeze the authorized 14-replay pilot's offline inputs; never touch hardware.

P/Q use the P1 generator at 1.5 mrad before truncation, the archived fixed
contact gate and j4 exclusion.  No post-gate rescaling, seed search, model
comparison or automatic motion release is performed.  The one G2v0 nominal
forward is exclusively an independent-A0 alignment reference.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import platform
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

PILOT_ID = 'pilot-response14-20260930'
BUDGET = {'max_attempts': 14, 'max_elapsed_s': 2400}
SEEDS = {'P': 3000, 'Q': 3001}
PROBE_N = 2000
PROBE_MRAD = 1.5
PEAK = .010
JOINT_FACTOR = 57324.840764
FPS = 20.0
KNOWN_BLOCKER = (
    'multi_joint_continuous：速度/加速度**无有出处又适用于本设备的上限**'
    '（G-v2 A_MAX 是阶跃拟合的模型参数，裸轨迹本身就超它；'
    '发送路径无软件限幅）⇒ **安全审查未完成**'
)


def canonical_json(value) -> bytes:
    def convert(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(type(obj).__name__)
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                       allow_nan=False, default=convert) + '\n').encode('utf-8')


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def build_runs() -> list[dict]:
    names = ['A0', 'A1', 'A2', 'BP_p_r1', 'BP_m_r1', 'BP_m_r2', 'BP_p_r2',
             'A3', 'BQ_m_r1', 'BQ_p_r1', 'BQ_p_r2', 'BQ_m_r2', 'A4', 'A5']
    rows = []
    for run_id in names:
        if run_id.startswith('A'):
            group, sign, repeat, filename = 'A', 0, int(run_id[1:]), 'A.npz'
        else:
            group, signchar, repeatname = run_id[1:].split('_')
            sign = 1 if signchar == 'p' else -1
            repeat = int(repeatname[1:])
            filename = f'{group}_{"plus" if sign == 1 else "minus"}.npz'
        rows.append({'run_id': run_id, 'arm': run_id, 'sequence': filename,
                     'group': group, 'sign': sign, 'repeat': repeat})
    assert len(rows) == 14 and len({r['run_id'] for r in rows}) == 14
    return rows


def generate_probe(seed: int, gate: np.ndarray, *, probe_factory=None) -> np.ndarray:
    if probe_factory is None:
        from lerobot.common.policies.act.b_correction_deploy import make_probe
        probe_factory = make_probe
    gate = np.asarray(gate)
    if gate.dtype != np.bool_ or gate.ndim != 1 or not 0 < len(gate) <= PROBE_N:
        raise ValueError('invalid fixed gate')
    raw = np.asarray(probe_factory(seed, PROBE_MRAD, n=PROBE_N), np.float32)
    if raw.shape != (PROBE_N, 6) or not np.isfinite(raw).all():
        raise ValueError('invalid probe factory output')
    if float(np.abs(raw).max()) > PEAK + 1e-12:
        raise ValueError('probe peak exceeds frozen limit; not repairing')
    da = raw[:len(gate)].copy()
    da[gate] = 0.0
    da[:, 3] = 0.0
    return da


def planned_counts(cmd7: np.ndarray, da: np.ndarray) -> np.ndarray:
    """Exactly mirror replay float32 addition -> list -> piper.write round."""
    cmd7, da = np.asarray(cmd7, np.float32), np.asarray(da, np.float32)
    if cmd7.ndim != 2 or cmd7.shape[1] != 7 or da.shape != (len(cmd7), 6):
        raise ValueError('command/probe shape mismatch')
    if not np.isfinite(cmd7).all() or not np.isfinite(da).all():
        raise ValueError('non-finite command/probe')
    target = (cmd7[:, :6] + da).astype(np.float32).tolist()
    return np.asarray([[round(x * JOINT_FACTOR) for x in row] for row in target],
                      dtype=np.int64)


def quantized_nominal(cmd7: np.ndarray, counts: np.ndarray) -> np.ndarray:
    result = np.asarray(cmd7, np.float64).copy()
    result[:, :6] = np.asarray(counts, np.float64) / JOINT_FACTOR
    # Mirror write's two multiplications and GripperCtrl(abs(gripper_range)).
    result[:, 6] = [abs(round(float(x) * 1000 * 1000)) / 1_000_000
                    for x in np.asarray(cmd7, np.float32)[:, 6]]
    return result


def check_source_arrays(cmd7, archived_cmd7, gate) -> None:
    cmd7, archived_cmd7 = np.asarray(cmd7, np.float32), np.asarray(archived_cmd7)
    if archived_cmd7.dtype != np.float32 or not np.array_equal(cmd7, archived_cmd7):
        raise ValueError('archived nominal differs from ab5/A_ep0 float32 nominal')
    if np.asarray(gate).dtype != np.bool_ or np.asarray(gate).shape != (len(cmd7),):
        raise ValueError('archived gate has invalid shape/dtype')
    if not np.isfinite(cmd7).all():
        raise ValueError('non-finite nominal')


def check_replay_result(msgs, degraded, blockers) -> dict:
    if degraded:
        raise ValueError(f'replay preflight degraded: {degraded}')
    unknown = [x for x in blockers if x != KNOWN_BLOCKER]
    if unknown:
        raise ValueError(f'unrecognized replay blocker: {unknown}')
    return {'msgs': list(msgs), 'degraded': list(degraded), 'blockers': list(blockers),
            'pending_engineering_authorization': list(blockers)}


def npz_bytes(**arrays) -> bytes:
    """NPZ bytes with fixed ZIP metadata, independent of wall-clock time."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_STORED) as archive:
        for name, value in sorted(arrays.items()):
            data = io.BytesIO()
            np.lib.format.write_array(data, np.asarray(value), allow_pickle=False)
            info = zipfile.ZipInfo(name + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, data.getvalue())
    return buffer.getvalue()


def freeze_files(outdir: Path, payloads: dict[str, bytes]) -> None:
    """Validate all conflicts first; identical files are never overwritten."""
    outdir = Path(outdir)
    for name, payload in payloads.items():
        if Path(name).name != name:
            raise ValueError('frozen filename must not contain directories')
        target = outdir / name
        if target.is_symlink() or (target.exists() and target.read_bytes() != payload):
            raise FileExistsError(f'refusing to overwrite frozen artifact: {target}')
    outdir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        target = outdir / name
        if not target.exists():
            with target.open('xb') as stream:
                stream.write(payload)


def calibration_reference(cmd7: np.ndarray, c0: np.ndarray) -> tuple[bytes, dict]:
    from freeze_pi_g2wall import G2v0
    nominal_q = quantized_nominal(cmd7, c0)
    n = len(cmd7)
    timebase = np.arange(n) / FPS
    model = G2v0()
    omega0 = np.asarray(model.omega({
        'name': PILOT_ID + '-A0-reference', 'cmd': nominal_q,
        't_send': timebase, 't_read': timebase,
        'tg_i': np.arange(n * 5) / (FPS * 5), 'gyro100': np.zeros((n * 5, 3)),
    }), np.float64)
    if omega0.shape != (n, 3) or not np.isfinite(omega0).all():
        raise ValueError(f'G2 nominal calibration reference invalid: {omega0.shape}')
    paths = ['outputs/step4_gvnext/rs/rs_fir.npz',
             'outputs/step4_gvnext/rq/rq_fir.npz', 'outputs/step4_gvnext/rq/select.json']
    metadata = {
        'purpose': 'independent A0 alignment only; no P/Q model predictions',
        'model': 'freeze_pi_g2wall.G2v0 (original v=0)',
        'W_sha256': model.man['W_sha256'],
        'weights': {p: file_sha(ROOT / p) for p in paths},
        'calibration_method': 'analyze_probe_3a.fit_frame; SHIFTS=0..6; A0 only',
        'coordinate_frame': 'native G2 output; no R applied',
        'time_axis': 'nominal uniform 20 Hz; actual replay timestamps recorded separately',
    }
    return npz_bytes(omega0=omega0, cmd7_quantized=nominal_q, planned_counts=c0), metadata


def make_artifacts() -> tuple[dict[str, bytes], dict]:
    gc.disable()
    from make_probe_3c import nominal, robot_limits_safe, checks
    from replay_probe import safety_checks
    import scipy

    source_path = ROOT / 'outputs/ab5/A_ep0.json'
    archived_path = ROOT / 'outputs/probe_3c/probe_3c_p.npz'
    pay = json.loads(source_path.read_text())
    cmd7, starts = nominal(pay)
    cmd7 = cmd7.astype(np.float32)
    with np.load(archived_path, allow_pickle=False) as old:
        gate = old['gate'].copy()
        check_source_arrays(cmd7, old['cmd7'], gate)
        old_meta = json.loads(str(old['meta_json'].item()))
        if old_meta.get('base') != 'outputs/ab5/A_ep0.json':
            raise ValueError('archived gate source identity mismatch')
        if not np.array_equal(starts, old['starts']):
            raise ValueError('archived chunk starts differ')
    lo, hi, mlo, mhi = robot_limits_safe()
    positive = {group: generate_probe(seed, gate) for group, seed in SEEDS.items()}
    deltas = {'A.npz': np.zeros((len(cmd7), 6), np.float32)}
    for group, da in positive.items():
        deltas[f'{group}_plus.npz'] = da
        deltas[f'{group}_minus.npz'] = -da
    audit = {'scope': 'offline software checks; not a safety proof or device receipt evidence',
             'sequences': {}, 'groups': {}, 'planned_counts_semantics':
             'float32 replay sum, converted to Python float, round(x*57324.840764)'}
    for group, da in positive.items():
        ok, stats, messages = checks(cmd7, da, gate, lo, hi, mlo, mhi,
                                     seam_frames=set(int(x) for x in starts))
        if not ok:
            raise ValueError(f'{group}: make_probe_3c checks failed: {messages}')
        audit['groups'][group] = {'checks': stats, 'messages': messages}
    payloads = {}
    counts = {}
    for name, da in deltas.items():
        messages, degraded, blockers = safety_checks(
            cmd7, da, probe_kind='multi_joint_continuous')
        detail = check_replay_result(messages, degraded, blockers)
        counts[name] = planned_counts(cmd7, da)
        detail.update({'rms_mrad_all_six': float(np.sqrt(np.mean(da.astype(float) ** 2)) * 1000),
                       'rms_mrad_per_joint': (np.sqrt(np.mean(da.astype(float) ** 2, axis=0)) * 1000).tolist(),
                       'peak_mrad': float(np.max(np.abs(da))) * 1000,
                       'j4_zero': bool(np.all(da[:, 3] == 0)),
                       'gate_zero': bool(np.all(da[gate] == 0))})
        audit['sequences'][name] = detail
        group = name[0]
        metadata = {'pilot_id': PILOT_ID, 'group': group, 'seed': SEEDS.get(group),
                    'nominal_mrad_before_truncation_gate_j4': PROBE_MRAD,
                    'no_post_gate_rescaling': True,
                    'planned_only_not_actual_device_receipt': True}
        payloads[name] = npz_bytes(cmd7=cmd7, da=da, gate=gate, starts=starts,
                                  planned_counts=counts[name],
                                  meta_json=np.array(canonical_json(metadata).decode()))
    for group in SEEDS:
        even = counts[f'{group}_plus.npz'] + counts[f'{group}_minus.npz'] - 2 * counts['A.npz']
        audit['groups'][group]['quantization_even'] = {
            'max_counts': int(np.abs(even).max()),
            'rms_mrad': float(np.sqrt(np.mean(even.astype(float) ** 2)) / JOINT_FACTOR * 1000),
            'meaning': 'input-layer asymmetry only; not an IMU nonlinearity estimate'}
        payloads[f'{group}_sent_counts.npz'] = npz_bytes(
            c0=counts['A.npz'], cp=counts[f'{group}_plus.npz'],
            cm=counts[f'{group}_minus.npz'], e_cmd_counts=even)
    payloads['calibration_reference.npz'], calibration = calibration_reference(cmd7, counts['A.npz'])
    payloads['audit.json'] = canonical_json(audit)
    code_paths = [Path(__file__), HERE / 'make_probe_3c.py', HERE / 'replay_probe.py',
                  HERE / 'freeze_pi_g2wall.py', HERE / 'analyze_probe_3a.py',
                  ROOT / 'lerobot/common/policies/act/b_correction_deploy.py',
                  ROOT / 'lerobot/common/robot_devices/motors/piper.py']
    manifest = {
        'schema': 'pilot-response14-v1', 'pilot_id': PILOT_ID, 'budget': BUDGET,
        'nominal_frames': len(cmd7), 'fps': FPS, 'runs': build_runs(),
        'files': {name: sha(data) for name, data in sorted(payloads.items())},
        'source': {str(p.relative_to(ROOT)): file_sha(p) for p in (source_path, archived_path)},
        'code_sha256': {str(p.relative_to(ROOT)): file_sha(p) for p in code_paths},
        'seeds': SEEDS, 'probe_kind': 'multi_joint_continuous',
        'amplitude': {'generator_mrad': PROBE_MRAD, 'generator_length': PROBE_N,
                      'normalization': 'P1: RMS over all 2000x6 generator entries before clipping',
                      'afterward': 'truncate, fixed archived gate, j4=0; no renormalization',
                      'peak_limit_mrad': 10.0},
        'j4': 'incremental j4 direction deliberately excluded; not a claim of no j4 response',
        'gate': {'source': 'outputs/probe_3c/probe_3c_p.npz:gate',
                 'frames': int(gate.sum()), 'array_sha256': sha(gate.tobytes())},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'scipy': scipy.__version__, 'device': 'cpu'},
        'calibration': calibration,
        'preflight_complete': True,
        'motion_release': False,
        'remaining_requirement': 'separate pilot-scoped engineering authorization at execution; '
                                 'known rate-limit blocker retained in audit.json',
    }
    payloads['manifest.json'] = canonical_json(manifest)
    return payloads, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, default=ROOT / 'outputs/pilot_response14')
    args = parser.parse_args()
    try:
        payloads, manifest = make_artifacts()
        freeze_files(args.outdir, payloads)
    except Exception as exc:
        print(f'OFFLINE PREFLIGHT FAILED: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
    print(f'Frozen/verified {len(payloads)} files: {args.outdir}')
    print(f'{manifest["nominal_frames"]} frames; 14 attempts / 2400 s budget; no hardware accessed')
    print(f'manifest sha256: {sha(payloads["manifest.json"])}')
    audit = json.loads(payloads['audit.json'])
    for group in SEEDS:
        row = audit['sequences'][f'{group}_plus.npz']
        print(f'{group}: RMS(all six)={row["rms_mrad_all_six"]:.6f} mrad; '
              f'peak={row["peak_mrad"]:.6f} mrad; gate/j4 zeros verified')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
