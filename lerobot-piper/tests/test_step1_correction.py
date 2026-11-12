"""Pure-NumPy contract tests for the frozen Step-1 correction runtime.

The module is loaded by file path so this small test does not import LeRobot's
top-level package (and therefore does not require draccus, MuJoCo, or torch).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_PATH = _ROOT / "lerobot/common/policies/act/step1_correction_runtime.py"
_SPEC = importlib.util.spec_from_file_location("_step1_correction_runtime_test", _RUNTIME_PATH)
assert _SPEC is not None and _SPEC.loader is not None
runtime = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = runtime
_SPEC.loader.exec_module(runtime)


def _manifest() -> dict:
    return {
        "version": "step1-linear-v1",
        "M": 15,
        "L": 3,
        "mu": 0.005,
        "hp_alpha": runtime.EXPECTED_HP_ALPHA,
        "b_limit": 0.002,
        "c_limit": 0.003,
        "grip_full": 0.035,
        "contact_threshold_ratio": 0.02,
        "contact_half_window": 10,
        "gate_threshold": 0.08,
        "feature_dim": 576,
    }


def _artifact(
    *,
    theta: np.ndarray | None = None,
    H: np.ndarray | None = None,
    epsilon_lms: float = 1e-12,
) -> runtime.Step1Artifact:
    if theta is None:
        theta = np.zeros((6, 576), dtype=np.float32)
    if H is None:
        H = np.zeros((16, 3, 6), dtype=np.float32)
    return runtime.Step1Artifact(
        theta=np.asarray(theta, dtype=np.float32),
        H=np.asarray(H, dtype=np.float32),
        R=np.eye(3, dtype=np.float32),
        b=np.zeros(3, dtype=np.float32),
        epsilon_lms=float(epsilon_lms),
        manifest=_manifest(),
    )


def _commands_and_ideal(n: int, accel_x: float | np.ndarray = 0.001):
    commands = np.zeros((n, 7), dtype=np.float32)
    ideal = np.zeros((n, 6), dtype=np.float32)
    ideal[:, 0] = accel_x
    return commands, ideal


def test_artifact_loader_accepts_only_frozen_contract(tmp_path: Path):
    path = tmp_path / "candidate.npz"
    arrays = {
        "theta": np.zeros((6, 576), dtype=np.float32),
        "H": np.zeros((16, 3, 6), dtype=np.float32),
        "R": np.eye(3, dtype=np.float32),
        "b": np.zeros(3, dtype=np.float32),
        "epsilon_lms": np.asarray(1e-6, dtype=np.float32),
    }
    np.savez_compressed(path, **arrays, manifest_json=np.asarray(json.dumps(_manifest())))

    loaded = runtime.load_step1_artifact(path)
    assert loaded.theta.shape == (6, 576)
    assert loaded.H.shape == (16, 3, 6)
    assert not loaded.theta.flags.writeable
    with pytest.raises(ValueError):
        loaded.theta[0, 0] = 1.0

    bad_manifest = _manifest()
    bad_manifest["L"] = 4
    np.savez_compressed(path, **arrays, manifest_json=np.asarray(json.dumps(bad_manifest)))
    with pytest.raises(runtime.Step1ArtifactError, match="manifest 'L'"):
        runtime.load_step1_artifact(path)


def test_command_features_clamp_every_preview_to_chunk_end():
    actions = (100 + np.arange(12, dtype=np.float32)).reshape(2, 6)
    ideal = np.arange(12, dtype=np.float32).reshape(2, 6)
    got = runtime.build_command_features(actions, ideal, M=1)

    block_0 = np.concatenate((ideal[0], ideal[1], ideal[1], actions[0], actions[1], actions[1]))
    block_1 = np.concatenate((ideal[1], ideal[1], ideal[1], actions[1], actions[1], actions[1]))
    assert got.shape == (2, 72)
    np.testing.assert_array_equal(got[0], np.concatenate((block_0, block_1)))
    np.testing.assert_array_equal(got[1], np.concatenate((block_1, block_1)))


def test_contact_window_zeros_b_correction():
    theta = np.zeros((6, 576), dtype=np.float32)
    theta[0, 0] = 1.0
    correction = runtime.Step1CorrectionRuntime(_artifact(theta=theta), mode="B")
    commands, ideal = _commands_and_ideal(25)
    commands[12:, 6] = 0.01
    correction.set_chunk(commands, ideal, ideal_is_calibrated=True)

    frames = [correction.step() for _ in range(25)]
    expected_mask = np.zeros(25, dtype=bool)
    expected_mask[2:23] = True  # jump at 12, inclusive +/-10 frames
    np.testing.assert_array_equal([frame.contact_gated for frame in frames], expected_mask)
    np.testing.assert_allclose([frame.delta[0] for frame in frames][2:23], 0.0)
    np.testing.assert_allclose([frames[0].delta[0], frames[-1].delta[0]], 0.001)


def test_h_taps_zero_and_fifteen_obey_three_frame_alignment():
    theta = np.zeros((6, 576), dtype=np.float32)
    theta[0, 0] = 1.0
    H = np.zeros((16, 3, 6), dtype=np.float32)
    H[0, 0, 0] = 1.0
    H[15, 1, 0] = 2.0
    correction = runtime.Step1CorrectionRuntime(_artifact(theta=theta, H=H), mode="C")
    commands, ideal = _commands_and_ideal(19)
    correction.set_chunk(commands, ideal, ideal_is_calibrated=True)

    frames = [correction.step(np.array([1.0, 0.0, 0.0], dtype=np.float32)) for _ in range(19)]
    assert all(frame.aligned_index is None for frame in frames[:3])
    assert frames[3].aligned_index == 0
    np.testing.assert_allclose(frames[3].predicted_response, [0.001, 0.0, 0.0], atol=1e-8)
    assert frames[18].aligned_index == 15
    np.testing.assert_allclose(frames[18].predicted_response, [0.001, 0.002, 0.0], atol=1e-8)


def test_high_pass_reset_initializes_from_first_sample():
    hp = runtime.CausalHighPass(runtime.EXPECTED_HP_ALPHA, channels=3)
    np.testing.assert_array_equal(hp.step(np.full(3, 3.0)), np.zeros(3))
    assert np.linalg.norm(hp.step(np.full(3, 4.0))) > 0.0
    hp.reset()
    np.testing.assert_array_equal(hp.step(np.full(3, -7.0)), np.zeros(3))


def test_lms_update_changes_z_only_on_the_next_action_frame():
    theta = np.zeros((6, 576), dtype=np.float32)
    theta[0, 0] = 1.0
    H = np.zeros((16, 3, 6), dtype=np.float32)
    H[0, 0, 0] = 1.0
    correction = runtime.Step1CorrectionRuntime(_artifact(theta=theta, H=H), mode="C")
    commands, ideal = _commands_and_ideal(6, np.linspace(0.0005, 0.0015, 6, dtype=np.float32))
    correction.set_chunk(commands, ideal, ideal_is_calibrated=True)

    frames = []
    for t in range(6):
        gyro = np.array([1.0 if t <= 3 else 2.0, 0.0, 0.0], dtype=np.float32)
        frames.append(correction.step(gyro))

    updated = frames[4]
    following = frames[5]
    assert updated.z_used == pytest.approx(1.0)
    assert updated.z_next < updated.z_used
    np.testing.assert_allclose(updated.delta, updated.delta_b)
    assert following.z_used == pytest.approx(updated.z_next)
    np.testing.assert_allclose(following.delta, following.z_used * following.delta_b)


def test_b_and_c_apply_their_frozen_per_joint_caps():
    theta = np.zeros((6, 576), dtype=np.float32)
    theta[0, 0] = 10.0  # 0.001 feature -> 10 mrad before clipping
    artifact = _artifact(theta=theta)
    commands, ideal = _commands_and_ideal(1)

    arm_b = runtime.Step1CorrectionRuntime(artifact, mode="B")
    arm_b.set_chunk(commands, ideal, ideal_is_calibrated=True)
    frame_b = arm_b.step()
    assert frame_b.b_saturated
    assert frame_b.delta[0] == pytest.approx(0.002)

    arm_c = runtime.Step1CorrectionRuntime(artifact, mode="C")
    arm_c.adapter.z = 1.5
    arm_c.set_chunk(commands, ideal, ideal_is_calibrated=True)
    frame_c = arm_c.step(None)
    assert np.max(np.abs(frame_c.delta_b)) <= 0.002 + 1e-9
    assert np.max(np.abs(frame_c.delta)) <= 0.003 + 1e-9
    assert frame_c.delta[0] == pytest.approx(0.003)


def test_missing_imu_consumes_c_frame_but_never_updates_z():
    theta = np.zeros((6, 576), dtype=np.float32)
    theta[0, 0] = 1.0
    commands, ideal = _commands_and_ideal(4)

    arm_c = runtime.Step1CorrectionRuntime(_artifact(theta=theta), mode="C")
    arm_c.set_chunk(commands, ideal, ideal_is_calibrated=True)
    frames = [arm_c.step(None) for _ in range(4)]
    assert all(frame.imu_valid is False and frame.nlms is None for frame in frames)
    assert all(frame.z_used == 1.0 and frame.z_next == 1.0 for frame in frames)
    assert arm_c.summary()["stale"] == 4

    arm_b = runtime.Step1CorrectionRuntime(_artifact(theta=theta), mode="B")
    arm_b.set_chunk(commands[:1], ideal[:1], ideal_is_calibrated=True)
    assert arm_b.step(None).imu_valid is None
    assert arm_b.summary()["stale"] == 0


def test_episode_reset_clears_chunk_history_filters_stats_and_z():
    theta = np.zeros((6, 576), dtype=np.float32)
    theta[0, 0] = 1.0
    correction = runtime.Step1CorrectionRuntime(_artifact(theta=theta), mode="C")
    commands, ideal = _commands_and_ideal(4)
    correction.set_chunk(commands, ideal, ideal_is_calibrated=True)
    correction.step(None)
    correction.adapter.z = 0.25

    correction.reset()
    summary = correction.summary()
    assert summary["frames"] == 0
    assert summary["chunk_remaining"] == 0
    assert summary["stale"] == 0
    assert summary["z_final"] == 1.0

    correction.set_chunk(commands, ideal, ideal_is_calibrated=True)
    first = correction.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert first.frame_index == 0
    assert first.aligned_index is None
    assert first.z_used == 1.0
