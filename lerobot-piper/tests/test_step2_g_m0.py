"""Tests for the Step 2 M0a synthetic contract, not tests of real G quality."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from sim.step2_g.m0_synthetic import (
    CalibrationSchema,
    M0Config,
    ParameterBounds,
    SyntheticSplit,
    build_synthetic_study,
    calibrate,
    evaluate_role_separated_holdout,
    run_m0,
)


def test_parameter_recovery_and_role_separated_holdout() -> None:
    config = M0Config()
    study = build_synthetic_study(config)
    artifact = calibrate(study.calibration, config.schema)

    np.testing.assert_allclose(artifact.parameters.damping, config.truth.damping, atol=1e-10)
    np.testing.assert_allclose(artifact.parameters.stiffness, config.truth.stiffness, atol=1e-10)
    np.testing.assert_allclose(artifact.parameters.gain, config.truth.gain, atol=1e-10)
    evaluation = evaluate_role_separated_holdout(study, artifact)
    assert evaluation["data_role"] == "role_separated_holdout_evaluation_only"
    assert evaluation["q_rmse"] < 1e-10
    assert evaluation["qdot_rmse"] < 1e-10


def test_reproducible_artifacts_and_manifest(tmp_path) -> None:
    first = run_m0(tmp_path / "first")
    second = run_m0(tmp_path / "second")
    for role in ("artifact", "evaluation", "manifest"):
        assert first[role].read_bytes() == second[role].read_bytes()

    manifest = json.loads(first["manifest"].read_text(encoding="utf-8"))
    assert manifest["label"].startswith("M0a role-separated synthetic contract")
    assert manifest["data_roles"]["calibration"]["allowed_use"].startswith("parameter fitting")
    holdout = manifest["data_roles"]["role_separated_holdout"]
    assert holdout["allowed_use"] == "post-fit evaluation only"
    assert set(manifest["artifacts"]) == {"calibration_artifact.json", "holdout_evaluation.json"}
    for filename, expected_hash in manifest["artifacts"].items():
        payload = (first["manifest"].parent / filename).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected_hash
    assert "not sealed" in holdout["seal"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_m0(tmp_path / "first")


def test_holdout_seed_cannot_change_fit_and_holdout_role_is_rejected() -> None:
    config_a = M0Config(holdout_seed=101)
    config_b = replace(config_a, holdout_seed=202)
    study_a = build_synthetic_study(config_a)
    study_b = build_synthetic_study(config_b)
    artifact_a = calibrate(study_a.calibration, config_a.schema)
    artifact_b = calibrate(study_b.calibration, config_b.schema)

    assert artifact_a.to_dict() == artifact_b.to_dict()
    assert study_a.holdout_descriptor["sha256"] != study_b.holdout_descriptor["sha256"]
    with pytest.raises(ValueError, match="holdout fitting is forbidden"):
        calibrate(study_a._holdout, config_a.schema)

    wrong_artifact = replace(artifact_a, calibration_data_sha256="0" * 64)
    with pytest.raises(ValueError, match="not calibrated from this study"):
        evaluate_role_separated_holdout(study_a, wrong_artifact)


def test_invalid_schema_gauge_and_unidentifiable_design_are_rejected() -> None:
    with pytest.raises(ValueError, match="gauge must be fixed"):
        CalibrationSchema(command_scale_gauge=0.5).validate()
    with pytest.raises(ValueError, match="n_joints must be positive"):
        CalibrationSchema(n_joints=0).validate()
    with pytest.raises(ValueError, match="invalid positive bounds"):
        CalibrationSchema(bounds=ParameterBounds(gain=(2.0, 0.3))).validate()

    schema = CalibrationSchema(n_joints=1)
    zero = np.zeros((1, 30, 1), dtype=float)
    unidentifiable = SyntheticSplit(
        role="calibration",
        q=zero,
        qdot=zero,
        command=zero,
        acceleration=zero,
        dt=0.01,
        seed=1,
    )
    with pytest.raises(ValueError, match="not identifiable"):
        calibrate(unidentifiable, schema)
