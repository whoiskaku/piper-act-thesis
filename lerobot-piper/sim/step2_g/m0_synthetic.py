"""Executable Step 2 M0 synthetic calibration contract.

This module deliberately uses a small, known second-order plant.  Its purpose
is to exercise data roles, identifiability checks, deterministic calibration,
artifact provenance, and role-separated holdout evaluation before any new
robot data are collected.  It is not a MuJoCo implementation, not the final G
model, and not evidence about real Piper dynamics.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


M0_LABEL = "M0a role-separated synthetic contract; not real G and not a model result"
HOLDOUT_SEAL = "role separation only; not sealed, secret, access-controlled, or confirmatory"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_arrays(arrays: dict[str, np.ndarray]) -> str:
    """Hash named arrays without relying on archive metadata or filesystem order."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(_canonical_json(list(array.shape)))
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class PlantParameters:
    """Per-joint parameters of qdd = gain*u - damping*qd - stiffness*q."""

    damping: tuple[float, ...]
    stiffness: tuple[float, ...]
    gain: tuple[float, ...]

    def validate(self, n_joints: int) -> None:
        for name in ("damping", "stiffness", "gain"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (n_joints,):
                raise ValueError(f"{name} must have exactly {n_joints} values")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class ParameterBounds:
    """Explicit physical search bounds, shared by all joints in M0."""

    damping: tuple[float, float] = (0.05, 4.0)
    stiffness: tuple[float, float] = (0.10, 8.0)
    gain: tuple[float, float] = (0.30, 2.0)

    def validate(self) -> None:
        for name in ("damping", "stiffness", "gain"):
            lower, upper = getattr(self, name)
            if not (np.isfinite(lower) and np.isfinite(upper) and 0.0 < lower < upper):
                raise ValueError(f"invalid positive bounds for {name}: {(lower, upper)}")


@dataclass(frozen=True)
class CalibrationSchema:
    """M0 schema and the one required identifiability gauge.

    A free command scale and the plant gain occur only as a product.  They
    cannot both be estimated.  M0 therefore fixes command_scale_gauge to 1.0
    and estimates gain; any other value is rejected instead of silently
    changing the model gauge.
    """

    n_joints: int = 3
    command_scale_gauge: float = 1.0
    bounds: ParameterBounds = ParameterBounds()
    max_design_condition: float = 1.0e8

    def validate(self) -> None:
        if self.n_joints < 1:
            raise ValueError("n_joints must be positive")
        if self.command_scale_gauge != 1.0:
            raise ValueError("command_scale_gauge must be fixed to 1.0 for gain identifiability")
        if not np.isfinite(self.max_design_condition) or self.max_design_condition <= 1.0:
            raise ValueError("max_design_condition must be finite and greater than 1")
        self.bounds.validate()


@dataclass(frozen=True)
class M0Config:
    """Fully specified deterministic synthetic study configuration."""

    schema: CalibrationSchema = CalibrationSchema()
    truth: PlantParameters = PlantParameters(
        damping=(0.55, 0.90, 1.25),
        stiffness=(1.40, 2.10, 3.00),
        gain=(0.80, 1.15, 1.45),
    )
    dt: float = 0.01
    steps_per_episode: int = 500
    calibration_episodes: int = 6
    holdout_episodes: int = 4
    calibration_seed: int = 2026082401
    holdout_seed: int = 2026082402

    def validate(self) -> None:
        self.schema.validate()
        self.truth.validate(self.schema.n_joints)
        if not (0.0 < self.dt <= 0.05):
            raise ValueError("dt must be in (0, 0.05]")
        if self.steps_per_episode < 20:
            raise ValueError("steps_per_episode must be at least 20")
        if self.calibration_episodes < 1 or self.holdout_episodes < 1:
            raise ValueError("both data roles require at least one episode")
        for name in ("damping", "stiffness", "gain"):
            values = np.asarray(getattr(self.truth, name))
            lower, upper = getattr(self.schema.bounds, name)
            if np.any(values < lower) or np.any(values > upper):
                raise ValueError(f"truth.{name} is outside the declared calibration bounds")


@dataclass(frozen=True)
class SyntheticSplit:
    """Observed synthetic trajectories with an enforced data-role label."""

    role: Literal["calibration", "holdout"]
    q: np.ndarray
    qdot: np.ndarray
    command: np.ndarray
    acceleration: np.ndarray
    dt: float
    seed: int

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "acceleration": self.acceleration,
            "command": self.command,
            "q": self.q,
            "qdot": self.qdot,
        }

    @property
    def sha256(self) -> str:
        metadata = _canonical_json({"dt": self.dt, "role": self.role, "seed": self.seed})
        return _sha256_bytes(metadata + _hash_arrays(self.arrays()).encode("ascii"))


@dataclass(frozen=True)
class SyntheticStudy:
    calibration: SyntheticSplit
    _holdout: SyntheticSplit

    @property
    def holdout_descriptor(self) -> dict[str, Any]:
        holdout = self._holdout
        return {
            "role": "role_separated_holdout",
            "episodes": int(holdout.q.shape[0]),
            "steps_per_episode": int(holdout.q.shape[1]),
            "sha256": holdout.sha256,
            "seal": HOLDOUT_SEAL,
        }


@dataclass(frozen=True)
class CalibrationArtifact:
    label: str
    model_equation: str
    parameters: PlantParameters
    schema: CalibrationSchema
    calibration_data_sha256: str
    calibration_seed: int
    design_rank_by_joint: tuple[int, ...]
    design_condition_by_joint: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _commands(rng: np.random.Generator, episodes: int, steps: int, n_joints: int, dt: float) -> np.ndarray:
    time = np.arange(steps, dtype=float) * dt
    commands = np.empty((episodes, steps, n_joints), dtype=float)
    for episode in range(episodes):
        for joint in range(n_joints):
            f1 = rng.uniform(0.15, 1.20)
            f2 = rng.uniform(1.30, 3.50)
            phase1, phase2 = rng.uniform(-np.pi, np.pi, size=2)
            signal = 0.65 * np.sin(2.0 * np.pi * f1 * time + phase1)
            signal += 0.25 * np.sin(2.0 * np.pi * f2 * time + phase2)
            signal += rng.normal(0.0, 0.08, size=steps)
            commands[episode, :, joint] = signal
    return commands


def _generate_split(
    role: Literal["calibration", "holdout"],
    seed: int,
    episodes: int,
    config: M0Config,
) -> SyntheticSplit:
    rng = np.random.default_rng(seed)
    n_joints = config.schema.n_joints
    steps = config.steps_per_episode
    command = _commands(rng, episodes, steps, n_joints, config.dt)
    q = np.empty_like(command)
    qdot = np.empty_like(command)
    acceleration = np.empty_like(command)
    damping = np.asarray(config.truth.damping)
    stiffness = np.asarray(config.truth.stiffness)
    gain = np.asarray(config.truth.gain)

    for episode in range(episodes):
        q_now = rng.uniform(-0.25, 0.25, size=n_joints)
        qdot_now = rng.uniform(-0.10, 0.10, size=n_joints)
        for step in range(steps):
            q[episode, step] = q_now
            qdot[episode, step] = qdot_now
            qdd = gain * command[episode, step] - damping * qdot_now - stiffness * q_now
            acceleration[episode, step] = qdd
            q_now = q_now + config.dt * qdot_now
            qdot_now = qdot_now + config.dt * qdd

    for array in (q, qdot, command, acceleration):
        array.setflags(write=False)
    return SyntheticSplit(role, q, qdot, command, acceleration, config.dt, seed)


def build_synthetic_study(config: M0Config) -> SyntheticStudy:
    """Generate independent deterministic calibration and sealed-holdout roles."""
    config.validate()
    calibration = _generate_split(
        "calibration", config.calibration_seed, config.calibration_episodes, config
    )
    holdout = _generate_split("holdout", config.holdout_seed, config.holdout_episodes, config)
    return SyntheticStudy(calibration=calibration, _holdout=holdout)


def calibrate(split: SyntheticSplit, schema: CalibrationSchema) -> CalibrationArtifact:
    """Fit calibration data only; a holdout-labelled split is rejected."""
    schema.validate()
    if split.role != "calibration":
        raise ValueError("calibrate accepts only role='calibration'; holdout fitting is forbidden")
    expected = (split.q.shape[0], split.q.shape[1], schema.n_joints)
    for name in ("q", "qdot", "command", "acceleration"):
        if getattr(split, name).shape != expected:
            raise ValueError(f"{name} shape does not match the calibration schema")

    fitted: dict[str, list[float]] = {"damping": [], "stiffness": [], "gain": []}
    ranks: list[int] = []
    conditions: list[float] = []
    for joint in range(schema.n_joints):
        design = np.column_stack(
            [
                -split.qdot[:, :, joint].reshape(-1),
                -split.q[:, :, joint].reshape(-1),
                split.command[:, :, joint].reshape(-1),
            ]
        )
        target = split.acceleration[:, :, joint].reshape(-1)
        rank = int(np.linalg.matrix_rank(design))
        condition = float(np.linalg.cond(design))
        if rank != 3 or not np.isfinite(condition) or condition > schema.max_design_condition:
            raise ValueError(
                f"joint {joint} calibration design is not identifiable: rank={rank}, condition={condition}"
            )
        coefficient, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        for name, value in zip(("damping", "stiffness", "gain"), coefficient, strict=True):
            lower, upper = getattr(schema.bounds, name)
            if value < lower - 1.0e-10 or value > upper + 1.0e-10:
                raise ValueError(f"fitted {name}[{joint}]={value} is outside declared bounds")
            fitted[name].append(float(np.clip(value, lower, upper)))
        ranks.append(rank)
        conditions.append(condition)

    return CalibrationArtifact(
        label=M0_LABEL,
        model_equation="qdd[j] = gain[j]*u[j] - damping[j]*qdot[j] - stiffness[j]*q[j]",
        parameters=PlantParameters(**{name: tuple(values) for name, values in fitted.items()}),
        schema=schema,
        calibration_data_sha256=split.sha256,
        calibration_seed=split.seed,
        design_rank_by_joint=tuple(ranks),
        design_condition_by_joint=tuple(conditions),
    )


def _simulate(split: SyntheticSplit, parameters: PlantParameters) -> tuple[np.ndarray, np.ndarray]:
    n_joints = split.q.shape[-1]
    parameters.validate(n_joints)
    damping = np.asarray(parameters.damping)
    stiffness = np.asarray(parameters.stiffness)
    gain = np.asarray(parameters.gain)
    q_pred = np.empty_like(split.q)
    qdot_pred = np.empty_like(split.qdot)
    for episode in range(split.q.shape[0]):
        q_now = split.q[episode, 0].copy()
        qdot_now = split.qdot[episode, 0].copy()
        for step in range(split.q.shape[1]):
            q_pred[episode, step] = q_now
            qdot_pred[episode, step] = qdot_now
            qdd = gain * split.command[episode, step] - damping * qdot_now - stiffness * q_now
            q_now = q_now + split.dt * qdot_now
            qdot_now = qdot_now + split.dt * qdd
    return q_pred, qdot_pred


def evaluate_role_separated_holdout(
    study: SyntheticStudy, artifact: CalibrationArtifact
) -> dict[str, Any]:
    """Evaluate the role-separated holdout after fitting; this is not a sealed test."""
    if artifact.calibration_data_sha256 != study.calibration.sha256:
        raise ValueError("artifact was not calibrated from this study's calibration split")
    holdout = study._holdout
    if holdout.role != "holdout":
        raise ValueError("holdout evaluation requires role='holdout'")
    q_pred, qdot_pred = _simulate(holdout, artifact.parameters)
    return {
        "label": M0_LABEL,
        "data_role": "role_separated_holdout_evaluation_only",
        "holdout_data_sha256": holdout.sha256,
        "holdout_seed": holdout.seed,
        "q_rmse": float(np.sqrt(np.mean(np.square(q_pred - holdout.q)))),
        "qdot_rmse": float(np.sqrt(np.mean(np.square(qdot_pred - holdout.qdot)))),
        "episodes": int(holdout.q.shape[0]),
        "steps_per_episode": int(holdout.q.shape[1]),
    }


def _write_json(path: Path, value: Any) -> str:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    return _sha256_bytes(payload.encode("utf-8"))


def run_m0(output_dir: Path, config: M0Config | None = None) -> dict[str, Path]:
    """Run M0 offline and write calibration, evaluation, and provenance artifacts."""
    config = config or M0Config()
    config.validate()
    targets = tuple(
        output_dir / name
        for name in ("calibration_artifact.json", "holdout_evaluation.json", "manifest.json")
    )
    existing = [path for path in targets if path.exists()]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"refusing to overwrite immutable M0 artifacts: {names}")
    output_dir.mkdir(parents=True, exist_ok=True)
    study = build_synthetic_study(config)

    # The calibration API receives only the calibration role.  Holdout data are
    # first consumed by the evaluator after the fitted artifact exists.
    artifact = calibrate(study.calibration, config.schema)
    artifact_path = output_dir / "calibration_artifact.json"
    artifact_sha256 = _write_json(artifact_path, artifact.to_dict())
    evaluation = evaluate_role_separated_holdout(study, artifact)
    evaluation_path = output_dir / "holdout_evaluation.json"
    evaluation_sha256 = _write_json(evaluation_path, evaluation)

    config_dict = asdict(config)
    implementation_path = Path(__file__).resolve()
    manifest = {
        "label": M0_LABEL,
        "contract_version": 1,
        "config": config_dict,
        "config_sha256": _sha256_bytes(_canonical_json(config_dict)),
        "seeds": {
            "calibration": config.calibration_seed,
            "role_separated_holdout": config.holdout_seed,
        },
        "data_roles": {
            "calibration": {
                "allowed_use": "parameter fitting and identifiability diagnostics",
                "sha256": study.calibration.sha256,
            },
            "role_separated_holdout": {
                "allowed_use": "post-fit evaluation only",
                "sha256": study.holdout_descriptor["sha256"],
                "seal": HOLDOUT_SEAL,
            },
        },
        "artifacts": {
            artifact_path.name: artifact_sha256,
            evaluation_path.name: evaluation_sha256,
        },
        "implementation": {
            "path": str(implementation_path.relative_to(implementation_path.parents[2])),
            "sha256": _sha256_bytes(implementation_path.read_bytes()),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "artifact": artifact_path,
        "evaluation": evaluation_path,
        "manifest": manifest_path,
    }
