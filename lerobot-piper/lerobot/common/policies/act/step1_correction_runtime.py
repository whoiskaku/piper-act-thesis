"""Pure NumPy runtime for the frozen Step-1 linear correction candidate.

This module deliberately has no robot, torch, or MuJoCo dependency.  It is the
single runtime definition shared by the offline solver and the eventual robot
wrapper:

* B: ``delta_b = clip(theta @ command_features, +/-2 mrad)``;
* C: ``delta_c = clip(z * delta_b, +/-3 mrad)``;
* C updates the scalar ``z`` with the causal, normalized LMS rule frozen in
  ``reports/PREREG_STEP1.md`` section 6.

The wrapper integration is intentionally kept outside this file.  A caller
loads an artifact, calls :meth:`Step1CorrectionRuntime.set_chunk` whenever ACT
produces a new raw chunk, then calls :meth:`Step1CorrectionRuntime.step` once
per actually executed frame.
"""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ARTIFACT_VERSION = "step1-linear-v1"
FPS = 20.0
EXPECTED_HP_ALPHA = 1.0 - math.exp(-2.0 * math.pi * 0.5 / FPS)

_EXPECTED_MANIFEST: dict[str, Any] = {
    "version": ARTIFACT_VERSION,
    "M": 15,
    "L": 3,
    "mu": 0.005,
    "hp_alpha": EXPECTED_HP_ALPHA,
    "b_limit": 0.002,
    "c_limit": 0.003,
    "grip_full": 0.035,
    "contact_threshold_ratio": 0.02,
    "contact_half_window": 10,
    "gate_threshold": 0.08,
    "feature_dim": 576,
}


class Step1ArtifactError(ValueError):
    """The frozen Step-1 artifact is absent, malformed, or incompatible."""


def _readonly_float32(value: np.ndarray) -> np.ndarray:
    out = np.asarray(value, dtype=np.float32).copy()
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class Step1Artifact:
    """Validated arrays and manifest stored in a ``step1-linear-v1`` NPZ."""

    theta: np.ndarray
    H: np.ndarray
    R: np.ndarray
    b: np.ndarray
    epsilon_lms: float
    manifest: Mapping[str, Any]
    path: Path | None = None


def _scalar_text(value: np.ndarray, name: str) -> str:
    arr = np.asarray(value)
    if arr.shape != ():
        raise Step1ArtifactError(f"{name} must be a scalar, got shape {arr.shape}")
    item = arr.item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    if isinstance(item, str):
        return item
    raise Step1ArtifactError(f"{name} must contain a JSON string, got {type(item).__name__}")


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    missing = sorted(set(_EXPECTED_MANIFEST) - set(manifest))
    if missing:
        raise Step1ArtifactError(f"manifest_json missing fields: {missing}")

    for key, expected in _EXPECTED_MANIFEST.items():
        actual = manifest[key]
        if isinstance(expected, float):
            try:
                actual_float = float(actual)
            except (TypeError, ValueError) as exc:
                raise Step1ArtifactError(f"manifest {key!r} is not numeric: {actual!r}") from exc
            # JSON writers may round hp_alpha for readability.  The tolerance is
            # still tight enough to reject a different filter or safety limit.
            atol = 1e-6 if key == "hp_alpha" else 1e-12
            if not math.isclose(actual_float, expected, rel_tol=0.0, abs_tol=atol):
                raise Step1ArtifactError(
                    f"manifest {key!r}={actual!r}, expected frozen value {expected!r}"
                )
        elif actual != expected:
            raise Step1ArtifactError(
                f"manifest {key!r}={actual!r}, expected frozen value {expected!r}"
            )


def load_step1_artifact(path: str | Path) -> Step1Artifact:
    """Load and strictly validate the frozen single-file NPZ artifact.

    Required NPZ members are ``theta``, ``H``, ``R``, ``b``,
    ``epsilon_lms``, and the scalar JSON string ``manifest_json``.
    Additional members are permitted for audit metadata.
    """

    artifact_path = Path(path)
    try:
        with np.load(artifact_path, allow_pickle=False) as data:
            required = {"theta", "H", "R", "b", "epsilon_lms", "manifest_json"}
            missing = sorted(required - set(data.files))
            if missing:
                raise Step1ArtifactError(f"artifact missing arrays: {missing}")

            theta_raw = np.asarray(data["theta"])
            h_raw = np.asarray(data["H"])
            r_raw = np.asarray(data["R"])
            b_raw = np.asarray(data["b"])
            epsilon_raw = np.asarray(data["epsilon_lms"])
            manifest_text = _scalar_text(data["manifest_json"], "manifest_json")
    except Step1ArtifactError:
        raise
    except (OSError, ValueError) as exc:
        raise Step1ArtifactError(f"cannot load Step-1 artifact {artifact_path}: {exc}") from exc

    try:
        manifest = json.loads(manifest_text)
    except json.JSONDecodeError as exc:
        raise Step1ArtifactError(f"manifest_json is invalid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise Step1ArtifactError("manifest_json must decode to a JSON object")
    _validate_manifest(manifest)

    if theta_raw.dtype != np.float32:
        raise Step1ArtifactError(f"theta dtype must be float32, got {theta_raw.dtype}")
    if h_raw.dtype != np.float32:
        raise Step1ArtifactError(f"H dtype must be float32, got {h_raw.dtype}")
    if theta_raw.shape != (6, 576):
        raise Step1ArtifactError(f"theta shape must be (6, 576), got {theta_raw.shape}")
    if h_raw.shape != (16, 3, 6):
        raise Step1ArtifactError(f"H shape must be (16, 3, 6), got {h_raw.shape}")
    if r_raw.shape != (3, 3):
        raise Step1ArtifactError(f"R shape must be (3, 3), got {r_raw.shape}")
    if b_raw.shape != (3,):
        raise Step1ArtifactError(f"b shape must be (3,), got {b_raw.shape}")
    if epsilon_raw.shape != ():
        raise Step1ArtifactError(
            f"epsilon_lms must be a scalar, got shape {epsilon_raw.shape}"
        )

    arrays = (theta_raw, h_raw, r_raw, b_raw, epsilon_raw)
    if not all(np.isfinite(x).all() for x in arrays):
        raise Step1ArtifactError("artifact contains NaN or infinity")
    epsilon_lms = float(epsilon_raw)
    if epsilon_lms <= 0.0:
        raise Step1ArtifactError(f"epsilon_lms must be > 0, got {epsilon_lms}")
    if "epsilon_lms" in manifest and not math.isclose(
        float(manifest["epsilon_lms"]), epsilon_lms, rel_tol=1e-6, abs_tol=0.0
    ):
        raise Step1ArtifactError("manifest epsilon_lms does not match NPZ epsilon_lms")

    # Copy out of the closed NpzFile and make accidental runtime mutation fail.
    return Step1Artifact(
        theta=_readonly_float32(theta_raw),
        H=_readonly_float32(h_raw),
        R=_readonly_float32(r_raw),
        b=_readonly_float32(b_raw),
        epsilon_lms=epsilon_lms,
        manifest=dict(manifest),
        path=artifact_path.resolve(),
    )


def calibrate_ideal_imu(ideal_raw: np.ndarray, R: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Map renderer-frame ideal IMU into the real IMU frame.

    This is byte-for-byte the convention used by ``FeatureAssembler``:
    accelerometer samples receive ``R`` and ``b``; gyro samples receive only
    ``R``.  Inputs and output have shape ``(T, 6)``.
    """

    ideal = np.asarray(ideal_raw, dtype=np.float32)
    rotation = np.asarray(R, dtype=np.float32)
    bias = np.asarray(b, dtype=np.float32)
    if ideal.ndim != 2 or ideal.shape[1] != 6:
        raise ValueError(f"ideal_raw must have shape (T, 6), got {ideal.shape}")
    if rotation.shape != (3, 3) or bias.shape != (3,):
        raise ValueError("R and b must have shapes (3, 3) and (3,)")
    if not np.isfinite(ideal).all():
        raise ValueError("ideal_raw contains NaN or infinity")

    out = np.empty_like(ideal, dtype=np.float32)
    out[:, :3] = ideal[:, :3] @ rotation.T + bias
    out[:, 3:] = ideal[:, 3:] @ rotation.T
    return out


def build_command_features(actions: np.ndarray, ideal: np.ndarray, M: int = 15) -> np.ndarray:
    """Construct the frozen command feature matrix, including end clamping.

    ``actions`` are raw ACT commands (only the first six joints are used), and
    ``ideal`` is already calibrated IMU in the real sensor frame.  For each
    chunk position ``t``, the returned row concatenates ``x[t+k]`` for
    ``k=0..M``, where ``x[j]`` is::

        [ideal[j], ideal[j+4], ideal[j+8],
         action[j], action[j+4], action[j+8]]

    All indices are clamped to the final frame of this chunk.  With ``M=15``
    the result has shape ``(T, 576)``.
    """

    action_array = np.asarray(actions, dtype=np.float32)
    ideal_array = np.asarray(ideal, dtype=np.float32)
    if action_array.ndim != 2 or action_array.shape[1] < 6:
        raise ValueError(f"actions must have shape (T, >=6), got {action_array.shape}")
    if ideal_array.shape != (len(action_array), 6):
        raise ValueError(
            f"ideal must have shape ({len(action_array)}, 6), got {ideal_array.shape}"
        )
    if len(action_array) == 0:
        raise ValueError("a command chunk must contain at least one frame")
    if M < 0:
        raise ValueError(f"M must be non-negative, got {M}")
    if not np.isfinite(action_array[:, :6]).all() or not np.isfinite(ideal_array).all():
        raise ValueError("actions/ideal contain NaN or infinity")

    n = len(action_array)
    arm = action_array[:, :6]
    rows = np.empty((n, (M + 1) * 36), dtype=np.float32)
    end = n - 1
    for t in range(n):
        blocks = []
        for k in range(M + 1):
            j = min(t + k, end)
            j4 = min(j + 4, end)
            j8 = min(j + 8, end)
            blocks.append(
                np.concatenate(
                    (ideal_array[j], ideal_array[j4], ideal_array[j8],
                     arm[j], arm[j4], arm[j8])
                )
            )
        rows[t] = np.concatenate(blocks)
    return rows


def command_contact_mask(
    grip: np.ndarray,
    *,
    grip_full: float = 0.035,
    threshold_ratio: float = 0.02,
    half_window: int = 10,
    previous_grip: float | None = None,
) -> np.ndarray:
    """Return frames within +/- ``half_window`` of a known grip-command jump."""

    values = np.asarray(grip, dtype=np.float32)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError(f"grip must be a non-empty 1-D array, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("grip contains NaN or infinity")
    if half_window < 0:
        raise ValueError("half_window must be non-negative")

    threshold = float(grip_full) * float(threshold_ratio)
    diffs = np.empty(len(values), dtype=np.float32)
    diffs[0] = 0.0 if previous_grip is None else abs(values[0] - float(previous_grip))
    if len(values) > 1:
        diffs[1:] = np.abs(np.diff(values))
    jumps = np.flatnonzero(diffs > threshold)
    mask = np.zeros(len(values), dtype=bool)
    for j in jumps:
        mask[max(0, j - half_window): min(len(mask), j + half_window + 1)] = True
    return mask


def causal_fir_at(H: np.ndarray, inputs: list[np.ndarray], index: int) -> np.ndarray:
    """Evaluate ``sum_tau H[tau] @ inputs[index-tau]`` at one causal index."""

    kernel = np.asarray(H, dtype=np.float32)
    if kernel.ndim != 3 or kernel.shape[2] != 6:
        raise ValueError(f"H must have shape (lags, outputs, 6), got {kernel.shape}")
    out = np.zeros(kernel.shape[1], dtype=np.float32)
    if index < 0:
        return out
    for tau in range(min(len(kernel), index + 1)):
        out += kernel[tau] @ np.asarray(inputs[index - tau], dtype=np.float32)
    return out


class CausalHighPass:
    """Frozen first-order causal DC remover ``x - EMA(x)``.

    Each episode starts the EMA at its first valid sample.  Consequently the
    first high-passed sample is exactly zero rather than an artificial reset
    transient from a zero-valued mean.
    """

    def __init__(self, alpha: float, channels: int = 3):
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)
        self.channels = int(channels)
        self.reset()

    def reset(self) -> None:
        self.mean = np.zeros(self.channels, dtype=np.float64)
        self._initialized = False

    def step(self, value: np.ndarray) -> np.ndarray:
        x = np.asarray(value, dtype=np.float64)
        if x.shape != (self.channels,) or not np.isfinite(x).all():
            raise ValueError(f"high-pass input must be finite shape ({self.channels},)")
        if not self._initialized:
            self.mean = x.copy()
            self._initialized = True
            return np.zeros(self.channels, dtype=np.float32)
        self.mean = (1.0 - self.alpha) * self.mean + self.alpha * x
        return (x - self.mean).astype(np.float32)


@dataclass(frozen=True)
class NLMSUpdate:
    z_before: float
    z_after: float
    e_filtered: np.ndarray
    p_filtered: np.ndarray
    gradient: float | None
    updated: bool


class ScalarNLMS:
    """Unit-testable scalar normalized-LMS state used by arm C."""

    def __init__(
        self,
        *,
        mu: float,
        epsilon: float,
        hp_alpha: float,
        z0: float = 1.0,
        z_min: float = 0.0,
        z_max: float = 1.5,
    ):
        if mu <= 0.0 or epsilon <= 0.0:
            raise ValueError("mu and epsilon must be positive")
        if not z_min <= z0 <= z_max:
            raise ValueError("z0 must lie inside [z_min, z_max]")
        self.mu = float(mu)
        self.epsilon = float(epsilon)
        self.z0 = float(z0)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self._hp_e = CausalHighPass(hp_alpha, channels=3)
        self._hp_p = CausalHighPass(hp_alpha, channels=3)
        self.reset()

    def reset(self) -> None:
        self.z = self.z0
        self._hp_e.reset()
        self._hp_p.reset()

    def step(self, e_raw: np.ndarray, p_raw: np.ndarray, *, enabled: bool) -> NLMSUpdate:
        """Filter one aligned pair and, if enabled, update ``z`` for the next frame."""

        e_filtered = self._hp_e.step(e_raw)
        p_filtered = self._hp_p.step(p_raw)
        before = self.z
        gradient: float | None = None
        if enabled:
            numerator = float(np.dot(e_filtered, p_filtered))
            denominator = float(np.dot(p_filtered, p_filtered)) + self.epsilon
            gradient = numerator / denominator
            self.z = float(np.clip(before - self.mu * gradient, self.z_min, self.z_max))
        return NLMSUpdate(
            z_before=before,
            z_after=self.z,
            e_filtered=e_filtered,
            p_filtered=p_filtered,
            gradient=gradient,
            updated=bool(enabled),
        )


@dataclass(frozen=True)
class Step1Frame:
    """One runtime decision plus the causal state update made after that decision."""

    frame_index: int
    chunk_index: int
    delta: np.ndarray
    delta_b: np.ndarray
    raw_delta_b: np.ndarray
    z_used: float
    z_next: float
    contact_gated: bool
    b_saturated: bool
    c_saturated: bool
    aligned_index: int | None
    raw_gap: np.ndarray | None
    predicted_response: np.ndarray | None
    gate: float | None
    imu_valid: bool | None
    lms_update_mask: bool | None
    nlms: NLMSUpdate | None


@dataclass
class _ChunkPlan:
    ideal: np.ndarray
    raw_delta_b: np.ndarray
    delta_b: np.ndarray
    contact: np.ndarray
    grip: np.ndarray
    pos: int = 0


class Step1CorrectionRuntime:
    """Streaming B/C candidate with explicit, robot-independent state.

    Parameters
    ----------
    artifact:
        A validated :class:`Step1Artifact` or path to its NPZ.
    mode:
        ``"B"`` for fixed command correction, or ``"C"`` for the same B
        correction scaled by the real-time IMU NLMS state.
    """

    def __init__(self, artifact: Step1Artifact | str | Path, mode: str = "C"):
        self.artifact = (
            artifact if isinstance(artifact, Step1Artifact) else load_step1_artifact(artifact)
        )
        normalized_mode = str(mode).upper()
        if normalized_mode not in {"B", "C"}:
            raise ValueError("mode must be 'B' or 'C'")
        self.mode = normalized_mode
        m = self.artifact.manifest
        self.M = int(m["M"])
        self.L = int(m["L"])
        self.mu = float(m["mu"])
        self.hp_alpha = float(m["hp_alpha"])
        self.b_limit = float(m["b_limit"])
        self.c_limit = float(m["c_limit"])
        self.grip_full = float(m["grip_full"])
        self.contact_threshold_ratio = float(m["contact_threshold_ratio"])
        self.contact_half_window = int(m["contact_half_window"])
        self.gate_threshold = float(m["gate_threshold"])
        self.adapter = ScalarNLMS(
            mu=self.mu,
            epsilon=self.artifact.epsilon_lms,
            hp_alpha=self.hp_alpha,
            z0=1.0,
            z_min=0.0,
            z_max=1.5,
        )
        self.reset()

    @property
    def z(self) -> float:
        return self.adapter.z

    @property
    def chunk_remaining(self) -> int:
        return 0 if self._chunk is None else len(self._chunk.delta_b) - self._chunk.pos

    def reset(self) -> None:
        """Reset all episode state, including the required ``z0 = 1``."""

        self.adapter.reset()
        self._chunk: _ChunkPlan | None = None
        self._b_history: list[np.ndarray] = []
        self._ideal_gyro_history: list[np.ndarray] = []
        self._contact_history: list[bool] = []
        self._grip_jump_history: list[bool] = []
        self._gyro_norms: deque[float] = deque(maxlen=5)
        self._last_grip: float | None = None
        self._frame = 0
        self._stats: dict[str, float | int] = {
            "frames": 0,
            "b_sat": 0,
            "c_sat": 0,
            "contact_gated": 0,
            "aligned_pairs": 0,
            "nlms_updates": 0,
            "nlms_skip_gate": 0,
            "nlms_skip_contact": 0,
            "z_sum": 0.0,
            "z_min": 1.0,
            "z_max": 1.0,
            "gradient_abs_sum": 0.0,
            "stale": 0,
            "nan": 0,
        }

    def set_chunk(
        self,
        actions: np.ndarray,
        ideal_imu: np.ndarray,
        *,
        ideal_is_calibrated: bool = False,
    ) -> None:
        """Prepare one raw ACT chunk and its renderer ideal IMU.

        A pending chunk must be fully consumed first.  ``actions`` must include
        the seventh (gripper) command because the frozen contact gate uses it.
        Renderer-frame ``ideal_imu`` is calibrated with artifact ``R,b`` by
        default; pass ``ideal_is_calibrated=True`` only when it is already in
        the real IMU frame.
        """

        if self.chunk_remaining:
            raise RuntimeError(
                f"cannot replace a chunk with {self.chunk_remaining} unexecuted frames"
            )
        action_array = np.asarray(actions, dtype=np.float32)
        if action_array.ndim != 2 or action_array.shape[1] < 7 or len(action_array) == 0:
            raise ValueError(f"actions must have shape (T, >=7), got {action_array.shape}")
        if not np.isfinite(action_array[:, :7]).all():
            raise ValueError("actions contain NaN or infinity")
        ideal = (
            np.asarray(ideal_imu, dtype=np.float32).copy()
            if ideal_is_calibrated
            else calibrate_ideal_imu(ideal_imu, self.artifact.R, self.artifact.b)
        )
        if ideal.shape != (len(action_array), 6) or not np.isfinite(ideal).all():
            raise ValueError(f"ideal_imu must have finite shape ({len(action_array)}, 6)")

        features = build_command_features(action_array, ideal, M=self.M)
        raw_b = (features @ self.artifact.theta.T).astype(np.float32)
        if not np.isfinite(raw_b).all():
            self._stats["nan"] = int(self._stats["nan"]) + int(
                np.count_nonzero(~np.isfinite(raw_b).all(axis=1))
            )
            raise FloatingPointError("theta produced NaN or infinity")
        delta_b = np.clip(raw_b, -self.b_limit, self.b_limit).astype(np.float32)
        contact = command_contact_mask(
            action_array[:, 6],
            grip_full=self.grip_full,
            threshold_ratio=self.contact_threshold_ratio,
            half_window=self.contact_half_window,
            previous_grip=self._last_grip,
        )
        delta_b[contact] = 0.0
        self._chunk = _ChunkPlan(
            ideal=ideal,
            raw_delta_b=raw_b,
            delta_b=delta_b,
            contact=contact,
            grip=action_array[:, 6].copy(),
        )

    @staticmethod
    def _gyro3(real_imu_or_gyro: np.ndarray) -> np.ndarray:
        value = np.asarray(real_imu_or_gyro, dtype=np.float32).reshape(-1)
        if value.shape == (6,):
            value = value[3:]
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError("real IMU input must be a finite gyro (3,) or IMU (6,)")
        return value

    def step(self, real_imu_or_gyro: np.ndarray | None = None) -> Step1Frame:
        """Execute one planned frame and update C's ``z`` for the next frame.

        Arm B does not read IMU.  Arm C normally receives the current
        wall-clock real gyro (or six-axis IMU).  A missing/malformed C sample
        safely skips LMS but still consumes this command frame and returns the
        current ``z * B``; the wrapper records the resulting ``stale`` and the
        experimental protocol reruns that triplet.  The correction returned for
        frame ``t`` uses ``z`` *before* the observation at ``t`` is used; the
        updated value is therefore first applied at frame ``t+1``.
        """

        if self._chunk is None or self._chunk.pos >= len(self._chunk.delta_b):
            raise RuntimeError("no prepared chunk frame; call set_chunk first")
        real_gyro = None
        if self.mode == "C":
            if real_imu_or_gyro is not None:
                try:
                    real_gyro = self._gyro3(real_imu_or_gyro)
                except ValueError:
                    real_gyro = None

        chunk = self._chunk
        k = chunk.pos
        t = self._frame
        raw_b = chunk.raw_delta_b[k].copy()
        delta_b = chunk.delta_b[k].copy()
        contact = bool(chunk.contact[k])
        b_saturated = bool(np.any(np.abs(raw_b) > self.b_limit))

        # Freeze the action decision before processing this wall-clock IMU:
        # the eventual NLMS update is for the next frame only.
        z_used = self.adapter.z if self.mode == "C" else 1.0
        raw_c = z_used * delta_b
        c_saturated = bool(np.any(np.abs(raw_c) > self.c_limit + 1e-12))
        delta = (
            np.clip(raw_c, -self.c_limit, self.c_limit).astype(np.float32)
            if self.mode == "C"
            else delta_b.copy()
        )

        self._b_history.append(delta_b.copy())  # deliberately unscaled by z
        self._ideal_gyro_history.append(chunk.ideal[k, 3:].copy())
        self._contact_history.append(contact)
        current_grip = float(chunk.grip[k])
        grip_jump = bool(
            self._last_grip is not None
            and abs(current_grip - self._last_grip)
            > self.grip_full * self.contact_threshold_ratio
        )
        self._grip_jump_history.append(grip_jump)

        aligned_index: int | None = None
        raw_gap: np.ndarray | None = None
        predicted: np.ndarray | None = None
        gate: float | None = None
        lms_update_mask: bool | None = None
        nlms: NLMSUpdate | None = None
        if self.mode == "C":
            if real_gyro is None:
                self._stats["stale"] = int(self._stats["stale"]) + 1
            i = t - self.L
            if i >= 0 and real_gyro is not None:
                # Match the existing aligned causal gate: its first element is
                # real wall-clock sample L (for gap index zero), not the three
                # pre-alignment warm-up samples.
                self._gyro_norms.append(float(np.linalg.norm(real_gyro)))
                gate = float(np.mean(self._gyro_norms))
                aligned_index = i
                raw_gap = (real_gyro - self._ideal_gyro_history[i]).astype(np.float32)
                predicted = causal_fir_at(self.artifact.H, self._b_history, i)
                # LMS's frozen mask is causal and distinct from the output
                # contact mask: no grip-command jump in indices i-9..i.
                recent_start = max(0, i - 9)
                no_recent_grip_jump = not any(
                    self._grip_jump_history[recent_start: i + 1]
                )
                moving = gate >= self.gate_threshold
                lms_update_mask = bool(moving and no_recent_grip_jump)
                nlms = self.adapter.step(
                    raw_gap,
                    predicted,
                    enabled=lms_update_mask,
                )
                self._stats["aligned_pairs"] = int(self._stats["aligned_pairs"]) + 1
                if nlms.updated:
                    self._stats["nlms_updates"] = int(self._stats["nlms_updates"]) + 1
                    self._stats["gradient_abs_sum"] = float(
                        self._stats["gradient_abs_sum"]
                    ) + abs(float(nlms.gradient))
                else:
                    if not moving:
                        self._stats["nlms_skip_gate"] = int(
                            self._stats["nlms_skip_gate"]
                        ) + 1
                    if not no_recent_grip_jump:
                        self._stats["nlms_skip_contact"] = int(
                            self._stats["nlms_skip_contact"]
                        ) + 1

        chunk.pos += 1
        self._last_grip = current_grip
        self._frame += 1
        self._stats["frames"] = int(self._stats["frames"]) + 1
        self._stats["b_sat"] = int(self._stats["b_sat"]) + int(b_saturated)
        self._stats["c_sat"] = int(self._stats["c_sat"]) + int(c_saturated)
        self._stats["contact_gated"] = int(self._stats["contact_gated"]) + int(contact)
        self._stats["z_sum"] = float(self._stats["z_sum"]) + z_used
        self._stats["z_min"] = min(float(self._stats["z_min"]), self.adapter.z)
        self._stats["z_max"] = max(float(self._stats["z_max"]), self.adapter.z)

        return Step1Frame(
            frame_index=t,
            chunk_index=k,
            delta=delta,
            delta_b=delta_b,
            raw_delta_b=raw_b,
            z_used=z_used,
            z_next=self.adapter.z if self.mode == "C" else 1.0,
            contact_gated=contact,
            b_saturated=b_saturated,
            c_saturated=c_saturated,
            aligned_index=aligned_index,
            raw_gap=raw_gap,
            predicted_response=predicted,
            gate=gate,
            imu_valid=(real_gyro is not None) if self.mode == "C" else None,
            lms_update_mask=lms_update_mask,
            nlms=nlms,
        )

    def summary(self) -> dict[str, Any]:
        """Return JSON-serializable episode statistics for audit logging."""

        frames = max(int(self._stats["frames"]), 1)
        aligned = max(int(self._stats["aligned_pairs"]), 1)
        sat_count = int(self._stats["b_sat"] if self.mode == "B" else self._stats["c_sat"])
        updates = int(self._stats["nlms_updates"])
        return {
            "artifact_version": self.artifact.manifest["version"],
            "artifact_path": str(self.artifact.path) if self.artifact.path else None,
            "mode": self.mode,
            "frames": int(self._stats["frames"]),
            "sat_pct": 100.0 * sat_count / frames,
            "b_preclip_sat_pct": 100.0 * int(self._stats["b_sat"]) / frames,
            "c_preclip_sat_pct": 100.0 * int(self._stats["c_sat"]) / frames,
            "contact_gated_pct": 100.0 * int(self._stats["contact_gated"]) / frames,
            "stale": int(self._stats["stale"]),
            "nan": int(self._stats["nan"]),
            "aligned_pairs": int(self._stats["aligned_pairs"]),
            "nlms_updates": updates,
            "nlms_update_pct": 100.0 * updates / aligned,
            "nlms_skip_gate": int(self._stats["nlms_skip_gate"]),
            "nlms_skip_contact": int(self._stats["nlms_skip_contact"]),
            "nlms_gradient_abs_mean": (
                float(self._stats["gradient_abs_sum"]) / updates if updates else 0.0
            ),
            "z_final": self.adapter.z if self.mode == "C" else 1.0,
            "z_mean_used": float(self._stats["z_sum"]) / frames,
            "z_min": float(self._stats["z_min"]),
            "z_max": float(self._stats["z_max"]),
            "chunk_remaining": self.chunk_remaining,
        }


__all__ = [
    "ARTIFACT_VERSION",
    "EXPECTED_HP_ALPHA",
    "CausalHighPass",
    "NLMSUpdate",
    "ScalarNLMS",
    "Step1Artifact",
    "Step1ArtifactError",
    "Step1CorrectionRuntime",
    "Step1Frame",
    "build_command_features",
    "calibrate_ideal_imu",
    "causal_fir_at",
    "command_contact_mask",
    "load_step1_artifact",
]
