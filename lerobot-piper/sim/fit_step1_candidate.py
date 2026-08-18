#!/usr/bin/env python3
"""Fit the preregistered Step-1 linear candidate from P0/P1/P2/P3 logs.

This is the single offline entry point for ``reports/PREREG_STEP1.md`` sections
6 and 8.  The fit deliberately uses no learned simulator:

* P1's exogenous probes identify the causal raw-gyro response ``H[0:16]``.
* P0 ep0--2 and P1 fit the raw aligned-gap predictor ``W``.  P1 targets have
  ``H * probe`` removed first.
* A rolling, history-zero local inverse is collapsed into one fixed matrix
  ``theta``.  Deployment therefore needs only ``theta @ x_tilde``.
* P2/P3 beta tests and the P0 ep3--4 8% gate are the only places where the
  centred 41-tap 0.5--9 Hz FIR is used.

Expected input is the JSON written by ``BCorrectionDeployWrapper``::

    outputs/ab4/P0_ep0.json ... P0_ep4.json
    outputs/ab4/P1_ep0.json ... P1_ep7.json
    outputs/ab4/P2_ep0.json ... P2_ep5.json
    outputs/ab4/P3_ep0.json ... P3_ep3.json

Typical use::

    python sim/fit_step1_candidate.py \\
      --data-root outputs/ab4 \\
      --calib outputs/p2_labels/pp2a-deploy/manifest.json \\
      --output outputs/step1_candidate/step1_linear_v1.npz

The NPZ is written only when all frozen gates pass.  A sidecar JSON is always
written (including on a gate failure), so a negative result remains auditable.
Run ``--self-test`` for a no-robot-data check of the pure-array core.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.signal import firwin


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# These are the deployment definitions.  Load the pure-NumPy source directly
# so ``--self-test`` does not execute lerobot.common.policies.__init__ (which
# pulls optional training dependencies such as draccus into an offline fit).
_RUNTIME_PATH = ROOT / "lerobot/common/policies/act/step1_correction_runtime.py"
_RUNTIME_SPEC = importlib.util.spec_from_file_location("_step1_runtime_shared", _RUNTIME_PATH)
if _RUNTIME_SPEC is None or _RUNTIME_SPEC.loader is None:
    raise ImportError(f"cannot load shared Step-1 runtime helpers from {_RUNTIME_PATH}")
_RUNTIME = importlib.util.module_from_spec(_RUNTIME_SPEC)
sys.modules[_RUNTIME_SPEC.name] = _RUNTIME
_RUNTIME_SPEC.loader.exec_module(_RUNTIME)
CausalHighPass = _RUNTIME.CausalHighPass
build_command_features = _RUNTIME.build_command_features
calibrate_ideal_imu = _RUNTIME.calibrate_ideal_imu
command_contact_mask = _RUNTIME.command_contact_mask

FS = 20.0
ALIGN_L = 3
M = 15
N_LAGS = M + 1
BASE_FEATURE_DIM = 36
FEATURE_DIM = (M + 1) * BASE_FEATURE_DIM
RIDGE_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
EXPECTED_COUNTS = {"P0": 5, "P1": 8, "P2": 6, "P3": 4}
EXPECTED_SEEDS = {
    "P1": (1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007),
    "P2": (2000, 2000, 2001, 2001, 2002, 2002),
    "P3": (1000, 1001, 1002, 1003),
}
EXPECTED_MRAD = {"P1": 1.5, "P2": 1.5, "P3": 3.0}
B_LIMIT = 0.002
C_LIMIT = 0.003
GRIP_FULL = 0.035
CONTACT_THRESHOLD_RATIO = 0.02
CONTACT_HALF_WINDOW = 10
GATE_THRESHOLD = 0.08
MU = 0.005
HP_ALPHA = 1.0 - math.exp(-2.0 * math.pi * 0.5 / FS)


@dataclass
class Episode:
    """Array representation used by every fitting routine.

    ``gap`` is aligned raw gyro gap: ``real[t + ALIGN_L] - ideal[t]``.
    ``da`` and features retain command-time index ``t``.
    """

    tier: str
    ep: int
    path: str
    gap: np.ndarray                 # (T, 3), raw aligned gyro gap
    da: np.ndarray                  # (T, 6), actually applied probe
    x: np.ndarray                   # (T, 36)
    x_tilde: np.ndarray             # (T, 576)
    contact_mask: np.ndarray        # (T,), candidate must be zero here
    seed: int | None = None
    probe_hash: str | None = None
    summary: dict | None = None


def _as_f64(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float64)


def causal_design(action: np.ndarray, n_lags: int = N_LAGS) -> np.ndarray:
    """Causal FIR design matrix, with zero history before episode start."""
    action = _as_f64(action)
    if action.ndim != 2 or action.shape[1] != 6:
        raise ValueError(f"action must have shape (T,6), got {action.shape}")
    out = np.zeros((len(action), n_lags * 6), dtype=np.float64)
    for lag in range(n_lags):
        if lag == 0:
            out[:, :6] = action
        elif lag < len(action):
            out[lag:, lag * 6:(lag + 1) * 6] = action[:-lag]
    return out


def causal_conv(H: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Apply ``H[lag, gyro_axis, joint]`` with causal, zero-padded history."""
    H, action = _as_f64(H), _as_f64(action)
    if H.ndim != 3 or H.shape[1:] != (3, 6):
        raise ValueError(f"H must have shape (lags,3,6), got {H.shape}")
    if action.ndim != 2 or action.shape[1] != 6:
        raise ValueError(f"action must have shape (T,6), got {action.shape}")
    out = np.zeros((len(action), 3), dtype=np.float64)
    for lag in range(len(H)):
        if lag == 0:
            out += action @ H[lag].T
        elif lag < len(action):
            out[lag:] += action[:-lag] @ H[lag].T
    return out


def bandpass41(x: np.ndarray) -> np.ndarray:
    """Frozen offline metric filter: centred 41-tap FIR, 0.5--9 Hz at 20 Hz."""
    x = _as_f64(x)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"bandpass input must have shape (T,3), got {x.shape}")
    kernel = firwin(41, [0.5, 9.0], fs=FS, pass_zero=False)
    return np.stack([np.convolve(x[:, j], kernel, mode="same") for j in range(3)], axis=1)


def ridge_scale(X: np.ndarray) -> float:
    """The preregistered ``trace(X.T X) / dim`` scale, protected only at zero."""
    X = _as_f64(X)
    return max(float(np.square(X).sum()) / max(X.shape[1], 1), np.finfo(float).tiny)


def _ridge_solve(X: np.ndarray, Y: np.ndarray, multiplier: float) -> tuple[np.ndarray, float]:
    X, Y = _as_f64(X), _as_f64(Y)
    scale = ridge_scale(X)
    lam = float(multiplier) * scale
    gram = X.T @ X
    coef = np.linalg.solve(gram + lam * np.eye(gram.shape[0]), X.T @ Y)
    return coef, lam


def fit_H(p1: Sequence[Episode], ridge_multiplier: float) -> tuple[np.ndarray, float]:
    """Fit raw aligned gyro response from P1 only.

    An episode-specific constant is removed from both sides.  This is a nuisance
    intercept for raw gyro bias, not a bandpass: all dynamic samples remain raw.
    The exogenous probe is what identifies H in the presence of task baseline.
    """
    if not p1:
        raise ValueError("fit_H needs at least one P1 episode")
    xs, ys = [], []
    for e in p1:
        X = causal_design(e.da)
        xs.append(X - X.mean(axis=0, keepdims=True))
        ys.append(_as_f64(e.gap) - np.mean(e.gap, axis=0, keepdims=True))
    coef, lam = _ridge_solve(np.concatenate(xs), np.concatenate(ys), ridge_multiplier)
    H = coef.reshape(N_LAGS, 6, 3).transpose(0, 2, 1)
    return H, lam


def baseline_target(ep: Episode, H: np.ndarray) -> np.ndarray:
    """Raw no-probe gap; P1's measured probe contribution is removed."""
    if ep.tier == "P1":
        return _as_f64(ep.gap) - causal_conv(H, ep.da)
    if ep.tier == "P0":
        return _as_f64(ep.gap)
    raise ValueError(f"baseline_target only accepts P0/P1, got {ep.tier}")


def fit_W(train: Sequence[Episode], H: np.ndarray,
          ridge_multiplier: float) -> tuple[np.ndarray, float]:
    """Fit ``e_hat[t] = W @ x[t]`` in raw aligned-gap space."""
    if not train:
        raise ValueError("fit_W needs training episodes")
    X = np.concatenate([_as_f64(e.x) for e in train])
    Y = np.concatenate([baseline_target(e, H) for e in train])
    coef, lam = _ridge_solve(X, Y, ridge_multiplier)       # (36,3)
    return coef.T, lam                                     # (3,36)


def solve_theta(H: np.ndarray, W: np.ndarray,
                ridge_multiplier: float) -> tuple[np.ndarray, float]:
    """Collapse the preregistered history-zero local inverse into ``theta``.

    For one action d at time t, the predicted horizon response is H[k] @ d.
    Thus K = -(Hstack.T Hstack + lambda I)^-1 Hstack.T, and each future
    36-vector is mapped by ``K_k @ W``.  No online optimisation remains.
    """
    H, W = _as_f64(H), _as_f64(W)
    if H.shape != (N_LAGS, 3, 6) or W.shape != (3, BASE_FEATURE_DIM):
        raise ValueError(f"bad H/W shapes: {H.shape}, {W.shape}")
    hs = H.reshape(N_LAGS * 3, 6)
    gram = hs.T @ hs
    scale = max(float(np.trace(gram)) / 6.0, np.finfo(float).tiny)
    lam = float(ridge_multiplier) * scale
    K = -np.linalg.solve(gram + lam * np.eye(6), hs.T)     # (6,48)
    theta = np.concatenate([K[:, 3*k:3*(k+1)] @ W for k in range(N_LAGS)], axis=1)
    if theta.shape != (6, FEATURE_DIM):
        raise AssertionError(f"internal theta shape error: {theta.shape}")
    return theta, lam


def candidate_action(ep: Episode, theta: np.ndarray) -> np.ndarray:
    """Frozen B-arm action including its preregistered clip and contact zeroing."""
    theta = _as_f64(theta)
    d = np.clip(_as_f64(ep.x_tilde) @ theta.T, -B_LIMIT, B_LIMIT)
    d[np.asarray(ep.contact_mask, bool)] = 0.0
    return d


def residual_score(episodes: Sequence[Episode], H: np.ndarray,
                   theta: np.ndarray) -> tuple[float, float]:
    """Return raw residual and baseline energies for held-out CV episodes."""
    residual, baseline = 0.0, 0.0
    for ep in episodes:
        base = baseline_target(ep, H)
        res = base + causal_conv(H, candidate_action(ep, theta))
        residual += float(np.square(res).sum())
        baseline += float(np.square(base).sum())
    return residual, baseline


def grouped_folds(episodes: Sequence[Episode], n_folds: int = 5) -> list[list[int]]:
    """Deterministic episode-level, tier-stratified folds (never frame splits)."""
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    by_tier: dict[str, list[int]] = {}
    for i, ep in enumerate(episodes):
        by_tier.setdefault(ep.tier, []).append(i)
    offset = 0
    for tier in sorted(by_tier):
        for j, idx in enumerate(by_tier[tier]):
            folds[(j + offset) % n_folds].append(idx)
        offset = (offset + len(by_tier[tier])) % n_folds
    if any(not fold for fold in folds):
        raise ValueError(f"cannot make {n_folds} nonempty grouped folds from {len(episodes)} episodes")
    return folds


def cross_validate(fit_episodes: Sequence[Episode],
                   grid: Sequence[float] = RIDGE_GRID) -> tuple[tuple[float, float, float], list[dict]]:
    """Joint 5-fold CV; every fold refits the complete H -> W -> theta chain."""
    if sum(e.tier == "P1" for e in fit_episodes) < 5:
        raise ValueError("five-fold full-pipeline CV needs at least five P1 episodes")
    folds = grouped_folds(fit_episodes, 5)
    totals = {(h, w, k): [0.0, 0.0] for h in grid for w in grid for k in grid}
    fold_report: list[dict] = []
    all_idx = set(range(len(fit_episodes)))

    for fold_i, val_idx in enumerate(folds):
        train_idx = sorted(all_idx - set(val_idx))
        train = [fit_episodes[i] for i in train_idx]
        val = [fit_episodes[i] for i in val_idx]
        p1_train = [e for e in train if e.tier == "P1"]
        fold_report.append({
            "fold": fold_i,
            "train": [f"{e.tier}:{e.ep}" for e in train],
            "validation": [f"{e.tier}:{e.ep}" for e in val],
        })
        for h_mult in grid:
            H, _ = fit_H(p1_train, h_mult)
            for w_mult in grid:
                W, _ = fit_W(train, H, w_mult)
                for k_mult in grid:
                    theta, _ = solve_theta(H, W, k_mult)
                    num, den = residual_score(val, H, theta)
                    acc = totals[(h_mult, w_mult, k_mult)]
                    acc[0] += num
                    acc[1] += den

    rows = []
    for key, (num, den) in totals.items():
        rows.append({"h_multiplier": key[0], "w_multiplier": key[1],
                     "k_multiplier": key[2], "residual_ratio": num / max(den, 1e-30),
                     "predicted_reduction": 1.0 - num / max(den, 1e-30)})
    rows.sort(key=lambda r: (r["residual_ratio"], r["h_multiplier"],
                             r["w_multiplier"], r["k_multiplier"]))
    best = rows[0]
    chosen = (best["h_multiplier"], best["w_multiplier"], best["k_multiplier"])
    for fr in fold_report:
        fr["chosen_by_global_cv"] = dict(best)
    return chosen, fold_report


def beta_ols(episodes: Sequence[Episode], H: np.ndarray) -> float:
    """Frozen through-origin beta using offline-bandpassed measured/predicted response."""
    numerator, denominator = 0.0, 0.0
    for ep in episodes:
        measured = bandpass41(ep.gap)
        predicted = bandpass41(causal_conv(H, ep.da))
        numerator += float((measured * predicted).sum())
        denominator += float(np.square(predicted).sum())
    if denominator <= 1e-30:
        raise ValueError("beta undefined: predicted P2/P3 probe response has zero energy")
    return numerator / denominator


def evaluate_p2_gate(p2: Sequence[Episode], H: np.ndarray) -> dict:
    by_seed: dict[int, list[Episode]] = {}
    for ep in p2:
        if ep.seed is None:
            raise ValueError(f"P2 ep{ep.ep} has no logged probe_seed")
        by_seed.setdefault(ep.seed, []).append(ep)
    beta_seed = {str(seed): beta_ols(by_seed[seed], H) for seed in sorted(by_seed)}
    values = np.asarray(list(beta_seed.values()), dtype=float)
    median = float(np.median(values))
    p2_pass = bool(median >= 0.5 and np.all(values > 0.0) and len(values) == 3)
    return {"p2_beta_by_seed": beta_seed, "p2_beta_median": median, "p2_pass": p2_pass}


def evaluate_beta_gates(p2: Sequence[Episode], p3: Sequence[Episode],
                        H: np.ndarray) -> dict:
    result = evaluate_p2_gate(p2, H)
    median = result["p2_beta_median"]
    beta_p3 = beta_ols(p3, H)
    ratio = beta_p3 / median if abs(median) > 1e-30 else float("nan")
    p3_pass = bool(np.isfinite(ratio) and 0.7 <= ratio <= 1.3)
    result.update({"p3_beta": beta_p3, "p3_over_p2_median": ratio, "p3_pass": p3_pass})
    return result


def evaluate_p0_gate(p0_holdout: Sequence[Episode], H: np.ndarray,
                     theta: np.ndarray) -> dict:
    reductions = []
    for ep in p0_holdout:
        base = bandpass41(ep.gap)
        response = bandpass41(causal_conv(H, candidate_action(ep, theta)))
        den = float(np.square(base).sum())
        reduction = 1.0 - float(np.square(base + response).sum()) / max(den, 1e-30)
        reductions.append(reduction)
    median = float(np.median(reductions))
    return {"p0_holdout_reduction": reductions, "p0_median_reduction": median,
            "p0_pass": bool(median >= 0.08)}


def lms_epsilon(fit_episodes: Sequence[Episode], H: np.ndarray,
                theta: np.ndarray) -> float:
    """One percent of the fitting-set median framewise ||highpass(p)||^2."""
    energy = []
    for ep in fit_episodes:
        p = causal_conv(H, candidate_action(ep, theta))
        hp = CausalHighPass(HP_ALPHA, channels=3)
        p_filtered = np.stack([hp.step(row) for row in p])
        energy.append(np.square(p_filtered.astype(np.float64)).sum(axis=1))
    median = float(np.median(np.concatenate(energy)))
    return max(0.01 * median, 1e-12)


def contact_mask(grip: np.ndarray) -> np.ndarray:
    """One-chunk compatibility wrapper around the runtime's strict-``>`` gate."""
    return command_contact_mask(
        np.asarray(grip, dtype=np.float32).reshape(-1),
        grip_full=GRIP_FULL,
        threshold_ratio=CONTACT_THRESHOLD_RATIO,
        half_window=CONTACT_HALF_WINDOW,
    )


def build_features(ideal: np.ndarray, action: np.ndarray,
                   chunk_caps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct features per chunk through the deployment's single source."""
    ideal = np.asarray(ideal, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)
    caps = np.asarray(chunk_caps, dtype=np.int64)
    n = len(ideal)
    if ideal.shape != (n, 6) or action.shape != (n, 6) or caps.shape != (n,):
        raise ValueError(f"bad feature inputs: ideal={ideal.shape}, action={action.shape}, caps={caps.shape}")
    parts = []
    start = 0
    while start < n:
        end = int(caps[start]) + 1
        if end <= start or end > n or not np.all(caps[start:end] == end - 1):
            raise ValueError("chunk_caps does not describe contiguous chunks")
        parts.append(build_command_features(action[start:end], ideal[start:end], M=M))
        start = end
    x_tilde = np.concatenate(parts, axis=0)
    return x_tilde[:, :BASE_FEATURE_DIM], x_tilde


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _episode_number(path: Path) -> int:
    match = re.search(r"_ep(\d+)\.json$", path.name)
    if not match:
        raise ValueError(f"cannot parse episode number from {path}")
    return int(match.group(1))


def discover_logs(data_root: Path, tiers: Sequence[str]) -> dict[str, list[Path]]:
    found = {}
    for tier in tiers:
        expected = EXPECTED_COUNTS[tier]
        paths = sorted(data_root.glob(f"{tier}_ep*.json"), key=_episode_number)
        if len(paths) != expected:
            raise FileNotFoundError(f"{tier}: expected exactly {expected} logs at "
                                    f"{data_root}/{tier}_ep*.json, found {len(paths)}")
        ids = [_episode_number(p) for p in paths]
        if ids != list(range(expected)):
            raise ValueError(f"{tier}: expected episode ids 0..{expected-1}, got {ids}")
        found[tier] = paths
    return found


def _render_logged_chunks(chunks: list[dict], renderer, n_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not chunks:
        raise ValueError("log lacks chunks[].raw; exact Step-1 features cannot be reconstructed")
    chunks = sorted(chunks, key=lambda c: int(c["start"]))
    total = max(int(c["start"]) + len(c["raw"]) for c in chunks)
    command = np.full((total, 7), np.nan, dtype=np.float64)
    ideal = np.full((total, 6), np.nan, dtype=np.float64)
    caps = np.full(total, -1, dtype=np.int64)
    previous = None
    previous_end = 0
    for ci, chunk in enumerate(chunks):
        start = int(chunk["start"])
        raw = np.asarray(chunk["raw"], dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 7:
            raise ValueError(f"chunk {ci} raw must be (T,7), got {raw.shape}")
        if start != previous_end:
            raise ValueError(f"chunk coverage is not contiguous: expected start {previous_end}, got {start}")
        rendered = np.asarray(renderer.render(raw, previous), dtype=np.float64)
        if rendered.shape != (len(raw), 6):
            raise ValueError(f"renderer returned {rendered.shape}, expected {(len(raw), 6)}")
        end = start + len(raw)
        command[start:end] = raw
        ideal[start:end] = rendered
        caps[start:end] = end - 1
        previous, previous_end = raw, end
    if n_frames > total:
        raise ValueError(f"log has {n_frames} frames but chunks cover only {total}")
    if not np.isfinite(command[:n_frames]).all() or not np.isfinite(ideal[:n_frames]).all():
        raise ValueError("non-finite or uncovered command/ideal rows")
    # Keep the complete logged chunks, including their not-yet-executed tail.
    # Runtime knew that tail when it made each decision, so truncating here to
    # n_frames would silently change the final chunk's preview features.
    return command, ideal, caps


def load_episode(path: Path, tier: str, R: np.ndarray, b: np.ndarray,
                 renderer, min_frames: int) -> Episode:
    payload = json.loads(path.read_text())
    frames = payload.get("frames", [])
    summary = payload.get("summary", {})
    if len(frames) < min_frames or int(summary.get("frames", 0)) < min_frames:
        raise ValueError(f"{path}: fewer than {min_frames} frames")
    if len(frames) != int(summary.get("frames", len(frames))):
        raise ValueError(f"{path}: len(frames)={len(frames)} != summary.frames={summary.get('frames')}")
    if any(f.get("imu") is None for f in frames):
        raise ValueError(f"{path}: missing IMU frame (stale/nan must be zero for fitting data)")
    real = np.asarray([f["imu"] for f in frames], dtype=np.float64)
    da = np.asarray([f["da"] for f in frames], dtype=np.float64)
    if real.shape != (len(frames), 6) or da.shape != (len(frames), 6):
        raise ValueError(f"{path}: bad IMU/da shapes {real.shape}/{da.shape}")
    if not np.isfinite(real).all() or not np.isfinite(da).all():
        raise ValueError(f"{path}: non-finite IMU/da")

    command, ideal_sim, caps = _render_logged_chunks(payload.get("chunks", []), renderer, len(frames))
    # Build features on the complete chunks *before* removing the final L
    # unalignable samples.  At those earlier decisions, the final chunk frames
    # were already known and runtime therefore previews them.
    ideal = calibrate_ideal_imu(ideal_sim.astype(np.float32),
                                np.asarray(R, np.float32), np.asarray(b, np.float32))
    x_full, x_tilde_full = build_features(ideal, command[:, :6], caps)
    mask_full = np.zeros(len(command), dtype=bool)
    previous_grip: float | None = None
    start = 0
    while start < len(command):
        end = int(caps[start]) + 1
        mask_full[start:end] = command_contact_mask(
            np.asarray(command[start:end, 6], dtype=np.float32),
            grip_full=GRIP_FULL,
            threshold_ratio=CONTACT_THRESHOLD_RATIO,
            half_window=CONTACT_HALF_WINDOW,
            previous_grip=previous_grip,
        )
        previous_grip = float(command[end - 1, 6])
        start = end

    n = min(len(frames) - ALIGN_L, len(command))
    if n <= M + 1:
        raise ValueError(f"{path}: too short after L={ALIGN_L} alignment")
    gap = real[ALIGN_L:ALIGN_L + n, 3:6] - ideal[:n, 3:6]
    seed = summary.get("probe_seed")
    return Episode(tier=tier, ep=_episode_number(path), path=str(path), gap=gap,
                   da=da[:n], x=x_full[:n], x_tilde=x_tilde_full[:n],
                   contact_mask=mask_full[:n],
                   seed=int(seed) if seed is not None else None,
                   probe_hash=summary.get("probe_hash"), summary=summary)


def validate_protocol(episodes: dict[str, list[Episode]]) -> None:
    for tier in ("P1", "P2", "P3"):
        if tier not in episodes:
            continue
        got_seeds = tuple(e.seed for e in episodes[tier])
        if got_seeds != EXPECTED_SEEDS[tier]:
            raise ValueError(f"{tier}: expected logged seeds {EXPECTED_SEEDS[tier]}, got {got_seeds}")
        for e in episodes[tier]:
            mrad = float((e.summary or {}).get("probe_mrad", float("nan")))
            if not math.isclose(mrad, EXPECTED_MRAD[tier], rel_tol=0.0, abs_tol=1e-8):
                raise ValueError(f"{tier} ep{e.ep}: expected {EXPECTED_MRAD[tier]} mrad, got {mrad}")
            if (e.summary or {}).get("mode") != "probe":
                raise ValueError(f"{tier} ep{e.ep}: summary.mode must be probe")
            if int((e.summary or {}).get("probe_overrun", 0)) != 0:
                raise ValueError(f"{tier} ep{e.ep}: probe_overrun != 0")
            if int((e.summary or {}).get("probe_wrapped", 0)) != 0:
                raise ValueError(f"{tier} ep{e.ep}: probe_wrapped != 0")
    for e in episodes.get("P0", []):
        if (e.summary or {}).get("mode") != "off":
            raise ValueError(f"P0 ep{e.ep}: summary.mode must be off")
        if np.max(np.abs(e.da)) > 1e-9:
            raise ValueError(f"P0 ep{e.ep}: logged da is not zero")
    if "P3" in episodes:
        p1_hash = {e.seed: e.probe_hash for e in episodes["P1"]}
        for e in episodes["P3"]:
            if not e.probe_hash or e.probe_hash != p1_hash.get(e.seed):
                raise ValueError(f"P3 ep{e.ep}: hash does not match P1 seed {e.seed}")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def fit_all(episodes: dict[str, list[Episode]]) -> tuple[np.ndarray, np.ndarray, float, dict]:
    fit_eps = episodes["P0"][:3] + episodes["P1"]
    chosen, folds = cross_validate(fit_eps)
    h_mult, w_mult, k_mult = chosen
    H, h_lambda = fit_H(episodes["P1"], h_mult)
    W, w_lambda = fit_W(fit_eps, H, w_mult)
    theta, k_lambda = solve_theta(H, W, k_mult)
    epsilon = lms_epsilon(fit_eps, H, theta)
    beta = evaluate_beta_gates(episodes["P2"], episodes["P3"], H)
    p0_gate = evaluate_p0_gate(episodes["P0"][3:5], H, theta)
    diagnostics = {
        "cv": {"folds": folds, "chosen_multipliers": {"H": h_mult, "W": w_mult, "K": k_mult},
               "actual_lambdas": {"H": h_lambda, "W": w_lambda, "K": k_lambda}},
        "beta_gates": beta,
        "p0_gate": p0_gate,
        "epsilon_lms": epsilon,
        "all_gates_pass": bool(beta["p2_pass"] and beta["p3_pass"] and p0_gate["p0_pass"]),
    }
    return theta, H, epsilon, diagnostics


def artifact_manifest(R: np.ndarray, b: np.ndarray, epsilon: float,
                      diagnostics: dict, logs: dict[str, list[Path]], calib_path: Path) -> dict:
    files = [{"tier": tier, "ep": _episode_number(path), "path": str(path),
              "sha256": _file_sha256(path)} for tier in sorted(logs) for path in logs[tier]]
    return {
        "version": "step1-linear-v1",
        "M": M,
        "L": ALIGN_L,
        "mu": MU,
        "hp_alpha": HP_ALPHA,
        "b_limit": B_LIMIT,
        "c_limit": C_LIMIT,
        "grip_full": GRIP_FULL,
        "contact_threshold_ratio": CONTACT_THRESHOLD_RATIO,
        "contact_half_window": CONTACT_HALF_WINDOW,
        "gate_threshold": GATE_THRESHOLD,
        "feature_dim": FEATURE_DIM,
        "h_shape": [N_LAGS, 3, 6],
        "theta_shape": [6, FEATURE_DIM],
        "epsilon_lms": epsilon,
        "fit_space": "raw_aligned_gyro_gap",
        "offline_metric": "41-tap FIR 0.5-9 Hz, centred",
        "H_fit_nuisance": "per-episode constant removed",
        "calibration": {"path": str(calib_path), "sha256": _file_sha256(calib_path),
                        "R": np.asarray(R).tolist(), "b": np.asarray(b).tolist()},
        "data": files,
        "diagnostics": diagnostics,
    }


def save_artifact(output: Path, theta: np.ndarray, H: np.ndarray, R: np.ndarray,
                  b: np.ndarray, epsilon: float, manifest: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_text = json.dumps(_jsonable(manifest), ensure_ascii=False, sort_keys=True)
    np.savez_compressed(
        output,
        theta=np.asarray(theta, dtype=np.float32),
        H=np.asarray(H, dtype=np.float32),
        R=np.asarray(R, dtype=np.float32),
        b=np.asarray(b, dtype=np.float32),
        epsilon_lms=np.asarray(epsilon, dtype=np.float32),
        manifest_json=np.asarray(manifest_text),
    )


def self_test() -> int:
    rng = np.random.default_rng(7)
    n = 900
    H_true = rng.normal(0.0, 0.08, (N_LAGS, 3, 6)) * np.exp(-np.arange(N_LAGS)[:, None, None] / 4)
    da = rng.normal(0.0, 1.0, (n, 6))
    # Core convolution/design must agree exactly.
    y = causal_conv(H_true, da)
    yd = causal_design(da) @ H_true.transpose(0, 2, 1).reshape(N_LAGS * 6, 3)
    assert np.allclose(y, yd, atol=1e-12)

    caps = np.full(n, n - 1, dtype=int)
    ideal = rng.normal(0.0, 0.1, (n, 6))
    action = rng.normal(0.0, 0.1, (n, 6))
    x, xt = build_features(ideal, action, caps)
    assert x.shape == (n, 36) and xt.shape == (n, 576)
    ep = Episode("P1", 0, "synthetic", y, da, x, xt, np.zeros(n, bool), seed=1)
    H_fit, _ = fit_H([ep], 1e-9)
    assert np.linalg.norm(H_fit - H_true) / np.linalg.norm(H_true) < 1e-5

    # With measured gap exactly equal to the predicted response, beta is one.
    beta = beta_ols([ep], H_true)
    assert abs(beta - 1.0) < 1e-10

    W = rng.normal(0.0, 1e-4, (3, BASE_FEATURE_DIM))
    theta, _ = solve_theta(H_true, W, 0.1)
    assert theta.shape == (6, FEATURE_DIM) and np.isfinite(theta).all()
    d = candidate_action(ep, theta)
    assert np.max(np.abs(d)) <= B_LIMIT + 1e-15

    mask = contact_mask(np.r_[np.zeros(100), np.full(100, GRIP_FULL)])
    assert mask[90:111].all() and not mask[89] and not mask[111]

    # A stopped episode may execute only part of its final 100-frame chunk.
    # Offline features must retain the entire chunk preview, exactly as runtime
    # did when the chunk was generated.
    class _IdentityRenderer:
        @staticmethod
        def render(raw, previous):
            return np.asarray(raw[:, :6], dtype=np.float32)

    raw = np.arange(10 * 7, dtype=np.float32).reshape(10, 7)
    cmd, ide, partial_caps = _render_logged_chunks(
        [{"start": 0, "raw": raw.tolist()}], _IdentityRenderer(), n_frames=8)
    assert len(cmd) == len(ide) == len(partial_caps) == 10
    assert np.all(partial_caps == 9)
    partial_x, _ = build_features(ide, cmd[:, :6], partial_caps)
    # At command t=4, the t+8 preview clamps to complete-chunk frame 9,
    # rather than the last executed frame 7.
    assert np.array_equal(partial_x[4, 30:36], cmd[9, :6].astype(np.float32))
    print("self-test: PASS (core arrays + partial-final-chunk preview)")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=ROOT / "outputs/ab4")
    parser.add_argument("--calib", type=Path,
                        default=ROOT / "outputs/p2_labels/pp2a-deploy/manifest.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/step1_candidate/step1_linear_v1.npz")
    parser.add_argument("--sidecar", type=Path, default=None,
                        help="default: OUTPUT with .json suffix; written even when a gate fails")
    parser.add_argument("--stage", choices=("p2", "full"), default="full",
                        help="p2: require only P0/P1/P2 and stop at the P2 beta gate; "
                             "full: require P3 and run all frozen gates")
    parser.add_argument("--min-frames", type=int, default=550)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    data_root = args.data_root if args.data_root.is_absolute() else ROOT / args.data_root
    calib_path = args.calib if args.calib.is_absolute() else ROOT / args.calib
    output = args.output if args.output.is_absolute() else ROOT / args.output
    default_sidecar = (output.with_name(output.stem + "_p2.json")
                       if args.stage == "p2" else output.with_suffix(".json"))
    sidecar = args.sidecar or default_sidecar
    if not sidecar.is_absolute():
        sidecar = ROOT / sidecar

    tiers = ("P0", "P1", "P2") if args.stage == "p2" else ("P0", "P1", "P2", "P3")
    logs = discover_logs(data_root, tiers)
    calibration = json.loads(calib_path.read_text())
    R = np.asarray(calibration["R"], dtype=np.float64)
    b = np.asarray(calibration.get("b", [0.0, 0.0, 0.0]), dtype=np.float64)
    if R.shape != (3, 3) or b.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(b).all():
        raise ValueError(f"bad calibration R/b shapes or values: {R.shape}/{b.shape}")

    # Import here so pure-array --self-test does not require MuJoCo.
    sys.path.insert(0, str(ROOT))
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    renderer = IdealChunkRenderer()
    episodes = {tier: [load_episode(p, tier, R, b, renderer, args.min_frames)
                       for p in paths] for tier, paths in logs.items()}
    validate_protocol(episodes)

    if args.stage == "p2":
        fit_eps = episodes["P0"][:3] + episodes["P1"]
        chosen, folds = cross_validate(fit_eps)
        h_mult, w_mult, k_mult = chosen
        H, h_lambda = fit_H(episodes["P1"], h_mult)
        p2_gate = evaluate_p2_gate(episodes["P2"], H)
        p2_report = {
            "version": "step1-linear-v1",
            "stage": "p2",
            "fit_space": "raw_aligned_gyro_gap",
            "cv": {"folds": folds,
                   "chosen_multipliers": {"H": h_mult, "W": w_mult, "K": k_mult},
                   "H_actual_lambda": h_lambda},
            "H_shape": list(H.shape),
            "H_float64_sha256": hashlib.sha256(np.ascontiguousarray(H).tobytes()).hexdigest(),
            "p2_gate": p2_gate,
            "calibration": {"path": str(calib_path), "sha256": _file_sha256(calib_path)},
            "data": [{"tier": tier, "ep": _episode_number(path), "path": str(path),
                      "sha256": _file_sha256(path)}
                     for tier in sorted(logs) for path in logs[tier]],
        }
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps(_jsonable(p2_report), ensure_ascii=False,
                                      indent=2, sort_keys=True) + "\n")
        print(f"P2 beta seeds: {p2_gate['p2_beta_by_seed']}  "
              f"median={p2_gate['p2_beta_median']:.4f} "
              f"=> {'PASS' if p2_gate['p2_pass'] else 'FAIL'}")
        print(f"sidecar: {sidecar}")
        print("P3 was not loaded or evaluated in --stage p2.")
        return 0 if p2_gate["p2_pass"] else 2

    theta, H, epsilon, diagnostics = fit_all(episodes)
    manifest = artifact_manifest(R, b, epsilon, diagnostics, logs, calib_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    bg = diagnostics["beta_gates"]
    pg = diagnostics["p0_gate"]
    print(f"P2 beta seeds: {bg['p2_beta_by_seed']}  median={bg['p2_beta_median']:.4f} "
          f"=> {'PASS' if bg['p2_pass'] else 'FAIL'}")
    print(f"P3 beta={bg['p3_beta']:.4f}, ratio={bg['p3_over_p2_median']:.4f} "
          f"=> {'PASS' if bg['p3_pass'] else 'FAIL'}")
    print(f"P0 ep3-4 reductions={[round(100*x, 2) for x in pg['p0_holdout_reduction']]}%, "
          f"median={100*pg['p0_median_reduction']:.2f}% "
          f"=> {'PASS' if pg['p0_pass'] else 'FAIL'}")
    print(f"sidecar: {sidecar}")
    if not diagnostics["all_gates_pass"]:
        print("Frozen gate failed: deployable NPZ was NOT written.")
        return 2

    save_artifact(output, theta, H, R, b, epsilon, manifest)
    print(f"artifact: {output}")
    print(f"theta={theta.shape}, H={H.shape}, epsilon_lms={epsilon:.8g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
