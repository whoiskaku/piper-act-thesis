#!/usr/bin/env python3
"""Stage 1.3 — NoisyPiperSim wrapper.

Wraps the MuJoCo Piper model and injects 3 noise layers per RESEARCH_PLAN_v4.3
and noise_params.json:

  Layer A (Action / motor): A_executed = A_commanded + ε_motor + overshoot(vel)
  Layer B (IMU sensor):     IMU_meas = IMU_sim + ε_accel + ε_gyro
  Layer C (Vibration):      IMU_meas += amp × sin(2π · 12 Hz · t + φ)

Used by Stage 2 to generate paired (A_clean, IMU_clean, A_noisy, IMU_noisy)
training data, and by Stage 5 sim-replay for the consistency filter.

Public API:
  sim = NoisyPiperSim(model_path, noise_params_path, seed=0)
  sim.reset(init_qpos=None)
  obs = sim.step(action, noise=True)
       → returns dict {qpos, qvel, imu_clean, imu_noisy}
  result = sim.execute_action_chunk(start_qpos, action_chunk)
       → runs two passes (clean + noisy) and returns 6-tuple paired sample
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np


CONTROL_HZ = 30.0
CONTROL_DT = 1.0 / CONTROL_HZ

# Piper action / state convention vs MuJoCo internal convention:
#   - Piper (and ACT / RefineNet) use 7 dims: [j1..j6, gripper]
#   - MuJoCo model has 8 actuators: [j1..j6, joint7 = grip, joint8 = -grip mirror]
# Conversion is required at the sim boundary. RefineNet I/O stays 7-dim.

ACTION_DIM_PIPER = 7
CTRL_DIM_MUJOCO = 8


def action_7_to_ctrl_8(a7: np.ndarray, grip_min: float = 0.0,
                       grip_max: float = 0.035) -> np.ndarray:
    """Map Piper 7-dim action to MuJoCo 8-dim ctrl (gripper finger mirror)."""
    a7 = np.asarray(a7, dtype=np.float64)
    if a7.ndim == 1:
        ctrl = np.zeros(CTRL_DIM_MUJOCO)
        ctrl[:6] = a7[:6]
        grip = float(np.clip(a7[6], grip_min, grip_max))
        ctrl[6] = grip
        ctrl[7] = -grip
        return ctrl
    elif a7.ndim == 2:
        ctrl = np.zeros((a7.shape[0], CTRL_DIM_MUJOCO))
        ctrl[:, :6] = a7[:, :6]
        grip = np.clip(a7[:, 6], grip_min, grip_max)
        ctrl[:, 6] = grip
        ctrl[:, 7] = -grip
        return ctrl
    else:
        raise ValueError(f"unsupported a7 shape: {a7.shape}")


def ctrl_8_to_action_7(ctrl8: np.ndarray) -> np.ndarray:
    """Reverse mapping: MuJoCo 8-dim ctrl back to Piper 7-dim action.

    Uses joint7 (gripper opening) as the gripper dimension; joint8 is its mirror
    so it carries no extra info.
    """
    ctrl8 = np.asarray(ctrl8, dtype=np.float64)
    if ctrl8.ndim == 1:
        a7 = np.zeros(ACTION_DIM_PIPER)
        a7[:6] = ctrl8[:6]
        a7[6] = ctrl8[6]
        return a7
    elif ctrl8.ndim == 2:
        a7 = np.zeros((ctrl8.shape[0], ACTION_DIM_PIPER))
        a7[:, :6] = ctrl8[:, :6]
        a7[:, 6] = ctrl8[:, 6]
        return a7
    else:
        raise ValueError(f"unsupported ctrl8 shape: {ctrl8.shape}")


def qpos_8_to_state_7(qpos8: np.ndarray) -> np.ndarray:
    """Reduce MuJoCo qpos (8-dim, with mirrored gripper finger) to Piper 7-dim state.

    Same shape rule as ctrl_8_to_action_7. Useful for handing sim state to ACT
    or RefineNet, which expect 7-dim Piper-convention state.
    """
    return ctrl_8_to_action_7(qpos8)


class NoisyPiperSim:
    def __init__(
        self,
        model_path: str | Path,
        noise_params_path: str | Path,
        seed: int = 0,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        self.n_phys_per_ctrl = int(round(CONTROL_DT / self.model.opt.timestep))
        self.dt_ctrl = CONTROL_DT
        self.nq = self.model.nq           # 8 (6 arm + 2 gripper fingers)
        self.nu = self.model.nu           # 8 actuators
        self.t = 0.0                       # control-step time

        self.rng = np.random.default_rng(seed)

        # ----- noise params -----
        with open(noise_params_path) as f:
            self.params = json.load(f)

        # Layer A — action noise
        active_joints = self.params["calibrated_from_v3"]["active_joints_indices"]
        # active_joints indices are in the 0..6 v3 action space (7 dims).
        # We map them onto the 8 actuator slots: actuators 0..5 are arm joints,
        # 6 is gripper joint7, 7 is gripper joint8 (mirror).
        self.active_actuators = list(active_joints)
        if 6 in self.active_actuators:
            self.active_actuators.append(7)

        h = self.params["heuristic_defaults_for_stage1"]
        self.sigma_motor = float(h["motor_noise_sigma_per_active_joint_rad"])
        self.overshoot_coef = float(h["overshoot_coefficient"])
        self.overshoot_vel_thresh = float(h["overshoot_velocity_threshold_rad_per_s"])

        # Layer B — IMU sensor noise
        gyro_sigma_dict = self.params["calibrated_from_v3"]["gyro_sensor_noise_sigma"]
        self.sigma_gyro = np.array([gyro_sigma_dict[k] for k in ("gx", "gy", "gz")])
        self.sigma_accel = np.full(3, float(h["accel_sensor_sigma_per_channel"]))

        # Layer C — vibration
        self.vib_freq = float(h["vibration_sinusoid_freq_hz"])
        self.vib_amp_accel = float(h["vibration_sinusoid_amplitude_accel"])
        self.vib_amp_gyro = float(h["vibration_sinusoid_amplitude_gyro"])
        # Random phase per channel, regenerated each reset (so consecutive eps differ)
        self.vib_phase = self.rng.uniform(0, 2 * np.pi, size=6)

        # For Layer A overshoot, track previous commanded action (for velocity calc)
        self.prev_ctrl = np.zeros(self.nu)

    # --------------------------------------------------------------- reset
    def reset(self, init_qpos: Optional[np.ndarray] = None):
        mujoco.mj_resetData(self.model, self.data)
        if init_qpos is not None:
            self.data.qpos[: len(init_qpos)] = init_qpos
            mujoco.mj_forward(self.model, self.data)
        self.t = 0.0
        self.prev_ctrl = self.data.qpos[: self.nu].copy()
        self.vib_phase = self.rng.uniform(0, 2 * np.pi, size=6)

    # --------------------------------------------------------------- noise injectors
    def _apply_action_noise(self, action_commanded: np.ndarray) -> np.ndarray:
        """Layer A: motor noise + velocity-dependent overshoot."""
        action_executed = action_commanded.copy().astype(np.float64)

        # 1) Gaussian motor noise on active actuators only
        for i in self.active_actuators:
            action_executed[i] += self.rng.normal(0.0, self.sigma_motor)

        # 2) Velocity-dependent overshoot:
        #    if |velocity| > threshold, push action further by overshoot_coef × vel²
        #    (sign-preserving — overshoots in direction of motion)
        vel = (action_commanded - self.prev_ctrl) / self.dt_ctrl
        for i in self.active_actuators:
            if abs(vel[i]) > self.overshoot_vel_thresh:
                action_executed[i] += self.overshoot_coef * np.sign(vel[i]) * vel[i] ** 2

        return action_executed

    def _apply_imu_noise(self, imu_clean: np.ndarray) -> np.ndarray:
        """Layer B + C: sensor noise (Gaussian) + low-freq vibration sinusoid."""
        out = imu_clean.copy().astype(np.float64)

        # Layer B: per-channel Gaussian
        out[:3] += self.rng.normal(0.0, self.sigma_accel, size=3)
        out[3:] += self.rng.normal(0.0, self.sigma_gyro, size=3)

        # Layer C: vibration sinusoid (one channel per IMU dim, random phase)
        phase = 2 * np.pi * self.vib_freq * self.t + self.vib_phase
        out[:3] += self.vib_amp_accel * np.sin(phase[:3])
        out[3:] += self.vib_amp_gyro * np.sin(phase[3:])

        return out

    # --------------------------------------------------------------- step
    def step(self, action_commanded: np.ndarray, noise: bool = True) -> dict:
        """One 30-Hz control step. action_commanded shape (nu,) = (8,)."""
        action_commanded = np.asarray(action_commanded, dtype=np.float64)
        if action_commanded.shape != (self.nu,):
            raise ValueError(f"action shape must be ({self.nu},), got {action_commanded.shape}")

        # Layer A
        action_executed = (
            self._apply_action_noise(action_commanded) if noise else action_commanded
        )
        self.data.ctrl[:] = action_executed

        # Physics (n_phys_per_ctrl steps to reach next control time)
        for _ in range(self.n_phys_per_ctrl):
            mujoco.mj_step(self.model, self.data)

        imu_clean = self.data.sensordata[:6].copy()  # 3 accel + 3 gyro
        imu_noisy = self._apply_imu_noise(imu_clean) if noise else imu_clean

        obs = {
            "qpos": self.data.qpos[: self.nq].copy(),
            "qvel": self.data.qvel[: self.model.nv].copy(),
            "imu_clean": imu_clean,
            "imu_noisy": imu_noisy,
            "action_commanded": action_commanded,
            "action_executed": action_executed.copy(),
            "t": self.t,
        }

        self.prev_ctrl = action_commanded.copy()
        self.t += self.dt_ctrl

        return obs

    # --------------------------------------------------------------- chunk
    def execute_action_chunk(
        self,
        start_qpos: Optional[np.ndarray],
        action_chunk: np.ndarray,
        common_seed: Optional[int] = None,
    ) -> dict:
        """Two-pass execution producing paired (clean, noisy) trajectories.

        For Stage 2 data generation. action_chunk shape (T, nu).
        Returns 6-tuple consistent with RESEARCH_PLAN_v4.3 §5 Stage 2:
            A_clean:   commanded actions     (T, nu)
            IMU_clean: noiseless sim IMU     (T, 6)
            A_noisy:   noise-injected exec   (T, nu)
            IMU_noisy: noisy sim IMU         (T, 6)
            qpos_clean / qpos_noisy: state trajectories (T, nq)
        """
        action_chunk = np.asarray(action_chunk, dtype=np.float64)
        T = action_chunk.shape[0]

        # Pass 1: clean (no noise)
        seed_save = self.rng
        if common_seed is not None:
            self.rng = np.random.default_rng(common_seed)
        self.reset(init_qpos=start_qpos)
        clean_qpos = np.zeros((T, self.nq))
        clean_imu = np.zeros((T, 6))
        for t in range(T):
            obs = self.step(action_chunk[t], noise=False)
            clean_qpos[t] = obs["qpos"]
            clean_imu[t] = obs["imu_clean"]

        # Pass 2: noisy
        if common_seed is not None:
            self.rng = np.random.default_rng(common_seed + 1)
        else:
            self.rng = seed_save
        self.reset(init_qpos=start_qpos)
        noisy_qpos = np.zeros((T, self.nq))
        noisy_imu = np.zeros((T, 6))
        noisy_action_exec = np.zeros((T, self.nu))
        for t in range(T):
            obs = self.step(action_chunk[t], noise=True)
            noisy_qpos[t] = obs["qpos"]
            noisy_imu[t] = obs["imu_noisy"]
            noisy_action_exec[t] = obs["action_executed"]

        # Convert MuJoCo 8-dim ctrl ↔ Piper 7-dim action at the sim boundary.
        # RefineNet / ACT operate in 7-dim space; only sim physics uses 8-dim.
        return {
            # Piper / RefineNet 7-dim convention (primary):
            "A_clean":  ctrl_8_to_action_7(action_chunk),       # (T, 7) target
            "A_noisy":  ctrl_8_to_action_7(noisy_action_exec),  # (T, 7) sim-executed
            "IMU_clean": clean_imu,                              # (T, 6)
            "IMU_noisy": noisy_imu,                              # (T, 6)

            # MuJoCo 8-dim (kept for debug / sim-side analysis):
            "A_clean_8":  action_chunk,
            "A_noisy_8":  noisy_action_exec,
            "qpos_clean": clean_qpos,                            # (T, 8) sim state
            "qpos_noisy": noisy_qpos,
        }
