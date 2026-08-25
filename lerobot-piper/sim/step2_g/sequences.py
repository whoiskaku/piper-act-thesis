#!/usr/bin/env python3
"""Step 2 激励序列生成器 —— 20 Hz 指令，供 `replay_probe.py` 开环复放。

设计依据全部来自零机时的结构性分析，不是猜的：

  · 绕 j 轴的等效惯量**只**取决于比 j 更远端的关节姿态
    ⇒ 激励 j 时要变的是 `j+1..6`（`step2_identifiability.py` 实测：
      改近端完全无效，probe_c 三位形的 `I_link@j3` 极差 0.0%）。
  · `b` 与 `kv` 精确简并 ⇒ 参数集已合并为 `c`，序列设计**不必**为分离二者服务
    （`step2_g/test_degeneracy.py`）。慢速双向扫描仍然要做 —— 它是为 `fc` 的。
  · 确定性内容在 **9–12 Hz** 截止（`step2_ceiling.py`）⇒ chirp 不必往上扩。
    且指令率只有 20 Hz，故 chirp 封顶 **6 Hz**（≥3.3 样本/周期）；
    更高频的激励靠**阶跃**的宽带内容提供。
  · **j4 在 pp2 里的指令范围是 `0.0…0.0`** —— 从未动过。j5 也几乎不动。
    这些关节的全部参数目前无数据可辨识，必须直接激励。

安全口径（重要）：
  本模块只做**关节限位**与**速度/加速度**检查。`piper_model.xml` 的碰撞是
  **全关**的（`contype/conaffinity` 非零 0/17），所以**不得**声称「碰撞安全通过」，
  只能说「关节限位通过」。序列会走出 pp2 已验证的包络（尤其 j4/j5），
  **上机前必须由人在 MuJoCo 里目视确认无碰撞**。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import mujoco

FPS = 20.0
DT = 1.0 / FPS
LIMIT_MARGIN_DEG = 5.0
CHIRP_F_MAX = 6.0                    # 指令率 20 Hz ⇒ ≥3.3 样本/周期
GRIP_HOLD = 0.0                      # 夹爪全程不动

# 速度/加速度上限：pp2 实测 p95；j4 在 pp2 里从未动过，借 j5 的（更保守的腕关节值）
VMAX_DEG = np.array([119.2, 293.7, 123.9, 58.3, 58.3, 143.7])
# 近端关节的停放姿态 = pp2 实测指令范围的中点 —— 真机已反复安全经过, 且远离限位。
# **不能**简单置 0: j2 的下限就是 0、j3 的上限就是 0, 置 0 即触 margin。
HOME_DEG = np.array([-4.9, 51.5, -25.2, 0.0, 9.4, 4.0])
AMAX_DEG = np.array([2429.0, 6475.0, 2448.0, 850.0, 850.0, 2976.0])


@dataclass
class Seq:
    name: str
    kind: str
    target: tuple[int, ...]
    cmd: np.ndarray                        # (T,7) rad
    pose_distal: np.ndarray                # 远端关节配置(rad)，决定 I_link[target]
    meta: dict = field(default_factory=dict)

    @property
    def sha(self) -> str:
        return hashlib.sha256(np.round(self.cmd, 9).tobytes()).hexdigest()[:12]


# ── 位形选择 ────────────────────────────────────────────────────────────
def link_inertia(m, d, M, q6: np.ndarray) -> np.ndarray:
    d.qpos[:6] = q6
    d.qpos[6] = d.qpos[7] = 0.0
    mujoco.mj_forward(m, d)
    mujoco.mj_fullM(m, M, d.qM)
    return np.diag(M)[:6].copy()


def pick_distal_poses(m, j: int, n: int = 3, n_sample: int = 3000,
                      seed: int = 20260824) -> list[np.ndarray]:
    """为激励关节 `j` 选 n 个**远端**配置，使 `I_link[j]` 的跨度尽量大。

    近端关节固定在 pp2 的典型姿态（不影响 `I_link[j]`，但决定重力负载）。
    """
    d, M = mujoco.MjData(m), np.zeros((m.nv, m.nv))
    lo = m.jnt_range[:6, 0] + np.deg2rad(LIMIT_MARGIN_DEG)
    hi = m.jnt_range[:6, 1] - np.deg2rad(LIMIT_MARGIN_DEG)
    home = np.deg2rad(HOME_DEG)
    if j >= 5:                                    # j6 无远端 ⇒ 只有一个位形
        return [home.copy()]
    rng = np.random.default_rng(seed + j)
    Q = rng.uniform(lo, hi, size=(n_sample, 6))
    Q[:, : j + 1] = home[: j + 1]                 # 近端停 HOME, 只变远端
    I = np.array([link_inertia(m, d, M, q)[j] for q in Q])
    order = np.argsort(I)
    idx = [order[0], order[-1]] if n <= 2 else \
          [order[0], order[len(order) // 2], order[-1]][:n]
    return [Q[i] for i in idx]


# ── 基元 ────────────────────────────────────────────────────────────────
def safe_center(m, j: int, amp_deg: float) -> float:
    """关节 j 的起始角(度)，保证 ±amp 全程留在限位内。

    必需，因为若干关节的限位**一端就是 0**（j2 下限 0、j3 上限 0），
    从 0 出发做双向摆动必然越界。
    """
    lo = np.rad2deg(m.jnt_range[j, 0]) + LIMIT_MARGIN_DEG + abs(amp_deg)
    hi = np.rad2deg(m.jnt_range[j, 1]) - LIMIT_MARGIN_DEG - abs(amp_deg)
    if lo > hi:                                   # 幅值过大，取中点并由调用方缩幅
        return float((lo + hi) / 2)
    return float(np.clip(0.0, lo, hi))


def _base(pose: np.ndarray, T: int) -> np.ndarray:
    c = np.zeros((T, 7))
    c[:, :6] = pose[None, :6]
    c[:, 6] = GRIP_HOLD
    return c


def _raised_cos(n: int) -> np.ndarray:
    """0→1 的平滑过渡（升余弦），端点速度与加速度均为 0。"""
    t = np.linspace(0.0, 1.0, n, endpoint=True)
    return 0.5 * (1.0 - np.cos(np.pi * t))


def smooth_step(j: int, pose: np.ndarray, amp_deg: float, rise_frames: int,
                hold_s: float = 1.5, n_rep: int = 2, q0_deg: float = 0.0) -> np.ndarray:
    """平滑阶跃（升余弦上升）+ 保持，双向各 n_rep 次。"""
    hold = int(hold_s * FPS)
    unit = rise_frames + hold
    T = 2 * n_rep * unit + hold
    c = _base(pose, T)
    q = np.full(T, np.deg2rad(q0_deg))
    k = hold
    for r in range(n_rep):
        for sgn in (+1, -1):
            q[k:k + rise_frames] = q[k - 1] + sgn * np.deg2rad(amp_deg) * _raised_cos(rise_frames)
            q[k + rise_frames:k + unit] = q[k + rise_frames - 1]
            k += unit
    c[:, j] = q
    return c


def slow_sweep(j: int, pose: np.ndarray, amp_deg: float, speed_dps: float,
               n_rev: int = 4, q0_deg: float = 0.0) -> np.ndarray:
    """慢速**双向**扫描（三角波）—— 换向处的死区是库仑摩擦 `fc` 的直接签名。"""
    leg = max(int(abs(amp_deg) / max(speed_dps, 1e-6) * FPS), 4)
    T = n_rev * leg + int(1.0 * FPS)
    c = _base(pose, T)
    q = np.full(T, np.deg2rad(q0_deg))
    k = int(0.5 * FPS)
    sgn = 1
    while k + leg <= T:
        q[k:k + leg] = q[k - 1] + sgn * np.deg2rad(amp_deg) * np.linspace(0, 1, leg)
        k += leg
        sgn = -sgn
    q[k:] = q[k - 1]
    c[:, j] = q
    return c


def chirp(j: int, pose: np.ndarray, amp_deg: float, f0: float = 0.3,
          f1: float = CHIRP_F_MAX, dur_s: float = 20.0, q0_deg: float = 0.0) -> np.ndarray:
    """对数扫频。封顶 `CHIRP_F_MAX` —— 指令率 20 Hz 限制，且 >12 Hz 无可复现信号。"""
    T = int(dur_s * FPS)
    c = _base(pose, T)
    t = np.arange(T) * DT
    k = np.log(f1 / f0) / dur_s
    ph = 2 * np.pi * f0 * (np.exp(k * t) - 1) / k
    env = np.clip(np.minimum(t / 1.0, (dur_s - t) / 1.0), 0, 1)      # 首尾渐入渐出
    c[:, j] = np.deg2rad(q0_deg) + np.deg2rad(amp_deg) * env * np.sin(ph)
    return c


def multisine(joints: tuple[int, ...], pose: np.ndarray, amps_deg: dict[int, float],
              dur_s: float = 25.0, n_tone: int = 12, seed: int = 0) -> np.ndarray:
    """多关节随机相位多正弦 —— 唯一能激励**惯量耦合**项的形式。"""
    T = int(dur_s * FPS)
    c = _base(pose, T)
    t = np.arange(T) * DT
    rng = np.random.default_rng(seed)
    f = np.logspace(np.log10(0.3), np.log10(CHIRP_F_MAX), n_tone)
    env = np.clip(np.minimum(t / 1.5, (dur_s - t) / 1.5), 0, 1)
    for j in joints:
        ph = rng.uniform(0, 2 * np.pi, n_tone)
        s = np.sum(np.sin(2 * np.pi * f[:, None] * t[None, :] + ph[:, None]), axis=0)
        s /= np.abs(s).max()
        c[:, j] = pose[j] + np.deg2rad(amps_deg[j]) * env * s
    return c


# ── 安全与度量 ──────────────────────────────────────────────────────────
def fit_limits(m, cmd: np.ndarray, j: int) -> tuple[np.ndarray, float]:
    """把关节 j 的幅值等比缩到速度/加速度上限内。→ (缩放后的指令, 缩放比)。

    对固定形状的轨迹 `v ∝ amp`、`a ∝ amp`，故一个标量缩放即可同时满足两者。
    慢速扫描的 `v` 由速度参数决定、与幅值无关，缩放只会缩短行程 —— 无副作用。
    """
    q0 = cmd[0, j]
    dev = cmd[:, j] - q0
    v = np.abs(np.diff(dev)) * FPS
    a = np.abs(np.diff(dev, n=2)) * FPS ** 2
    sv = np.deg2rad(VMAX_DEG[j]) / max(v.max(), 1e-12)
    sa = np.deg2rad(AMAX_DEG[j]) / max(a.max(), 1e-12)
    s = float(min(1.0, sv, sa))
    if s < 1.0:
        cmd = cmd.copy()
        cmd[:, j] = q0 + dev * s
    return cmd, s


def max_amp(m, j: int, gen, probe_deg: float = 1.0) -> float:
    """给定波形生成器，关节 j 在速度/加速度限内的**可达最大幅值**(度)。

    必需，否则绝对度数的幅值阶梯会在限位处**塌成同一条序列**
    （实测 j4 的 a4 与 a12 缩放后 hash 相同 ⇒ 重复采集、灵敏度重复计数）。
    """
    _, s = fit_limits(m, gen(probe_deg), j)
    return probe_deg * s


def check_limits(m, cmd: np.ndarray) -> dict:
    """**只**检查关节限位与速度/加速度。碰撞在模型里是关闭的 ⇒ 不做碰撞判定。"""
    q = cmd[:, :6]
    lo = m.jnt_range[:6, 0] + np.deg2rad(LIMIT_MARGIN_DEG)
    hi = m.jnt_range[:6, 1] - np.deg2rad(LIMIT_MARGIN_DEG)
    v = np.abs(np.diff(q, axis=0)) * FPS
    a = np.abs(np.diff(q, n=2, axis=0)) * FPS ** 2
    return {
        "limit_ok": bool((q >= lo).all() and (q <= hi).all()),
        "vel_ok": bool((v.max(0) <= np.deg2rad(VMAX_DEG) + 1e-9).all()),
        "acc_ok": bool((a.max(0) <= np.deg2rad(AMAX_DEG) + 1e-9).all()),
        "v_max_deg": np.rad2deg(v.max(0)),
        "a_max_deg": np.rad2deg(a.max(0)),
        "q_lo_deg": np.rad2deg(q.min(0)), "q_hi_deg": np.rad2deg(q.max(0)),
    }


def inertia_span(m, cmd: np.ndarray, j: int) -> tuple[float, float]:
    d, M = mujoco.MjData(m), np.zeros((m.nv, m.nv))
    I = np.array([link_inertia(m, d, M, q)[j] for q in cmd[::10, :6]])
    return float(I.min()), float(I.max())
