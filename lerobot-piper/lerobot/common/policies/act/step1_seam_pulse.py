"""Step 1 v2 —— 接缝触发的三参数脉冲补正。**无 G、无 H、无求逆。**

预注册: `reports/PREREG_STEP1_V2.md`（采集前冻结）。本文件是那份规格的**唯一**实现。

    θ = (A_prox, A_wrist, τ)

每个新 chunk 到来时（首个 chunk 不补正）:

    d    = q_prev,last − q_new,first                       ∈ R⁶   纯指令, 决策时可得
    v_G  = d_G / max(‖d_G‖∞, 2 mrad)                       G ∈ {prox=j1-3, wrist=j4-6}
    δa_G(k) = A_G · v_G · exp(−k/τ)                        k = 接缝后第几帧

方向取自**当时的指令跳变**, 不是固定向量, 也不来自任何响应模型。`A_G` 可为负 ⇒
SPSA 能发现"应该强化而非平滑接缝"。夹爪不修改。

为什么不用固定符号的方向: 本次 25 个接缝里 j6 恰好 25/25 同号、j2 80% 同号, 但 n=25
且很可能是本任务结构的偶然; 绑死符号则换任务/换会话可能整体反向。取自 d 则方向永远
由当时的指令自己决定, 且自动按跳变幅度缩放。

结构性安全性质（不依赖数值巧合）:
  ‖v_G‖∞ ≤ 1 恒成立（∞范数归一化）且 |A_G| ≤ 2 mrad ⇒ |δa| ≤ 2 mrad
  ⇒ ±2 mrad 的裁剪是**冗余**兜底, 正常永不触发（`sat_pct` 应恒为 0）。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.common.policies.act.step1_correction_runtime import command_contact_mask

# ── 预注册冻结的常数（改动须同步 PREREG_STEP1_V2.md §2 并留改动记录）──────────
A_LIMIT = 0.002          # |A_G| 上界 (rad) = 2 mrad
DA_LIMIT = 0.002         # δa 硬裁剪 (rad) —— 冗余兜底
V_FLOOR = 0.002          # v_G 归一化的分母下限 (rad); 实测从未触发(见 §7.3)
TAU_MIN, TAU_MAX = 2.0, 10.0
GRIP_FULL = 0.035
CONTACT_THRESHOLD_RATIO = 0.02
CONTACT_HALF_WINDOW = 10
PROX = slice(0, 3)       # j1–j3
WRIST = slice(3, 6)      # j4–j6


class SeamPulseError(ValueError):
    """θ 越界或格式不对。"""


@dataclass(frozen=True)
class SeamPulseTheta:
    """θ = (A_prox, A_wrist, τ)。A 单位 rad, τ 单位帧。"""

    a_prox: float
    a_wrist: float
    tau: float

    def __post_init__(self) -> None:
        for nm, v in (("a_prox", self.a_prox), ("a_wrist", self.a_wrist)):
            if not math.isfinite(v) or abs(v) > A_LIMIT + 1e-12:
                raise SeamPulseError(f"{nm}={v} 超出 ±{A_LIMIT} rad")
        if not math.isfinite(self.tau) or not (TAU_MIN - 1e-9 <= self.tau <= TAU_MAX + 1e-9):
            raise SeamPulseError(f"tau={self.tau} 超出 [{TAU_MIN}, {TAU_MAX}] 帧")

    # ---- 归一化坐标 (预注册 §3.2): 全部落在 [-1, 1] ----
    def to_x(self) -> np.ndarray:
        return np.array([self.a_prox / A_LIMIT,
                         self.a_wrist / A_LIMIT,
                         (self.tau - 6.0) / 4.0], dtype=np.float64)

    @classmethod
    def from_x(cls, x) -> "SeamPulseTheta":
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape != (3,):
            raise SeamPulseError(f"x 必须是 3 维, 得到 {x.shape}")
        x = np.clip(x, -1.0, 1.0)
        return cls(a_prox=float(x[0] * A_LIMIT),
                   a_wrist=float(x[1] * A_LIMIT),
                   tau=float(x[2] * 4.0 + 6.0))

    def to_dict(self) -> dict[str, float]:
        return {"a_prox": self.a_prox, "a_wrist": self.a_wrist, "tau": self.tau}

    @classmethod
    def load(cls, path: str | Path) -> "SeamPulseTheta":
        d = json.loads(Path(path).read_text())
        d = d.get("theta", d)
        missing = {"a_prox", "a_wrist", "tau"} - set(d)
        if missing:
            raise SeamPulseError(f"θ 文件缺字段 {sorted(missing)}: {path}")
        return cls(a_prox=float(d["a_prox"]), a_wrist=float(d["a_wrist"]),
                   tau=float(d["tau"]))

    def save(self, path: str | Path, **extra: Any) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"theta": self.to_dict(), **extra}, indent=1))


def seam_direction(prev_last: np.ndarray, new_first: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(v_prox, v_wrist)。d = q_prev,last − q_new,first, 每组按 ∞ 范数归一化。

    分母取 max(‖d_G‖∞, V_FLOOR) ⇒ ‖v_G‖∞ ≤ 1 恒成立, 小跳变不会被放大。
    """
    prev_last = np.asarray(prev_last, dtype=np.float64).reshape(-1)[:6]
    new_first = np.asarray(new_first, dtype=np.float64).reshape(-1)[:6]
    d = prev_last - new_first
    out = []
    for sl in (PROX, WRIST):
        dg = d[sl]
        out.append(dg / max(float(np.abs(dg).max()), V_FLOOR))
    return out[0], out[1]


def seam_pulse_chunk(theta: SeamPulseTheta, prev_last: np.ndarray | None,
                     chunk_cmd: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """整个 chunk 的 δa (n,6)、接触掩码 (n,)、**裁剪前越界**掩码 (n,)。

    第三项按预注册对 `sat_pct` 的定义("裁剪前越界帧比例")统计。因 ‖v_G‖∞ ≤ 1 且
    |A_G| ≤ 2 mrad, 它**结构上恒为 False** —— 一旦为真就说明实现被改坏了。

    `prev_last is None`（首个 chunk）⇒ δa 全零, 与预注册 §2.5 一致。
    `chunk_cmd`: (n,7) 本 chunk 的**原始 ACT 指令**（未加 δa）。
    """
    chunk_cmd = np.asarray(chunk_cmd, dtype=np.float64)
    if chunk_cmd.ndim != 2 or chunk_cmd.shape[1] < 7:
        raise SeamPulseError(f"chunk_cmd 需为 (n,7), 得到 {chunk_cmd.shape}")
    n = len(chunk_cmd)
    contact = np.asarray(command_contact_mask(
        chunk_cmd[:, 6].astype(np.float32), grip_full=GRIP_FULL,
        threshold_ratio=CONTACT_THRESHOLD_RATIO,
        half_window=CONTACT_HALF_WINDOW), bool)

    da = np.zeros((n, 6), dtype=np.float64)
    if prev_last is None:
        return da, contact, np.zeros(n, dtype=bool)

    v_prox, v_wrist = seam_direction(prev_last, chunk_cmd[0])
    decay = np.exp(-np.arange(n, dtype=np.float64) / theta.tau)      # k = 0..n-1
    da[:, PROX] = theta.a_prox * decay[:, None] * v_prox[None, :]
    da[:, WRIST] = theta.a_wrist * decay[:, None] * v_wrist[None, :]

    over = np.abs(da).max(axis=1) > DA_LIMIT * (1.0 + 1e-9)   # **裁剪前**是否越界
    da = np.clip(da, -DA_LIMIT, DA_LIMIT)       # 冗余兜底, 正常不触发
    da[contact] = 0.0                            # 接触窗归零 (预注册 §6 ⑥-4)
    return da, contact, over


class SeamPulseRuntime:
    """流式接口: 每 chunk 调 `set_chunk`, 每帧调 `step`。**不读 IMU, 不用任何模型。**

    IMU 只在 **episode 之间**（由 SPSA 编排器离线算 loss）使用 —— 这是 v2 与
    v1 的根本区别之一。
    """

    def __init__(self, theta: SeamPulseTheta, verbose: bool = True):
        self.theta = theta
        self.verbose = verbose
        self.reset()
        if verbose:
            print(f"[seam_pulse] A_prox={theta.a_prox*1e3:+.3f} "
                  f"A_wrist={theta.a_wrist*1e3:+.3f} mrad  τ={theta.tau:.2f} 帧  "
                  f"(|δa|≤{DA_LIMIT*1e3:.0f}mrad 结构保证)")

    def reset(self) -> None:
        """每集清零 —— 不跨 episode 继承 (预注册 §2.5)。"""
        self._prev_last: np.ndarray | None = None
        self._da: np.ndarray | None = None
        self._contact: np.ndarray | None = None
        self._over: np.ndarray | None = None
        self._pos = 0
        self._chunks = 0
        self.stats = {"frames": 0, "corrected": 0, "contact_gated": 0,
                      "first_chunk_skipped": 0, "sat": 0, "abs_sum": 0.0}

    def set_chunk(self, chunk_cmd: np.ndarray) -> None:
        """chunk_cmd: (n,7) 原始 ACT 指令。必须在发送本 chunk 第一帧之前调用。"""
        raw = np.asarray(chunk_cmd, dtype=np.float64)
        da, contact, over = seam_pulse_chunk(self.theta, self._prev_last, raw)
        if self._prev_last is None:
            self.stats["first_chunk_skipped"] += len(raw)
        self._da = da
        self._contact = contact
        self._over = over
        self._pos = 0
        self._chunks += 1
        self._prev_last = raw[-1, :6].copy()

    def step(self) -> np.ndarray:
        """本帧的 δa (6,)。chunk 用尽后返回零并计数（正常不该发生）。"""
        if self._da is None or self._pos >= len(self._da):
            self.stats["frames"] += 1
            return np.zeros(6, dtype=np.float32)
        d = self._da[self._pos]
        gated = bool(self._contact[self._pos])
        over = bool(self._over[self._pos])
        self._pos += 1
        self.stats["frames"] += 1
        self.stats["abs_sum"] += float(np.abs(d).mean())
        if gated:
            self.stats["contact_gated"] += 1
        elif np.any(d != 0.0):
            self.stats["corrected"] += 1
        if over:                     # **裁剪前越界**(结构上不该发生)
            self.stats["sat"] += 1
        return d.astype(np.float32)

    def summary(self) -> dict[str, Any]:
        n = max(self.stats["frames"], 1)
        return {"mode": "step1_v2", "theta": self.theta.to_dict(),
                "frames": self.stats["frames"], "chunks": self._chunks,
                "corrected_pct": 100.0 * self.stats["corrected"] / n,
                "contact_gated_pct": 100.0 * self.stats["contact_gated"] / n,
                "first_chunk_skipped": self.stats["first_chunk_skipped"],
                "sat_pct": 100.0 * self.stats["sat"] / n,
                "da_mean_mrad": 1e3 * self.stats["abs_sum"] / n}
