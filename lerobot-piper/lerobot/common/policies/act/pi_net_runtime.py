"""`pi-g2env-v1` 的真机运行时 —— **纯 π 网络**，运行时不含任何模型或求解器。

结构
----
    chunk 生成那一刻:  [上一 chunk 指令; 本 chunk 指令] → π → δa[0..99]   （一次前向）
    逐帧:              δa[k] → 接触门控 → ε 硬限幅 → 叠到 ACT 指令上

**运行时不需要 G-2 / H / Ĥ_θ / 解析求解器，也不跑 MuJoCo。**
G-2 只在离线训练时充当环境，训练完它的控制知识就存在 π 的权重里了
（用户 2026-09-16 定稿的结构）。

与 `g2wall` 的两个结构差别
--------------------------
1. **没有空档。** `g2wall` 要在后台跑 G-2 前向 + ideal 渲染（~150 ms），
   实测每 chunk 空档 3–4 帧；而墙的能量 **52.2% 在 chunk 头 4 帧**
   ⇒ 那是它从 15.50% 掉到 ~7.6% 的原因。π 一次前向 <2 ms，**第 0 帧就能补**。
2. **不读 IMU。** v1 是纯前馈 —— 因为 G-2 是指令的确定性函数，
   仿真里的 IMU/编码器携带不了额外信息，放进去只会让 π 去拟合仿真特征。

⚠ 事前预测（上机前写死，见 `reports/RUNBOOK_pi_net_ACD.md`）
------------------------------------------------------------
用真机已验证的 `H`（β=0.955，且正确预测了 C 臂的 −7.6% vs 实测 −8.0%）推算：
**`D − A` ≈ −0.1%**，对照 `C − A` 实测 **−8.0%**。
π 的 δa RMS **2.16 mrad**（C 臂的 4.0×）、峰值**顶到 ε=10 mrad**。
成因：方向学自 G-2 的增量响应，而它与真机的 `cos` 仅 **0.12**。

⚠ 安全
------
ε=10 mrad 在网络内由 `tanh` 结构性保证，**运行时外面再限幅一次**兜底；
接触门控与 `g2wall` 同口径（由夹爪指令决定，逐 chunk 带 `previous_grip`）。
幅度在已验证范围内：P3 开环探针跑过 3 mrad RMS / 10 mrad 峰值。
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lerobot.common.policies.act.step1_correction_runtime import command_contact_mask

EPS = 0.010                 # **瞬时峰值**硬界（与 g2wall 同，保证 C/D 可比）
CHUNK = 100
GRIP_FULL = 0.035
CONTACT_THRESHOLD_RATIO = 0.02
CONTACT_HALF_WINDOW = 10

_REPO = Path(__file__).resolve().parents[4]
_SIM_PATHS = ("sim", "sim/step4_gvnext")


class PiNetError(RuntimeError):
    """启动期校验失败 —— 只在 `__init__` 抛出，**运行中绝不抛**。"""


def _ensure_sim_path() -> None:
    for rel in _SIM_PATHS:
        p = str(_REPO / rel)
        if p not in sys.path:
            sys.path.insert(0, p)
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))


@dataclass
class PiNetFrame:
    delta: np.ndarray
    k: int
    contact_gated: bool
    saturated: bool
    stage: str                  # "pi" / "none"


class PiNetRuntime:
    """逐 chunk 一次前向 → 逐帧发。**运行中不抛异常**（异常 → δa=0 并计数）。"""

    def __init__(self, ckpt_path: str | Path, device="cpu", verbose: bool = True):
        _ensure_sim_path()
        self.device = device
        self.ckpt_path = Path(ckpt_path)
        try:
            ck = torch.load(self.ckpt_path, map_location=device, weights_only=False)
        except Exception as exc:                        # 启动期，可抛
            raise PiNetError(f"读不到 π 检查点 {ckpt_path}: {exc}") from exc
        for k in ("state_dict", "norm_mu", "norm_sd", "eps_rad"):
            if k not in ck:
                raise PiNetError(f"检查点缺字段 `{k}`（不是 train_pi_g2env 产出的？）")
        if abs(float(ck["eps_rad"]) - EPS) > 1e-12:
            raise PiNetError(f"检查点的 ε={ck['eps_rad']} 与运行时 {EPS} 不一致")
        try:
            import train_pi_g2env as T
            self._T = T
            self.net = T.PiNet().to(device)
            self.net.load_state_dict(ck["state_dict"])
            self.net.eval()
        except Exception as exc:
            raise PiNetError(f"装载 π 失败: {exc}") from exc
        self.mu = np.asarray(ck["norm_mu"], np.float32)
        self.sd = np.asarray(ck["norm_sd"], np.float32)
        self.ckpt_sha256 = hashlib.sha256(self.ckpt_path.read_bytes()).hexdigest()
        self.val_episodes = ck.get("val_episodes", [])
        self.n_params = sum(p.numel() for p in self.net.parameters())
        self.reset()
        self._warmup()
        if verbose:
            print(f"[pi_net] {self.ckpt_path.name} ✓  {self.n_params} 参数 · "
                  f"ε={EPS*1e3:.0f}mrad(tanh 结构硬界) · **纯前馈、不读 IMU** · "
                  f"sha {self.ckpt_sha256[:8]}…")

    # ---- 生命周期 ----
    def reset(self) -> None:
        self._plan: np.ndarray | None = None
        self._contact: np.ndarray | None = None
        self._k = 0
        self._prev_cmd: np.ndarray | None = None
        self._prev_grip: float | None = None
        self.stats = {"frames": 0, "gated": 0, "sat": 0, "no_plan": 0,
                      "infer_ms": [], "fail": 0}

    def _warmup(self) -> None:
        """先跑一发，别让第一个 chunk 付掉 torch 的首次分配开销。"""
        try:
            self.begin_chunk(np.zeros((CHUNK, 7)))
            self.reset()
        except Exception as exc:                        # 启动期，可抛
            raise PiNetError(f"π 前向自检失败: {exc}") from exc

    # ---- chunk 装配（**同步**，一次前向）----
    def begin_chunk(self, cmd_raw: np.ndarray) -> None:
        """chunk 生成那一帧调用。失败 → 本 chunk δa=0，**不抛**。"""
        c = np.asarray(cmd_raw, np.float64)
        T_ = len(c)
        t0 = time.perf_counter()
        self._contact = command_contact_mask(
            c[:, 6].astype(np.float32), grip_full=GRIP_FULL,
            threshold_ratio=CONTACT_THRESHOLD_RATIO,
            half_window=CONTACT_HALF_WINDOW, previous_grip=self._prev_grip)
        try:
            prev = (self._prev_cmd if self._prev_cmd is not None
                    else np.zeros((CHUNK, 7), np.float32))
            cur = np.zeros((CHUNK, 7), np.float32); cur[:min(CHUNK, T_)] = c[:CHUNK]
            pv = np.zeros((CHUNK, 7), np.float32); pv[:min(CHUNK, len(prev))] = prev[:CHUNK]
            seq = np.concatenate([pv, cur], 0)
            d1 = np.concatenate([np.zeros((1, 7), np.float32), np.diff(seq, axis=0)], 0)
            d2 = np.concatenate([np.zeros((1, 7), np.float32), np.diff(d1, axis=0)], 0)
            rel = seq - seq[CHUNK:CHUNK + 1]
            x = np.concatenate([rel, d1, d2], 1).T[None]          # (1,21,200)
            xn = (x - self.mu) / self.sd
            with torch.no_grad():
                da = self.net(torch.from_numpy(xn.astype(np.float32)).to(self.device))
            self._plan = da[0].cpu().numpy().astype(np.float64)   # (100,6)
        except Exception as exc:
            self.stats["fail"] += 1
            self._plan = None
            print(f"[pi_net] ⚠ 前向失败 → 本 chunk δa=0: {type(exc).__name__}: {exc}")
        self.stats["infer_ms"].append((time.perf_counter() - t0) * 1e3)
        self._prev_cmd = c[:CHUNK].astype(np.float32)
        self._prev_grip = float(c[-1, 6])
        self._k = 0

    # ---- 逐帧 ----
    def step(self) -> PiNetFrame:
        k = self._k
        self._k += 1
        self.stats["frames"] += 1
        zero = np.zeros(6, np.float32)
        if self._plan is None or k >= len(self._plan):
            self.stats["no_plan"] += 1
            return PiNetFrame(zero, k, False, False, "none")
        d = self._plan[k]
        if not np.isfinite(d).all():
            self.stats["no_plan"] += 1
            return PiNetFrame(zero, k, False, False, "nan")
        sat = bool(np.any(np.abs(d) >= EPS * 0.999))
        self.stats["sat"] += int(sat)
        d = np.clip(d, -EPS, EPS)                       # 网络内已 tanh，这里兜底
        contact = bool(self._contact is not None and k < len(self._contact)
                       and self._contact[k])
        if contact:
            self.stats["gated"] += 1
            d = np.zeros(6)
        return PiNetFrame(d.astype(np.float32), k, contact, sat, "pi")

    def summary(self) -> dict:
        s, n = self.stats, max(self.stats["frames"], 1)
        v = s["infer_ms"]
        return {"artifact": self.ckpt_path.stem, "n_params": self.n_params,
                "ckpt_sha256": self.ckpt_sha256,
                "frames": s["frames"], "gated_pct": 100 * s["gated"] / n,
                "sat_pct": 100 * s["sat"] / n, "no_plan_pct": 100 * s["no_plan"] / n,
                "infer_ms_p95": float(np.percentile(v, 95)) if v else 0.0,
                "infer_ms_max": float(max(v)) if v else 0.0,
                "forward_fail": s["fail"], "eps_rad": EPS,
                "feedforward_only": True, "reads_imu": False,
                "blackout_frames": 0,
                "val_episodes": self.val_episodes}


__all__ = ["PiNetRuntime", "PiNetFrame", "PiNetError", "EPS", "CHUNK"]
