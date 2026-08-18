"""B 校正模块 — 真机部署 wrapper（π v1 = TCN, 2026-07-22 定档）。

把已加载的 frozen ACTPolicy 包一层, 在**每一帧**（非 chunk 生成时刻）插入实时校正:

    ACT chunk ──▶ 逐帧发送 ──▶ [ + δa(t) ] ──▶ 机器人
                                  ▲
                    π(51维因果特征) ◀── 实时 IMU + 编码器 + chunk 内 ideal 预览

与 refine_deploy 的三点根本不同（这是本模块存在的理由）:
  1. **逐帧**校正（RefineNet 是整 chunk 一次性精修）;
  2. **实时 IMU 在环**（指导教員 2026-07-06 的硬要求）;
  3. 输出有 **ε=10mrad 结构硬界**（tanh×ε）, 且**只改 6 臂关节, 夹爪原样透传**。

三方 A/B（env 门控, 与 refine_deploy 同风格）:
    A 裸 ACT:      不设 BCORR_CKPT（或 BCORR_MODE=off）
    B 逆算前馈:    BCORR_CKPT=<ckpt> BCORR_MODE=ff      （FF1/2, 指导教員的逆算案对照臂）
    C 本方法:      BCORR_CKPT=<ckpt> BCORR_MODE=full    （π 实时校正）
    诊断:          BCORR_MODE=pi_only（π 但不含 FF）/ record（只落盘特征不施加）

安全（方案 C, 用户 2026-07-15 选定）:
  · ε 硬界 10mrad（tanh, 结构性不可逾越）
  · 静止 gate 归零（ε 尺度体检: 静止段传递增益仅 0.574, 该处本就修不动）
  · 每帧断言: |δa|≤ε、无 NaN、IMU 新鲜（陈旧则本帧 δa 置零并计数）

⚠️ 真机首跑核对清单（`--dry-run` 会逐项打印）:
  (1) observation.state 单位确为 mdeg（转 rad 后 |q|<3.2）
  (2) IMU 标定 R 与本次安装一致（用当日 rollout 重算, 见 manifest）
  (3) ideal 渲染耗时 < 1 个 chunk 周期（5s）
  (4) 首 L=3 帧 δa 恒为 0（误差窗未填满）
  (5) 全程 sat_pct ≈ 0、gate 触发率与离线一致
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from lerobot.common.policies.act.b_runtime import (
    CHUNK, DT, EPS_B, FEAT_DIM, WIN, FeatureAssembler, IdealChunkRenderer,
    apply_ff12, load_tcn,
)

_MDEG_TO_RAD = math.pi / 180.0 / 1000.0
_GRIP_SCALE = 1.0 / 800000.0        # 与 refine_deploy 一致
MODES = ("off", "ff", "pi_only", "full", "record", "probe")


def make_probe(seed: int, mrad: float, n: int = 4000,
               fs: float = 20.0, lo: float = 0.5, hi: float = 9.0) -> np.ndarray:
    """带限白噪声 δa —— **与状态无关的外生激励**，开环系统辨识专用。

    为什么必须外生: 此前 D/E/F 三条臂的 δa 都由 π 从状态算出(闭环), 于是辨识出的
    "响应" H 里混着控制器 —— 实测 E 与 F(同日同机, 只差 FF 开关)的 H 互不通用,
    用 E 的 H 求解、F 上评估会从 −86.7% 翻成 +187.8%(reports/STEP1_GATE.md)。
    只有与状态无关的激励才能把被控对象与控制器分开。

    频带取 0.5–9 Hz(评分带), 六关节独立, 定种子可完全复现, 硬界 ε 兜底。
    """
    from scipy.signal import butter, filtfilt
    rng = np.random.default_rng(int(seed))
    x = rng.standard_normal((int(n), 6))
    ba = butter(3, [lo / (fs / 2), hi / (fs / 2)], "bandpass")
    x = filtfilt(*ba, x, axis=0)
    x *= (float(mrad) * 1e-3) / np.sqrt((x ** 2).mean())
    return np.clip(x, -EPS_B, EPS_B).astype(np.float32)


class BCorrectionDeployWrapper:
    """Drop-in policy: frozen ACT + 逐帧实时 δa 校正（IMU 在环）。"""

    def __init__(self, act_policy, pi_ckpt, device, R, b, mode="full",
                 lag=3, ff_lag=2, ff_blend=4, log_path=None, verbose=True,
                 probe_seed=0, probe_mrad=3.0):
        assert mode in MODES, f"mode 须为 {MODES}"
        self.act = act_policy
        self.device = device
        self.mode = mode
        self.pi_ckpt = pi_ckpt
        self.verbose = verbose
        self.use_pi = mode in ("full", "pi_only", "record")
        self.apply_delta = mode in ("full", "pi_only", "probe")   # record 只观测不施加
        self.use_ff = mode in ("full", "ff")
        # probe: 外生激励, **不读 IMU、不经 π** —— 序列由种子完全决定
        self.probe_seed, self.probe_mrad = int(probe_seed), float(probe_mrad)
        self.probe = make_probe(probe_seed, probe_mrad) if mode == "probe" else None

        self.pi = load_tcn(pi_ckpt).to(device) if self.use_pi else None
        self.renderer = IdealChunkRenderer() if self.use_pi else None
        self.asm = FeatureAssembler(R, b, L=lag) if self.use_pi else None
        self.ff_lag, self.ff_blend = ff_lag, ff_blend

        self.n_action_steps = act_policy.config.n_action_steps
        self._queue = deque(maxlen=self.n_action_steps)
        self._chunk_raw = None          # 本 chunk 原始 ACT 指令(未加 δa), 供 ideal 渲染
        self._prev_tail = None          # 上一 chunk 尾部(FF blend 用)
        self._k = 0                     # chunk 内帧号
        self.stats = {"frames": 0, "gated": 0, "sat": 0, "stale": 0, "nan": 0,
                      "da_abs_sum": 0.0, "render_ms": [], "infer_ms": []}
        self._log = [] if log_path else None
        self._chunks = [] if log_path else None
        self.log_path = Path(log_path) if log_path else None
        self._ep_count = 0                  # 多集自动加集号后缀
        if verbose:
            print(f"[b_corr] mode={mode} ε={EPS_B*1000:.0f}mrad L={lag} "
                  f"feat={FEAT_DIM} win={WIN} π={'on' if self.use_pi else 'off'} "
                  f"ff={'on' if self.use_ff else 'off'}"
                  + (f" probe=seed{self.probe_seed}/{self.probe_mrad:g}mrad"
                     f"(RMS {np.sqrt((self.probe**2).mean())*1e3:.2f}, "
                     f"clip {np.mean(np.abs(self.probe) >= EPS_B*0.999)*100:.1f}%)"
                     if self.probe is not None else ""))

    # ---- 透传 ACTPolicy 接口 ----
    def __getattr__(self, name):
        return getattr(self.__dict__["act"], name) if "act" in self.__dict__ else None

    def reset(self):
        # control_robot 每集开始调 reset() → 借此把上一集落盘并推进集号(多集独立文件)
        self._flush()
        if self._log:
            self._ep_count += 1
            self._log, self._chunks = [], []
            self.stats = {k: (0 if not isinstance(v, list) else [])
                          for k, v in self.stats.items()}
        self.act.reset()
        self._queue.clear()
        self._chunk_raw = None
        self._prev_tail = None
        self._k = 0
        if self._chunks is not None:
            self._chunks = []
        if self.asm is not None:
            self.asm.reset()

    def to(self, device):
        self.act.to(device)
        if self.pi is not None:
            self.pi.to(device)
        return self

    def eval(self):
        self.act.eval()
        return self

    def parameters(self):
        return self.act.parameters()

    # ---- 内部 ----
    def _state_rad(self, batch):
        """observation.state (1,7) mdeg → 6 关节角 rad（π 的 q̇ 特征用）。"""
        s = batch["observation.state"].detach().cpu().numpy().reshape(-1)
        return (s[:6] * _MDEG_TO_RAD).astype(np.float32)

    def _new_chunk(self, batch):
        """ACT 生成 chunk → (可选 FF) → 渲染 ideal → 喂给特征装配器。"""
        b = self.act.normalize_inputs(batch)
        if self.act.config.image_features:
            b = dict(b)
            b["observation.images"] = torch.stack(
                [b[k] for k in self.act.config.image_features], dim=-4)
        chunk = self.act.model(b)[0][:, : self.n_action_steps]
        chunk = self.act.unnormalize_outputs({"action": chunk})["action"]   # (1,T,7) raw
        cmd = chunk[0].detach().cpu().numpy().astype(np.float32)            # (T,7)
        if self._log is not None:            # 原始 ACT 指令(FF/δa 之前) — A/B 分析的 ideal 基准
            self._chunks.append({"start": self.stats["frames"], "raw": cmd.tolist()})

        if self.use_ff:
            cmd = apply_ff12(cmd, self._prev_tail, lag=self.ff_lag, blend=self.ff_blend)

        if self.use_pi:
            t0 = time.perf_counter()
            ideal = self.renderer.render(cmd, self._prev_tail)              # (T,6) sim系
            self.stats["render_ms"].append((time.perf_counter() - t0) * 1e3)
            self.asm.on_new_chunk(cmd, ideal)

        self._chunk_raw = cmd
        self._prev_tail = cmd.copy()
        self._k = 0
        return torch.from_numpy(cmd).to(self.device).unsqueeze(0)           # (1,T,7)

    def _delta(self, batch):
        """一帧的 δa(6,)。IMU 陈旧/异常 → 置零并计数（fail-safe）。"""
        try:
            imu6 = batch["observation.imu_real"].detach().cpu().numpy().reshape(-1)
        except (KeyError, AttributeError):
            self.stats["stale"] += 1
            return np.zeros(6, np.float32)
        if imu6.shape[0] != 6 or not np.isfinite(imu6).all():
            self.stats["stale"] += 1
            return np.zeros(6, np.float32)

        t0 = time.perf_counter()
        win, gate = self.asm.step(imu6.astype(np.float32), self._state_rad(batch))
        # 窗未填满(前 WIN 帧)不施加: 此时 TCN 的补零区受 conv bias 影响, 与训练时的
        # 全序列前向不一致(实测前 12 帧差 0.23mrad, 之后 9e-6mrad)。宁可不修。
        if self.asm.i <= WIN:
            self.stats["warmup"] = self.stats.get("warmup", 0) + 1
            return np.zeros(6, np.float32)
        with torch.no_grad():
            out = self.pi(torch.from_numpy(win).unsqueeze(0).to(self.device))
        self.stats["infer_ms"].append((time.perf_counter() - t0) * 1e3)
        da = out[0, -1].detach().cpu().numpy().astype(np.float32)           # 取窗末=当帧

        if not np.isfinite(da).all():
            self.stats["nan"] += 1
            return np.zeros(6, np.float32)
        # 静止 gate（与训练一致的判据）
        if gate < 0.04:
            self.stats["gated"] += 1
            da = np.zeros(6, np.float32)
        if np.abs(da).max() > 0.95 * EPS_B:
            self.stats["sat"] += 1
        return np.clip(da, -EPS_B, EPS_B)                                   # 冗余硬界

    @torch.no_grad()
    def select_action(self, batch):
        if len(self._queue) == 0:
            chunk = self._new_chunk(batch)
            self._queue.extend(chunk.transpose(0, 1))
        a = self._queue.popleft()                                           # (1,7)

        da = np.zeros(6, np.float32)
        if self.probe is not None:                       # 开环激励: 与状态无关
            da = self.probe[self.stats["frames"] % len(self.probe)]
            a = a.clone()
            a[0, :6] = a[0, :6] + torch.from_numpy(da).to(a.device)
        elif self.use_pi:
            da = self._delta(batch)
            if self.apply_delta:
                a = a.clone()
                a[0, :6] = a[0, :6] + torch.from_numpy(da).to(a.device)     # 夹爪不动
        self.stats["frames"] += 1
        self.stats["da_abs_sum"] += float(np.abs(da).mean())
        if self._log is not None:
            imu = batch.get("observation.imu_real")
            self._log.append({
                "k": self._k, "da": da.tolist(),
                "a": a[0].detach().cpu().numpy().tolist(),
                "imu": imu.detach().cpu().numpy().reshape(-1).tolist() if imu is not None else None,
                "state": batch["observation.state"].detach().cpu().numpy().reshape(-1).tolist()})
            if self.stats["frames"] % 20 == 0:      # 每 1s 增量落盘(不依赖有人调 summary)
                self._flush()
        self._k += 1
        return a

    def _summary_dict(self):
        s, n = self.stats, max(self.stats["frames"], 1)
        # ckpt/FF 状态必须落盘: C 臂那次没记, 事后无法确认用的是哪个 seed,
        # 也无法从日志本身判断 π 与 FF 的占比(2026-07-29)
        return {"mode": self.mode, "pi_ckpt": str(self.pi_ckpt),
                "probe_seed": self.probe_seed if self.probe is not None else None,
                "probe_mrad": self.probe_mrad if self.probe is not None else None,
                "use_ff": bool(self.use_ff), "use_pi": bool(self.use_pi),
                "ff_lag": int(self.ff_lag), "ff_blend": int(self.ff_blend),
                "ff_blend_effective": int(max(self.ff_blend - self.ff_lag, 0)),
                "frames": n, "ep": self._ep_count,
                "gated_pct": 100 * s["gated"] / n, "sat_pct": 100 * s["sat"] / n,
                "stale": s["stale"], "nan": s["nan"],
                "warmup": s.get("warmup", 0), "da_mean_mrad": 1e3 * s["da_abs_sum"] / n,
                "infer_ms_p95": float(np.percentile(s["infer_ms"], 95)) if s["infer_ms"] else 0.0,
                "render_ms_max": float(max(s["render_ms"])) if s["render_ms"] else 0.0}

    def _flush(self):
        """当前集落盘到 <stem>_ep<k>.<suf> (覆盖)。每 20 帧 + 每集 reset 时调用。"""
        if self.log_path is None or not self._log:
            return
        st = self.log_path
        p = st.with_name(f"{st.stem}_ep{self._ep_count}{st.suffix}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(
            {"summary": self._summary_dict(), "frames": self._log, "chunks": self._chunks}))

    def summary(self):
        self._flush()
        return self._summary_dict()
        return out


def build_b_correction_policy(act_policy, pi_ckpt, device, calib_json,
                              mode="full", log_path=None, ff_lag=None, ff_blend=None,
                              probe_seed=None, probe_mrad=None):
    """calib_json: 含 {"R": 3x3, "b": 3} 的标定文件（p2_labels/<tag>/manifest.json 同格式）。"""
    m = json.loads(Path(calib_json).read_text())
    R = np.array(m["R"], np.float32)
    b = np.array(m.get("b", [0, 0, 0]), np.float32)
    assert abs(np.linalg.det(R) - 1.0) < 1e-3, f"R 非旋转矩阵 (det={np.linalg.det(R):.4f})"
    # 2026-07-29: ff_lag/ff_blend 原写死。实测残余滞后 A/D=3 帧、B(lag=2)=1 帧
    # ⇒ lag=2 欠补偿 1 帧。且 lag 会**丢掉** blend 的前 lag 帧
    # ⇒ 有效 blend = blend − lag（实测 blend=4,lag=2 时只有 k0/k1 起作用）。
    # 两者不独立, 必须一起调, 故一并开放。
    kw = {}
    if ff_lag is not None:
        kw["ff_lag"] = int(ff_lag)
    if ff_blend is not None:
        kw["ff_blend"] = int(ff_blend)
    if probe_seed is not None:
        kw["probe_seed"] = int(probe_seed)
    if probe_mrad is not None:
        kw["probe_mrad"] = float(probe_mrad)
    return BCorrectionDeployWrapper(act_policy, pi_ckpt, device, R, b,
                                    mode=mode, log_path=log_path, **kw)
