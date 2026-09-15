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
    C' G-2 预测墙:  BCORR_CKPT=<pi_g2wall_v1.json> BCORR_MODE=g2wall
                   （`pi-g2wall-v1`, 自由参数 **0**; **不读 IMU**、**不含 FF**）
    D  纯 π 网络:   BCORR_CKPT=<pi_g2env_v1.pt> BCORR_MODE=pi_net
                   （`pi-g2env-v1`，在冻结 G-2 环境里训出；运行时**无 G-2/H/求解器**）
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

import hashlib
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
from lerobot.common.policies.act.step1_seam_pulse import (
    SeamPulseRuntime, SeamPulseTheta,
)
from lerobot.common.policies.act.step1_correction_runtime import (
    Step1CorrectionRuntime,
    Step1Frame,
)

_MDEG_TO_RAD = math.pi / 180.0 / 1000.0
_GRIP_SCALE = 1.0 / 800000.0        # 与 refine_deploy 一致
MODES = ("off", "ff", "pi_only", "full", "record", "probe",
         "step1_b", "step1_c", "step1_v2", "g2wall", "pi_net")


PROBE_N = 2000                   # 100 s @20Hz —— 覆盖 30 s 一集, 正常不会绕回


def probe_waveform(seed: int, n: int = PROBE_N, fs: float = 20.0,
                   lo: float = 0.5, hi: float = 9.0) -> np.ndarray:
    """**单位 RMS** 的带限白噪声波形 —— 只由 seed 决定, **与幅值无关**。

    与 make_probe 分开是为了满足预注册(PREREG_STEP1.md §8)的两条要求:
      · P3 复用 P1 的 seed 只放大幅值 ⇒ 波形必须逐点相同, 未触界处严格成比例
      · 波形 hash 用于事后证明"确实是同一段激励" ⇒ hash 必须与幅值无关
    """
    from scipy.signal import butter, filtfilt
    rng = np.random.default_rng(int(seed))
    x = rng.standard_normal((int(n), 6))
    ba = butter(3, [lo / (fs / 2), hi / (fs / 2)], "bandpass")
    x = filtfilt(*ba, x, axis=0)
    return (x / np.sqrt((x ** 2).mean())).astype(np.float32)


def probe_hash(seed: int, n: int = PROBE_N) -> str:
    """波形指纹(与幅值无关)。同 seed ⇒ 同 hash, 用于核对 P3 确实复用了 P1 的激励。"""
    import hashlib
    return hashlib.sha256(probe_waveform(seed, n).tobytes()).hexdigest()[:16]


def make_probe(seed: int, mrad: float, n: int = PROBE_N,
               fs: float = 20.0, lo: float = 0.5, hi: float = 9.0) -> np.ndarray:
    """带限白噪声 δa —— **与状态无关的外生激励**，开环系统辨识专用。

    为什么必须外生: 此前 D/E/F 三条臂的 δa 都由 π 从状态算出(闭环), 于是辨识出的
    "响应" H 里混着控制器 —— 实测 E 与 F(同日同机, 只差 FF 开关)的 H 互不通用,
    用 E 的 H 求解、F 上评估会从 −86.7% 翻成 +187.8%(reports/STEP1_GATE.md)。
    只有与状态无关的激励才能把被控对象与控制器分开。

    频带取 0.5–9 Hz(评分带), 六关节独立, 定种子可完全复现, 硬界 ε 兜底。
    幅值 = 单位波形 × mrad, **再**裁剪 ⇒ 未触界处 mrad 加倍即输出严格加倍。
    """
    w = probe_waveform(seed, n, fs, lo, hi)
    return np.clip(w * (float(mrad) * 1e-3), -EPS_B, EPS_B).astype(np.float32)


def parse_seed_schedule(spec, fallback: int) -> list[int]:
    """"1000,1001,..." → [1000, 1001, ...]。空则退回单 seed(旧行为)。

    **必须支持显式表**而非 base+ep_id: P2 要同 seed 重复两集("2000,2000,2001,..."),
    P3 要复用 P1 的 seed("1000,1001,1002,1003") —— 递增式生成表达不了这两种。
    """
    if spec is None or str(spec).strip() == "":
        return [int(fallback)]
    return [int(s) for s in str(spec).replace(" ", "").split(",") if s]


class BCorrectionDeployWrapper:
    """Drop-in policy: frozen ACT + 逐帧实时 δa 校正（IMU 在环）。"""

    def __init__(self, act_policy, pi_ckpt, device, R, b, mode="full",
                 lag=3, ff_lag=2, ff_blend=4, log_path=None, verbose=True,
                 probe_seed=0, probe_mrad=3.0, probe_seeds=None,
                 theta_path=None):
        assert mode in MODES, f"mode 须为 {MODES}"
        self.act = act_policy
        self.device = device
        self.mode = mode
        self.pi_ckpt = pi_ckpt
        self.verbose = verbose
        self.use_pi = mode in ("full", "pi_only", "record")
        self.use_step1 = mode in ("step1_b", "step1_c")
        # v2: 接缝脉冲, **不读 IMU、不用任何模型**(PREREG_STEP1_V2.md)
        self.use_seam = mode == "step1_v2"
        # g2wall: **G-2 预测墙 + H 解析求解**(pi-g2wall-v1, 自由参数 0)。
        # **不读 IMU**(纯前馈: 只吃本 chunk 的 ACT 指令)、**不含 FF**(用户 2026-09-15)。
        self.use_g2wall = mode == "g2wall"
        # pi_net: **纯 π 网络**(pi-g2env-v1)。运行时无 G-2/H/求解器/MuJoCo,
        # 一次前向出整 chunk ⇒ **没有空档**(g2wall 每 chunk 空 3-4 帧)。
        # **不读 IMU**(v1 纯前馈); FF 不在本线。
        self.use_pinet = mode == "pi_net"
        self.apply_delta = mode in ("full", "pi_only", "probe", "step1_b", "step1_c",
                                    "step1_v2", "g2wall", "pi_net")
        self.use_ff = mode in ("full", "ff")
        # probe: 外生激励, **不读 IMU、不经 π** —— 序列由种子完全决定
        # **逐集换 seed**(PREREG_STEP1.md §8): 旧实现整批只有 1 个噪声实现 ——
        # probe 表在 __init__ 一次性生成, 而 reset() 把 frames 归零 ⇒ 每集从第 0 帧重放。
        self.probe_mrad = float(probe_mrad)
        self.probe_seeds = parse_seed_schedule(probe_seeds, probe_seed)
        self.probe_seed = self.probe_seeds[0]
        self._probe_ep = 0            # 集号(独立于 _ep_count, 不依赖是否开日志)
        self._probe_i = 0             # 集内帧号(独立于 stats["frames"])
        self._probe_wrapped = 0       # 集长超出波形长度的次数(应为 0)
        self._probe_overrun = 0       # 集数超出 seed 表的次数(应为 0)
        self._probe_clamped = False   # 本集是否因超表而沿用了末位 seed
        self.probe = None
        if mode in ("probe",):
            self._load_probe()

        self.seam = None
        if self.use_seam:
            src = theta_path if theta_path else pi_ckpt
            if not src:
                raise ValueError("step1_v2 需要 θ 文件: 设 BCORR_THETA 或 BCORR_CKPT")
            self.theta_path = str(src)
            self.seam = SeamPulseRuntime(SeamPulseTheta.load(src), verbose=verbose)

        self.pi = load_tcn(pi_ckpt).to(device) if self.use_pi else None
        self.step1 = (Step1CorrectionRuntime(
            pi_ckpt, mode="B" if mode == "step1_b" else "C")
            if self.use_step1 else None)
        if self.step1 is not None:
            if not np.allclose(self.step1.artifact.R, R, atol=1e-6, rtol=0.0):
                raise ValueError("Step1 artifact 的 R 与 BCORR_CALIB 不一致")
            if not np.allclose(self.step1.artifact.b, b, atol=1e-6, rtol=0.0):
                raise ValueError("Step1 artifact 的 b 与 BCORR_CALIB 不一致")
            self._artifact_sha256 = hashlib.sha256(Path(pi_ckpt).read_bytes()).hexdigest()
        else:
            self._artifact_sha256 = None

        # ── g2wall: 启动期校验冻结件(H sha / G-2 W sha / 求解器常量) ──────────
        # 校验失败在这里**抛**(还没上机, 早失败好); 运行中的异常一律 δa 置零并记录。
        self.g2w = None
        if self.use_g2wall:
            from lerobot.common.policies.act.g2wall_runtime import G2WallRuntime
            if not pi_ckpt:
                raise ValueError("g2wall 需要 manifest: 设 BCORR_CKPT=…/pi_g2wall_v1.json")
            self.g2w = G2WallRuntime(pi_ckpt, R, b, verbose=verbose)
            self._artifact_sha256 = hashlib.sha256(Path(pi_ckpt).read_bytes()).hexdigest()
            # 标定差异只**报告**不阻拦: 冻结评价用的是 step1_H_v1 里的标定(pp2a-deploy),
            # 而部署应当用**当日本会话**的标定。已实测重拟 R 救不回换会话的损失
            # (ab2: 26.3% → 25.4%, 两 R 差 4.8° 属正常绑带漂移) ⇒ 这是泛化问题。
            try:
                frozen_R = np.asarray(json.loads(
                    (Path(__file__).resolve().parents[4]
                     / "outputs/step1_H/step1_H_v1.json").read_text()
                )["calibration"]["R"], np.float64)
                ang = np.degrees(np.arccos(np.clip(
                    (np.trace(frozen_R.T @ np.asarray(R, np.float64)) - 1) / 2, -1, 1)))
                self._calib_delta_deg = float(ang)
                if verbose:
                    print(f"[g2wall] 本次标定 R 与冻结评价所用 R 相差 **{ang:.2f}°** "
                          f"（只报告不阻拦；换会话的损失**不是**标定问题）")
            except Exception as exc:
                self._calib_delta_deg = None
                if verbose:
                    print(f"[g2wall] ⚠ 标定比对跳过: {type(exc).__name__}: {exc}")
            # **进程卸载**(默认开): G-2 与 ideal 渲染都是持 GIL 的逐帧 Python 循环,
            # 放线程里会把 chunk 边界附近的帧顶到 59.7-63.2 ms(门禁 60)。换进程后
            # 主进程只剩 pickle 几 KB + 逐帧 0.15 ms 求解。spawn 而非 fork(CUDA)。
            self._g2w_pool = None
            if os.environ.get("BCORR_G2WALL_PROC", "1") != "0":
                from lerobot.common.policies.act.g2wall_runtime import make_worker_pool
                self._g2w_pool = make_worker_pool(pi_ckpt, R, b,
                                                  self.g2w.prefix, self.g2w.n_a)
                if verbose and self._g2w_pool is not None:
                    print(f"[g2wall] 进程卸载 **开**（stage A/B 在子进程，不抢主线程 GIL）")

        # ── pi_net: 启动期校验检查点 ──────────────────────────────────────
        self.pinet = None
        if self.use_pinet:
            from lerobot.common.policies.act.pi_net_runtime import PiNetRuntime
            if not pi_ckpt:
                raise ValueError("pi_net 需要检查点: 设 BCORR_CKPT=…/pi_g2env_v1.pt")
            self.pinet = PiNetRuntime(pi_ckpt, device=device, verbose=verbose)
            self._artifact_sha256 = self.pinet.ckpt_sha256

        # ⚠ pi_net **不需要** renderer —— 运行时不跑 MuJoCo
        self.renderer = (IdealChunkRenderer()
                         if (self.use_pi or self.use_step1 or self.use_g2wall) else None)
        # ── chunk 渲染移出 20 Hz 关键路径（2026-09-14）─────────────────────
        # 原先 render 同步跑在 chunk 边界那一帧上，与 ACT 推理/FF/plan 挤同一个
        # 50 ms 预算。实测 MuJoCo **释放 GIL**（后台渲染时主线程只多 3 ms/202 ms），
        # 故改为后台线程；渲染未就绪的帧 δa 置零（保守，且只影响 chunk 头 1 帧）。
        from concurrent.futures import ThreadPoolExecutor
        self._render_pool = ThreadPoolExecutor(max_workers=1) if self.renderer else None
        self._pending = None            # (future, cmd) 或 None
        # g2wall 是**两段式**: A(渲 ideal + G-2 头段) 先落地, B(整 chunk) 随后回填。
        # 两段都在同一个单 worker 池上排队, 天然保证 A 先于 B。
        self.g2wall_defer = int(os.environ.get("BCORR_G2WALL_DEFER", 1))
        self._g2w_submit = None         # (cmd, prev) 待提交的 stage A
        self._pending_a = None          # (future, t0) 或 None —— 返回 ideal
        self._pending_b = None          # (future, t0) 或 None
        self._frame_t = None            # 上一帧的墙钟, 用于帧间隔统计
        self.asm = FeatureAssembler(R, b, L=lag) if self.use_pi else None
        self.ff_lag, self.ff_blend = ff_lag, ff_blend

        self.n_action_steps = act_policy.config.n_action_steps
        self._queue = deque(maxlen=self.n_action_steps)
        self._chunk_raw = None          # 本 chunk 原始 ACT 指令(未加 δa), 供 ideal 渲染
        self._prev_tail = None          # 上一 chunk 尾部(FF blend 用)
        self._k = 0                     # chunk 内帧号
        self.stats = {"frames": 0, "gated": 0, "sat": 0, "stale": 0, "nan": 0,
                      "da_abs_sum": 0.0, "render_ms": [], "infer_ms": [],
                      "plan_ms": []}
        self._last_step1_frame: Step1Frame | None = None
        self._last_g2w_frame = None
        self._last_pinet_frame = None
        self._log = [] if log_path else None
        self._chunks = [] if log_path else None
        self.log_path = Path(log_path) if log_path else None
        self._ep_count = 0                  # 多集自动加集号后缀
        if verbose:
            print(f"[b_corr] mode={mode} ε={EPS_B*1000:.0f}mrad L={lag} "
                  f"feat={FEAT_DIM} win={WIN} π={'on' if self.use_pi else 'off'} "
                  f"ff={'on' if self.use_ff else 'off'} "
                  f"step1={'B' if mode == 'step1_b' else ('C' if mode == 'step1_c' else 'off')}"
                  + (" seam=on(v2: 运行时不读 IMU)" if self.use_seam else "")
                  + (" g2wall=on(**不读 IMU**, 纯前馈; FF 不在本线)" if self.use_g2wall else "")
                  + (" pi_net=on(**纯 π 网络**; 无 G-2/H/求解器; 无空档)" if self.use_pinet else "")
                  + (f" probe: {len(self.probe_seeds)} seeds "
                     f"{self.probe_seeds[:4]}{'…' if len(self.probe_seeds) > 4 else ''} "
                     f"@{self.probe_mrad:g}mrad" if self.probe is not None else ""))

    # ---- 透传 ACTPolicy 接口 ----
    def __getattr__(self, name):
        return getattr(self.__dict__["act"], name) if "act" in self.__dict__ else None

    def _load_probe(self):
        """按当前集号取 seed 表里的 seed, 生成本集波形。"""
        i = self._probe_ep
        # 超表只**标记**, 待本集真的产生帧时才计数 —— 否则末集之后的那次 reset()
        # 会多记一次(集号已推进但那一集根本没跑)。
        self._probe_clamped = i >= len(self.probe_seeds)
        if self._probe_clamped:
            i = len(self.probe_seeds) - 1
        self.probe_seed = self.probe_seeds[i]
        self.probe = make_probe(self.probe_seed, self.probe_mrad)
        self._probe_i = 0
        if self.verbose:
            print(f"[b_corr] ep{self._probe_ep}: probe seed={self.probe_seed} "
                  f"mrad={self.probe_mrad:g} hash={probe_hash(self.probe_seed)} "
                  f"RMS={np.sqrt((self.probe**2).mean())*1e3:.3f}mrad "
                  f"clip={np.mean(np.abs(self.probe) >= EPS_B*0.999)*100:.2f}%")

    def reset(self):
        # control_robot 每集开始调 reset() → 借此把上一集落盘并推进集号(多集独立文件)
        # **顺序要紧**: 先 flush(写的是刚跑完那一集, 必须带**那一集**的 seed),
        # 再推进集号并换波形。反过来会把下一集的 seed 写进上一集的日志。
        had_frames = self.stats["frames"] > 0
        self._flush()
        if self.probe is not None and had_frames:
            self._probe_ep += 1
            self._load_probe()
        if self._log:
            self._ep_count += 1
            self._log, self._chunks = [], []
        if had_frames:
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
        if self.seam is not None:
            self.seam.reset()          # 每集清零, 不跨 episode 继承(v2 §2.5)
        if self.step1 is not None:
            # 与是否开日志无关：z/HP/H 历史绝不能跨 episode 继承。
            self.step1.reset()
        if self.pinet is not None:
            self.pinet.reset()
        if self.g2w is not None:
            # resp(已提交 δa 的未来响应)与热身前缀绝不能跨 episode 继承。
            self.g2w.reset()
        self._pending_a = self._pending_b = self._g2w_submit = None
        self._last_step1_frame = None
        self._last_g2w_frame = None
        self._last_pinet_frame = None

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

        if self.seam is not None:
            # 用**原始 ACT 指令**算接缝差 —— 跳变是 ACT 输出的性质, 与我们自己加的 δa 无关
            self.seam.set_chunk(cmd)

        if self.use_ff:
            cmd = apply_ff12(cmd, self._prev_tail, lag=self.ff_lag, blend=self.ff_blend)

        if self.use_pi or self.use_step1:
            # **不在本帧渲染** —— 提交后台, 由 `_drain_pending()` 在就绪后装配。
            prev = None if self._prev_tail is None else self._prev_tail.copy()
            t0 = time.perf_counter()
            fut = self._render_pool.submit(self.renderer.render, cmd, prev)
            self._pending = (fut, cmd, t0)

        if self.use_g2wall:
            # 同步只做便宜的三件事(建计划 / 接触门控 / 帧号归零)。
            # ⚠ 帧号必须**此刻**归零: 后台没落地的那几帧 `step()` 仍照常推进 `_k`,
            #   若等 stage A 落地再归零, 空档帧会被当成 chunk 内偏移 ⇒ δa 整条错位。
            self.g2w.begin_chunk(cmd)
            prev = None if self._prev_tail is None else self._prev_tail.copy()
            # ⚠ **不在本帧提交** stage A。后台的 G-2 前向是持 GIL 的 Python 循环
            # （`kin_gyro_fs`/`_sim` 逐帧调 mujoco），会**抢主线程时间** —— 而
            # chunk 边界这一帧恰好最忙（ACT 推理在这里）。实测 ACT=40 ms 时
            # 该帧到 **63.2 ms**，超 60 ms 门禁（`sim/smoke_g2wall.py` 检 5b）。
            # 推迟 `g2wall_defer` 帧提交：空档不增（提交晚 1 帧但落地仍在同一帧），
            # 却把争抢挪到只做 0.14 ms 求解的空闲帧上。
            self._g2w_submit = (cmd, prev)
            self._pending_a = self._pending_b = None

        if self.use_pinet:
            # **同步**一次前向（实测 <2 ms）⇒ 第 0 帧就有 δa，没有空档
            self.pinet.begin_chunk(cmd)

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

    def _delta_step1(self, batch):
        """Step1 B/C 一帧。IMU 异常时仍消费该帧，但跳过 LMS 更新。"""
        imu6 = None
        try:
            value = batch["observation.imu_real"].detach().cpu().numpy().reshape(-1)
            if value.shape == (6,) and np.isfinite(value).all():
                imu6 = value.astype(np.float32)
            else:
                self.stats["stale"] += 1
        except (KeyError, AttributeError):
            self.stats["stale"] += 1

        t0 = time.perf_counter()
        frame = self.step1.step(imu6)
        self.stats["infer_ms"].append((time.perf_counter() - t0) * 1e3)
        self._last_step1_frame = frame
        saturated = frame.b_saturated if self.mode == "step1_b" else frame.c_saturated
        self.stats["sat"] += int(saturated)
        self.stats["gated"] += int(frame.contact_gated)
        if not np.isfinite(frame.delta).all():
            self.stats["nan"] += 1
            return np.zeros(6, np.float32)
        return frame.delta

    def _submit_g2wall(self) -> None:
        """到点后把 stage A 递给后台（推迟 `g2wall_defer` 帧，见 `_new_chunk` 的注释）。"""
        if self._g2w_submit is None or self.g2w._k < self.g2wall_defer:
            return
        cmd, prev = self._g2w_submit
        self._g2w_submit = None
        t0 = time.perf_counter()
        if self._g2w_pool is not None:
            from lerobot.common.policies.act.g2wall_runtime import _worker_build
            pre = self.g2w._prefix
            self._pending_a = (self._g2w_pool.submit(
                _worker_build, cmd, prev, pre, min(self.g2w.n_a, len(cmd))), t0, cmd, prev, pre)
        else:
            def _stage_a(c=cmd, pv=prev):
                ideal = self.renderer.render(c, pv)  # 必须**整 chunk**(filtfilt 非因果)
                self.g2w.fill_stage_a(ideal)
                return ideal
            self._pending_a = (self._render_pool.submit(_stage_a), t0)

    def _drain_g2wall(self) -> None:
        """非阻塞收割两段后台。**不返回 waiting** —— g2wall 空档期靠 `serve_to=0`
        让 `step()` 自然输出 0，同时照常推进 `_k` 与 `resp`。"""
        if self._pending_a is not None and self._pending_a[0].done():
            fut, t0 = self._pending_a[0], self._pending_a[1]
            self.stats["render_ms"].append((time.perf_counter() - t0) * 1e3)
            try:
                if self._g2w_pool is not None:
                    from lerobot.common.policies.act.g2wall_runtime import (
                        _worker_build, apply_stage)
                    e0, m = fut.result()
                    apply_stage(self.g2w, e0, m, "A")
                    _, _, cmd, prev, pre = self._pending_a
                    self._pending_b = (self._g2w_pool.submit(
                        _worker_build, cmd, prev, pre, len(cmd)), time.perf_counter())
                else:
                    ideal = fut.result()
                    self._pending_b = (self._render_pool.submit(
                        self.g2w.fill_stage_b, ideal), time.perf_counter())
            except Exception as exc:                       # 运行中**不抛**
                self.stats["nan"] += 1
                if self.verbose:
                    print(f"[g2wall] ⚠ stage A 后台任务异常 → 本 chunk δa 保持 0: "
                          f"{type(exc).__name__}: {exc}")
            self._pending_a = None
        if self._pending_b is not None and self._pending_b[0].done():
            try:
                r = self._pending_b[0].result()
                if self._g2w_pool is not None:
                    from lerobot.common.policies.act.g2wall_runtime import apply_stage
                    apply_stage(self.g2w, r[0], r[1], "B")
            except Exception as exc:
                self.stats["nan"] += 1
                if self.verbose:
                    print(f"[g2wall] ⚠ stage B 后台任务异常 → 沿用 stage A: "
                          f"{type(exc).__name__}: {exc}")
            self._pending_b = None

    def _drain_pending(self) -> bool:
        """后台 ideal 就绪则装配。→ 是否**仍在等待**（等待中 δa 必须置零）。"""
        if self._pending is None:
            return False
        fut, cmd, t0 = self._pending
        if not fut.done():
            self.stats["render_pending_frames"] = \
                self.stats.get("render_pending_frames", 0) + 1
            return True
        ideal = fut.result()
        self.stats["render_ms"].append((time.perf_counter() - t0) * 1e3)
        if self.use_pi:
            self.asm.on_new_chunk(cmd, ideal)
        if self.use_step1:
            t1 = time.perf_counter()
            self.step1.set_chunk(cmd, ideal)
            self.stats["plan_ms"].append((time.perf_counter() - t1) * 1e3)
        self._pending = None
        return False

    @torch.no_grad()
    def select_action(self, batch):
        # 帧间隔统计（部署门禁看的是**这个**，不是后台渲染本身的耗时）
        _now = time.perf_counter()
        if self._frame_t is not None:
            self.stats.setdefault("frame_ms", []).append((_now - self._frame_t) * 1e3)
        self._frame_t = _now
        if len(self._queue) == 0:
            chunk = self._new_chunk(batch)
            self._queue.extend(chunk.transpose(0, 1))
        a = self._queue.popleft()                                           # (1,7)

        da = np.zeros(6, np.float32)
        if self.probe is not None:                       # 开环激励: 与状态无关
            if self._probe_i == 0 and self._probe_clamped:   # 本集确实跑了且超表
                self._probe_overrun += 1
                print(f"[b_corr] ⚠ 第 {self._probe_ep} 集超出 seed 表长度 "
                      f"{len(self.probe_seeds)} —— 沿用末位 seed {self.probe_seed}, "
                      f"已计入 probe_overrun。这是**协议偏离**, 分析时须剔除或说明。")
            if self._probe_i >= len(self.probe):         # 集长超出波形 —— 正常不该发生
                self._probe_wrapped += 1
            da = self.probe[self._probe_i % len(self.probe)]
            self._probe_i += 1
            a = a.clone()
            a[0, :6] = a[0, :6] + torch.from_numpy(da).to(a.device)
        elif self.seam is not None:                      # v2 接缝脉冲: 不读 IMU
            da = self.seam.step()
            a = a.clone()
            a[0, :6] = a[0, :6] + torch.from_numpy(da).to(a.device)
        elif self.use_pinet:
            fr = self.pinet.step()
            self._last_pinet_frame = fr
            da = fr.delta
            if self.apply_delta and np.any(da):
                a = a.clone()
                a[0, :6] = a[0, :6] + torch.from_numpy(da).to(a.device)   # 夹爪不动
        elif self.use_g2wall:
            # 纯前馈: **不读 IMU**。每帧都必须调 step() —— 空档期它输出 0 但会
            # 推进 `_k` 与 `resp`(已提交 δa 的未来响应), 少调一次整条就错位。
            self._submit_g2wall()
            self._drain_g2wall()
            fr = self.g2w.step()
            self._last_g2w_frame = fr
            da = fr.delta
            if fr.stage == "none":
                self.stats["g2w_blackout"] = self.stats.get("g2w_blackout", 0) + 1
            if self.apply_delta and np.any(da):
                a = a.clone()
                a[0, :6] = a[0, :6] + torch.from_numpy(da).to(a.device)     # 夹爪不动
        elif self.use_pi or self.use_step1:
            # 后台 ideal 未就绪 ⇒ δa 置零（保守）。渲染实测 30 ms，故通常只影响
            # 每个 chunk 的头 1 帧；这一帧由 FF 的 blend 覆盖，不留空档。
            waiting = self._drain_pending()
            da = (np.zeros(6, np.float32) if waiting else
                  (self._delta_step1(batch) if self.use_step1 else self._delta(batch)))
            if self.apply_delta and not waiting:
                a = a.clone()
                a[0, :6] = a[0, :6] + torch.from_numpy(da).to(a.device)     # 夹爪不动
        self.stats["frames"] += 1
        self.stats["da_abs_sum"] += float(np.abs(da).mean())
        if self._log is not None:
            imu = batch.get("observation.imu_real")
            row = {
                "k": self._k, "da": da.tolist(),
                "a": a[0].detach().cpu().numpy().tolist(),
                "imu": imu.detach().cpu().numpy().reshape(-1).tolist() if imu is not None else None,
                "state": batch["observation.state"].detach().cpu().numpy().reshape(-1).tolist()}
            if self._last_pinet_frame is not None:
                pf = self._last_pinet_frame
                row.update({"pi_stage": pf.stage, "pi_k": pf.k,
                            "pi_contact": pf.contact_gated, "pi_sat": pf.saturated})
            if self._last_g2w_frame is not None:
                gf = self._last_g2w_frame
                row.update({"g2w_stage": gf.stage, "g2w_k": gf.k,
                            "g2w_horizon": gf.horizon,
                            "g2w_contact": gf.contact_gated,
                            "g2w_sat": gf.saturated,
                            "e0hat": None if gf.e0hat is None else gf.e0hat.tolist()})
            if self._last_step1_frame is not None:
                sf = self._last_step1_frame
                row.update({
                    "da_preclip": sf.raw_delta_b.tolist(),
                    "da_b": sf.delta_b.tolist(),
                    "z_used": sf.z_used,
                    "z_next": sf.z_next,
                    "contact_gate": sf.contact_gated,
                    "lms_update": sf.lms_update_mask,
                    "gap_raw": sf.raw_gap.tolist() if sf.raw_gap is not None else None,
                    "p_raw": (sf.predicted_response.tolist()
                              if sf.predicted_response is not None else None),
                })
            self._log.append(row)
            if self.stats["frames"] % 20 == 0:      # 每 1s 增量落盘(不依赖有人调 summary)
                self._flush()
        self._k += 1
        return a

    def _summary_dict(self):
        s, n = self.stats, max(self.stats["frames"], 1)
        # ckpt/FF 状态必须落盘: C 臂那次没记, 事后无法确认用的是哪个 seed,
        # 也无法从日志本身判断 π 与 FF 的占比(2026-07-29)
        pr = self.probe is not None
        seam = self.seam.summary() if self.seam is not None else {}
        out = {"mode": self.mode, "pi_ckpt": str(self.pi_ckpt),
                **({"theta": seam["theta"], "theta_path": self.theta_path,
                    "seam_corrected_pct": seam["corrected_pct"],
                    "seam_contact_gated_pct": seam["contact_gated_pct"],
                    "seam_first_chunk_skipped": seam["first_chunk_skipped"],
                    "seam_chunks": seam["chunks"]} if self.seam is not None else {}),
                # 逐集落盘**实际用的** seed 与波形 hash —— 事后核对 P3 是否真复用了 P1 激励
                "probe_seed": self.probe_seed if pr else None,
                "probe_hash": probe_hash(self.probe_seed) if pr else None,
                "probe_mrad": self.probe_mrad if pr else None,
                "probe_ep": self._probe_ep if pr else None,
                "probe_seed_schedule": self.probe_seeds if pr else None,
                "probe_rms_mrad": float(np.sqrt((self.probe ** 2).mean()) * 1e3) if pr else None,
                "probe_clip_pct": float(np.mean(np.abs(self.probe) >= EPS_B * 0.999) * 100) if pr else None,
                "probe_wrapped": self._probe_wrapped, "probe_overrun": self._probe_overrun,
                "use_ff": bool(self.use_ff), "use_pi": bool(self.use_pi),
                "ff_lag": int(self.ff_lag), "ff_blend": int(self.ff_blend),
                "ff_blend_effective": int(max(self.ff_blend - self.ff_lag, 0)),
                "frames": n, "ep": self._ep_count,
                "gated_pct": 100 * s["gated"] / n, "sat_pct": 100 * s["sat"] / n,
                "stale": s["stale"], "nan": s["nan"],
                "warmup": s.get("warmup", 0), "da_mean_mrad": 1e3 * s["da_abs_sum"] / n,
                "infer_ms_p95": float(np.percentile(s["infer_ms"], 95)) if s["infer_ms"] else 0.0,
                "render_ms_max": float(max(s["render_ms"])) if s["render_ms"] else 0.0,
                "plan_ms_max": float(max(s.get("plan_ms", []))) if s.get("plan_ms") else 0.0,
                "use_step1": bool(self.use_step1),
                "artifact_sha256": self._artifact_sha256}
        if self.step1 is not None:
            out["step1"] = self.step1.summary()
        if self.pinet is not None:
            out["pi_net"] = self.pinet.summary()   # ⚠ summary 必须**无副作用**
        if self.g2w is not None:
            out["g2wall"] = self.g2w.summary()
            out["g2wall"]["blackout_frames"] = int(s.get("g2w_blackout", 0))
            out["g2wall"]["blackout_pct"] = 100.0 * s.get("g2w_blackout", 0) / n
            out["g2wall"]["calib_delta_deg"] = getattr(self, "_calib_delta_deg", None)
            out["g2wall"]["defer_frames"] = self.g2wall_defer
            out["g2wall"]["process_offload"] = self._g2w_pool is not None
            out["g2wall"]["stage_a_wall_ms_max"] = (
                float(max(s["render_ms"])) if s["render_ms"] else 0.0)
        return out

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
                              probe_seed=None, probe_mrad=None, probe_seeds=None,
                              theta_path=None):
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
    if probe_seeds is not None:
        kw["probe_seeds"] = probe_seeds
    if theta_path is not None:
        kw["theta_path"] = theta_path
    return BCorrectionDeployWrapper(act_policy, pi_ckpt, device, R, b,
                                    mode=mode, log_path=log_path, **kw)
