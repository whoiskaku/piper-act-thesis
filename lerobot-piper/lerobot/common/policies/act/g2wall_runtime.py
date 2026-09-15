"""`pi-g2wall-v1` 的真机运行时 —— **G-2 预测墙 + H 解析求解**，自由参数 **0** 个。

结构
----
    ê₀[k] = calib(ω̂_G2(u_chunk))[k+ALIGN_L] − ideal[k]      ← G-2 预测「会错多少」
    δa[k] = clip( −(HᵀH+λI)⁻¹Hᵀ·(ê₀[k:k+K] + resp) , ±ε )，接触窗归零
            K = min(M, T−1−k)                               ← chunk 钳位

冻结件：G-2 原始 v=0（`rq_fir` + `rs_fir`）、H = `step1-H-v1`。
本模块**不含任何可训练参数**，只是把冻结件按部署时序跑起来。

为什么是两段式后台（这一条决定了能不能用）
------------------------------------------
实测（`sim/verify_g2wall_deploy.py`，ab4 P0）：

| chunk 内位置 | 占墙的带内能量 | 帧数占比 | 平均能量 |
|---|---|---|---|
| **第 0–4 帧** | **52.2%** | 4% | **13.05×** |
| 第 4–8 帧 | 8.2% | 4% | 2.06× |
| 其余 92 帧 | 39.6% | 92% | ≤0.67× |

⇒ **墙的能量一半在 chunk 头 4 帧**（与已记录的「边界跳变 → IMU 瞬态 2×」一致）。
于是后台延迟的代价极其陡峭：

    空档 0 帧 15.50%  ·  1 帧 11.88%(77%)  ·  2 帧 9.24%(60%)  ·  4 帧 7.63%(49%)

单帧 50 ms 的延迟就要付 23%。故**不能**把整个 chunk 的 G-2 前向放后台等：

  · **stage A**（同步，chunk 边界那一帧）：只算**头 `N_A=32` 帧** + 热身前缀 `P=20`，
    足以给出帧 0..`W_A−1` 的 δa（`W_A = N_A − M − ALIGN_L = 14`）。
    模型缓存后实测数十 ms 量级。
  · **stage B**（后台）：整 chunk 前向，回填帧 `W_A` 之后。它有 14 帧(700 ms)的余裕。

⚠ stage A 与 stage B 在帧 0..`W_A−1` 上**数值相同**（FIR 因果、中心差分只 ±1 帧、
前缀相同）—— 这一点由 `sim/verify_g2wall_runtime.py` 实测核对，不靠推断。

为什么需要热身前缀 `P`
----------------------
G-2 的伺服 rollout 与 FIR 历史在 chunk 边界会被截断。接上上一 chunk 尾部 20 帧
指令后，逐 chunk 前向相对整集前向只掉 **−1.42 pp**（16.92% → 15.50%）；
`P=10` 已基本够，`P≥20` 饱和。
⚠ chunk 内的 ±1 帧前瞻**合法**（整 chunk 指令在 t₀ 全部已知，与 ideal 渲染同理）；
被修的只是**跨 chunk 的状态延续**。

⚠ 已知风险（必须随结果报告）
----------------------------
**换会话掉 25.8 pp**：ab4 P0 裸 52.1% → ab2 A 裸 26.3%，且已证**不是标定问题**
（在 ab2 上独立重拟 R 只把 26.3% 动到 25.4%，而两个 R 差 4.8° 属正常绑带漂移）
⇒ **上机当天重算标定修不了**，这是泛化问题。

⚠ 以上全部是**离线**数字，被控对象是 H（模型），**不是真机验证**。
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lerobot.common.policies.act.step1_correction_runtime import (
    calibrate_ideal_imu,
    command_contact_mask,
)

# 冻结常量（与 sim/fit_step1_candidate.py、sim/delivery_g2_predictor.py 一致）
ALIGN_L = 3
M = 15
N_LAGS = M + 1
EPS = 0.010                 # **瞬时峰值**硬界（非 RMS）
KMULT = 1.0
FPS = 20.0

PREFIX = 20                 # 热身前缀默认值（实测 P≥20 饱和，P=10 已基本够）
N_A = 32                    # stage A 前向的 chunk 头长度默认值
# stage A 的 ê₀ **尾部有一段不可用**：ω̂ 里 `np.gradient`(dq→ddq)、`kin_gyro_fs` 的
# 中心差分都会在序列末端退化成单边差分，再叠上 ê₀ 的 `k+ALIGN_L` 钳位。
# 实测（`sim/verify_g2wall_runtime.py` 检 4）：32 帧的 stage A 中**末 7 帧**与整 chunk
# 的 stage B 不同（第 25 帧起分叉）。取 8 留一帧余量。
# ⚠ 这一段**不是**「不能输出 δa 的帧」，而是「求解器不能**读到**的帧」——
#   帧 k 会读到 ê₀[k+M]，所以可服务帧数要再减 M。
TAIL_GUARD = 8
W_A = N_A - M - TAIL_GUARD  # = 9，stage A 能可靠供给的帧数
# 接触门控常量 —— 与 sim/fit_step1_candidate.py 的 GRIP_FULL / CONTACT_* 必须一致
GRIP_FULL = 0.035
CONTACT_THRESHOLD_RATIO = 0.02
CONTACT_HALF_WINDOW = 10

_REPO = Path(__file__).resolve().parents[4]
_SIM_PATHS = ("sim", "sim/step4_gvnext", "sim/step2_g", "sim/step3_canobs")


class G2WallError(RuntimeError):
    """启动期校验失败 —— 只在 `__init__` 抛出，**运行中绝不抛**。"""


def _ensure_sim_path() -> None:
    for rel in _SIM_PATHS:
        p = str(_REPO / rel)
        if p not in sys.path:
            sys.path.insert(0, p)
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))


def install_model_cache(share: bool = True) -> None:
    """把 `mujoco.MjModel.from_xml_path` 换成**缓存解析**。

    G-2 的前向里 XML 被解析 **3 次/chunk**（`_sim` 一次、`kin_gyro_fs` 两次），
    每次实测 ~58 ms，占了绝大部分开销。

    - `share=False`：每次发独立深拷贝（~6 ms/次）。**永远安全**。
    - `share=True`（默认）：发同一个对象。安全性依据是**两条都成立**：
      `kin_gyro_fs` 根本不改模型（只写 `MjData` 的 `qpos/qvel`）；
      `_sim` 会把 θ 写进模型，但部署只走 `predict_qpos`，写的**永远是同一组冻结 θ**
      ⇒ 该写入**幂等**。
      ⚠ 因此 `predict_with_theta`（候选评估专用，会写入**别的** θ）
      **绝不能**与本缓存共存 —— `G2WallRuntime` 不调它，且下面直接把它禁掉。

    ⚠ 无论哪种模式，数值与原路径**逐位一致**，由 `sim/verify_g2wall_runtime.py` 检 1 核对。
    """
    import mujoco

    if getattr(mujoco.MjModel.from_xml_path, "_g2wall_cached", None) == share:
        return
    orig = getattr(mujoco.MjModel.from_xml_path, "_g2wall_orig",
                   mujoco.MjModel.from_xml_path)
    cache: dict[str, object] = {}

    def cached(path, *a, **kw):
        if a or kw:                       # 非默认调用一律走原路径，不进缓存
            return orig(path, *a, **kw)
        key = str(path)
        if key not in cache:
            cache[key] = orig(path)
        return cache[key] if share else copy.deepcopy(cache[key])

    cached._g2wall_cached = share          # type: ignore[attr-defined]
    cached._g2wall_orig = orig             # type: ignore[attr-defined]
    mujoco.MjModel.from_xml_path = cached  # type: ignore[assignment]

    if share:                              # 共享模式下堵死会写别的 θ 的入口
        _ensure_sim_path()
        try:
            import g_q_v1 as _GQ

            def _blocked(*a, **kw):
                raise G2WallError(
                    "共享模型缓存下禁止 predict_with_theta（会把**别的** θ 写进"
                    "共享模型）。需要它请先 install_model_cache(share=False)。")

            _GQ.predict_with_theta = _blocked
        except Exception:                  # 导不进来就不必堵
            pass


@dataclass
class G2WallFrame:
    """一帧的诊断记录（落盘用）。"""

    delta: np.ndarray
    k: int
    horizon: int
    contact_gated: bool
    saturated: bool
    stage: str                  # "A" / "B" / "none"
    e0hat: np.ndarray | None


class _ChunkPlan:
    """⚠ `valid_to` 与 `serve_to` 是**两件事**，混用会静默改掉求解器的 horizon。

    - `valid_to`：ê₀ 在 `[0, valid_to)` 上**数值有效** —— 求解器取 horizon 时用它钳位。
    - `serve_to`：允许**输出 δa** 的帧数。stage A 只输出 `W_A` 帧，
      但它算出的 ê₀ 一直有效到 `N_A`，正好够帧 `W_A−1` 看满 `M+ALIGN_L` 的 horizon。

    （第一版把两者写成同一个字段 ⇒ stage A 期间 horizon 被钳到第 `W_A−1` 帧，
    逐帧 δa 与离线口径差到 3.42 mrad。由 `verify_g2wall_runtime` 检 5 抓出。）
    """

    __slots__ = ("e0hat", "contact", "T", "valid_to", "serve_to", "stage")

    def __init__(self, T: int):
        self.e0hat = np.zeros((T, 3))
        self.contact = np.zeros(T, bool)
        self.T = T
        self.valid_to = 0
        self.serve_to = 0
        self.stage = "none"


class G2WallRuntime:
    """逐 chunk 前向 G-2 → 逐帧解析求解 δa。**运行中不抛异常**（异常输入 → δa=0 并计数）。"""

    def __init__(self, manifest_path: str | Path, R: np.ndarray, b: np.ndarray,
                 verbose: bool = True, prefix: int = PREFIX, n_a: int = N_A,
                 share_model: bool = True):
        _ensure_sim_path()
        install_model_cache(share=share_model)
        self.prefix = int(prefix)
        self.n_a = int(n_a)
        self.w_a = self.n_a - M - TAIL_GUARD
        if self.w_a < 1:
            raise G2WallError(
                f"N_A={n_a} 太小: W_A = N_A−M−TAIL_GUARD = {self.w_a} < 1")
        self.manifest_path = Path(manifest_path)
        try:
            man = json.loads(self.manifest_path.read_text())
        except Exception as exc:                       # 启动期，可抛
            raise G2WallError(f"读不到 manifest {manifest_path}: {exc}") from exc
        if man.get("artifact") != "pi-g2wall-v1":
            raise G2WallError(f"manifest artifact 应为 pi-g2wall-v1, 实为 {man.get('artifact')}")
        self.manifest = man
        comp = man["components"]

        # ── H（被控对象模型）── sha 必须对上
        h_path = _REPO / "outputs/step1_H/step1_H_v1.npz"
        try:
            self.H = np.ascontiguousarray(np.load(h_path)["H"], float)
        except Exception as exc:
            raise G2WallError(f"读不到 H {h_path}: {exc}") from exc
        h_sha = hashlib.sha256(self.H.tobytes()).hexdigest()
        if h_sha != comp["H"]["sha256"]:
            raise G2WallError(f"H sha256 不符: {h_sha[:16]}… ≠ {comp['H']['sha256'][:16]}…")
        if self.H.shape != (N_LAGS, 3, 6):
            raise G2WallError(f"H 形状应为 {(N_LAGS, 3, 6)}, 实为 {self.H.shape}")

        # ── G-2（原始冻结 v=0）── W sha 必须对上 manifest
        rq = _REPO / "outputs/step4_gvnext/rq"
        rs = _REPO / "outputs/step4_gvnext/rs"
        try:
            zs = np.load(rs / "rs_fir.npz")
            self.W_rs = np.ascontiguousarray(zs["W"], float)
            self.nlag_rs = int(zs["nlag"])
            self.zq = np.load(rq / "rq_fir.npz")
            self.sq = json.loads((rq / "select.json").read_text())
        except Exception as exc:
            raise G2WallError(f"读不到 G-2 冻结件: {exc}") from exc
        w_sha = hashlib.sha256(self.W_rs.tobytes()).hexdigest()
        if w_sha != comp["G2"]["W_sha256"]:
            raise G2WallError(f"G-2 W sha256 不符: {w_sha[:16]}… ≠ "
                              f"{comp['G2']['W_sha256'][:16]}…")

        # ── 求解器常量 ── 与 manifest 一致
        sol = comp["solver"]
        if int(sol["align_L"]) != ALIGN_L or abs(float(sol["eps_rad"]) - EPS) > 1e-12 \
                or abs(float(sol["lambda_K"]) - KMULT) > 1e-12:
            raise G2WallError(f"求解器常量与 manifest 不一致: {sol}")
        self.R = np.asarray(R, np.float32)
        self.b = np.asarray(b, np.float32)
        hs_full = self.H.reshape(N_LAGS * 3, 6)
        self.lam = KMULT * max(float(np.trace(hs_full.T @ hs_full)) / 6.0,
                               np.finfo(float).tiny)
        self._Hs = [self.H[:k + 1].reshape((k + 1) * 3, 6) for k in range(N_LAGS)]
        self._chol = [np.linalg.inv(h.T @ h + self.lam * np.eye(6)) @ h.T
                      for h in self._Hs]          # 预分解，逐帧只剩一次矩阵乘
        self.verbose = verbose
        self.reset()
        # 预热：首次 import/解析发生在这里，而不是第一帧
        self._warmup()
        if verbose:
            print(f"[g2wall] pi-g2wall-v1 ✓  自由参数 0  ε={EPS*1e3:.0f}mrad(峰值) "
                  f"P={self.prefix} N_A={self.n_a} W_A={self.w_a}  H sha {h_sha[:8]}… "
                  f"G2 sha {w_sha[:8]}…")

    # ---- G-2 前向 ----
    def _omega(self, cmd: np.ndarray) -> np.ndarray:
        """指令片段(n,7) → 机体系角速度 ω̂(n,3)。**与冻结 `G2.omega` 同一代码路径。**"""
        from train_rq import fir_design
        from train_rs import rs_features
        from ceiling_200hz import kin_gyro_fs
        import g_q_v1 as GQ

        n = len(cmd)
        t = np.arange(n) / FPS
        dd = {"cmd": np.asarray(cmd, float), "t_send": t, "t_read": t,
              "tg_i": np.arange(n * 5) / (FPS * 5)}
        qh_f = np.radians(GQ.predict_qpos(dd["cmd"], t_send=t, t_read=t))
        qh0 = np.stack([np.interp(t, t, qh_f[:, j]) for j in range(6)], 1)
        u = np.stack([np.interp(t, t, dd["cmd"][:, j]) for j in range(6)], 1)
        g = np.interp(t, t, dd["cmd"][:, 6])
        dq = np.gradient(qh0, 1 / FPS, axis=0)
        ddq = np.gradient(dq, 1 / FPS, axis=0)
        X = np.concatenate([u, qh0, dq, ddq, u - qh0, g[:, None],
                            np.gradient(g, 1 / FPS)[:, None]], 1)
        qh = qh0 + fir_design(X, self.sq["nlag"]) @ self.zq["W"]
        return kin_gyro_fs(qh, FPS)[:n] + fir_design(
            rs_features(dd, t, qh)[:n], self.nlag_rs) @ self.W_rs

    def _warmup(self) -> None:
        try:
            self._omega(np.zeros((self.prefix + self.n_a, 7)))
        except Exception as exc:                       # 启动期，可抛
            raise G2WallError(f"G-2 前向自检失败: {exc}") from exc

    # ---- 生命周期 ----
    def reset(self) -> None:
        self._plan: _ChunkPlan | None = None
        self._k = 0
        self._resp = np.zeros((N_LAGS, 3))      # 已提交 δa 的未来响应（偏移 0..M）
        self._prefix: np.ndarray | None = None  # 上一 chunk 尾部 PREFIX 帧原始指令
        self._cur_cmd: np.ndarray | None = None  # 本 chunk 的原始指令
        self._prev_grip: float | None = None
        self.stats = {"frames": 0, "gated": 0, "sat": 0, "no_plan": 0,
                      "stage_a_ms": [], "stage_b_ms": [], "solve_ms": [],
                      "stage_a_fail": 0, "stage_b_fail": 0,
                      "frames_stage_a": 0, "frames_stage_b": 0}

    # ---- chunk 装配 ----
    def _build(self, cmd_raw: np.ndarray, ideal: np.ndarray, n_head: int) -> tuple:
        """→ (ê₀[0:n_head], 有效长度)。`ideal` 已是标定后的传感器系角速度 (≥n_head,3)。"""
        cmd = np.asarray(cmd_raw, float)
        head = cmd[:n_head]
        pre = self._prefix if self._prefix is not None else np.zeros((0, 7))
        seq = np.concatenate([pre, head], 0) if len(pre) else head
        w = self._omega(seq)[len(pre):]
        w6 = np.zeros((len(w), 6), np.float32)
        w6[:, 3:] = w
        ws = calibrate_ideal_imu(w6, self.R, self.b)[:, 3:].astype(float)
        m = min(n_head, len(ws), len(ideal))
        e0 = np.empty((m, 3))
        for k in range(m):
            e0[k] = ws[min(k + ALIGN_L, len(ws) - 1)] - ideal[k]
        return e0, m

    def begin_chunk(self, cmd_raw: np.ndarray) -> None:
        """**同步**，在 chunk 生成那一帧调用。只做三件便宜事，绝不跑 G-2：

        建计划、算接触门控（纯 numpy）、把帧号归零。

        ⚠ `_k` 的归零**必须在这里**，不能放进 stage A。stage A 是后台的，
        而空档期间 `step()` 仍要照常推进 `_k` 与 `resp` —— 若等 stage A 落地再归零，
        空档跑掉的帧数会被当成 chunk 内的偏移，整条 δa 错位。
        """
        c = np.asarray(cmd_raw, float)
        plan = _ChunkPlan(len(c))
        plan.contact = command_contact_mask(
            c[:, 6].astype(np.float32), grip_full=GRIP_FULL,
            threshold_ratio=CONTACT_THRESHOLD_RATIO,
            half_window=CONTACT_HALF_WINDOW, previous_grip=self._prev_grip)
        self._prefix = (self._cur_cmd[-self.prefix:].copy()
                        if self._cur_cmd is not None and self.prefix > 0
                        else np.zeros((0, 7)))
        self._cur_cmd = c
        self._prev_grip = float(c[-1, 6])
        self._plan = plan
        self._k = 0

    def fill_stage_a(self, ideal: np.ndarray) -> None:
        """**后台**：只截断 **G-2 的前向**（头 `N_A` 帧 + 热身前缀），供给帧 0..`W_A−1`。

        ⚠ `ideal` 必须是**整 chunk** 的渲染结果 —— `IdealChunkRenderer` 用
        `filtfilt`（零相位、非因果），渲染前缀 ≠ 整渲染的前缀（实测差 8.23）。
        整 chunk 渲染只要 ~30 ms，不是瓶颈。

        失败 → 本 chunk 头部保持 δa=0，**不抛**。
        """
        plan, cmd = self._plan, self._cur_cmd
        if plan is None or cmd is None:
            return
        t0 = time.perf_counter()
        try:
            e0, m = self._build(cmd, ideal, min(self.n_a, plan.T))
            plan.e0hat[:m] = e0
            plan.valid_to = m                       # ê₀ 有效到第 m 帧
            plan.serve_to = min(self.w_a, plan.T)   # 但只输出前 W_A 帧
            plan.stage = "A"
        except Exception as exc:
            self.stats["stage_a_fail"] += 1
            if self.verbose:
                print(f"[g2wall] ⚠ stage A 失败 → 本 chunk 头部 δa=0: "
                      f"{type(exc).__name__}: {exc}")
        self.stats["stage_a_ms"].append((time.perf_counter() - t0) * 1e3)

    def fill_stage_b(self, ideal: np.ndarray) -> None:
        """**后台**：整 chunk 前向，回填 `W_A` 之后。失败 → 保留 stage A 的部分。"""
        plan, cmd = self._plan, self._cur_cmd
        if plan is None or cmd is None:
            return
        t0 = time.perf_counter()
        try:
            e0, m = self._build(cmd, ideal, plan.T)
            plan.e0hat[:m] = e0
            plan.valid_to = max(plan.valid_to, m)
            plan.serve_to = max(plan.serve_to, m)
            plan.stage = "B"
        except Exception as exc:
            self.stats["stage_b_fail"] += 1
            if self.verbose:
                print(f"[g2wall] ⚠ stage B 失败 → 沿用 stage A: "
                      f"{type(exc).__name__}: {exc}")
        self.stats["stage_b_ms"].append((time.perf_counter() - t0) * 1e3)

    # ---- 逐帧 ----
    def step(self) -> G2WallFrame:
        """一帧的 δa(6,)。无计划/越界 → 置零并计数（**不抛**）。"""
        plan = self._plan
        k = self._k
        self._k += 1
        self.stats["frames"] += 1
        zero = np.zeros(6, np.float32)
        if plan is None or k >= plan.T or k >= plan.serve_to:
            self.stats["no_plan"] += 1
            self._advance(zero)
            return G2WallFrame(zero, k, 0, False, False, "none", None)

        t0 = time.perf_counter()
        K = int(min(M, plan.T - 1 - k))
        if K < 0:
            K = 0
        idx = np.minimum(np.arange(k, k + K + 1), plan.valid_to - 1)
        tgt = plan.e0hat[idx] + self._resp[:K + 1]
        d = -(self._chol[K] @ tgt.reshape(-1))
        self.stats["solve_ms"].append((time.perf_counter() - t0) * 1e3)

        if not np.isfinite(d).all():
            self.stats["no_plan"] += 1
            self._advance(zero)
            return G2WallFrame(zero, k, K, False, False, "nan", None)
        sat = bool(np.any(np.abs(d) > EPS))
        self.stats["sat"] += int(sat)
        d = np.clip(d, -EPS, EPS)
        contact = bool(plan.contact[k])
        if contact:
            self.stats["gated"] += 1
            d = np.zeros(6)
        self.stats["frames_stage_a" if plan.stage == "A" else "frames_stage_b"] += 1
        da = d.astype(np.float32)
        self._advance(da)
        return G2WallFrame(da, k, K, contact, sat, plan.stage, plan.e0hat[k].copy())

    def _advance(self, da: np.ndarray) -> None:
        """提交本帧 δa 的未来响应，并把 `resp` 环形前移一帧。"""
        if np.any(da):
            self._resp += self.H @ np.asarray(da, float)
        self._resp[:-1] = self._resp[1:]
        self._resp[-1] = 0.0

    def summary(self) -> dict:
        s, n = self.stats, max(self.stats["frames"], 1)

        def p(key, q=95):
            v = s[key]
            return float(np.percentile(v, q)) if v else 0.0

        return {
            "artifact": "pi-g2wall-v1", "free_parameters": 0,
            "frames": s["frames"], "gated_pct": 100 * s["gated"] / n,
            "sat_pct": 100 * s["sat"] / n, "no_plan_pct": 100 * s["no_plan"] / n,
            "frames_stage_a": s["frames_stage_a"], "frames_stage_b": s["frames_stage_b"],
            "stage_a_ms_p95": p("stage_a_ms"), "stage_a_ms_max": p("stage_a_ms", 100),
            "stage_b_ms_p95": p("stage_b_ms"), "stage_b_ms_max": p("stage_b_ms", 100),
            "solve_ms_p95": p("solve_ms"),
            "stage_a_fail": s["stage_a_fail"], "stage_b_fail": s["stage_b_fail"],
            "prefix": self.prefix, "n_a": self.n_a, "w_a": self.w_a, "eps_rad": EPS,
            "known_risk": self.manifest.get("known_risk", {}),
        }


__all__ = ["G2WallRuntime", "G2WallFrame", "G2WallError", "install_model_cache",
           "ALIGN_L", "M", "N_LAGS", "EPS", "PREFIX", "N_A", "W_A", "TAIL_GUARD"]
