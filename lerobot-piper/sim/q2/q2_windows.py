"""阶段二的 δa 时间轴规则与固定 H 的窗口响应 —— 预注册 §2、§3.5（只在这里实现）。

δa 的可见性（控制周期索引，与指令同一个时间轴）：
- 过去段 t0−19 … t0：日志记录的请求补正量；
- 当前 chunk t0+1 … t1：离线条件预测使用日志中的当前 chunk 补正序列（离线给定的条件，不代表历史部署时在起点就知道全部补正）；
- 超出 t1（以及超出集尾）：取 0。
固定 H：沿用原有的 gap 时间轴 `Δê_t = Σ_{τ=0}^{15} H[τ] · δa_{t−τ}`，不再额外平移 L = 3；卷积包含过去的 δa，不在每个 chunk 清零重启。
第 t 个误差只用到第 t 周期为止的 δa，而计分的 t ≤ t1 ⇒ 始终可见。
"""
from __future__ import annotations
import hashlib

import numpy as np

import q1_data as D1

H_FILE = D1.ROOT / "outputs/step1_H/step1_H_v1.npz"
H_SHA16 = "89d118d4b69837f0"               # float64 sha256 前 16 位（step1-linear-v1 同一个核）


def da_visible(ep, w, upto: int) -> np.ndarray:
    """δa 第 0 … upto 行；超出 t1 或超出集尾的行取 0。"""
    out = np.zeros((upto + 1, 6))
    n = min(upto, w.t1, ep.T - 1) + 1
    if n > 0:
        out[:n] = np.asarray(ep.da, np.float64)[:n]
    return out


def da_span(ep, w, lo: int, hi: int) -> np.ndarray:
    """第 lo … hi 行（含）的可见 δa；< 0 的行取 0。"""
    vis = da_visible(ep, w, max(hi, 0))
    k = np.arange(lo, hi + 1)
    out = np.zeros((len(k), 6))
    ok = k >= 0
    out[ok] = vis[k[ok]]
    return out


def load_h() -> np.ndarray:
    z = np.load(H_FILE, allow_pickle=False)
    H = np.ascontiguousarray(z["H"], np.float64)
    got = hashlib.sha256(H.tobytes()).hexdigest()[:16]
    if got != H_SHA16:
        raise RuntimeError(f"{H_FILE}: 固定 H 的指纹 {got} ≠ 冻结值 {H_SHA16}")
    return H


def h_full(H, ep) -> np.ndarray:
    """整集 gap 索引上的 `Σ_τ H[τ]·δa_{t−τ}` → (T, 3)（零历史只在集首，不在 chunk 边界）。"""
    import fit_step1_candidate as S1
    return S1.causal_conv(np.asarray(H, np.float64), np.asarray(ep.da, np.float64))


def h_window(H, ep, w) -> np.ndarray:
    """窗口内 t ∈ [t0, s1] 的 H 响应 → (n_scored, 3)。"""
    return h_full(H, ep)[w.t0:w.s1 + 1]
